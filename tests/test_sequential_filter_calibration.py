import tempfile
import unittest
from pathlib import Path

import torch

from tfmplayground.experiments.calibrate_sequential_latent_filter import (
    BoundedTemperatures,
    CalibrationConfig,
    observed_temperature_loss,
    rank_candidates,
    run,
    split_seed_bases,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.sequential_latent_filter import (
    NanoTabPFNSequentialLatentFilter,
    SequentialFilterLogits,
    filter_sequential_logits,
    load_sequential_filter_checkpoint,
    save_sequential_filter_checkpoint,
    sequential_filter_checkpoint,
)


def tiny_backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=3,
    )


def simple_raw(stream_count: int = 8) -> SequentialFilterLogits:
    stream_logits = torch.zeros(2, stream_count, 2, 2)
    stream_logits[:, :, 0, 0] = 2.0
    stream_logits[:, :, 1, 0] = -1.0
    query_logits = torch.zeros(2, 4, 2, 2)
    query_logits[:, :, 0, 0] = 2.0
    query_logits[:, :, 0, 1] = -2.0
    query_logits[:, :, 1, 0] = -2.0
    query_logits[:, :, 1, 1] = 2.0
    return SequentialFilterLogits(stream_logits, query_logits, torch.zeros(2, 2, 8))


class SequentialFilterCalibrationTests(unittest.TestCase):
    def test_identity_filtering_matches_model_forward_exactly(self):
        torch.manual_seed(81)
        model = NanoTabPFNSequentialLatentFilter(tiny_backbone()).eval()
        support_x = torch.randn(2, 4, 1)
        support_y = torch.randint(0, 2, (2, 4)).float()
        stream_x = torch.randn(2, 8, 1)
        stream_y = torch.randint(0, 2, (2, 8)).float()
        query_x = torch.randn(2, 4, 1)
        raw = model.raw_logits(support_x, support_y, stream_x, query_x)
        cached = filter_sequential_logits(raw, stream_y)
        direct = model(support_x, support_y, stream_x, stream_y, query_x)
        torch.testing.assert_close(cached.stream_logits, direct.stream_logits, rtol=0, atol=0)
        torch.testing.assert_close(cached.query_logits, direct.query_logits, rtol=0, atol=0)
        torch.testing.assert_close(cached.log_weights, direct.log_weights, rtol=0, atol=0)
        torch.testing.assert_close(cached.joint_probabilities(), direct.joint_probabilities(), rtol=0, atol=0)

    def test_evidence_softening_reduces_final_log_odds_and_preserves_order(self):
        raw = simple_raw()
        labels = torch.zeros(2, 8)
        identity = filter_sequential_logits(raw, labels)
        softened = filter_sequential_logits(raw, labels, evidence_logit_scale=0.1)
        identity_odds = identity.log_weights[:, -1, 0] - identity.log_weights[:, -1, 1]
        softened_odds = softened.log_weights[:, -1, 0] - softened.log_weights[:, -1, 1]
        self.assertTrue(torch.all(softened_odds.abs() < identity_odds.abs()))

        order = torch.tensor([4, 1, 7, 0, 5, 2, 6, 3])
        reordered_raw = SequentialFilterLogits(raw.stream_logits[:, order], raw.query_logits, raw.slots)
        reordered = filter_sequential_logits(reordered_raw, labels[:, order], evidence_logit_scale=0.1)
        torch.testing.assert_close(softened.log_weights[:, -1], reordered.log_weights[:, -1], atol=1e-6, rtol=1e-6)

    def test_query_sharpening_reduces_incoherent_mass(self):
        raw = simple_raw()
        labels = torch.zeros(2, 8)
        identity = filter_sequential_logits(raw, labels, query_temperature=1.0)
        sharpened = filter_sequential_logits(raw, labels, query_temperature=0.5)

        def incoherent(prediction):
            joint = prediction.joint_probabilities()[:, 0]
            return 1.0 - joint[:, 0] - joint[:, -1]

        self.assertTrue(torch.all(incoherent(sharpened) < incoherent(identity)))

    def test_old_checkpoint_defaults_to_identity_temperatures(self):
        model = NanoTabPFNSequentialLatentFilter(tiny_backbone())
        checkpoint = sequential_filter_checkpoint(
            model,
            training_config={"initial_support_count": 128, "stream_count": 128},
            source_checkpoint_sha256="source",
            stage="controlled",
        )
        checkpoint["architecture"].pop("evidence_logit_scale")
        checkpoint["architecture"].pop("query_temperature")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pth"
            torch.save(checkpoint, path)
            loaded, _ = load_sequential_filter_checkpoint(path)
        self.assertEqual(loaded.evidence_logit_scale, 1.0)
        self.assertEqual(loaded.query_temperature, 1.0)

    def test_seed_ranges_are_deterministic_and_disjoint(self):
        first = split_seed_bases(2402)
        second = split_seed_bases(2402)
        self.assertEqual(first, second)
        self.assertEqual(len(first.values()), len(set(first.values())))

    def test_only_temperature_scalars_receive_gradients(self):
        torch.manual_seed(82)
        model = NanoTabPFNSequentialLatentFilter(tiny_backbone())
        raw = simple_raw()
        labels = torch.zeros(2, 8)
        query_y = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        calibrator = BoundedTemperatures(0.1, 0.7)
        evidence_scale, query_temperature = calibrator()
        loss, _ = observed_temperature_loss(raw, labels, query_y, evidence_scale, query_temperature)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in calibrator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_candidate_ranking_is_deterministic(self):
        records = [
            {
                "candidate": "identity",
                "passed": False,
                "passed_checks": 11,
                "max_normalized_violation": 1.0,
                "final_marginal_js": 0.02,
                "identity_distance": 0.0,
            },
            {
                "candidate": "grid",
                "passed": True,
                "passed_checks": 14,
                "max_normalized_violation": 0.0,
                "final_marginal_js": 0.03,
                "identity_distance": 0.9,
            },
            {
                "candidate": "learned",
                "passed": True,
                "passed_checks": 14,
                "max_normalized_violation": 0.0,
                "final_marginal_js": 0.01,
                "identity_distance": 1.0,
            },
        ]
        first = rank_candidates(records)
        second = rank_candidates(records)
        self.assertEqual(list(first.candidate), ["learned", "grid", "identity"])
        self.assertTrue(first.equals(second))

    def test_calibration_cli_smoke_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = NanoTabPFNSequentialLatentFilter(tiny_backbone())
            controlled = root / "controlled.pth"
            save_sequential_filter_checkpoint(
                controlled,
                model,
                training_config={"initial_support_count": 128, "stream_count": 128},
                source_checkpoint_sha256="original",
                stage="controlled",
                controlled_gate={
                    "passed": False,
                    "metrics": {
                        "query_permutation_max_error": 0.0,
                        "evidence_order_weight_max_error": 0.0,
                        "evidence_order_joint_max_error": 0.0,
                    },
                },
            )
            official = root / "official.pth"
            torch.save(
                {
                    "architecture": {
                        "embedding_size": 8,
                        "num_attention_heads": 2,
                        "mlp_hidden_size": 16,
                        "num_layers": 1,
                        "num_outputs": 3,
                    },
                    "model": model.backbone.state_dict(),
                },
                official,
            )
            output = root / "run"
            result = run(
                CalibrationConfig(
                    checkpoint=str(controlled),
                    official_checkpoint=str(official),
                    output_dir=str(output),
                    device="cpu",
                    grid_trials=1,
                    optimization_trials=1,
                    selection_trials=1,
                    final_trials=1,
                    scalar_steps=1,
                    scalar_batch_size=1,
                    ordinary_evaluation_batches=0,
                    run_tabicl_on_pass=False,
                    plots=False,
                )
            )
            self.assertEqual(result, output.resolve())
            required = (
                "temperature_grid.csv",
                "learned_temperature_curve.csv",
                "selection_ranking.csv",
                "selection.json",
                "final_selected_gate.json",
                "calibrated_checkpoint.pth",
                "ordinary_accuracy.json",
                "decision.json",
            )
            self.assertTrue(all((output / name).is_file() for name in required))


if __name__ == "__main__":
    unittest.main()
