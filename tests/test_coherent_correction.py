import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.train_coherent_correction import (
    CorrectionConfig,
    controlled_observation_batch,
    crossfit_loss,
    should_launch_variational,
    variational_loss,
)
from tfmplayground.experiments.train_coherent_correction import main as correction_main
from tfmplayground.experiments.train_coherent_hypotheses import (
    CoherentTrainingConfig,
    save_consistency_checkpoint,
)
from tfmplayground.models.coherent_correction import (
    NanoTabPFNCrossFitHypothesisModel,
    NanoTabPFNVariationalHypothesisModel,
    load_correction_checkpoint,
    permutation_invariant_folds,
    save_correction_checkpoint,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=3,
    )


class CoherentCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.support_x = torch.randn(2, 8, 1)
        self.support_y = torch.randint(0, 2, (2, 8)).float()
        self.query_x = torch.randn(2, 4, 1)

    def _predict(self, model):
        return model(
            ((torch.cat((self.support_x, self.query_x), dim=1)), self.support_y),
            train_test_split_index=self.support_x.shape[1],
        )

    def test_folds_are_balanced_deterministic_and_permutation_equivariant(self):
        folds = permutation_invariant_folds(self.support_x, num_partitions=2)
        torch.testing.assert_close(folds, permutation_invariant_folds(self.support_x, num_partitions=2))
        self.assertTrue(torch.all(folds.sum(-1) == 4))

        order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
        permuted = permutation_invariant_folds(self.support_x[:, order], num_partitions=2)
        torch.testing.assert_close(permuted, folds[:, :, order])

    def test_heldout_label_cannot_change_its_prediction_logits(self):
        model = NanoTabPFNCrossFitHypothesisModel(tiny_backbone()).eval()
        original = self._predict(model)
        changed_y = self.support_y.clone()
        changed_y[:, 3] = 1.0 - changed_y[:, 3]
        changed = model(
            (torch.cat((self.support_x, self.query_x), dim=1), changed_y),
            train_test_split_index=8,
        )
        torch.testing.assert_close(
            original.heldout_logits[:, :, 3],
            changed.heldout_logits[:, :, 3],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(original.row_log_evidence[:, 3], changed.row_log_evidence[:, 3]))

    def test_crossfit_scores_each_row_and_has_finite_gradients(self):
        model = NanoTabPFNCrossFitHypothesisModel(tiny_backbone())
        prediction = self._predict(model)
        self.assertEqual(prediction.heldout_logits.shape, (2, 2, 8, 2, 2))
        self.assertTrue(torch.isfinite(prediction.row_log_evidence).all())
        torch.testing.assert_close(prediction.slot_log_weights.exp().sum(-1), torch.ones(2))
        torch.testing.assert_close(prediction.joint_probabilities().sum(-1), torch.ones(2))
        (-prediction.joint_log_probabilities().mean() + prediction.alignment_loss()).backward()
        self.assertIsNotNone(model.slot_decoder[0].weight.grad)
        self.assertTrue(torch.isfinite(model.slot_decoder[0].weight.grad).all())

    def test_neutral_duplication_has_no_slot_preference_when_slots_are_identical(self):
        model = NanoTabPFNCrossFitHypothesisModel(tiny_backbone()).eval()
        with torch.no_grad():
            model.hypothesis_queries[1].copy_(model.hypothesis_queries[0])
            model.slot_prior_logits.zero_()
        prediction = self._predict(model)
        doubled = model(
            (
                torch.cat((self.support_x.repeat(1, 2, 1), self.query_x), dim=1),
                self.support_y.repeat(1, 2),
            ),
            train_test_split_index=16,
        )
        expected = torch.full((2, 2), 0.5)
        torch.testing.assert_close(prediction.slot_log_weights.exp(), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(doubled.slot_log_weights.exp(), expected, atol=1e-6, rtol=0)

    def test_support_and_query_permutation_invariance(self):
        model = NanoTabPFNCrossFitHypothesisModel(tiny_backbone()).eval()
        canonical = self._predict(model).joint_probabilities()
        row_order = torch.tensor([3, 6, 0, 7, 2, 5, 1, 4])
        support_permuted = model(
            (
                torch.cat((self.support_x[:, row_order], self.query_x), dim=1),
                self.support_y[:, row_order],
            ),
            train_test_split_index=8,
        ).joint_probabilities()
        torch.testing.assert_close(support_permuted, canonical, atol=1e-6, rtol=1e-6)

        query_order = torch.tensor([2, 0, 3, 1])
        query_permuted = model(
            (torch.cat((self.support_x, self.query_x[:, query_order]), dim=1), self.support_y),
            train_test_split_index=8,
        ).joint_probabilities()
        outcomes = torch.arange(16)
        bits = (outcomes[:, None] >> torch.arange(3, -1, -1)) & 1
        permuted_index = (bits[:, query_order] * (2 ** torch.arange(3, -1, -1))).sum(-1)
        remapped = torch.empty_like(query_permuted)
        remapped[:, outcomes] = query_permuted[:, permuted_index]
        torch.testing.assert_close(remapped, canonical, atol=1e-6, rtol=1e-6)

    def test_checkpoint_round_trips_for_both_model_types(self):
        for model in (
            NanoTabPFNCrossFitHypothesisModel(tiny_backbone()),
            NanoTabPFNVariationalHypothesisModel(tiny_backbone()),
        ):
            expected = self._predict(model.eval()).joint_probabilities()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "correction.pth"
                save_correction_checkpoint(
                    path,
                    model,
                    training_config={"seed": 21},
                    source_checkpoint_sha256="source",
                    stage="test",
                )
                loaded, metadata = load_correction_checkpoint(path)
                actual = self._predict(loaded.eval()).joint_probabilities()
            torch.testing.assert_close(actual, expected)
            self.assertEqual(metadata["model_type"], model.model_type)
            self.assertEqual(metadata["source_checkpoint_sha256"], "source")

    def test_tiny_crossfit_and_variational_losses_decrease(self):
        config = CorrectionConfig(device="cpu", batch_size=2, accumulate_gradients=1)
        rng = np.random.default_rng(8)
        batches = [
            controlled_observation_batch(config, rng, zero_evidence=zero_evidence, support_size=16)
            for zero_evidence in (True, False)
        ]
        for model, loss_function in (
            (NanoTabPFNCrossFitHypothesisModel(tiny_backbone(), num_partitions=1), crossfit_loss),
            (
                NanoTabPFNVariationalHypothesisModel(tiny_backbone()),
                lambda candidate, data: variational_loss(candidate, data, beta=0.01),
            ),
        ):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            def combined_loss(candidate=model, candidate_loss=loss_function):
                return sum(candidate_loss(candidate, batch)[0] for batch in batches) / len(batches)

            initial = float(combined_loss().detach())
            for _ in range(8):
                optimizer.zero_grad()
                loss = combined_loss()
                loss.backward()
                optimizer.step()
            self.assertLess(float(combined_loss().detach()), initial)

    def test_fallback_gate_runs_exactly_on_failed_crossfit_or_direct_request(self):
        self.assertFalse(should_launch_variational("crossfit", True, {"passed": True}))
        self.assertTrue(should_launch_variational("crossfit", True, {"passed": False}))
        self.assertFalse(should_launch_variational("crossfit", False, {"passed": False}))
        self.assertTrue(should_launch_variational("variational", False))

    def test_crossfit_cli_smoke_writes_selection_and_checkpoint(self):
        backbone = tiny_backbone()
        original = {
            "architecture": {
                "embedding_size": backbone.embedding_size,
                "num_attention_heads": backbone.num_attention_heads,
                "mlp_hidden_size": backbone.mlp_hidden_size,
                "num_layers": backbone.num_layers,
                "num_outputs": backbone.num_outputs,
            },
            "model": backbone.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original_path = directory / "original.pth"
            consistency_path = directory / "consistency.pth"
            output_dir = directory / "run"
            torch.save(original, original_path)
            save_consistency_checkpoint(
                consistency_path,
                backbone,
                CoherentTrainingConfig(),
                "original",
            )
            result = correction_main(
                [
                    "--checkpoint",
                    str(original_path),
                    "--consistency-checkpoint",
                    str(consistency_path),
                    "--biased-checkpoint",
                    str(directory / "missing.pth"),
                    "--output-dir",
                    str(output_dir),
                    "--controlled-only",
                    "--no-fallback-on-failure",
                    "--batch-size",
                    "1",
                    "--accumulate-gradients",
                    "1",
                    "--crossfit-frozen-steps",
                    "1",
                    "--crossfit-unfrozen-steps",
                    "0",
                    "--validation-interval",
                    "1",
                    "--evaluation-trials",
                    "1",
                    "--ordinary-evaluation-batches",
                    "1",
                    "--num-partitions",
                    "1",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "crossfit_checkpoint.pth").is_file())
            self.assertTrue((output_dir / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
