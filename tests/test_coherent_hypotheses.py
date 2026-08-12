import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.coherent_hypotheses import (
    enumerate_chain_joint_torch,
    sequence_to_canonical_indices,
)
from tfmplayground.experiments.train_coherent_hypotheses import (
    CoherentTrainingConfig,
    consistency_loss,
    controlled_batch,
    hypothesis_loss,
    train_consistency_model,
    train_hypothesis_model,
)
from tfmplayground.experiments.train_coherent_hypotheses import main as training_main
from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader
from tfmplayground.models.hypothesis import (
    NanoTabPFNHypothesisModel,
    load_hypothesis_checkpoint,
    save_hypothesis_checkpoint,
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


class CoherentHypothesisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.support_x = torch.randn(2, 4, 1)
        self.support_y = torch.randint(0, 2, (2, 4)).float()
        self.query_x = torch.randn(2, 3, 1)

    def test_embedding_refactor_preserves_decoder_path(self):
        model = tiny_backbone().eval()
        full_x = torch.cat((self.support_x, self.query_x), dim=1)
        with torch.no_grad():
            encoded = model.encode_table((full_x, self.support_y), train_test_split_index=4)
            expected = model.decoder(encoded[:, 4:, -1, :])
            actual = model((full_x, self.support_y), train_test_split_index=4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hypothesis_shapes_normalization_and_permutation_invariance(self):
        model = NanoTabPFNHypothesisModel(tiny_backbone(), num_hypotheses=2).eval()
        full_x = torch.cat((self.support_x, self.query_x), dim=1)
        prediction = model((full_x, self.support_y), train_test_split_index=4)
        self.assertEqual(prediction.slot_logits.shape, (2, 3, 2, 2))
        self.assertEqual(prediction.slot_log_weights.shape, (2, 2))
        self.assertEqual(prediction.row_log_evidence.shape, (2, 4, 2))
        torch.testing.assert_close(prediction.slot_log_weights.exp().sum(-1), torch.ones(2))
        canonical = prediction.joint_probabilities()
        torch.testing.assert_close(canonical.sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6)

        order = (2, 0, 1)
        reordered_prediction = type(prediction)(
            prediction.slot_logits[:, order], prediction.slot_log_weights, prediction.row_log_evidence
        )
        sequence_joint = reordered_prediction.joint_probabilities()
        mapping = sequence_to_canonical_indices(order)
        remapped = sequence_joint.index_select(1, torch.argsort(mapping))
        torch.testing.assert_close(remapped, canonical, atol=1e-6, rtol=1e-6)

    def test_hypothesis_model_has_finite_gradients(self):
        model = NanoTabPFNHypothesisModel(tiny_backbone(), num_hypotheses=2)
        full_x = torch.cat((self.support_x, self.query_x), dim=1)
        loss = -model((full_x, self.support_y), train_test_split_index=4).joint_log_probabilities().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_differentiable_chain_enumerator_normalizes_and_backpropagates(self):
        model = tiny_backbone()
        joint = enumerate_chain_joint_torch(model, self.support_x, self.support_y, self.query_x, (2, 0, 1))
        self.assertEqual(joint.shape, (2, 8))
        torch.testing.assert_close(joint.sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6)
        (-joint[:, 0].log().mean()).backward()
        self.assertIsNotNone(model.decoder.linear2.weight.grad)

    def test_hypothesis_checkpoint_round_trip(self):
        model = NanoTabPFNHypothesisModel(tiny_backbone(), num_hypotheses=2).eval()
        full_x = torch.cat((self.support_x, self.query_x), dim=1)
        expected = model((full_x, self.support_y), train_test_split_index=4).joint_probabilities()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hypothesis.pth"
            save_hypothesis_checkpoint(
                path,
                model,
                training_config={"seed": 13},
                source_checkpoint_sha256="abc",
                stage="slots",
            )
            loaded, metadata = load_hypothesis_checkpoint(path)
            actual = loaded.eval()((full_x, self.support_y), train_test_split_index=4).joint_probabilities()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["source_checkpoint_sha256"], "abc")
        self.assertEqual(metadata["training_config"], {"seed": 13})

    def test_repeated_identical_evidence_adds_linearly(self):
        row_scores = torch.tensor([[[0.2, -0.1]]])
        one = torch.log_softmax(row_scores.sum(1), dim=-1)
        eight = torch.log_softmax(row_scores.expand(-1, 8, -1).sum(1), dim=-1)
        self.assertGreater(float((eight[0, 0] - eight[0, 1]).abs()), float((one[0, 0] - one[0, 1]).abs()))
        np.testing.assert_allclose((eight[0, 0] - eight[0, 1]).item(), 8 * 0.3, atol=1e-6)

    def test_training_losses_and_short_stages_are_finite(self):
        config = CoherentTrainingConfig(
            device="cpu",
            query_count=2,
            batch_size=1,
            accumulate_gradients=1,
            consistency_steps=1,
            slot_frozen_steps=1,
            slot_unfrozen_steps=1,
            validation_interval=1,
            controlled_only=True,
        )
        rng = np.random.default_rng(9)
        batch = controlled_batch(config, rng, support_sizes=(4,), evidence_counts=(2,))
        backbone = tiny_backbone()
        loss, metrics = consistency_loss(backbone, batch, (1, 0))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

        trained_backbone, history = train_consistency_model(backbone, config, rng)
        self.assertEqual(len(history), 1)
        hypothesis_model = NanoTabPFNHypothesisModel(trained_backbone)
        slot_loss, slot_metrics = hypothesis_loss(hypothesis_model, batch)
        self.assertTrue(torch.isfinite(slot_loss))
        self.assertTrue(all(np.isfinite(value) for value in slot_metrics.values()))
        _, slot_history = train_hypothesis_model(hypothesis_model, config, rng)
        self.assertEqual(len(slot_history), 2)

    def test_tiny_slot_model_overfits_ambiguous_and_identified_batches(self):
        torch.manual_seed(3)
        rng = np.random.default_rng(4)
        config = CoherentTrainingConfig(
            device="cpu", query_count=2, batch_size=4, controlled_only=True
        )
        batches = [
            controlled_batch(config, rng, support_sizes=(4,), evidence_counts=(evidence_count,))
            for evidence_count in (0, 4)
        ]
        model = NanoTabPFNHypothesisModel(tiny_backbone())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        def combined_loss():
            return sum(hypothesis_loss(model, batch)[0] for batch in batches) / len(batches)

        initial = float(combined_loss().detach())
        for _ in range(15):
            optimizer.zero_grad()
            loss = combined_loss()
            loss.backward()
            optimizer.step()
        self.assertLess(float(combined_loss().detach()), 0.6 * initial)

    def test_installed_tabicl_prior_adapter_produces_binary_batches(self):
        loader = TabICLPriorDataLoader(
            num_steps=1,
            batch_size=2,
            num_datapoints_min=16,
            num_datapoints_max=17,
            min_features=1,
            max_features=2,
            max_num_classes=2,
            prior_type="dummy",
            device=torch.device("cpu"),
        )
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape[0], 2)
        self.assertEqual(batch["y"].shape[:2], batch["x"].shape[:2])
        self.assertTrue(set(batch["y"].unique().tolist()) <= {0, 1})

    def test_training_cli_writes_both_stage_checkpoints(self):
        backbone = tiny_backbone()
        original_checkpoint = {
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
            checkpoint_path = directory / "original.pth"
            output_dir = directory / "run"
            torch.save(original_checkpoint, checkpoint_path)
            result = training_main(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-dir",
                    str(output_dir),
                    "--controlled-only",
                    "--query-count",
                    "2",
                    "--batch-size",
                    "1",
                    "--accumulate-gradients",
                    "1",
                    "--consistency-steps",
                    "1",
                    "--slot-frozen-steps",
                    "1",
                    "--slot-unfrozen-steps",
                    "0",
                    "--validation-interval",
                    "1",
                    "--evaluation-trials",
                    "1",
                    "--ordinary-evaluation-batches",
                    "1",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "consistency_checkpoint.pth").is_file())
            self.assertTrue((output_dir / "hypothesis_checkpoint.pth").is_file())
            self.assertTrue((output_dir / "acceptance.json").is_file())


if __name__ == "__main__":
    unittest.main()
