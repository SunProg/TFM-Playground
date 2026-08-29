import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.evaluate_bayesian_tabarena import (
    compare_uncertainty_to_vanilla_entropy,
    evaluate_uncertainty_split,
    posterior_collapse_diagnostics,
    static_metrics,
)
from tfmplayground.experiments.summarize_static_bayesian_sweep import summarize_sweep
from tfmplayground.experiments.train_bayesian_nanotabpfn import (
    StaticBayesianTrainingConfig,
    _ordinary_accuracy,
    controlled_static_batch,
    random_label_diagnostic_batch,
    static_bayesian_loss,
    structured_static_batch,
    train_frozen_stage,
    train_full_stage,
    validate_static_config,
)
from tfmplayground.models.hypothesis import (
    HypothesisPrediction,
    NanoTabPFNBayesianModel,
    load_bayesian_checkpoint,
    save_bayesian_checkpoint,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_model(k=2):
    return NanoTabPFNBayesianModel(NanoTabPFNModel(8, 2, 16, 1, 3), num_hypotheses=k)


class StaticBayesianTests(unittest.TestCase):
    def test_k_and_probability_invariants(self):
        for hypotheses in (4, 8):
            with self.subTest(hypotheses=hypotheses):
                model = tiny_model(hypotheses).eval()
                support_x = torch.randn(2, 5, 1)
                support_y = torch.randint(0, 2, (2, 5)).float()
                query_x = torch.randn(2, 3, 1)
                prediction = model(support_x, support_y, query_x)
                self.assertEqual(prediction.query_probabilities.shape, (2, 3, hypotheses, 2))
                torch.testing.assert_close(prediction.posterior_weights.sum(-1), torch.ones(2))
                torch.testing.assert_close(
                    prediction.joint_probabilities().sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6
                )
                self.assertTrue(torch.isfinite(prediction.mutual_information()).all())
                self.assertTrue(torch.isfinite(prediction.epistemic_variance()).all())
                self.assertTrue(torch.isfinite(prediction.effective_sample_size()).all())
                resampled = prediction.systematic_resample_indices(
                    2,
                    generator=torch.Generator().manual_seed(3),
                )
                self.assertEqual(resampled.shape, (2, 2))
                self.assertTrue(((resampled >= 0) & (resampled < hypotheses)).all())

    def test_mean_is_exactly_vanilla_and_permutations_are_equivariant(self):
        torch.manual_seed(11)
        model = tiny_model(2).eval()
        support_x = torch.randn(1, 8, 3)
        support_y = torch.randint(0, 2, (1, 8)).float()
        query_x = torch.randn(1, 4, 3)
        prediction = model(support_x, support_y, query_x)
        vanilla = model.backbone(support_x, support_y, query_x)[..., :2].softmax(-1)
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla, atol=1e-6, rtol=1e-6)
        self.assertLessEqual(float(prediction.mean_preservation_error().max().detach()), 1e-6)

        support_order = torch.tensor([6, 1, 4, 0, 7, 2, 5, 3])
        reordered_support = model(support_x[:, support_order], support_y[:, support_order], query_x)
        torch.testing.assert_close(
            reordered_support.marginal_probabilities(),
            prediction.marginal_probabilities(),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            reordered_support.posterior_weights,
            prediction.posterior_weights,
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            reordered_support.mutual_information(),
            prediction.mutual_information(),
            atol=1e-5,
            rtol=1e-5,
        )
        query_order = torch.tensor([2, 0, 3, 1])
        reordered_query = model(support_x, support_y, query_x[:, query_order])
        torch.testing.assert_close(
            reordered_query.query_probabilities,
            prediction.query_probabilities[:, query_order],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_uncertainty_diagnostics(self):
        identical = HypothesisPrediction(
            torch.zeros(1, 2, 2, 2), torch.log(torch.tensor([[0.5, 0.5]])), torch.zeros(1, 3, 2)
        )
        torch.testing.assert_close(identical.mutual_information(), torch.zeros(1, 2))
        disagreeing = HypothesisPrediction(
            torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]]),
            torch.log(torch.tensor([[0.5, 0.5]])),
            torch.zeros(1, 1, 2),
        )
        self.assertGreater(float(disagreeing.mutual_information()), 0.5)

    def test_controlled_generalized_loss_and_checkpoint(self):
        config = StaticBayesianTrainingConfig(num_hypotheses=4, query_count=3, batch_size=2, device="cpu")
        batch = controlled_static_batch(config, np.random.default_rng(4))
        model = tiny_model(4)
        loss, _ = static_bayesian_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bayesian.pth"
            save_bayesian_checkpoint(
                path, model, training_config={"query_count": 3}, source_checkpoint_sha256="x", stage="frozen"
            )
            loaded, metadata = load_bayesian_checkpoint(path)
            # Slot attention resamples its slot initialization while training, so
            # the round-trip is compared in eval mode, where the draw is seeded.
            expected = model.eval()(batch.support_x, batch.support_y, batch.query_x).joint_probabilities()
            actual = loaded.eval()(batch.support_x, batch.support_y, batch.query_x).joint_probabilities()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["model_type"], "nanotabpfn_bayesian")

    def test_validation_and_metric_helpers(self):
        with self.assertRaises(ValueError):
            validate_static_config(StaticBayesianTrainingConfig(num_hypotheses=5, query_count=2))
        with self.assertRaises(ValueError):
            validate_static_config(StaticBayesianTrainingConfig(min_delta=-1e-4))
        metrics = static_metrics(np.array([0, 1, 1, 0]), np.array([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]))
        self.assertTrue(np.isfinite(metrics["nll"]))
        diagnostics = posterior_collapse_diagnostics(np.array([[0.5, 0.5], [0.99, 0.01]]))
        self.assertEqual(diagnostics["collapse_fraction"], 0.5)
        comparison = compare_uncertainty_to_vanilla_entropy(
            np.array([0, 1, 1, 0]),
            np.array([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]),
            np.array([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]),
            np.array([0.1, 0.2, 0.8, 0.3]),
        )
        self.assertIn("bayesian_aurc", comparison)

    def test_selection_loss_removes_irreducible_target_entropy(self):
        config = StaticBayesianTrainingConfig(num_hypotheses=4, query_count=3, batch_size=2, device="cpu")
        batch = controlled_static_batch(config, np.random.default_rng(13), evidence_count=0, noise=0.1)
        _, metrics = static_bayesian_loss(tiny_model(4), batch)
        self.assertGreaterEqual(metrics["selection_loss"], 0.0)
        self.assertGreaterEqual(metrics["joint_kl"], 0.0)
        self.assertLessEqual(metrics["selection_loss"], metrics["loss"])
        # The per-slot and per-weight terms are gone: they scored slots against
        # named candidate tasks, which is the assignment supervision that slot
        # competition replaces.  What is left is permutation invariant.
        expected = (
            metrics["joint_kl"] / (config.query_count * np.log(2))
            + 0.25 * metrics["mi_loss"]
            + 0.25 * metrics["variance_loss"]
        )
        self.assertAlmostEqual(metrics["selection_loss"], expected, places=6)
        for removed in ("weight_loss", "weight_kl", "slot_loss", "slot_kl"):
            self.assertNotIn(removed, metrics)

    def test_structured_curriculum_and_frozen_training(self):
        try:
            import tabicl  # noqa: F401
        except ImportError:
            self.skipTest("TabICL is optional outside the training environment.")
        config = StaticBayesianTrainingConfig(
            num_hypotheses=2,
            query_count=3,
            batch_size=1,
            support_size=8,
            min_features=2,
            max_features=2,
            frozen_steps=1,
            validation_interval=1,
            validation_episodes=3,
            ordinary_evaluation_batches=1,
            device="cpu",
        )
        rng = np.random.default_rng(9)
        ambiguous = structured_static_batch(config, rng, condition="ambiguous")
        identifiable = structured_static_batch(config, rng, condition="identifiable")
        noisy = structured_static_batch(config, rng, condition="noisy")
        self.assertTrue(ambiguous.controlled)
        self.assertTrue(identifiable.controlled)
        self.assertTrue(noisy.controlled)
        self.assertGreater(float(ambiguous.hypothesis_probabilities.var(dim=1).mean()), 0)
        torch.testing.assert_close(
            noisy.hypothesis_probabilities.var(dim=1),
            torch.zeros_like(noisy.hypothesis_probabilities.var(dim=1)),
        )
        diagnostic = random_label_diagnostic_batch(config, rng)
        self.assertFalse(diagnostic.controlled)

        model = tiny_model(2)
        backbone_before = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
        trained, history, validation = train_frozen_stage(model, config)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(validation))
        for name, value in trained.backbone.state_dict().items():
            torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            train_full_stage(trained, config)

    def test_k8_large_support_structured_episode(self):
        try:
            import tabicl  # noqa: F401
        except ImportError:
            self.skipTest("TabICL is optional outside the training environment.")
        config = StaticBayesianTrainingConfig(
            num_hypotheses=8,
            query_count=8,
            batch_size=1,
            support_size=512,
            min_features=2,
            max_features=2,
            device="cpu",
        )
        episode = structured_static_batch(config, np.random.default_rng(2402), condition="ambiguous")
        self.assertEqual(episode.support_x.shape, (1, 512, 2))
        self.assertEqual(episode.query_x.shape, (1, 8, 2))
        self.assertEqual(episode.posterior.shape, (1, 8))
        self.assertEqual(episode.hypothesis_probabilities.shape, (1, 8, 8))
        self.assertTrue(torch.isfinite(episode.support_x).all())
        self.assertTrue(torch.isfinite(episode.posterior).all())
        torch.testing.assert_close(episode.posterior.sum(dim=-1), torch.ones(1))

    def test_sweep_summary_applies_required_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (loss, preserved) in enumerate(((1.2, True), (0.8, False))):
                training = root / f"trial-{index}" / "training"
                training.mkdir(parents=True)
                (training / "config.json").write_text(
                    json.dumps(
                        {
                            "num_hypotheses": 4 + 4 * index,
                            "learning_rate": 1e-4,
                            "likelihood_temperature": 0.01,
                        }
                    )
                )
                (training / "selection.json").write_text(
                    json.dumps(
                        {
                            "validation_loss": loss,
                            "acceptance": {
                                "finite_validation_loss": True,
                                "backbone_unchanged": True,
                                "mean_preserved_at_1e-6": preserved,
                                "ordinary_accuracy_identical": True,
                                "passed": preserved,
                            },
                        }
                    )
                )
            summary_path = summarize_sweep(root)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["completed_trials"], 2)
            self.assertEqual(summary["eligible_trials"], 1)
            self.assertEqual(summary["selected"]["trial"], "trial-0")

    def test_ordinary_accuracy_moves_model_to_configured_device(self):
        config = StaticBayesianTrainingConfig(
            num_hypotheses=2,
            query_count=2,
            batch_size=1,
            support_size=8,
            min_features=2,
            max_features=2,
            ordinary_evaluation_batches=1,
            device="cpu",
        )
        model = tiny_model(2)
        accuracy = _ordinary_accuracy(model, config)
        self.assertTrue(np.isfinite(accuracy))
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_split_evaluation_uses_matched_context_and_preserves_mean(self):
        rng = np.random.default_rng(17)
        X_train = rng.normal(size=(12, 2)).astype(np.float32)
        y_train = np.tile(np.array([0, 1]), 6)
        X_test = rng.normal(size=(4, 2)).astype(np.float32)
        y_test = np.array([0, 1, 1, 0])
        model = tiny_model(2).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bayesian_path = directory / "bayesian.pth"
            vanilla_path = directory / "vanilla.pth"
            save_bayesian_checkpoint(
                bayesian_path,
                model,
                training_config={},
                source_checkpoint_sha256="test",
                stage="mean_preserving_frozen",
            )
            backbone = model.backbone
            torch.save(
                {
                    "architecture": {
                        "num_layers": backbone.num_layers,
                        "embedding_size": backbone.embedding_size,
                        "num_attention_heads": backbone.num_attention_heads,
                        "mlp_hidden_size": backbone.mlp_hidden_size,
                        "num_outputs": backbone.num_outputs,
                    },
                    "model": backbone.state_dict(),
                },
                vanilla_path,
            )
            report = evaluate_uncertainty_split(
                X_train,
                y_train,
                X_test,
                y_test,
                checkpoint=str(bayesian_path),
                vanilla_checkpoint=str(vanilla_path),
                context_size=8,
            )
        self.assertTrue(report["context_indices_identical"])
        self.assertFalse(report["query_labels_used_for_construction"])
        self.assertLessEqual(report["max_mean_probability_difference"], 1e-6)


if __name__ == "__main__":
    unittest.main()
