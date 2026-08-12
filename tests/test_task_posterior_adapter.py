import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.evaluate_task_posterior_tabarena import (
    _installed_tabarena_revision,
)
from tfmplayground.experiments.task_posterior_acceptance import (
    no_harm_gate,
    paired_bootstrap_gate,
)
from tfmplayground.experiments.train_task_posterior_adapter import (
    TaskPosteriorTrainingConfig,
    choose_ordinary_episode,
    contrastive_episode_objective,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    load_task_posterior_checkpoint,
    match_regimes_to_particles,
    regime_posterior_supervision_loss,
    save_task_posterior_checkpoint,
    task_posterior_loss,
)
from tfmplayground.tabarena_model import (
    APPLICABILITY_CONSTRAINTS,
    TabArenaTaskPosteriorModel,
)
from tfmplayground.task_posterior_interface import TaskPosteriorClassifier


def tiny_backbone(num_outputs: int = 3) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=2,
        num_outputs=num_outputs,
    )


class TaskPosteriorAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3).eval()
        generator = torch.Generator().manual_seed(42)
        self.context_x = torch.randn(2, 12, 5, generator=generator)
        self.context_y = torch.randint(0, 3, (2, 12), generator=generator)
        self.query_x = torch.randn(2, 6, 5, generator=generator)

    def test_zero_initialized_particles_are_exactly_vanilla_multiclass(self):
        prediction = self.model(self.context_x, self.context_y, self.query_x, class_count=3)
        vanilla = self.model.backbone(self.context_x, self.context_y.float(), self.query_x)[..., :3]
        torch.testing.assert_close(prediction.vanilla_logits, vanilla, rtol=0, atol=0)
        torch.testing.assert_close(
            prediction.particle_logits,
            vanilla[:, :, None].expand(-1, -1, 4, -1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla.softmax(-1), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(
            prediction.log_weights.exp(),
            torch.full_like(prediction.log_weights, 0.25),
            rtol=0,
            atol=0,
        )

    def test_slots_are_label_conditioned_and_context_permutation_invariant(self):
        original = self.model(self.context_x, self.context_y, self.query_x, class_count=3)
        changed_y = self.context_y.clone()
        changed_y[:, 0] = (changed_y[:, 0] + 1) % 3
        changed = self.model(self.context_x, changed_y, self.query_x, class_count=3)
        self.assertFalse(torch.allclose(original.slots, changed.slots))

        order = torch.tensor([8, 2, 11, 0, 5, 4, 7, 1, 10, 3, 9, 6])
        permuted = self.model(self.context_x[:, order], self.context_y[:, order], self.query_x, class_count=3)
        torch.testing.assert_close(original.slots, permuted.slots, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            original.marginal_probabilities(),
            permuted.marginal_probabilities(),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_candidate_supervision_and_ordinary_loss_are_finite(self):
        prediction = self.model(self.context_x, self.context_y, self.query_x, class_count=3)
        target = torch.randint(0, 3, (2, 6))
        candidates = torch.stack((target, (target + 1) % 3), dim=1)
        losses = task_posterior_loss(prediction, target, candidate_y=candidates)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertIsNotNone(self.model.residual_decoder.output.weight.grad)

        ordinary = task_posterior_loss(prediction, target)
        self.assertEqual(float(ordinary.specialization), 0.0)
        self.assertGreater(float(ordinary.posterior.detach()), 0.0)
        self.assertTrue(torch.isfinite(ordinary.total))

    def test_regimes_are_matched_once_to_distinct_particles(self):
        prediction = self.model(self.context_x, self.context_y, self.query_x, class_count=3)
        active = torch.tensor([0, 1])
        candidate = torch.randint(0, 3, (2, 2, 6))
        assignment = match_regimes_to_particles(prediction, candidate)
        self.assertEqual(assignment.particle_for_regime.shape, (2, 2))
        self.assertTrue(all(len(set(row.tolist())) == 2 for row in assignment.particle_for_regime))
        loss = regime_posterior_supervision_loss(prediction, active, assignment)
        self.assertTrue(torch.isfinite(loss))
        matched = prediction.matched_marginal_probabilities(assignment.particle_for_regime)
        torch.testing.assert_close(matched.sum(-1), torch.ones_like(matched[..., 0]))

    def test_episode_objective_uses_candidate_stream_and_query_labels(self):
        batch = SimpleNamespace(
            initial_support_x=self.context_x[:, :5],
            initial_support_y=self.context_y[:, :5],
            stream_x=self.context_x[:, 5:],
            stream_y=self.context_y[:, 5:],
            query_x=self.query_x,
            query_y=torch.randint(0, 3, (2, 6)),
        )
        batch.candidate_stream_y = torch.stack((batch.stream_y, (batch.stream_y + 1) % 3), dim=1)
        batch.candidate_query_y = torch.stack((batch.query_y, (batch.query_y + 1) % 3), dim=1)
        objective = contrastive_episode_objective(self.model, batch, TaskPosteriorTrainingConfig())
        self.assertTrue(torch.isfinite(objective.total))
        self.assertGreater(float(objective.prior_only.specialization.detach()), 0)
        self.assertIsNotNone(objective.prior_only.assignment)
        torch.testing.assert_close(
            objective.prior_only.assignment.particle_for_regime,
            objective.updated.assignment.particle_for_regime,
        )
        self.assertEqual(
            choose_ordinary_episode(step=10, seed=4),
            choose_ordinary_episode(step=10, seed=4),
        )

    def test_checkpoint_roundtrip(self):
        expected = self.model(self.context_x, self.context_y, self.query_x, class_count=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            save_task_posterior_checkpoint(
                path,
                self.model,
                training_config={"seed": 41},
                lineage={"backbone": "tiny"},
                data_provenance={
                    "real_meta_training_datasets": [],
                    "synthetic_prior_families": ["unit-test"],
                    "tabarena_overlap": [],
                    "tabarena_checked_against_commit": "test-only",
                },
            )
            loaded, metadata = load_task_posterior_checkpoint(path)
        actual = loaded.eval()(self.context_x, self.context_y, self.query_x, class_count=3)
        torch.testing.assert_close(expected.particle_logits, actual.particle_logits, rtol=0, atol=0)
        self.assertEqual(metadata["architecture"]["particle_count"], 4)
        self.assertEqual(metadata["architecture"]["context_mode"], "iid_set")

    def test_sequential_mode_is_explicit(self):
        sequential = NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(), particle_count=4, max_classes=3, context_mode="sequential"
        ).eval()
        with self.assertRaisesRegex(ValueError, "forward_sequential"):
            sequential(self.context_x, self.context_y, self.query_x, class_count=3)
        result = sequential.forward_sequential(
            self.context_x[:, :6],
            self.context_y[:, :6],
            self.context_x[:, 6:],
            self.context_y[:, 6:],
            self.query_x,
            class_count=3,
        )
        self.assertEqual(result.particle_logits.shape, (2, 6, 4, 3))


class TaskPosteriorInterfaceTests(unittest.TestCase):
    def test_pinned_tabarena_checkout_revision_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "packages" / "tabarena" / "__init__.py"
            package.parent.mkdir(parents=True)
            package.touch()
            git = root / ".git"
            (git / "refs" / "heads").mkdir(parents=True)
            (git / "HEAD").write_text("ref: refs/heads/main\n")
            (git / "refs" / "heads" / "main").write_text("abc123\n")
            self.assertEqual(_installed_tabarena_revision(str(package)), "abc123")

    def test_sklearn_classifier_is_deterministic_and_preserves_labels(self):
        rng = np.random.default_rng(91)
        X = pd.DataFrame(
            {
                "number": rng.normal(size=30),
                "category": np.tile(["a", "b", "c"], 10),
            }
        )
        y = np.tile(np.array(["left", "middle", "right"]), 10)
        model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3)
        classifier = TaskPosteriorClassifier(
            model,
            particle_count=4,
            context_size=18,
            context_ensembles=2,
            random_state=7,
            device="cpu",
            query_chunk_size=4,
            num_mem_chunks=1,
        ).fit(X, y)
        first = classifier.predict_proba(X.iloc[:7])
        second = classifier.predict_proba(X.iloc[:7])
        np.testing.assert_allclose(first, second, atol=0, rtol=0)
        np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(first.shape, (7, 3))
        self.assertTrue(set(classifier.predict(X.iloc[:7])).issubset(set(y)))

    def test_autogluon_wrapper_declares_scope_without_optional_dependency(self):
        self.assertEqual(APPLICABILITY_CONSTRAINTS["max_train_rows"], 10_000)
        self.assertFalse(APPLICABILITY_CONSTRAINTS["regression"])
        self.assertEqual(TabArenaTaskPosteriorModel.get_applicability_constraints()["max_classes"], 10)

    def test_predeclared_acceptance_gates(self):
        adapter = np.linspace(0.65, 0.8, 40)
        baseline = adapter - 0.03
        result = paired_bootstrap_gate(adapter, baseline, bootstrap_samples=500, random_state=3)
        self.assertTrue(result.passes)
        self.assertGreater(result.confidence_low, 0)
        self.assertTrue(no_harm_gate([0.7, 0.8], [0.701, 0.801]))
        self.assertFalse(no_harm_gate([0.69, 0.79], [0.7, 0.8]))


if __name__ == "__main__":
    unittest.main()
