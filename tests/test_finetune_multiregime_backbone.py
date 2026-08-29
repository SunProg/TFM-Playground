import unittest

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import SCM_FAMILIES
from tfmplayground.experiments.finetune_multiregime_backbone import (
    BackboneFinetuneConfig,
    _draw_condition,
    _draw_multiregime_source,
    _draw_scm_family,
    finetune,
    multiregime_diagnostics,
    query_cross_entropy,
    sample_condition_episode,
    validate,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone(seed: int = 0) -> NanoTabPFNModel:
    torch.manual_seed(seed)
    return NanoTabPFNModel(16, 2, 32, 2, 3)


class CurriculumTests(unittest.TestCase):
    def test_draw_condition_respects_probability(self):
        rng = np.random.default_rng(11)
        drawn = {_draw_condition(rng, 0.3) for _ in range(200)}
        self.assertEqual(drawn, {"prior", "multiregime"})

    def test_draw_condition_extremes(self):
        rng = np.random.default_rng(0)
        self.assertTrue(all(_draw_condition(rng, 0.0) == "prior" for _ in range(20)))
        rng = np.random.default_rng(1)
        self.assertTrue(all(_draw_condition(rng, 1.0) == "multiregime" for _ in range(20)))

    def test_draw_scm_family_covers_both_pretraining_families(self):
        rng = np.random.default_rng(2)
        self.assertEqual({_draw_scm_family(rng) for _ in range(100)}, set(SCM_FAMILIES))

    def test_draw_multiregime_source_includes_cross_family_case(self):
        rng = np.random.default_rng(3)
        drawn = {_draw_multiregime_source(rng) for _ in range(100)}
        self.assertEqual(drawn, {"mlp_scm", "tree_scm", ("mlp_scm", "tree_scm")})


class EpisodeSamplingTests(unittest.TestCase):
    """Both curriculum shares come from the official TabICL SCM prior, same family."""

    def test_multiregime_condition_returns_regime_tagged_episode(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(0), "multiregime", config, family="mlp_scm")
        self.assertEqual(episode.condition, "multiregime")
        self.assertEqual(episode.family, "mlp_scm")
        self.assertIsNotNone(episode.query_regime_source)

    def test_prior_condition_has_no_regime_tag_and_matching_family(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(0), "prior", config, family="mlp_scm")
        self.assertEqual(episode.family, "mlp_scm")
        self.assertIsNone(episode.query_regime_source)

    def test_tree_scm_is_available_for_both_conditions(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        prior_episode = sample_condition_episode(np.random.default_rng(2), "prior", config, family="tree_scm")
        multiregime_episode = sample_condition_episode(np.random.default_rng(3), "multiregime", config, family="tree_scm")
        self.assertEqual(prior_episode.family, "tree_scm")
        self.assertEqual(multiregime_episode.family, "tree_scm")

    def test_cross_family_multiregime_uses_one_shared_feature_table(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        episode = sample_condition_episode(
            np.random.default_rng(4),
            "multiregime",
            config,
            family=("mlp_scm", "tree_scm"),
        )
        self.assertEqual(episode.family, "mlp_scm+tree_scm")
        self.assertEqual(episode.support_x.shape[-1], episode.query_x.shape[-1])
        self.assertIsNotNone(episode.query_regime_source)


class LossAndDiagnosticsTests(unittest.TestCase):
    def test_query_cross_entropy_is_finite_and_differentiable(self):
        model = tiny_backbone(1)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(2), "multiregime", config)
        loss = query_cross_entropy(model, episode)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(norm > 0 for norm in gradient_norms))

    def test_diagnostics_no_leakage_and_expected_keys(self):
        """multiregime_diagnostics reads only support/query features and the diagnostic-only regime tag."""
        model = tiny_backbone(3)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(5), "multiregime", config)
        result = multiregime_diagnostics(model, episode)
        for key in ("multiregime_base_error", "multiregime_other_error", "multiregime_error_gap"):
            self.assertIn(key, result)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_diagnostics_empty_for_episodes_without_regime_tag(self):
        model = tiny_backbone(4)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(6), "prior", config)
        self.assertEqual(multiregime_diagnostics(model, episode), {})


class TrainingLoopTests(unittest.TestCase):
    def test_finetune_runs_and_updates_every_parameter(self):
        model = tiny_backbone(7)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        config = BackboneFinetuneConfig(
            batch_size=1,
            support_size=16,
            query_count=4,
            max_steps=6,
            validation_interval=3,
            validation_episodes=2,
            patience=100,
            device="cpu",
        )
        trained_model, history, selection = finetune(model, config)
        self.assertEqual(len(history), config.max_steps)
        self.assertIn("best_step", selection)
        changed = [
            name
            for name, parameter in trained_model.named_parameters()
            if not torch.equal(parameter.detach(), before[name])
        ]
        # Every parameter group should get at least one gradient step across a
        # curriculum that touches both conditions, including the decoder and
        # feature/target encoders which both conditions exercise the same way.
        self.assertTrue(len(changed) >= len(before) // 2)

    def test_fixed_budget_saves_requested_trajectory_steps(self):
        model = tiny_backbone(9)
        config = BackboneFinetuneConfig(
            batch_size=1,
            support_size=16,
            query_count=4,
            max_steps=6,
            validation_interval=2,
            checkpoint_interval=2,
            validation_episodes=2,
            early_stopping=False,
            device="cpu",
        )
        saved_steps = []
        _, history, selection = finetune(
            model,
            config,
            checkpoint_callback=lambda _model, step, _validation: saved_steps.append(step),
        )
        self.assertEqual([row["step"] for row in history], list(range(1, 7)))
        self.assertEqual(saved_steps, [2, 4, 6])
        self.assertFalse(selection["early_stopping"])
        self.assertEqual(selection["executed_steps"], 6)

    def test_validate_returns_finite_metrics(self):
        model = tiny_backbone(8)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, validation_episodes=6, device="cpu")
        metrics = validate(model, config)
        self.assertIn("cross_entropy", metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
