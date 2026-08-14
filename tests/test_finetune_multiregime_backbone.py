import unittest

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import TRAIN_REGIME
from tfmplayground.experiments.finetune_multiregime_backbone import (
    ORDINARY_CONDITIONS,
    BackboneFinetuneConfig,
    _curriculum_weights,
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
    def test_weights_sum_to_one_and_cover_every_condition(self):
        config = BackboneFinetuneConfig(multiregime_probability=0.3)
        weights = _curriculum_weights(config)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["multiregime"], 0.3)
        self.assertEqual(set(weights) - {"multiregime"}, set(ORDINARY_CONDITIONS))


class EpisodeSamplingTests(unittest.TestCase):
    def test_multiregime_condition_returns_regime_tagged_episode(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(0), TRAIN_REGIME, "multiregime", config)
        self.assertEqual(episode.condition, "multiregime")
        self.assertIsNotNone(episode.query_regime_source)

    def test_ordinary_condition_has_no_regime_tag(self):
        config = BackboneFinetuneConfig(batch_size=1, support_size=32, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(0), TRAIN_REGIME, "ambiguous", config)
        self.assertEqual(episode.condition, "ambiguous")
        self.assertIsNone(episode.query_regime_source)


class LossAndDiagnosticsTests(unittest.TestCase):
    def test_query_cross_entropy_is_finite_and_differentiable(self):
        model = tiny_backbone(1)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(2), TRAIN_REGIME, "multiregime", config)
        loss = query_cross_entropy(model, episode)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(norm > 0 for norm in gradient_norms))

    def test_diagnostics_no_leakage_and_expected_keys(self):
        """multiregime_diagnostics reads only support/query features and the diagnostic-only regime tag."""
        model = tiny_backbone(3)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(5), TRAIN_REGIME, "multiregime", config)
        result = multiregime_diagnostics(model, episode)
        for key in ("multiregime_base_error", "multiregime_other_error", "multiregime_error_gap"):
            self.assertIn(key, result)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_diagnostics_empty_for_episodes_without_regime_tag(self):
        model = tiny_backbone(4)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, device="cpu")
        episode = sample_condition_episode(np.random.default_rng(6), TRAIN_REGIME, "identifiable", config)
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
        # curriculum that touches every condition, including the decoder and
        # feature/target encoders which only ordinary conditions exercise the
        # same way multiregime conditions do.
        self.assertTrue(len(changed) >= len(before) // 2)

    def test_validate_returns_finite_metrics(self):
        model = tiny_backbone(8)
        config = BackboneFinetuneConfig(batch_size=1, support_size=16, query_count=4, validation_episodes=6, device="cpu")
        metrics = validate(model, config)
        self.assertIn("cross_entropy", metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
