import copy
import dataclasses
import unittest

import numpy as np
import torch

from tfmplayground.experiments.train_four_mode_particle_filter import (
    COMPETING_EVIDENCE_CENTER,
    FOUR_MODE_VECTORS,
    FourModeConfig,
    _effective_features,
    four_mode_loss,
    generate_four_mode_episodes,
    validate_config,
)
from tfmplayground.models.adaptive_particle_filter import expand_two_to_k_particles
from tfmplayground.models.integrated_latent_filter import NanoTabPFNIntegratedLatentFilter
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=2,
        num_outputs=3,
    )


class FourModeParticleFilterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.config = FourModeConfig(
            device="cpu",
            prior_count=8,
            update_count=8,
            query_count=4,
            batch_size=2,
            accumulate_gradients=1,
            steps=1,
            evaluation_trials=2,
        )
        source = NanoTabPFNIntegratedLatentFilter(tiny_backbone())
        self.model = expand_two_to_k_particles(source, tiny_backbone(), particle_count=4)

    def test_generator_exposes_all_four_coherent_modes(self):
        for mode in range(4):
            batch = generate_four_mode_episodes(
                self.config,
                np.random.default_rng(29),
                condition=f"consistent_{mode}",
                batch_size=2,
            )
            expected = FOUR_MODE_VECTORS[mode].expand(2, -1)
            torch.testing.assert_close(batch.query_y, expected)
            self.assertEqual(batch.posterior.shape, (2, 9, 4))
            torch.testing.assert_close(batch.posterior.sum(-1), torch.ones_like(batch.posterior[..., 0]))
            self.assertTrue((batch.posterior[:, -1, mode] > 0.99).all())

    def test_label_does_not_change_own_preupdate_prediction(self):
        batch = generate_four_mode_episodes(self.config, np.random.default_rng(30), condition="noisy", batch_size=2)
        original = self.model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        changed = copy.copy(batch)
        changed.stream_y = batch.stream_y.clone()
        changed.stream_y[:, 3] = 1 - changed.stream_y[:, 3]
        altered = self.model(
            changed.initial_support_x,
            changed.initial_support_y,
            changed.stream_x,
            changed.stream_y,
            changed.query_x,
        )
        torch.testing.assert_close(original.stream_logits[:, 3], altered.stream_logits[:, 3], rtol=0, atol=0)
        torch.testing.assert_close(original.log_weights[:, 3], altered.log_weights[:, 3], rtol=0, atol=0)
        self.assertFalse(torch.equal(original.log_weights[:, 4], altered.log_weights[:, 4]))

    def test_four_mode_loss_is_finite_and_trains_particle_head(self):
        batch = generate_four_mode_episodes(
            self.config, np.random.default_rng(31), condition="consistent_2", batch_size=2
        )
        prediction = self.model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        loss, metrics = four_mode_loss(prediction, batch, self.config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.ambiguity_gate.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.particle_model.decoder.parameters()))
        self.assertTrue(np.isfinite(list(metrics.values())).all())


class CompetingFeatureTests(unittest.TestCase):
    """feature_mode="competing": regions are hosted on columns, rivals come from the
    prior's own uniform(-1, 1) range rather than being separable N(0, 1) noise."""

    def setUp(self):
        torch.manual_seed(29)
        self.noise = FourModeConfig(
            device="cpu",
            prior_count=8,
            update_count=8,
            query_count=4,
            batch_size=2,
            accumulate_gradients=1,
            steps=1,
            evaluation_trials=2,
        )
        self.competing = dataclasses.replace(self.noise, feature_mode="competing")
        source = NanoTabPFNIntegratedLatentFilter(tiny_backbone())
        self.model = expand_two_to_k_particles(source, tiny_backbone(), particle_count=4)

    def _episode(self, config, **kwargs):
        return generate_four_mode_episodes(config, np.random.default_rng(37), **kwargs)

    def test_default_mode_is_noise(self):
        self.assertEqual(FourModeConfig().feature_mode, "noise")

    def test_noise_mode_leaves_the_informative_column_untouched_by_width(self):
        """Regression guard on the shared signal path: adding noise columns must not
        perturb column 0, so an F=3 episode's column 0 equals the F=1 episode's."""
        for condition in ("neutral", "consistent_2", "contradictory"):
            single = self._episode(self.noise, condition=condition, num_features=1)
            wide = self._episode(self.noise, condition=condition, num_features=3)
            self.assertEqual(wide.stream_x.shape[-1], 3)
            for name in ("initial_support_x", "stream_x", "query_x"):
                torch.testing.assert_close(getattr(wide, name)[..., :1], getattr(single, name), rtol=0, atol=0)

    def test_competing_mode_clamps_width_up_to_one_column_per_region(self):
        self.assertEqual(_effective_features(1, 4, "competing"), 2)
        self.assertEqual(_effective_features(5, 4, "competing"), 5)
        self.assertEqual(_effective_features(1, 1, "competing"), 1)
        # The noise encoder has no such floor.
        self.assertEqual(_effective_features(1, 4, "noise"), 1)
        batch = self._episode(self.competing, condition="consistent_1", num_features=1, num_modes=4)
        self.assertEqual(batch.stream_x.shape[-1], 2)

    def test_competing_rows_host_evidence_in_exactly_one_column(self):
        for num_modes in (2, 4):
            for num_features in (2, 3, 5):
                batch = self._episode(
                    self.competing,
                    condition="consistent_1",
                    num_modes=num_modes,
                    num_features=num_features,
                )
                for table in (batch.stream_x, batch.query_x):
                    values = table.numpy()
                    self.assertEqual(values.shape[-1], num_features)
                    evidence = values > 1.0
                    self.assertTrue((evidence.sum(-1) == 1).all())
                    rivals = values[~evidence]
                    self.assertTrue((np.abs(rivals) <= 1.0).all())
                # Every region's cluster sits at the same value: identity is the column.
                hosted = batch.query_x.numpy()[batch.query_x.numpy() > 1.0]
                self.assertTrue(
                    (np.abs(hosted - COMPETING_EVIDENCE_CENTER) <= 0.15).all(),
                    "competing regions must share one cluster center",
                )

    def test_competing_support_stays_inside_the_prior_range(self):
        """The support must not reveal which column hosts a region -- every support
        column, informative or rival, lies in the same uniform(-1, 1) range."""
        batch = self._episode(self.competing, condition="consistent_3", num_modes=4, num_features=5)
        support = batch.initial_support_x.numpy()
        self.assertEqual(support.shape[-1], 5)
        self.assertTrue((np.abs(support) <= 1.0).all())

    def test_competing_hosting_columns_are_not_pinned(self):
        widths = set()
        for seed in range(12):
            batch = generate_four_mode_episodes(
                self.competing,
                np.random.default_rng(seed),
                condition="consistent_1",
                num_modes=2,
                num_features=4,
            )
            evidence = batch.query_x.numpy() > 1.0
            widths.add(tuple(evidence[0].argmax(-1).tolist()))
        self.assertGreater(len(widths), 1, "hosting column must be resampled per episode")

    def test_trivial_mode_hosts_the_prior_rule_on_one_column(self):
        batch = self._episode(self.competing, num_modes=1, num_features=4)
        self.assertEqual(batch.stream_x.shape[-1], 4)
        self.assertTrue((np.abs(batch.stream_x.numpy()) <= 1.0).all())
        self.assertEqual(batch.query_x.shape[1], 2)

    def test_competing_label_does_not_change_own_preupdate_prediction(self):
        batch = self._episode(self.competing, condition="noisy", num_modes=4, num_features=3)
        original = self.model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        changed = copy.copy(batch)
        changed.stream_y = batch.stream_y.clone()
        changed.stream_y[:, 3] = 1 - changed.stream_y[:, 3]
        altered = self.model(
            changed.initial_support_x,
            changed.initial_support_y,
            changed.stream_x,
            changed.stream_y,
            changed.query_x,
        )
        torch.testing.assert_close(original.stream_logits[:, 3], altered.stream_logits[:, 3], rtol=0, atol=0)
        self.assertFalse(torch.equal(original.log_weights[:, 4], altered.log_weights[:, 4]))

    def test_competing_loss_is_finite_and_trains_the_same_parameter_groups(self):
        batch = self._episode(self.competing, condition="consistent_2", num_modes=4, num_features=5)
        prediction = self.model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        loss, metrics = four_mode_loss(prediction, batch, self.competing)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.ambiguity_gate.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.particle_model.decoder.parameters()))
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_unknown_feature_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(dataclasses.replace(self.noise, feature_mode="rivals"))


if __name__ == "__main__":
    unittest.main()
