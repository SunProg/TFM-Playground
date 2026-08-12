import unittest

import torch

from tfmplayground.experiments.structural_latents import StructuralLatentSchema, probe_r2
from tfmplayground.experiments.structural_probe_analysis import (
    CAPACITIES,
    REPRESENTATIONS,
    StructuralProbeConfig,
    _build_probe,
    _splits,
    fit_probe,
    score_representations,
)


def tiny_config(**kwargs) -> StructuralProbeConfig:
    defaults = dict(
        checkpoint="unused.pth",
        output_dir="unused",
        episodes=32,
        probe_epochs=250,
        patience=40,
        hidden_size=16,
        learning_rates=(1e-2,),
        max_features=3,
    )
    defaults.update(kwargs)
    return StructuralProbeConfig(**defaults)


class ProbeConfigTests(unittest.TestCase):
    def test_rejects_degenerate_splits_and_budgets(self):
        with self.assertRaises(ValueError):
            tiny_config(episodes=4).validate()
        with self.assertRaises(ValueError):
            tiny_config(validation_fraction=0.5, test_fraction=0.4).validate()
        with self.assertRaises(ValueError):
            tiny_config(learning_rates=()).validate()
        with self.assertRaises(ValueError):
            tiny_config(patience=0).validate()
        tiny_config().validate()


class SplitTests(unittest.TestCase):
    def test_splits_partition_every_row_exactly_once(self):
        config = tiny_config()
        train, validation, test = _splits(400, config)
        combined = torch.cat((train, validation, test)).sort().values
        torch.testing.assert_close(combined, torch.arange(400))
        self.assertEqual(len(validation), 60)
        self.assertEqual(len(test), 60)

    def test_splits_are_deterministic_under_a_fixed_seed(self):
        first = _splits(200, tiny_config(seed=7))[0]
        second = _splits(200, tiny_config(seed=7))[0]
        third = _splits(200, tiny_config(seed=8))[0]
        torch.testing.assert_close(first, second)
        self.assertFalse(bool(torch.equal(first, third)))


class ProbeCapacityTests(unittest.TestCase):
    def test_every_capacity_builds_and_maps_to_the_target_width(self):
        for capacity in CAPACITIES:
            probe = _build_probe(12, 5, capacity, hidden_size=16)
            self.assertEqual(tuple(probe(torch.randn(7, 12)).shape), (7, 5))

    def test_unknown_capacity_is_refused(self):
        with self.assertRaises(ValueError):
            _build_probe(12, 5, "transformer", hidden_size=16)

    def test_deeper_capacities_have_more_parameters(self):
        sizes = [sum(p.numel() for p in _build_probe(12, 5, c, hidden_size=16).parameters()) for c in CAPACITIES]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])


class ProbeFitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = tiny_config()
        self.schema = StructuralLatentSchema(max_features=3)
        rows, width = 600, 8
        generator = torch.Generator().manual_seed(1)
        self.features = torch.randn(rows, width, generator=generator)
        projection = torch.randn(width, self.schema.latent_dim, generator=generator)
        self.targets = self.features @ projection

    def test_a_linearly_recoverable_target_is_recovered(self):
        predictions, diagnostics = fit_probe(self.features, self.targets, self.config, capacity="linear")
        _, _, test_index = _splits(self.features.shape[0], self.config)
        report = probe_r2(predictions, self.targets[test_index], self.schema)
        self.assertGreater(report["mean_continuous_r2"], 0.95)
        self.assertGreater(diagnostics["train_rows"], 0)
        self.assertEqual(diagnostics["test_rows"], float(len(test_index)))

    def test_an_uninformative_representation_cannot_beat_the_mean(self):
        generator = torch.Generator().manual_seed(2)
        noise = torch.randn_like(self.features)
        predictions, _ = fit_probe(noise, self.targets, self.config, capacity="linear")
        _, _, test_index = _splits(noise.shape[0], self.config)
        report = probe_r2(predictions, self.targets[test_index], self.schema)
        # Predicting the mean scores exactly zero; noise cannot do meaningfully better.
        self.assertLess(report["mean_continuous_r2"], 0.05)
        del generator

    def test_a_shuffled_target_is_not_recoverable(self):
        generator = torch.Generator().manual_seed(3)
        shuffled = self.targets[torch.randperm(self.targets.shape[0], generator=generator)]
        predictions, _ = fit_probe(self.features, shuffled, self.config, capacity="linear")
        _, _, test_index = _splits(self.features.shape[0], self.config)
        report = probe_r2(predictions, shuffled[test_index], self.schema)
        self.assertLess(report["mean_continuous_r2"], 0.05)

    def test_fitting_is_deterministic_under_a_fixed_seed(self):
        first, _ = fit_probe(self.features, self.targets, self.config, capacity="mlp1")
        second, _ = fit_probe(self.features, self.targets, self.config, capacity="mlp1")
        torch.testing.assert_close(first, second, rtol=0, atol=0)


class ScoreRepresentationsTests(unittest.TestCase):
    def test_scores_cover_every_cell_and_separate_the_candidate_subsets(self):
        torch.manual_seed(0)
        config = tiny_config(probe_epochs=40, patience=10)
        schema = StructuralLatentSchema(max_features=config.max_features)
        rows = 400
        generator = torch.Generator().manual_seed(5)
        cached = {name: torch.randn(rows, 6, generator=generator) for name in REPRESENTATIONS}
        cached["targets"] = torch.rand(rows, schema.latent_dim, generator=generator)
        cached["is_true_candidate"] = torch.arange(rows) % 2 == 0

        frame = score_representations(cached, config)
        self.assertEqual(set(frame.representation), set(REPRESENTATIONS))
        self.assertEqual(set(frame.capacity), set(CAPACITIES))
        self.assertEqual(set(frame.targets), {"real", "shuffled_null"})
        self.assertEqual(set(frame.subset), {"all", "true_candidate", "false_candidate"})
        expected = len(REPRESENTATIONS) * len(CAPACITIES) * 2 * 3
        self.assertEqual(len(frame), expected)
        self.assertTrue(frame.mean_continuous_r2.notna().all())

    def test_random_features_score_at_or_below_zero_on_random_targets(self):
        torch.manual_seed(0)
        config = tiny_config(probe_epochs=40, patience=10)
        schema = StructuralLatentSchema(max_features=config.max_features)
        generator = torch.Generator().manual_seed(6)
        cached = {name: torch.randn(400, 6, generator=generator) for name in REPRESENTATIONS}
        cached["targets"] = torch.rand(400, schema.latent_dim, generator=generator)
        cached["is_true_candidate"] = torch.zeros(400, dtype=torch.bool)
        frame = score_representations(cached, config)
        self.assertLess(frame.mean_continuous_r2.max(), 0.10)


if __name__ == "__main__":
    unittest.main()
