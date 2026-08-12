import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.experiments.region_split_episodes import (
    RegionSplitConfig,
    generate_region_split_episodes,
    region_split_acceptance,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def write_dump(path: Path, records: int = 400, rows: int = 150, features: int = 5, seed: int = 0) -> str:
    """A dump whose labels are a real function of the features.

    A threshold rule on feature 0 plus a little noise, so a region split leaves
    both sides populated and the prior block is genuinely two-class.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(records, rows, features)).astype(np.float32)
    y = (x[:, :, 0] + 0.5 * x[:, :, 1] > 0).astype(np.float32)
    flip = rng.random((records, rows)) < 0.05
    y = np.where(flip, 1 - y, y)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("num_features", data=np.full(records, features, dtype="i4"))
        handle.create_dataset("num_datapoints", data=np.full(records, rows, dtype="i4"))
        handle.create_dataset("train_test_split_index", data=np.full(records, rows // 2, dtype="i4"))
        handle.create_dataset("max_num_classes", data=np.array([2]))
        handle.create_dataset("original_batch_size", data=np.array([1]))
        handle.create_dataset("problem_type", data="classification", dtype=h5py.string_dtype())
    return str(path)


def small_config(**kwargs) -> RegionSplitConfig:
    defaults = dict(initial_support_count=24, stream_count=8, query_count=8, min_features=5, max_features=5)
    defaults.update(kwargs)
    return RegionSplitConfig(**defaults)


class RegionSplitConfigTests(unittest.TestCase):
    def test_second_scm_requires_the_on_the_fly_prior(self):
        with self.assertRaises(ValueError):
            small_config(alternative_mode="second_scm", source="h5").validate()
        small_config(alternative_mode="second_scm", source="tabicl").validate()

    def test_rejects_unknown_modes_and_degenerate_layouts(self):
        with self.assertRaises(ValueError):
            small_config(alternative_mode="swap").validate()
        with self.assertRaises(ValueError):
            small_config(source="openml").validate()
        with self.assertRaises(ValueError):
            small_config(initial_support_count=2).validate()
        with self.assertRaises(ValueError):
            small_config(split_quantile_range=(0.8, 0.2)).validate()
        with self.assertRaises(ValueError):
            small_config(split_quantile_range=(0.0, 0.5)).validate()


class RegionSplitEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = write_dump(Path(cls._directory.name) / "dump.h5")

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def batch(self, **kwargs):
        config = small_config(**kwargs)
        return config, generate_region_split_episodes(config, np.random.default_rng(0), batch_size=4, path=self.path)

    def test_prior_block_is_exactly_consistent_with_both_candidates(self):
        _, batch = self.batch()
        torch.testing.assert_close(
            batch.candidate_support_y[:, 0], batch.candidate_support_y[:, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(batch.support_disagreement, torch.zeros(4), rtol=0, atol=0)

    def test_every_stream_and_query_row_discriminates(self):
        _, batch = self.batch()
        self.assertTrue(bool((batch.candidate_stream_y[:, 0] != batch.candidate_stream_y[:, 1]).all()))
        self.assertTrue(bool((batch.candidate_query_y[:, 0] != batch.candidate_query_y[:, 1]).all()))
        torch.testing.assert_close(batch.stream_disagreement, torch.ones(4), rtol=0, atol=0)
        torch.testing.assert_close(batch.query_disagreement, torch.ones(4), rtol=0, atol=0)

    def test_observed_labels_follow_the_true_candidate_on_every_block(self):
        _, batch = self.batch()
        index = torch.arange(4)
        task = batch.candidate_task
        torch.testing.assert_close(
            batch.initial_support_y, batch.candidate_support_y[index, task].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(batch.stream_y, batch.candidate_stream_y[index, task].float(), rtol=0, atol=0)
        torch.testing.assert_close(batch.query_y, batch.candidate_query_y[index, task].long(), rtol=0, atol=0)

    def test_block_shapes_match_the_configured_layout(self):
        config, batch = self.batch()
        self.assertEqual(tuple(batch.initial_support_x.shape[:2]), (4, config.initial_support_count))
        self.assertEqual(tuple(batch.stream_x.shape[:2]), (4, config.stream_count))
        self.assertEqual(tuple(batch.query_x.shape[:2]), (4, config.query_count))

    def test_structural_targets_differ_between_candidates(self):
        _, batch = self.batch()
        self.assertIsNotNone(batch.candidate_structural_z)
        gap = (batch.candidate_structural_z[:, 0] - batch.candidate_structural_z[:, 1]).abs().max()
        self.assertGreater(float(gap), 0.0)
        self.assertEqual(tuple(batch.structural_feature_mask.shape), (4, 5))

    def test_structural_targets_can_be_disabled(self):
        _, batch = self.batch(compute_structural_latents=False)
        self.assertIsNone(batch.candidate_structural_z)
        self.assertIsNone(batch.structural_feature_mask)

    def test_generation_is_deterministic_under_a_fixed_seed(self):
        config = small_config()
        first = generate_region_split_episodes(config, np.random.default_rng(3), batch_size=4, path=self.path)
        second = generate_region_split_episodes(config, np.random.default_rng(3), batch_size=4, path=self.path)
        torch.testing.assert_close(first.initial_support_x, second.initial_support_x, rtol=0, atol=0)
        torch.testing.assert_close(first.candidate_task, second.candidate_task, rtol=0, atol=0)

    def test_a_dump_path_is_required_for_the_h5_source(self):
        with self.assertRaises(ValueError):
            generate_region_split_episodes(small_config(), np.random.default_rng(0), batch_size=2)

    def test_impossible_layouts_are_reported_rather_than_silently_shrunk(self):
        config = small_config(initial_support_count=140, stream_count=64, query_count=64)
        with self.assertRaises(RuntimeError):
            generate_region_split_episodes(config, np.random.default_rng(0), batch_size=2, path=self.path)


class RegionSplitAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = write_dump(Path(cls._directory.name) / "dump.h5", records=800)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_acceptance_reports_the_construction_invariants_exactly(self):
        torch.manual_seed(0)
        backbone = NanoTabPFNModel(
            embedding_size=8, num_attention_heads=2, mlp_hidden_size=16, num_layers=2, num_outputs=2
        )
        report = region_split_acceptance(
            small_config(),
            backbone,
            np.random.default_rng(0),
            episodes=16,
            batch_size=4,
            path=self.path,
        )
        self.assertEqual(report["support_disagreement"], 0.0)
        self.assertEqual(report["stream_disagreement"], 1.0)
        self.assertEqual(report["query_disagreement"], 1.0)
        self.assertEqual(report["discriminating_query_rows"], 8.0)
        # An untrained backbone cannot identify anything; the invariants above are
        # what this test pins.  The ambiguous/resolvable flags are measured for
        # real against a pretrained checkpoint, not asserted here.
        for key in ("prior_only_identification", "prior_stream_identification", "identification_gain"):
            self.assertTrue(np.isfinite(report[key]))


if __name__ == "__main__":
    unittest.main()
