import json
import tempfile
import unittest
from pathlib import Path

from tfmplayground.experiments.multiregime_sweep import (
    FINAL_SEEDS,
    configuration_flags,
    configuration_label,
    screening_configurations,
    summarize_sweep,
)


class GridTests(unittest.TestCase):
    def test_grid_size_and_uniqueness(self):
        configurations = screening_configurations()
        self.assertEqual(len(configurations), 18)
        labels = {configuration_label(configuration) for configuration in configurations}
        self.assertEqual(len(labels), 18)

    def test_configuration_flags_round_trips_every_axis(self):
        configurations = screening_configurations()
        for index, configuration in enumerate(configurations):
            flags = configuration_flags(index)
            self.assertIn(f"--learning-rate {configuration['learning_rate']:g}", flags)
            self.assertIn(f"--multiregime-probability {configuration['multiregime_probability']:g}", flags)
            self.assertIn(f"--support-size {configuration['support_size']}", flags)

    def test_final_uses_larger_budget_and_given_seed(self):
        screening_flags = configuration_flags(0)
        final_flags = configuration_flags(0, final=True, seed=2403)
        self.assertIn("--seed 2403", final_flags)
        self.assertNotIn("--seed 2403", screening_flags)

    def test_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            configuration_flags(18)


class SummarizeTests(unittest.TestCase):
    def test_ranks_by_validation_cross_entropy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (loss, learning_rate) in enumerate([(0.9, 1e-5), (0.5, 3e-5), (0.7, 1e-4)]):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "learning_rate": learning_rate,
                            "multiregime_probability": 0.3,
                            "support_size": 128,
                            "seed": 2402,
                        }
                    )
                )
                (run_dir / "selection.json").write_text(
                    json.dumps({"best_step": 100, "best_validation_cross_entropy": loss})
                )
            summary = summarize_sweep(root, top_k=1)
            self.assertEqual(len(summary["runs"]), 3)
            self.assertEqual(summary["selected"][0]["learning_rate"], 3e-5)
            self.assertEqual(summary["final_seeds"], list(FINAL_SEEDS))

    def test_skips_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "config.json").write_text("{}")
            # No selection.json: this run should be skipped, not crash.
            summary = summarize_sweep(root)
            self.assertEqual(summary["runs"], [])


if __name__ == "__main__":
    unittest.main()
