import unittest

from tfmplayground.experiments.detection_ceiling import (
    CLASS_COUNTS,
    CONTAMINATIONS,
    FEATURE_COUNTS,
    SUPPORT_SIZES,
    ceiling_cells,
    main,
    measure,
)


class CeilingGridTests(unittest.TestCase):
    def test_grid_covers_both_targets_and_is_append_only(self):
        cells = ceiling_cells()
        classification = [c for c in cells if c["target"] == "classification"]
        regression = [c for c in cells if c["target"] == "regression"]
        self.assertEqual(
            len(classification), len(SUPPORT_SIZES) * len(FEATURE_COUNTS) * len(CONTAMINATIONS) * len(CLASS_COUNTS)
        )
        self.assertEqual(len(regression), len(SUPPORT_SIZES) * len(FEATURE_COUNTS) * len(CONTAMINATIONS))
        # Classification first, so adding regression cells did not renumber them.
        self.assertEqual([c["target"] for c in cells[: len(classification)]], ["classification"] * len(classification))
        self.assertEqual(len({tuple(sorted(c.items())) for c in cells}), len(cells))

    def test_index_bounds(self):
        for index in (-1, len(ceiling_cells())):
            with self.assertRaises((IndexError, SystemExit)):
                main([f"--index={index}"])


class MeasurementTests(unittest.TestCase):
    """Two episodes only: this asserts the contract, not the ceiling itself."""

    def test_classification_cell_reports_a_reachable_ceiling(self):
        cell = {"target": "classification", "num_classes": 3, "features": 4, "contamination": 0.3, "support": 64}
        row = measure(cell, episodes=2, seed=1)
        self.assertGreater(row["restricted"]["n"], 0)
        # Restricted drops the collided positives, so its ceiling is 1.0 while
        # the unrestricted one is capped by how many rows are identifiable.
        self.assertLess(row["unrestricted_ceiling"], 1.0)
        self.assertAlmostEqual(
            row["unrestricted_ceiling"], row["identifiable"] + (1 - row["identifiable"]) * 0.5, places=6
        )
        self.assertIsNone(row["rule_separation"])

    def test_regression_cell_has_no_collided_positives(self):
        """The whole point of a continuous target: two independent rules never
        agree, so every contaminated row is identifiable and the two scorings
        coincide."""
        cell = {"target": "regression", "num_classes": 0, "features": 4, "contamination": 0.3, "support": 64}
        row = measure(cell, episodes=2, seed=1)
        self.assertEqual(row["identifiable"], 1.0)
        self.assertEqual(row["unrestricted_ceiling"], 1.0)
        self.assertEqual(row["restricted"]["mean"], row["unrestricted"]["mean"])
        self.assertIsNotNone(row["rule_separation"])


if __name__ == "__main__":
    unittest.main()
