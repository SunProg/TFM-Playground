import unittest

from tfmplayground.experiments.detection_ceiling import (
    CLASS_COUNTS,
    SEPARATIONS,
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
        # The plain blocks: everything before the mechanism blocks were added.
        plain = [c for c in cells if c.get("mechanism", "plain") == "plain"]
        classification = [c for c in plain if c["target"] == "classification"]
        regression = [c for c in plain if c["target"] == "regression"]
        self.assertEqual(
            len(classification), len(SUPPORT_SIZES) * len(FEATURE_COUNTS) * len(CONTAMINATIONS) * len(CLASS_COUNTS)
        )
        # Separation cells, plus the gate-and-separation block on 12 features.
        self.assertEqual(
            len(regression),
            len(SEPARATIONS) * len(SUPPORT_SIZES) * len(FEATURE_COUNTS) * len(CONTAMINATIONS)
            + len(SEPARATIONS[1:]) * len(SUPPORT_SIZES) * len(CONTAMINATIONS),
        )
        # Every mechanism the module offers is actually exercised by the grid.
        self.assertEqual({c.get("mechanism", "plain") for c in cells}, {"plain", "repeated", "anchors"})
        # Classification first, so adding regression cells did not renumber them.
        self.assertEqual([c["target"] for c in cells[: len(classification)]], ["classification"] * len(classification))
        self.assertEqual(len({tuple(sorted(c.items())) for c in cells}), len(cells))

    def test_regression_indices_are_stable_across_the_separation_block(self):
        """Indices 32-39 were already in flight when separation was added, and a
        pending array task reads this grid when it starts -- so appending must
        not move them."""
        cells = ceiling_cells()
        regression = [(i, c) for i, c in enumerate(cells) if c["target"] == "regression"]
        zero = [i for i, c in regression if c["separation"] == 0.0]
        self.assertEqual(len(zero), len(SUPPORT_SIZES) * len(FEATURE_COUNTS) * len(CONTAMINATIONS))
        # Every zero-separation regression cell precedes every separated one, so
        # appending separation did not move indices 32-39.  Scoped to regression
        # cells: classification cells carry no separation key at all.
        separated = [i for i, c in regression if c["separation"] > 0.0]
        self.assertGreater(min(separated), max(zero))
        self.assertEqual(sorted({c["separation"] for _, c in regression}), sorted(SEPARATIONS))

    def test_separation_pushes_the_rules_apart(self):
        """The half of mixture-of-experts identifiability the prior never had:
        two independent draws land wherever chance puts them, so raising this
        must measurably increase the gap between the rules."""
        import numpy as np

        from tfmplayground.experiments.detection_ceiling import _continuous_candidates

        gaps = {}
        for separation in (0.0, 0.9):
            rng = np.random.default_rng(4)
            gaps[separation] = float(
                np.mean(
                    [
                        np.mean(np.abs(rules[0] - rules[1]))
                        for _, rules in (_continuous_candidates(72, 4, rng, separation=separation) for _ in range(3))
                    ]
                )
            )
        self.assertGreater(gaps[0.9], 1.3 * gaps[0.0])

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
