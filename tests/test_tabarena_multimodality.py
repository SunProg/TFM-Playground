import json
import unittest

import numpy as np
import pandas as pd

from tfmplayground.experiments.evaluate_tabarena_multimodality import (
    HYPOTHESIS_NAMES,
    MultimodalityConfig,
    _effective_count,
    _fit_hypotheses,
    _hypothesis_metrics,
    _pairwise_js,
    _sample_episode,
    _softmax_log_weights,
    _update_weights,
)


class TabArenaMultimodalityTests(unittest.TestCase):
    def test_identical_hypotheses_have_zero_disagreement(self):
        probabilities = np.full((6, 4), 0.4)
        weights = np.full(4, 0.25)
        config = MultimodalityConfig(include_vanilla=False)
        metrics = _hypothesis_metrics(probabilities, weights, config)
        self.assertEqual(_pairwise_js(probabilities, weights), 0.0)
        self.assertEqual(metrics["max_pairwise_disagreement"], 0.0)
        self.assertEqual(metrics["ambiguous"], 0)
        self.assertEqual(_effective_count(weights), 4.0)

    def test_two_plausible_disagreeing_hypotheses_are_flagged(self):
        probabilities = np.column_stack([np.full(8, 0.05), np.full(8, 0.95), np.full(8, 0.5), np.full(8, 0.5)])
        weights = np.array([0.45, 0.45, 0.05, 0.05])
        config = MultimodalityConfig(include_vanilla=False, disagreement_threshold=0.2)
        metrics = _hypothesis_metrics(probabilities, weights, config)
        self.assertEqual(metrics["plausible_hypothesis_count"], 2)
        self.assertEqual(metrics["ambiguous"], 1)
        self.assertGreater(metrics["plausible_max_disagreement"], 0.8)

    def test_weight_update_prefers_hypothesis_that_explains_stream(self):
        initial = _softmax_log_weights(np.array([0.5, 0.5]), temperature=1.0)
        probabilities = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        updated = _update_weights(initial, probabilities, np.array([1, 1, 1]))
        self.assertAlmostEqual(float(updated.sum()), 1.0)
        self.assertGreater(updated[0], 0.99)

    def test_hypothesis_bank_is_deterministic_and_finite(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(48, 4))
        y = (x[:, 0] + 0.25 * x[:, 1] > 0).astype(int)
        first_models, first_losses, first_oob = _fit_hypotheses(x, y, seed=17)
        second_models, second_losses, second_oob = _fit_hypotheses(x, y, seed=17)
        self.assertEqual(len(first_models), len(HYPOTHESIS_NAMES))
        np.testing.assert_allclose(first_losses, second_losses)
        np.testing.assert_array_equal(first_oob, second_oob)
        self.assertTrue(np.isfinite(first_losses).all())

    def test_episode_metrics_are_reproducible_and_query_labels_only_score(self):
        rng = np.random.default_rng(21)
        frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
        labels = (frame["a"].to_numpy() + frame["b"].to_numpy() > 0).astype(int)
        config = MultimodalityConfig(
            seed=12,
            context_sizes=(16,),
            stream_size=8,
            query_size=12,
            include_vanilla=False,
        )
        train_indices = np.arange(0, 96)
        test_indices = np.arange(96, 120)
        first = _sample_episode(
            frame,
            labels,
            train_indices,
            test_indices,
            16,
            config,
            np.random.default_rng(101),
            None,
            1,
            "toy",
            0,
        )
        second = _sample_episode(
            frame,
            labels,
            train_indices,
            test_indices,
            16,
            config,
            np.random.default_rng(101),
            None,
            1,
            "toy",
            0,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(map(float, json.loads(first["updated_weights"]))), 1.0)
        self.assertTrue(np.isfinite(first["mixture_query_nll"]))
        self.assertIn("oracle_best_hypothesis_query_nll", first)


if __name__ == "__main__":
    unittest.main()
