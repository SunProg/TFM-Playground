import unittest

import numpy as np

from tfmplayground.experiments.regime_ladder_diagnostic import (
    RUNGS,
    evaluate_rung,
    fixed_rule_episode,
    run_ladder,
    sample_rung_episode,
    train_rung,
)


class FixedRuleEpisodeTests(unittest.TestCase):
    def test_shapes_and_determinism_of_the_rule(self):
        rng = np.random.default_rng(0)
        episode = fixed_rule_episode(rng, batch_size=3, support_size=16, query_count=5, features=4)
        self.assertEqual(tuple(episode.support_x.shape), (3, 16, 4))
        self.assertEqual(tuple(episode.query_y.shape), (3, 5))
        # The rule itself, not just its shape: z is exactly 1[x0 > 0].
        self.assertTrue(bool(((episode.query_x[..., 0] > 0).long() == episode.query_y).all()))


class SampleRungEpisodeTests(unittest.TestCase):
    def test_every_rung_produces_a_usable_regime_target(self):
        rng = np.random.default_rng(1)
        for rung in RUNGS:
            episode = sample_rung_episode(
                rung,
                rng,
                batch_size=2,
                support_size=32,
                query_count=8,
                features=4,
                contamination=0.3,
                regime_coherence=4.0,
            )
            self.assertEqual(tuple(episode.support_x.shape), (2, 32, 4))
            self.assertEqual(tuple(episode.query_y.shape), (2, 8))
            self.assertTrue(set(episode.query_y.unique().tolist()) <= {0, 1})


class LadderSmokeTests(unittest.TestCase):
    """Tiny budgets: this checks the harness is wired correctly, not that the
    trained numbers mean anything -- that needs a real step budget."""

    def test_fixed_rung_is_close_to_ceiling_for_both_scorers(self):
        model = train_rung(
            "fixed",
            steps=200,
            batch_size=8,
            support_size=64,
            query_count=16,
            features=4,
            contamination=0.3,
            regime_coherence=4.0,
            seed=0,
        )
        metrics = evaluate_rung(
            model,
            "fixed",
            episodes=8,
            batch_size=4,
            support_size=64,
            query_count=16,
            features=4,
            contamination=0.3,
            regime_coherence=4.0,
            seed=1,
        )
        self.assertGreater(metrics["rf_auc"], 0.95)
        self.assertGreater(metrics["model_auc"], 0.8)

    def test_run_ladder_covers_every_rung_in_order(self):
        rows = run_ladder(
            steps=2,
            train_batch_size=2,
            eval_episodes=2,
            eval_batch_size=2,
            support_size=16,
            query_count=4,
            features=4,
            contamination=0.3,
            regime_coherence=4.0,
            seed=0,
        )
        self.assertEqual([row["rung"] for row in rows], list(RUNGS))
        for row in rows:
            self.assertIn("model_auc", row)
            self.assertIn("rf_auc", row)
            self.assertIn("gap", row)


if __name__ == "__main__":
    unittest.main()
