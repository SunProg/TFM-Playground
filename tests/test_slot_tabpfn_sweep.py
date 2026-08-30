import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import sample_scm_multiregime_episode
from tfmplayground.experiments.dump_multiregime_episodes import (
    DumpConfig,
    MultiregimeDumpLoader,
    _shard_episodes,
    dump_multiregime_episodes,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    PlainPretrainingConfig,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    multiregime_probability as plain_multiregime_probability,
)
from tfmplayground.experiments.pretrain_slot_tabpfn import (
    PRIOR_MODES,
    SlotPretrainingConfig,
    multiregime_probability,
    summarize_samples,
    support_binding_scores,
    validate_config,
)
from tfmplayground.experiments.slot_tabpfn_sweep import (
    MULTIREGIME_SHARE,
    SCREENING_STEPS,
    SLOT_COUNTS,
    configuration_flags,
    configuration_label,
    screening_configurations,
    summarize_sweep,
)

MAX_STEPS = 1000


def config(prior_mode: str, **kwargs) -> SlotPretrainingConfig:
    return SlotPretrainingConfig(prior_mode=prior_mode, max_steps=MAX_STEPS, device="cpu", **kwargs)


class CurriculumTests(unittest.TestCase):
    """The four arms, and the guarantee that three of them match the vanilla script."""

    def test_constant_modes(self):
        for step in (0, 1, MAX_STEPS // 2, MAX_STEPS):
            self.assertEqual(multiregime_probability(config("plain"), step), 0.0)
            self.assertEqual(multiregime_probability(config("multiregime"), step), 1.0)
            self.assertEqual(multiregime_probability(config("mixed"), step), 0.30)

    def test_mixed_share_is_configurable(self):
        self.assertAlmostEqual(multiregime_probability(config("mixed", multiregime_share=0.5), 7), 0.5)

    def test_curriculum_shape(self):
        curriculum = config("curriculum")
        self.assertEqual(multiregime_probability(curriculum, 0), 0.0)
        # Flat at zero through the first 10% of the budget.
        self.assertEqual(multiregime_probability(curriculum, int(0.10 * MAX_STEPS)), 0.0)
        # Linear ramp to 0.5 at 50%, then flat.
        self.assertAlmostEqual(multiregime_probability(curriculum, int(0.30 * MAX_STEPS)), 0.25, places=6)
        self.assertAlmostEqual(multiregime_probability(curriculum, int(0.50 * MAX_STEPS)), 0.5, places=6)
        self.assertAlmostEqual(multiregime_probability(curriculum, MAX_STEPS), 0.5, places=6)

    def test_shared_modes_match_the_plain_implementation_exactly(self):
        """This is what stops the two scripts drifting apart.

        `plain`, `multiregime` and `curriculum` must stay byte-identical to the
        vanilla pretraining script, because the sweep is compared against runs
        produced by it.  `mixed` is the only arm this module defines itself.
        """
        for prior_mode in ("plain", "multiregime", "curriculum"):
            plain = replace(PlainPretrainingConfig(), prior_mode=prior_mode, max_steps=MAX_STEPS)
            slot = config(prior_mode)
            for step in range(0, MAX_STEPS + 1, 37):
                with self.subTest(prior_mode=prior_mode, step=step):
                    self.assertAlmostEqual(
                        multiregime_probability(slot, step),
                        plain_multiregime_probability(plain, step),
                        places=12,
                    )

    def test_config_validation(self):
        validate_config(config("mixed"))
        with self.assertRaises(ValueError):
            validate_config(config("nonsense"))
        with self.assertRaises(ValueError):
            validate_config(config("mixed", multiregime_share=1.5))
        with self.assertRaises(ValueError):
            validate_config(replace(config("plain"), max_steps=0))


class SweepTests(unittest.TestCase):
    def test_grid_is_slot_count_outermost_with_unique_labels(self):
        configurations = screening_configurations()
        self.assertEqual(len(configurations), len(PRIOR_MODES) * len(SLOT_COUNTS))
        # Slot count outermost, so the original K=2 arms keep indices 0..3 and an
        # in-flight array is not re-mapped by extending the grid.
        self.assertEqual([c["prior_mode"] for c in configurations[:4]], list(PRIOR_MODES))
        self.assertTrue(all(c["num_slots"] == 2 for c in configurations[:4]))
        self.assertEqual([configuration_label(c) for c in configurations[:4]], list(PRIOR_MODES))
        labels = [configuration_label(c) for c in configurations]
        self.assertEqual(len(set(labels)), len(labels))

    def test_flags_carry_the_arm_and_hold_everything_else_fixed(self):
        configurations = screening_configurations()
        flag_sets = [configuration_flags(index) for index in range(len(configurations))]
        for configuration, flags in zip(configurations, flag_sets, strict=True):
            self.assertIn(f"--prior-mode {configuration['prior_mode']}", flags)
            self.assertIn(f"--num-slots {configuration['num_slots']}", flags)
            self.assertIn(f"--max-steps {SCREENING_STEPS}", flags)
            self.assertIn(f"--multiregime-share {MULTIREGIME_SHARE:g}", flags)
        # Prior mode and slot count are the only two axes: strip both and every
        # cell's remaining flags must be byte identical, so the grid really
        # isolates them and nothing else drifts between cells.
        stripped = {
            flags.replace(f"--prior-mode {c['prior_mode']} ", "").replace(f"--num-slots {c['num_slots']} ", "")
            for c, flags in zip(configurations, flag_sets, strict=True)
        }
        self.assertEqual(len(stripped), 1)

    def test_seed_override_and_index_bounds(self):
        self.assertIn("--seed 99", configuration_flags(0, seed=99))
        for index in (-1, len(screening_configurations())):
            with self.assertRaises(IndexError):
                configuration_flags(index)

    def test_summarize_ranks_by_multiregime_cross_entropy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, prior_mode, value in (
                ("run-a", "plain", 0.90),
                ("run-b", "curriculum", 0.40),
                ("run-c", "mixed", 0.60),
            ):
                run = root / name
                run.mkdir()
                (run / "config.json").write_text(json.dumps({"prior_mode": prior_mode, "num_slots": 2, "seed": 2402}))
                (run / "selection.json").write_text(
                    json.dumps(
                        {
                            "multiregime_cross_entropy": value,
                            "query_cross_entropy": 0.5,
                            "gate_regime_auc": 0.7,
                            "gate_entropy": 0.3,
                        }
                    )
                )
            (root / "unfinished").mkdir()
            ranked = summarize_sweep(root)
        self.assertEqual([run["prior_mode"] for run in ranked], ["curriculum", "mixed", "plain"])
        self.assertEqual(ranked[0]["gate_regime_auc"], 0.7)


class SupportBindingTests(unittest.TestCase):
    """The mechanism metric: did slot competition partition the context by regime?"""

    def test_perfect_binding_scores_one(self):
        attention = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        source = torch.tensor([[0, 0, 1, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_auc"], 1.0, places=6)
        self.assertAlmostEqual(scores["support_binding_purity"], 1.0, places=6)
        self.assertAlmostEqual(scores["support_attention_entropy"], 0.0, places=6)
        self.assertAlmostEqual(scores["support_regime_base_rate"], 0.5, places=6)

    def test_uniform_attention_is_chance_with_maximal_entropy(self):
        attention = torch.full((1, 6, 2), 0.5)
        source = torch.tensor([[0, 0, 0, 1, 1, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_auc"], 0.5, places=6)
        self.assertAlmostEqual(scores["support_attention_entropy"], 1.0, places=6)

    def test_scores_are_invariant_to_slot_order(self):
        """Slots are anonymous, so relabelling them must not move the metric."""
        attention = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]])
        source = torch.tensor([[0, 0, 1, 1]])
        original = support_binding_scores(attention, source)
        flipped = support_binding_scores(attention.flip(-1), source)
        self.assertEqual(original, flipped)

    def test_purity_is_read_against_the_base_rate(self):
        """One slot claiming everything already scores the majority fraction."""
        attention = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        source = torch.tensor([[0, 0, 0, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_purity"], 0.75, places=6)
        self.assertAlmostEqual(scores["support_regime_base_rate"], 0.75, places=6)

    def test_missing_or_single_valued_tags_yield_no_scores(self):
        attention = torch.full((1, 4, 2), 0.5)
        self.assertEqual(support_binding_scores(attention, None), {})
        self.assertEqual(support_binding_scores(attention, torch.zeros(1, 4, dtype=torch.long)), {})

    def test_sampler_actually_populates_the_support_tag(self):
        """The tag used to be computed and discarded; it must now survive."""
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(3),
            family="mlp_scm",
            batch_size=1,
            support_size=32,
            query_count=8,
            contamination=0.4,
            device="cpu",
        )
        self.assertIsNotNone(episode.support_regime_source)
        self.assertEqual(episode.support_regime_source.shape, episode.support_y.shape)
        self.assertEqual(set(episode.support_regime_source.reshape(-1).tolist()), {0, 1})
        # It must survive a device move alongside the query-side tag.
        moved = episode.to("cpu")
        self.assertIsNotNone(moved.support_regime_source)


class MultiregimeDumpTests(unittest.TestCase):
    """The dump must reproduce what on-the-fly generation would have given."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        dump_multiregime_episodes(
            DumpConfig(
                output=str(cls.root / "shard-000.h5"),
                episodes=12,
                batch_size=4,
                support_size=16,
                query_count=4,
                contamination=0.25,
                seed=7,
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _loader(self, **kwargs):
        return MultiregimeDumpLoader(self.root, batch_size=4, **kwargs)

    def test_episode_shapes_and_split(self):
        episode = self._loader().sample()
        self.assertEqual(episode.support_x.shape[:2], (4, 16))
        self.assertEqual(episode.query_x.shape[:2], (4, 4))
        self.assertEqual(episode.support_y.shape, (4, 16))
        self.assertEqual(episode.query_y.shape, (4, 4))
        self.assertEqual(episode.support_x.shape[2], episode.query_x.shape[2])

    def test_regime_tags_survive_the_round_trip(self):
        """Without these the slot-binding diagnostic cannot be computed at all."""
        episode = self._loader().sample()
        self.assertEqual(episode.support_regime_source.shape, episode.support_y.shape)
        self.assertEqual(episode.query_regime_source.shape, episode.query_y.shape)
        tags = torch.cat((episode.support_regime_source, episode.query_regime_source), dim=1)
        self.assertEqual(sorted(set(tags.reshape(-1).tolist())), [0, 1])
        # Contamination is the fraction of rows relabelled under the second regime.
        self.assertAlmostEqual(float(tags.float().mean()), 0.25, places=6)

    def test_candidates_and_posterior_are_preserved(self):
        episode = self._loader().sample()
        self.assertEqual(episode.candidate_support_positive.shape, (4, 2, 16))
        self.assertEqual(episode.candidate_query_positive.shape, (4, 2, 4))
        torch.testing.assert_close(episode.posterior.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_dump_is_usable_as_a_training_batch(self):
        episode = self._loader().sample()
        self.assertEqual(episode.condition, "multiregime")
        self.assertTrue(torch.isfinite(episode.support_x).all())
        self.assertEqual(set(episode.query_y.reshape(-1).tolist()) - {0, 1}, set())

    def test_exhausting_the_shard_cycles_rather_than_stopping(self):
        loader = self._loader()
        for _ in range(6):  # 12 episodes at batch 4 -> wraps partway through
            self.assertEqual(loader.sample().support_x.shape[0], 4)

    def test_shards_split_the_work_without_overlap(self):
        counts = [_shard_episodes(DumpConfig(output="", episodes=10, shard_index=i, num_shards=3)) for i in range(3)]
        self.assertEqual(sum(counts), 10)
        self.assertEqual(counts, [4, 3, 3])

    def test_invalid_shard_configuration_raises(self):
        with self.assertRaises(ValueError):
            dump_multiregime_episodes(DumpConfig(output="", episodes=1, shard_index=3, num_shards=3))


class SummarySampleTests(unittest.TestCase):
    """Every logged metric carries dispersion, not just a mean."""

    def test_reports_mean_spread_and_interval(self):
        summary = summarize_samples("metric", [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(summary["metric"], 2.5)
        self.assertEqual(summary["metric_n"], 4)
        self.assertAlmostEqual(summary["metric_std"], 1.2909944, places=6)
        self.assertAlmostEqual(summary["metric_stderr"], 0.6454972, places=6)
        # 95% normal approximation, the convention this repo's summaries use.
        self.assertAlmostEqual(summary["metric_ci_low"], 2.5 - 1.96 * 0.6454972, places=6)
        self.assertAlmostEqual(summary["metric_ci_high"], 2.5 + 1.96 * 0.6454972, places=6)

    def test_single_sample_has_no_interval(self):
        summary = summarize_samples("metric", [0.7])
        self.assertEqual(summary, {"metric": 0.7, "metric_n": 1})

    def test_zero_variance_collapses_the_interval(self):
        """Purity pinned to the base rate is a real signal, not missing data."""
        summary = summarize_samples("metric", [0.6875] * 5)
        self.assertAlmostEqual(summary["metric_ci_low"], 0.6875)
        self.assertAlmostEqual(summary["metric_ci_high"], 0.6875)

    def test_empty_samples_yield_nothing(self):
        self.assertEqual(summarize_samples("metric", []), {})


if __name__ == "__main__":
    unittest.main()
