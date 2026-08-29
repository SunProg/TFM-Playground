import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import sample_scm_multiregime_episode
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
    support_binding_scores,
    validate_config,
)
from tfmplayground.experiments.slot_tabpfn_sweep import (
    MULTIREGIME_SHARE,
    SCREENING_STEPS,
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
    def test_one_configuration_per_prior_mode_in_order(self):
        configurations = screening_configurations()
        self.assertEqual([c["prior_mode"] for c in configurations], list(PRIOR_MODES))
        labels = [configuration_label(c) for c in configurations]
        self.assertEqual(len(set(labels)), len(labels))

    def test_flags_carry_the_arm_and_hold_everything_else_fixed(self):
        flag_sets = [configuration_flags(index) for index in range(len(PRIOR_MODES))]
        for prior_mode, flags in zip(PRIOR_MODES, flag_sets, strict=True):
            self.assertIn(f"--prior-mode {prior_mode}", flags)
            self.assertIn(f"--max-steps {SCREENING_STEPS}", flags)
            self.assertIn(f"--multiregime-share {MULTIREGIME_SHARE:g}", flags)
        # Only the prior mode differs between arms: strip it and every arm's
        # remaining flags must be byte identical, so the sweep really isolates
        # the prior composition.
        without_mode = {
            flags.replace(f"--prior-mode {prior_mode} ", "")
            for prior_mode, flags in zip(PRIOR_MODES, flag_sets, strict=True)
        }
        self.assertEqual(len(without_mode), 1)

    def test_seed_override_and_index_bounds(self):
        self.assertIn("--seed 99", configuration_flags(0, seed=99))
        for index in (-1, len(PRIOR_MODES)):
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
                (run / "config.json").write_text(json.dumps({"prior_mode": prior_mode, "seed": 2402}))
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


if __name__ == "__main__":
    unittest.main()
