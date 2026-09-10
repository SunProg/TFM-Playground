import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v3 import (
    FAMILIES,
    OriginalPrior,
    V3Config,
    group_posterior,
    oracle_probabilities,
    sample_episode,
)
from tfmplayground.experiments.pretrain_multiregime_v3 import (
    KINDS,
    PilotConfig,
    build_model,
    log_predictions,
    mixture_probability,
    preflight,
    run,
    state_hash,
)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = V3Config(support_size=24, query_size=12, max_features=4, num_groups=4)

    def test_reproducible_mechanisms_and_support_independent_of_query_count(self):
        for family in FAMILIES[1:]:
            with self.subTest(family=family):
                first = sample_episode(self.config, family=family, seed=31)
                same = sample_episode(self.config, family=family, seed=31)
                more = sample_episode(replace(self.config, query_size=19), family=family, seed=31)
                self.assertEqual(first.tensor_hash(), same.tensor_hash())
                self.assertEqual(first.metadata["mechanism_hash"], more.metadata["mechanism_hash"])
                torch.testing.assert_close(first.support_x, more.support_x, atol=0, rtol=0)
                torch.testing.assert_close(first.support_y, more.support_y, atol=0, rtol=0)
                np.testing.assert_array_equal(first.support_components, more.support_components)
                self.assertEqual(first.support_x.shape, (1, 24, 8))

    def test_shared_rule_and_zero_separation_have_no_predictive_regime_difference(self):
        for family in FAMILIES[1:]:
            config = replace(self.config, separation=0, num_regimes=4)
            episode = sample_episode(config, family=family, seed=19)
            p = episode.query_components
            np.testing.assert_array_equal(p, np.repeat(p[:, :1], 4, axis=1))

    def test_persistent_groups_share_assignments_but_codes_are_not_regime_labels(self):
        episode = sample_episode(self.config, family="persistent", seed=18)
        groups = np.concatenate((episode.support_groups, episode.query_groups))
        z = np.concatenate((episode.support_z, episode.query_z))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(z[groups == group])), 1)
        x = torch.cat((episode.support_x, episode.query_x), dim=1)[0]
        codes = x[:, self.config.max_features :].numpy()
        np.testing.assert_array_equal(codes.sum(-1), np.ones(len(codes)))
        # G>K ensures at least two distinct observed identities share a regime.
        self.assertGreater(len(np.unique(groups)), len(np.unique(z)))

    def test_oracle_uses_neither_query_labels_nor_realized_query_regimes(self):
        for family in FAMILIES[1:]:
            episode = sample_episode(self.config, family=family, seed=15)
            changed = replace(episode, query_y=1 - episode.query_y, query_z=1 - episode.query_z)
            before, after = oracle_probabilities(episode), oracle_probabilities(changed)
            np.testing.assert_array_equal(before["marginal"], after["marginal"])
            np.testing.assert_allclose(before["query_responsibilities"].sum(-1), 1)
            self.assertTrue(np.isfinite(before["marginal"]).all())

    def test_group_evidence_accumulates_and_unseen_group_keeps_prior(self):
        prior = np.array([0.5, 0.5])
        probabilities = np.tile([0.9, 0.1], (6, 1))
        groups = np.zeros(6, dtype=int)
        posterior = group_posterior(prior, probabilities, np.ones(6), groups, 2)
        self.assertGreater(posterior[0, 0], 0.999)
        np.testing.assert_array_equal(posterior[1], prior)
        reverse = group_posterior(prior, probabilities, np.zeros(6), groups, 2)
        np.testing.assert_allclose(posterior[0], reverse[0, ::-1])

    def test_original_replay_preserves_global_model_rng_and_has_no_invented_oracle(self):
        original = OriginalPrior(self.config)
        torch.manual_seed(23)
        state = torch.get_rng_state().clone()
        first = original.sample(17)
        torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(first.tensor_hash(), original.sample(17).tensor_hash())
        self.assertEqual(oracle_probabilities(first), {})
        self.assertTrue(set(first.query_y.flatten().tolist()).issubset({0, 1}))


class TrainingTests(unittest.TestCase):
    def config(self):
        return PilotConfig(
            steps=2,
            micro_batch_size=1,
            accumulate_gradients=1,
            support_size=16,
            query_size=4,
            max_features=3,
            num_groups=3,
            width=16,
            hidden=32,
            layers=1,
            heads=2,
            validation_interval=2,
            validation_episodes=1,
            warmup_steps=1,
        )

    def test_fixed_mixture_and_curriculum_are_explicit(self):
        self.assertEqual(mixture_probability("original", 100, 100), 0)
        self.assertEqual(mixture_probability("fixed", 1, 100), 0.5)
        self.assertEqual(mixture_probability("curriculum", 10, 100), 0)
        self.assertAlmostEqual(mixture_probability("curriculum", 25, 100), 0.25)
        self.assertEqual(mixture_probability("curriculum", 100, 100), 0.5)

    def test_models_have_identical_initial_backbones_and_do_not_read_tags(self):
        config = self.config()
        episode = sample_episode(config.generator(), family="persistent", seed=77)
        hashes = []
        for kind in KINDS:
            model, fingerprint = build_model(config, kind)
            hashes.append(fingerprint)
            model.eval()
            with torch.no_grad():
                first, _ = log_predictions(model, [episode], "cpu")
                altered = replace(
                    episode,
                    support_z=1 - episode.support_z,
                    query_z=1 - episode.query_z,
                    query_y=1 - episode.query_y,
                    support_components=None,
                    query_components=None,
                )
                second, _ = log_predictions(model, [altered], "cpu")
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(len(set(hashes)), 1)

    def test_end_to_end_matched_stream_checkpoints_and_source_gate(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight(config, root / "gate")
            gate_path = root / "gate" / "execution_gate.json"
            # index 0 and 3 share mode="original" (index // len(MODES) selects
            # kind, index % len(MODES) selects mode) but differ in kind
            # (plain vs. the first table_slot condition) -- training_episode's
            # seed formula and mixture_probability(mode, ...) never read kind,
            # so the two cells' training streams must still match.
            results = [run(config, index=i, output=root / str(i), gate_path=gate_path) for i in (0, 3)]
            self.assertEqual(results[0]["training_stream_hash"], results[1]["training_stream_hash"])
            self.assertEqual(results[0]["evaluation_bank_hashes"], results[1]["evaluation_bank_hashes"])
            for i in (0, 3):
                checkpoint = torch.load(root / str(i) / "checkpoint.pth", weights_only=False)
                self.assertEqual(checkpoint["step"], 2)
                backbone = {
                    k.removeprefix("backbone."): v
                    for k, v in checkpoint["model"].items()
                    if i == 0 or k.startswith("backbone.")
                }
                self.assertNotEqual(state_hash(backbone), checkpoint["metadata"]["initial_backbone_hash"])
            gate = json.loads(gate_path.read_text())
            gate["source_hash"] = "different-source"
            gate_path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "matching source"):
                run(config, index=0, output=root / "reject", gate_path=gate_path)


if __name__ == "__main__":
    unittest.main()
