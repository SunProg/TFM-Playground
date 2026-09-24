import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v2 import stack_regime_episodes, tensor_hash
from tfmplayground.experiments.scm_prior_learning import SCMBatchSampler
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode


class SCMRegimePriorTests(unittest.TestCase):
    def test_classical_and_neural_evaluation_use_identical_episodes(self):
        sampler = SCMBatchSampler(SCMRegimeConfig())
        episode = sampler.episode(3_000_000)
        x, y, z = sampler(3_000_000)
        np.testing.assert_array_equal(x, torch.cat([episode.support_x, episode.query_x], 1).numpy())
        np.testing.assert_array_equal(y, torch.cat([episode.support_y.long(), episode.query_y], 1).numpy())
        np.testing.assert_array_equal(z, torch.cat([episode.support_z, episode.query_z], 1).numpy())

    def test_reproducibility_and_global_rng_isolation(self):
        config = SCMRegimeConfig()
        first = sample_scm_regime_episode(config, seed=123)
        state = torch.random.get_rng_state().clone()
        second = sample_scm_regime_episode(config, seed=123)
        self.assertEqual(tensor_hash(first), tensor_hash(second))
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertNotEqual(
            first.metadata["mechanism_hash"], sample_scm_regime_episode(config, seed=124).metadata["mechanism_hash"]
        )

    def test_query_count_does_not_select_mechanisms_or_change_support(self):
        config = SCMRegimeConfig(query_size=16)
        small = sample_scm_regime_episode(config, seed=8)
        large = sample_scm_regime_episode(replace(config, query_size=256), seed=8)
        self.assertEqual(small.metadata["mechanism_hash"], large.metadata["mechanism_hash"])
        self.assertEqual(small.metadata["thresholds"], large.metadata["thresholds"])
        torch.testing.assert_close(small.support_x, large.support_x, rtol=0, atol=0)
        torch.testing.assert_close(small.support_y, large.support_y, rtol=0, atol=0)

    def test_separation_changes_cue_only_and_zero_is_uninformative(self):
        config = SCMRegimeConfig()
        easy = sample_scm_regime_episode(config, seed=9)
        hard = sample_scm_regime_episode(replace(config, cue_separation=0), seed=9)
        for name in ("support_y", "query_y", "support_z", "query_z", "counterfactual_query_probabilities"):
            torch.testing.assert_close(getattr(easy, name), getattr(hard, name), rtol=0, atol=0)
        torch.testing.assert_close(easy.support_x[:, :, 1:], hard.support_x[:, :, 1:], rtol=0, atol=0)
        self.assertTrue(torch.all(hard.query_gate_probabilities == 0.5))

    def test_no_label_leakage_and_batch_compatibility(self):
        episode = sample_scm_regime_episode(SCMRegimeConfig(max_regimes=4), seed=7)
        changed = copy.deepcopy(episode)
        changed.query_y.fill_(0)
        changed.support_z.fill_(0)
        changed.query_z.fill_(0)
        changed.counterfactual_query_probabilities.fill_(0)
        for a, b in zip(episode.latent_inputs(), changed.latent_inputs(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(stack_regime_episodes([episode, episode]).support_x.shape, (2, 128, 3))
        self.assertEqual(episode.oracle_inputs()[0].shape, (1, 128, 7))
        self.assertTrue(torch.all(episode.query_gate_probabilities[:, :, 2:] == 0))
        torch.testing.assert_close(episode.query_gate_probabilities.sum(-1), torch.ones(1, 256))

    def test_nontrivial_rules_and_visible_routing(self):
        routing, disagreements = [], []
        for seed in range(10):
            e = sample_scm_regime_episode(SCMRegimeConfig(), seed=seed)
            routing.append(((e.query_x[:, :, 0] > 0) == e.query_z).float().mean().item())
            disagreements.append(e.metadata["calibration_disagreement"])
            expected_y = e.counterfactual_query_probabilities.gather(1, e.query_z[:, None, :]).squeeze(1)
            torch.testing.assert_close(expected_y.long(), e.query_y)
        self.assertGreater(np.mean(routing), 0.99)
        self.assertTrue(all(0.25 <= d <= 0.75 for d in disagreements))

    def test_noise_and_validation(self):
        e = sample_scm_regime_episode(SCMRegimeConfig(label_noise=0.5), seed=1)
        self.assertTrue(torch.all(e.counterfactual_query_probabilities == 0.5))
        for kwargs in (
            {"cue_separation": float("nan")},
            {"num_layers": 1},
            {"min_disagreement": 0.8},
            {"regime_probability": 1.0},
            {"label_noise": -1},
        ):
            with self.assertRaises(ValueError):
                SCMRegimeConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
