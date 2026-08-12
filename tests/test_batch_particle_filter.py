import unittest

import numpy as np
import torch

from tfmplayground.experiments.train_particle_regime_comparison import _episode
from tfmplayground.models.batch_particle_filter import BatchCausalParticleFilter
from tfmplayground.models.integrated_latent_filter import NanoTabPFNIntegratedLatentFilter
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.particle_online import BatchParticleOnlineClassifier


def tiny_model(context_limit=12):
    torch.manual_seed(4)
    backbone = NanoTabPFNModel(embedding_size=8, num_attention_heads=2, mlp_hidden_size=16, num_layers=2, num_outputs=3)
    particle = NanoTabPFNIntegratedLatentFilter(backbone, num_hypotheses=2)
    return BatchCausalParticleFilter(particle, backbone, context_limit=context_limit)


class BatchParticleFilterTests(unittest.TestCase):
    def test_current_labels_cannot_change_committed_probabilities(self):
        model = tiny_model().eval()
        x = torch.randn(1, 5, 3)
        state = model.initial_state(3)
        pending = model.predict_batch(state, x)
        committed = pending.probabilities.detach().clone()
        left = model.reveal_batch(pending, torch.zeros(1, 5, dtype=torch.long))
        right = model.reveal_batch(pending, torch.ones(1, 5, dtype=torch.long))
        torch.testing.assert_close(pending.probabilities, committed, rtol=0, atol=0)
        self.assertFalse(torch.equal(left.context_y, right.context_y))

    def test_transition_occurs_only_in_predict(self):
        model = tiny_model()
        state = model.initial_state(2)
        state = type(state)(
            log_weights=torch.tensor([[-0.01, -4.61]]),
            previous_surprise=state.previous_surprise,
            context_x=state.context_x,
            context_y=state.context_y,
        )
        pending = model.predict_batch(state, torch.zeros(1, 3, 2))
        expected = model._transition(state.log_weights)
        torch.testing.assert_close(pending.unmasked_log_weights, expected)
        revealed = model.reveal_batch(pending, torch.zeros(1, 3, dtype=torch.long))
        # With equal first-batch particle likelihoods, reveal normalisation changes no weights.
        torch.testing.assert_close(revealed.log_weights, expected - torch.logsumexp(expected, -1, keepdim=True))

    def test_context_cap_is_recent_or_deterministically_stratified(self):
        model = tiny_model(context_limit=4)
        x = torch.arange(12, dtype=torch.float32)[None, :, None]
        y = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
        recent_x, _ = model._capped_context(x, y, temporal=True)
        torch.testing.assert_close(recent_x.flatten(), torch.tensor([8.0, 9.0, 10.0, 11.0]))
        first = model._capped_context(x, y, temporal=False)
        second = model._capped_context(x, y, temporal=False)
        torch.testing.assert_close(first[0], second[0])
        self.assertEqual(first[1].flatten().bincount(minlength=2).tolist(), [2, 2])

    def test_online_adapter_requires_predict_then_reveal(self):
        adapter = BatchParticleOnlineClassifier(tiny_model())
        x = np.zeros((4, 2), dtype=np.float32)
        adapter.update(x, np.array([0, 1, 0, 1]))  # known initial/few-shot context is allowed
        prediction = adapter.predict_proba(x)
        self.assertEqual(prediction.shape, (4, 2))
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(x)
        adapter.update(x, np.array([0, 1, 0, 1]))
        self.assertEqual(adapter.predict_proba(x).shape, (4, 2))

    def test_generator_is_deterministic_and_has_fixed_episode_batches(self):
        left = _episode(91, 16, "cpu")
        right = _episode(91, 16, "cpu")
        torch.testing.assert_close(left.x, right.x)
        self.assertEqual(left.batch_slices, right.batch_slices)
        self.assertLessEqual(left.x.shape[-1], 16)


if __name__ == "__main__":
    unittest.main()
