import unittest

import torch

from tfmplayground.models.slot_attention import SlotAttention, slot_assignment_entropy

BATCH, INPUTS, SLOTS, WIDTH, HIDDEN = 2, 9, 3, 8, 16


def tiny_module(**kwargs) -> SlotAttention:
    torch.manual_seed(41)
    return SlotAttention(SLOTS, WIDTH, HIDDEN, **kwargs)


def tiny_inputs(seed: int = 7) -> torch.Tensor:
    return torch.randn(BATCH, INPUTS, WIDTH, generator=torch.Generator().manual_seed(seed))


class SlotAttentionTests(unittest.TestCase):
    def setUp(self):
        self.module = tiny_module()
        self.inputs = tiny_inputs()

    def test_shapes(self):
        slots, attention = self.module(self.inputs, generator=torch.Generator().manual_seed(1))
        self.assertEqual(slots.shape, (BATCH, SLOTS, WIDTH))
        self.assertEqual(attention.shape, (BATCH, INPUTS, SLOTS))
        self.assertTrue(torch.isfinite(slots).all())
        self.assertTrue(torch.isfinite(attention).all())

    def test_competitive_attention_normalizes_over_slots(self):
        """The defining property: each input row distributes its mass across slots.

        The pre-existing heads normalized over support rows instead, which is why
        their slots never specialized.  This is the assertion that separates the
        two, and the one that would have caught the original misnomer.
        """
        _, attention = self.module(self.inputs, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(BATCH, INPUTS), atol=1e-6, rtol=1e-6)

    def test_non_competitive_attention_normalizes_over_inputs(self):
        module = tiny_module(competitive=False)
        _, attention = module(self.inputs, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(attention.sum(dim=-2), torch.ones(BATCH, SLOTS), atol=1e-6, rtol=1e-6)

    def test_one_iteration_non_competitive_matches_a_hand_written_weighted_mean(self):
        """The ablation really is the old mechanism, not a differently broken one."""
        module = tiny_module(num_iterations=1, competitive=False)
        slots_init = torch.randn(BATCH, SLOTS, WIDTH, generator=torch.Generator().manual_seed(3))
        slots, _ = module(self.inputs, slots=slots_init.clone())

        normalized = module.norm_inputs(self.inputs)
        k = module.project_k(normalized)
        v = module.project_v(normalized)
        q = module.project_q(module.norm_slots(slots_init)) * WIDTH**-0.5
        attention = torch.matmul(k, q.transpose(-1, -2)).softmax(dim=-2)
        weights = attention + module.epsilon
        weights = weights / weights.sum(dim=-2, keepdim=True)
        updates = torch.matmul(weights.transpose(-1, -2), v)
        expected = module.gru(updates.reshape(BATCH * SLOTS, WIDTH), slots_init.reshape(BATCH * SLOTS, WIDTH)).reshape(
            BATCH, SLOTS, WIDTH
        )
        expected = expected + module.mlp(module.norm_mlp(expected))

        torch.testing.assert_close(slots, expected, atol=0, rtol=0)

    def test_permuting_inputs_leaves_slots_unchanged(self):
        order = torch.tensor([4, 0, 8, 2, 6, 1, 7, 3, 5])
        slots, attention = self.module(self.inputs, slots=self._fixed_slots())
        permuted_slots, permuted_attention = self.module(self.inputs[:, order], slots=self._fixed_slots())
        torch.testing.assert_close(slots, permuted_slots, atol=2e-6, rtol=2e-6)
        # The attention follows the rows, so it permutes rather than staying put.
        torch.testing.assert_close(attention[:, order], permuted_attention, atol=2e-6, rtol=2e-6)

    def test_permuting_the_slot_init_permutes_the_slots(self):
        order = torch.tensor([2, 0, 1])
        slots, attention = self.module(self.inputs, slots=self._fixed_slots())
        permuted_slots, permuted_attention = self.module(self.inputs, slots=self._fixed_slots()[:, order])
        torch.testing.assert_close(slots[:, order], permuted_slots, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(attention[:, :, order], permuted_attention, atol=2e-6, rtol=2e-6)

    def test_seeded_generator_is_reproducible_and_seed_sensitive(self):
        first, _ = self.module(self.inputs, generator=torch.Generator().manual_seed(11))
        again, _ = self.module(self.inputs, generator=torch.Generator().manual_seed(11))
        other, _ = self.module(self.inputs, generator=torch.Generator().manual_seed(12))
        torch.testing.assert_close(first, again, atol=0, rtol=0)
        self.assertFalse(torch.allclose(first, other))

    def test_sampled_slots_start_distinct(self):
        """Identical slots are a fixed point of the competition; the noise breaks it."""
        slots = self.module.initial_slots(BATCH, generator=torch.Generator().manual_seed(5))
        self.assertEqual(slots.shape, (BATCH, SLOTS, WIDTH))
        for first in range(SLOTS):
            for second in range(first + 1, SLOTS):
                self.assertFalse(torch.allclose(slots[:, first], slots[:, second]))

    def test_gradients_reach_every_component(self):
        slots, _ = self.module(self.inputs, generator=torch.Generator().manual_seed(2))
        slots.square().mean().backward()
        for name in ("slots_mu", "slots_log_sigma"):
            parameter = getattr(self.module, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (self.module.project_q, self.module.project_k, self.module.project_v, self.module.gru):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_configurations_raise(self):
        for num_slots, slot_size, hidden in ((0, WIDTH, HIDDEN), (SLOTS, 0, HIDDEN), (SLOTS, WIDTH, 0)):
            with self.assertRaises(ValueError):
                SlotAttention(num_slots, slot_size, hidden)
        with self.assertRaises(ValueError):
            SlotAttention(SLOTS, WIDTH, HIDDEN, num_iterations=0)
        with self.assertRaises(ValueError):
            SlotAttention(SLOTS, WIDTH, HIDDEN, epsilon=0.0)
        with self.assertRaises(ValueError):
            self.module(torch.randn(BATCH, INPUTS))
        with self.assertRaises(ValueError):
            self.module(torch.randn(BATCH, INPUTS, WIDTH + 1))
        with self.assertRaises(ValueError):
            self.module(self.inputs, slots=torch.randn(BATCH, SLOTS + 1, WIDTH))

    def _fixed_slots(self) -> torch.Tensor:
        return torch.randn(BATCH, SLOTS, WIDTH, generator=torch.Generator().manual_seed(19))


class SlotAssignmentEntropyTests(unittest.TestCase):
    def test_bounds(self):
        uniform = torch.full((1, 4, SLOTS), 1.0 / SLOTS)
        torch.testing.assert_close(slot_assignment_entropy(uniform), torch.ones(1, 4), atol=1e-6, rtol=1e-6)
        onehot = torch.zeros(1, 4, SLOTS)
        onehot[..., 0] = 1.0
        torch.testing.assert_close(slot_assignment_entropy(onehot), torch.zeros(1, 4), atol=1e-6, rtol=1e-6)

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            slot_assignment_entropy(torch.zeros(4, SLOTS))


if __name__ == "__main__":
    unittest.main()
