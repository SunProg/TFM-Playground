import unittest

import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer
from tfmplayground.models.slot_backbone import (
    SlotTransformerEncoderLayer,
    collect_support_attention,
    install_slot_layers,
    slot_layer_parameters,
)

BATCH, SUPPORT, QUERY, FEATURES, SLOTS = 2, 12, 5, 3, 3


def backbone() -> NanoTabPFNModel:
    torch.manual_seed(0)
    return NanoTabPFNModel(16, 2, 32, 2, 3)


def episode(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(BATCH, SUPPORT, FEATURES, generator=g),
        torch.randint(0, 2, (BATCH, SUPPORT), generator=g).float(),
        torch.randn(BATCH, QUERY, FEATURES, generator=g),
    )


class SlotBackboneTests(unittest.TestCase):
    def setUp(self):
        self.plain = backbone().eval()
        self.slotted = install_slot_layers(backbone(), num_slots=SLOTS).eval()
        self.slotted.load_state_dict(self.plain.state_dict(), strict=False)
        self.support_x, self.support_y, self.query_x = episode()

    def test_every_layer_is_replaced(self):
        self.assertTrue(all(isinstance(b, SlotTransformerEncoderLayer) for b in self.slotted.transformer_blocks))
        self.assertEqual(len(self.slotted.transformer_blocks), len(self.plain.transformer_blocks))

    def test_zero_gate_is_exactly_the_pretrained_backbone(self):
        """Slots must earn their influence: an untrained layer changes nothing."""
        expected = self.plain(self.support_x, self.support_y, self.query_x)
        actual = self.slotted(self.support_x, self.support_y, self.query_x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_opening_the_gate_changes_the_output(self):
        with torch.no_grad():
            for layer in self.slotted.transformer_blocks:
                layer.row_gate.fill_(1.0)
        expected = self.plain(self.support_x, self.support_y, self.query_x)
        actual = self.slotted(self.support_x, self.support_y, self.query_x)
        self.assertFalse(torch.allclose(actual, expected))

    def test_support_attention_is_a_per_row_distribution_over_slots(self):
        self.slotted(self.support_x, self.support_y, self.query_x)
        attention = collect_support_attention(self.slotted)
        self.assertEqual(attention.shape, (BATCH, SUPPORT, SLOTS))
        torch.testing.assert_close(attention.sum(-1), torch.ones(BATCH, SUPPORT), atol=1e-5, rtol=1e-5)

    def test_gradients_reach_the_slot_machinery(self):
        model = install_slot_layers(backbone(), num_slots=SLOTS)
        model(self.support_x, self.support_y, self.query_x).square().mean().backward()
        for layer in model.transformer_blocks:
            self.assertIsNotNone(layer.row_gate.grad)
            self.assertIsNotNone(layer.slot_attention.slots_mu.grad)
            self.assertIsNotNone(layer.write_back.in_proj_weight.grad)

    def test_slot_parameters_exclude_the_pretrained_ones(self):
        slot_only = sum(p.numel() for p in slot_layer_parameters(self.slotted))
        total = sum(p.numel() for p in self.slotted.parameters())
        pretrained = sum(p.numel() for p in self.plain.parameters())
        self.assertEqual(total - slot_only, pretrained)

    def test_installing_twice_is_a_no_op(self):
        again = install_slot_layers(self.slotted, num_slots=SLOTS)
        self.assertTrue(all(isinstance(b, SlotTransformerEncoderLayer) for b in again.transformer_blocks))
        self.assertEqual(sum(p.numel() for p in again.parameters()), sum(p.numel() for p in self.slotted.parameters()))

    def test_from_pretrained_rejects_foreign_parameters(self):
        layer = TransformerEncoderLayer(16, 2, 32)
        adapted = SlotTransformerEncoderLayer.from_pretrained(
            layer, num_slots=2, num_slot_iterations=3, competitive_slots=True
        )
        for name, parameter in layer.named_parameters():
            torch.testing.assert_close(dict(adapted.named_parameters())[name], parameter, atol=0, rtol=0)

    def test_attention_comes_from_the_deepest_layer(self):
        self.slotted(self.support_x, self.support_y, self.query_x)
        deepest = self.slotted.transformer_blocks[-1].last_support_attention
        torch.testing.assert_close(collect_support_attention(self.slotted), deepest, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
