import tempfile
import unittest
from pathlib import Path

import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import (
    NanoTabPFNSlotRegimeModel,
    SlotLogitsAdapter,
    load_slot_regime_checkpoint,
    save_slot_regime_checkpoint,
    slot_regime_loss,
)

BATCH, SUPPORT, QUERY, FEATURES, SLOTS = 2, 12, 5, 3, 2


def tiny_backbone(num_outputs: int = 3) -> NanoTabPFNModel:
    torch.manual_seed(41)
    return NanoTabPFNModel(16, 2, 32, 2, num_outputs)


def tiny_model(**kwargs) -> NanoTabPFNSlotRegimeModel:
    return NanoTabPFNSlotRegimeModel(tiny_backbone(), num_slots=SLOTS, max_classes=2, **kwargs)


def tiny_episode(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(BATCH, SUPPORT, FEATURES, generator=generator),
        torch.randint(0, 2, (BATCH, SUPPORT), generator=generator).float(),
        torch.randn(BATCH, QUERY, FEATURES, generator=generator),
    )


class SlotRegimeModelTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model().eval()
        self.support_x, self.support_y, self.query_x = tiny_episode()

    def test_shapes(self):
        prediction = self.model(self.support_x, self.support_y, self.query_x)
        self.assertEqual(prediction.slot_logits.shape, (BATCH, QUERY, SLOTS, 2))
        self.assertEqual(prediction.log_gate.shape, (BATCH, QUERY, SLOTS))
        self.assertEqual(prediction.support_attention.shape, (BATCH, SUPPORT, SLOTS))

    def test_gate_and_marginal_are_normalized(self):
        prediction = self.model(self.support_x, self.support_y, self.query_x)
        torch.testing.assert_close(prediction.gate().sum(-1), torch.ones(BATCH, QUERY), atol=1e-6, rtol=1e-6)
        marginal = prediction.marginal_probabilities()
        torch.testing.assert_close(marginal.sum(-1), torch.ones(BATCH, QUERY), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(marginal).all())

    def test_support_attention_assigns_each_row_across_slots(self):
        """Competitive slot attention: a support row distributes its mass over slots."""
        prediction = self.model(self.support_x, self.support_y, self.query_x)
        torch.testing.assert_close(
            prediction.support_attention.sum(-1), torch.ones(BATCH, SUPPORT), atol=1e-6, rtol=1e-6
        )

    def test_both_calling_conventions_agree(self):
        direct = self.model(self.support_x, self.support_y, self.query_x)
        concatenated = self.model(
            (torch.cat((self.support_x, self.query_x), dim=1), self.support_y),
            train_test_split_index=SUPPORT,
        )
        torch.testing.assert_close(direct.marginal_probabilities(), concatenated.marginal_probabilities())

    def test_support_permutation_invariance(self):
        order = torch.tensor([8, 2, 11, 0, 5, 4, 7, 1, 10, 3, 9, 6])
        original = self.model(self.support_x, self.support_y, self.query_x)
        permuted = self.model(self.support_x[:, order], self.support_y[:, order], self.query_x)
        torch.testing.assert_close(
            original.marginal_probabilities(), permuted.marginal_probabilities(), atol=2e-5, rtol=2e-5
        )

    def test_query_permutation_equivariance(self):
        order = torch.tensor([3, 0, 4, 1, 2])
        original = self.model(self.support_x, self.support_y, self.query_x)
        permuted = self.model(self.support_x, self.support_y, self.query_x[:, order])
        torch.testing.assert_close(
            original.marginal_probabilities()[:, order],
            permuted.marginal_probabilities(),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_support_labels_change_the_prediction(self):
        original = self.model(self.support_x, self.support_y, self.query_x)
        flipped = self.support_y.clone()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        changed = self.model(self.support_x, flipped, self.query_x)
        self.assertFalse(torch.allclose(original.marginal_probabilities(), changed.marginal_probabilities()))

    def test_loss_is_finite_and_trains_backbone_and_slots(self):
        model = tiny_model()
        prediction = model(self.support_x, self.support_y, self.query_x)
        target = torch.randint(0, 2, (BATCH, QUERY))
        loss = slot_regime_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Pretraining trains the whole model, so gradients must reach the
        # backbone as well as the slot module.
        self.assertIsNotNone(model.backbone.transformer_blocks[0].linear1.weight.grad)
        for name in ("slots_mu", "slots_log_sigma"):
            self.assertIsNotNone(getattr(model.slot_binding, name).grad, name)
        self.assertIsNotNone(model.slot_binding.gru.weight_ih.grad)
        self.assertIsNotNone(model.query_projection.weight.grad)

    def test_loss_rejects_mismatched_targets(self):
        prediction = self.model(self.support_x, self.support_y, self.query_x)
        with self.assertRaises(ValueError):
            slot_regime_loss(prediction, torch.zeros(BATCH, QUERY + 1))

    def test_invalid_configurations_raise(self):
        with self.assertRaises(ValueError):
            NanoTabPFNSlotRegimeModel(tiny_backbone(), num_slots=0)
        with self.assertRaises(ValueError):
            NanoTabPFNSlotRegimeModel(tiny_backbone(), max_classes=1)
        with self.assertRaises(ValueError):
            NanoTabPFNSlotRegimeModel(tiny_backbone(num_outputs=2), max_classes=3)
        with self.assertRaises(TypeError):
            self.model(self.support_x, self.support_y)
        with self.assertRaises(TypeError):
            self.model((torch.cat((self.support_x, self.query_x), dim=1), self.support_y))

    def test_adapter_returns_logits_matching_the_mixture(self):
        adapter = SlotLogitsAdapter(self.model).eval()
        logits = adapter(self.support_x, self.support_y, self.query_x)
        self.assertEqual(logits.shape, (BATCH, QUERY, 2))
        torch.testing.assert_close(
            logits.softmax(-1),
            self.model(self.support_x, self.support_y, self.query_x).marginal_probabilities(),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_checkpoint_roundtrip_reproduces_predictions(self):
        expected = self.model(self.support_x, self.support_y, self.query_x).marginal_probabilities()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot_regime.pth"
            save_slot_regime_checkpoint(path, self.model, training_config={"seed": 41})
            loaded, checkpoint = load_slot_regime_checkpoint(path)
        actual = loaded.eval()(self.support_x, self.support_y, self.query_x).marginal_probabilities()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(checkpoint["architecture"]["num_slots"], SLOTS)
        self.assertEqual(checkpoint["training_config"], {"seed": 41})

    def test_non_competitive_variant_normalizes_over_support_rows(self):
        model = tiny_model(competitive_slots=False).eval()
        prediction = model(self.support_x, self.support_y, self.query_x)
        torch.testing.assert_close(prediction.support_attention.sum(-2), torch.ones(BATCH, SLOTS), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
