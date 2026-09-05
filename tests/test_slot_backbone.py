import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer
from tfmplayground.models.slot_backbone import (
    COMPATIBILITY_MODES,
    SlotBackboneMixtureModel,
    SlotBackboneModel,
    SlotTransformerEncoderLayer,
    bind_support_labels,
    collect_support_attention,
    install_slot_layers,
    slot_layer_parameters,
)
from tfmplayground.models.slot_regime import SlotRegimePrediction, slot_regime_loss

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

    def test_slots_contribute_from_initialization(self):
        """The slot path must not be free to ignore.

        An earlier version gated it additively from zero, so an untrained layer
        was bit-exact with the plain backbone.  The optimizer then simply never
        opened it: after 10,000 steps the gates had moved ~1e-4 and the arm
        reproduced vanilla to four decimals.  Half of each row state now comes
        through the slots at initialization, so the output must differ.
        """
        expected = self.plain(self.support_x, self.support_y, self.query_x)
        actual = self.slotted(self.support_x, self.support_y, self.query_x)
        self.assertFalse(torch.allclose(actual, expected))

    def test_mix_starts_at_one_half(self):
        for layer in self.slotted.transformer_blocks:
            self.assertAlmostEqual(float(torch.sigmoid(layer.slot_mix)), 0.5, places=6)

    def test_driving_the_mix_to_zero_recovers_the_plain_backbone(self):
        """Silencing the slots is possible, but the optimizer has to choose it."""
        with torch.no_grad():
            for layer in self.slotted.transformer_blocks:
                layer.slot_mix.fill_(-40.0)  # sigmoid -> 0
        expected = self.plain(self.support_x, self.support_y, self.query_x)
        actual = self.slotted(self.support_x, self.support_y, self.query_x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_support_attention_is_a_per_row_distribution_over_slots(self):
        self.slotted(self.support_x, self.support_y, self.query_x)
        attention = collect_support_attention(self.slotted)
        self.assertEqual(attention.shape, (BATCH, SUPPORT, SLOTS))
        torch.testing.assert_close(attention.sum(-1), torch.ones(BATCH, SUPPORT), atol=1e-5, rtol=1e-5)

    def test_gradients_reach_the_slot_machinery(self):
        model = install_slot_layers(backbone(), num_slots=SLOTS)
        model(self.support_x, self.support_y, self.query_x).square().mean().backward()
        for layer in model.transformer_blocks:
            self.assertIsNotNone(layer.slot_mix.grad)
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

    def test_slot_position_selects_the_expected_states_and_call_count(self):
        """Single placements run once on their boundary; the dual placement
        runs on both sides of feature attention and nowhere else."""
        source = torch.randn(BATCH, SUPPORT + QUERY, FEATURES + 1, 16)
        for position in (
            "before_feature",
            "after_feature",
            "before_and_after_feature",
            "after_datapoint",
        ):
            layer = SlotTransformerEncoderLayer(16, 2, 32, slot_position=position).eval()
            post_feature = layer.feature_attention_stage(source)
            post_datapoint = layer.datapoint_attention_stage(
                post_feature, train_test_split_index=SUPPORT
            )
            with mock.patch.object(layer, "_slot_write_back", side_effect=lambda value: value) as apply_slots:
                layer(source, train_test_split_index=SUPPORT)
            expected = {
                "before_feature": (source,),
                "after_feature": (post_feature,),
                "before_and_after_feature": (source, post_feature),
                "after_datapoint": (post_datapoint,),
            }[position]
            self.assertEqual(apply_slots.call_count, len(expected))
            for call, expected_state in zip(apply_slots.call_args_list, expected, strict=True):
                torch.testing.assert_close(call.args[0], expected_state, atol=0, rtol=0)

    def test_historical_slot_position_is_the_default(self):
        for layer in self.slotted.transformer_blocks:
            self.assertEqual(layer.slot_position, "after_datapoint")

    def test_unknown_slot_position_is_rejected(self):
        with self.assertRaises(ValueError):
            install_slot_layers(backbone(), num_slots=SLOTS, slot_position="between_attentions")

    def test_dual_placement_reuses_one_slot_parameter_set(self):
        before = install_slot_layers(backbone(), num_slots=SLOTS, slot_position="before_feature")
        dual = install_slot_layers(backbone(), num_slots=SLOTS, slot_position="before_and_after_feature")
        self.assertEqual(sum(p.numel() for p in before.parameters()), sum(p.numel() for p in dual.parameters()))


class CompatibilityTests(unittest.TestCase):
    """What a slot's claim on a support row is scored by.

    The dot product asks whether a row *resembles* a slot, which is the right
    question for pixels and the wrong one here: what marks a minority-regime row
    is that its label disagrees with what the majority hypothesis predicts at
    its features.  These check that the alternative scores are wired in, that
    the default is untouched, and that a missing label is loud rather than a
    silent fallback.
    """

    def setUp(self):
        self.support_x, self.support_y, self.query_x = episode()

    def _model(self, compatibility: str):
        return install_slot_layers(backbone(), num_slots=SLOTS, compatibility=compatibility).eval()

    def test_default_is_unchanged(self):
        """Every earlier run scored by dot product; retyping the model and adding
        the compatibility switch must not have moved that path at all."""
        torch.manual_seed(0)
        explicit = self._model("dot")(self.support_x, self.support_y, self.query_x)
        torch.manual_seed(0)
        default = install_slot_layers(backbone(), num_slots=SLOTS).eval()(
            self.support_x, self.support_y, self.query_x
        )
        torch.testing.assert_close(explicit, default, atol=0, rtol=0)

    def test_each_mode_scores_the_rows_differently(self):
        attentions = {}
        for mode in COMPATIBILITY_MODES:
            model = self._model(mode)
            model(self.support_x, self.support_y, self.query_x)
            attentions[mode] = collect_support_attention(model)
            self.assertEqual(attentions[mode].shape, (BATCH, SUPPORT, SLOTS))
            torch.testing.assert_close(
                attentions[mode].sum(-1), torch.ones(BATCH, SUPPORT), atol=1e-5, rtol=1e-5
            )
        self.assertFalse(torch.allclose(attentions["dot"], attentions["likelihood"]))
        self.assertFalse(torch.allclose(attentions["dot"], attentions["additive"]))

    def test_the_label_actually_changes_the_competition(self):
        """The point of the likelihood score: flipping a row's label must move
        which slot claims it.  Under the dot product the label reaches the row
        state too, but nothing forces the competition to use it."""
        model = self._model("likelihood")
        model(self.support_x, self.support_y, self.query_x)
        before = collect_support_attention(model).clone()
        flipped = self.support_y.clone()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        model(self.support_x, flipped, self.query_x)
        self.assertFalse(torch.allclose(before[:, 0], collect_support_attention(model)[:, 0]))

    def test_labels_are_bound_automatically_by_the_model(self):
        """Every caller reaches the layers through `encode_table`, so none of
        them has to know some layers score differently -- and none can forget."""
        model = self._model("likelihood")
        self.assertIsInstance(model, SlotBackboneModel)
        model(self.support_x, self.support_y, self.query_x)
        for layer in model.transformer_blocks:
            self.assertEqual(layer.support_labels.shape, (BATCH, SUPPORT))

    def test_a_missing_label_raises_rather_than_falling_back(self):
        """A silent fallback to the dot product is exactly what made twelve
        earlier runs measure a model nobody configured."""
        model = self._model("likelihood")
        bind_support_labels(model, None)
        support = torch.randn(BATCH, SUPPORT, 16)
        with self.assertRaises(RuntimeError):
            model.transformer_blocks[0]._log_likelihood(support, torch.randn(BATCH, SLOTS, 16))

    def test_gradients_reach_the_evidence_parameters(self):
        for mode in ("likelihood", "additive"):
            model = install_slot_layers(backbone(), num_slots=SLOTS, compatibility=mode)
            model(self.support_x, self.support_y, self.query_x).square().mean().backward()
            for layer in model.transformer_blocks:
                self.assertIsNotNone(layer.class_keys.weight.grad)
                if mode == "additive":
                    self.assertIsNotNone(layer.evidence_scale.grad)

    def test_evidence_enters_at_unit_weight(self):
        """Not behind a zero-initialized gate: `slot_mix` already had to be
        rescued from exactly that, having been ignored for 10,000 steps."""
        for layer in self._model("additive").transformer_blocks:
            self.assertAlmostEqual(float(torch.nn.functional.softplus(layer.evidence_scale.detach())), 1.0, places=4)

    def test_evidence_parameters_are_counted_as_slot_machinery(self):
        model = self._model("additive")
        slot_only = sum(p.numel() for p in slot_layer_parameters(model))
        pretrained = sum(p.numel() for p in backbone().parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters()) - slot_only, pretrained)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            install_slot_layers(backbone(), num_slots=SLOTS, compatibility="cosine")


class MixtureReadoutTests(unittest.TestCase):
    """The objective has to mention the slots, or nothing rewards using them.

    `slot_backbone` trains on one cross entropy over the finished
    representation, so the loss cannot see which slot claimed which row and
    one-slot-takes-all costs nothing -- measured across two studies and both
    compatibility functions, `purity - base` was exactly zero everywhere.  This
    readout decodes one prediction per slot and trains on the mixture NLL, so
    the loss decomposes over slots while the competition still runs inside the
    layers.
    """

    def setUp(self):
        self.support_x, self.support_y, self.query_x = episode()
        self.model = SlotBackboneMixtureModel(install_slot_layers(backbone(), num_slots=SLOTS)).eval()

    def test_shapes_and_a_normalized_mixture(self):
        p = self.model(self.support_x, self.support_y, self.query_x)
        self.assertEqual(p.slot_logits.shape, (BATCH, QUERY, SLOTS, 2))
        self.assertEqual(p.log_gate.shape, (BATCH, QUERY, SLOTS))
        self.assertEqual(p.support_attention.shape, (BATCH, SUPPORT, SLOTS))
        torch.testing.assert_close(p.gate().sum(-1), torch.ones(BATCH, QUERY), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            p.marginal_probabilities().sum(-1), torch.ones(BATCH, QUERY), atol=1e-5, rtol=1e-5
        )

    def test_the_loss_reaches_the_competition(self):
        """The whole point: gradient from the mixture NLL must arrive at the slot
        machinery inside the layers, not stop at a decoder bolted on top."""
        model = SlotBackboneMixtureModel(install_slot_layers(backbone(), num_slots=SLOTS))
        target = torch.randint(0, 2, (BATCH, QUERY))
        slot_regime_loss(model(self.support_x, self.support_y, self.query_x), target).backward()
        self.assertIsNotNone(model.decoder.body[0].weight.grad)
        for layer in model.backbone.transformer_blocks:
            self.assertIsNotNone(layer.slot_attention.slots_mu.grad)
            self.assertIsNotNone(layer.slot_attention.gru.weight_ih.grad)
            self.assertIsNotNone(layer.slot_mix.grad)

    def test_new_placement_losses_reach_slots_write_back_and_decoder(self):
        for position in ("before_feature", "before_and_after_feature"):
            model = SlotBackboneMixtureModel(
                install_slot_layers(backbone(), num_slots=SLOTS, slot_position=position)
            )
            target = torch.randint(0, 2, (BATCH, QUERY))
            slot_regime_loss(model(self.support_x, self.support_y, self.query_x), target).backward()
            self.assertIsNotNone(model.decoder.body[0].weight.grad)
            for layer in model.backbone.transformer_blocks:
                self.assertIsNotNone(layer.slot_attention.slots_mu.grad)
                self.assertIsNotNone(layer.write_back.in_proj_weight.grad)
                self.assertIsNotNone(layer.slot_mix.grad)

    def test_slots_are_kept_undetached_for_that_gradient(self):
        """`last_support_attention` is detached because it is only ever scored;
        `last_slots` must not be, because the loss trains through it."""
        model = SlotBackboneMixtureModel(install_slot_layers(backbone(), num_slots=SLOTS))
        model(self.support_x, self.support_y, self.query_x)
        for layer in model.backbone.transformer_blocks:
            self.assertTrue(layer.last_slots.requires_grad)
            self.assertFalse(layer.last_support_attention.requires_grad)

    def test_marginal_is_invariant_to_slot_order(self):
        """`logsumexp` over slots is a sum, so it cannot depend on their order --
        which is why no slot-to-regime matching appears in the loss."""
        p = self.model(self.support_x, self.support_y, self.query_x)
        permutation = torch.tensor([1, 2, 0])
        shuffled = SlotRegimePrediction(
            slot_logits=p.slot_logits[:, :, permutation],
            log_gate=p.log_gate[:, :, permutation],
            support_attention=p.support_attention[:, :, permutation],
        )
        torch.testing.assert_close(
            shuffled.marginal_log_probabilities(), p.marginal_log_probabilities(), atol=1e-6, rtol=1e-6
        )

    def test_both_calling_conventions_agree(self):
        """Training calls `model(support_x, support_y, query_x)`; the TabArena
        predictor calls `model((x, y), train_test_split_index=...)`.  Supporting
        only the first is what killed the first submission of this model at its
        first epoch boundary, after the training loop had run fine for 500 steps.
        """
        expected = self.model(self.support_x, self.support_y, self.query_x)
        table = torch.cat((self.support_x, self.query_x), dim=1)
        actual = self.model(
            (table, self.support_y), train_test_split_index=SUPPORT, num_mem_chunks=1
        )
        torch.testing.assert_close(
            actual.marginal_log_probabilities(), expected.marginal_log_probabilities(), atol=1e-6, rtol=1e-6
        )

    def test_the_tabarena_predictor_accepts_it(self):
        """The exact call path that failed: `predict_vanilla` chunks the query
        rows and uses the concatenated-table interface.  Exercising the real
        function rather than imitating it is the point -- the interface test
        above would have passed against a wrapper that still mismatched here.
        """
        from tfmplayground.experiments.evaluate_integrated_tabarena import predict_vanilla
        from tfmplayground.models.slot_regime import SlotLogitsAdapter

        rng = np.random.default_rng(0)
        probabilities = predict_vanilla(
            SlotLogitsAdapter(self.model),
            rng.normal(size=(SUPPORT, FEATURES)).astype(np.float32),
            rng.integers(0, 2, SUPPORT).astype(np.float32),
            rng.normal(size=(7, FEATURES)).astype(np.float32),
            device="cpu",
            query_chunk_size=4,  # smaller than the query count, so it chunks
            num_mem_chunks=1,
        )
        self.assertEqual(probabilities.shape, (7,))
        self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))

    def test_the_concatenated_interface_requires_its_split(self):
        table = torch.cat((self.support_x, self.query_x), dim=1)
        with self.assertRaises(TypeError):
            self.model((table, self.support_y))
        with self.assertRaises(TypeError):
            self.model((table, self.support_y), train_test_split_index=SUPPORT, nonsense=1)

    def test_a_backbone_without_slot_layers_is_rejected(self):
        with self.assertRaises(ValueError):
            SlotBackboneMixtureModel(backbone())

    def test_checkpoint_round_trip_through_the_plain_signature(self):
        """Every evaluator calls this like a NanoTabPFNModel, so it has to come
        back wrapped and return logits of the ordinary shape."""
        from tfmplayground.models.slot_regime import load_checkpoint_for_inference

        expected = self.model(self.support_x, self.support_y, self.query_x).marginal_log_probabilities()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(
                {
                    "architecture": {
                        "embedding_size": 16,
                        "num_attention_heads": 2,
                        "mlp_hidden_size": 32,
                        "num_layers": 2,
                        "num_outputs": 3,
                        "model_kind": "slot_backbone_mixture",
                        "num_slots": SLOTS,
                        "max_classes": 2,
                    },
                    "model": self.model.state_dict(),
                },
                path,
            )
            restored = load_checkpoint_for_inference(path)
        self.assertTrue(
            all(layer.slot_position == "after_datapoint" for layer in restored.backbone.transformer_blocks)
        )
        actual = restored(self.support_x, self.support_y, self.query_x)
        self.assertEqual(actual.shape, (BATCH, QUERY, 2))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_new_checkpoint_preserves_feature_boundary_positions(self):
        from tfmplayground.models.slot_regime import load_checkpoint_for_inference

        for position in ("before_feature", "after_feature", "before_and_after_feature"):
            model = SlotBackboneMixtureModel(
                install_slot_layers(backbone(), num_slots=SLOTS, slot_position=position)
            ).eval()
            expected = model(self.support_x, self.support_y, self.query_x).marginal_log_probabilities()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pth"
                torch.save(
                    {
                        "architecture": {
                            "embedding_size": 16,
                            "num_attention_heads": 2,
                            "mlp_hidden_size": 32,
                            "num_layers": 2,
                            "num_outputs": 3,
                            "model_kind": "slot_backbone_mixture",
                            "num_slots": SLOTS,
                            "max_classes": 2,
                            "slot_position": position,
                        },
                        "model": model.state_dict(),
                    },
                    path,
                )
                restored = load_checkpoint_for_inference(path)
            self.assertTrue(all(layer.slot_position == position for layer in restored.backbone.transformer_blocks))
            torch.testing.assert_close(
                restored(self.support_x, self.support_y, self.query_x), expected, atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
