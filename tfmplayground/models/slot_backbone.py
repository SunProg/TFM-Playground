"""Slot attention *inside* the nanoTabPFN transformer, not on top of it.

The head in ``slot_regime.py`` reads the target column after the backbone has
finished: six layers of full attention across rows, so every row's state is
already a global mixture of every other row's.  Slot attention is then asked to
un-mix something already mixed, and it does not: measured across twelve
configurations, support-row purity equalled the majority-class base rate exactly,
meaning every row's argmax landed on one slot.

The vision model does not face this.  Slot attention there reads CNN features
whose receptive field is *local*, so pixels belonging to different objects have
genuinely different features before any competition happens.  The tabular
analogue of "before the mixing" is inside the block, not after the stack.

So the slots here live in every transformer layer.  After each layer's datapoint
attention they read the support rows, compete for them, and a learned share of
every row state is reconstructed from them.  Being in the loop, they shape the
representation rather than only reading the finished one, and can carry a regime
distinction forward that full row-attention would otherwise average away.

The share starts at one half, and that matters.  The first version of this layer
used a ``tanh`` gate initialized at zero, copying the adapter convention
elsewhere in the package, so an untrained layer was exactly the pretrained one.
That is right for an adapter bolted onto pretrained weights and wrong here:
these models train from scratch, so there is no behaviour to preserve, and a
zero-initialized additive residual is a side-path the optimizer can ignore at no
cost.  It did exactly that -- after 10,000 steps the gates had moved about 1e-4
from zero and the arm reproduced plain nanoTabPFN to four decimal places on
every metric.  Blending instead of adding means silencing the slots is now work
the optimizer has to choose to do.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer
from tfmplayground.models.slot_attention import SlotAttention


class SlotTransformerEncoderLayer(TransformerEncoderLayer):
    """A pretrained layer whose row states pass through a slot bottleneck.

    Subclasses the stock layer and overrides only ``adapt_after_datapoint_attention``,
    the hook the architecture already exposes for research heads, so the
    pretrained parameters keep their state-dictionary names and a vanilla
    checkpoint still loads.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        num_slots: int = 2,
        num_slot_iterations: int = 3,
        competitive_slots: bool = True,
        **kwargs: Any,
    ):
        super().__init__(embedding_size, nhead, mlp_hidden_size, **kwargs)
        self.num_slots = num_slots
        self.slot_attention = SlotAttention(
            num_slots,
            embedding_size,
            mlp_hidden_size,
            num_iterations=num_slot_iterations,
            competitive=competitive_slots,
        )
        self.write_back = nn.MultiheadAttention(embedding_size, nhead, batch_first=True)
        self.write_norm = nn.LayerNorm(embedding_size)
        # A convex blend, not an additive residual behind a zero gate.
        #
        # The first version gated the slot path with tanh(row_gate) initialized
        # at zero, so the layer began as the exact pretrained layer.  That is
        # right for an adapter bolted onto pretrained weights, and wrong here:
        # training is from scratch, so there is no behaviour to preserve, and a
        # zero-initialized additive residual is a side-path the optimizer can
        # ignore for free.  It did -- after 10,000 steps the gates had moved
        # ~1e-4 and the model reproduced vanilla to four decimals.
        #
        # sigmoid(slot_mix) starts at 0.5, so half of every row state comes
        # through the slots from step one.  Suppressing them is now something
        # the optimizer has to actively do, rather than the default.
        self.slot_mix = nn.Parameter(torch.zeros(()))
        self._split: int | None = None
        self.last_support_attention: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        layer: TransformerEncoderLayer,
        *,
        num_slots: int,
        num_slot_iterations: int,
        competitive_slots: bool,
    ) -> SlotTransformerEncoderLayer:
        adapted = cls(
            layer.norm1.normalized_shape[0],
            layer.self_attention_between_features.num_heads,
            layer.linear1.out_features,
            num_slots=num_slots,
            num_slot_iterations=num_slot_iterations,
            competitive_slots=competitive_slots,
        )
        missing, unexpected = adapted.load_state_dict(layer.state_dict(), strict=False)
        if unexpected:
            raise ValueError(f"Unexpected pretrained parameters for a slot layer: {sorted(unexpected)}")
        if any(not name.startswith(("slot_attention", "write_back", "write_norm", "slot_mix")) for name in missing):
            raise ValueError("The slot layer did not receive every pretrained parameter.")
        return adapted

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        # The hook signature carries no split index, so stash it for the hook.
        self._split = train_test_split_index
        return super().forward(src, train_test_split_index, num_mem_chunks=num_mem_chunks)

    def adapt_after_datapoint_attention(self, src: torch.Tensor) -> torch.Tensor:
        """Slots read the support rows, compete, and write back into every row.

        ``src`` is ``(batch, rows, columns, embedding)``.  The target column is
        the one carrying label information, so the competition runs over the
        support rows of that column, and the resulting slots are broadcast back
        to every row -- query rows included, which is how an unlabelled query
        inherits a regime assignment it could not compete for itself.
        """
        if self._split is None:
            return src
        target = src[:, :, -1, :]
        support = target[:, : self._split]
        if support.shape[1] < 1:
            return src
        slots, attention = self.slot_attention(support)
        self.last_support_attention = attention.detach()
        reconstruction = self.write_norm(self.write_back(target, slots, slots, need_weights=False)[0])
        mix = torch.sigmoid(self.slot_mix)
        updated = (1.0 - mix) * target + mix * reconstruction
        return torch.cat((src[:, :, :-1, :], updated[:, :, None, :]), dim=2)


def install_slot_layers(
    backbone: NanoTabPFNModel,
    *,
    num_slots: int = 2,
    num_slot_iterations: int = 3,
    competitive_slots: bool = True,
) -> NanoTabPFNModel:
    """Replace every transformer layer in place with a slot-equipped copy."""
    for index, layer in enumerate(backbone.transformer_blocks):
        if isinstance(layer, SlotTransformerEncoderLayer):
            continue
        backbone.transformer_blocks[index] = SlotTransformerEncoderLayer.from_pretrained(
            layer,
            num_slots=num_slots,
            num_slot_iterations=num_slot_iterations,
            competitive_slots=competitive_slots,
        )
    return backbone


def slot_layer_parameters(backbone: NanoTabPFNModel):
    """Only the slot machinery, for training it against a frozen backbone."""
    for layer in backbone.transformer_blocks:
        if isinstance(layer, SlotTransformerEncoderLayer):
            for module in (layer.slot_attention, layer.write_back, layer.write_norm):
                yield from module.parameters()
            yield layer.slot_mix


def collect_support_attention(backbone: NanoTabPFNModel) -> torch.Tensor | None:
    """Per-row slot attention from the deepest slot layer that produced any.

    The deepest layer is the one whose competition had the most processed
    representation to work with, so it is the one the binding diagnostics should
    score.
    """
    found = None
    for layer in backbone.transformer_blocks:
        if isinstance(layer, SlotTransformerEncoderLayer) and layer.last_support_attention is not None:
            found = layer.last_support_attention
    return found


__all__ = [
    "SlotTransformerEncoderLayer",
    "collect_support_attention",
    "install_slot_layers",
    "slot_layer_parameters",
]
