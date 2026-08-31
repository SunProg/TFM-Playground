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

#: How a slot scores its claim on a support row.
#:
#: ``dot``
#:     Locatello's own compatibility, ``<k(h_n), q(slot_k)> / sqrt(E)``: does
#:     this row *resemble* this slot?  Right for pixels, whose own features
#:     determine which object they belong to.
#: ``likelihood``
#:     ``log p(y_n | x_n, slot_k)``: does this slot's hypothesis *explain* this
#:     row's label?  What marks a minority-regime row is that its label
#:     disagrees with what the majority hypothesis predicts at its features --
#:     a residual against a hypothesis, not a similarity to a prototype.
#: ``additive``
#:     ``dot + softplus(scale) * log p``.  Nests ``dot`` exactly at scale
#:     ``-inf`` and ``likelihood`` at ``+inf``, so it keeps the module genuinely
#:     Slot Attention while giving the label evidence a way in.
COMPATIBILITY_MODES = ("dot", "likelihood", "additive")


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
        compatibility: str = "dot",
        max_classes: int = 2,
        **kwargs: Any,
    ):
        super().__init__(embedding_size, nhead, mlp_hidden_size, **kwargs)
        if compatibility not in COMPATIBILITY_MODES:
            raise ValueError(f"compatibility must be one of {COMPATIBILITY_MODES}, got {compatibility!r}.")
        self.num_slots = num_slots
        self.compatibility = compatibility
        self.max_classes = max_classes
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
        # One key per class, so the row's own label selects which key competes.
        # This is the decoder of a bilinear per-slot classifier, written as a
        # projection: `log p(y|x,s) = <W_y h(x), s> - logsumexp_c <W_c h(x), s>`.
        # The second term is the point -- without it a slot wins rows by growing
        # its norm rather than by predicting them correctly.
        self.class_keys = (
            nn.Linear(embedding_size, max_classes * embedding_size) if compatibility != "dot" else None
        )
        # softplus(0.5413) = 1.0: the label evidence enters at the same weight as
        # the dot product rather than as a zero-initialized side path, which is
        # the mistake `slot_mix` already had to be rescued from.
        self.evidence_scale = nn.Parameter(torch.full((), 0.5413)) if compatibility == "additive" else None
        self._split: int | None = None
        self.support_labels: torch.Tensor | None = None
        self.last_support_attention: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        layer: TransformerEncoderLayer,
        *,
        num_slots: int,
        num_slot_iterations: int,
        competitive_slots: bool,
        compatibility: str = "dot",
        max_classes: int = 2,
    ) -> SlotTransformerEncoderLayer:
        adapted = cls(
            layer.norm1.normalized_shape[0],
            layer.self_attention_between_features.num_heads,
            layer.linear1.out_features,
            num_slots=num_slots,
            num_slot_iterations=num_slot_iterations,
            competitive_slots=competitive_slots,
            compatibility=compatibility,
            max_classes=max_classes,
        )
        missing, unexpected = adapted.load_state_dict(layer.state_dict(), strict=False)
        if unexpected:
            raise ValueError(f"Unexpected pretrained parameters for a slot layer: {sorted(unexpected)}")
        own = ("slot_attention", "write_back", "write_norm", "slot_mix", "class_keys", "evidence_scale")
        if any(not name.startswith(own) for name in missing):
            raise ValueError("The slot layer did not receive every pretrained parameter.")
        return adapted

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        # The hook signature carries no split index, so stash it for the hook.
        self._split = train_test_split_index
        return super().forward(src, train_test_split_index, num_mem_chunks=num_mem_chunks)

    def _log_likelihood(self, support: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """``log p(y_n | x_n, slot_k)`` for every support row and slot.

        ``support`` is ``(batch, rows, embedding)`` -- the target column, so each
        state is already label-conditioned -- and ``slots`` is ``(batch, slots,
        embedding)``.  Returns ``(batch, rows, slots)``.

        Written as a bilinear classifier so the algebra stays visible: the score
        is still a dot product against the slot, with the row's label selecting
        which of ``max_classes`` keys is used, minus the log-partition over
        classes.  The subtraction is what stops a slot from claiming every row
        by growing large, which is the degree of freedom the bare dot product
        leaves open and that one-slot-takes-all exploits.
        """
        if self.class_keys is None or self.support_labels is None:
            raise RuntimeError(
                f"compatibility={self.compatibility!r} needs the support labels; "
                "call bind_support_labels(backbone, support_y) before the forward pass."
            )
        batch, rows, embedding = support.shape
        labels = self.support_labels
        if labels.shape[:2] != (batch, rows):
            raise ValueError(f"support_labels must have shape {(batch, rows)}, got {tuple(labels.shape)}.")
        keys = self.class_keys(support).reshape(batch, rows, self.max_classes, embedding)
        # (batch, rows, classes, slots); scaled like the dot-product path so the
        # logits start at a comparable magnitude.
        scores = torch.einsum("brce,bke->brck", keys, slots) * embedding**-0.5
        index = labels.long().clamp_(0, self.max_classes - 1)
        chosen = scores.gather(2, index[:, :, None, None].expand(batch, rows, 1, self.num_slots)).squeeze(2)
        return chosen - scores.logsumexp(dim=2)

    def _compatibility(self, support: torch.Tensor):
        """The score the slots compete under, or ``None`` for Locatello's own."""
        if self.compatibility == "dot":
            return None

        def score(slots: torch.Tensor) -> torch.Tensor:
            evidence = self._log_likelihood(support, slots)
            if self.compatibility == "likelihood":
                return evidence
            # additive: keep the learned similarity and add the label evidence,
            # so the module still nests Locatello's compatibility exactly.
            normalized = self.slot_attention.norm_inputs(support)
            keys = self.slot_attention.project_k(normalized)
            queries = self.slot_attention.project_q(slots) * self.slot_attention.slot_size**-0.5
            dot = torch.matmul(keys, queries.transpose(-1, -2))
            assert self.evidence_scale is not None
            return dot + nn.functional.softplus(self.evidence_scale) * evidence

        return score

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
        slots, attention = self.slot_attention(support, compatibility=self._compatibility(support))
        self.last_support_attention = attention.detach()
        reconstruction = self.write_norm(self.write_back(target, slots, slots, need_weights=False)[0])
        mix = torch.sigmoid(self.slot_mix)
        updated = (1.0 - mix) * target + mix * reconstruction
        return torch.cat((src[:, :, :-1, :], updated[:, :, None, :]), dim=2)


class SlotBackboneModel(NanoTabPFNModel):
    """A ``NanoTabPFNModel`` whose layers hold slots and receive the labels.

    A likelihood compatibility needs each support row's label, and the layer
    hook is handed only the embedded table.  Overriding ``encode_table`` is what
    makes that automatic: every caller -- training, validation, TabArena, the
    locked-episode evaluators -- goes through it, so none of them has to know
    that some layers score differently, and none of them can forget.

    The labels are not extra model input.  The target column the slots already
    read is built from exactly this tensor, and it holds support rows only, so
    nothing here is visible that ``encode_table`` was not already given.

    Adds no parameters, so the state dictionary is identical to the plain
    model's and checkpoints move between the two freely.
    """

    def encode_table(self, src, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        _, y_src = src
        labels = y_src.squeeze(-1) if y_src.ndim > 2 else y_src
        bind_support_labels(self, labels)
        return super().encode_table(src, train_test_split_index, num_mem_chunks=num_mem_chunks)


def install_slot_layers(
    backbone: NanoTabPFNModel,
    *,
    num_slots: int = 2,
    num_slot_iterations: int = 3,
    competitive_slots: bool = True,
    compatibility: str = "dot",
    max_classes: int = 2,
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
            compatibility=compatibility,
            max_classes=max_classes,
        )
    # Retype in place rather than rebuilding: the caller already holds this
    # object, `SlotBackboneModel` adds no parameters and no constructor
    # arguments, and every existing call site keeps its reference and its state
    # dictionary.  Rebuilding would mean recovering the backbone's constructor
    # arguments here, which this function is not given.
    backbone.__class__ = SlotBackboneModel
    return backbone


def bind_support_labels(backbone: NanoTabPFNModel, support_y: torch.Tensor | None) -> None:
    """Hand every slot layer the support labels for the batch about to be run.

    A likelihood compatibility has to know what each support row's label
    actually was, and the layer hook receives only the embedded table.  The
    labels are *not* extra model input: the target column the slots already read
    is built from them, so this exposes nothing the layer could not see.  Query
    labels never appear here.

    Layers that score by dot product ignore this; the ones that do not raise if
    it was never called, because a silent fallback to the dot product is exactly
    the failure that made twelve earlier runs measure a model nobody intended.
    """
    for layer in backbone.transformer_blocks:
        if isinstance(layer, SlotTransformerEncoderLayer):
            layer.support_labels = support_y


def slot_layer_parameters(backbone: NanoTabPFNModel):
    """Only the slot machinery, for training it against a frozen backbone."""
    for layer in backbone.transformer_blocks:
        if isinstance(layer, SlotTransformerEncoderLayer):
            for module in (layer.slot_attention, layer.write_back, layer.write_norm):
                yield from module.parameters()
            yield layer.slot_mix
            if layer.class_keys is not None:
                yield from layer.class_keys.parameters()
            if layer.evidence_scale is not None:
                yield layer.evidence_scale


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
    "COMPATIBILITY_MODES",
    "SlotBackboneModel",
    "SlotTransformerEncoderLayer",
    "bind_support_labels",
    "collect_support_attention",
    "install_slot_layers",
    "slot_layer_parameters",
]
