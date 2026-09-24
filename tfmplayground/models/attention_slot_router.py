"""Two-hop attention retrieval: query -> slots -> support labels, no MLP anywhere.

Replaces both halves of ``ReconstructionRouter``'s MLP usage (``_SlotDecoder``
for support-row reconstruction, and again for query classification) with the
same attention-retrieval idiom TabPFN-3's many-class decoder uses
(``p_m = softmax_n(q_m . k_n) @ y_n``), applied twice:

  1. support rows attend to slots (query=blind support row, key=slot) to get
     ``responsibilities`` -- this stage still needs the row's identity to be
     genuinely per-row and permutation-covariant, which is why the query here
     is the row's own *blind* (label-masked) representation rather than
     ``row_positions`` (see the conversation this file resolves: row order in
     these SCM/multiregime episodes carries no cross-episode signal, so a
     position-keyed decoder can only ever learn a position-conditioned
     average -- content-derived, label-blind queries fix that without
     reopening the shortcut a fully content-visible query would allow).
  2. each slot's own "value" is the responsibility-weighted average of the
     *real* support labels assigned to it, then query rows attend to slots
     (query=blind query row, key=slot) and predict directly as the
     attention-weighted mixture of those slot values.

There is no separate ``gate``/``slot_logits``/``log_gate`` marginalization:
the second attention's output already is the class probability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.table_slot import TableSlotAdapter
from tfmplayground.models.tabpfn_frozen import FrozenTabPFNBackbone


class AttentionSlotRouter(nn.Module):
    def __init__(self, *, scope, width=24, hidden=48, layers=2, num_slots=4, heads=3, num_classes=2):
        super().__init__()
        self.backbone = NanoTabPFNModel(
            embedding_size=width, num_attention_heads=heads, mlp_hidden_size=hidden, num_layers=layers, num_outputs=2
        )
        self.adapter = TableSlotAdapter(width, hidden, scope=scope, num_slots=num_slots)
        # One projection for both support and query rows: a row's blind
        # representation asks the same question of the slots ("which of you
        # do I belong to / look like") whichever side of the support/query
        # split it happens to sit on -- there is no structural reason for two
        # independently-learned notions of row-to-slot fit here.
        self.row_query = nn.Linear(width, width, bias=False)
        self.slot_key = nn.Linear(width, width, bias=False)
        self.width = width
        self.num_classes = num_classes
        self.last = {}

    def forward(self, support_x, support_y, query_x):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), 1)
        # One encode_table call does both jobs: NanoTabPFNModel's TargetEncoder
        # pads query positions up to num_rows using mean(support_y) internally
        # (nanotabpfn.py's TargetEncoder.forward), so query rows already get
        # the same label-blind treatment a separate masked pass would give
        # them -- while support rows keep their real, per-row labels. No
        # reason to blind the support side too: unlike ReconstructionRouter,
        # nothing here reconstructs a row's own embedding from its own query,
        # so there is no shortcut a label-visible support query could take.
        table = self.backbone.encode_table((x, support_y.float()), split)

        state = self.adapter(table, split)
        slots = state.slots
        # state.pooled_rows, not an independent table.mean(2): for cell/
        # cell_and_data scope the adapter rewrites the table with the
        # cell-level competition's own output before pooling rows from it, so
        # this is the representation the slots were actually built to relate
        # to. Pooling the raw table instead (as an earlier version of this
        # model did) bypasses that rewrite entirely, so row_query and
        # slot_key would live in inconsistent spaces for those two scopes --
        # plausibly why cell/cell_and_data trained meaningfully worse than
        # data scope in the first local pilots.
        rows = state.pooled_rows
        slot_key = self.slot_key(slots)
        row_query = self.row_query(rows)  # one projection, sliced below -- see __init__

        # 1. support rows (real labels) attend to slots -> responsibilities
        inner_similarity = row_query[:, :split] @ slot_key.transpose(-1, -2) / self.width**0.5
        responsibilities = inner_similarity.softmax(-1)

        # 2. each slot's value = responsibility-weighted average of real support labels
        y_onehot = F.one_hot(support_y.long(), num_classes=self.num_classes).to(inner_similarity.dtype)
        slot_weight = responsibilities / (responsibilities.sum(1, keepdim=True) + 1e-8)
        slot_value = torch.einsum("bsk,bsc->bkc", slot_weight, y_onehot)

        # 3. query rows (label-blind: mean-padded by TargetEncoder) attend to slots -> prediction directly
        outer_similarity = row_query[:, split:] @ slot_key.transpose(-1, -2) / self.width**0.5
        alpha = outer_similarity.softmax(-1)
        prediction = torch.einsum("bqk,bkc->bqc", alpha, slot_value)

        # Diagnostic only (not trained on): can a support row's own retrieval
        # mechanism recover its own label? slot_value already includes row i's
        # own label in its average, so comparing prediction[i] against y[i]
        # directly would be trivially self-fulfilling in proportion to
        # slot_weight[i,k] -- leave row i out of each slot's value before
        # scoring it, the way leave-one-out kNN excludes a point from its own
        # neighborhood, so this only reads real generalization, not self-match.
        with torch.no_grad():
            loo_numerator = slot_value[:, None] - slot_weight[..., None] * y_onehot[:, :, None, :]
            loo_denominator = (1 - slot_weight)[..., None] + 1e-8
            slot_value_loo = loo_numerator / loo_denominator
            support_prediction = torch.einsum("bsk,bskc->bsc", responsibilities, slot_value_loo)
            support_nll = attention_slot_router_loss(support_prediction, support_y)
            support_accuracy = (support_prediction.argmax(-1) == support_y).float().mean()

        self.last = {
            "support_nll": support_nll.detach(),
            "support_accuracy": support_accuracy.detach(),
            "mask_row_std": responsibilities.std(1, unbiased=False).mean().detach(),
            "slot_std": slots.std(1, unbiased=False).mean().detach(),
            "query_gate_std": alpha.std(1, unbiased=False).mean().detach(),
            # Per-row peakedness of alpha, distinct from query_gate_std (which
            # is variation *across* rows, not how confident any one row's own
            # distribution over slots is). max prob 1/num_slots = uniform
            # (no preferred slot); entropy log(num_slots) = uniform, 0 = one-hot.
            "query_confidence": alpha.max(-1).values.mean().detach(),
            "query_entropy": (-(alpha * (alpha + 1e-8).log()).sum(-1)).mean().detach(),
        }
        return prediction, responsibilities


class AttentionSlotRouterFrozenTabPFN(nn.Module):
    """``AttentionSlotRouter``, but the backbone is a frozen real TabPFN model
    instead of a jointly-trained ``NanoTabPFNModel``.

    ``FrozenTabPFNBackbone`` is a plain object, not an ``nn.Module``, so it
    contributes no parameters to ``self.parameters()`` -- only ``adapter``,
    ``row_query``, and ``slot_key`` ever receive gradients. See
    ``tabpfn_frozen.py``'s module docstring for the column-count caveat this
    backbone swap introduces (TabPFN groups pairs of raw feature columns into
    one embedding column).
    """

    def __init__(
        self,
        *,
        scope,
        hidden=48,
        num_slots=4,
        num_classes=2,
        layer_index=-1,
        tabpfn_device="cpu",
        tabpfn_model_path=None,
        tabpfn_random_state=0,
    ):
        super().__init__()
        self.backbone = FrozenTabPFNBackbone(
            layer_index=layer_index,
            device=tabpfn_device,
            model_path=tabpfn_model_path,
            random_state=tabpfn_random_state,
        )
        width = self.backbone.embedding_size
        self.adapter = TableSlotAdapter(width, hidden, scope=scope, num_slots=num_slots)
        self.row_query = nn.Linear(width, width, bias=False)
        self.slot_key = nn.Linear(width, width, bias=False)
        self.width = width
        self.num_classes = num_classes
        self.last = {}

    def forward(self, support_x, support_y, query_x):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), 1)
        # Raw integer labels, not support_y.float(): real TabPFNClassifier.fit
        # is a classifier, unlike nanotabpfn's continuous TargetEncoder -- see
        # FrozenTabPFNBackbone.encode_table's docstring.
        table = self.backbone.encode_table((x, support_y), split)

        state = self.adapter(table, split)
        slots = state.slots
        rows = state.pooled_rows
        slot_key = self.slot_key(slots)
        row_query = self.row_query(rows)

        inner_similarity = row_query[:, :split] @ slot_key.transpose(-1, -2) / self.width**0.5
        responsibilities = inner_similarity.softmax(-1)

        y_onehot = F.one_hot(support_y.long(), num_classes=self.num_classes).to(inner_similarity.dtype)
        slot_weight = responsibilities / (responsibilities.sum(1, keepdim=True) + 1e-8)
        slot_value = torch.einsum("bsk,bsc->bkc", slot_weight, y_onehot)

        outer_similarity = row_query[:, split:] @ slot_key.transpose(-1, -2) / self.width**0.5
        alpha = outer_similarity.softmax(-1)
        prediction = torch.einsum("bqk,bkc->bqc", alpha, slot_value)

        with torch.no_grad():
            loo_numerator = slot_value[:, None] - slot_weight[..., None] * y_onehot[:, :, None, :]
            loo_denominator = (1 - slot_weight)[..., None] + 1e-8
            slot_value_loo = loo_numerator / loo_denominator
            support_prediction = torch.einsum("bsk,bskc->bsc", responsibilities, slot_value_loo)
            support_nll = attention_slot_router_loss(support_prediction, support_y)
            support_accuracy = (support_prediction.argmax(-1) == support_y).float().mean()

        self.last = {
            "support_nll": support_nll.detach(),
            "support_accuracy": support_accuracy.detach(),
            "mask_row_std": responsibilities.std(1, unbiased=False).mean().detach(),
            "slot_std": slots.std(1, unbiased=False).mean().detach(),
            "query_gate_std": alpha.std(1, unbiased=False).mean().detach(),
            "query_confidence": alpha.max(-1).values.mean().detach(),
            "query_entropy": (-(alpha * (alpha + 1e-8).log()).sum(-1)).mean().detach(),
        }
        return prediction, responsibilities


def attention_slot_router_loss(prediction: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
    """NLL of the true query label under ``prediction`` -- already a probability, not logits."""
    log_probability = prediction.clamp_min(1e-12).log()
    return F.nll_loss(log_probability.flatten(0, 1), query_y.reshape(-1).long())
