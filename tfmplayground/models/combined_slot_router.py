"""Feature-slot attention, then data-slot attention, on the enriched result.

Combines the two routers built this session: FeatureSlotRouter's column-level
attention (learned per-slot feature-relevance keys, shared across rows and
episodes) enriches each row's pooled representation, and AttentionSlotRouter's
row-level attention (learned per-slot data-relevance keys) then routes that
enriched representation to real support labels exactly as before.

This is the same two-stage shape TableSlotAdapter's cell_and_data scope has
(feature competition rewrites the table, row competition runs on the
rewritten table) -- but built from the two mechanisms validated this session
instead of SlotAttention's per-row-then-averaged-across-rows path, which is
what made plain `cell`/`cell_and_data` scope a poor fit for
AttentionSlotRouter's row-to-slot retrieval in the first place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class CombinedSlotRouter(nn.Module):
    def __init__(
        self,
        *,
        width=24,
        hidden=48,
        layers=2,
        num_feature_slots=4,
        num_data_slots=4,
        heads=3,
        num_classes=2,
    ):
        super().__init__()
        self.backbone = NanoTabPFNModel(
            embedding_size=width, num_attention_heads=heads, mlp_hidden_size=hidden, num_layers=layers, num_outputs=2
        )

        # --- feature step: which columns matter to this row ---
        self.slot_feature_key = nn.Parameter(torch.randn(num_feature_slots, width) / width**0.5)
        self.column_key = nn.Linear(width, width, bias=False)
        self.feature_score = nn.Linear(width, 1, bias=False)
        # Residual gate blending the feature-weighted summary into the plain
        # pooled row -- same role table_slot.py's feature_mix plays for its
        # own cell_and_data rewrite, and the same zero-initialized parameter
        # convention: sigmoid(0)=0.5, an even blend, not a passive one. A
        # near-zero *gate* would let training silently ignore the feature
        # step by driving mix toward 0; starting already-active at 0.5 means
        # the feature signal has to be learned away, not defaulted away.
        self.feature_mix = nn.Parameter(torch.zeros(()))

        # --- data step: which regime this (enriched) row belongs to ---
        self.data_slot_key = nn.Parameter(torch.randn(num_data_slots, width) / width**0.5)
        self.row_query = nn.Linear(width, width, bias=False)

        self.width = width
        self.num_feature_slots = num_feature_slots
        self.num_data_slots = num_data_slots
        self.num_classes = num_classes
        self.last = {}

    def forward(self, support_x, support_y, query_x):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), 1)
        table = self.backbone.encode_table((x, support_y.float()), split)  # (B, R, C, E)

        # === feature step ===
        column_key = self.column_key(table)
        feature_similarity = (
            torch.einsum("brce,fe->brfc", column_key, self.slot_feature_key) / self.width**0.5
        )
        column_attention = feature_similarity.softmax(-1)  # over columns, per (row, feature-slot)
        feature_slot_view = torch.einsum("brfc,brce->brfe", column_attention, table)  # (B,R,F,E)
        feature_relevance = self.feature_score(feature_slot_view).squeeze(-1)  # (B,R,F)
        feature_responsibility = feature_relevance.softmax(-1)  # competitive over feature-slots
        feature_summary = torch.einsum("brf,brfe->bre", feature_responsibility, feature_slot_view)

        # Residual addition, not a convex blend: (1-mix)*plain + mix*feature
        # would trade the two signals off against each other -- whatever mix
        # settles on necessarily dampens plain_rows to make room for the
        # feature signal. Adding instead means plain_rows is never
        # diminished; the feature step can only contribute on top of it, not
        # compete with it for the same budget.
        plain_rows = table.mean(2)
        mix = self.feature_mix.sigmoid()
        rows = plain_rows + mix * feature_summary

        # === data step (AttentionSlotRouter's mechanism, on enriched rows) ===
        row_query = self.row_query(rows)
        data_similarity = torch.einsum("bre,de->brd", row_query, self.data_slot_key) / self.width**0.5
        data_attention = data_similarity.softmax(-1)
        support_responsibility = data_attention[:, :split]
        query_responsibility = data_attention[:, split:]

        y_onehot = F.one_hot(support_y.long(), num_classes=self.num_classes).to(data_similarity.dtype)
        slot_weight = support_responsibility / (support_responsibility.sum(1, keepdim=True) + 1e-8)
        slot_value = torch.einsum("bsd,bsc->bdc", slot_weight, y_onehot)

        prediction = torch.einsum("bqd,bdc->bqc", query_responsibility, slot_value)

        with torch.no_grad():
            loo_numerator = slot_value[:, None] - slot_weight[..., None] * y_onehot[:, :, None, :]
            loo_denominator = (1 - slot_weight)[..., None] + 1e-8
            slot_value_loo = loo_numerator / loo_denominator
            support_prediction = torch.einsum("bsd,bsdc->bsc", support_responsibility, slot_value_loo)
            support_nll = combined_slot_router_loss(support_prediction, support_y)
            support_accuracy = (support_prediction.argmax(-1) == support_y).float().mean()

        self.last = {
            "support_nll": support_nll.detach(),
            "support_accuracy": support_accuracy.detach(),
            "feature_mix": mix.detach(),
            "feature_responsibility_std": feature_responsibility.std(-1, unbiased=False).mean().detach(),
            "mask_row_std": support_responsibility.std(1, unbiased=False).mean().detach(),
            "query_gate_std": query_responsibility.std(1, unbiased=False).mean().detach(),
        }
        return prediction, support_responsibility


def combined_slot_router_loss(prediction: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
    """NLL of the true label under `prediction` -- already a probability, not logits."""
    log_probability = prediction.clamp_min(1e-12).log()
    return F.nll_loss(log_probability.flatten(0, 1), query_y.reshape(-1).long())
