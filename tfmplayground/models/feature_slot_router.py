"""Route rows by which columns matter to them, not by row-pooled similarity.

AttentionSlotRouter's slots live in the space of pooled *rows* (one vector per
row, columns averaged away before anything competes). This model's slots live
in the space of *columns*: each of the K slots is a learned "feature
relevance" key that attends over one row's own C columns to build a
slot-specific weighted view of that row, before anything about rows competes.

This is a direct answer to what AttentionSlotRouter's `cell` scope was
actually doing without saying so: TableSlotAdapter's `feature_slots` clusters
one row's own columns internally then discards the row axis by averaging
across rows (`cell_state()`'s `slots[:, :split].mean(dim=1)`) -- a
feature-clustering mechanism pressed into serving as row/regime slots, which
AttentionSlotRouter's query-to-slot retrieval was never built to consume.
Here the feature axis stays the organizing principle end to end: slots keep
their column-relevance identity, rows are scored against that identity
directly, and nothing about columns is averaged away before it matters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class FeatureSlotRouter(nn.Module):
    def __init__(self, *, width=24, hidden=48, layers=2, num_slots=4, heads=3, num_classes=2):
        super().__init__()
        self.backbone = NanoTabPFNModel(
            embedding_size=width, num_attention_heads=heads, mlp_hidden_size=hidden, num_layers=layers, num_outputs=2
        )
        # One key per slot, shared across every row and every episode: what
        # column-pattern this slot specializes in. Not derived from any one
        # row -- it has to generalize across rows for the column-attention
        # below to mean anything.
        self.slot_feature_key = nn.Parameter(torch.randn(num_slots, width) / width**0.5)
        self.column_key = nn.Linear(width, width, bias=False)
        # Shared across slots on purpose: "how relevant is this weighted
        # view" should mean the same thing regardless of which slot produced
        # it, the same way support_query/query_query got unified into one
        # row_query in AttentionSlotRouter for the analogous reason.
        self.slot_score = nn.Linear(width, 1, bias=False)
        self.width = width
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.last = {}

    def forward(self, support_x, support_y, query_x):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), 1)
        table = self.backbone.encode_table((x, support_y.float()), split)  # (B, R, C, E)

        # 1. each slot attends over each row's own C columns -> a per-(row,
        #    slot) column-weighted view of that row. Independent per row, so
        #    nothing here depends on row order.
        column_key = self.column_key(table)
        similarity = torch.einsum("brce,ke->brkc", column_key, self.slot_feature_key) / self.width**0.5
        column_attention = similarity.softmax(-1)
        slot_view = torch.einsum("brkc,brce->brke", column_attention, table)

        # 2. which slot's view of this row is most relevant -> responsibility,
        #    competitive over slots (softmax over k, not over rows).
        relevance = self.slot_score(slot_view).squeeze(-1)
        responsibility = relevance.softmax(-1)
        support_responsibility = responsibility[:, :split]
        query_responsibility = responsibility[:, split:]

        # 3. each slot's value = responsibility-weighted average of real support labels
        y_onehot = F.one_hot(support_y.long(), num_classes=self.num_classes).to(relevance.dtype)
        slot_weight = support_responsibility / (support_responsibility.sum(1, keepdim=True) + 1e-8)
        slot_value = torch.einsum("bsk,bsc->bkc", slot_weight, y_onehot)

        # 4. query prediction = query's own responsibility-weighted mixture of slot values
        prediction = torch.einsum("bqk,bkc->bqc", query_responsibility, slot_value)

        with torch.no_grad():
            loo_numerator = slot_value[:, None] - slot_weight[..., None] * y_onehot[:, :, None, :]
            loo_denominator = (1 - slot_weight)[..., None] + 1e-8
            slot_value_loo = loo_numerator / loo_denominator
            support_prediction = torch.einsum(
                "bsk,bskc->bsc", support_responsibility, slot_value_loo
            )
            support_nll = feature_slot_router_loss(support_prediction, support_y)
            support_accuracy = (support_prediction.argmax(-1) == support_y).float().mean()

        self.last = {
            "support_nll": support_nll.detach(),
            "support_accuracy": support_accuracy.detach(),
            "mask_row_std": support_responsibility.std(1, unbiased=False).mean().detach(),
            "query_gate_std": query_responsibility.std(1, unbiased=False).mean().detach(),
            "column_attention_std": column_attention.std(-1, unbiased=False).mean().detach(),
        }
        return prediction, support_responsibility


def feature_slot_router_loss(prediction: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
    """NLL of the true label under `prediction` -- already a probability, not logits."""
    log_probability = prediction.clamp_min(1e-12).log()
    return F.nll_loss(log_probability.flatten(0, 1), query_y.reshape(-1).long())
