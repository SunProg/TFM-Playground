"""Experimental reconstruction routing for matched learning pilots.

Kept separate from submitted TableSlotModel jobs. Cell slots are aligned to
support row zero for this pilot; this is not a row-permutation-invariant medoid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import SlotRegimePrediction, _SlotDecoder
from tfmplayground.models.table_slot import TableSlotAdapter, row_positions


def bounded_attention(similarity, errors, beta=0.5, eta=0.25):
    errors = errors.float()
    u = (errors / (errors + errors.mean(-1, keepdim=True) + 1e-8)).detach()
    base = similarity.float().softmax(-1)
    if eta == 0:
        return base
    return (1 - eta) * base + eta * (similarity.float() - beta * u[:, None]).softmax(-1)


def align_cell_slots(slots):
    """Align every support row to row zero; apply returned indices to masks too."""
    b, s, k, e = slots.shape
    normalized = F.normalize(slots.detach(), dim=-1)
    costs = (1 - torch.einsum("bke,bsje->bskj", normalized[:, 0], normalized)).cpu().numpy()
    indices = np.empty((b, s, k), dtype=np.int64)
    for batch in range(b):
        for row in range(s):
            _, permutation = linear_sum_assignment(costs[batch, row])
            indices[batch, row] = permutation
    indices = torch.as_tensor(indices, device=slots.device)
    return slots.gather(2, indices[..., None].expand(b, s, k, e)), indices


class ReconstructionRouter(nn.Module):
    def __init__(self, *, scope, variant, width=24, hidden=48, layers=2, num_slots=4, heads=3):
        super().__init__()
        if scope not in ("data", "cell", "cell_and_data") or variant not in ("masks", "values"):
            raise ValueError("Unknown scope or routing variant")
        self.scope, self.variant = scope, variant
        self.backbone = NanoTabPFNModel(
            embedding_size=width, num_attention_heads=heads, mlp_hidden_size=hidden, num_layers=layers, num_outputs=2
        )
        self.adapter = TableSlotAdapter(width, hidden, scope=scope, num_slots=num_slots)
        self.adapter.embedding_reconstruction = scope != "cell"
        self.embedding_decoder = _SlotDecoder(width, hidden, width)
        self.class_decoder = _SlotDecoder(width, hidden, 2)
        self.query_key = nn.Linear(width, width, bias=False)
        self.support_key = nn.Linear(width, width, bias=False)
        # Construct in both arms to preserve matched initialization/random streams.
        self.value = nn.Linear(width, width)
        self.context = nn.Linear(width, width, bias=False)
        self.width = width
        self.last = {}

    def forward(self, support_x, support_y, query_x, *, eta=0.25, shuffle_errors=False, drop_context=False):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), 1)
        table = self.backbone.encode_table((x, support_y.float()), split)
        blind_y = support_y.float().mean(1, keepdim=True).expand_as(support_y)
        blind = self.backbone.encode_table((x, blind_y), split)
        b, _, c, e = table.shape
        cell_masks = None
        if self.scope == "cell":
            # Cell decoder sees per-row slots and column addresses only.
            positions = row_positions(c, e, table)
            cell_input = table + positions[:, None]
            feature_slots, attention = self.adapter.feature_slots(cell_input.reshape(-1, c, e))
            feature_slots = feature_slots.reshape(b, -1, self.adapter.num_slots, e)
            attention = attention.reshape(b, -1, c, self.adapter.num_slots)
            raw_support_slots = feature_slots[:, :split]
            values, masks = self.embedding_decoder(
                positions.expand(b * split, -1, -1), raw_support_slots.reshape(b * split, -1, e)
            )
            cell_masks = masks.softmax(-1).reshape(b, split, c, -1)
            reconstructed = (values * masks.softmax(-1)[..., None]).sum(2).reshape(b, split, c, e)
            target = table[:, :split].detach()
            slots_by_row, permutation = align_cell_slots(raw_support_slots)
            cell_masks = cell_masks.gather(-1, permutation[:, :, None].expand(-1, -1, c, -1))
            responsibilities = cell_masks.mean(2)
            slots = slots_by_row.mean(1)
            rewritten = self.adapter.feature_norm(
                self.adapter.feature_write(torch.einsum("brck,brke->brce", attention, feature_slots))
            )
            mix = self.adapter.feature_mix.sigmoid()
            rows = ((1 - mix) * table + mix * rewritten).mean(2)
            errors = (reconstructed.float() - target.float()).square().mean((-1, -2))
        else:
            state = self.adapter(table, split)
            slots, rows = state.slots, state.pooled_rows
            target = self.adapter.last_embedding_target
            values, masks = self.embedding_decoder(row_positions(split, e, slots).expand(b, -1, -1), slots)
            responsibilities = masks.softmax(-1)
            reconstructed = (values * responsibilities[..., None]).sum(2)
            errors = (reconstructed.float() - target.float()).square().mean(-1)
        mse = errors.mean()
        blind_rows = blind.mean(2)
        q = self.query_key(blind_rows[:, split:])
        keys = self.support_key(blind_rows[:, :split])
        similarity = q @ keys.transpose(-1, -2) / self.width**0.5
        scored_errors = errors.flip(1) if shuffle_errors else errors
        a = bounded_attention(similarity, scored_errors, eta=eta)
        if self.scope == "cell" and self.variant == "values":
            # Retain cell structure until query matching; same column weights
            # retrieve labelled values and aggregate aligned reconstruction masks.
            cell_keys = self.support_key(blind[:, :split])
            cell_attention = (torch.einsum("bqe,bice->bqic", q, cell_keys) / e**0.5).softmax(-1)
            r = torch.einsum("bqic,bick->bqik", cell_attention, cell_masks.detach())
            gate = torch.einsum("bqi,bqik->bqk", a, r)
            retrieved = torch.einsum("bqic,bice->bqie", cell_attention, self.value(table[:, :split]))
            context = torch.einsum("bqi,bqie->bqe", a, retrieved)
        else:
            gate = a @ responsibilities.detach()
            context = a @ self.value(table[:, :split].mean(2))
        query = rows[:, split:]
        if self.variant == "values" and not drop_context:
            query = query + self.context(context)
        logits, _ = self.class_decoder(query, slots)
        output = SlotRegimePrediction(logits, gate.clamp_min(1e-12).log(), responsibilities)
        self.last = {
            "mse": mse,
            "mask_row_std": responsibilities.std(1, unbiased=False).mean().detach(),
            "slot_std": slots.std(1, unbiased=False).mean().detach(),
            "expert_probability_std": logits.softmax(-1).std(2, unbiased=False).mean().detach(),
            "query_gate_std": gate.std(1, unbiased=False).mean().detach(),
            "embedding_variance": target.var(unbiased=False).detach(),
            "target_row_std": target.std(1, unbiased=False).mean().detach(),
        }
        return output, mse
