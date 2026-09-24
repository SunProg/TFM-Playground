"""Causal ablation model for query routing and slot supervision.

The three switches are deliberately orthogonal:

``query_retrieval``
    Route a query through label-blind query-to-support attention and lift the
    retrieved support assignments into a query-specific slot gate.
``slot_loss``
    Add support-label reconstruction and balanced/sharp slot supervision.
``blind_routing``
    Keep the labelled backbone pass available only for support values.  Slot
    construction, query routing, and query decoding use a label-blind pass.

All parameters are present for every arm so the combination sweep changes the
information path and objective, not the parameter count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_attention import SlotAttention, slot_assignment_entropy
from tfmplayground.models.slot_regime import SlotRegimePrediction, _SlotDecoder, slot_mi_loss


class CausalRoutingModel(nn.Module):
    """A fixed-capacity slot model with independently switchable fixes."""

    model_type = "causal_routing_ablation"

    def __init__(
        self,
        *,
        query_retrieval: bool,
        slot_loss: bool,
        blind_routing: bool,
        width: int = 192,
        hidden: int = 768,
        layers: int = 6,
        heads: int = 6,
        num_slots: int = 4,
    ) -> None:
        super().__init__()
        self.query_retrieval = bool(query_retrieval)
        self.slot_loss_enabled = bool(slot_loss)
        self.blind_routing = bool(blind_routing)
        self.num_slots = num_slots
        self.width = width
        self.backbone = NanoTabPFNModel(
            embedding_size=width,
            num_attention_heads=heads,
            mlp_hidden_size=hidden,
            num_layers=layers,
            num_outputs=2,
        )
        self.slot_binding = SlotAttention(
            num_slots,
            width,
            hidden,
            num_iterations=3,
            competitive=True,
            max_log_sigma=2.0,
        )
        self.decoder = _SlotDecoder(width, hidden, 2)
        # These modules remain allocated in every arm to keep parameter count
        # and initialization structure matched across the causal sweep.
        self.query_key = nn.Linear(width, width, bias=False)
        self.support_key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.context = nn.Linear(width, width, bias=False)
        self.last: dict[str, torch.Tensor] = {}

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor):
        split = support_x.shape[1]
        x = torch.cat((support_x, query_x), dim=1)
        labelled = self.backbone.encode_table((x, support_y.float()), split)
        blind_y = support_y.float().mean(dim=1, keepdim=True).expand_as(support_y)
        blind = self.backbone.encode_table((x, blind_y), split)
        labelled_rows = labelled[:, :, -1, :]
        blind_rows = blind[:, :, -1, :]

        routing_rows = blind_rows if self.blind_routing else labelled_rows
        support_routing = routing_rows[:, :split]
        query_routing = routing_rows[:, split:]
        slots, support_attention = self.slot_binding(support_routing)

        query_rows = blind_rows[:, split:] if self.blind_routing else labelled_rows[:, split:]
        if self.query_retrieval:
            query_keys = self.query_key(query_routing)
            support_keys = self.support_key(support_routing)
            similarity = query_keys @ support_keys.transpose(-1, -2) / self.width**0.5
            row_attention = similarity.softmax(dim=-1)
            log_gate = (row_attention @ support_attention).clamp_min(1e-8).log()
            # Labels enter through values only.  This is the direct support
            # information path that a query can use after retrieval.
            context = row_attention @ self.value(labelled_rows[:, :split])
            query_rows = query_rows + self.context(context)
        else:
            row_attention = query_rows.new_zeros(query_rows.shape[0], query_rows.shape[1], split)

        slot_logits, mask_logits = self.decoder(query_rows, slots)
        if not self.query_retrieval:
            log_gate = F.log_softmax(mask_logits, dim=-1)

        support_logits, _ = self.decoder(routing_rows[:, :split], slots)
        support_log_probabilities = F.log_softmax(support_logits, dim=-1)
        support_log_probabilities = torch.logsumexp(
            support_attention.clamp_min(1e-8).log()[..., None] + support_log_probabilities, dim=2
        )
        query_entropy = (
            -(log_gate.exp() * log_gate).sum(-1) / torch.log(log_gate.new_tensor(self.num_slots))
        ).mean()
        row_attention_entropy = (
            -(row_attention.clamp_min(1e-8) * row_attention.clamp_min(1e-8).log()).sum(-1)
            / torch.log(row_attention.new_tensor(max(2, split)))
        ).mean()
        self.last = {
            "support_entropy": slot_assignment_entropy(support_attention).mean().detach(),
            "support_confidence": support_attention.max(dim=-1).values.mean().detach(),
            "query_entropy": query_entropy.detach(),
            "query_confidence": log_gate.exp().max(dim=-1).values.mean().detach(),
            "query_gate_std": log_gate.exp().std(dim=1, unbiased=False).mean().detach(),
            "row_attention_entropy": row_attention_entropy.detach(),
            "slot_probability_std": slot_logits.softmax(-1).std(dim=2, unbiased=False).mean().detach(),
        }
        prediction = SlotRegimePrediction(
            slot_logits=slot_logits,
            log_gate=log_gate,
            support_attention=support_attention,
            support_reconstruction_log_probabilities=support_log_probabilities,
        )
        return prediction

    def slot_loss(self, prediction: SlotRegimePrediction, support_y: torch.Tensor) -> torch.Tensor:
        """Support-label loss that makes support assignments consequential."""
        if not self.slot_loss_enabled:
            return support_y.new_zeros((), dtype=torch.float32)
        if prediction.support_reconstruction_log_probabilities is None:
            raise RuntimeError("support reconstruction was not produced")
        support_nll = F.nll_loss(
            prediction.support_reconstruction_log_probabilities.flatten(0, 1), support_y.reshape(-1).long()
        )
        mi = slot_mi_loss(prediction.support_attention)
        # The support NLL trains the assigned slot to predict; MI keeps the
        # assignment from becoming uniformly diffuse or one-slot collapse.
        return support_nll + 0.05 * mi
