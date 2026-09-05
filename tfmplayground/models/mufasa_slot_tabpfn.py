"""MUFASA-style multi-layer slot readout for nanoTabPFN.

The backbone remains an ordinary nanoTabPFN.  Selected intermediate target-row
representations each receive an independent slot-attention module; slots are
re-ordered with a detached Hungarian assignment, then fused before the shared
per-slot decoder.  Unlike the historical slot-backbone adapter this does not
write slots into every transformer block, so the layer complementarity remains
available to the final predictor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_attention import SlotAttention
from tfmplayground.models.slot_regime import SlotRegimePrediction, _SlotDecoder


class MufasaSlotTabPFNModel(nn.Module):
    """A standard backbone with independently tapped, fused slot modules."""

    model_type = "multiregime_v2_mufasa_slot_tabpfn"

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        num_slots: int = 4,
        layer_indices: Sequence[int] = (2, 3, 4, 5),
        num_slot_iterations: int = 3,
        max_classes: int = 2,
    ):
        super().__init__()
        layers = tuple(int(index) for index in layer_indices)
        if not layers:
            raise ValueError("MUFASA needs at least one tapped layer.")
        if len(set(layers)) != len(layers):
            raise ValueError("layer_indices must be unique.")
        if any(index < 0 or index >= backbone.num_layers for index in layers):
            raise ValueError(f"layer_indices must lie in [0, {backbone.num_layers}).")
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")
        self.backbone = backbone
        self.num_slots = num_slots
        self.layer_indices = layers
        self.max_classes = max_classes
        self.slot_modules = nn.ModuleList(
            SlotAttention(
                num_slots,
                backbone.embedding_size,
                backbone.mlp_hidden_size,
                num_iterations=num_slot_iterations,
                competitive=True,
            )
            for _ in layers
        )
        self.layer_logits = nn.Parameter(torch.zeros(len(layers)))
        self.slot_fusion = nn.Sequential(
            nn.Linear(2 * backbone.embedding_size, backbone.embedding_size),
            nn.GELU(),
            nn.Linear(backbone.embedding_size, backbone.embedding_size),
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(2 * backbone.embedding_size, backbone.embedding_size),
            nn.GELU(),
            nn.Linear(backbone.embedding_size, backbone.embedding_size),
        )
        self.decoder = _SlotDecoder(backbone.embedding_size, backbone.mlp_hidden_size, max_classes)
        self.last_support_attention: torch.Tensor | None = None
        self.last_support_attention_for_loss: torch.Tensor | None = None
        self.last_slots: torch.Tensor | None = None
        self.last_layer_alignments: list[np.ndarray] = []
        #: Diagnostics for the non-finite matching cost that ended four runs.
        self.alignment_fallbacks = 0
        self.first_alignment_fallback: dict[str, object] | None = None

    @staticmethod
    def _encode_layers(
        backbone: NanoTabPFNModel,
        source: tuple[torch.Tensor, torch.Tensor],
        split: int,
        layer_indices: tuple[int, ...],
        num_mem_chunks: int = 1,
    ) -> dict[int, torch.Tensor]:
        x_src, y_src = source
        if y_src.ndim < x_src.ndim:
            y_src = y_src.unsqueeze(-1)
        x_encoded = backbone.feature_encoder(x_src, split)
        y_encoded = backbone.target_encoder(y_src, x_encoded.shape[1])
        states = torch.cat((x_encoded, y_encoded), dim=2)
        taps: dict[int, torch.Tensor] = {}
        for index, block in enumerate(backbone.transformer_blocks):
            states = block(states, train_test_split_index=split, num_mem_chunks=num_mem_chunks)
            if index in layer_indices:
                taps[index] = states
        return taps

    def _alignment(
        self,
        reference_slots: torch.Tensor,
        candidate_slots: torch.Tensor,
        reference_attention: torch.Tensor,
        candidate_attention: torch.Tensor,
    ) -> list[np.ndarray]:
        """Detached per-episode matching; indexing preserves gradients."""
        alignments: list[np.ndarray] = []
        for batch in range(reference_slots.shape[0]):
            ref = nn.functional.normalize(reference_slots[batch].detach(), dim=-1)
            candidate = nn.functional.normalize(candidate_slots[batch].detach(), dim=-1)
            slot_cost = 1.0 - ref @ candidate.T
            ref_mask = reference_attention[batch].detach().T
            candidate_mask = candidate_attention[batch].detach().T
            mask_cost = torch.cdist(ref_mask, candidate_mask, p=2) / max(1, ref_mask.shape[1]) ** 0.5
            cost = (slot_cost + 0.5 * mask_cost).cpu().numpy()
            if not np.isfinite(cost).all():
                # ``linear_sum_assignment`` rejects a non-finite matrix rather
                # than degrading, so one bad episode would end a run that is
                # otherwise training normally.  The identity is what the
                # matching returns for an already-aligned pair.
                self.alignment_fallbacks += 1
                if self.first_alignment_fallback is None:
                    self.first_alignment_fallback = {
                        "cost_nan": int(np.isnan(cost).sum()),
                        "cost_inf": int(np.isinf(cost).sum()),
                        "reference_slots_finite": bool(torch.isfinite(reference_slots).all()),
                        "candidate_slots_finite": bool(torch.isfinite(candidate_slots).all()),
                        "reference_attention_finite": bool(torch.isfinite(reference_attention).all()),
                        "candidate_attention_finite": bool(torch.isfinite(candidate_attention).all()),
                        "candidate_slot_absmax": float(candidate_slots.detach().abs().max()),
                    }
                alignments.append(np.arange(reference_slots.shape[1], dtype=np.int64))
                continue
            rows, columns = linear_sum_assignment(cost)
            permutation = np.empty(reference_slots.shape[1], dtype=np.int64)
            permutation[rows] = columns
            alignments.append(permutation)
        return alignments

    @staticmethod
    def _reorder(values: torch.Tensor, alignments: list[np.ndarray]) -> torch.Tensor:
        reordered = []
        for batch, permutation in enumerate(alignments):
            index = torch.as_tensor(permutation, device=values.device)
            # The slot axis is the leading axis after removing the batch
            # dimension for both ``(B,S,E)`` slots and ``(B,S,N)`` masks.
            reordered.append(values[batch].index_select(0, index))
        return torch.stack(reordered, dim=0)

    def forward(self, *args: Any, **kwargs: Any) -> SlotRegimePrediction:
        if len(args) == 3:
            support_x, support_y, query_x = args
            source = (
                torch.cat((support_x, query_x), dim=1) if query_x is not None else support_x,
                support_y,
            )
            split = support_x.shape[1]
        elif len(args) == 1 and isinstance(args[0], tuple):
            source = args[0]
            split = kwargs.pop("train_test_split_index", None)
            if split is None:
                raise TypeError("train_test_split_index is required for the table interface.")
        else:
            raise TypeError("Expected (support_x, support_y, query_x) or ((x, y), train_test_split_index=...).")
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        split = int(split)
        taps = self._encode_layers(self.backbone, source, split, self.layer_indices, num_mem_chunks)
        layer_slots: list[torch.Tensor] = []
        layer_attention: list[torch.Tensor] = []
        layer_queries: list[torch.Tensor] = []
        for index, slot_module in zip(self.layer_indices, self.slot_modules, strict=True):
            states = taps[index]
            support = states[:, :split, -1, :]
            slots, attention = slot_module(support)
            layer_slots.append(slots)
            layer_attention.append(attention)
            layer_queries.append(states[:, split:, -1, :])

        reference_index = len(layer_slots) - 1
        reference_slots = layer_slots[reference_index]
        reference_attention = layer_attention[reference_index]
        aligned_slots: list[torch.Tensor] = []
        aligned_attention: list[torch.Tensor] = []
        alignments: list[np.ndarray] = []
        for slots, attention in zip(layer_slots, layer_attention, strict=True):
            permutation = self._alignment(reference_slots, slots, reference_attention, attention)
            # Keep the first batch's permutation for each tapped layer as a
            # compact diagnostic; matching itself is per episode and detached.
            if permutation:
                alignments.append(permutation[0])
            aligned_slots.append(self._reorder(slots, permutation))
            aligned_attention.append(self._reorder(attention.transpose(1, 2), permutation).transpose(1, 2))
        # Each layer has its own alignment, while fused values retain the
        # autograd path.  ``last_layer_alignments`` is diagnostic-only.
        self.last_layer_alignments = alignments
        weights = self.layer_logits.softmax(0)
        stacked_slots = torch.stack(aligned_slots, dim=0)
        stacked_attention = torch.stack(aligned_attention, dim=0)
        stacked_queries = torch.stack(layer_queries, dim=0)
        mean_slots = (weights[:, None, None, None] * stacked_slots).sum(0)
        max_slots = stacked_slots.max(0).values
        fused_slots = self.slot_fusion(torch.cat((mean_slots, max_slots), dim=-1))
        mean_queries = (weights[:, None, None, None] * stacked_queries).sum(0)
        max_queries = stacked_queries.max(0).values
        fused_queries = self.query_fusion(torch.cat((mean_queries, max_queries), dim=-1))
        fused_attention = (weights[:, None, None, None] * stacked_attention).sum(0)
        slot_logits, mask_logits = self.decoder(fused_queries, fused_slots)
        self.last_support_attention = fused_attention.detach()
        self.last_support_attention_for_loss = fused_attention
        self.last_slots = fused_slots
        return SlotRegimePrediction(
            slot_logits=slot_logits,
            log_gate=nn.functional.log_softmax(mask_logits, dim=-1),
            support_attention=fused_attention,
        )


def build_mufasa_model(architecture: dict[str, Any]) -> MufasaSlotTabPFNModel:
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["num_outputs"],
    )
    return MufasaSlotTabPFNModel(
        backbone,
        num_slots=architecture["num_slots"],
        layer_indices=tuple(architecture.get("slot_layer_indices", (2, 3, 4, 5))),
        num_slot_iterations=architecture.get("num_slot_iterations", 3),
        max_classes=architecture.get("max_classes", 2),
    )


__all__ = ["MufasaSlotTabPFNModel", "build_mufasa_model"]
