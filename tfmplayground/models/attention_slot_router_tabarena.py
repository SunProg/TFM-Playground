"""Inference adapter for AttentionSlotRouter checkpoints on TabArena."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tfmplayground.models.attention_slot_router import AttentionSlotRouter, AttentionSlotRouterFrozenTabPFN


class AttentionSlotRouterTabArenaAdapter(nn.Module):
    """Expose AttentionSlotRouter through nanoTabPFN's concatenated-table API."""

    def __init__(self, model: AttentionSlotRouter):
        super().__init__()
        self.model = model

    def forward(
        self,
        source: tuple[torch.Tensor, torch.Tensor],
        *,
        train_test_split_index: int,
        **_: Any,
    ) -> torch.Tensor:
        table, support_y = source
        split = int(train_test_split_index)
        support_x = table[:, :split]
        query_x = table[:, split:]
        prediction, _ = self.model(support_x, support_y, query_x)
        # prediction is already a normalized probability distribution (the
        # attention-retrieval mixture, not logits) -- log() of it is valid
        # logits under softmax, the same trick SlotLogitsAdapter and
        # ReconstructionTabArenaAdapter use, so every existing evaluator that
        # does model(...)[..., :2].softmax(-1) keeps working unchanged.
        return prediction.clamp_min(1e-12).log()


def build_attention_slot_router_inference(architecture: dict[str, Any]) -> AttentionSlotRouterTabArenaAdapter:
    """Build the inference adapter described by a wrapped checkpoint."""
    model = AttentionSlotRouter(
        scope=architecture["scope"],
        width=architecture["width"],
        hidden=architecture["hidden"],
        layers=architecture["layers"],
        heads=architecture["heads"],
        num_slots=architecture.get("num_slots", 4),
    )
    return AttentionSlotRouterTabArenaAdapter(model)


def build_attention_slot_router_frozen_tabpfn_inference(
    architecture: dict[str, Any],
) -> AttentionSlotRouterTabArenaAdapter:
    """Build the inference adapter for an AttentionSlotRouterFrozenTabPFN checkpoint.

    ``tabpfn_device`` is deliberately left at its "cpu" default:
    evaluate_tabarena_small.py's own call site never threads its ``--device``
    through to load_checkpoint_for_inference (every other model_kind gets
    re-homed per-call inside predict_vanilla instead, which only works
    because those backbones are real nn.Module submodules -- FrozenTabPFN-
    Backbone is a plain object, so a later ``adapter.to(device)`` would be a
    no-op for it regardless). Real TabPFN's own inference is CPU-capable and
    fast enough at TabArena's small-protocol sizes (subsample <= 2048,
    matching its own CPU sample limit) -- only the adapter head (row_query/
    slot_key/TableSlotAdapter, real nn.Parameters) benefits from a later
    ``.to(device)`` call.
    """
    model = AttentionSlotRouterFrozenTabPFN(
        scope=architecture["scope"],
        hidden=architecture["hidden"],
        num_slots=architecture.get("num_slots", 4),
        layer_index=architecture.get("tabpfn_layer_index", -1),
        tabpfn_model_path=architecture.get("tabpfn_model_path"),
    )
    return AttentionSlotRouterTabArenaAdapter(model)
