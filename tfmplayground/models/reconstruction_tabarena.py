"""Inference adapters for reconstruction-routing checkpoints on TabArena."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.reconstruction_routing import ReconstructionRouter
from tfmplayground.models.table_slot import TableSlotModel


class ReconstructionTabArenaAdapter(nn.Module):
    """Expose reconstruction models through nanoTabPFN's concatenated-table API."""

    def __init__(self, model: nn.Module, *, plain: bool = False):
        super().__init__()
        self.model = model
        self.plain = plain

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
        if self.plain:
            encoded = self.model.encode_table((table, support_y.float()), split)
            return self.model.decoder(encoded[:, split:, -1])
        output = self.model(support_x, support_y, query_x)
        prediction = output[0] if isinstance(output, tuple) else output
        return prediction.marginal_log_probabilities()


def build_reconstruction_inference(architecture: dict[str, Any]) -> ReconstructionTabArenaAdapter:
    """Build the inference adapter described by a wrapped checkpoint."""
    width = architecture["width"]
    hidden = architecture["hidden"]
    layers = architecture["layers"]
    heads = architecture["heads"]
    num_slots = architecture.get("num_slots", 4)
    arm = architecture.get("arm", "vanilla")
    backbone = NanoTabPFNModel(
        embedding_size=width,
        num_attention_heads=heads,
        mlp_hidden_size=hidden,
        num_layers=layers,
        num_outputs=2,
    )
    if arm == "vanilla":
        return ReconstructionTabArenaAdapter(backbone, plain=True)
    if arm == "values":
        model = ReconstructionRouter(
            scope=architecture["scope"],
            variant="values",
            width=width,
            hidden=hidden,
            layers=layers,
            heads=heads,
            num_slots=num_slots,
        )
        return ReconstructionTabArenaAdapter(model)
    if arm not in ("label_alpha", "embedding_mse"):
        raise ValueError(f"Unsupported reconstruction arm: {arm!r}")
    model = TableSlotModel(
        backbone,
        mode="head",
        scope=architecture["scope"],
        num_slots=num_slots,
        query_routing_mode="decoder",
        reconstruction_mixture="alpha",
        embedding_reconstruction=arm == "embedding_mse",
    )
    return ReconstructionTabArenaAdapter(model)
