"""Latent-input nanoTabPFN with an auxiliary per-row regime head.

The prediction path is the ordinary nanoTabPFN path.  The regime head is only
used by the v2 training objective: it reads the final target-column state for
each support/query row and predicts the diagnostic regime assignment ``z``.
At evaluation time the module still accepts the normal three-argument
``(support_x, support_y, query_x)`` interface and returns ordinary class logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class SupervisedNanoTabPFNModel(nn.Module):
    """Standard nanoTabPFN plus a training-only regime classification head."""

    model_type = "supervised_tabpfn"

    def __init__(self, backbone: NanoTabPFNModel, max_regimes: int = 4):
        super().__init__()
        if max_regimes < 1:
            raise ValueError("max_regimes must be positive.")
        self.backbone = backbone
        self.max_regimes = max_regimes
        self.regime_head = nn.Linear(backbone.embedding_size, max_regimes)
        self.last_regime_logits: torch.Tensor | None = None

    def _split_arguments(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) == 3:
            support_x, support_y, query_x = args
            x = support_x if query_x is None else torch.cat((support_x, query_x), dim=1)
            split = support_x.shape[1]
            source = (x, support_y)
        elif len(args) == 1 and isinstance(args[0], tuple):
            source = args[0]
            split = kwargs.pop("train_test_split_index", None)
            if split is None:
                raise TypeError("train_test_split_index is required for the concatenated-table interface.")
        else:
            raise TypeError("Expected (support_x, support_y, query_x) or ((x, y), train_test_split_index=...).")
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return source, int(split), num_mem_chunks

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        source, split, num_mem_chunks = self._split_arguments(args, kwargs)
        x_source, y_source = source
        encoded = self.backbone.encode_table(
            (x_source, y_source.float()),
            train_test_split_index=split,
            num_mem_chunks=num_mem_chunks,
        )
        # The target-column state is row-level and includes support labels, but
        # query labels remain mean-padded by TargetEncoder.  Thus no query y or
        # z is exposed to the model input.
        row_states = encoded[:, :, -1, :]
        self.last_regime_logits = self.regime_head(row_states)
        return self.backbone.decoder(row_states[:, split:])


def supervised_regime_loss(model: SupervisedNanoTabPFNModel, z: torch.Tensor) -> torch.Tensor:
    """Cross-entropy auxiliary loss for support and query regime assignments."""
    if model.last_regime_logits is None:
        raise RuntimeError("The supervised regime head has not run a forward pass.")
    logits = model.last_regime_logits
    if logits.shape[:2] != z.shape:
        raise ValueError("z must have shape (batch, support + query rows).")
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), z.reshape(-1).long())


def build_supervised_tabpfn_model(architecture: dict[str, Any]) -> SupervisedNanoTabPFNModel:
    """Reconstruct the model from a v2 checkpoint architecture dictionary."""
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["num_outputs"],
    )
    return SupervisedNanoTabPFNModel(
        backbone,
        max_regimes=int(architecture.get("max_regimes", 4)),
    )


__all__ = [
    "SupervisedNanoTabPFNModel",
    "build_supervised_tabpfn_model",
    "supervised_regime_loss",
]
