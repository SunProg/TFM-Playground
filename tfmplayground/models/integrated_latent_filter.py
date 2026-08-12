"""Latent tokens integrated between frozen or trainable nanoTabPFN blocks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.sequential_latent_filter import (
    SequentialFilterLogits,
    SequentialFilterPrediction,
    filter_sequential_logits,
)


@dataclass(frozen=True)
class IntegratedFilterLogits(SequentialFilterLogits):
    latent_states: torch.Tensor
    adapter_gates: torch.Tensor
    stream_residuals: torch.Tensor
    query_residuals: torch.Tensor
    stream_disagreement_gates: torch.Tensor
    query_disagreement_gates: torch.Tensor
    final_row_states: torch.Tensor


@dataclass(frozen=True)
class IntegratedFilterPrediction(SequentialFilterPrediction):
    latent_states: torch.Tensor
    adapter_gates: torch.Tensor
    stream_residuals: torch.Tensor
    query_residuals: torch.Tensor
    stream_disagreement_gates: torch.Tensor
    query_disagreement_gates: torch.Tensor
    final_row_states: torch.Tensor


class IntegratedLatentAdapter(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int):
        super().__init__()
        self.latent_attention = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)
        self.latent_norm = nn.LayerNorm(embedding_size)
        self.row_attention = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)
        self.row_norm = nn.LayerNorm(embedding_size)
        self.row_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        latents: torch.Tensor,
        support_states: torch.Tensor,
        target_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_delta = self.latent_attention(latents, support_states, support_states, need_weights=False)[0]
        latents = self.latent_norm(latents + latent_delta)
        row_delta = self.row_attention(target_states, latents, latents, need_weights=False)[0]
        gate = self.row_gate.tanh()
        return latents, target_states + gate * self.row_norm(row_delta)


class CenteredResidualDecoder(nn.Module):
    def __init__(self, embedding_size: int, residual_logit_bound: float | None = None):
        super().__init__()
        self.residual_logit_bound = residual_logit_bound
        self.base_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), nn.GELU(), nn.Linear(embedding_size, 1)
        )
        self.compatibility_head = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size), nn.GELU(), nn.Linear(embedding_size, 1)
        )
        self.disagreement_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), nn.GELU(), nn.Linear(embedding_size, 1)
        )

    def forward(
        self, row_states: torch.Tensor, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_count = row_states.shape[1]
        rows = row_states[:, :, None].expand(-1, -1, latents.shape[1], -1)
        slots = latents[:, None].expand(-1, row_count, -1, -1)
        compatibility = self.compatibility_head(torch.cat((rows, slots, rows * slots), dim=-1)).squeeze(-1)
        centered = compatibility - compatibility.mean(dim=2, keepdim=True)
        disagreement_gate = self.disagreement_head(row_states).sigmoid()
        residuals = disagreement_gate * centered
        if self.residual_logit_bound is not None:
            residuals = self.residual_logit_bound * torch.tanh(residuals / self.residual_logit_bound)
        binary_log_odds = self.base_head(row_states).squeeze(-1)[:, :, None] + residuals
        logits = torch.stack((-0.5 * binary_log_odds, 0.5 * binary_log_odds), dim=-1)
        return logits, residuals, disagreement_gate


class NanoTabPFNIntegratedLatentFilter(nn.Module):
    model_type = "nanotabpfn_integrated_latent_filter"

    def __init__(self, backbone: NanoTabPFNModel, num_hypotheses: int = 2, num_outputs: int = 2):
        super().__init__()
        if num_hypotheses < 1 or num_outputs != 2:
            raise ValueError("The integrated filter requires at least one binary hypothesis.")
        self.backbone = backbone
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
        self.evidence_logit_scale = 1.0
        self.evidence_disagreement_threshold: float | None = None
        self.evidence_disagreement_js_threshold: float | None = None
        self.query_temperature = 1.0
        self.transition_probability = 0.0
        self.trainability_stage = "frozen"
        embedding_size = backbone.embedding_size
        self.initial_latents = nn.Parameter(torch.empty(num_hypotheses, embedding_size))
        nn.init.normal_(self.initial_latents, std=embedding_size**-0.5)
        self.adapters = nn.ModuleList(
            [IntegratedLatentAdapter(embedding_size, backbone.num_attention_heads) for _ in range(backbone.num_layers)]
        )
        self.decoder = CenteredResidualDecoder(embedding_size)
        self.set_trainability("frozen")

    def set_query_temperature(self, value: float) -> None:
        if not 0 < value <= 1:
            raise ValueError("query temperature must be in (0, 1].")
        self.query_temperature = float(value)

    def set_evidence_logit_scale(self, value: float) -> None:
        if value <= 0:
            raise ValueError("evidence logit scale must be positive.")
        self.evidence_logit_scale = float(value)

    def set_transition_probability(self, value: float) -> None:
        if not 0 <= value < 1:
            raise ValueError("transition_probability must be in [0, 1).")
        self.transition_probability = float(value)

    def set_residual_logit_bound(self, value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("residual_logit_bound must be positive or None.")
        self.decoder.residual_logit_bound = None if value is None else float(value)

    def set_evidence_disagreement_threshold(self, value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("evidence disagreement threshold must be positive.")
        self.evidence_disagreement_threshold = None if value is None else float(value)
        if value is not None:
            self.evidence_disagreement_js_threshold = None

    def set_evidence_disagreement_js_threshold(self, value: float | None) -> None:
        if value is not None and not 0 < value <= 1:
            raise ValueError("evidence disagreement JS threshold must be in (0, 1].")
        self.evidence_disagreement_js_threshold = None if value is None else float(value)
        if value is not None:
            self.evidence_disagreement_threshold = None

    def set_trainability(self, stage: str) -> None:
        if stage not in {"frozen", "partial", "full"}:
            raise ValueError("stage must be frozen, partial, or full.")
        self.requires_grad_(True)
        self.backbone.requires_grad_(False)
        if stage == "partial":
            for block in self.backbone.transformer_blocks[-2:]:
                block.requires_grad_(True)
        elif stage == "full":
            self.backbone.requires_grad_(True)
        self.trainability_stage = stage
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def raw_logits(
        self,
        initial_support_x: torch.Tensor,
        initial_support_y: torch.Tensor,
        stream_x: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int = 1,
    ) -> IntegratedFilterLogits:
        if initial_support_x.ndim != 3 or stream_x.ndim != 3 or query_x.ndim != 3:
            raise ValueError("Support, stream, and query features must be three-dimensional.")
        support_count = initial_support_x.shape[1]
        stream_count = stream_x.shape[1]
        all_x = torch.cat((initial_support_x, stream_x, query_x), dim=1)
        y = initial_support_y
        if y.ndim < all_x.ndim:
            y = y.unsqueeze(-1)
        feature_states = self.backbone.feature_encoder(all_x, support_count)
        target_states = self.backbone.target_encoder(y, all_x.shape[1])
        states = torch.cat((feature_states, target_states), dim=2)
        latents = self.initial_latents[None].expand(all_x.shape[0], -1, -1)
        latent_history = []
        for block, adapter in zip(self.backbone.transformer_blocks, self.adapters, strict=True):
            states = block(states, train_test_split_index=support_count, num_mem_chunks=num_mem_chunks)
            target_column = states[:, :, -1, :]
            latents, updated_targets = adapter(
                latents,
                target_column[:, :support_count],
                target_column[:, support_count:],
            )
            states = states.clone()
            states[:, support_count:, -1, :] = updated_targets
            latent_history.append(latents)

        final_targets = states[:, support_count:, -1, :]
        stream_states = final_targets[:, :stream_count]
        query_states = final_targets[:, stream_count:]
        stream_logits, stream_residuals, stream_gates = self.decoder(stream_states, latents)
        query_logits, query_residuals, query_gates = self.decoder(query_states, latents)
        return IntegratedFilterLogits(
            stream_logits=stream_logits,
            query_logits=query_logits,
            slots=latents,
            latent_states=torch.stack(latent_history, dim=1),
            adapter_gates=torch.stack([adapter.row_gate.tanh() for adapter in self.adapters]),
            stream_residuals=stream_residuals,
            query_residuals=query_residuals,
            stream_disagreement_gates=stream_gates,
            query_disagreement_gates=query_gates,
            final_row_states=final_targets,
        )

    def forward(
        self,
        initial_support_x: torch.Tensor,
        initial_support_y: torch.Tensor,
        stream_x: torch.Tensor,
        stream_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int = 1,
    ) -> IntegratedFilterPrediction:
        raw = self.raw_logits(
            initial_support_x,
            initial_support_y,
            stream_x,
            query_x,
            num_mem_chunks=num_mem_chunks,
        )
        prediction = filter_sequential_logits(
            SequentialFilterLogits(raw.stream_logits, raw.query_logits, raw.slots),
            stream_y,
            evidence_logit_scale=self.evidence_logit_scale,
            evidence_disagreement_threshold=self.evidence_disagreement_threshold,
            evidence_disagreement_js_threshold=self.evidence_disagreement_js_threshold,
            query_temperature=self.query_temperature,
            transition_probability=self.transition_probability,
        )
        return IntegratedFilterPrediction(
            **prediction.__dict__,
            latent_states=raw.latent_states,
            adapter_gates=raw.adapter_gates,
            stream_residuals=raw.stream_residuals,
            query_residuals=raw.query_residuals,
            stream_disagreement_gates=raw.stream_disagreement_gates,
            query_disagreement_gates=raw.query_disagreement_gates,
            final_row_states=raw.final_row_states,
        )


def backbone_sha256(model: NanoTabPFNIntegratedLatentFilter) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def integrated_checkpoint(
    model: NanoTabPFNIntegratedLatentFilter,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    lineage: dict[str, Any],
    controlled_gate: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backbone = model.backbone
    checkpoint = {
        "model_type": model.model_type,
        "architecture": {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "num_hypotheses": model.num_hypotheses,
            "num_outputs": model.num_outputs,
            "evidence_logit_scale": model.evidence_logit_scale,
            "evidence_disagreement_threshold": model.evidence_disagreement_threshold,
            "evidence_disagreement_js_threshold": (model.evidence_disagreement_js_threshold),
            "query_temperature": model.query_temperature,
            "transition_probability": model.transition_probability,
            "residual_logit_bound": model.decoder.residual_logit_bound,
        },
        "model": model.state_dict(),
        "training_config": training_config,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "stage": stage,
        "trainability_stage": model.trainability_stage,
        "lineage": lineage,
        "backbone_sha256": backbone_sha256(model),
        "controlled_gate": controlled_gate,
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    return checkpoint


def save_integrated_checkpoint(path: str | Path, model: NanoTabPFNIntegratedLatentFilter, **kwargs) -> None:
    torch.save(integrated_checkpoint(model, **kwargs), Path(path))


def load_integrated_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNIntegratedLatentFilter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("model_type") != NanoTabPFNIntegratedLatentFilter.model_type:
        raise ValueError("Checkpoint is not an integrated latent filter.")
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    model = NanoTabPFNIntegratedLatentFilter(
        backbone,
        num_hypotheses=architecture["num_hypotheses"],
        num_outputs=architecture["num_outputs"],
    )
    model.load_state_dict(checkpoint["model"])
    model.set_evidence_logit_scale(architecture.get("evidence_logit_scale", 1.0))
    model.set_evidence_disagreement_threshold(architecture.get("evidence_disagreement_threshold"))
    model.set_evidence_disagreement_js_threshold(architecture.get("evidence_disagreement_js_threshold"))
    model.set_query_temperature(architecture.get("query_temperature", 1.0))
    model.set_transition_probability(architecture.get("transition_probability", 0.0))
    model.set_residual_logit_bound(architecture.get("residual_logit_bound"))
    model.set_trainability(checkpoint.get("trainability_stage", "frozen"))
    return model, checkpoint
