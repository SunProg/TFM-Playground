"""Minimal sequential two-hypothesis filter on frozen nanoTabPFN embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _binary_vectors(query_count: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(2**query_count, device=device)
    shifts = torch.arange(query_count - 1, -1, -1, device=device)
    return ((indices[:, None] >> shifts[None, :]) & 1).long()


@dataclass(frozen=True)
class SequentialFilterLogits:
    """Cached uncalibrated outputs of the frozen backbone and latent head."""

    stream_logits: torch.Tensor
    query_logits: torch.Tensor
    slots: torch.Tensor


@dataclass(frozen=True)
class SequentialFilterPrediction:
    """Complete trajectory produced by a sequential latent filter.

    ``log_weights[:, t]`` is the normalized filter state after exactly ``t``
    stream labels. Consequently, ``prequential_log_likelihood[:, t]`` was
    calculated from ``log_weights[:, t]`` before label ``t`` was incorporated.
    """

    stream_logits: torch.Tensor
    query_logits: torch.Tensor
    log_weights: torch.Tensor
    prequential_log_likelihood: torch.Tensor
    row_log_likelihood: torch.Tensor
    slots: torch.Tensor

    def slot_joint_log_probabilities(self) -> torch.Tensor:
        """Factorized query-vector log probabilities for each fixed slot."""
        batch_size, query_count, slot_count, class_count = self.query_logits.shape
        if class_count != 2:
            raise ValueError("Canonical joint enumeration requires binary logits.")
        outcomes = _binary_vectors(query_count, self.query_logits.device)
        log_probabilities = F.log_softmax(self.query_logits, dim=-1)
        expanded = log_probabilities[:, None].expand(-1, outcomes.shape[0], -1, -1, -1)
        gather_index = outcomes[None, :, :, None, None].expand(batch_size, -1, -1, slot_count, 1)
        # (batch, outcomes, queries, slots) -> (batch, slots, outcomes)
        return expanded.gather(-1, gather_index).squeeze(-1).sum(dim=2).transpose(1, 2)

    def joint_log_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        """Canonical query-vector mixture log probabilities at filter states."""
        weights = self.log_weights if states is None else self.log_weights.index_select(1, states)
        slot_joint = self.slot_joint_log_probabilities()
        return torch.logsumexp(weights[..., None] + slot_joint[:, None], dim=2)

    def joint_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        return self.joint_log_probabilities(states).exp()

    def marginal_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        """Per-query class probabilities at every requested filter state."""
        weights = self.log_weights if states is None else self.log_weights.index_select(1, states)
        slot_probabilities = F.softmax(self.query_logits, dim=-1)
        return torch.einsum("bsk,bqkc->bsqc", weights.exp(), slot_probabilities)


class NanoTabPFNSequentialLatentFilter(nn.Module):
    """Two fixed latent states whose probabilities receive online Bayes updates."""

    model_type = "nanotabpfn_sequential_latent_filter"

    def __init__(self, backbone: NanoTabPFNModel, num_hypotheses: int = 2, num_outputs: int = 2):
        super().__init__()
        if num_hypotheses != 2 or num_outputs != 2:
            raise ValueError("The minimal sequential filter requires two binary hypotheses.")
        self.backbone = backbone
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
        self.evidence_logit_scale = 1.0
        self.query_temperature = 1.0
        self.transition_probability = 0.0
        embedding_size = backbone.embedding_size

        self.hypothesis_queries = nn.Parameter(torch.empty(num_hypotheses, embedding_size))
        nn.init.normal_(self.hypothesis_queries, std=embedding_size**-0.5)
        self.slot_attention = nn.MultiheadAttention(
            embedding_size,
            backbone.num_attention_heads,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(embedding_size)
        self.slot_decoder = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, num_outputs),
        )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The official checkpoint is always an embedding service, never a
        # trainable or stochastic part of this experiment.
        self.backbone.eval()
        return self

    def head_parameters(self):
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("backbone."))

    def _decode(self, row_embeddings: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        row_count = row_embeddings.shape[1]
        rows = row_embeddings[:, :, None].expand(-1, -1, self.num_hypotheses, -1)
        expanded_slots = slots[:, None].expand(-1, row_count, -1, -1)
        return self.slot_decoder(torch.cat((rows, expanded_slots, rows * expanded_slots), dim=-1))

    def set_temperatures(self, evidence_logit_scale: float, query_temperature: float) -> None:
        if not 0 < evidence_logit_scale <= 1:
            raise ValueError("evidence_logit_scale must be in (0, 1].")
        if not 0 < query_temperature <= 1:
            raise ValueError("query_temperature must be in (0, 1].")
        self.evidence_logit_scale = float(evidence_logit_scale)
        self.query_temperature = float(query_temperature)

    def set_transition_probability(self, value: float) -> None:
        if not 0 <= value < 1:
            raise ValueError("transition_probability must be in [0, 1).")
        self.transition_probability = float(value)

    def raw_logits(
        self,
        initial_support_x: torch.Tensor,
        initial_support_y: torch.Tensor,
        stream_x: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int = 1,
    ) -> SequentialFilterLogits:
        """Encode and decode an episode once without applying calibration."""
        if initial_support_x.ndim != 3 or stream_x.ndim != 3 or query_x.ndim != 3:
            raise ValueError("Support, stream, and query features must have shape (batch, rows, features).")
        if initial_support_y.shape != initial_support_x.shape[:2]:
            raise ValueError("initial_support_y must have shape (batch, support rows).")

        support_count = initial_support_x.shape[1]
        stream_count = stream_x.shape[1]
        all_x = torch.cat((initial_support_x, stream_x, query_x), dim=1)
        with torch.no_grad():
            encoded = self.backbone.encode_table(
                (all_x, initial_support_y),
                support_count,
                num_mem_chunks=num_mem_chunks,
            )[:, :, -1, :].detach()

        support_embeddings = encoded[:, :support_count]
        stream_embeddings = encoded[:, support_count : support_count + stream_count]
        query_embeddings = encoded[:, support_count + stream_count :]
        seeds = self.hypothesis_queries[None].expand(encoded.shape[0], -1, -1)
        attended = self.slot_attention(seeds, support_embeddings, support_embeddings, need_weights=False)[0]
        slots = self.slot_norm(seeds + attended)
        return SequentialFilterLogits(
            stream_logits=self._decode(stream_embeddings, slots),
            query_logits=self._decode(query_embeddings, slots),
            slots=slots,
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
        evidence_logit_scale: float | torch.Tensor | None = None,
        query_temperature: float | torch.Tensor | None = None,
    ) -> SequentialFilterPrediction:
        if stream_y.shape != stream_x.shape[:2]:
            raise ValueError("stream_y must have shape (batch, stream rows).")
        raw = self.raw_logits(
            initial_support_x,
            initial_support_y,
            stream_x,
            query_x,
            num_mem_chunks=num_mem_chunks,
        )
        return filter_sequential_logits(
            raw,
            stream_y,
            evidence_logit_scale=(self.evidence_logit_scale if evidence_logit_scale is None else evidence_logit_scale),
            query_temperature=self.query_temperature if query_temperature is None else query_temperature,
            transition_probability=self.transition_probability,
        )


def filter_sequential_logits(
    raw: SequentialFilterLogits,
    stream_y: torch.Tensor,
    *,
    evidence_logit_scale: float | torch.Tensor = 1.0,
    evidence_disagreement_threshold: float | None = None,
    evidence_disagreement_js_threshold: float | None = None,
    query_temperature: float | torch.Tensor = 1.0,
    transition_probability: float = 0.0,
) -> SequentialFilterPrediction:
    """Apply temperatures and causal online updates to cached raw logits.

    ``transition_probability`` is the probability of leaving the current
    particle before the next row is observed.  A positive value prevents a
    posterior from becoming irreversibly concentrated and is therefore needed
    for regime-switching streams.  Zero retains the historical static-task
    behaviour exactly.
    """
    if stream_y.shape != raw.stream_logits.shape[:2]:
        raise ValueError("stream_y must match the cached stream rows.")
    stream_logits = raw.stream_logits * evidence_logit_scale
    query_logits = raw.query_logits / query_temperature
    stream_log_probabilities = F.log_softmax(stream_logits, dim=-1)
    num_hypotheses = stream_logits.shape[2]
    if not 0 <= transition_probability < 1:
        raise ValueError("transition_probability must be in [0, 1).")
    if transition_probability > 0 and num_hypotheses < 2:
        raise ValueError("A positive transition prior requires at least two hypotheses.")
    labels = stream_y.long()[:, :, None, None].expand(-1, -1, num_hypotheses, 1)
    row_log_likelihood = stream_log_probabilities.gather(-1, labels).squeeze(-1)
    if evidence_disagreement_threshold is not None and evidence_disagreement_js_threshold is not None:
        raise ValueError("Select either max-min or JS disagreement gating, not both.")
    update_log_likelihood = row_log_likelihood
    if evidence_disagreement_threshold is not None:
        if evidence_disagreement_threshold <= 0:
            raise ValueError("evidence disagreement threshold must be positive.")
        class_one = stream_log_probabilities[..., 1].exp()
        disagreement = class_one.max(dim=-1).values - class_one.min(dim=-1).values
        gate = (disagreement >= evidence_disagreement_threshold).to(row_log_likelihood.dtype)
        common = row_log_likelihood.mean(dim=-1, keepdim=True)
        update_log_likelihood = common + gate[..., None] * (row_log_likelihood - common)
    elif evidence_disagreement_js_threshold is not None:
        if not 0 < evidence_disagreement_js_threshold <= 1:
            raise ValueError("evidence disagreement JS threshold must be in (0, 1].")
        # Generalized Jensen-Shannon divergence of the particle Bernoulli
        # predictions, normalized to [0, 1]. Uniform particle weights make the
        # statistic invariant to exact particle duplication and therefore much
        # less sensitive to K than a maximum-minus-minimum range.
        probabilities = stream_log_probabilities.exp()
        mixture = probabilities.mean(dim=2)
        mixture_entropy = -(mixture * mixture.clamp_min(1e-12).log()).sum(dim=-1)
        particle_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=2)
        disagreement_js = (mixture_entropy - particle_entropy).clamp_min(0) / math.log(2)
        gate = (disagreement_js >= evidence_disagreement_js_threshold).to(row_log_likelihood.dtype)
        common = row_log_likelihood.mean(dim=-1, keepdim=True)
        update_log_likelihood = common + gate[..., None] * (row_log_likelihood - common)
    current = stream_logits.new_full((stream_logits.shape[0], num_hypotheses), -math.log(num_hypotheses))
    states = [current]
    prequential = []
    for index in range(stream_logits.shape[1]):
        if transition_probability > 0:
            stay = math.log1p(-transition_probability)
            switch = math.log(transition_probability / (num_hypotheses - 1))
            transition = current.new_full((num_hypotheses, num_hypotheses), switch)
            transition.diagonal().fill_(stay)
            current = torch.logsumexp(current[:, :, None] + transition[None], dim=1)
        predictive = row_log_likelihood[:, index]
        prequential.append(torch.logsumexp(current + predictive, dim=-1))
        current = current + update_log_likelihood[:, index]
        current = current - torch.logsumexp(current, dim=-1, keepdim=True)
        states.append(current)

    if prequential:
        prequential_tensor = torch.stack(prequential, dim=1)
    else:
        prequential_tensor = stream_logits.new_empty((stream_logits.shape[0], 0))
    return SequentialFilterPrediction(
        stream_logits=stream_logits,
        query_logits=query_logits,
        log_weights=torch.stack(states, dim=1),
        prequential_log_likelihood=prequential_tensor,
        row_log_likelihood=row_log_likelihood,
        slots=raw.slots,
    )


def sequential_filter_checkpoint(
    model: NanoTabPFNSequentialLatentFilter,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    controlled_gate: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backbone = model.backbone
    checkpoint: dict[str, Any] = {
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
            "query_temperature": model.query_temperature,
            "transition_probability": model.transition_probability,
        },
        "model": model.state_dict(),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "training_config": training_config,
        "stage": stage,
        "controlled_gate": controlled_gate,
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    return checkpoint


def save_sequential_filter_checkpoint(
    path: str | Path,
    model: NanoTabPFNSequentialLatentFilter,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    controlled_gate: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    torch.save(
        sequential_filter_checkpoint(
            model,
            training_config=training_config,
            source_checkpoint_sha256=source_checkpoint_sha256,
            stage=stage,
            controlled_gate=controlled_gate,
            optimizer_state=optimizer_state,
        ),
        Path(path),
    )


def load_sequential_filter_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNSequentialLatentFilter, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if checkpoint.get("model_type") != NanoTabPFNSequentialLatentFilter.model_type:
        raise ValueError("Checkpoint is not a nanoTabPFN sequential latent filter.")
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    model = NanoTabPFNSequentialLatentFilter(
        backbone,
        num_hypotheses=architecture["num_hypotheses"],
        num_outputs=architecture["num_outputs"],
    )
    model.load_state_dict(checkpoint["model"])
    model.set_temperatures(
        architecture.get("evidence_logit_scale", 1.0),
        architecture.get("query_temperature", 1.0),
    )
    model.set_transition_probability(architecture.get("transition_probability", 0.0))
    model.freeze_backbone()
    return model, checkpoint
