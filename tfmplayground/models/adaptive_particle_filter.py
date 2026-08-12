"""Adaptive K-particle filter with an exact frozen nanoTabPFN fallback."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.integrated_latent_filter import (
    IntegratedFilterPrediction,
    NanoTabPFNIntegratedLatentFilter,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.sequential_latent_filter import _binary_vectors


@dataclass(frozen=True)
class AdaptiveParticlePrediction:
    particle: IntegratedFilterPrediction
    vanilla_stream_logits: torch.Tensor
    vanilla_query_logits: torch.Tensor
    ambiguity_probability: torch.Tensor

    @property
    def stream_logits(self) -> torch.Tensor:
        return self.particle.stream_logits

    @property
    def query_logits(self) -> torch.Tensor:
        return self.particle.query_logits

    @property
    def log_weights(self) -> torch.Tensor:
        return self.particle.log_weights

    @property
    def slots(self) -> torch.Tensor:
        return self.particle.slots

    @property
    def stream_residuals(self) -> torch.Tensor:
        return self.particle.stream_residuals

    @property
    def query_residuals(self) -> torch.Tensor:
        return self.particle.query_residuals

    @property
    def prequential_log_likelihood(self) -> torch.Tensor:
        raise AttributeError("Use prequential_log_likelihood_for(stream_y).")

    def prequential_log_likelihood_for(self, stream_y: torch.Tensor) -> torch.Tensor:
        base_log_prob = F.log_softmax(self.vanilla_stream_logits, dim=-1)
        base_observed = base_log_prob.gather(-1, stream_y.long().unsqueeze(-1)).squeeze(-1)
        alpha = self.ambiguity_probability[:, None]
        # Guard both tails symmetrically: sigmoid saturates to exactly 1.0 in float32 for
        # logits above ~17, and log1p(-1.0) is -inf, which turns into NaN in the backward
        # pass and cannot be recovered by gradient clipping.
        return torch.logaddexp(
            (1 - alpha).clamp_min(1e-12).log() + base_observed,
            alpha.clamp_min(1e-12).log() + self.particle.prequential_log_likelihood,
        )

    def slot_joint_log_probabilities(self) -> torch.Tensor:
        return self.particle.slot_joint_log_probabilities()

    def particle_marginal_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        return self.particle.marginal_probabilities(states)

    def vanilla_marginal_probabilities(self) -> torch.Tensor:
        return self.vanilla_query_logits.softmax(-1)

    def marginal_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        particle = self.particle.marginal_probabilities(states)
        base = self.vanilla_marginal_probabilities()[:, None]
        alpha = self.ambiguity_probability[:, None, None, None]
        return (1 - alpha) * base + alpha * particle

    def matched_marginal_probabilities(
        self, matched_particles: torch.Tensor, states: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Suppress particles not assigned to an episode's supervised regimes."""

        if matched_particles.ndim != 2 or matched_particles.shape[0] != self.log_weights.shape[0]:
            raise ValueError("matched_particles must have shape (batch, supervised regimes).")
        weights = self.log_weights if states is None else self.log_weights.index_select(1, states)
        mask = torch.zeros_like(weights, dtype=torch.bool)
        indices = matched_particles[:, None].expand(-1, weights.shape[1], -1)
        mask.scatter_(2, indices, True)
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = (weights - torch.logsumexp(weights, dim=-1, keepdim=True)).exp()
        particle = torch.einsum(
            "bsk,bqkc->bsqc",
            weights,
            self.query_logits.softmax(-1),
        )
        base = self.vanilla_marginal_probabilities()[:, None]
        alpha = self.ambiguity_probability[:, None, None, None]
        return (1 - alpha) * base + alpha * particle

    def vanilla_joint_probabilities(self) -> torch.Tensor:
        probabilities = self.vanilla_marginal_probabilities()
        outcomes = _binary_vectors(probabilities.shape[1], probabilities.device)
        expanded = probabilities[:, None].expand(-1, outcomes.shape[0], -1, -1)
        indices = outcomes[None, :, :, None].expand(probabilities.shape[0], -1, -1, 1)
        return expanded.gather(-1, indices).squeeze(-1).prod(dim=-1)

    def joint_probabilities(self, states: torch.Tensor | None = None) -> torch.Tensor:
        particle = self.particle.joint_probabilities(states)
        base = self.vanilla_joint_probabilities()[:, None]
        alpha = self.ambiguity_probability[:, None, None]
        return (1 - alpha) * base + alpha * particle

    def effective_particle_count(self, states: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.log_weights if states is None else self.log_weights.index_select(1, states)
        return 1.0 / weights.exp().square().sum(-1)


class AdaptiveKParticleFilter(nn.Module):
    """Integrated particles activated only when a support-derived gate requests them."""

    model_type = "nanotabpfn_adaptive_k_particle_filter"

    def __init__(
        self,
        particle_model: NanoTabPFNIntegratedLatentFilter,
        vanilla_backbone: NanoTabPFNModel,
        *,
        initial_ambiguity_probability: float = 0.01,
    ):
        super().__init__()
        if not 0 < initial_ambiguity_probability < 1:
            raise ValueError("initial ambiguity probability must be in (0, 1).")
        self.particle_model = particle_model
        self.vanilla_backbone = vanilla_backbone
        self.vanilla_backbone.requires_grad_(False).eval()
        embedding_size = particle_model.backbone.embedding_size
        self.ambiguity_gate = nn.Sequential(
            nn.LayerNorm(embedding_size * 2),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1),
        )
        nn.init.zeros_(self.ambiguity_gate[-1].weight)
        nn.init.constant_(
            self.ambiguity_gate[-1].bias,
            math.log(initial_ambiguity_probability / (1 - initial_ambiguity_probability)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vanilla_backbone.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def _vanilla_logits(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        target_x: torch.Tensor,
        *,
        num_mem_chunks: int,
    ) -> torch.Tensor:
        return self.vanilla_backbone(
            (torch.cat((support_x, target_x), dim=1), support_y),
            train_test_split_index=support_x.shape[1],
            num_mem_chunks=num_mem_chunks,
        )[..., :2].detach()

    def forward(
        self,
        initial_support_x: torch.Tensor,
        initial_support_y: torch.Tensor,
        stream_x: torch.Tensor,
        stream_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int = 1,
        ambiguity_override: float | torch.Tensor | None = None,
    ) -> AdaptiveParticlePrediction:
        particle = self.particle_model(
            initial_support_x,
            initial_support_y,
            stream_x,
            stream_y,
            query_x,
            num_mem_chunks=num_mem_chunks,
        )
        pooled = torch.cat((particle.slots.mean(dim=1), particle.slots.std(dim=1, unbiased=False)), dim=-1)
        alpha = self.ambiguity_gate(pooled).squeeze(-1).sigmoid()
        if ambiguity_override is not None:
            alpha = torch.as_tensor(ambiguity_override, device=alpha.device, dtype=alpha.dtype).expand_as(alpha)
        targets = torch.cat((stream_x, query_x), dim=1)
        vanilla = self._vanilla_logits(
            initial_support_x,
            initial_support_y,
            targets,
            num_mem_chunks=num_mem_chunks,
        )
        return AdaptiveParticlePrediction(
            particle=particle,
            vanilla_stream_logits=vanilla[:, : stream_x.shape[1]],
            vanilla_query_logits=vanilla[:, stream_x.shape[1] :],
            ambiguity_probability=alpha,
        )


def expand_two_to_k_particles(
    source: NanoTabPFNIntegratedLatentFilter,
    vanilla_backbone: NanoTabPFNModel,
    *,
    particle_count: int = 4,
) -> AdaptiveKParticleFilter:
    if source.num_hypotheses != 2 or particle_count < 2:
        raise ValueError("Expansion requires a two-particle source and K >= 2.")
    expanded = NanoTabPFNIntegratedLatentFilter(copy.deepcopy(source.backbone), num_hypotheses=particle_count)
    transferable = {name: value for name, value in source.state_dict().items() if name != "initial_latents"}
    expanded.load_state_dict(transferable, strict=False)
    with torch.no_grad():
        for index in range(particle_count):
            expanded.initial_latents[index].copy_(source.initial_latents[index % 2])
    expanded.set_query_temperature(source.query_temperature)
    expanded.set_evidence_logit_scale(source.evidence_logit_scale)
    expanded.set_evidence_disagreement_threshold(source.evidence_disagreement_threshold)
    expanded.set_evidence_disagreement_js_threshold(source.evidence_disagreement_js_threshold)
    expanded.set_transition_probability(source.transition_probability)
    expanded.set_residual_logit_bound(source.decoder.residual_logit_bound)
    expanded.set_trainability(source.trainability_stage)
    return AdaptiveKParticleFilter(expanded, copy.deepcopy(vanilla_backbone))


def save_adaptive_checkpoint(
    path: str | Path,
    model: AdaptiveKParticleFilter,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    controlled_gate: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    backbone = model.particle_model.backbone
    payload = {
        "model_type": model.model_type,
        "architecture": {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "particle_count": model.particle_model.num_hypotheses,
            "evidence_logit_scale": model.particle_model.evidence_logit_scale,
            "evidence_disagreement_threshold": (model.particle_model.evidence_disagreement_threshold),
            "evidence_disagreement_js_threshold": (model.particle_model.evidence_disagreement_js_threshold),
            "query_temperature": model.particle_model.query_temperature,
            "transition_probability": model.particle_model.transition_probability,
            "residual_logit_bound": model.particle_model.decoder.residual_logit_bound,
        },
        "model": model.state_dict(),
        "training_config": training_config,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "controlled_gate": controlled_gate,
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, Path(path))


def load_adaptive_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[AdaptiveKParticleFilter, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("model_type") != AdaptiveKParticleFilter.model_type:
        raise ValueError("Checkpoint is not an adaptive K-particle filter.")
    architecture = payload["architecture"]

    def backbone() -> NanoTabPFNModel:
        return NanoTabPFNModel(
            num_layers=architecture["num_layers"],
            embedding_size=architecture["embedding_size"],
            num_attention_heads=architecture["num_attention_heads"],
            mlp_hidden_size=architecture["mlp_hidden_size"],
            num_outputs=architecture["backbone_num_outputs"],
        )

    particle = NanoTabPFNIntegratedLatentFilter(backbone(), num_hypotheses=architecture["particle_count"])
    particle.set_evidence_logit_scale(architecture.get("evidence_logit_scale", 1.0))
    particle.set_evidence_disagreement_threshold(architecture.get("evidence_disagreement_threshold"))
    particle.set_evidence_disagreement_js_threshold(architecture.get("evidence_disagreement_js_threshold"))
    particle.set_query_temperature(architecture.get("query_temperature", 1.0))
    particle.set_transition_probability(architecture.get("transition_probability", 0.0))
    particle.set_residual_logit_bound(architecture.get("residual_logit_bound"))
    model = AdaptiveKParticleFilter(particle, backbone())
    model.load_state_dict(payload["model"])
    return model, payload


def checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
