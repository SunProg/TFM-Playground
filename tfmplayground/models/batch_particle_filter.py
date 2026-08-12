"""Batch-causal K-particle filtering with an exact cumulative vanilla anchor.

The causal boundary is represented by two methods instead of convention:
``predict_batch`` commits probabilities without labels and returns a pending
update; ``reveal_batch`` is the only operation which accepts those labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from tfmplayground.models.integrated_latent_filter import NanoTabPFNIntegratedLatentFilter
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


@dataclass(frozen=True)
class BatchParticleState:
    """All information revealed before the next batch."""

    log_weights: torch.Tensor
    previous_surprise: torch.Tensor
    context_x: torch.Tensor
    context_y: torch.Tensor
    batches_seen: int = 0


@dataclass(frozen=True)
class PendingBatchUpdate:
    """Immutable prediction committed before the corresponding labels exist."""

    prior_state: BatchParticleState
    unmasked_log_weights: torch.Tensor
    transitioned_log_weights: torch.Tensor
    x: torch.Tensor
    probabilities: torch.Tensor
    particle_log_probabilities: torch.Tensor
    particle_logits: torch.Tensor
    vanilla_logits: torch.Tensor
    ambiguity_probability: torch.Tensor
    slots: torch.Tensor
    residuals: torch.Tensor


class BatchCausalParticleFilter(nn.Module):
    """Frozen-backbone particle model for delayed-label batches."""

    model_type = "nanotabpfn_batch_causal_particle_filter"

    def __init__(
        self,
        particle_model: NanoTabPFNIntegratedLatentFilter,
        vanilla_backbone: NanoTabPFNModel,
        *,
        context_limit: int = 1024,
        transition_probability: float = 0.05,
        residual_logit_bound: float = 4.0,
        initial_ambiguity_probability: float = 0.01,
    ):
        super().__init__()
        if context_limit < 2:
            raise ValueError("context_limit must be at least two.")
        if not 0 <= transition_probability < 1:
            raise ValueError("transition_probability must be in [0, 1).")
        if not 0 < initial_ambiguity_probability < 1:
            raise ValueError("initial_ambiguity_probability must be in (0, 1).")
        self.particle_model = particle_model
        self.vanilla_backbone = vanilla_backbone
        self.context_limit = int(context_limit)
        self.transition_probability = float(transition_probability)
        self.particle_model.set_transition_probability(0.0)  # transition is applied here, once per batch
        self.particle_model.set_residual_logit_bound(residual_logit_bound)
        self.particle_model.set_trainability("frozen")
        self.vanilla_backbone.requires_grad_(False).eval()
        embedding = particle_model.backbone.embedding_size
        self.ambiguity_gate = nn.Sequential(
            nn.LayerNorm(2 * embedding + 4),
            nn.Linear(2 * embedding + 4, embedding),
            nn.GELU(),
            nn.Linear(embedding, 1),
        )
        nn.init.zeros_(self.ambiguity_gate[-1].weight)
        nn.init.constant_(
            self.ambiguity_gate[-1].bias,
            math.log(initial_ambiguity_probability / (1 - initial_ambiguity_probability)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.particle_model.backbone.eval()
        self.vanilla_backbone.eval()
        return self

    def initial_state(
        self,
        feature_count: int,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> BatchParticleState:
        if feature_count < 1 or batch_size < 1:
            raise ValueError("feature_count and batch_size must be positive.")
        device = device or self.particle_model.initial_latents.device
        count = self.particle_model.num_hypotheses
        return BatchParticleState(
            log_weights=torch.full((batch_size, count), -math.log(count), device=device, dtype=dtype),
            previous_surprise=torch.zeros(batch_size, device=device, dtype=dtype),
            context_x=torch.empty((batch_size, 0, feature_count), device=device, dtype=dtype),
            context_y=torch.empty((batch_size, 0), device=device, dtype=torch.long),
        )

    def _transition(self, log_weights: torch.Tensor) -> torch.Tensor:
        count = log_weights.shape[-1]
        probability = self.transition_probability
        if probability == 0 or count == 1:
            return log_weights
        transition = torch.full(
            (count, count), probability / (count - 1), device=log_weights.device, dtype=log_weights.dtype
        )
        transition.fill_diagonal_(1 - probability)
        return torch.logsumexp(log_weights[:, :, None] + transition.log()[None], dim=1)

    @staticmethod
    def _clean(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _raw_predictions(
        self, state: BatchParticleState, x: torch.Tensor, num_mem_chunks: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, rows = x.shape[:2]
        if state.context_x.shape[1] == 0:
            particle_logits = torch.zeros(
                batch, rows, self.particle_model.num_hypotheses, 2, device=x.device, dtype=x.dtype
            )
            vanilla_logits = torch.zeros(batch, rows, 2, device=x.device, dtype=x.dtype)
            slots = self.particle_model.initial_latents[None].expand(batch, -1, -1)
            residuals = torch.zeros(batch, rows, self.particle_model.num_hypotheses, device=x.device, dtype=x.dtype)
            return particle_logits, vanilla_logits, slots, residuals
        raw = self.particle_model.raw_logits(
            state.context_x,
            state.context_y.float(),
            state.context_x[:, :0],
            x,
            num_mem_chunks=num_mem_chunks,
        )
        with torch.no_grad():
            vanilla_logits = self.vanilla_backbone(
                (torch.cat((state.context_x, x), dim=1), state.context_y.float()),
                train_test_split_index=state.context_x.shape[1],
                num_mem_chunks=num_mem_chunks,
            )[..., :2]
        return raw.query_logits, vanilla_logits.detach(), raw.slots, raw.query_residuals

    def predict_batch(
        self,
        state: BatchParticleState,
        x: torch.Tensor,
        *,
        matched_particles: torch.Tensor | None = None,
        num_mem_chunks: int = 1,
    ) -> PendingBatchUpdate:
        """Transition once and commit every row using identical pre-reveal weights."""

        x = self._clean(x)
        if x.ndim != 3 or x.shape[0] != state.log_weights.shape[0] or x.shape[2] != state.context_x.shape[2]:
            raise ValueError("x must have shape (batch, rows, features) matching state.")
        unmasked = self._transition(state.log_weights)
        transitioned = unmasked
        if matched_particles is not None:
            if matched_particles.ndim != 2 or matched_particles.shape[0] != x.shape[0]:
                raise ValueError("matched_particles must have shape (batch, candidates).")
            mask = torch.zeros_like(transitioned, dtype=torch.bool)
            mask.scatter_(1, matched_particles.long(), True)
            transitioned = transitioned.masked_fill(~mask, float("-inf"))
            transitioned = transitioned - torch.logsumexp(transitioned, dim=-1, keepdim=True)
        particle_logits, vanilla_logits, slots, residuals = self._raw_predictions(state, x, num_mem_chunks)
        particle_log_probabilities = particle_logits.log_softmax(-1)
        particle_probability = torch.einsum("bk,brkc->brc", transitioned.exp(), particle_log_probabilities.exp())
        vanilla_probability = vanilla_logits.softmax(-1)
        gate_weights = unmasked.exp()
        entropy = -(gate_weights * unmasked.nan_to_num()).sum(-1) / math.log(max(2, gate_weights.shape[-1]))
        top = gate_weights.topk(min(2, gate_weights.shape[-1]), dim=-1).values
        margin = top[:, 0] - (top[:, 1] if top.shape[-1] > 1 else 0)
        disagreement = particle_log_probabilities.exp().var(dim=2, unbiased=False).mean(dim=(1, 2))
        gate_features = torch.cat(
            (
                slots.mean(1),
                slots.std(1, unbiased=False),
                entropy[:, None],
                margin[:, None],
                state.previous_surprise[:, None],
                disagreement[:, None],
            ),
            dim=-1,
        )
        alpha = self.ambiguity_gate(gate_features).squeeze(-1).sigmoid()
        probabilities = (1 - alpha[:, None, None]) * vanilla_probability + alpha[:, None, None] * particle_probability
        return PendingBatchUpdate(
            prior_state=state,
            unmasked_log_weights=unmasked,
            transitioned_log_weights=transitioned,
            x=x,
            probabilities=probabilities,
            particle_log_probabilities=particle_log_probabilities,
            particle_logits=particle_logits,
            vanilla_logits=vanilla_logits,
            ambiguity_probability=alpha,
            slots=slots,
            residuals=residuals,
        )

    def restrict_pending(self, pending: PendingBatchUpdate, matched_particles: torch.Tensor) -> PendingBatchUpdate:
        """Apply episode matching without rerunning a transition or prediction."""

        weights = pending.unmasked_log_weights
        mask = torch.zeros_like(weights, dtype=torch.bool)
        mask.scatter_(1, matched_particles.long(), True)
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = weights - torch.logsumexp(weights, dim=-1, keepdim=True)
        particle = torch.einsum("bk,brkc->brc", weights.exp(), pending.particle_log_probabilities.exp())
        vanilla = pending.vanilla_logits.softmax(-1)
        alpha = pending.ambiguity_probability
        probability = (1 - alpha[:, None, None]) * vanilla + alpha[:, None, None] * particle
        return replace(pending, transitioned_log_weights=weights, probabilities=probability)

    def _capped_context(self, x: torch.Tensor, y: torch.Tensor, *, temporal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] <= self.context_limit:
            return x, y
        if temporal:
            return x[:, -self.context_limit :], y[:, -self.context_limit :]
        # Deterministic stratification, performed independently for each episode.
        selected_x, selected_y = [], []
        for batch in range(x.shape[0]):
            indices = []
            for label in (0, 1):
                candidates = torch.nonzero(y[batch] == label, as_tuple=False).flatten()
                quota = self.context_limit // 2
                if candidates.numel():
                    positions = torch.linspace(
                        0,
                        candidates.numel() - 1,
                        min(quota, candidates.numel()),
                        device=x.device,
                    )
                    indices.append(candidates[positions.round().long()])
            joined = torch.cat(indices) if indices else torch.arange(x.shape[1], device=x.device)
            if joined.numel() < self.context_limit:
                remaining = torch.tensor(
                    [i for i in range(x.shape[1]) if i not in set(joined.tolist())], device=x.device
                )[: self.context_limit - joined.numel()]
                joined = torch.cat((joined, remaining))
            joined = joined[: self.context_limit].sort().values
            selected_x.append(x[batch, joined])
            selected_y.append(y[batch, joined])
        return torch.stack(selected_x), torch.stack(selected_y)

    def reveal_batch(
        self, pending: PendingBatchUpdate, y: torch.Tensor, *, temporal: bool = True
    ) -> BatchParticleState:
        """Update posterior and context after all probabilities are committed."""

        y = y.to(device=pending.x.device, dtype=torch.long)
        if y.shape != pending.x.shape[:2]:
            raise ValueError("y must align with the pending batch rows.")
        observed = pending.particle_log_probabilities.gather(
            -1, y[:, :, None, None].expand(-1, -1, pending.particle_logits.shape[2], 1)
        ).squeeze(-1)
        batch_likelihood = observed.sum(1)
        # Matching may suppress particles in the committed mixture, but the raw
        # posterior remains trainable so the unmatched-mass loss is meaningful.
        evidence = pending.unmasked_log_weights + batch_likelihood
        posterior = evidence - torch.logsumexp(evidence, dim=-1, keepdim=True)
        surprise = -torch.logsumexp(evidence, dim=-1) / y.shape[1]
        context_x = torch.cat((pending.prior_state.context_x, pending.x), dim=1)
        context_y = torch.cat((pending.prior_state.context_y, y), dim=1)
        context_x, context_y = self._capped_context(context_x, context_y, temporal=temporal)
        return replace(
            pending.prior_state,
            log_weights=posterior,
            previous_surprise=surprise,
            context_x=context_x,
            context_y=context_y,
            batches_seen=pending.prior_state.batches_seen + 1,
        )
