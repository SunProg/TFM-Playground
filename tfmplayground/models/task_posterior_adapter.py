"""Vanilla-anchored, label-conditioned particle adapter for IID tables.

This module intentionally lives beside, rather than replaces, the sequential
latent filter.  Existing checkpoints encode a chronology-dependent experiment;
the model below is the TabArena-facing representation.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel

SLOT_MODES = ("deterministic", "gaussian")
# sigmoid/exp guards.  adaptive_particle_filter.py:64 is the precedent: an
# unguarded log on one side of a bounded quantity produced unrecoverable NaN
# gradients that clipping could not rescue.
SLOT_LOG_SIGMA_MIN = -6.0
SLOT_LOG_SIGMA_MAX = 2.0


@dataclass(frozen=True)
class TaskPosteriorPrediction:
    """Predictions and task-posterior diagnostics for one table."""

    vanilla_logits: torch.Tensor
    particle_logits: torch.Tensor
    log_weights: torch.Tensor
    slots: torch.Tensor
    residuals: torch.Tensor
    # Present only when the adapter carries a structural probe.  The probe is
    # off the prediction path, so these never influence particle_logits.
    structural: torch.Tensor | None = None
    kl: torch.Tensor | None = None

    def particle_probabilities(self) -> torch.Tensor:
        return self.particle_logits.softmax(dim=-1)

    def marginal_probabilities(self) -> torch.Tensor:
        return torch.einsum("bk,bqkc->bqc", self.log_weights.exp(), self.particle_probabilities())

    def matched_marginal_probabilities(self, matched_particles: torch.Tensor) -> torch.Tensor:
        """Mixture after suppressing particles unmatched to supervised regimes."""

        if matched_particles.ndim != 2 or matched_particles.shape[0] != self.log_weights.shape[0]:
            raise ValueError("matched_particles must have shape (batch, supervised regimes).")
        mask = torch.zeros_like(self.log_weights, dtype=torch.bool)
        mask.scatter_(1, matched_particles.long(), True)
        weights = self.log_weights.masked_fill(~mask, float("-inf"))
        weights = (weights - torch.logsumexp(weights, dim=-1, keepdim=True)).exp()
        return torch.einsum("bk,bqkc->bqc", weights, self.particle_probabilities())

    def ambiguity(self) -> torch.Tensor:
        """Normalized posterior entropy; zero means one effective task."""
        weights = self.log_weights.exp()
        entropy = -(weights * self.log_weights).sum(dim=-1)
        return entropy / math.log(weights.shape[-1]) if weights.shape[-1] > 1 else entropy

    def top_two_margin(self) -> torch.Tensor:
        if self.log_weights.shape[-1] == 1:
            return torch.ones_like(self.log_weights[:, 0])
        top = self.log_weights.exp().topk(2, dim=-1).values
        return top[:, 0] - top[:, 1]

    def effective_particle_count(self) -> torch.Tensor:
        """Inverse participation ratio ``1 / sum(w^2)``; one means full collapse."""
        return 1.0 / self.log_weights.exp().square().sum(dim=-1).clamp_min(1e-12)

    def slot_dispersion(self) -> torch.Tensor:
        """Mean per-dimension standard deviation across particles.

        Collapse in *representation* space, which the weight-based effective
        count cannot see: distinct weights over identical slots still report a
        healthy effective count.
        """
        if self.slots.shape[1] < 2:
            return torch.zeros_like(self.slots[:, 0, 0])
        return self.slots.std(dim=1).mean(dim=-1)


class _ResidualParticleDecoder(nn.Module):
    def __init__(self, embedding_size: int, class_count: int, residual_logit_bound: float | None = None):
        super().__init__()
        self.residual_logit_bound = residual_logit_bound
        self.hidden = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size),
            nn.GELU(),
        )
        self.output = nn.Linear(embedding_size, class_count)
        # The adapter must start as the exact pretrained model, for every slot.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, rows: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        row_count, slot_count = rows.shape[1], slots.shape[1]
        expanded_rows = rows[:, :, None].expand(-1, -1, slot_count, -1)
        expanded_slots = slots[:, None].expand(-1, row_count, -1, -1)
        features = torch.cat((expanded_rows, expanded_slots, expanded_rows * expanded_slots), dim=-1)
        residuals = self.output(self.hidden(features))
        if self.residual_logit_bound is not None:
            residuals = self.residual_logit_bound * torch.tanh(residuals / self.residual_logit_bound)
        return residuals


class StructuralLatentProbe(nn.Module):
    """Reads a structural task summary out of a particle slot.

    The probe never contributes to ``particle_logits``, so an adapter carrying
    one is still numerically identical to vanilla at initialization.  The final
    ``family_count`` outputs are logits for a one-hot family block; every
    earlier output is squashed into ``[0, 1]`` to match the targets produced by
    ``tfmplayground.experiments.structural_latents``.
    """

    def __init__(self, embedding_size: int, latent_dim: int, family_count: int):
        super().__init__()
        if latent_dim <= family_count:
            raise ValueError("latent_dim must exceed family_count so continuous entries remain.")
        if family_count < 2:
            raise ValueError("family_count must be at least two.")
        self.latent_dim = latent_dim
        self.family_count = family_count
        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, latent_dim)

    @property
    def continuous_dim(self) -> int:
        return self.latent_dim - self.family_count

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        raw = self.output(self.norm(slots))
        continuous = torch.sigmoid(raw[..., : self.continuous_dim])
        return torch.cat((continuous, raw[..., self.continuous_dim :]), dim=-1)


class NanoTabPFNTaskPosteriorAdapter(nn.Module):
    """Permutation-invariant task posterior whose particles correct vanilla.

    ``context_mode='iid_set'`` is the production default.  The complete labeled
    context participates in slot construction.  ``sequential`` is retained only
    as an explicit controlled-research path via :meth:`forward_sequential`.
    """

    model_type = "nanotabpfn_task_posterior_adapter"

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        particle_count: int = 4,
        max_classes: int = 10,
        context_mode: str = "iid_set",
        residual_logit_bound: float | None = None,
        slot_mode: str = "deterministic",
        slot_sample_seed: int = 0,
        structural_latent_dim: int | None = None,
        structural_family_count: int = 2,
        structural_detach: bool = False,
    ):
        super().__init__()
        if particle_count < 1:
            raise ValueError("particle_count must be positive.")
        if max_classes < 2 or max_classes > 10:
            raise ValueError("max_classes must be between 2 and 10.")
        if max_classes > backbone.num_outputs:
            raise ValueError("max_classes cannot exceed the backbone output count.")
        if context_mode not in {"iid_set", "sequential"}:
            raise ValueError("context_mode must be 'iid_set' or 'sequential'.")
        if residual_logit_bound is not None and residual_logit_bound <= 0:
            raise ValueError("residual_logit_bound must be positive or None.")
        if slot_mode not in SLOT_MODES:
            raise ValueError(f"slot_mode must be one of {SLOT_MODES}.")
        self.backbone = backbone
        self.particle_count = particle_count
        self.max_classes = max_classes
        self.context_mode = context_mode
        self.slot_mode = slot_mode
        self.slot_sample_seed = slot_sample_seed
        self.structural_detach = structural_detach
        embedding_size = backbone.embedding_size
        # The deterministic arm carries one learned query per particle and gets
        # its diversity from that bank.  The Gaussian arm carries a single query
        # summarizing the evidence into (mu, log sigma) and gets its diversity
        # from sampling, which also makes the particle count a runtime knob.
        query_count = particle_count if slot_mode == "deterministic" else 1
        self.slot_queries = nn.Parameter(torch.empty(query_count, embedding_size))
        nn.init.normal_(self.slot_queries, std=embedding_size**-0.5)
        self.evidence_attention = nn.MultiheadAttention(embedding_size, backbone.num_attention_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(embedding_size)
        self.posterior_projection: nn.Linear | None = None
        if slot_mode == "gaussian":
            projection = nn.Linear(embedding_size, 2 * embedding_size)
            # Start at sigma = 1 exactly; the mean half keeps its default init so
            # it carries gradient from the first step.
            with torch.no_grad():
                projection.weight[embedding_size:].zero_()
                projection.bias[embedding_size:].zero_()
            self.posterior_projection = projection
        self.posterior_head = nn.Linear(embedding_size, 1)
        nn.init.zeros_(self.posterior_head.weight)
        nn.init.zeros_(self.posterior_head.bias)
        self.residual_decoder = _ResidualParticleDecoder(embedding_size, max_classes, residual_logit_bound)
        self.structural_probe: StructuralLatentProbe | None = None
        if structural_latent_dim is not None:
            self.structural_probe = StructuralLatentProbe(
                embedding_size, structural_latent_dim, structural_family_count
            )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def adapter_parameters(self):
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("backbone."))

    def _validate(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor, class_count: int
    ) -> None:
        if context_x.ndim != 3 or query_x.ndim != 3:
            raise ValueError("Context and query features must have shape (batch, rows, features).")
        if context_y.shape != context_x.shape[:2]:
            raise ValueError("context_y must have shape (batch, context rows).")
        if context_x.shape[0] != query_x.shape[0] or context_x.shape[2] != query_x.shape[2]:
            raise ValueError("Context and query batch/feature dimensions must match.")
        if context_x.shape[1] < 2:
            raise ValueError("At least two labeled context rows are required.")
        if not 2 <= class_count <= self.max_classes:
            raise ValueError(f"class_count must be between 2 and {self.max_classes}.")

    def _slot_noise(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        """Reparameterization noise, deterministic outside training.

        ``TaskPosteriorClassifier`` promises reproducible ensembles for a given
        ``random_state``, and the paired bootstrap gate is only valid if repeated
        ``predict_proba`` calls agree.  Evaluation therefore draws from a seeded
        generator rather than the global stream; training uses the global stream
        so samples vary across steps while a global ``manual_seed`` still makes
        the whole run reproducible.
        """
        shape = (batch, self.particle_count, self.backbone.embedding_size)
        if self.training:
            noise = torch.randn(shape, device=reference.device, dtype=reference.dtype)
        else:
            generator = torch.Generator().manual_seed(self.slot_sample_seed)
            noise = torch.randn(shape, generator=generator).to(device=reference.device, dtype=reference.dtype)
        return noise

    def _build_slots(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        seeds = self.slot_queries[None].expand(evidence.shape[0], -1, -1)
        attended = self.evidence_attention(seeds, evidence, evidence, need_weights=False)[0]
        summary = self.slot_norm(seeds + attended)
        if self.slot_mode == "deterministic":
            return summary, None
        assert self.posterior_projection is not None
        mean, log_sigma = self.posterior_projection(summary).chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(SLOT_LOG_SIGMA_MIN, SLOT_LOG_SIGMA_MAX)
        slots = mean + log_sigma.exp() * self._slot_noise(evidence.shape[0], summary)
        kl = 0.5 * (mean.square() + (2.0 * log_sigma).exp() - 1.0 - 2.0 * log_sigma).sum(-1).squeeze(-1)
        return slots, kl

    def _decode_encoded(self, encoded: torch.Tensor, context_count: int, class_count: int) -> TaskPosteriorPrediction:
        # Target-column support states contain label embeddings after repeated
        # feature/row attention, hence each evidence item depends on both x_i and y_i.
        evidence = encoded[:, :context_count, -1, :]
        query_states = encoded[:, context_count:, -1, :]
        slots, kl = self._build_slots(evidence)
        log_weights = F.log_softmax(self.posterior_head(slots).squeeze(-1), dim=-1)
        vanilla_logits = self.backbone.decoder(query_states)[..., :class_count]
        residuals = self.residual_decoder(query_states, slots)[..., :class_count]
        particle_logits = vanilla_logits[:, :, None, :] + residuals
        structural = None
        if self.structural_probe is not None:
            structural = self.structural_probe(slots.detach() if self.structural_detach else slots)
        return TaskPosteriorPrediction(
            vanilla_logits=vanilla_logits,
            particle_logits=particle_logits,
            log_weights=log_weights,
            slots=slots,
            residuals=residuals,
            structural=structural,
            kl=kl,
        )

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        class_count: int | None = None,
        num_mem_chunks: int = 1,
    ) -> TaskPosteriorPrediction:
        if self.context_mode != "iid_set":
            raise ValueError(
                "A sequential adapter requires forward_sequential(initial_x, initial_y, stream_x, stream_y, query_x)."
            )
        class_count = self.max_classes if class_count is None else class_count
        self._validate(context_x, context_y, query_x, class_count)
        all_x = torch.cat((context_x, query_x), dim=1)
        encoded = self.backbone.encode_table(
            (all_x, context_y.float()),
            train_test_split_index=context_x.shape[1],
            num_mem_chunks=num_mem_chunks,
        )
        return self._decode_encoded(encoded, context_x.shape[1], class_count)

    def forward_sequential(
        self,
        initial_x: torch.Tensor,
        initial_y: torch.Tensor,
        stream_x: torch.Tensor,
        stream_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        class_count: int | None = None,
        num_mem_chunks: int = 1,
    ) -> TaskPosteriorPrediction:
        """Compatibility mode: slots use initial evidence; weights use stream labels."""
        if self.context_mode != "sequential":
            raise ValueError("forward_sequential is only available in context_mode='sequential'.")
        class_count = self.max_classes if class_count is None else class_count
        self._validate(initial_x, initial_y, query_x, class_count)
        if stream_y.shape != stream_x.shape[:2]:
            raise ValueError("stream_y must have shape (batch, stream rows).")
        targets = torch.cat((stream_x, query_x), dim=1)
        encoded = self.backbone.encode_table(
            (torch.cat((initial_x, targets), dim=1), initial_y.float()),
            train_test_split_index=initial_x.shape[1],
            num_mem_chunks=num_mem_chunks,
        )
        prediction = self._decode_encoded(encoded, initial_x.shape[1], class_count)
        stream_count = stream_x.shape[1]
        stream_logits = prediction.particle_logits[:, :stream_count]
        labels = stream_y.long()[:, :, None, None].expand(-1, -1, self.particle_count, 1)
        evidence = F.log_softmax(stream_logits, dim=-1).gather(-1, labels).squeeze(-1).sum(1)
        log_weights = prediction.log_weights + evidence
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        return TaskPosteriorPrediction(
            vanilla_logits=prediction.vanilla_logits[:, stream_count:],
            particle_logits=prediction.particle_logits[:, stream_count:],
            log_weights=log_weights,
            slots=prediction.slots,
            residuals=prediction.residuals[:, stream_count:],
            structural=prediction.structural,
            kl=prediction.kl,
        )


@dataclass(frozen=True)
class TaskPosteriorLoss:
    total: torch.Tensor
    mixture: torch.Tensor
    specialization: torch.Tensor
    coherence: torch.Tensor
    diversity: torch.Tensor
    residual: torch.Tensor
    posterior: torch.Tensor
    structural: torch.Tensor
    kl: torch.Tensor
    assignment: RegimeParticleAssignment | None = None


@dataclass(frozen=True)
class RegimeParticleAssignment:
    """Episode-level one-to-one regime assignment, reused across its trajectory."""

    particle_for_regime: torch.Tensor


def match_regimes_to_particles(
    prediction: TaskPosteriorPrediction, candidate_y: torch.Tensor
) -> RegimeParticleAssignment:
    """Match supervised regime candidates to distinct particles once per episode."""

    if candidate_y.ndim != 3 or candidate_y.shape[0] != prediction.particle_logits.shape[0]:
        raise ValueError("candidate_y must have shape (batch, regimes, rows).")
    batch, rows, particles, _ = prediction.particle_logits.shape
    if candidate_y.shape[2] != rows or candidate_y.shape[1] > particles:
        raise ValueError("Candidate rows must match predictions and regimes cannot exceed particles.")
    log_probabilities = F.log_softmax(prediction.particle_logits, dim=-1)
    regimes = candidate_y.shape[1]
    costs = []
    for regime in range(regimes):
        labels = candidate_y[:, regime, :, None, None].long().expand(-1, -1, particles, 1)
        costs.append(-log_probabilities.gather(-1, labels).squeeze(-1).mean(1))
    pair_cost = torch.stack(costs, dim=1)
    choices = list(itertools.permutations(range(particles), regimes))
    selected = []
    for batch_index in range(batch):
        totals = torch.stack(
            [
                sum(pair_cost[batch_index, regime, particle] for regime, particle in enumerate(choice))
                for choice in choices
            ]
        )
        selected.append(choices[int(totals.detach().argmin())])
    return RegimeParticleAssignment(torch.tensor(selected, device=prediction.log_weights.device, dtype=torch.long))


def regime_posterior_supervision_loss(
    prediction: TaskPosteriorPrediction,
    active_regime: torch.Tensor,
    assignment: RegimeParticleAssignment,
    *,
    unmatched_weight: float = 0.1,
) -> torch.Tensor:
    """Supervise posterior mass after labels arrive and suppress unused slots."""

    if active_regime.shape != (prediction.log_weights.shape[0],):
        raise ValueError("active_regime must have shape (batch,).")
    mapping = assignment.particle_for_regime
    if mapping.shape[0] != prediction.log_weights.shape[0]:
        raise ValueError("Assignment batch dimension does not match prediction.")
    target = mapping.gather(1, active_regime.long()[:, None]).squeeze(1)
    supervised = F.nll_loss(prediction.log_weights, target)
    mask = torch.zeros_like(prediction.log_weights, dtype=torch.bool)
    mask.scatter_(1, mapping, True)
    unmatched_mass = prediction.log_weights.exp().masked_fill(mask, 0).sum(-1).mean()
    return supervised + unmatched_weight * unmatched_mass


def structural_latent_loss(
    prediction: TaskPosteriorPrediction,
    structural_z: torch.Tensor,
    assignment: RegimeParticleAssignment,
    *,
    family_count: int,
    feature_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score each matched particle's probe output against its candidate's structure.

    ``structural_z`` has shape ``(batch, candidates, latent_dim)`` and follows the
    layout of :class:`tfmplayground.experiments.structural_latents.StructuralLatentSchema`:
    a per-feature relevance block, then scalar summaries, then a family one-hot
    as the final ``family_count`` entries.  Candidates reach particles through
    the same label-space assignment the specialization loss uses, so a particle
    is never asked to describe a task it was not matched to.

    ``feature_mask`` has shape ``(batch, max_features)`` and zeroes the padded
    relevance entries of narrower tables.
    """
    if prediction.structural is None:
        raise ValueError("The adapter was built without a structural probe.")
    if structural_z.ndim != 3:
        raise ValueError("structural_z must have shape (batch, candidates, latent_dim).")
    batch, candidates, latent_dim = structural_z.shape
    if latent_dim != prediction.structural.shape[-1]:
        raise ValueError("structural_z and the probe output disagree on latent_dim.")
    if latent_dim <= family_count:
        raise ValueError("latent_dim must exceed family_count.")
    mapping = assignment.particle_for_regime
    if mapping.shape != (batch, candidates):
        raise ValueError("The assignment must match the batch and candidate dimensions.")

    continuous_dim = latent_dim - family_count
    index = mapping[:, :, None].expand(-1, -1, latent_dim)
    matched = prediction.structural.gather(1, index)
    continuous_error = (matched[..., :continuous_dim] - structural_z[..., :continuous_dim]).square()

    weights = torch.ones_like(continuous_error)
    if feature_mask is not None:
        features = feature_mask.shape[-1]
        if feature_mask.shape != (batch, features) or features > continuous_dim:
            raise ValueError("feature_mask must have shape (batch, max_features) within the continuous block.")
        weights[..., :features] = feature_mask[:, None, :].to(weights.dtype)
    continuous = (continuous_error * weights).sum() / weights.sum().clamp_min(1.0)

    # An all-zero family block means the source could not recover the generating
    # family (empirical dumps store observations only).  Zero is not a valid
    # one-hot, so it is unambiguous: drop those rows instead of training the
    # probe towards an arbitrary argmax of zeros.
    family_block = structural_z[..., continuous_dim:].reshape(batch * candidates, family_count)
    known = family_block.sum(-1) > 0
    family = continuous.new_zeros(())
    if bool(known.any()):
        family_logits = matched[..., continuous_dim:].reshape(batch * candidates, family_count)
        family = F.cross_entropy(family_logits[known], family_block[known].argmax(-1))
    return continuous + family


def task_posterior_loss(
    prediction: TaskPosteriorPrediction,
    target_y: torch.Tensor,
    *,
    candidate_y: torch.Tensor | None = None,
    specialization_weight: float = 0.25,
    coherence_weight: float = 0.10,
    diversity_weight: float = 0.02,
    residual_weight: float = 0.01,
    ordinary_posterior_weight: float = 0.02,
    structural_z: torch.Tensor | None = None,
    structural_feature_mask: torch.Tensor | None = None,
    structural_family_count: int = 2,
    structural_weight: float = 0.0,
    kl_weight: float = 0.0,
    assignment: RegimeParticleAssignment | None = None,
) -> TaskPosteriorLoss:
    """Primary mixture CE plus directly matched candidate-task supervision.

    ``candidate_y`` has shape ``(batch, candidates, query rows)``.  Candidate
    tasks are assigned one-to-one to particles by minimum detached CE.  This is
    deliberately semantic supervision; diversity is only a weak auxiliary.
    """
    if target_y.shape != prediction.particle_logits.shape[:2]:
        raise ValueError("target_y must match the prediction batch and query dimensions.")
    probabilities = prediction.marginal_probabilities().clamp_min(1e-12)
    mixture = F.nll_loss(probabilities.log().flatten(0, 1), target_y.long().flatten())
    particle_log_probs = F.log_softmax(prediction.particle_logits, dim=-1)
    batch, queries, particles, _ = particle_log_probs.shape

    zero = mixture.new_zeros(())
    specialization = zero
    coherence = zero
    # Ordinary episodes use slot zero as the canonical no-correction task.  A
    # fixed canonical slot breaks the otherwise stationary uniform-posterior
    # symmetry and teaches one effective hypothesis without inventing ambiguity.
    posterior = -prediction.log_weights[:, 0].mean() if candidate_y is None else zero
    if candidate_y is not None:
        if candidate_y.ndim != 3 or candidate_y.shape[0] != batch or candidate_y.shape[2] != queries:
            raise ValueError("candidate_y must have shape (batch, candidates, query rows).")
        candidates = candidate_y.shape[1]
        if candidates > particles:
            raise ValueError("There cannot be more candidate tasks than particles.")
        pair_cost = []
        for candidate in range(candidates):
            labels = candidate_y[:, candidate].long()[:, :, None, None].expand(-1, -1, particles, 1)
            pair_cost.append(-particle_log_probs.gather(-1, labels).squeeze(-1).mean(1))
        costs = torch.stack(pair_cost, dim=1)  # batch, candidate, particle
        assigned_losses = []
        coherent_losses = []
        if assignment is None:
            assignment = match_regimes_to_particles(prediction, candidate_y)
        if assignment.particle_for_regime.shape != (batch, candidates):
            raise ValueError("The reusable assignment must match batch and candidate dimensions.")
        for batch_index in range(batch):
            chosen = assignment.particle_for_regime[batch_index].tolist()
            assigned_losses.extend(costs[batch_index, c, p] for c, p in enumerate(chosen))
            for candidate, particle in enumerate(chosen):
                labels = candidate_y[batch_index, candidate].long()
                coherent_losses.append(
                    -particle_log_probs[batch_index, torch.arange(queries), particle, labels].sum() / max(queries, 1)
                )
        specialization = torch.stack(assigned_losses).mean()
        coherence = torch.stack(coherent_losses).mean()

    # Structure-space supervision needs the same candidate-to-particle matching
    # as the label-space terms, so it is only available on paired episodes.
    structural = zero
    if structural_z is not None and structural_weight != 0.0:
        if assignment is None:
            raise ValueError("Structural supervision requires candidate_y so particles can be matched.")
        structural = structural_latent_loss(
            prediction,
            structural_z,
            assignment,
            family_count=structural_family_count,
            feature_mask=structural_feature_mask,
        )

    kl = zero if prediction.kl is None else prediction.kl.mean()

    particle_probs = particle_log_probs.exp()
    mean_particle = particle_probs.mean(dim=2)
    mean_entropy = -(mean_particle * mean_particle.clamp_min(1e-12).log()).sum(-1)
    component_entropy = -(particle_probs * particle_probs.clamp_min(1e-12).log()).sum(-1).mean(2)
    diversity = -(mean_entropy - component_entropy).mean()
    residual = prediction.residuals.square().mean()
    total = (
        mixture
        + specialization_weight * specialization
        + coherence_weight * coherence
        + diversity_weight * diversity
        + residual_weight * residual
        + ordinary_posterior_weight * posterior
        + structural_weight * structural
        + kl_weight * kl
    )
    return TaskPosteriorLoss(
        total, mixture, specialization, coherence, diversity, residual, posterior, structural, kl, assignment
    )


def task_posterior_checkpoint(
    model: NanoTabPFNTaskPosteriorAdapter,
    *,
    training_config: dict[str, Any],
    lineage: dict[str, Any],
    data_provenance: dict[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name, value in model.backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    backbone = model.backbone
    return {
        "model_type": model.model_type,
        "architecture": {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "particle_count": model.particle_count,
            "max_classes": model.max_classes,
            "context_mode": model.context_mode,
            "residual_logit_bound": model.residual_decoder.residual_logit_bound,
            "slot_mode": model.slot_mode,
            "slot_sample_seed": model.slot_sample_seed,
            "structural_latent_dim": None if model.structural_probe is None else model.structural_probe.latent_dim,
            "structural_family_count": 2 if model.structural_probe is None else model.structural_probe.family_count,
            "structural_detach": model.structural_detach,
        },
        "model": model.state_dict(),
        "training_config": training_config,
        "lineage": lineage,
        "data_provenance": data_provenance,
        "backbone_sha256": digest.hexdigest(),
    }


def save_task_posterior_checkpoint(path: str | Path, model: NanoTabPFNTaskPosteriorAdapter, **kwargs: Any) -> None:
    torch.save(task_posterior_checkpoint(model, **kwargs), Path(path))


def load_task_posterior_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNTaskPosteriorAdapter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("model_type") != NanoTabPFNTaskPosteriorAdapter.model_type:
        raise ValueError("Checkpoint is not a task-posterior adapter.")
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    model = NanoTabPFNTaskPosteriorAdapter(
        backbone,
        particle_count=architecture["particle_count"],
        max_classes=architecture["max_classes"],
        context_mode=architecture.get("context_mode", "iid_set"),
        residual_logit_bound=architecture.get("residual_logit_bound"),
        slot_mode=architecture.get("slot_mode", "deterministic"),
        slot_sample_seed=architecture.get("slot_sample_seed", 0),
        structural_latent_dim=architecture.get("structural_latent_dim"),
        structural_family_count=architecture.get("structural_family_count", 2),
        structural_detach=architecture.get("structural_detach", False),
    )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
