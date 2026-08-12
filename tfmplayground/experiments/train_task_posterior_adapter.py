"""Training utilities for contrastively supervised task-posterior adapters.

The objective accepts the repository's ordinary and paired-prior episode
batches.  It uses candidate stream *and* query labels when available and mixes
paired episodes with ordinary single-task episodes at the caller level.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    TaskPosteriorLoss,
    task_posterior_loss,
)


@dataclass(frozen=True)
class TaskPosteriorTrainingConfig:
    particle_count: int = 4
    ordinary_episode_fraction: float = 0.35
    specialization_weight: float = 0.25
    coherence_weight: float = 0.10
    diversity_weight: float = 0.02
    residual_weight: float = 0.01
    ordinary_posterior_weight: float = 0.02
    min_context_size: int = 32
    max_context_size: int = 1024
    min_features: int = 1
    max_features: int = 100
    max_classes: int = 10

    def validate(self) -> None:
        if self.particle_count != 4:
            raise ValueError("K=4 is frozen until the representation gate passes.")
        if not 0 <= self.ordinary_episode_fraction <= 1:
            raise ValueError("ordinary_episode_fraction must be in [0, 1].")
        if not 2 <= self.min_context_size <= self.max_context_size <= 1024:
            raise ValueError("Context curriculum must stay between 2 and 1,024 rows.")
        if not 1 <= self.min_features <= self.max_features:
            raise ValueError("Feature curriculum bounds are invalid.")
        if not 2 <= self.max_classes <= 10:
            raise ValueError("Classification curriculum supports at most 10 classes.")


@dataclass(frozen=True)
class EpisodeObjective:
    total: torch.Tensor
    prior_only: TaskPosteriorLoss
    updated: TaskPosteriorLoss


def _candidate_labels(batch, prefix: str) -> torch.Tensor | None:
    value = getattr(batch, f"candidate_{prefix}_y", None)
    return None if value is None else value.long()


def contrastive_episode_objective(
    model: NanoTabPFNTaskPosteriorAdapter,
    batch,
    config: TaskPosteriorTrainingConfig,
) -> EpisodeObjective:
    """Compute prior-only and evidence-updated losses for an episode batch.

    Paired batches directly match candidate tasks to slots over the candidate
    stream+query targets.  Ordinary batches have no candidate tensors and
    therefore train only mixture accuracy and a small zero-residual preference.
    """
    config.validate()
    if model.context_mode != "iid_set":
        raise ValueError("The TabArena training objective requires context_mode='iid_set'.")
    if model.particle_count != config.particle_count:
        raise ValueError("Model and training particle counts do not match.")
    labels = torch.cat(
        (
            batch.initial_support_y.long().flatten(),
            batch.stream_y.long().flatten(),
            batch.query_y.long().flatten(),
        )
    )
    class_count = int(labels.max().item() + 1)
    class_count = max(class_count, 2)

    prior_targets_x = torch.cat((batch.stream_x, batch.query_x), dim=1)
    prior_targets_y = torch.cat((batch.stream_y.long(), batch.query_y.long()), dim=1)
    candidate_stream = _candidate_labels(batch, "stream")
    candidate_query = _candidate_labels(batch, "query")
    prior_candidates = None
    if candidate_stream is not None and candidate_query is not None:
        prior_candidates = torch.cat((candidate_stream, candidate_query), dim=2)
    prior_prediction = model(
        batch.initial_support_x,
        batch.initial_support_y.long(),
        prior_targets_x,
        class_count=class_count,
    )
    prior_loss = task_posterior_loss(
        prior_prediction,
        prior_targets_y,
        candidate_y=prior_candidates,
        specialization_weight=config.specialization_weight,
        coherence_weight=config.coherence_weight,
        diversity_weight=config.diversity_weight,
        residual_weight=config.residual_weight,
        ordinary_posterior_weight=config.ordinary_posterior_weight,
    )

    complete_context_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
    complete_context_y = torch.cat((batch.initial_support_y.long(), batch.stream_y.long()), dim=1)
    updated_prediction = model(
        complete_context_x,
        complete_context_y,
        batch.query_x,
        class_count=class_count,
    )
    updated_loss = task_posterior_loss(
        updated_prediction,
        batch.query_y.long(),
        candidate_y=candidate_query,
        specialization_weight=config.specialization_weight,
        coherence_weight=config.coherence_weight,
        diversity_weight=config.diversity_weight,
        residual_weight=config.residual_weight,
        ordinary_posterior_weight=config.ordinary_posterior_weight,
        assignment=prior_loss.assignment,
    )
    return EpisodeObjective(
        total=0.5 * (prior_loss.total + updated_loss.total),
        prior_only=prior_loss,
        updated=updated_loss,
    )


def choose_ordinary_episode(*, step: int, seed: int, ordinary_episode_fraction: float = 0.35) -> bool:
    """Deterministic curriculum switch, stable across resumed training runs."""
    if not 0 <= ordinary_episode_fraction <= 1:
        raise ValueError("ordinary_episode_fraction must be in [0, 1].")
    generator = torch.Generator().manual_seed(seed + step * 1_000_003)
    return bool(torch.rand((), generator=generator) < ordinary_episode_fraction)
