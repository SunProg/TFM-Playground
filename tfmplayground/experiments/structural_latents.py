"""Structural summaries of a candidate task, used as particle supervision.

The task-posterior adapter is supervised in label space only: a particle is told
which candidate label vector it should reproduce.  That says nothing about what
a particle should *represent*, which is the diagnosis recorded in
``ADAPTIVE_PARTICLE_FILTER_RESEARCH.md`` for why particles collapse.

This module produces a fixed-width structural vector per candidate task -
per-feature relevance, label noise, boundary complexity, class balance, and the
generating SCM family - so a probe head can be supervised in structure space
instead.

Scope of the estimate.  Feature relevance, noise, complexity, and balance are
computed from the episode's own rows and the candidate's labels, not read out of
the generator's internals.  TabICL's SCM classes synthesize their own feature
matrix rather than accepting one, so an interventional relevance would require a
generator-side API that does not exist today; :func:`structural_latent_vector`
takes a materialized ``(x, y)`` pair precisely so an exact estimator can replace
the data-derived one later without touching any caller.  Only the family entry
is exact generator truth.

Like every ``candidate_*`` tensor in this repository these vectors are
evaluator- and loss-only.  They are never an input to a model's forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FAMILIES: tuple[str, ...] = ("mlp_scm", "tree_scm")

# Continuous entries beyond the per-feature relevance block, in order.
SCALAR_NAMES: tuple[str, ...] = ("label_noise", "boundary_complexity", "class_balance")


@dataclass(frozen=True)
class StructuralLatentSchema:
    """Layout of a structural latent vector.

    The family one-hot is always the final block so that the loss only needs to
    know ``family_count`` and never imports this module.
    """

    max_features: int
    families: tuple[str, ...] = FAMILIES

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features must be positive.")
        if len(self.families) < 2:
            raise ValueError("At least two families are required for a one-hot block.")

    @property
    def family_count(self) -> int:
        return len(self.families)

    @property
    def continuous_dim(self) -> int:
        return self.max_features + len(SCALAR_NAMES)

    @property
    def latent_dim(self) -> int:
        return self.continuous_dim + self.family_count

    @property
    def relevance_slice(self) -> slice:
        return slice(0, self.max_features)

    @property
    def scalar_slice(self) -> slice:
        return slice(self.max_features, self.continuous_dim)

    @property
    def family_slice(self) -> slice:
        return slice(self.continuous_dim, self.latent_dim)

    def dimension_names(self) -> tuple[str, ...]:
        relevance = tuple(f"relevance_{index}" for index in range(self.max_features))
        family = tuple(f"family_{name}" for name in self.families)
        return relevance + SCALAR_NAMES + family

    def family_index(self, family: str) -> int:
        if family not in self.families:
            raise ValueError(f"Unknown family {family!r}; expected one of {self.families}.")
        return self.families.index(family)


def _class_counts(y: torch.Tensor, class_count: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(y.long(), num_classes=class_count).to(torch.float64)


def _entropy(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum(-1, keepdim=True).clamp_min(1e-12)
    probabilities = counts / total
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)


def _stump_scan(column: torch.Tensor, onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best single-threshold split on one feature.

    Returns ``(normalized_information_gain, accuracy)``.  The gain is divided by
    the parent entropy so it lands in ``[0, 1]`` and stays comparable across
    episodes with different class balances.
    """
    rows = column.shape[0]
    order = torch.argsort(column)
    sorted_column = column[order]
    sorted_onehot = onehot[order]
    cumulative = sorted_onehot.cumsum(0)
    total = cumulative[-1]
    parent_entropy = _entropy(total)
    majority = total.max() / rows

    if rows < 2:
        return torch.zeros((), dtype=torch.float64), majority
    valid = sorted_column[1:] > sorted_column[:-1]
    if not bool(valid.any()):
        return torch.zeros((), dtype=torch.float64), majority

    left = cumulative[:-1]
    right = total - left
    left_count = left.sum(-1)
    right_count = right.sum(-1)
    split_entropy = (left_count * _entropy(left) + right_count * _entropy(right)) / rows
    gain = (parent_entropy - split_entropy).masked_fill(~valid, float("-inf")).max().clamp_min(0.0)
    if parent_entropy <= 1e-12:
        normalized_gain = torch.zeros((), dtype=torch.float64)
    else:
        normalized_gain = (gain / parent_entropy).clamp(0.0, 1.0)

    accuracy = ((left.max(-1).values + right.max(-1).values) / rows).masked_fill(~valid, float("-inf")).max()
    return normalized_gain, torch.maximum(accuracy, majority)


def _nearest_neighbour_disagreement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fraction of rows whose nearest neighbour carries a different label."""
    rows = x.shape[0]
    if rows < 2:
        return torch.zeros((), dtype=torch.float64)
    centered = x - x.mean(0, keepdim=True)
    standardized = centered / centered.std(0, keepdim=True).clamp_min(1e-8)
    distances = torch.cdist(standardized, standardized)
    distances.fill_diagonal_(float("inf"))
    neighbours = distances.argmin(1)
    return (y[neighbours] != y).to(torch.float64).mean()


def structural_latent_vector(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    schema: StructuralLatentSchema,
    family: str | None,
    class_count: int | None = None,
) -> torch.Tensor:
    """Summarize one candidate task as a fixed-width structural vector.

    ``x`` is ``(rows, features)`` and ``y`` is ``(rows,)``.  Features beyond
    ``x.shape[1]`` are zero-padded; use :func:`structural_feature_mask` to build
    the matching mask so the loss ignores the padding.

    Every block except the family one-hot is computed from ``(x, y)`` alone, so
    an empirical source that has lost its generator identity can still supply
    them.  Pass ``family=None`` there: the family block becomes all-zero, which
    :func:`tfmplayground.models.task_posterior_adapter.structural_latent_loss`
    reads as "unknown" and drops from the objective.  A zero block is never a
    valid one-hot, so an unknown family cannot be mistaken for a known one.
    """
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be (rows, features) and y must be (rows,).")
    features = x.shape[1]
    if features > schema.max_features:
        raise ValueError(f"x has {features} features but the schema allows {schema.max_features}.")

    # Always summarize on CPU in float64: these are small per-episode statistics
    # whose accumulations want the precision, and MPS has no float64 at all.
    # Callers move the result back to the batch device.
    x = x.detach().cpu().to(torch.float64)
    y = y.detach().cpu().long()
    class_count = int(y.max().item()) + 1 if class_count is None else class_count
    class_count = max(class_count, 2)
    onehot = _class_counts(y, class_count)

    relevance = torch.zeros(schema.max_features, dtype=torch.float64)
    best_accuracy = (onehot.sum(0).max() / x.shape[0]).to(torch.float64)
    for index in range(features):
        gain, accuracy = _stump_scan(x[:, index], onehot)
        relevance[index] = gain
        best_accuracy = torch.maximum(best_accuracy, accuracy)

    scalars = torch.stack(
        (
            _nearest_neighbour_disagreement(x, y),
            (1.0 - best_accuracy).clamp(0.0, 1.0),
            (_entropy(onehot.sum(0)) / torch.tensor(class_count, dtype=torch.float64).log()).clamp(0.0, 1.0),
        )
    )

    family_onehot = torch.zeros(schema.family_count, dtype=torch.float64)
    if family is not None:
        family_onehot[schema.family_index(family)] = 1.0
    return torch.cat((relevance, scalars, family_onehot)).to(torch.float32)


def probe_r2(
    predictions: list[torch.Tensor] | torch.Tensor,
    targets: list[torch.Tensor] | torch.Tensor,
    schema: StructuralLatentSchema,
) -> dict[str, float | None]:
    """Per-dimension coefficient of determination over matched candidate tasks.

    Inputs are ``(rows, latent_dim)`` or any shape that flattens to it, as lists
    of batches or single tensors.  R2 is against each dimension's own mean, so a
    probe that only learns the dataset-wide average scores zero rather than
    looking informative, and a probe worse than that mean scores below zero.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = [predictions]
    if isinstance(targets, torch.Tensor):
        targets = [targets]
    if not predictions:
        return {}
    predicted = torch.cat([value.reshape(-1, value.shape[-1]) for value in predictions]).double().cpu()
    actual = torch.cat([value.reshape(-1, value.shape[-1]) for value in targets]).double().cpu()
    if predicted.shape != actual.shape:
        raise ValueError("Predictions and targets must flatten to the same shape.")
    residual = (actual - predicted).square().sum(0)
    variance = (actual - actual.mean(0, keepdim=True)).square().sum(0)
    scores = torch.where(variance > 1e-12, 1.0 - residual / variance.clamp_min(1e-12), torch.zeros_like(variance))
    names = schema.dimension_names()
    report: dict[str, float | None] = {name: float(scores[index]) for index, name in enumerate(names)}
    # The family block is trained as logits, so only the continuous entries have
    # a meaningful R2; report the family separately as accuracy.
    report["mean_continuous_r2"] = float(scores[: schema.continuous_dim].mean())
    # An all-zero family block means the source lost the generating family.
    # Scoring it would compare two arbitrary argmax ties, so report nothing.
    family_target = actual[:, schema.family_slice]
    known = family_target.sum(-1) > 0
    report["family_accuracy"] = (
        float((predicted[known][:, schema.family_slice].argmax(-1) == family_target[known].argmax(-1)).double().mean())
        if bool(known.any())
        else None
    )
    report["family_known_fraction"] = float(known.double().mean())
    return report


def structural_feature_mask(features: int, *, schema: StructuralLatentSchema) -> torch.Tensor:
    """Boolean-valued mask over the relevance block; ``1`` marks a real feature."""
    if not 1 <= features <= schema.max_features:
        raise ValueError(f"features must be in [1, {schema.max_features}].")
    mask = torch.zeros(schema.max_features, dtype=torch.float32)
    mask[:features] = 1.0
    return mask
