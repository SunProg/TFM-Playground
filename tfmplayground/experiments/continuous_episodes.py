"""Synthetic training episodes for the slot-free continuous posterior.

An episode contains a labelled support table, an unlabelled query table, and
``C`` candidate data-generating functions evaluated on the *same* feature rows.
``C`` is a property of the episode only.  It varies over ``{2, 4, 8, 16}`` and
never changes the model architecture, which has no hypothesis count.

Function families are held out for model selection: the training regime and the
validation regime use disjoint families *and* disjoint parameter ranges, so a
validated improvement cannot come from memorized generators.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

#: Analytic families that need no external prior package.
ANALYTIC_FAMILIES = (
    "linear",
    "threshold",
    "tree",
    "sparse_interaction",
    "smooth",
    "dense_interaction",
)
#: Structured TabICL/SCM families.
SCM_FAMILIES = ("mlp_scm", "tree_scm")
ALL_FAMILIES = ANALYTIC_FAMILIES + SCM_FAMILIES

#: Families seen during training.
TRAIN_FAMILIES = ("linear", "threshold", "tree", "sparse_interaction", "smooth", "mlp_scm")
#: Families never seen during training; model selection uses these.
HELDOUT_FAMILIES = ("dense_interaction", "tree_scm")

CONDITIONS = ("ambiguous", "identifiable", "noisy", "paired")
#: Share of support rows on which the candidates genuinely disagree.  This is
#: the *only* thing separating an ambiguous episode from an identifiable one.
AMBIGUOUS_IDENTIFYING_FRACTION = 0.0
IDENTIFIABLE_IDENTIFYING_FRACTION = (0.25, 1.0)
#: Structured label-noise levels ``eta``.
NOISE_LEVELS = (0.00, 0.05, 0.10, 0.20, 0.35)
SUPPORT_SIZES = (32, 64, 128, 256, 512)
CANDIDATE_COUNTS = (2, 4, 8, 16)

#: Likelihood guard so that a zero-noise episode still has a finite posterior.
MIN_EFFECTIVE_NOISE = 1e-3


@dataclass(frozen=True)
class EpisodeRegime:
    """A disjoint slice of the generator's parameter space.

    Attributes:
        families: function families that may be drawn.
        min_features / max_features: feature-count range.
        imbalance_range: quantile range for the decision threshold, which
            controls class imbalance.
        scale_exponent: base-ten exponent range for per-column feature scales.
    """

    families: tuple[str, ...]
    min_features: int = 2
    max_features: int = 12
    imbalance_range: tuple[float, float] = (0.35, 0.65)
    scale_exponent: tuple[float, float] = (-1.0, 1.0)


TRAIN_REGIME = EpisodeRegime(TRAIN_FAMILIES, 2, 12, (0.35, 0.65), (-1.0, 1.0))
#: Held-out families *and* held-out parameter ranges.
HELDOUT_REGIME = EpisodeRegime(HELDOUT_FAMILIES, 13, 16, (0.20, 0.35), (-2.0, -1.0))


@dataclass
class ContinuousEpisode:
    """One batch of episodes sharing shapes, condition, and candidate count.

    Attributes:
        support_x: ``(batch, support, feature)`` labelled support features.
        support_y: ``(batch, support)`` binary support labels.
        query_x: ``(batch, query, feature)`` query features.
        query_y: ``(batch, query)`` realized binary query labels.
        candidate_query_positive: ``(batch, candidate, query)`` class-1
            probability ``theta_qh`` of candidate ``h`` for query ``q``.
        candidate_support_positive: ``(batch, candidate, support)`` the same
            quantity on support rows.
        posterior: ``(batch, candidate)`` exact candidate posterior ``rho_h``.
        label_noise: the controlled noise probability ``eta``.
    """

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    candidate_query_positive: torch.Tensor
    candidate_support_positive: torch.Tensor
    posterior: torch.Tensor
    label_noise: float
    condition: str
    family: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_candidates(self) -> int:
        return self.candidate_query_positive.shape[1]

    @property
    def query_count(self) -> int:
        return self.query_x.shape[1]

    def to(self, device: torch.device | str) -> ContinuousEpisode:
        return ContinuousEpisode(
            self.support_x.to(device),
            self.support_y.to(device),
            self.query_x.to(device),
            self.query_y.to(device),
            self.candidate_query_positive.to(device),
            self.candidate_support_positive.to(device),
            self.posterior.to(device),
            self.label_noise,
            self.condition,
            self.family,
            dict(self.metadata),
        )


# --------------------------------------------------------------------------- #
# Analytic candidate functions
# --------------------------------------------------------------------------- #
def _labels_from_scores(scores: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.quantile(scores, quantile)
    return (scores > threshold).astype(np.int64)


def _linear_scores(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    weights = rng.normal(size=x.shape[1])
    return x @ weights


def _threshold_scores(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    count = min(x.shape[1], int(rng.integers(1, 4)))
    columns = rng.choice(x.shape[1], size=count, replace=False)
    weights = rng.normal(size=count)
    cuts = np.quantile(x[:, columns], rng.uniform(0.25, 0.75, size=count), axis=0).diagonal()
    return ((x[:, columns] > cuts[None, :]).astype(np.float64) * weights[None, :]).sum(axis=1)


def _tree_scores(rng: np.random.Generator, x: np.ndarray, depth: int = 3) -> np.ndarray:
    leaf = np.zeros(x.shape[0], dtype=np.int64)
    for level in range(depth):
        column = int(rng.integers(x.shape[1]))
        cut = float(np.quantile(x[:, column], rng.uniform(0.3, 0.7)))
        leaf = leaf * 2 + (x[:, column] > cut).astype(np.int64)
        del level
    values = rng.normal(size=2**depth)
    return values[leaf]


def _sparse_interaction_scores(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    count = min(x.shape[1], int(rng.integers(2, 4)))
    columns = rng.choice(x.shape[1], size=count, replace=False)
    product = np.prod(x[:, columns], axis=1)
    linear = x[:, columns] @ rng.normal(size=count)
    return product + 0.5 * linear


def _dense_interaction_scores(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    features = x.shape[1]
    matrix = rng.normal(size=(features, features)) / math.sqrt(features)
    matrix = 0.5 * (matrix + matrix.T)
    return np.einsum("nf,fg,ng->n", x, matrix, x)


def _smooth_scores(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    components = int(rng.integers(2, 5))
    projections = rng.normal(size=(x.shape[1], components)) / math.sqrt(x.shape[1])
    frequency = rng.uniform(0.5, 2.5, size=components)
    phase = rng.uniform(0, 2 * math.pi, size=components)
    amplitude = rng.normal(size=components)
    return (amplitude[None, :] * np.sin(frequency[None, :] * (x @ projections) + phase[None, :])).sum(axis=1)


_ANALYTIC_SCORES = {
    "linear": _linear_scores,
    "threshold": _threshold_scores,
    "tree": _tree_scores,
    "sparse_interaction": _sparse_interaction_scores,
    "dense_interaction": _dense_interaction_scores,
    "smooth": _smooth_scores,
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _scm_candidates(
    family: str,
    num_candidates: int,
    rows: int,
    features: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw structured TabICL/SCM candidates on one shared feature table."""
    from tabicl.prior._dataset import SCMPrior
    from tabicl.prior._mlp_scm import MLPSCM
    from tabicl.prior._reg2cls import Reg2Cls
    from tabicl.prior._tree_scm import TreeSCM

    from tfmplayground.experiments.prior_bimodal_episodes import (
        PriorBimodalConfig,
        _rng_state,
        _sample_params,
        _set_rng_state,
    )

    prior_type = "mlp_scm" if family == "mlp_scm" else "tree_scm"
    episode_config = PriorBimodalConfig(
        initial_support_count=max(2, rows // 2),
        stream_count=0,
        query_count=1,
        min_features=features,
        max_features=features,
        device="cpu",
        prior_type=prior_type,
    )
    prior = SCMPrior(
        batch_size=1,
        min_features=features,
        max_features=features,
        max_classes=2,
        min_seq_len=rows,
        max_seq_len=rows + 1,
        min_train_size=max(2, rows // 2),
        max_train_size=max(3, rows // 2 + 1),
        prior_type=prior_type,
        n_jobs=1,
        device="cpu",
    )
    outer_state = _rng_state()
    try:
        for _attempt in range(max_attempts):
            _seed_all(int(rng.integers(0, 2**31 - 1)))
            params = _sample_params(prior, episode_config, rows, features)
            prior_cls = MLPSCM if params["prior_type"] == "mlp_scm" else TreeSCM
            models = []
            for _candidate in range(num_candidates):
                _seed_all(int(rng.integers(0, 2**31 - 1)))
                with torch.no_grad():
                    models.append(prior_cls(**params))
            # Covariates are sampled exactly once, so every candidate is a
            # different latent function on the same table rather than a
            # different dataset.
            _seed_all(int(rng.integers(0, 2**31 - 1)))
            with torch.no_grad():
                shared_raw_x = models[0].xsampler.sample()
            processed_x = None
            labels = []
            for model in models:
                _seed_all(int(rng.integers(0, 2**31 - 1)))
                with torch.no_grad():
                    value = model.layers(shared_raw_x)
                    if value.shape[-1] == 1:
                        value = value.squeeze(-1)
                    x_value, y_value = Reg2Cls(params)(shared_raw_x.clone(), value)
                if processed_x is None:
                    processed_x = x_value.float()
                labels.append(y_value)
            if (
                processed_x is not None
                and torch.isfinite(processed_x).all()
                and all(torch.isfinite(value).all() and value.unique().numel() == 2 for value in labels)
            ):
                x_array = processed_x.reshape(-1, features).cpu().numpy().astype(np.float64)
                y_array = torch.stack(labels).reshape(num_candidates, -1).cpu().numpy().astype(np.int64)
                return x_array[:rows], y_array[:, :rows]
    finally:
        _set_rng_state(outer_state)
    raise RuntimeError(f"Could not draw finite binary {family} candidates within {max_attempts} attempts.")


def sample_candidate_pool(
    regime: EpisodeRegime,
    rng: np.random.Generator,
    *,
    family: str,
    num_candidates: int,
    rows: int,
    features: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return shared features ``(rows, feature)`` and labels ``(candidate, rows)``."""
    if features is None:
        features = int(rng.integers(regime.min_features, regime.max_features + 1))
    if family in SCM_FAMILIES:
        x, labels = _scm_candidates(family, num_candidates, rows, features, rng)
        return x, labels, {"features": features, "family": family}
    if family not in _ANALYTIC_SCORES:
        raise ValueError(f"Unknown function family {family!r}.")
    x = rng.normal(size=(rows, features))
    # Feature-scale variation, which also produces irrelevant columns whenever
    # a candidate function ignores them.
    scales = 10.0 ** rng.uniform(*regime.scale_exponent, size=features)
    x = x * scales[None, :]
    score_function = _ANALYTIC_SCORES[family]
    labels = []
    for _candidate in range(num_candidates):
        quantile = float(rng.uniform(*regime.imbalance_range))
        labels.append(_labels_from_scores(score_function(rng, x), quantile))
    return x, np.stack(labels), {"features": features, "family": family}


# --------------------------------------------------------------------------- #
# Episode assembly
# --------------------------------------------------------------------------- #
def available_support_sizes(max_support_size: int) -> tuple[int, ...]:
    """Support sizes at or below a compute budget, keeping the smallest as a floor."""
    allowed = tuple(size for size in SUPPORT_SIZES if size <= max_support_size)
    if not allowed:
        raise ValueError(f"max_support_size={max_support_size} excludes every supported size {SUPPORT_SIZES}.")
    return allowed


def _candidate_probabilities(labels: np.ndarray, noise: float) -> np.ndarray:
    """``eta`` for the opposite class and ``1 - eta`` for the selected class."""
    return noise + (1.0 - 2.0 * noise) * labels.astype(np.float64)


def exact_candidate_posterior(
    support_labels: np.ndarray,
    candidate_support_positive: np.ndarray,
    *,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Exact posterior ``rho_h`` from support labels under the known noise model."""
    probability = np.clip(candidate_support_positive, 1e-6, 1 - 1e-6)
    log_likelihood = (
        support_labels[None, :] * np.log(probability) + (1 - support_labels[None, :]) * np.log1p(-probability)
    ).sum(axis=1)
    if prior is not None:
        log_likelihood = log_likelihood + np.log(np.clip(prior, 1e-12, None))
    log_likelihood = log_likelihood - log_likelihood.max()
    weights = np.exp(log_likelihood)
    return weights / weights.sum()


def _select_rows(
    labels: np.ndarray,
    condition: str,
    support_size: int,
    query_count: int,
    rng: np.random.Generator,
    identifying_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pick support and query rows, varying only how much evidence identifies.

    Queries are always the highest-disagreement rows and the support is always
    drawn from the same low-disagreement base region, so ambiguous and
    identifiable episodes differ *only* in ``identifying_fraction`` -- the share
    of support rows on which the candidates genuinely disagree.

    Selecting the support from a different region per condition, as an earlier
    version did, made the conditions differ in support difficulty as well as in
    evidence content.  That is a family-specific surface cue rather than a fact
    about the posterior, and an encoder probe showed heads latching onto it and
    then *inverting* on held-out function families.

    Returns the support indices, the query indices, and how many support rows
    are identifying, which the caller needs in order to force the rest to agree.
    """
    if not 0.0 <= identifying_fraction <= 1.0:
        raise ValueError("identifying_fraction must lie in [0, 1].")
    if condition == "noisy":
        # Every candidate is identical here, so disagreement is zero everywhere
        # and any ordering is arbitrary.
        order = rng.permutation(labels.shape[1])
        return order[:support_size], order[support_size : support_size + query_count], 0
    disagreement = labels.astype(np.float64).var(axis=0)
    descending = np.argsort(-disagreement, kind="stable")
    query_indices = descending[:query_count]
    remaining = descending[query_count:]
    identifying_count = int(round(identifying_fraction * support_size))
    identifying_count = min(identifying_count, max(0, remaining.size - support_size))
    agreeing_count = support_size - identifying_count
    # ``remaining`` is sorted by descending disagreement: the head identifies
    # the candidate, the tail is uninformative about it.
    identifying = remaining[:identifying_count]
    agreeing = remaining[remaining.size - agreeing_count :] if agreeing_count else remaining[:0]
    support_indices = np.concatenate((agreeing, identifying))
    return support_indices, query_indices, identifying_count


def _build_item(
    regime: EpisodeRegime,
    rng: np.random.Generator,
    *,
    family: str,
    condition: str,
    num_candidates: int,
    support_size: int,
    query_count: int,
    noise: float,
    features: int,
    identifying_fraction: float = 0.0,
    extra_support: int = 0,
) -> dict[str, np.ndarray]:
    rows = max(support_size + query_count + extra_support + 16, 4 * (support_size + query_count))
    x, labels, _info = sample_candidate_pool(
        regime, rng, family=family, num_candidates=num_candidates, rows=rows, features=features
    )
    if condition == "noisy":
        # Candidates agree everywhere, so all remaining uncertainty is aleatoric.
        labels = np.repeat(labels[:1], num_candidates, axis=0)
    support_indices, query_indices, identifying_count = _select_rows(
        labels, condition, support_size + extra_support, query_count, rng, identifying_fraction
    )
    # The non-identifying part of the support is made exactly uninformative, so
    # the only evidence about which candidate is active is the identifying rows.
    # With many candidates, naturally agreeing rows are too rare to select.
    agreeing_indices = support_indices[: support_indices.size - identifying_count]
    labels[:, agreeing_indices] = labels[0, agreeing_indices]
    candidate_support = _candidate_probabilities(labels[:, support_indices], noise)
    candidate_query = _candidate_probabilities(labels[:, query_indices], noise)
    truth = int(rng.integers(num_candidates))
    support_clean = labels[truth, support_indices]
    query_clean = labels[truth, query_indices]
    support_y = np.logical_xor(support_clean, rng.random(support_clean.shape) < noise).astype(np.float32)
    query_y = np.logical_xor(query_clean, rng.random(query_clean.shape) < noise).astype(np.int64)
    posterior = exact_candidate_posterior(support_y.astype(np.int64), candidate_support)
    return {
        "support_x": x[support_indices].astype(np.float32),
        "support_y": support_y,
        "query_x": x[query_indices].astype(np.float32),
        "query_y": query_y,
        "candidate_support_positive": candidate_support.astype(np.float32),
        "candidate_query_positive": candidate_query.astype(np.float32),
        "posterior": posterior.astype(np.float32),
        "truth": np.asarray(truth),
    }


def _stack(items: list[dict[str, np.ndarray]], key: str) -> torch.Tensor:
    return torch.from_numpy(np.stack([item[key] for item in items]))


def _assemble(
    items: list[dict[str, np.ndarray]],
    *,
    noise: float,
    condition: str,
    family: str,
    metadata: dict[str, Any],
) -> ContinuousEpisode:
    return ContinuousEpisode(
        _stack(items, "support_x"),
        _stack(items, "support_y"),
        _stack(items, "query_x"),
        _stack(items, "query_y"),
        _stack(items, "candidate_query_positive"),
        _stack(items, "candidate_support_positive"),
        _stack(items, "posterior"),
        noise,
        condition,
        family,
        metadata,
    )


def sample_episode(
    rng: np.random.Generator,
    *,
    regime: EpisodeRegime = TRAIN_REGIME,
    condition: str = "ambiguous",
    batch_size: int = 2,
    num_candidates: int | None = None,
    support_size: int | None = None,
    query_count: int | None = None,
    noise: float | None = None,
    family: str | None = None,
    device: torch.device | str = "cpu",
    max_support_size: int = max(SUPPORT_SIZES),
) -> ContinuousEpisode:
    """Draw one batch of episodes with an exact candidate posterior."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    family = family or str(rng.choice(regime.families))
    num_candidates = num_candidates or int(rng.choice(CANDIDATE_COUNTS))
    support_size = support_size or int(rng.choice(available_support_sizes(max_support_size)))
    query_count = query_count or int(rng.integers(4, 9))
    if not 4 <= query_count <= 8:
        raise ValueError("query_count must lie between four and eight for exact joint enumeration.")
    noise = float(rng.choice(NOISE_LEVELS)) if noise is None else float(noise)
    effective_noise = max(noise, MIN_EFFECTIVE_NOISE)
    # One feature count per batch keeps every episode in the batch stackable.
    features = int(rng.integers(regime.min_features, regime.max_features + 1))
    identifying_fraction = (
        float(rng.uniform(*IDENTIFIABLE_IDENTIFYING_FRACTION))
        if condition == "identifiable"
        else AMBIGUOUS_IDENTIFYING_FRACTION
    )
    items = [
        _build_item(
            regime,
            rng,
            family=family,
            condition=condition,
            num_candidates=num_candidates,
            support_size=support_size,
            query_count=query_count,
            noise=effective_noise,
            features=features,
            identifying_fraction=identifying_fraction,
        )
        for _ in range(batch_size)
    ]
    episode = _assemble(
        items,
        noise=effective_noise,
        condition=condition,
        family=family,
        metadata={
            "num_candidates": num_candidates,
            "support_size": support_size,
            "query_count": query_count,
            "features": features,
            "identifying_fraction": identifying_fraction,
        },
    )
    return episode.to(device)


def sample_paired_episode(
    rng: np.random.Generator,
    *,
    regime: EpisodeRegime = TRAIN_REGIME,
    batch_size: int = 2,
    num_candidates: int | None = None,
    support_size: int | None = None,
    query_count: int | None = None,
    noise: float | None = None,
    family: str | None = None,
    device: torch.device | str = "cpu",
    identifying_fraction: float = 0.5,
    max_support_size: int = max(SUPPORT_SIZES),
) -> tuple[ContinuousEpisode, ContinuousEpisode]:
    """Return a matched pair differing only in how identifying the support is.

    Both arms share the candidate functions, the query rows, the support size,
    and the support region.  The first arm has no identifying support rows and
    the second has ``identifying_fraction`` of them, so a correct posterior must
    not increase its mutual information from the first to the second.  Holding
    the support size fixed makes this a sharper test than the earlier
    prefix-versus-extension construction, in which the arms also differed in
    how much data they had.
    """
    if not 0 < identifying_fraction <= 1:
        raise ValueError("identifying_fraction must lie in (0, 1].")
    family = family or str(rng.choice(regime.families))
    num_candidates = num_candidates or int(rng.choice(CANDIDATE_COUNTS))
    support_size = support_size or int(rng.choice(available_support_sizes(max_support_size)))
    query_count = query_count or int(rng.integers(4, 9))
    noise = float(rng.choice(NOISE_LEVELS)) if noise is None else float(noise)
    effective_noise = max(noise, MIN_EFFECTIVE_NOISE)
    features = int(rng.integers(regime.min_features, regime.max_features + 1))

    short_items: list[dict[str, np.ndarray]] = []
    long_items: list[dict[str, np.ndarray]] = []
    for _ in range(batch_size):
        rows = max(support_size + query_count + 16, 4 * (support_size + query_count))
        x, labels, _info = sample_candidate_pool(
            regime, rng, family=family, num_candidates=num_candidates, rows=rows, features=features
        )
        truth = int(rng.integers(num_candidates))
        for fraction, bucket in ((0.0, short_items), (identifying_fraction, long_items)):
            # Each arm gets its own copy so that forcing one arm's agreeing rows
            # to agree cannot leak into the other.
            arm_labels = labels.copy()
            support_indices, query_indices, identifying_count = _select_rows(
                arm_labels, "paired", support_size, query_count, rng, fraction
            )
            agreeing = support_indices[: support_indices.size - identifying_count]
            arm_labels[:, agreeing] = arm_labels[0, agreeing]
            candidate_support = _candidate_probabilities(arm_labels[:, support_indices], effective_noise)
            candidate_query = _candidate_probabilities(arm_labels[:, query_indices], effective_noise)
            clean = arm_labels[truth, support_indices]
            support_y = np.logical_xor(clean, rng.random(clean.shape) < effective_noise).astype(np.float32)
            query_clean = arm_labels[truth, query_indices]
            query_y = np.logical_xor(
                query_clean, rng.random(query_clean.shape) < effective_noise
            ).astype(np.int64)
            bucket.append(
                {
                    "support_x": x[support_indices].astype(np.float32),
                    "support_y": support_y,
                    "query_x": x[query_indices].astype(np.float32),
                    "query_y": query_y,
                    "candidate_support_positive": candidate_support.astype(np.float32),
                    "candidate_query_positive": candidate_query.astype(np.float32),
                    "posterior": exact_candidate_posterior(
                        support_y.astype(np.int64), candidate_support
                    ).astype(np.float32),
                    "truth": np.asarray(truth),
                }
            )
    metadata = {
        "num_candidates": num_candidates,
        "support_size": support_size,
        "query_count": query_count,
        "identifying_fraction": identifying_fraction,
    }
    short = _assemble(short_items, noise=effective_noise, condition="paired", family=family, metadata=metadata)
    long = _assemble(long_items, noise=effective_noise, condition="paired", family=family, metadata=metadata)
    return short.to(device), long.to(device)


def random_label_episode(
    rng: np.random.Generator,
    *,
    batch_size: int = 2,
    support_size: int = 64,
    query_count: int = 4,
    features: int = 4,
    device: torch.device | str = "cpu",
) -> ContinuousEpisode:
    """Diagnostic-only episodes with independent random labels.

    These are never training batches.  A correct model should report high
    expected conditional entropy and low epistemic mutual information.
    """
    support_x = rng.normal(size=(batch_size, support_size, features)).astype(np.float32)
    query_x = rng.normal(size=(batch_size, query_count, features)).astype(np.float32)
    support_y = rng.integers(0, 2, size=(batch_size, support_size)).astype(np.float32)
    query_y = rng.integers(0, 2, size=(batch_size, query_count)).astype(np.int64)
    half = np.full((batch_size, 1, query_count), 0.5, dtype=np.float32)
    half_support = np.full((batch_size, 1, support_size), 0.5, dtype=np.float32)
    return ContinuousEpisode(
        torch.from_numpy(support_x),
        torch.from_numpy(support_y),
        torch.from_numpy(query_x),
        torch.from_numpy(query_y),
        torch.from_numpy(half),
        torch.from_numpy(half_support),
        torch.ones(batch_size, 1),
        0.5,
        "random_label_diagnostic",
        "random",
    ).to(device)


def curriculum_condition(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Draw an episode condition from the balanced curriculum."""
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("curriculum weights must sum to one.")
    draw = float(rng.random())
    cumulative = 0.0
    for condition in CONDITIONS:
        cumulative += weights[condition]
        if draw < cumulative:
            return condition
    return CONDITIONS[-1]
