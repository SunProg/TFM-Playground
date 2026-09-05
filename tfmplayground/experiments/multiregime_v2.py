"""Generalized shared-covariate multi-regime binary-classification episodes.

This module is deliberately independent of the original two-regime
contamination generator.  A single feature table is sampled for an episode,
all regime functions are evaluated on that table, and the diagnostic regime
labels are kept outside the latent model input.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch

Backend = Literal["analytic", "tabicl_scm"]
SingleRegimeSource = Literal["legacy", "matched"]
DIFFERENCE_COMPONENTS = ("coefficients", "nonlinear", "feature_subset", "decision_boundary")


@dataclass(frozen=True)
class RegimeGeneratorConfig:
    """Configuration for one family of reproducible v2 episodes."""

    backend: Backend = "analytic"
    max_regimes: int = 4
    regime_separation: float = 1.0
    imbalance_ratio: float = 1.0
    gate_strength: float = 1.0
    difference_components: tuple[str, ...] = ("coefficients",)
    support_size: int = 128
    query_size: int = 256
    min_features: int = 2
    max_features: int = 12
    label_noise: float = 0.0
    seed: int = 2402
    single_regime_source: SingleRegimeSource = "matched"

    def __post_init__(self) -> None:
        if self.backend not in ("analytic", "tabicl_scm"):
            raise ValueError("backend must be 'analytic' or 'tabicl_scm'.")
        if self.max_regimes < 1:
            raise ValueError("max_regimes must be positive.")
        if self.regime_separation < 0:
            raise ValueError("regime_separation must be non-negative.")
        if not 0 < self.imbalance_ratio <= 1:
            raise ValueError("imbalance_ratio must lie in (0, 1].")
        if self.gate_strength < 0:
            raise ValueError("gate_strength must be non-negative.")
        if self.support_size < 1 or self.query_size < 1:
            raise ValueError("support_size and query_size must be positive.")
        if self.min_features < 1 or self.max_features < self.min_features:
            raise ValueError("feature bounds must satisfy 1 <= min_features <= max_features.")
        if not 0 <= self.label_noise <= 0.5:
            raise ValueError("label_noise must lie in [0, 0.5].")
        if self.single_regime_source not in ("legacy", "matched"):
            raise ValueError("single_regime_source must be 'legacy' or 'matched'.")
        components = tuple(dict.fromkeys(self.difference_components))
        if not components:
            raise ValueError("difference_components cannot be empty.")
        unknown = sorted(set(components) - set(DIFFERENCE_COMPONENTS))
        if unknown:
            raise ValueError(f"Unknown difference components: {unknown}.")
        if self.backend == "tabicl_scm" and components != ("coefficients",):
            raise ValueError(
                "tabicl_scm uses independent target networks; analytic components "
                "nonlinear, feature_subset, and decision_boundary are unsupported."
            )
        object.__setattr__(self, "difference_components", components)


@dataclass
class RegimeEpisode:
    """One v2 episode; regime tensors are diagnostic and never latent inputs.

    Model tensors include a leading batch dimension of one.  Counterfactual
    arrays are padded on their regime axis to ``max_regimes``.
    """

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    support_z: torch.Tensor
    query_z: torch.Tensor
    active_regime_mask: torch.Tensor
    support_gate_probabilities: torch.Tensor
    query_gate_probabilities: torch.Tensor
    counterfactual_support_probabilities: torch.Tensor
    counterfactual_query_probabilities: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_regimes(self) -> int:
        return int(self.active_regime_mask.sum().item())

    def latent_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The complete unsupervised model view; notably, it contains no z."""
        return self.support_x, self.support_y, self.query_x

    def oracle_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append fixed-width one-hot regime columns for the oracle control."""
        width = self.active_regime_mask.shape[-1]
        support = torch.nn.functional.one_hot(self.support_z.long(), width).to(self.support_x.dtype)
        query = torch.nn.functional.one_hot(self.query_z.long(), width).to(self.query_x.dtype)
        return torch.cat((self.support_x, support), -1), self.support_y, torch.cat((self.query_x, query), -1)

    def to(self, device: str | torch.device) -> RegimeEpisode:
        values = {
            name: getattr(self, name).to(device)
            for name in (
                "support_x",
                "support_y",
                "query_x",
                "query_y",
                "support_z",
                "query_z",
                "active_regime_mask",
                "support_gate_probabilities",
                "query_gate_probabilities",
                "counterfactual_support_probabilities",
                "counterfactual_query_probabilities",
            )
        }
        return RegimeEpisode(**values, metadata=dict(self.metadata))


def stack_regime_episodes(episodes: Sequence[RegimeEpisode]) -> RegimeEpisode:
    """Stack fixed-shape episodes into one model batch.

    Every generator episode uses padded feature and regime axes, so a batch can
    be formed without exposing diagnostic ``z`` to the latent model inputs.
    Per-episode metadata remains owned by the caller (training writes it as
    JSONL); the returned container is intended for tensor computation only.
    """
    if not episodes:
        raise ValueError("episodes must not be empty.")
    fields = (
        "support_x",
        "support_y",
        "query_x",
        "query_y",
        "support_z",
        "query_z",
        "active_regime_mask",
        "support_gate_probabilities",
        "query_gate_probabilities",
        "counterfactual_support_probabilities",
        "counterfactual_query_probabilities",
    )
    values = {}
    for field_name in fields:
        tensors = [getattr(episode, field_name) for episode in episodes]
        if any(tensor.shape[1:] != tensors[0].shape[1:] for tensor in tensors[1:]):
            raise ValueError(f"Cannot stack episodes with different {field_name} shapes.")
        values[field_name] = torch.cat(tensors, dim=0)
    return RegimeEpisode(**values, metadata={"batched": True, "batch_size": len(episodes)})


def requested_regime_weights(num_regimes: int, imbalance_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric target weights in random regime order."""
    if num_regimes < 1:
        raise ValueError("num_regimes must be positive.")
    if not 0 < imbalance_ratio <= 1:
        raise ValueError("imbalance_ratio must lie in (0, 1].")
    weights = np.exp(np.linspace(0.0, math.log(imbalance_ratio), num_regimes))
    weights = weights[rng.permutation(num_regimes)]
    return weights / weights.sum()


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values)
        low, high = np.percentile(finite, (1.0, 99.0))
        values = np.nan_to_num(
            values,
            nan=float(np.median(finite)),
            posinf=float(high),
            neginf=float(low),
        )
        values = np.clip(values, low, high)
    centered = values - values.mean()
    scale = values.std()
    return centered / scale if scale > 1e-12 else centered


def _calibrate_gate_intercepts(
    logits: np.ndarray, target: np.ndarray, *, tolerance: float = 1e-10, max_iterations: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """Find softmax intercepts whose empirical column means equal ``target``."""
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Gate logits must be finite after standardization.")
    # Avoid numerically saturated softmax rows from heavy-tailed SCM draws;
    # this bound is far outside the range needed for a calibrated probability
    # and keeps the Newton Jacobian well-conditioned.
    logits = np.clip(logits, -30.0, 30.0)
    if logits.shape[1] == 1:
        return np.zeros(1, dtype=np.float64), np.ones_like(logits, dtype=np.float64)
    # Fix the final intercept to zero (the softmax is invariant to a shared
    # additive constant) and solve the remaining K-1 equations with Newton
    # steps.  The former multiplicative fixed-point update can oscillate for
    # highly concentrated SCM gates.
    intercepts = np.log(target)
    intercepts -= intercepts[-1]
    for _ in range(max_iterations):
        shifted = logits + intercepts[None, :]
        shifted -= shifted.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        realized = probability.mean(axis=0)
        error = realized[:-1] - target[:-1]
        if np.max(np.abs(error)) <= tolerance:
            return intercepts - intercepts.mean(), probability
        p_free = probability[:, :-1]
        jacobian = np.diag(realized[:-1]) - (p_free.T @ p_free) / len(probability)
        jacobian.flat[:: len(error) + 1] += 1e-12
        try:
            step = np.linalg.solve(jacobian, -error)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
        old_error = float(np.max(np.abs(error)))
        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = intercepts.copy()
            candidate[:-1] += damping * step
            shifted_candidate = logits + candidate[None, :]
            shifted_candidate -= shifted_candidate.max(axis=1, keepdims=True)
            probability_candidate = np.exp(shifted_candidate)
            probability_candidate /= probability_candidate.sum(axis=1, keepdims=True)
            candidate_error = probability_candidate.mean(axis=0)[:-1] - target[:-1]
            if float(np.max(np.abs(candidate_error))) < old_error:
                intercepts, probability = candidate, probability_candidate
                accepted = True
                break
        if not accepted:
            # A final small step prevents numerical plateaus from spinning
            # for thousands of iterations; the explicit convergence check
            # below still guards the requested tolerance.
            intercepts[:-1] += 0.01 * step
    raise RuntimeError("Softmax gate intercept calibration did not converge.")


def _pairwise_rms(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.sqrt(np.mean((values[left] - values[right]) ** 2)) for left in range(len(values)) for right in range(left)],
        dtype=np.float64,
    )


def _normalize_deltas(raw: np.ndarray) -> tuple[np.ndarray, float]:
    """Center across regimes and give the mean pairwise RMS unit scale."""
    if raw.shape[0] == 1:
        return np.zeros_like(raw), 0.0
    centered = raw - raw.mean(axis=0, keepdims=True)
    distance = float(_pairwise_rms(centered).mean())
    if distance <= 1e-12:
        # A rare SCM draw can collapse all target-network outputs (for
        # example after an overflow fallback).  Treat it as an explicitly
        # zero-separation episode rather than aborting a locked benchmark.
        return np.zeros_like(raw), 0.0
    return centered / distance, distance


def _analytic_scores(
    x: np.ndarray, num_regimes: int, components: Sequence[str], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = x.shape[1]
    base_weights = rng.normal(size=features) / math.sqrt(features)
    base = _standardize(x @ base_weights)
    raw = np.zeros((num_regimes, len(x)), dtype=np.float64)
    parameters: dict[str, Any] = {"base_weights": base_weights.tolist(), "components": {}}
    if "coefficients" in components:
        weights = rng.normal(size=(num_regimes, features)) / math.sqrt(features)
        raw += weights @ x.T
        parameters["components"]["coefficients"] = weights.tolist()
    if "nonlinear" in components:
        projections = rng.normal(size=(num_regimes, features)) / math.sqrt(features)
        frequencies = rng.uniform(0.5, 2.5, size=num_regimes)
        phases = rng.uniform(0, 2 * math.pi, size=num_regimes)
        raw += np.sin((x @ projections.T) * frequencies[None, :] + phases[None, :]).T
        parameters["components"]["nonlinear"] = {
            "projections": projections.tolist(),
            "frequencies": frequencies.tolist(),
            "phases": phases.tolist(),
        }
    if "feature_subset" in components:
        masks = np.zeros((num_regimes, features))
        weights = rng.normal(size=(num_regimes, features))
        for regime in range(num_regimes):
            count = int(rng.integers(1, features + 1))
            masks[regime, rng.choice(features, size=count, replace=False)] = 1.0
        sparse = weights * masks / np.sqrt(np.maximum(1.0, masks.sum(axis=1, keepdims=True)))
        raw += sparse @ x.T
        parameters["components"]["feature_subset"] = {"masks": masks.tolist(), "weights": weights.tolist()}
    if "decision_boundary" in components:
        projections = rng.normal(size=(num_regimes, features)) / math.sqrt(features)
        offsets = rng.normal(scale=0.5, size=num_regimes)
        raw += np.square((x @ projections.T) - offsets[None, :]).T
        parameters["components"]["decision_boundary"] = {
            "projections": projections.tolist(),
            "offsets": offsets.tolist(),
        }
    return base, raw, parameters


def _tabicl_scm_scores(
    rows: int, features: int, num_regimes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate multiple TabICL target networks on one raw SCM X sample."""
    from tabicl.prior._dataset import SCMPrior
    from tabicl.prior._mlp_scm import MLPSCM

    from tfmplayground.experiments.prior_bimodal_episodes import (
        PriorBimodalConfig,
        _rng_state,
        _sample_params,
        _set_rng_state,
    )

    prior_config = PriorBimodalConfig(
        initial_support_count=max(2, rows // 2),
        stream_count=0,
        query_count=1,
        min_features=features,
        max_features=features,
        device="cpu",
        prior_type="mlp_scm",
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
        prior_type="mlp_scm",
        n_jobs=1,
        device="cpu",
    )
    outer = _rng_state()
    seeds: list[int] = []
    try:
        parameter_seed = int(rng.integers(0, 2**31 - 1))
        seeds.append(parameter_seed)
        random.seed(parameter_seed)
        np.random.seed(parameter_seed)
        torch.manual_seed(parameter_seed)
        params = _sample_params(prior, prior_config, rows, features, num_classes=2)
        models = []
        for _ in range(num_regimes + 1):
            model_seed = int(rng.integers(0, 2**31 - 1))
            seeds.append(model_seed)
            random.seed(model_seed)
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            # TabICL initializes its parameter tensors with in-place masking.
            # Keep this construction inference-only so newer PyTorch versions
            # do not reject the writes to leaf tensors that require grad.
            with torch.no_grad():
                models.append(MLPSCM(**params))
        x_seed = int(rng.integers(0, 2**31 - 1))
        seeds.append(x_seed)
        random.seed(x_seed)
        np.random.seed(x_seed)
        torch.manual_seed(x_seed)
        nonfinite_counts: list[int] = []
        with torch.no_grad():
            shared_raw_x = models[0].xsampler.sample()
            outputs = []
            for model in models:
                value = model.layers(shared_raw_x)
                array = value.reshape(-1).cpu().numpy().astype(np.float64)[:rows]
                invalid = ~np.isfinite(array)
                nonfinite_counts.append(int(invalid.sum()))
                if invalid.any():
                    # Some TabICL SCM draws overflow their nonlinear layers.
                    # Keep the episode usable while retaining the shared-X
                    # construction: replace only invalid values with robust
                    # finite quantiles from that same network output.
                    finite = array[np.isfinite(array)]
                    if finite.size == 0:
                        array = np.zeros_like(array)
                    else:
                        low, high = np.percentile(finite, (1.0, 99.0))
                        array = np.nan_to_num(
                            array,
                            nan=float(np.median(finite)),
                            posinf=float(high),
                            neginf=float(low),
                        )
                        array = np.clip(array, low, high)
                outputs.append(array)
        raw_x = shared_raw_x.reshape(-1, features).cpu().numpy().astype(np.float64)[:rows]
        base = _standardize(outputs[0])
        raw = np.stack([_standardize(value) for value in outputs[1:]])
        return raw_x, base, raw, {
            "family": "mlp_scm",
            "seeds": seeds,
            "nonfinite_output_values": nonfinite_counts,
        }
    finally:
        _set_rng_state(outer)


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_hash(episode: RegimeEpisode) -> str:
    digest = hashlib.sha256()
    for name in (
        "support_x",
        "support_y",
        "query_x",
        "query_y",
        "support_z",
        "query_z",
        "active_regime_mask",
        "support_gate_probabilities",
        "query_gate_probabilities",
        "counterfactual_support_probabilities",
        "counterfactual_query_probabilities",
    ):
        tensor = getattr(episode, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sample_regime_episode(
    config: RegimeGeneratorConfig,
    *,
    num_regimes: int,
    seed: int | None = None,
    episode_id: str | None = None,
    x_sampler: Callable[[np.random.Generator, int, int], np.ndarray] | None = None,
) -> RegimeEpisode:
    """Sample one matched episode.

    ``single_regime_source='legacy'`` is intentionally handled by
    :func:`legacy_single_regime_batch`; asking this matched generator for that
    mode would otherwise create a deceptively non-legacy tensor stream.
    """
    if not 1 <= num_regimes <= config.max_regimes:
        raise ValueError(f"num_regimes must lie in [1, {config.max_regimes}].")
    if num_regimes == 1 and config.single_regime_source == "legacy":
        raise ValueError("Use legacy_single_regime_batch for byte-identical legacy K=1 sampling.")
    episode_seed = config.seed if seed is None else int(seed)
    rng = np.random.default_rng(episode_seed)
    rows = config.support_size + config.query_size
    features = int(rng.integers(config.min_features, config.max_features + 1))
    if config.backend == "analytic":
        sampler = x_sampler or (lambda generator, count, width: generator.normal(size=(count, width)))
        x_actual = np.asarray(sampler(rng, rows, features), dtype=np.float64)
        if x_actual.shape != (rows, features):
            raise ValueError(f"x_sampler returned {x_actual.shape}, expected {(rows, features)}.")
        base, raw_delta, function_parameters = _analytic_scores(
            x_actual, num_regimes, config.difference_components, rng
        )
    else:
        if x_sampler is not None:
            raise ValueError("x_sampler cannot override the tabicl_scm covariate sampler.")
        x_actual, base, raw_delta, function_parameters = _tabicl_scm_scores(rows, features, num_regimes, rng)
    delta, raw_delta_scale = _normalize_deltas(raw_delta)
    scores = base[None, :] + config.regime_separation * delta
    threshold = float(np.median(scores))
    deterministic = (scores > threshold).astype(np.int64)
    counterfactual_probability = config.label_noise + (1 - 2 * config.label_noise) * deterministic

    weights = requested_regime_weights(num_regimes, config.imbalance_ratio, rng)
    # One episode-level latent linear gate is shared by every regime.  Regime
    # slopes turn that scalar into K logits; only the intercepts are calibrated.
    gate_weights = rng.normal(size=features)
    latent_gate = _standardize(x_actual @ gate_weights)
    gate_slopes = np.linspace(-1.0, 1.0, num_regimes)[rng.permutation(num_regimes)]
    gate_logits = config.gate_strength * latent_gate[:, None] * gate_slopes[None, :]
    gate_intercepts, gate_probability = _calibrate_gate_intercepts(gate_logits, weights)
    z = np.asarray([rng.choice(num_regimes, p=row) for row in gate_probability], dtype=np.int64)
    clean_y = deterministic[z, np.arange(rows)]
    flips = rng.random(rows) < config.label_noise
    y = np.logical_xor(clean_y, flips).astype(np.int64)

    padded_x = np.zeros((rows, config.max_features), dtype=np.float32)
    padded_x[:, :features] = x_actual.astype(np.float32)
    padded_gate = np.zeros((rows, config.max_regimes), dtype=np.float32)
    padded_gate[:, :num_regimes] = gate_probability.astype(np.float32)
    padded_cf = np.zeros((config.max_regimes, rows), dtype=np.float32)
    padded_cf[:num_regimes] = counterfactual_probability.astype(np.float32)
    active = np.zeros(config.max_regimes, dtype=bool)
    active[:num_regimes] = True
    split = config.support_size
    pairwise_scores = _pairwise_rms(scores)
    probability_distances = _pairwise_rms(counterfactual_probability)
    disagreements = np.asarray(
        [
            np.mean(deterministic[left] != deterministic[right])
            for left in range(num_regimes)
            for right in range(left)
        ],
        dtype=np.float64,
    )
    counts = np.bincount(z, minlength=num_regimes)
    class_balance = [float(y[z == regime].mean()) if counts[regime] else None for regime in range(num_regimes)]
    identifier = episode_id or f"seed-{episode_seed}-k-{num_regimes}"
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "episode_id": identifier,
        "seed": episode_seed,
        "backend": config.backend,
        "num_regimes": num_regimes,
        "max_regimes": config.max_regimes,
        "features": features,
        "support_size": config.support_size,
        "query_size": config.query_size,
        "requested_proportions": weights.tolist(),
        "imbalance_ratio": config.imbalance_ratio,
        "realized_gate_proportions": gate_probability.mean(axis=0).tolist(),
        "realized_proportions": (counts / rows).tolist(),
        "counts": counts.tolist(),
        "support_counts": np.bincount(z[:split], minlength=num_regimes).tolist(),
        "query_counts": np.bincount(z[split:], minlength=num_regimes).tolist(),
        "requested_alpha": config.regime_separation,
        "realized_score_distance": float(pairwise_scores.mean()) if pairwise_scores.size else 0.0,
        "realized_probability_distance": (
            float(probability_distances.mean()) if probability_distances.size else 0.0
        ),
        "deterministic_label_disagreement": float(disagreements.mean()) if disagreements.size else 0.0,
        "overall_class_balance": float(y.mean()),
        "per_regime_class_balance": class_balance,
        "label_noise": config.label_noise,
        "gate_strength": config.gate_strength,
        "gate_weights": gate_weights.tolist(),
        "gate_slopes": gate_slopes.tolist(),
        "gate_intercepts": gate_intercepts.tolist(),
        "function_parameters": function_parameters,
        "raw_delta_scale": raw_delta_scale,
        "threshold": threshold,
    }
    metadata["gate_parameter_hash"] = _hash_json(
        {
            "weights": metadata["gate_weights"],
            "slopes": metadata["gate_slopes"],
            "intercepts": metadata["gate_intercepts"],
        }
    )
    metadata["function_parameter_hash"] = _hash_json(function_parameters)
    episode = RegimeEpisode(
        support_x=torch.from_numpy(padded_x[:split])[None],
        support_y=torch.from_numpy(y[:split].astype(np.float32))[None],
        query_x=torch.from_numpy(padded_x[split:])[None],
        query_y=torch.from_numpy(y[split:].astype(np.int64))[None],
        support_z=torch.from_numpy(z[:split])[None],
        query_z=torch.from_numpy(z[split:])[None],
        active_regime_mask=torch.from_numpy(active)[None],
        support_gate_probabilities=torch.from_numpy(padded_gate[:split])[None],
        query_gate_probabilities=torch.from_numpy(padded_gate[split:])[None],
        counterfactual_support_probabilities=torch.from_numpy(padded_cf[:, :split])[None],
        counterfactual_query_probabilities=torch.from_numpy(padded_cf[:, split:])[None],
        metadata=metadata,
    )
    metadata["tensor_hash"] = tensor_hash(episode)
    return episode


def legacy_single_regime_batch(prior: Any) -> Any:
    """Delegate K=1 sampling unchanged to the existing TabICL prior iterator."""
    return next(iter(prior))


def episode_metadata_json(episode: RegimeEpisode) -> str:
    """Stable JSONL representation of all per-episode provenance."""
    return json.dumps(episode.metadata, sort_keys=True, allow_nan=False)


def config_dict(config: RegimeGeneratorConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "Backend",
    "DIFFERENCE_COMPONENTS",
    "RegimeEpisode",
    "RegimeGeneratorConfig",
    "config_dict",
    "episode_metadata_json",
    "legacy_single_regime_batch",
    "requested_regime_weights",
    "sample_regime_episode",
    "stack_regime_episodes",
    "tensor_hash",
]
