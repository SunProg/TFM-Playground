"""Related binary mechanisms with soft, independent, or persistent assignments.

All likelihoods and regime tags are diagnostic. Only X, support Y, and
episode-local group codes enter a model. Group codes never encode regime IDs.
"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.special import expit, softmax

FAMILIES = ("original", "shared_rule", "soft_gate", "independent", "persistent")


@dataclass(frozen=True)
class V3Config:
    support_size: int = 128
    query_size: int = 32
    min_features: int = 2
    max_features: int = 12
    num_groups: int = 8
    calibration_size: int = 256
    num_regimes: int = 2
    separation: float = 1.0
    gate_strength: float = 1.0
    imbalance_ratio: float = 0.5

    def __post_init__(self):
        if self.support_size < 2 or self.query_size < 1:
            raise ValueError("Need at least two support rows and one query row.")
        if not 2 <= self.min_features <= self.max_features:
            raise ValueError("Feature bounds must satisfy 2 <= min <= max.")
        if not 1 <= self.num_regimes <= 4 or self.num_groups < 2:
            raise ValueError("Need 1-4 regimes and at least two group codes.")
        if self.separation < 0 or self.gate_strength < 0 or self.calibration_size < 16:
            raise ValueError("Invalid separation, gate strength, or calibration size.")
        if not 0 < self.imbalance_ratio <= 1:
            raise ValueError("imbalance_ratio must be in (0, 1].")


@dataclass
class V3Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    support_z: np.ndarray
    query_z: np.ndarray
    support_groups: np.ndarray
    query_groups: np.ndarray
    support_gate: np.ndarray
    query_gate: np.ndarray
    support_components: np.ndarray | None
    query_components: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def latent_inputs(self):
        return self.support_x, self.support_y, self.query_x

    def tensor_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (*self.latent_inputs(), self.query_y):
            digest.update(value.contiguous().numpy().tobytes())
        return digest.hexdigest()


def group_posterior(
    prior: np.ndarray, components: np.ndarray, labels: np.ndarray, groups: np.ndarray, num_groups: int
) -> np.ndarray:
    """Known-mechanism posterior, using visible support labels only."""
    p = np.clip(components, 1e-12, 1 - 1e-12)
    likelihood = labels[:, None] * np.log(p) + (1 - labels[:, None]) * np.log1p(-p)
    logs = np.tile(np.log(np.clip(prior, 1e-12, 1)), (num_groups, 1))
    np.add.at(logs, groups, likelihood)
    return softmax(logs, axis=-1)


def oracle_probabilities(episode: V3Episode) -> dict[str, np.ndarray]:
    """Parameter-aware references, never conditioned on query Y or query Z.

    The separate privileged output uses query Z explicitly and is labeled as
    such. Original-prior likelihoods are unavailable and are not fabricated.
    """
    if episode.query_components is None:
        return {}
    gate = episode.query_gate
    if episode.metadata["family"] == "persistent":
        posterior = group_posterior(
            np.asarray(episode.metadata["weights"]),
            episode.support_components,
            episode.support_y.numpy().reshape(-1),
            episode.support_groups,
            episode.metadata["num_groups"],
        )
        gate = posterior[episode.query_groups]
    components = episode.query_components
    return {
        "marginal": (gate * components).sum(-1),
        "without_group_evidence": (episode.query_gate * components).sum(-1),
        "privileged_regime": components[np.arange(len(components)), episode.query_z],
        "query_responsibilities": gate,
    }


def support_responsibilities(episode: V3Episode) -> np.ndarray:
    """Parameter-aware support assignment reference, including visible Y."""
    if episode.support_components is None:
        raise ValueError("Original-prior component likelihoods are unavailable.")
    labels = episode.support_y.numpy().reshape(-1)
    if episode.metadata["family"] == "persistent":
        posterior = group_posterior(
            np.asarray(episode.metadata["weights"]),
            episode.support_components,
            labels,
            episode.support_groups,
            episode.metadata["num_groups"],
        )
        return posterior[episode.support_groups]
    p = np.clip(episode.support_components, 1e-12, 1 - 1e-12)
    log_likelihood = labels[:, None] * np.log(p) + (1 - labels[:, None]) * np.log1p(-p)
    return softmax(np.log(np.clip(episode.support_gate, 1e-12, 1)) + log_likelihood, axis=-1)


def _features(rng, count, mechanism):
    """A triangular nonlinear SCM; fixed mechanisms, independent row noise."""
    weights, offsets, scale = mechanism
    x = rng.normal(size=(count, len(offsets))) * scale + offsets
    for j in range(1, x.shape[1]):
        parents = x[:, :j] @ weights[:j, j]
        x[:, j] += 0.6 * parents + 0.4 * np.tanh(parents)
    return x


def _basis(x):
    return np.concatenate((x, np.tanh(x), x * np.roll(x, 1, axis=1), (x > 0).astype(float)), axis=1)


def _codes(rng, count, num_groups):
    return rng.integers(num_groups, size=count)


def _model_x(x, groups, config, code_permutation):
    padded = np.zeros((len(x), config.max_features + config.num_groups), dtype=np.float32)
    padded[:, : x.shape[1]] = x
    padded[:, config.max_features :] = np.eye(config.num_groups)[groups][:, code_permutation]
    return torch.from_numpy(padded)[None]


def sample_episode(config: V3Config, *, family: str, seed: int) -> V3Episode:
    if family not in FAMILIES[1:]:
        raise ValueError("Use OriginalPrior for original episodes; unknown v3 family.")
    # The support stream and mechanism do not depend on query count. Each
    # random stream has one role; row count changes cannot move another stream.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    parameter_rng, calibration_rng, sx_rng, qx_rng, sy_rng, qy_rng, sg_rng, qg_rng, z_rng = streams
    d = int(parameter_rng.integers(config.min_features, config.max_features + 1))
    k = config.num_regimes
    weights = np.triu(parameter_rng.normal(0, 0.4 / np.sqrt(d), (d, d)), 1)
    weights *= parameter_rng.random((d, d)) < 0.4
    feature_mechanism = (weights, parameter_rng.normal(0, 0.5, d), parameter_rng.uniform(0.5, 1.5, d))
    calibration = _features(calibration_rng, config.calibration_size, feature_mechanism)
    location, scale = calibration.mean(0), calibration.std(0).clip(0.1)
    calibration = (calibration - location) / scale
    support = (_features(sx_rng, config.support_size, feature_mechanism) - location) / scale
    query = (_features(qx_rng, config.query_size, feature_mechanism) - location) / scale
    phi_cal = _basis(calibration)
    common = parameter_rng.normal(size=4 * d)
    # Some tasks are mostly linear, others additive, interacting, or thresholded.
    mechanism_family = int(parameter_rng.integers(4))
    common *= 0.15
    common[mechanism_family * d : (mechanism_family + 1) * d] *= 1 / 0.15
    deltas = np.zeros((4 * d, k))
    for regime in range(k):
        columns = parameter_rng.choice(d, size=int(parameter_rng.integers(1, min(3, d) + 1)), replace=False)
        for column in columns:
            term = int(parameter_rng.integers(4)) * d + column
            deltas[term, regime] = parameter_rng.normal()
    deltas -= deltas.mean(axis=1, keepdims=True)
    base_cal, delta_cal = phi_cal @ common, phi_cal @ deltas
    base_center, base_scale = base_cal.mean(), max(base_cal.std(), 0.1)
    delta_scale = max(np.sqrt(np.mean(delta_cal**2)), 0.1)
    intercept = float(parameter_rng.normal(0, 0.8))
    base_strength = float(parameter_rng.uniform(0.8, 2.0))
    alpha = 0.0 if family == "shared_rule" else config.separation

    def probabilities(x):
        phi = _basis(x)
        base = base_strength * (phi @ common - base_center) / base_scale + intercept
        return expit(base[:, None] + alpha * (phi @ deltas) / delta_scale)

    prior = np.exp(np.linspace(0, np.log(config.imbalance_ratio), k))
    prior = prior[parameter_rng.permutation(k)]
    prior /= prior.sum()
    gate_weights = parameter_rng.normal(size=(4 * d, k))
    gate_cal = phi_cal @ gate_weights
    gate_center, gate_scale = gate_cal.mean(0), gate_cal.std(0).clip(0.1)

    def gate(x):
        if family in ("independent", "persistent"):
            return np.tile(prior, (len(x), 1))
        scores = (_basis(x) @ gate_weights - gate_center) / gate_scale
        return softmax(config.gate_strength * scores + np.log(prior), axis=-1)

    sg = _codes(sg_rng, config.support_size, config.num_groups)
    qg = _codes(qg_rng, config.query_size, config.num_groups)
    group_z = z_rng.choice(k, size=config.num_groups, p=prior)
    sp, qp = probabilities(support), probabilities(query)
    spi, qpi = gate(support), gate(query)

    def outcomes(p, pi, groups, rng):
        if family == "persistent":
            z = group_z[groups]
        else:
            z = (rng.random(len(p))[:, None] > pi.cumsum(-1)).sum(-1).clip(max=k - 1)
        y = (rng.random(len(p)) < p[np.arange(len(p)), z]).astype(np.int64)
        return z, y

    sz, sy = outcomes(sp, spi, sg, sy_rng)
    qz, qy = outcomes(qp, qpi, qg, qy_rng)
    code_permutation = parameter_rng.permutation(config.num_groups)
    mechanism_hash = hashlib.sha256()
    for value in (*feature_mechanism, location, scale, common, deltas, gate_weights, prior):
        mechanism_hash.update(np.asarray(value).tobytes())
    mechanism_hash.update(np.asarray([intercept, base_strength, alpha, config.gate_strength]).tobytes())
    episode = V3Episode(
        _model_x(support, sg, config, code_permutation),
        torch.from_numpy(sy.astype(np.float32))[None],
        _model_x(query, qg, config, code_permutation),
        torch.from_numpy(qy)[None],
        sz,
        qz,
        sg,
        qg,
        spi,
        qpi,
        sp,
        qp,
        {
            "family": family,
            "seed": seed,
            "num_regimes": k,
            "features": d,
            "num_groups": config.num_groups,
            "weights": prior.tolist(),
            "separation": alpha,
            "gate_strength": config.gate_strength,
            "mechanism_family": mechanism_family,
            "mechanism_hash": mechanism_hash.hexdigest(),
            "support_counts": np.bincount(sz, minlength=k).tolist(),
            "query_counts": np.bincount(qz, minlength=k).tolist(),
        },
    )
    episode.metadata["tensor_hash"] = episode.tensor_hash()
    return episode


@contextmanager
def preserved_cpu_rng(seed: int):
    """TabICL uses global RNGs; generation must not consume model randomness."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


class OriginalPrior:
    """Original mix_scm targets, with identical nuisance-code augmentation.

    No substitute labels or fallback prior are allowed. Retry only nonfinite
    draws and record the retry count. Oracle likelihoods are unavailable.
    """

    def __init__(self, config: V3Config):
        from tfmplayground.external_priors import TabICLPriorDataLoader

        self.config = config
        with preserved_cpu_rng(713):
            self.loader = TabICLPriorDataLoader(
                num_steps=1,
                batch_size=1,
                num_datapoints_min=config.support_size + config.query_size,
                num_datapoints_max=config.support_size + config.query_size + 1,
                min_features=config.min_features,
                max_features=config.max_features,
                max_num_classes=2,
                device=torch.device("cpu"),
                prior_type="mix_scm",
                min_train_size=config.support_size,
                max_train_size=config.support_size + 1,
            )
        self.loader.pd.prior.n_jobs = 1

    def sample(self, seed: int) -> V3Episode:
        config = self.config
        for attempt in range(32):
            with preserved_cpu_rng(seed + attempt * 10_000_019), torch.no_grad():
                batch = next(iter(self.loader))
            x, y = batch["x"][0].numpy(), batch["y"][0].numpy().reshape(-1)
            if np.isfinite(x).all() and np.isfinite(y).all():
                break
        else:
            raise RuntimeError("Original prior produced 32 nonfinite draws; refusing to replace the prior.")
        if not np.isin(y, (0, 1)).all() or batch["train_test_split_index"] != config.support_size:
            raise ValueError("Original prior violated the requested binary label/split contract.")
        split = config.support_size
        rng = np.random.default_rng(seed)
        sg = _codes(rng, split, config.num_groups)
        qg = _codes(rng, config.query_size, config.num_groups)
        permutation = rng.permutation(config.num_groups)
        episode = V3Episode(
            _model_x(x[:split], sg, config, permutation),
            torch.from_numpy(y[:split].astype(np.float32))[None],
            _model_x(x[split:], qg, config, permutation),
            torch.from_numpy(y[split:].astype(np.int64))[None],
            np.zeros(split, dtype=int),
            np.zeros(config.query_size, dtype=int),
            sg,
            qg,
            np.ones((split, 1)),
            np.ones((config.query_size, 1)),
            None,
            None,
            {
                "family": "original",
                "seed": seed,
                "num_regimes": 1,
                "features": x.shape[1],
                "num_groups": config.num_groups,
                "nonfinite_retries": attempt,
                "prior_type": "mix_scm",
                "group_codes": "independent_nuisance",
                "oracle_available": False,
            },
        )
        episode.metadata["tensor_hash"] = episode.tensor_hash()
        return episode
