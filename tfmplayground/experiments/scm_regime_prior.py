"""Learnable two-regime prior using TabICL MLPSCM mechanisms.

This is a constrained predictive SCM family (observed root causes -> nonlinear
MLP -> target), plus z -> noisy regime cue and z -> choice of target mechanism.
It does not sample the unrestricted TabICL hyperprior. Mechanism selection and
binary thresholds use independent calibration causes, never support/query rows.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v2 import RegimeEpisode, tensor_hash


@dataclass(frozen=True)
class SCMRegimeConfig:
    support_size: int = 128
    query_size: int = 256
    task_features: int = 2
    max_regimes: int = 2
    cue_separation: float = 3.0
    regime_probability: float = 0.5
    num_layers: int = 2
    hidden_dim: int = 8
    init_std: float = 0.5
    label_noise: float = 0.0
    calibration_size: int = 512
    min_disagreement: float = 0.25
    max_disagreement: float = 0.75
    max_attempts: int = 64

    def __post_init__(self):
        for name in ("support_size", "query_size", "task_features", "hidden_dim", "max_attempts"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.max_regimes < 2 or self.num_layers < 2 or self.calibration_size < 32:
            raise ValueError("Require max_regimes >= 2, num_layers >= 2, calibration_size >= 32.")
        if not math.isfinite(self.cue_separation) or self.cue_separation < 0:
            raise ValueError("cue_separation must be finite and non-negative.")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("init_std must be finite and positive.")
        if not 0 < self.regime_probability < 1 or not 0 <= self.label_noise <= 0.5:
            raise ValueError("Require regime_probability in (0,1), label_noise in [0,0.5].")
        if not 0 <= self.min_disagreement <= self.max_disagreement <= 1:
            raise ValueError("Require 0 <= min_disagreement <= max_disagreement <= 1.")


def sample_scm_regime_episode(config: SCMRegimeConfig, *, seed: int) -> RegimeEpisode:
    """Return a standard RegimeEpisode; latent_inputs() excludes all diagnostics."""
    from tabicl.prior._mlp_scm import MLPSCM
    from tabicl.prior._utils import XSampler

    rng = np.random.default_rng(seed)
    # Dedicated streams make support and selected mechanisms invariant to query count.
    cal_seed, sx_seed, qx_seed, sr_seed, qr_seed = rng.integers(0, 2**31 - 1, 5).tolist()

    def causes(count, cause_seed):
        torch.random.default_generator.manual_seed(cause_seed)
        return XSampler(count, config.task_features, sampling="normal", pre_stats=False).sample()

    def scores(models, x):
        return np.stack([model.layers(x).squeeze(-1).numpy() for model in models])

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        calibration = causes(config.calibration_size, cal_seed)
        for _attempt in range(1, config.max_attempts + 1):
            mechanism_seeds = rng.integers(0, 2**31 - 1, 2).tolist()
            models = []
            for mechanism_seed in mechanism_seeds:
                torch.random.default_generator.manual_seed(mechanism_seed)
                models.append(
                    MLPSCM(
                        seq_len=config.calibration_size,
                        num_features=config.task_features,
                        is_causal=False,
                        num_layers=config.num_layers,
                        hidden_dim=config.hidden_dim,
                        mlp_activations=torch.nn.Tanh,
                        init_std=config.init_std,
                        block_wise_dropout=False,
                        mlp_dropout_prob=0.0,
                        noise_std=0.0,
                        pre_sample_noise_std=False,
                        sampling="normal",
                        pre_sample_cause_stats=False,
                        device="cpu",
                    )
                )
            cal_scores = scores(models, calibration)
            if not np.isfinite(cal_scores).all() or np.any(cal_scores.std(axis=1) < 1e-6):
                continue
            thresholds = np.median(cal_scores, axis=1)
            cal_labels = cal_scores > thresholds[:, None]
            disagreement = float((cal_labels[0] != cal_labels[1]).mean())
            if config.min_disagreement <= disagreement <= config.max_disagreement:
                break
        else:
            raise RuntimeError("No non-degenerate SCM pair met the calibration disagreement bounds.")

        support_causes = causes(config.support_size, sx_seed)
        query_causes = causes(config.query_size, qx_seed)
        support_scores = scores(models, support_causes)
        query_scores = scores(models, query_causes)
        digest = hashlib.sha256()
        for model in models:
            for value in model.state_dict().values():
                digest.update(value.numpy().tobytes())

    def pack(x, values, row_seed):
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite SCM outputs; no silent replacement is allowed.")
        chooser = np.random.default_rng(row_seed)
        n = len(x)
        z = (chooser.random(n) < config.regime_probability).astype(np.int64)
        cue = config.cue_separation * (2 * z - 1) + chooser.normal(size=n)
        features = np.column_stack([cue, x.numpy()]).astype(np.float32)
        cf = (values > thresholds[:, None]).astype(np.float32)
        clean = cf[z, np.arange(n)].astype(bool)
        y = np.logical_xor(clean, chooser.random(n) < config.label_noise).astype(np.int64)
        cf = config.label_noise + (1 - 2 * config.label_noise) * cf
        log_odds = 2 * config.cue_separation * cue + math.log(
            config.regime_probability / (1 - config.regime_probability)
        )
        p1 = np.exp(-np.logaddexp(0, -log_odds))
        gate = np.zeros((n, config.max_regimes), dtype=np.float32)
        gate[:, :2] = np.column_stack([1 - p1, p1])
        padded_cf = np.zeros((config.max_regimes, n), dtype=np.float32)
        padded_cf[:2] = cf
        return features, y, z, gate, padded_cf

    sx, sy, sz, sg, sc = pack(support_causes, support_scores, sr_seed)
    qx, qy, qz, qg, qc = pack(query_causes, query_scores, qr_seed)
    active = torch.zeros(1, config.max_regimes, dtype=torch.bool)
    active[:, :2] = True
    episode = RegimeEpisode(
        support_x=torch.from_numpy(sx)[None],
        support_y=torch.from_numpy(sy).float()[None],
        query_x=torch.from_numpy(qx)[None],
        query_y=torch.from_numpy(qy)[None],
        support_z=torch.from_numpy(sz)[None],
        query_z=torch.from_numpy(qz)[None],
        active_regime_mask=active,
        support_gate_probabilities=torch.from_numpy(sg)[None],
        query_gate_probabilities=torch.from_numpy(qg)[None],
        counterfactual_support_probabilities=torch.from_numpy(sc)[None],
        counterfactual_query_probabilities=torch.from_numpy(qc)[None],
        metadata={
            "family": "tabicl_mlp_scm_clustered",
            "seed": int(seed),
            "config": asdict(config),
            "is_causal": False,
            "mechanism_noise_std": 0.0,
            "activation": "Tanh",
            "mechanism_seeds": mechanism_seeds,
            "mechanism_hash": digest.hexdigest(),
            "thresholds": thresholds.tolist(),
            "calibration_disagreement": disagreement,
            "pair_attempts": _attempt,
            "cue_feature_index": 0,
            "query_rule_disagreement": float(
                ((query_scores[0] > thresholds[0]) != (query_scores[1] > thresholds[1])).mean()
            ),
        },
    )
    episode.metadata["tensor_hash"] = tensor_hash(episode)
    return episode
