"""Train four adaptive particles on a genuine four-mode online task."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.experiments.evaluate_integrated_tabarena import release_device_memory
from tfmplayground.experiments.train_sequential_latent_filter import (
    _canonical_permutation_indices,
    _effective_milestones,
    _js_divergence,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import (
    AdaptiveKParticleFilter,
    AdaptiveParticlePrediction,
    checkpoint_sha256,
    expand_two_to_k_particles,
    load_adaptive_checkpoint,
    save_adaptive_checkpoint,
)
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed

FEATURE_MODES = ("noise", "competing")
# Cluster center shared by every region under feature_mode="competing": region identity
# is carried by which column holds the value, not by the value itself.
COMPETING_EVIDENCE_CENTER = 2.0


def _num_regions(num_modes: int) -> int:
    regions = round(math.log2(num_modes))
    if 2**regions != num_modes:
        raise ValueError("num_modes must be a power of 2.")
    return regions


def _mode_bits(num_modes: int) -> torch.Tensor:
    """Binary digits of each mode index, shape (num_modes, num_regions)."""
    regions = _num_regions(num_modes)
    return torch.tensor(
        [[int(bit) for bit in format(mode, f"0{regions}b")] for mode in range(num_modes)], dtype=torch.long
    )


def _mode_vectors(num_modes: int) -> torch.Tensor:
    """Query-length mode vectors: each region's bit duplicated across its 2 probe points."""
    return _mode_bits(num_modes).repeat_interleave(2, dim=1)


def _mode_outcomes(num_modes: int) -> tuple[int, ...]:
    """Canonical dot-product outcome for each mode's duplicated query vector."""
    vectors = _mode_vectors(num_modes)
    width = vectors.shape[1]
    weights = torch.tensor([2 ** (width - 1 - i) for i in range(width)])
    return tuple(int(value) for value in (vectors * weights).sum(-1).tolist())


def _mode_conditions(num_modes: int) -> tuple[str, ...]:
    return ("neutral", *(f"consistent_{mode}" for mode in range(num_modes)), "contradictory", "noisy")


def _query_points(num_regions: int, spacing: float = 1.0) -> np.ndarray:
    """Two probe points per region. spacing=0 collocates every region's cluster, which is
    what feature_mode="competing" wants: there, regions are told apart by column, not value."""
    points = []
    for region in range(num_regions):
        center = COMPETING_EVIDENCE_CENTER + region * spacing
        points.extend((center, center + 0.1))
    return np.array(points, dtype=np.float32)


# Kept for backwards-compatible imports; equivalent to the num_modes=4 case.
FOUR_MODE_VECTORS = _mode_vectors(4)
FOUR_MODE_OUTCOMES = _mode_outcomes(4)
FOUR_MODE_CONDITIONS = _mode_conditions(4)


@dataclass
class FourModeBatch:
    initial_support_x: torch.Tensor
    initial_support_y: torch.Tensor
    stream_x: torch.Tensor
    stream_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    latent_mode: torch.Tensor
    initial_mode: torch.Tensor
    posterior: torch.Tensor
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class FourModeConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    source_checkpoint: str = "runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth"
    evaluation_checkpoint: str | None = None
    output_dir: str | None = None
    device: str = "cpu"
    particle_count: int = 4
    num_modes: int = 4
    mode_curriculum: tuple[int, ...] | None = None
    num_features: int = 1
    feature_curriculum: tuple[int, ...] | None = None
    feature_log_range: tuple[int, int] | None = None
    feature_mode: str = "noise"
    prior_count: int = 128
    update_count: int = 128
    ratio_curriculum: tuple[float, ...] | None = None
    query_count: int = 4
    batch_size: int = 2
    accumulate_gradients: int = 8
    steps: int = 1000
    learning_rate: float = 1e-4
    gate_learning_rate: float = 1e-3
    initial_particle_jitter: float = 0.05
    diversity_weight: float = 0.05
    coherence_weight: float = 0.10
    residual_weight: float = 0.01
    specialization_weight: float = 0.0
    evidence_logit_scale: float = 1.0
    evidence_disagreement_threshold: float | None = None
    evidence_disagreement_js_threshold: float | None = 0.30
    evaluation_trials: int = 256


def validate_config(config: FourModeConfig) -> None:
    if config.prior_count < 1 or config.update_count < 1:
        raise ValueError("prior_count and update_count must be positive.")
    if config.ratio_curriculum is not None:
        if min(config.ratio_curriculum) <= 0 or max(config.ratio_curriculum) >= 1:
            raise ValueError("ratio_curriculum entries must be prior fractions strictly in (0, 1).")
    curriculum = config.mode_curriculum or (config.num_modes,)
    if config.mode_curriculum is None:
        expected_query_count = 2 if config.num_modes == 1 else 2 * _num_regions(config.num_modes)
        if config.query_count != expected_query_count:
            raise ValueError(
                f"num_modes={config.num_modes} requires query_count={expected_query_count} (2 probe points per region)."
            )
    # NOTE: the historical `update_count <= prior_count` guard is deliberately not enforced here.
    # The model architecture (NanoTabPFNIntegratedLatentFilter) has no such dependency; the guard
    # only reflected the protocol the original checkpoints were trained under. Lifting it lets us
    # measure update-heavy ratios. The guard is still enforced in evaluate_integrated_tabarena.py.
    if min(config.batch_size, config.accumulate_gradients, config.evaluation_trials) < 1:
        raise ValueError("Training and evaluation counts must be positive.")
    if config.evaluation_checkpoint is None and config.steps < 1:
        raise ValueError("Training steps must be positive.")
    if config.evaluation_checkpoint is None and config.initial_particle_jitter <= 0:
        raise ValueError("Four-mode training requires positive symmetry-breaking jitter.")
    if config.evidence_logit_scale <= 0:
        raise ValueError("Evidence logit scale must be positive.")
    if config.evidence_disagreement_threshold is not None and config.evidence_disagreement_threshold <= 0:
        raise ValueError("Evidence disagreement threshold must be positive.")
    if config.evidence_disagreement_js_threshold is not None and not 0 < config.evidence_disagreement_js_threshold <= 1:
        raise ValueError("Evidence disagreement JS threshold must be in (0, 1].")
    if config.evidence_disagreement_threshold is not None and config.evidence_disagreement_js_threshold is not None:
        raise ValueError("Select either max-min or JS disagreement gating, not both.")
    for modes in curriculum:
        _num_regions(modes)
    if config.particle_count < max(curriculum):
        raise ValueError("Particle count must be at least max(mode_curriculum) to represent every hypothesis.")
    if config.feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}.")
    feature_curriculum = config.feature_curriculum or (config.num_features,)
    if min(feature_curriculum) < 1:
        raise ValueError("num_features (and every entry in feature_curriculum) must be at least 1.")
    if config.feature_log_range is not None:
        low, high = config.feature_log_range
        if low < 1 or high < low:
            raise ValueError("feature_log_range must be (low, high) with 1 <= low <= high.")
        # feature_curriculum may still be set alongside feature_log_range: it becomes the
        # discrete evaluation-reporting grid while training samples continuously.
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def _with_noise_features(signal: np.ndarray, num_features: int, rng: np.random.Generator) -> np.ndarray:
    """Append (num_features - 1) pure-noise distractor columns after the informative feature."""
    if num_features == 1:
        return signal
    noise = rng.normal(0.0, 1.0, size=(*signal.shape[:-1], num_features - 1)).astype(np.float32)
    return np.concatenate([signal, noise], axis=-1)


def _effective_features(num_features: int, num_modes: int, feature_mode: str) -> int:
    """Feature width actually used, after the competing-mode 'one column per region' floor.

    Under feature_mode="competing" every region needs its own hosting column, so a
    requested count below log2(num_modes) is not representable and is clamped up.
    """
    if feature_mode != "competing" or num_modes <= 1:
        return num_features
    return max(num_features, _num_regions(num_modes))


def _competing_feature_columns(num_features: int, num_regions: int, rng: np.random.Generator) -> np.ndarray:
    """Per-episode assignment of columns to roles, so no column index is privileged.

    Entry 0 hosts the prior's sign rule (and region 0, mirroring the single-feature
    task where both live on column 0); entries 1..num_regions-1 host the remaining
    regions; the tail entries are rivals.
    """
    if num_features < max(num_regions, 1):
        raise ValueError("competing features require at least one column per region.")
    return rng.permutation(num_features)


def _competing_features(
    values: np.ndarray, host: np.ndarray, num_features: int, rng: np.random.Generator
) -> np.ndarray:
    """Place each row's informative value in its host column and fill the rest with rivals.

    values: (batch, rows) informative value for each row.
    host:   column index hosting that value, broadcastable to values' shape.
    """
    table = _rival_values((*values.shape, num_features), rng)
    indices = np.broadcast_to(host, values.shape)[..., None].astype(np.intp)
    np.put_along_axis(table, indices, values[..., None].astype(np.float32), axis=-1)
    return table


def _rival_values(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Rival-column fill: the same uniform(-1, 1) range the prior support is drawn from.

    Drawing rivals from the prior's own input range is what makes them competing
    explanations rather than separable noise -- under the prior's sign rule each
    rival column offers a live, support-consistent account of the stream labels.
    """
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def generate_four_mode_episodes(
    config: FourModeConfig,
    rng: np.random.Generator,
    *,
    condition: str | None = None,
    batch_size: int | None = None,
    num_modes: int | None = None,
    num_features: int | None = None,
    prior_count: int | None = None,
    update_count: int | None = None,
) -> FourModeBatch:
    batch_size = config.batch_size if batch_size is None else batch_size
    num_modes = config.num_modes if num_modes is None else num_modes
    num_features = config.num_features if num_features is None else num_features
    num_features = _effective_features(num_features, num_modes, config.feature_mode)
    prior_count = config.prior_count if prior_count is None else prior_count
    update_count = config.update_count if update_count is None else update_count
    conditions_list = _mode_conditions(num_modes) if num_modes > 1 else ("trivial",)
    if condition is not None and condition not in conditions_list:
        raise ValueError(f"Unknown mode condition: {condition}")
    device = torch.device(config.device)
    competing = config.feature_mode == "competing"
    columns = (
        _competing_feature_columns(num_features, _num_regions(num_modes) if num_modes > 1 else 0, rng)
        if competing
        else None
    )

    if num_modes == 1:
        # No hidden state at all: query points reuse the prior's own sign-threshold rule,
        # so the answer is already fully determined by context the model has already seen.
        support_x = np.empty((batch_size, prior_count, 1), dtype=np.float32)
        support_y = np.empty((batch_size, prior_count), dtype=np.float32)
        stream_x = np.empty((batch_size, update_count, 1), dtype=np.float32)
        stream_y = np.empty((batch_size, update_count), dtype=np.int64)
        query_x = np.empty((batch_size, 2, 1), dtype=np.float32)
        query_y = np.empty((batch_size, 2), dtype=np.int64)
        posterior = np.ones((batch_size, update_count + 1, 1), dtype=np.float32)
        for batch_index in range(batch_size):
            negative = rng.uniform(-1.0, -0.05, prior_count // 2)
            positive = rng.uniform(0.05, 1.0, prior_count - prior_count // 2)
            prior = np.concatenate((negative, positive))
            rng.shuffle(prior)
            support_x[batch_index, :, 0] = prior
            support_y[batch_index] = prior > 0
            stream_values = rng.uniform(-1.0, 1.0, update_count)
            stream_x[batch_index, :, 0] = stream_values
            stream_y[batch_index] = stream_values > 0
            query_values = rng.uniform(-1.0, 1.0, 2)
            query_x[batch_index, :, 0] = query_values
            query_y[batch_index] = query_values > 0
        if competing:
            # The trivial task has no regions, so a single hosting column carries the
            # prior's sign rule and every other column is an unrelated rival.
            assert columns is not None
            host = np.asarray(columns[0])
            support_x = _competing_features(support_x[..., 0], host, num_features, rng)
            stream_x = _competing_features(stream_x[..., 0], host, num_features, rng)
            query_x = _competing_features(query_x[..., 0], host, num_features, rng)
        else:
            support_x = _with_noise_features(support_x, num_features, rng)
            stream_x = _with_noise_features(stream_x, num_features, rng)
            query_x = _with_noise_features(query_x, num_features, rng)
        return FourModeBatch(
            initial_support_x=torch.as_tensor(support_x, device=device),
            initial_support_y=torch.as_tensor(support_y, device=device),
            stream_x=torch.as_tensor(stream_x, device=device),
            stream_y=torch.as_tensor(stream_y, device=device),
            query_x=torch.as_tensor(query_x, device=device),
            query_y=torch.as_tensor(query_y, device=device),
            latent_mode=torch.zeros(batch_size, dtype=torch.long, device=device),
            initial_mode=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            posterior=torch.as_tensor(posterior, device=device),
            conditions=("trivial",) * batch_size,
        )

    num_regions = _num_regions(num_modes)
    modes = _mode_vectors(num_modes).numpy()
    mode_bits = _mode_bits(num_modes).numpy()
    # Under feature_mode="competing" every region's cluster sits at the same value, so the
    # spacing that separates regions along the value axis collapses to zero.
    region_spacing = 0.0 if competing else 1.0
    query_points = _query_points(num_regions, spacing=region_spacing)
    query_width = 2 * num_regions
    switch_index = update_count // 8

    support_x = np.empty((batch_size, prior_count, 1), dtype=np.float32)
    support_y = np.empty((batch_size, prior_count), dtype=np.float32)
    stream_x = np.empty((batch_size, update_count, 1), dtype=np.float32)
    stream_y = np.empty((batch_size, update_count), dtype=np.int64)
    query_x = np.tile(query_points[None, :, None], (batch_size, 1, 1))
    query_y = np.empty((batch_size, query_width), dtype=np.int64)
    stream_region = np.empty((batch_size, update_count), dtype=np.int64)
    latent_mode = np.empty(batch_size, dtype=np.int64)
    initial_mode = np.full(batch_size, -1, dtype=np.int64)
    posterior = np.empty((batch_size, update_count + 1, num_modes), dtype=np.float32)
    conditions = []
    for batch_index in range(batch_size):
        selected_condition = condition or conditions_list[batch_index % len(conditions_list)]
        conditions.append(selected_condition)
        negative = rng.uniform(-1.0, -0.05, prior_count // 2)
        positive = rng.uniform(0.05, 1.0, prior_count - prior_count // 2)
        prior = np.concatenate((negative, positive))
        rng.shuffle(prior)
        support_x[batch_index, :, 0] = prior
        support_y[batch_index] = prior > 0
        target = (
            int(selected_condition.split("_", 1)[1])
            if selected_condition.startswith("consistent_")
            else int(rng.integers(num_modes))
        )
        latent_mode[batch_index] = target
        query_y[batch_index] = modes[target]
        flip_rate = 0.01
        if selected_condition == "neutral":
            values = rng.uniform(-1.0, 1.0, update_count)
            labels = values > 0
            region = np.full(update_count, -1)
        else:
            region = np.arange(update_count) % num_regions
            centers = COMPETING_EVIDENCE_CENTER + region.astype(np.float32) * region_spacing
            values = rng.normal(centers, 0.04)
            if selected_condition == "contradictory":
                first = int(rng.integers(num_modes))
                target = num_modes - 1 - first
                initial_mode[batch_index] = first
                latent_mode[batch_index] = target
                query_y[batch_index] = modes[target]
                active = np.where(np.arange(update_count) < switch_index, first, target)
                labels = mode_bits[active, region]
            else:
                labels = mode_bits[target, region]
                if selected_condition == "noisy":
                    flip_rate = float(rng.choice((0.10, 0.25, 0.40)))
                    labels = np.logical_xor(labels, rng.random(update_count) < flip_rate)
        stream_x[batch_index, :, 0] = values
        stream_y[batch_index] = labels
        stream_region[batch_index] = region
        log_posterior = np.full(num_modes, -math.log(num_modes), dtype=np.float64)
        posterior[batch_index, 0] = np.exp(log_posterior)
        for update_index, (observed, evidence_region) in enumerate(zip(labels, region, strict=True)):
            if evidence_region >= 0:
                predictions = mode_bits[:, evidence_region]
                likelihood = np.where(predictions == observed, 1 - flip_rate, flip_rate)
                log_posterior += np.log(likelihood)
                maximum = log_posterior.max()
                log_posterior -= maximum + np.log(np.exp(log_posterior - maximum).sum())
            posterior[batch_index, update_index + 1] = np.exp(log_posterior)
    if competing:
        assert columns is not None
        # Region r is hosted on columns[r]; neutral rows (region -1) follow the prior's own
        # sign rule, which lives on columns[0] just as it does on feature 0 today.
        support_x = _competing_features(support_x[..., 0], np.asarray(columns[0]), num_features, rng)
        stream_x = _competing_features(stream_x[..., 0], columns[np.maximum(stream_region, 0)], num_features, rng)
        query_x = _competing_features(
            query_x[..., 0],
            columns[np.arange(query_width) // 2][None, :],
            num_features,
            rng,
        )
    else:
        support_x = _with_noise_features(support_x, num_features, rng)
        stream_x = _with_noise_features(stream_x, num_features, rng)
        query_x = _with_noise_features(query_x, num_features, rng)
    return FourModeBatch(
        initial_support_x=torch.as_tensor(support_x, device=device),
        initial_support_y=torch.as_tensor(support_y, device=device),
        stream_x=torch.as_tensor(stream_x, device=device),
        stream_y=torch.as_tensor(stream_y, device=device),
        query_x=torch.as_tensor(query_x, device=device),
        query_y=torch.as_tensor(query_y, device=device),
        latent_mode=torch.as_tensor(latent_mode, device=device),
        initial_mode=torch.as_tensor(initial_mode, device=device),
        posterior=torch.as_tensor(posterior, device=device),
        conditions=tuple(conditions),
    )


def _pairwise_js(slot_joint: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            _js_divergence(slot_joint[:, left], slot_joint[:, right])
            for left in range(slot_joint.shape[1])
            for right in range(left + 1, slot_joint.shape[1])
        ],
        dim=1,
    )


def four_mode_loss(
    prediction: AdaptiveParticlePrediction,
    batch: FourModeBatch,
    config: FourModeConfig,
    *,
    num_modes: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    num_modes = config.num_modes if num_modes is None else num_modes
    prequential = -prediction.prequential_log_likelihood_for(batch.stream_y).mean()
    if num_modes == 1:
        # No hidden mode to disambiguate: only train stream/query accuracy, no
        # joint/coherence/diversity terms since there is nothing to keep distinct.
        marginal = prediction.marginal_probabilities()
        labels = batch.query_y[:, None, :, None].expand(-1, marginal.shape[1], -1, 1)
        marginal_loss = -marginal.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
        total = prequential + marginal_loss
        return total, {
            "loss": float(total.detach()),
            "prequential_loss": float(prequential.detach()),
            "marginal_loss": float(marginal_loss.detach()),
            "ambiguity_probability": float(prediction.ambiguity_probability.mean().detach()),
        }
    outcomes = _mode_outcomes(num_modes)
    width = batch.query_y.shape[-1]
    weights = torch.tensor([2 ** (width - 1 - i) for i in range(width)], device=batch.query_y.device)
    joint = prediction.joint_probabilities()
    outcome = (batch.query_y * weights).sum(-1)
    selected = joint.gather(-1, outcome[:, None, None].expand(-1, joint.shape[1], 1)).squeeze(-1)
    trajectory = -selected.clamp_min(1e-12).log().mean()
    marginal = prediction.marginal_probabilities()
    labels = batch.query_y[:, None, :, None].expand(-1, marginal.shape[1], -1, 1)
    marginal_loss = -marginal.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
    slot_joint = prediction.slot_joint_log_probabilities().exp()
    valid_slot = slot_joint[:, :, list(outcomes)]
    coherent = valid_slot.sum(-1)
    coherence = (1 - coherent).mean()
    if config.specialization_weight > 0:
        representative_slots = valid_slot.argmax(dim=1)
        diversity_joint = slot_joint.gather(
            1,
            representative_slots[:, :, None].expand(-1, -1, slot_joint.shape[-1]),
        )
    else:
        diversity_joint = slot_joint
    pairwise = _pairwise_js(diversity_joint)
    diversity = F.relu(0.20 - pairwise).mean()
    coverage = -valid_slot.max(dim=1).values.clamp_min(1e-12).log().mean()
    residual = prediction.stream_residuals.abs().mean()
    total = (
        prequential
        + trajectory
        + marginal_loss
        + config.coherence_weight * coherence
        + config.diversity_weight * diversity
        + config.residual_weight * residual
        + config.specialization_weight * coverage
    )
    return total, {
        "loss": float(total.detach()),
        "prequential_loss": float(prequential.detach()),
        "trajectory_loss": float(trajectory.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "coherence_loss": float(coherence.detach()),
        "diversity_loss": float(diversity.detach()),
        "minimum_pair_js": float(pairwise.min(1).values.mean().detach()),
        "coverage_loss": float(coverage.detach()),
        "ambiguity_probability": float(prediction.ambiguity_probability.mean().detach()),
    }


def train_four_mode(
    model: AdaptiveKParticleFilter, config: FourModeConfig
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    model.particle_model.set_trainability("frozen")
    model.vanilla_backbone.requires_grad_(False)
    model.to(config.device).train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.particle_model.trainable_parameters(), "lr": config.learning_rate},
            {"params": model.ambiguity_gate.parameters(), "lr": config.gate_learning_rate},
        ]
    )
    rng = np.random.default_rng(config.seed + 100_000)
    mode_curriculum = config.mode_curriculum or (config.num_modes,)
    # Interleave every (num_modes, condition) pair across the mode curriculum so each gradient
    # step sees a mix of task-complexity levels, rather than a separate model per setting.
    schedule = [
        (modes, cond) for modes in mode_curriculum for cond in (_mode_conditions(modes) if modes > 1 else ("trivial",))
    ]
    feature_curriculum = config.feature_curriculum or (config.num_features,)
    feature_counter = [0]

    def draw_features() -> int:
        if config.feature_log_range is not None:
            log_low, log_high = math.log(config.feature_log_range[0]), math.log(config.feature_log_range[1])
            return int(round(math.exp(rng.uniform(log_low, log_high))))
        value = feature_curriculum[feature_counter[0] % len(feature_curriculum)]
        feature_counter[0] += 1
        return value

    total_rows = config.prior_count + config.update_count
    ratio_curriculum = config.ratio_curriculum or (config.prior_count / total_rows,)
    ratio_counter = [0]

    def draw_ratio() -> tuple[int, int]:
        fraction = ratio_curriculum[ratio_counter[0] % len(ratio_curriculum)]
        ratio_counter[0] += 1
        task_prior = min(max(round(total_rows * fraction), 1), total_rows - 1)
        return task_prior, total_rows - task_prior

    history = []
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for accumulation in range(config.accumulate_gradients):
            modes, condition = schedule[((step - 1) * config.accumulate_gradients + accumulation) % len(schedule)]
            features = _effective_features(draw_features(), modes, config.feature_mode)
            task_prior_count, task_update_count = draw_ratio()
            batch = generate_four_mode_episodes(
                config,
                rng,
                condition=condition,
                num_modes=modes,
                num_features=features,
                prior_count=task_prior_count,
                update_count=task_update_count,
            )
            prediction = model(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.stream_y,
                batch.query_x,
            )
            loss, metrics = four_mode_loss(prediction, batch, config, num_modes=modes)
            (loss / config.accumulate_gradients).backward()
            release_device_memory(config.device)
            tag = f"modes{modes}_feat{features}_ratio{task_prior_count}-{task_update_count}"
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
                totals[f"{tag}_{key}"] = totals.get(f"{tag}_{key}", 0.0) + value
                counts[f"{tag}_{key}"] = counts.get(f"{tag}_{key}", 0) + 1
        for key, count in counts.items():
            totals[key] /= count
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()
        history.append({"step": step, **totals})
    return history, optimizer


@torch.no_grad()
def _invariance(model: AdaptiveKParticleFilter, batch: FourModeBatch, rng: np.random.Generator) -> dict[str, float]:
    original = model(batch.initial_support_x, batch.initial_support_y, batch.stream_x, batch.stream_y, batch.query_x)
    query_width = batch.query_x.shape[1]
    query_order = tuple(rng.permutation(query_width))
    permuted = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x[:, query_order],
    )
    mapping = _canonical_permutation_indices(query_order, batch.query_x.device)
    order = torch.as_tensor(rng.permutation(config_update := batch.stream_x.shape[1]), device=batch.stream_x.device)
    assert config_update == batch.stream_y.shape[1]
    reordered = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x[:, order],
        batch.stream_y[:, order],
        batch.query_x,
    )
    return {
        "query_permutation_max_error": float(
            (original.joint_probabilities() - permuted.joint_probabilities()[:, :, mapping]).abs().max()
        ),
        "evidence_order_weight_max_error": float(
            (original.log_weights[:, -1].exp() - reordered.log_weights[:, -1].exp()).abs().max()
        ),
        "evidence_order_joint_max_error": float(
            (original.joint_probabilities()[:, -1] - reordered.joint_probabilities()[:, -1]).abs().max()
        ),
    }


@torch.no_grad()
def _evaluate_trivial_mode(
    model: AdaptiveKParticleFilter,
    config: FourModeConfig,
    num_features: int,
    prior_count: int,
    update_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Zero-ambiguity floor: the query is already determined by the prior the model has seen."""
    model.to(config.device).eval()
    rng = np.random.default_rng(config.seed + 500_000)
    batch = generate_four_mode_episodes(
        config,
        rng,
        condition="trivial",
        batch_size=config.evaluation_trials,
        num_modes=1,
        num_features=num_features,
        prior_count=prior_count,
        update_count=update_count,
    )
    prediction = model(batch.initial_support_x, batch.initial_support_y, batch.stream_x, batch.stream_y, batch.query_x)
    marginal = prediction.vanilla_marginal_probabilities()
    correct = marginal.argmax(-1) == batch.query_y
    release_device_memory(config.device)
    rows = [
        {
            "trial": trial,
            "correct": float(correct[trial].float().mean()),
            "ambiguity_probability": float(prediction.ambiguity_probability[trial]),
        }
        for trial in range(config.evaluation_trials)
    ]
    trials = pd.DataFrame(rows)
    metrics = {
        "trivial_accuracy": float(trials["correct"].mean()),
        "trivial_mean_ambiguity": float(trials["ambiguity_probability"].mean()),
    }
    checks = {
        "trivial_accuracy": metrics["trivial_accuracy"] >= 0.95,
        "trivial_low_ambiguity": metrics["trivial_mean_ambiguity"] <= 0.10,
    }
    report = {
        "threshold_profile": "trivial_mode_poc_v1",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
    }
    return trials, report


@torch.no_grad()
def _evaluate_single_mode_count(
    model: AdaptiveKParticleFilter,
    config: FourModeConfig,
    num_modes: int,
    num_features: int,
    prior_count: int,
    update_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if num_modes == 1:
        return _evaluate_trivial_mode(model, config, num_features, prior_count, update_count)
    outcomes = _mode_outcomes(num_modes)
    conditions_list = _mode_conditions(num_modes)
    switch_index = update_count // 8
    model.to(config.device).eval()
    milestones = _effective_milestones(update_count)
    states = torch.as_tensor(milestones, device=config.device)
    rows = []
    invariance = {
        name: 0.0
        for name in (
            "query_permutation_max_error",
            "evidence_order_weight_max_error",
            "evidence_order_joint_max_error",
        )
    }
    distinct = []
    for condition_index, condition in enumerate(conditions_list):
        rng = np.random.default_rng(config.seed + 500_000 + condition_index)
        batch = generate_four_mode_episodes(
            config,
            rng,
            condition=condition,
            batch_size=config.evaluation_trials,
            num_modes=num_modes,
            num_features=num_features,
            prior_count=prior_count,
            update_count=update_count,
        )
        prediction = model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        weights = prediction.log_weights.index_select(1, states).exp()
        joint = prediction.joint_probabilities(states)
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        valid_slot = slot_joint[:, :, list(outcomes)]
        assignment = valid_slot.argmax(-1)
        representative_slots = valid_slot.argmax(dim=1)
        distinct.extend([len(torch.unique(item)) == num_modes for item in representative_slots])
        representative_joint = slot_joint.gather(
            1,
            representative_slots[:, :, None].expand(-1, -1, slot_joint.shape[-1]),
        )
        representative_pairwise = _pairwise_js(representative_joint)
        mode_weights = torch.stack(
            [(weights * (assignment == mode)[:, None]).sum(-1) for mode in range(num_modes)],
            dim=-1,
        )
        exact = batch.posterior.index_select(1, states)
        predicted_mode_probability = joint[:, :, list(outcomes)]
        predicted_mode_probability /= predicted_mode_probability.sum(-1, keepdim=True).clamp_min(1e-12)
        posterior_js = _js_divergence(exact, predicted_mode_probability)
        subset = FourModeBatch(
            initial_support_x=batch.initial_support_x[:16],
            initial_support_y=batch.initial_support_y[:16],
            stream_x=batch.stream_x[:16],
            stream_y=batch.stream_y[:16],
            query_x=batch.query_x[:16],
            query_y=batch.query_y[:16],
            latent_mode=batch.latent_mode[:16],
            initial_mode=batch.initial_mode[:16],
            posterior=batch.posterior[:16],
            conditions=batch.conditions[:16],
        )
        for name, value in _invariance(model, subset, rng).items():
            invariance[name] = max(invariance[name], value)
        for trial in range(config.evaluation_trials):
            target = int(batch.latent_mode[trial])
            initial = int(batch.initial_mode[trial])
            opposite = target if initial < 0 else target
            for state_index, milestone in enumerate(milestones):
                row = {
                    "condition": condition,
                    "trial": trial,
                    "milestone": milestone,
                    **{
                        f"mode_weight_{mode}": float(mode_weights[trial, state_index, mode])
                        for mode in range(num_modes)
                    },
                    "supporting_weight": float(mode_weights[trial, state_index, target]),
                    "initial_supporting_weight": float(
                        mode_weights[trial, state_index, initial if initial >= 0 else target]
                    ),
                    "opposite_weight": float(mode_weights[trial, state_index, opposite]),
                    "incoherent_mass": float((1 - joint[trial, state_index, list(outcomes)].sum()).clamp_min(0)),
                    "posterior_js": float(posterior_js[trial, state_index]),
                    "minimum_pair_js": float(representative_pairwise[trial].min()),
                    "ambiguity_probability": float(prediction.ambiguity_probability[trial]),
                    "effective_particles": float(prediction.effective_particle_count(states)[trial, state_index]),
                }
                rows.append(row)
        release_device_memory(config.device)
    trials = pd.DataFrame(rows)
    final = update_count
    mode_columns = [f"mode_weight_{mode}" for mode in range(num_modes)]
    neutral_initial = (
        trials[(trials.condition == "neutral") & (trials.milestone == 0)].set_index("trial")[mode_columns].sort_index()
    )
    neutral = trials[(trials.condition == "neutral") & (trials.milestone == final)].set_index("trial").sort_index()
    neutral_deviation = neutral[mode_columns] - neutral_initial
    monotonic = {}
    concentration = {}
    for mode in range(num_modes):
        condition = f"consistent_{mode}"
        curve = trials[trials.condition == condition].groupby("milestone").supporting_weight.mean()
        monotonic[condition] = float(curve.diff().min())
        concentration[condition] = float(curve.loc[final])
    contradiction = trials[trials.condition == "contradictory"]
    at_switch = contradiction[contradiction.milestone == switch_index].set_index("trial")
    at_final = contradiction[contradiction.milestone == final].set_index("trial")
    metrics = {
        "neutral_mean_total_variation": float(0.5 * neutral_deviation.abs().sum(1).mean()),
        "neutral_max_signed_drift": float(neutral_deviation.mean().abs().max()),
        "largest_consistent_weight_drop": monotonic,
        "final_consistent_supporting_weight": concentration,
        "contradiction_confidence_drop": float(
            (at_switch.initial_supporting_weight - at_final.initial_supporting_weight).mean()
        ),
        "contradiction_reversed_weight": float(at_final.opposite_weight.mean()),
        "final_incoherent_mass": float(trials[trials.milestone == final].incoherent_mass.mean()),
        "minimum_slot_joint_js": float(trials[trials.milestone == final].minimum_pair_js.mean()),
        "distinct_slots_fraction": float(np.mean(distinct)),
        **invariance,
    }
    checks = {
        "neutral_total_variation": metrics["neutral_mean_total_variation"] <= 0.05,
        "neutral_signed_drift": metrics["neutral_max_signed_drift"] <= 0.01,
        **{f"{name}_monotonic": value >= -0.02 for name, value in monotonic.items()},
        **{f"{name}_concentration": value >= 0.90 for name, value in concentration.items()},
        "contradiction_confidence_drop": metrics["contradiction_confidence_drop"] >= 0.40,
        "contradiction_reversal": metrics["contradiction_reversed_weight"] >= 0.70,
        "incoherent_mass": metrics["final_incoherent_mass"] <= 0.10,
        "slot_joint_diversity": metrics["minimum_slot_joint_js"] >= 0.20,
        "query_permutation": invariance["query_permutation_max_error"] <= 1e-5,
        "evidence_order_weights": invariance["evidence_order_weight_max_error"] <= 1e-5,
        "evidence_order_joint": invariance["evidence_order_joint_max_error"] <= 1e-5,
        "distinct_slots": metrics["distinct_slots_fraction"] >= 0.90,
    }
    report = {
        "threshold_profile": f"n_mode_poc_v1_modes{num_modes}",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
    }
    return trials, report


def evaluate_four_mode(model: AdaptiveKParticleFilter, config: FourModeConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate across every (num_modes, num_features, ratio) triple in the curricula."""
    mode_curriculum = config.mode_curriculum or (config.num_modes,)
    feature_curriculum = config.feature_curriculum or (config.num_features,)
    total_rows = config.prior_count + config.update_count
    ratio_curriculum = config.ratio_curriculum or (config.prior_count / total_rows,)
    per_cell_trials = []
    per_cell_reports = {}
    for num_modes in mode_curriculum:
        for num_features in feature_curriculum:
            effective_features = _effective_features(num_features, num_modes, config.feature_mode)
            for fraction in ratio_curriculum:
                prior_count = min(max(round(total_rows * fraction), 1), total_rows - 1)
                update_count = total_rows - prior_count
                cell = (
                    f"modes{num_modes}_feat{num_features}"
                    f"{'' if effective_features == num_features else f'as{effective_features}'}"
                    f"_ratio{prior_count}-{update_count}"
                )
                trials, report = _evaluate_single_mode_count(
                    model, config, num_modes, num_features, prior_count, update_count
                )
                release_device_memory(config.device)
                trials = trials.copy()
                trials["num_modes"] = num_modes
                trials["num_features"] = num_features
                # Requested vs. actually used width: under feature_mode="competing" a request
                # below log2(num_modes) is clamped up, so two requested counts can name the
                # same cell. Keep both columns so that is visible rather than silent.
                trials["effective_features"] = effective_features
                trials["prior_count"] = prior_count
                trials["update_count"] = update_count
                per_cell_trials.append(trials)
                per_cell_reports[cell] = report
    combined_trials = pd.concat(per_cell_trials, ignore_index=True)
    report = {
        "threshold_profile": (
            f"n_mode_curriculum_v1_modes{'-'.join(str(m) for m in mode_curriculum)}"
            f"_feat{'-'.join(str(f) for f in feature_curriculum)}"
            f"_{config.feature_mode}"
            f"_ratio{'-'.join(str(round(f, 2)) for f in ratio_curriculum)}"
        ),
        "per_cell": per_cell_reports,
        "passed": all(r["passed"] for r in per_cell_reports.values()),
        "failure_reasons": {key: r["failure_reasons"] for key, r in per_cell_reports.items() if not r["passed"]},
    }
    return combined_trials, report


def run(config: FourModeConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "four_mode_particle_filter" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    if config.evaluation_checkpoint is not None:
        model, _ = load_adaptive_checkpoint(config.evaluation_checkpoint)
        if model.particle_model.num_hypotheses != config.particle_count:
            raise ValueError("Evaluation checkpoint particle count does not match configuration.")
        history = [{"step": 0, "evaluation_only": True}]
        optimizer = None
        source_hash = checkpoint_sha256(config.evaluation_checkpoint)
    else:
        source, _ = load_integrated_checkpoint(config.source_checkpoint)
        vanilla = init_model_from_state_dict_file(config.official_checkpoint)
        model = expand_two_to_k_particles(source, vanilla, particle_count=config.particle_count)
        with torch.no_grad():
            generator = torch.Generator(device=model.particle_model.initial_latents.device)
            generator.manual_seed(config.seed + 17)
            model.particle_model.initial_latents[2:].add_(
                torch.randn(
                    model.particle_model.initial_latents[2:].shape,
                    generator=generator,
                    device=model.particle_model.initial_latents.device,
                )
                * config.initial_particle_jitter
            )
        history, optimizer = train_four_mode(model, config)
        source_hash = checkpoint_sha256(config.source_checkpoint)
    model.particle_model.set_evidence_logit_scale(config.evidence_logit_scale)
    model.particle_model.set_evidence_disagreement_threshold(config.evidence_disagreement_threshold)
    model.particle_model.set_evidence_disagreement_js_threshold(config.evidence_disagreement_js_threshold)
    pd.DataFrame(history).to_csv(output_dir / "learning_curves.csv", index=False)
    trials, report = evaluate_four_mode(model, config)
    trials.to_csv(output_dir / "metrics.csv", index=False)
    trials.groupby(["condition", "milestone"], as_index=False).mean(numeric_only=True).to_csv(
        output_dir / "summary.csv", index=False
    )
    (output_dir / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    save_adaptive_checkpoint(
        output_dir / "checkpoint.pth",
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=source_hash,
        controlled_gate=report,
        optimizer_state=None if optimizer is None else optimizer.state_dict(),
    )
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--source-checkpoint",
        default="runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth",
    )
    parser.add_argument("--evaluation-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--particle-count", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--num-features", type=int, default=1)
    parser.add_argument(
        "--mode-curriculum",
        type=str,
        default=None,
        help="Comma-separated list of mode counts to interleave during training, e.g. '2,4,8'.",
    )
    parser.add_argument(
        "--feature-curriculum",
        type=str,
        default=None,
        help="Comma-separated list of feature counts to interleave during training, e.g. '1,4,8'.",
    )
    parser.add_argument(
        "--feature-log-range",
        type=str,
        default=None,
        help="'low,high' - sample num_features log-uniformly from this range each episode instead of a fixed list.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="noise",
        help=(
            "'noise' appends N(0,1) distractor columns after one informative feature. "
            "'competing' hosts each region on its own column and draws every other column "
            "from the prior's own uniform(-1,1) range, so rivals are support-consistent "
            "explanations rather than separable noise (requires num_features >= log2(num_modes); "
            "smaller requests are clamped up)."
        ),
    )
    parser.add_argument("--prior-count", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument(
        "--ratio-curriculum",
        type=str,
        default=None,
        help=(
            "Comma-separated list of prior fractions of (prior_count+update_count) to interleave "
            "during training, e.g. '0.25,0.5,0.75'."
        ),
    )
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-particle-jitter", type=float, default=0.05)
    parser.add_argument("--specialization-weight", type=float, default=0.0)
    parser.add_argument("--evidence-logit-scale", type=float, default=1.0)
    disagreement = parser.add_mutually_exclusive_group()
    disagreement.add_argument("--evidence-disagreement-threshold", type=float, default=argparse.SUPPRESS)
    disagreement.add_argument("--evidence-disagreement-js-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--evaluation-trials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = vars(build_parser().parse_args(argv))
    if "evidence_disagreement_threshold" in arguments and "evidence_disagreement_js_threshold" not in arguments:
        arguments["evidence_disagreement_js_threshold"] = None
    if arguments["mode_curriculum"] is not None:
        arguments["mode_curriculum"] = tuple(int(value) for value in arguments["mode_curriculum"].split(","))
    if arguments["feature_curriculum"] is not None:
        arguments["feature_curriculum"] = tuple(int(value) for value in arguments["feature_curriculum"].split(","))
    if arguments["feature_log_range"] is not None:
        low, high = (int(value) for value in arguments["feature_log_range"].split(","))
        arguments["feature_log_range"] = (low, high)
    if arguments["ratio_curriculum"] is not None:
        arguments["ratio_curriculum"] = tuple(float(value) for value in arguments["ratio_curriculum"].split(","))
    output = run(FourModeConfig(**arguments))
    print(f"Wrote four-mode particle-filter artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
