"""Train the slot-free continuous Bayesian uncertainty head for nanoTabPFN.

The mean path is a frozen vanilla nanoTabPFN and is never optimized.  Only the
uncertainty path is trained, in one of three modes: a completely frozen
uncertainty backbone, a frozen backbone with residual adapters (the primary
model), or a fully trainable uncertainty copy.

The objective compares *distributions*, never individual samples to candidate
identities: there is no matching step and no candidate ordering anywhere in the
loss.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.continuous_episodes import (
    HELDOUT_REGIME,
    TRAIN_REGIME,
    ContinuousEpisode,
    EpisodeRegime,
    available_support_sizes,
    curriculum_condition,
    sample_episode,
    sample_multiregime_episode,
    sample_paired_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.continuous_posterior import (
    NanoTabPFNBetaConcentrationModel,
    NanoTabPFNContinuousPosteriorModel,
    all_binary_outcomes,
    binary_entropy,
    project_candidate_posterior,
    save_continuous_checkpoint,
)
from tfmplayground.utils import get_default_device, set_randomness_seed

LOG_TWO = math.log(2.0)
#: Maximum binary variance, used to normalize the variance and covariance losses.
MAX_BINARY_VARIANCE = 0.25


@dataclass(frozen=True)
class ContinuousTrainingConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    require_cuda: bool = False
    # Architecture
    model_type: str = "continuous"  # "continuous" or "beta"
    uncertainty_mode: str = "adapters"  # "frozen", "adapters", or "full"
    context_mode: str = "deepsets"  # "deepsets" or "cross_attention"
    adapter_bottleneck: int = 32
    latent_dim: int = 32
    num_samples: int = 32
    inference_seed: int = 0
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 1500
    validation_interval: int = 100
    patience: int = 8
    min_delta: float = 1e-4
    batch_size: int = 2
    accumulate_gradients: int = 1
    gradient_clip: float = 1.0
    # Curriculum
    ambiguous_probability: float = 0.30
    identifiable_probability: float = 0.20
    noisy_probability: float = 0.15
    paired_probability: float = 0.15
    multiregime_probability: float = 0.20
    # Loss weights
    energy_weight: float = 1.0
    mutual_information_weight: float = 0.5
    variance_weight: float = 0.5
    covariance_weight: float = 0.25
    joint_weight: float = 1.0
    monotonicity_weight: float = 0.25
    #: Multiplies the mutual-information and variance weights only.  The moment
    #: terms are numerically much smaller than the joint query loss, so this is
    #: swept rather than asserted; 1.0 reproduces the declared weights exactly.
    moment_weight: float = 1.0
    # Validation
    validation_episodes: int = 12
    validation_support_size: int = 128
    validation_query_count: int = 6
    include_scm_families: bool = True
    #: Compute budget: training episodes never draw a larger support table.
    max_support_size: int = 512

    def curriculum(self) -> dict[str, float]:
        return {
            "ambiguous": self.ambiguous_probability,
            "identifiable": self.identifiable_probability,
            "noisy": self.noisy_probability,
            "paired": self.paired_probability,
            "multiregime": self.multiregime_probability,
        }


@dataclass
class TeacherTargets:
    """Mean-preserving projection of the exact candidate posterior."""

    projected_positive: torch.Tensor
    weights: torch.Tensor
    safe_scale: torch.Tensor
    mutual_information: torch.Tensor
    epistemic_variance: torch.Tensor
    epistemic_covariance: torch.Tensor
    joint: torch.Tensor
    exact_mutual_information: torch.Tensor
    exact_epistemic_variance: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)


def _weighted_joint(positive: torch.Tensor, weights: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    """``sum_h rho_h prod_q Bernoulli(theta_qh)`` for every binary label vector."""
    probability = positive.clamp(1e-6, 1 - 1e-6)
    # (batch, outcome, candidate, query)
    selected = outcomes[None, :, None, :] * probability[:, None] + (1.0 - outcomes[None, :, None, :]) * (
        1.0 - probability[:, None]
    )
    per_candidate = selected.log().sum(dim=-1)
    return (weights[:, None, :] * per_candidate.exp()).sum(dim=-1)


def teacher_targets(episode: ContinuousEpisode, base_positive: torch.Tensor) -> TeacherTargets:
    """Project the exact candidate posterior onto the fixed vanilla mean.

    The projection keeps the exact candidate weights ``rho_h`` and the direction
    of candidate disagreement, and contracts the deviations only as far as the
    fixed mean requires.
    """
    weights = episode.posterior.to(base_positive.dtype)
    candidate = episode.candidate_query_positive.to(base_positive.dtype)
    projected, scale = project_candidate_posterior(base_positive, candidate, weights)

    exact_mean = (weights[:, :, None] * candidate).sum(dim=1)
    exact_expected_entropy = (weights[:, :, None] * binary_entropy(candidate)).sum(dim=1)
    exact_mi = (binary_entropy(exact_mean) - exact_expected_entropy).clamp_min(0.0)
    exact_variance = (weights[:, :, None] * (candidate - exact_mean[:, None]).square()).sum(dim=1)

    expected_entropy = (weights[:, :, None] * binary_entropy(projected)).sum(dim=1)
    mutual_information = (binary_entropy(base_positive) - expected_entropy).clamp_min(0.0)
    deviation = projected - base_positive[:, None, :]
    variance = (weights[:, :, None] * deviation.square()).sum(dim=1)
    covariance = torch.einsum("bh,bhq,bhr->bqr", weights, deviation, deviation)
    outcomes = all_binary_outcomes(episode.query_count, device=base_positive.device).to(base_positive.dtype)
    joint = _weighted_joint(projected, weights, outcomes)
    return TeacherTargets(
        projected,
        weights,
        scale,
        mutual_information,
        variance,
        covariance,
        joint,
        exact_mi,
        exact_variance,
    )


def energy_distance(
    sample_positive: torch.Tensor,
    teacher_positive: torch.Tensor,
    teacher_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted energy distance between anonymous samples and candidates.

    Query vectors are compared with Euclidean distance normalized by
    ``sqrt(query_count)``, so the value does not grow with the query count.
    """
    query_count = sample_positive.shape[-1]
    normalizer = math.sqrt(query_count)
    num_samples = sample_positive.shape[1]
    cross = torch.cdist(sample_positive, teacher_positive) / normalizer
    within_model = torch.cdist(sample_positive, sample_positive) / normalizer
    within_teacher = torch.cdist(teacher_positive, teacher_positive) / normalizer
    cross_term = (cross * teacher_weights[:, None, :]).sum(dim=-1).mean(dim=-1)
    model_term = within_model.sum(dim=(-1, -2)) / (num_samples**2)
    teacher_term = torch.einsum("bh,bhg,bg->b", teacher_weights, within_teacher, teacher_weights)
    return (2.0 * cross_term - model_term - teacher_term).clamp_min(0.0).mean()


def continuous_losses(
    prediction,
    episode: ContinuousEpisode,
    config: ContinuousTrainingConfig,
    *,
    targets: TeacherTargets | None = None,
) -> tuple[torch.Tensor, dict[str, float], TeacherTargets]:
    """Permutation-free distributional loss for one episode batch."""
    base_positive = prediction.base_positive.detach()
    targets = teacher_targets(episode, base_positive) if targets is None else targets
    samples = prediction.sample_positive

    energy = energy_distance(samples, targets.projected_positive, targets.weights)
    mi_loss = F.mse_loss(prediction.mutual_information() / LOG_TWO, targets.mutual_information / LOG_TWO)
    variance_loss = F.mse_loss(
        prediction.epistemic_variance() / MAX_BINARY_VARIANCE, targets.epistemic_variance / MAX_BINARY_VARIANCE
    )
    covariance_loss = F.mse_loss(
        prediction.epistemic_covariance() / MAX_BINARY_VARIANCE, targets.epistemic_covariance / MAX_BINARY_VARIANCE
    )
    model_log_joint = prediction.joint_log_probabilities()
    joint_loss = -(targets.joint * model_log_joint).sum(dim=-1).mean()
    teacher_entropy = -(targets.joint * targets.joint.clamp_min(1e-12).log()).sum(dim=-1).mean()
    joint_kl = (joint_loss - teacher_entropy).clamp_min(0.0)

    mutual_information_weight = config.moment_weight * config.mutual_information_weight
    variance_weight = config.moment_weight * config.variance_weight
    total = (
        config.energy_weight * energy
        + mutual_information_weight * mi_loss
        + variance_weight * variance_loss
        + config.covariance_weight * covariance_loss
        + config.joint_weight * joint_loss
    )
    metrics = {
        "energy_distance": float(energy.detach()),
        "mutual_information_loss": float(mi_loss.detach()),
        "variance_loss": float(variance_loss.detach()),
        "covariance_loss": float(covariance_loss.detach()),
        "joint_query_loss": float(joint_loss.detach()),
        "joint_query_kl": float(joint_kl.detach()),
        "predicted_mutual_information": float(prediction.mutual_information().mean().detach()),
        "teacher_mutual_information": float(targets.mutual_information.mean().detach()),
        "expected_conditional_entropy": float(prediction.expected_conditional_entropy().mean().detach()),
        "teacher_safe_scale": float(targets.safe_scale.mean().detach()),
        "mean_preservation_error": float(prediction.mean_preservation_error().max().detach()),
        # Dispersion diagnostics: a healthy gate with a low shape ratio means
        # the samples are spiky and the safe bound is throttling the bulk.
        "dispersion_gate": float(prediction.dispersion_gate.mean().detach()),
        "dispersion_bound": float(prediction.dispersion_bound.mean().detach()),
        "deviation_shape_ratio": float(prediction.deviation_shape_ratio().mean().detach()),
        # The selection loss excludes the teacher's own joint entropy, which
        # depends on the episode's candidate count rather than on model quality.
        "selection_loss": float(
            (
                config.energy_weight * energy
                + config.mutual_information_weight * mi_loss
                + config.variance_weight * variance_loss
                + config.covariance_weight * covariance_loss
                + config.joint_weight * joint_kl / (episode.query_count * LOG_TWO)
            ).detach()
        ),
    }
    return total, metrics, targets


def evidence_monotonicity_loss(short_prediction, long_prediction) -> torch.Tensor:
    """Penalize mutual information that grows after identifying evidence."""
    increase = long_prediction.mutual_information() - short_prediction.mutual_information()
    return increase.clamp_min(0.0).mean()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def validate_config(config: ContinuousTrainingConfig) -> None:
    if config.model_type not in {"continuous", "beta"}:
        raise ValueError("model_type must be 'continuous' or 'beta'.")
    if config.uncertainty_mode not in {"frozen", "adapters", "full"}:
        raise ValueError("uncertainty_mode must be 'frozen', 'adapters', or 'full'.")
    if config.context_mode not in {"deepsets", "cross_attention"}:
        raise ValueError("context_mode must be 'deepsets' or 'cross_attention'.")
    if config.model_type == "beta" and config.uncertainty_mode != "adapters":
        raise ValueError("The Beta ablation is defined with the adapter-equipped uncertainty encoder.")
    for name in ("batch_size", "validation_interval", "patience", "accumulate_gradients", "validation_episodes"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be at least one.")
    if config.max_steps < 1:
        raise ValueError("max_steps must be at least one.")
    if config.min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if config.num_samples < 2 or config.num_samples % 2:
        raise ValueError("num_samples must be an even number of at least two.")
    weights = config.curriculum()
    if any(value < 0 for value in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("curriculum probabilities must be non-negative and sum to one.")
    if not 4 <= config.validation_query_count <= 8:
        raise ValueError("validation_query_count must lie between four and eight.")
    available_support_sizes(config.max_support_size)


def build_model(config: ContinuousTrainingConfig, backbone) -> Any:
    if config.model_type == "continuous":
        return NanoTabPFNContinuousPosteriorModel(
            backbone,
            uncertainty_mode=config.uncertainty_mode,
            adapter_bottleneck=config.adapter_bottleneck,
            context_mode=config.context_mode,
            latent_dim=config.latent_dim,
            num_samples=config.num_samples,
            inference_seed=config.inference_seed,
        )
    return NanoTabPFNBetaConcentrationModel(
        backbone,
        uncertainty_mode=config.uncertainty_mode,
        adapter_bottleneck=config.adapter_bottleneck,
        context_mode=config.context_mode,
        num_samples=config.num_samples,
        inference_seed=config.inference_seed,
    )


def _regime(config: ContinuousTrainingConfig, base: EpisodeRegime) -> EpisodeRegime:
    if config.include_scm_families:
        return base
    families = tuple(name for name in base.families if not name.endswith("_scm"))
    if not families:
        raise ValueError("Disabling SCM families left the regime empty.")
    return EpisodeRegime(families, base.min_features, base.max_features, base.imbalance_range, base.scale_exponent)


def _episode_loss(
    model,
    config: ContinuousTrainingConfig,
    rng: np.random.Generator,
    regime: EpisodeRegime,
    condition: str,
    *,
    sample_seed: int,
    support_size: int | None = None,
    query_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the full objective for one drawn episode (or paired episodes)."""
    if condition == "paired":
        short, long = sample_paired_episode(
            rng,
            regime=regime,
            batch_size=config.batch_size,
            support_size=support_size,
            query_count=query_count,
            device=config.device,
            max_support_size=config.max_support_size,
        )
        short_prediction = model(
            short.support_x, short.support_y, short.query_x, sample_seed=sample_seed, num_samples=config.num_samples
        )
        long_prediction = model(
            long.support_x, long.support_y, long.query_x, sample_seed=sample_seed, num_samples=config.num_samples
        )
        short_loss, short_metrics, _ = continuous_losses(short_prediction, short, config)
        long_loss, long_metrics, _ = continuous_losses(long_prediction, long, config)
        monotonicity = evidence_monotonicity_loss(short_prediction, long_prediction)
        total = 0.5 * (short_loss + long_loss) + config.monotonicity_weight * monotonicity
        metrics = {key: 0.5 * (short_metrics[key] + long_metrics[key]) for key in short_metrics}
        metrics["evidence_monotonicity_loss"] = float(monotonicity.detach())
        metrics["paired_mutual_information_drop"] = float(
            (short_prediction.mutual_information().mean() - long_prediction.mutual_information().mean()).detach()
        )
        metrics["selection_loss"] = metrics["selection_loss"] + config.monotonicity_weight * float(
            monotonicity.detach()
        )
        return total, metrics
    if condition == "multiregime":
        episode = sample_multiregime_episode(
            rng,
            regime=regime,
            batch_size=config.batch_size,
            support_size=support_size,
            query_count=query_count,
            device=config.device,
            max_support_size=config.max_support_size,
        )
        prediction = model(
            episode.support_x,
            episode.support_y,
            episode.query_x,
            sample_seed=sample_seed,
            num_samples=config.num_samples,
        )
        total, metrics, _ = continuous_losses(prediction, episode, config)
        metrics["evidence_monotonicity_loss"] = 0.0
        metrics.update(_multiregime_error_gap(prediction, episode))
        return total, metrics
    episode = sample_episode(
        rng,
        regime=regime,
        condition=condition,
        batch_size=config.batch_size,
        support_size=support_size,
        query_count=query_count,
        device=config.device,
        max_support_size=config.max_support_size,
    )
    prediction = model(
        episode.support_x,
        episode.support_y,
        episode.query_x,
        sample_seed=sample_seed,
        num_samples=config.num_samples,
    )
    total, metrics, _ = continuous_losses(prediction, episode, config)
    metrics["evidence_monotonicity_loss"] = 0.0
    return total, metrics


def _multiregime_error_gap(prediction, episode: ContinuousEpisode) -> dict[str, float]:
    """Diagnostic only, never part of the loss.

    Two different gaps, and only one of them can move during training:

    - ``multiregime_*_error`` / ``multiregime_error_gap``: mean absolute error
      of the *frozen* vanilla mean (``prediction.base_positive``) on base- vs.
      other-regime query rows. This is a fixed property of the untouched mean
      path and is identical every validation call regardless of how well the
      uncertainty head trains -- a baseline characterization only, not a
      training signal.
    - ``multiregime_*_mutual_information`` / ``multiregime_mutual_information_gap``:
      the *trainable* uncertainty head's predicted mutual information on base-
      vs. other-regime rows. This is what should move if the head learns to
      flag contaminated queries as more uncertain; a model with no such signal
      shows a gap near zero throughout training, the same negative result as
      every other condition in this trial.
    """
    if episode.query_regime_source is None:
        return {}
    source = episode.query_regime_source
    base_mask = source == 0
    other_mask = source == 1
    result: dict[str, float] = {}

    information = prediction.mutual_information().detach()
    if base_mask.any():
        result["multiregime_base_mutual_information"] = float(information[base_mask].mean())
    if other_mask.any():
        result["multiregime_other_mutual_information"] = float(information[other_mask].mean())
    if base_mask.any() and other_mask.any():
        result["multiregime_mutual_information_gap"] = (
            result["multiregime_other_mutual_information"] - result["multiregime_base_mutual_information"]
        )

    predicted_mean = prediction.base_positive.detach()
    error = (predicted_mean - episode.query_y.to(predicted_mean.dtype)).abs()
    if base_mask.any():
        result["multiregime_base_error"] = float(error[base_mask].mean())
    if other_mask.any():
        result["multiregime_other_error"] = float(error[other_mask].mean())
    if base_mask.any() and other_mask.any():
        result["multiregime_error_gap"] = result["multiregime_other_error"] - result["multiregime_base_error"]
    return result


@torch.no_grad()
def validation_metrics(model, config: ContinuousTrainingConfig) -> dict[str, float]:
    """Held-out-*family* validation; training never sees these generators."""
    model.eval()
    rng = np.random.default_rng(config.seed + 10_001)
    regime = _regime(config, HELDOUT_REGIME)
    totals: dict[str, list[float]] = {}
    conditions = ("ambiguous", "identifiable", "noisy", "paired", "multiregime")
    for index in range(config.validation_episodes):
        condition = conditions[index % len(conditions)]
        _loss, metrics = _episode_loss(
            model,
            config,
            rng,
            regime,
            condition,
            sample_seed=config.inference_seed,
            support_size=config.validation_support_size,
            query_count=config.validation_query_count,
        )
        for key, value in metrics.items():
            totals.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(values)) for key, values in totals.items()}


def train(model, config: ContinuousTrainingConfig) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    parameters = model.trainable_parameters()
    if not parameters:
        raise ValueError("The selected configuration has no trainable parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed + 1)
    regime = _regime(config, TRAIN_REGIME)
    weights = config.curriculum()
    model.to(config.device)
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_metrics: dict[str, Any] = {}
    best_value = math.inf
    best_step = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for micro in range(config.accumulate_gradients):
            condition = curriculum_condition(rng, weights)
            loss, metrics = _episode_loss(
                model,
                config,
                rng,
                regime,
                condition,
                sample_seed=config.seed + 7919 * step + micro,
            )
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        row: dict[str, Any] = {"step": step, "gradient_norm": float(grad_norm), **totals}
        if step % config.validation_interval == 0 or step == config.max_steps:
            validation = validation_metrics(model, config)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation["selection_loss"] < best_value - config.min_delta:
                best_value = validation["selection_loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                best_metrics = validation
                best_step = step
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    selection = {
        "best_step": best_step,
        "best_validation_selection_loss": best_value,
        "validation_metrics": best_metrics,
        "optimizer_state": best_optimizer,
        "steps_run": history[-1]["step"] if history else 0,
    }
    return model, history, selection


def _environment_report(config: ContinuousTrainingConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": config.device,
    }
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
    return report


def run_training(config: ContinuousTrainingConfig) -> Path:
    validate_config(config)
    if config.require_cuda:
        assert torch.cuda.is_available(), "CUDA is required for this run but torch.cuda.is_available() is False."
    set_randomness_seed(config.seed)
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    output = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "continuous_bayesian" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    environment = _environment_report(config)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment, sort_keys=True), flush=True)

    backbone = init_model_from_state_dict_file(str(checkpoint_path))
    model = build_model(config, backbone).to(config.device)
    frozen_mean_reference = {
        name: value.detach().cpu().clone() for name, value in model.mean_backbone.state_dict().items()
    }
    model, history, selection = train(model, config)

    mean_backbone_unchanged = all(
        torch.equal(frozen_mean_reference[name], value.detach().cpu())
        for name, value in model.mean_backbone.state_dict().items()
    )
    optimizer_state = selection.pop("optimizer_state")
    best_path = output / "best.pth"
    save_continuous_checkpoint(
        best_path,
        model,
        training_config=asdict(config),
        source_checkpoint_path=str(checkpoint_path),
        source_checkpoint_sha256=source_hash,
        stage=f"{config.model_type}_{config.uncertainty_mode}",
        step=selection["best_step"],
        optimizer_state=optimizer_state,
        validation_metrics=selection["validation_metrics"],
        random_seeds={"training_seed": config.seed, "inference_seed": config.inference_seed},
        selection=selection,
    )
    summary = {
        "selected_checkpoint": str(best_path),
        "mean_backbone_unchanged": mean_backbone_unchanged,
        "environment": environment,
        **selection,
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "learning_curves.csv").write_text(_history_csv(history))
    (output / "source_checkpoint.json").write_text(
        json.dumps({"path": str(checkpoint_path), "sha256": source_hash}, indent=2) + "\n"
    )
    return output.resolve()


def _history_csv(rows: list[dict[str, Any]]) -> str:
    keys = sorted({key for row in rows for key in row})
    body = "\n".join(",".join(str(row.get(key, "")) for key in keys) for row in rows)
    return ",".join(keys) + "\n" + body + ("\n" if rows else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ContinuousTrainingConfig()
    integer_fields = (
        "seed",
        "adapter_bottleneck",
        "latent_dim",
        "num_samples",
        "inference_seed",
        "max_steps",
        "validation_interval",
        "patience",
        "batch_size",
        "accumulate_gradients",
        "validation_episodes",
        "validation_support_size",
        "validation_query_count",
        "max_support_size",
    )
    for name in integer_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    float_fields = (
        "learning_rate",
        "weight_decay",
        "min_delta",
        "gradient_clip",
        "ambiguous_probability",
        "identifiable_probability",
        "noisy_probability",
        "paired_probability",
        "multiregime_probability",
        "energy_weight",
        "mutual_information_weight",
        "variance_weight",
        "covariance_weight",
        "joint_weight",
        "monotonicity_weight",
        "moment_weight",
    )
    for name in float_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--model-type", choices=("continuous", "beta"), default=defaults.model_type)
    parser.add_argument("--uncertainty-mode", choices=("frozen", "adapters", "full"), default=defaults.uncertainty_mode)
    parser.add_argument("--context-mode", choices=("deepsets", "cross_attention"), default=defaults.context_mode)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-scm-families", dest="include_scm_families", action="store_false")
    parser.set_defaults(include_scm_families=defaults.include_scm_families)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    output_path = run_training(ContinuousTrainingConfig(**vars(arguments)))
    print(f"Wrote continuous Bayesian training artifacts to {output_path}")
