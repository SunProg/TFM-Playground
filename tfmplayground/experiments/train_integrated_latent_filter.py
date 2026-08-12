"""Train an integrated latent-token nanoTabPFN with progressive unfreezing."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from tfmplayground.experiments.calibrate_sequential_latent_filter import (
    QUERY_TEMPERATURES,
    CachedCondition,
    candidate_record,
    evaluate_cached_split,
    rank_candidates,
)
from tfmplayground.experiments.train_sequential_latent_filter import (
    CONDITIONS,
    SequentialEpisodeBatch,
    SequentialFilterConfig,
    _outcome_indices,
    _save_plots,
    controlled_gate_report,
    evaluate_controlled,
    evaluate_ordinary_accuracy,
    generate_controlled_episodes,
    make_tabicl_iterator,
    next_tabicl_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.integrated_latent_filter import (
    NanoTabPFNIntegratedLatentFilter,
    load_integrated_checkpoint,
    save_integrated_checkpoint,
)
from tfmplayground.models.sequential_latent_filter import SequentialFilterLogits
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class IntegratedTrainingConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    prior_count: int = 128
    update_count: int = 128
    query_count: int = 4
    batch_size: int = 2
    accumulate_gradients: int = 8
    frozen_steps: int = 1000
    partial_extra_steps: int = 500
    full_extra_steps: int = 500
    validation_interval: int = 50
    patience: int = 6
    head_learning_rate: float = 1e-4
    partial_backbone_learning_rate: float = 1e-5
    full_backbone_learning_rate: float = 5e-6
    coherence_weight: float = 0.10
    residual_weight: float = 0.01
    diversity_weight: float = 0.05
    diversity_target: float = 0.20
    calibration_trials: int = 128
    selection_trials: int = 128
    final_trials: int = 256
    ordinary_evaluation_batches: int = 8
    tabicl_steps: int = 500
    plots: bool = True


def validate_config(config: IntegratedTrainingConfig) -> None:
    if config.prior_count != 128 or config.update_count != 128:
        raise ValueError("The integrated experiment requires the preregistered 128:128 design.")
    if config.update_count > config.prior_count:
        raise ValueError("Updates cannot exceed prior rows.")
    if config.query_count != 4:
        raise ValueError("Exactly four final queries are required.")
    for name in (
        "batch_size",
        "accumulate_gradients",
        "validation_interval",
        "patience",
        "calibration_trials",
        "selection_trials",
        "final_trials",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def _episode_config(
    config: IntegratedTrainingConfig, *, seed: int, trials: int, ordinary_batches: int | None = None
) -> SequentialFilterConfig:
    return SequentialFilterConfig(
        seed=seed,
        checkpoint=config.checkpoint,
        device=config.device,
        initial_support_count=config.prior_count,
        stream_count=config.update_count,
        query_count=config.query_count,
        batch_size=config.batch_size,
        controlled_steps=0,
        tabicl_steps=config.tabicl_steps,
        evaluation_trials=trials,
        ordinary_evaluation_batches=(
            config.ordinary_evaluation_batches if ordinary_batches is None else ordinary_batches
        ),
        plots=config.plots,
    )


def _min_pairwise_slot_js(slot_joint: torch.Tensor) -> torch.Tensor:
    """Minimum Jensen-Shannon divergence over all particle pairs, shape (batch,).

    `slot_joint` is (batch, num_particles, num_outcomes) of probabilities. Reducing by the
    minimum keeps the diversity signal honest for K>2: the closest pair is what matters, since
    any two collapsed particles are a collapse regardless of how far the others are.
    """
    num_particles = slot_joint.shape[1]
    if num_particles < 2:
        return slot_joint.new_zeros(slot_joint.shape[0])
    pairwise = []
    for left in range(num_particles):
        for right in range(left + 1, num_particles):
            first, second = slot_joint[:, left], slot_joint[:, right]
            midpoint = 0.5 * (first + second)
            pairwise.append(
                0.5
                * (
                    (first * (first.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
                    + (second * (second.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
                )
            )
    return torch.stack(pairwise, dim=-1).amin(-1)


def integrated_loss(
    model: NanoTabPFNIntegratedLatentFilter,
    batch: SequentialEpisodeBatch,
    config: IntegratedTrainingConfig,
    *,
    controlled: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x,
    )
    prequential = -prediction.prequential_log_likelihood.mean()
    joint = prediction.joint_probabilities()
    indices = _outcome_indices(batch.query_y)
    selected = joint.gather(-1, indices[:, None, None].expand(-1, joint.shape[1], 1)).squeeze(-1)
    trajectory = -selected.clamp_min(1e-12).log().mean()
    marginals = prediction.marginal_probabilities()
    labels = batch.query_y[:, None, :, None].expand(-1, marginals.shape[1], -1, 1)
    marginal = -marginals.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
    slot_joint = prediction.slot_joint_log_probabilities().exp()
    # Minimum pairwise JS across particles. For K=2 this is exactly the single pair, so the
    # value is unchanged from the previous hardcoded slot_joint[:, 0] vs slot_joint[:, 1];
    # for K>2 it correctly measures the *closest* pair rather than ignoring particles 2..K-1.
    slot_js = _min_pairwise_slot_js(slot_joint)
    diversity = F.relu(config.diversity_target - slot_js).mean()
    incoherent = (1.0 - slot_joint[:, :, 0] - slot_joint[:, :, -1]).clamp_min(0).mean()
    residual = prediction.stream_residuals.abs().mean()
    total = prequential + trajectory + marginal + config.diversity_weight * diversity
    if controlled:
        total = total + config.coherence_weight * incoherent + config.residual_weight * residual
    return total, {
        "loss": float(total.detach()),
        "prequential_loss": float(prequential.detach()),
        "trajectory_loss": float(trajectory.detach()),
        "marginal_loss": float(marginal.detach()),
        "diversity_loss": float(diversity.detach()),
        "coherence_loss": float(incoherent.detach()),
        "residual_loss": float(residual.detach()),
        "slot_joint_js": float(slot_js.mean().detach()),
    }


def _batch_for_microstep(config: IntegratedTrainingConfig, global_microstep: int) -> SequentialEpisodeBatch:
    episode_config = _episode_config(config, seed=config.seed, trials=config.batch_size)
    return generate_controlled_episodes(
        episode_config,
        np.random.default_rng(config.seed + 10_000 + global_microstep),
        batch_size=config.batch_size,
    )


@torch.no_grad()
def _validation_loss(model: NanoTabPFNIntegratedLatentFilter, config: IntegratedTrainingConfig) -> float:
    model.eval()
    values = []
    episode_config = _episode_config(config, seed=config.seed + 50_000, trials=config.batch_size)
    for index, condition in enumerate(CONDITIONS):
        batch = generate_controlled_episodes(
            episode_config,
            np.random.default_rng(config.seed + 50_000 + index),
            condition=condition,
            batch_size=config.batch_size,
        )
        value, _ = integrated_loss(model, batch, config)
        values.append(float(value))
    model.train()
    return float(np.mean(values))


@torch.no_grad()
def _cache_controlled_split(
    model: NanoTabPFNIntegratedLatentFilter,
    config: IntegratedTrainingConfig,
    *,
    seed_base: int,
    trials: int,
) -> list[CachedCondition]:
    """Cache raw integrated logits once for query-temperature calibration."""
    model.to(config.device).eval()
    episode_config = _episode_config(config, seed=seed_base, trials=trials)
    cached = []
    for condition_index, condition in enumerate(CONDITIONS):
        batch = generate_controlled_episodes(
            episode_config,
            np.random.default_rng(seed_base + condition_index),
            condition=condition,
            batch_size=trials,
        )
        raw = model.raw_logits(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.query_x,
        )
        cached.append(
            CachedCondition(
                condition=condition,
                raw=SequentialFilterLogits(raw.stream_logits.detach(), raw.query_logits.detach(), raw.slots.detach()),
                stream_y=batch.stream_y,
                query_y=batch.query_y,
                latent_class=batch.latent_class,
                initial_evidence_class=batch.initial_evidence_class,
                exact_p1=batch.exact_p1,
            )
        )
    return cached


def train_segment(
    model: NanoTabPFNIntegratedLatentFilter,
    config: IntegratedTrainingConfig,
    *,
    candidate: str,
    trainability: str,
    start_step: int,
    end_step: int,
) -> tuple[NanoTabPFNIntegratedLatentFilter, list[dict[str, Any]], torch.optim.Optimizer]:
    model.set_query_temperature(1.0)
    model.set_trainability(trainability)
    model.to(config.device).train()
    head = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    backbone = [parameter for name, parameter in model.named_parameters() if name.startswith("backbone.")]
    backbone_lr = (
        config.partial_backbone_learning_rate if trainability == "partial" else config.full_backbone_learning_rate
    )
    parameter_groups = [{"params": [p for p in head if p.requires_grad], "lr": config.head_learning_rate}]
    trainable_backbone = [p for p in backbone if p.requires_grad]
    if trainable_backbone:
        parameter_groups.append({"params": trainable_backbone, "lr": backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-2)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale = 0
    trainable = list(model.trainable_parameters())
    for step in range(start_step + 1, end_step + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for accumulation in range(config.accumulate_gradients):
            global_microstep = (step - 1) * config.accumulate_gradients + accumulation
            batch = _batch_for_microstep(config, global_microstep)
            loss, metrics = integrated_loss(model, batch, config)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row: dict[str, Any] = {
            "candidate": candidate,
            "trainability": trainability,
            "step": step,
            **totals,
        }
        if step % config.validation_interval == 0 or step == end_step:
            row["validation_loss"] = _validation_loss(model, config)
            if row["validation_loss"] < best_validation:
                best_validation = row["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history, optimizer


def _selection_rank(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    return frame.sort_values(
        by=[
            "passed",
            "passed_checks",
            "max_normalized_violation",
            "final_marginal_js",
            "ordinary_accuracy",
            "trainable_parameters",
        ],
        ascending=[False, False, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def evaluate_candidate(
    model: NanoTabPFNIntegratedLatentFilter,
    config: IntegratedTrainingConfig,
    *,
    name: str,
    stage_index: int,
    baseline,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration_seed = config.seed + 100_000 + stage_index * 10_000
    cache = _cache_controlled_split(
        model,
        config,
        seed_base=calibration_seed,
        trials=config.calibration_trials,
    )
    temperature_records = []
    for temperature in QUERY_TEMPERATURES:
        trials, _, report = evaluate_cached_split(
            cache,
            evidence_logit_scale=1.0,
            query_temperature=temperature,
            stream_count=config.update_count,
            invariance={
                "query_permutation_max_error": 0.0,
                "evidence_order_weight_max_error": 0.0,
                "evidence_order_joint_max_error": 0.0,
            },
        )
        temperature_records.append(candidate_record(name, 1.0, temperature, trials, report))
    temperature_ranking = rank_candidates(temperature_records)
    selected_temperature = float(temperature_ranking.iloc[0].query_temperature)
    model.set_query_temperature(selected_temperature)

    selection_config = _episode_config(
        config, seed=config.seed + 200_000 + stage_index * 10_000 - 1000, trials=config.selection_trials
    )
    trials, summary, invariance = evaluate_controlled(model, selection_config)
    report = controlled_gate_report(trials, invariance, stream_count=config.update_count)
    ordinary_config = _episode_config(
        config,
        seed=config.seed + 300_000,
        trials=config.selection_trials,
        ordinary_batches=min(2, config.ordinary_evaluation_batches),
    )
    set_randomness_seed(config.seed + 300_000)
    ordinary = evaluate_ordinary_accuracy({name: model}, baseline, ordinary_config)
    record = candidate_record(name, 1.0, selected_temperature, trials, report)
    record["ordinary_accuracy"] = ordinary[name]
    record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record["trainability"] = model.trainability_stage
    return (
        record,
        trials,
        summary,
        {
            "gate": report,
            "temperature_ranking": temperature_ranking,
            "ordinary": ordinary,
        },
    )


def _save_candidate(
    output_dir: Path,
    name: str,
    model: NanoTabPFNIntegratedLatentFilter,
    config: IntegratedTrainingConfig,
    source_hash: str,
    lineage: dict[str, Any],
    history: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    evaluation: dict[str, Any],
) -> Path:
    pd.DataFrame(history).to_csv(output_dir / f"{name}_learning_curves.csv", index=False)
    trials.to_csv(output_dir / f"{name}_selection_metrics.csv", index=False)
    summary.to_csv(output_dir / f"{name}_selection_summary.csv", index=False)
    evaluation["temperature_ranking"].to_csv(output_dir / f"{name}_temperature_ranking.csv", index=False)
    (output_dir / f"{name}_gate.json").write_text(json.dumps(evaluation["gate"], indent=2) + "\n")
    path = output_dir / f"{name}_checkpoint.pth"
    save_integrated_checkpoint(
        path,
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=source_hash,
        stage=name,
        lineage=lineage,
        controlled_gate=evaluation["gate"],
        optimizer_state=optimizer.state_dict(),
    )
    return path


def train_tabicl(
    model: NanoTabPFNIntegratedLatentFilter,
    config: IntegratedTrainingConfig,
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.head_learning_rate)
    episode_config = _episode_config(config, seed=config.seed + 800_000, trials=config.batch_size)
    iterator = make_tabicl_iterator(episode_config, config.tabicl_steps * 100)
    rng = np.random.default_rng(config.seed + 800_000)
    history = []
    for step in range(1, config.tabicl_steps + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _ in range(config.accumulate_gradients):
            batch = (
                generate_controlled_episodes(episode_config, rng, batch_size=config.batch_size)
                if step % 2
                else next_tabicl_episode(iterator, episode_config, rng)
            )
            loss, metrics = integrated_loss(model, batch, config, controlled=bool(step % 2))
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()
        history.append({"step": step, "source": "controlled" if step % 2 else "tabicl", **totals})
    return history, optimizer


def resume_tabicl_stage(output_dir: str | Path, *, device: str | None = None) -> Path:
    """Run the gated TabICL stage from an already selected controlled checkpoint."""
    output_dir = Path(output_dir).resolve()
    config_values = json.loads((output_dir / "config.json").read_text())
    if device is not None:
        config_values["device"] = device
    config_values["output_dir"] = None
    config = IntegratedTrainingConfig(**config_values)
    final_gate = json.loads((output_dir / "final_gate.json").read_text())
    if not final_gate["passed"]:
        raise ValueError("TabICL requires a selected controlled model that passed every final gate.")
    model, metadata = load_integrated_checkpoint(output_dir / "selected_checkpoint.pth")
    model.to(config.device)
    history, optimizer = train_tabicl(model, config)
    pd.DataFrame(history).to_csv(output_dir / "tabicl_learning_curves.csv", index=False)
    save_integrated_checkpoint(
        output_dir / "tabicl_checkpoint.pth",
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=metadata["source_checkpoint_sha256"],
        stage="tabicl",
        lineage={"parent": "selected_checkpoint.pth"},
        controlled_gate=final_gate,
        optimizer_state=optimizer.state_dict(),
    )
    selection = {
        "selected_candidate": metadata["lineage"]["selected_from"],
        "final_gate_passed": True,
        "tabicl_ran": True,
        "failure_reasons": [],
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return output_dir


@torch.no_grad()
def complete_postrun_artifacts(output_dir: str | Path, *, device: str | None = None) -> Path:
    """Add comparable ordinary accuracy and per-stage latent/residual diagnostics."""
    output_dir = Path(output_dir).resolve()
    values = json.loads((output_dir / "config.json").read_text())
    if device is not None:
        values["device"] = device
    values["output_dir"] = None
    config = IntegratedTrainingConfig(**values)
    baseline = init_model_from_state_dict_file(config.checkpoint)
    selected, _ = load_integrated_checkpoint(output_dir / "selected_checkpoint.pth")
    models = {"controlled": selected}
    tabicl_path = output_dir / "tabicl_checkpoint.pth"
    if tabicl_path.exists():
        models["tabicl"] = load_integrated_checkpoint(tabicl_path)[0]
    ordinary_config = _episode_config(config, seed=config.seed + 750_000, trials=config.final_trials)
    set_randomness_seed(config.seed + 750_000)
    ordinary = evaluate_ordinary_accuracy(models, baseline, ordinary_config)
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary, indent=2) + "\n")

    rows = []
    episode_config = _episode_config(config, seed=config.seed + 760_000, trials=16)
    for checkpoint_path in sorted(output_dir.glob("*_checkpoint.pth")):
        if checkpoint_path.name == "initial_checkpoint.pth":
            stage_name = "initial"
        else:
            stage_name = checkpoint_path.stem.removesuffix("_checkpoint")
        model, metadata = load_integrated_checkpoint(checkpoint_path)
        model.to(config.device).eval()
        for condition_index, condition in enumerate(CONDITIONS):
            batch = generate_controlled_episodes(
                episode_config,
                np.random.default_rng(config.seed + 760_000 + condition_index),
                condition=condition,
                batch_size=16,
            )
            raw = model.raw_logits(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.query_x,
            )
            cosine = F.cosine_similarity(raw.latent_states[:, :, 0], raw.latent_states[:, :, 1], dim=-1)
            for layer in range(raw.latent_states.shape[1]):
                rows.append(
                    {
                        "stage": stage_name,
                        "condition": condition,
                        "layer": layer,
                        "adapter_gate": float(raw.adapter_gates[layer].cpu()),
                        "latent_pair_cosine": float(cosine[:, layer].mean().cpu()),
                        "stream_residual_abs_mean": float(raw.stream_residuals.abs().mean().cpu()),
                        "query_residual_abs_mean": float(raw.query_residuals.abs().mean().cpu()),
                        "stream_disagreement_gate_mean": float(raw.stream_disagreement_gates.mean().cpu()),
                        "query_disagreement_gate_mean": float(raw.query_disagreement_gates.mean().cpu()),
                        "backbone_sha256": metadata["backbone_sha256"],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "latent_residual_diagnostics.csv", index=False)
    return output_dir


def run(config: IntegratedTrainingConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "integrated_latent_filter" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    baseline = init_model_from_state_dict_file(str(checkpoint_path))
    initial = NanoTabPFNIntegratedLatentFilter(copy.deepcopy(baseline)).to(config.device)
    save_integrated_checkpoint(
        output_dir / "initial_checkpoint.pth",
        initial,
        training_config=asdict(config),
        source_checkpoint_sha256=source_hash,
        stage="initial",
        lineage={"parent": None},
    )

    candidates: list[dict[str, Any]] = []
    candidate_paths: dict[str, Path] = {}
    stage_counter = 0

    def train_and_evaluate(
        name: str, parent: Path, trainability: str, start_step: int, end_step: int
    ) -> dict[str, Any]:
        nonlocal stage_counter
        model, _ = load_integrated_checkpoint(parent)
        model, history, optimizer = train_segment(
            model,
            config,
            candidate=name,
            trainability=trainability,
            start_step=start_step,
            end_step=end_step,
        )
        record, trials, summary, evaluation = evaluate_candidate(
            model, config, name=name, stage_index=stage_counter, baseline=baseline
        )
        stage_counter += 1
        path = _save_candidate(
            output_dir,
            name,
            model,
            config,
            source_hash,
            {"parent": str(parent), "start_step": start_step, "end_step": end_step},
            history,
            optimizer,
            trials,
            summary,
            evaluation,
        )
        record["checkpoint"] = str(path)
        candidate_paths[name] = path
        candidates.append(record)
        return record

    initial_path = output_dir / "initial_checkpoint.pth"
    frozen = train_and_evaluate("frozen", initial_path, "frozen", 0, config.frozen_steps)
    level_candidates = [frozen]
    if not frozen["passed"]:
        partial_curriculum = train_and_evaluate(
            "partial_curriculum",
            candidate_paths["frozen"],
            "partial",
            config.frozen_steps,
            config.frozen_steps + config.partial_extra_steps,
        )
        partial_fresh = train_and_evaluate(
            "partial_fresh",
            initial_path,
            "partial",
            0,
            config.frozen_steps + config.partial_extra_steps,
        )
        level_candidates = [partial_curriculum, partial_fresh]
        if not any(record["passed"] for record in level_candidates):
            full_curriculum = train_and_evaluate(
                "full_curriculum",
                candidate_paths["partial_curriculum"],
                "full",
                config.frozen_steps + config.partial_extra_steps,
                config.frozen_steps + config.partial_extra_steps + config.full_extra_steps,
            )
            full_fresh = train_and_evaluate(
                "full_fresh",
                initial_path,
                "full",
                0,
                config.frozen_steps + config.partial_extra_steps + config.full_extra_steps,
            )
            level_candidates = [full_curriculum, full_fresh]

    passing = [record for record in level_candidates if record["passed"]]
    ranking = _selection_rank(passing if passing else candidates)
    ranking.to_csv(output_dir / "candidate_ranking.csv", index=False)
    selected_name = str(ranking.iloc[0].candidate)
    selected, _ = load_integrated_checkpoint(candidate_paths[selected_name])
    final_config = _episode_config(config, seed=config.seed + 700_000 - 1000, trials=config.final_trials)
    final_trials, final_summary, final_invariance = evaluate_controlled(selected, final_config)
    final_gate = controlled_gate_report(final_trials, final_invariance, stream_count=config.update_count)
    final_trials.to_csv(output_dir / "final_trajectory_metrics.csv", index=False)
    final_summary.to_csv(output_dir / "final_summary.csv", index=False)
    (output_dir / "final_gate.json").write_text(json.dumps(final_gate, indent=2) + "\n")
    if config.plots:
        _save_plots(final_summary, output_dir, "final")
    save_integrated_checkpoint(
        output_dir / "selected_checkpoint.pth",
        selected,
        training_config=asdict(config),
        source_checkpoint_sha256=source_hash,
        stage="selected",
        lineage={"selected_from": selected_name, "checkpoint": str(candidate_paths[selected_name])},
        controlled_gate=final_gate,
    )

    ordinary_config = _episode_config(config, seed=config.seed + 750_000, trials=config.final_trials)
    ordinary = evaluate_ordinary_accuracy({"selected": selected}, baseline, ordinary_config)
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary, indent=2) + "\n")
    tabicl_ran = False
    if final_gate["passed"]:
        history, optimizer = train_tabicl(selected, config)
        pd.DataFrame(history).to_csv(output_dir / "tabicl_learning_curves.csv", index=False)
        save_integrated_checkpoint(
            output_dir / "tabicl_checkpoint.pth",
            selected,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="tabicl",
            lineage={"parent": "selected_checkpoint.pth"},
            controlled_gate=final_gate,
            optimizer_state=optimizer.state_dict(),
        )
        tabicl_ran = True
    selection = {
        "selected_candidate": selected_name,
        "final_gate_passed": bool(final_gate["passed"]),
        "tabicl_ran": tabicl_ran,
        "failure_reasons": final_gate["failure_reasons"],
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--prior-count", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--frozen-steps", type=int, default=1000)
    parser.add_argument("--partial-extra-steps", type=int, default=500)
    parser.add_argument("--full-extra-steps", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--partial-backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--full-backbone-learning-rate", type=float, default=5e-6)
    parser.add_argument("--coherence-weight", type=float, default=0.10)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--diversity-target", type=float, default=0.20)
    parser.add_argument("--calibration-trials", type=int, default=128)
    parser.add_argument("--selection-trials", type=int, default=128)
    parser.add_argument("--final-trials", type=int, default=256)
    parser.add_argument("--ordinary-evaluation-batches", type=int, default=8)
    parser.add_argument("--tabicl-steps", type=int, default=500)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    output_dir = run(IntegratedTrainingConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote integrated latent-filter artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
