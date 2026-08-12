"""Post-hoc temperature calibration for the 128:128 sequential latent filter."""

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
from torch import nn

from tfmplayground.experiments.train_sequential_latent_filter import (
    CONDITIONS,
    SequentialFilterConfig,
    _effective_milestones,
    _js_divergence,
    _outcome_indices,
    _save_plots,
    controlled_gate_report,
    evaluate_controlled,
    evaluate_ordinary_accuracy,
    generate_controlled_episodes,
    train_head,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.sequential_latent_filter import (
    NanoTabPFNSequentialLatentFilter,
    SequentialFilterLogits,
    filter_sequential_logits,
    load_sequential_filter_checkpoint,
    save_sequential_filter_checkpoint,
)
from tfmplayground.utils import get_default_device, set_randomness_seed

EVIDENCE_LOGIT_SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00)
QUERY_TEMPERATURES = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 2402
    checkpoint: str = (
        "runs/sequential_latent_filter/full-controlled-prior128-update128-valid-20260807/controlled_checkpoint.pth"
    )
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    prior_count: int = 128
    update_count: int = 128
    query_count: int = 4
    grid_trials: int = 128
    optimization_trials: int = 128
    selection_trials: int = 128
    final_trials: int = 256
    scalar_steps: int = 200
    scalar_batch_size: int = 64
    scalar_learning_rate: float = 0.02
    ordinary_evaluation_batches: int = 8
    tabicl_steps: int = 500
    run_tabicl_on_pass: bool = True
    plots: bool = True


@dataclass(frozen=True)
class CachedCondition:
    condition: str
    raw: SequentialFilterLogits
    stream_y: torch.Tensor
    query_y: torch.Tensor
    latent_class: torch.Tensor
    initial_evidence_class: torch.Tensor
    exact_p1: torch.Tensor


class BoundedTemperatures(nn.Module):
    """Two scalar calibration parameters with fixed preregistered bounds."""

    def __init__(self, evidence_logit_scale: float, query_temperature: float):
        super().__init__()
        self.evidence_raw = nn.Parameter(self._inverse(evidence_logit_scale, 0.01, 1.0))
        self.query_raw = nn.Parameter(self._inverse(query_temperature, 0.30, 1.0))

    @staticmethod
    def _inverse(value: float, lower: float, upper: float) -> torch.Tensor:
        clipped = min(max(float(value), lower + 1e-6), upper - 1e-6)
        probability = (clipped - lower) / (upper - lower)
        return torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float32)

    @staticmethod
    def _bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * raw.sigmoid()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bounded(self.evidence_raw, 0.01, 1.0), self._bounded(self.query_raw, 0.30, 1.0)


def validate_config(config: CalibrationConfig) -> None:
    if config.prior_count != 128 or config.update_count != 128:
        raise ValueError("This preregistered calibration requires exactly 128 prior rows and 128 updates.")
    if config.update_count > config.prior_count:
        raise ValueError("The update set cannot exceed the prior set.")
    if config.query_count != 4:
        raise ValueError("Calibration requires exactly four queries.")
    for name in (
        "grid_trials",
        "optimization_trials",
        "selection_trials",
        "final_trials",
        "scalar_steps",
        "scalar_batch_size",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def split_seed_bases(seed: int) -> dict[str, int]:
    return {
        "grid": seed + 200_000,
        "optimization": seed + 300_000,
        "selection": seed + 400_000,
        "final": seed + 500_000,
        "ordinary": seed + 600_000,
    }


def _episode_config(config: CalibrationConfig, *, seed: int, trials: int) -> SequentialFilterConfig:
    return SequentialFilterConfig(
        seed=seed,
        checkpoint=config.official_checkpoint,
        device=config.device,
        initial_support_count=config.prior_count,
        stream_count=config.update_count,
        query_count=config.query_count,
        batch_size=min(16, trials),
        controlled_steps=0,
        tabicl_steps=config.tabicl_steps,
        evaluation_trials=trials,
        ordinary_evaluation_batches=config.ordinary_evaluation_batches,
        plots=config.plots,
    )


@torch.no_grad()
def cache_controlled_split(
    model: NanoTabPFNSequentialLatentFilter,
    config: CalibrationConfig,
    *,
    seed_base: int,
    trials: int,
) -> list[CachedCondition]:
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


def _summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    excluded = {"condition", "trial", "milestone"}
    metrics = [column for column in trials.columns if column not in excluded]
    rows = []
    for (condition, milestone), group in trials.groupby(["condition", "milestone"], sort=False):
        for metric in metrics:
            values = group[metric]
            count = int(values.count())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            sem = std / math.sqrt(count) if count else math.nan
            mean = float(values.mean())
            rows.append(
                {
                    "condition": condition,
                    "milestone": milestone,
                    "metric": metric,
                    "count": count,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                }
            )
    return pd.DataFrame(rows)


def _selected_weights(weights: torch.Tensor, slot: torch.Tensor) -> np.ndarray:
    indices = slot[:, None, None].expand(-1, weights.shape[1], 1)
    return weights.gather(2, indices).squeeze(-1).detach().cpu().numpy().reshape(-1)


@torch.no_grad()
def evaluate_cached_split(
    cached: list[CachedCondition],
    *,
    evidence_logit_scale: float,
    query_temperature: float,
    stream_count: int,
    invariance: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    milestones = _effective_milestones(stream_count)
    frames = []
    for item in cached:
        prediction = filter_sequential_logits(
            item.raw,
            item.stream_y,
            evidence_logit_scale=evidence_logit_scale,
            query_temperature=query_temperature,
        )
        states = torch.as_tensor(milestones, device=item.stream_y.device)
        weights = prediction.log_weights.index_select(1, states).exp()
        joint = prediction.joint_probabilities(states)
        marginals = prediction.marginal_probabilities(states)
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        slot_js = _js_divergence(slot_joint[:, 0], slot_joint[:, 1])
        class_zero_slot = slot_joint[:, :, 0].argmax(dim=1)
        class_one_slot = slot_joint[:, :, -1].argmax(dim=1)
        different_slots = class_zero_slot != class_one_slot
        exact_p1 = item.exact_p1.index_select(1, states)
        exact_binary = torch.stack((1.0 - exact_p1, exact_p1), dim=-1)
        marginal_js = _js_divergence(exact_binary, marginals.mean(dim=2))
        target_slot = torch.where(item.latent_class.bool(), class_one_slot, class_zero_slot)
        initial_slot = torch.where(
            item.initial_evidence_class == 1,
            class_one_slot,
            torch.where(item.initial_evidence_class == 0, class_zero_slot, target_slot),
        )
        opposite_slot = torch.where(item.initial_evidence_class == 1, class_zero_slot, class_one_slot)
        state_count = len(milestones)

        weight_values = weights.detach().cpu().numpy()
        frames.append(
            pd.DataFrame(
                {
                    "condition": item.condition,
                    "trial": np.repeat(np.arange(item.stream_y.shape[0]), state_count),
                    "milestone": np.tile(np.asarray(milestones), item.stream_y.shape[0]),
                    "weight_0": weight_values[:, :, 0].reshape(-1),
                    "weight_1": weight_values[:, :, 1].reshape(-1),
                    "incoherent_mass": (1.0 - joint[:, :, 0] - joint[:, :, -1])
                    .clamp_min(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1),
                    "marginal_js": marginal_js.detach().cpu().numpy().reshape(-1),
                    "mean_p1": marginals[:, :, :, 1].mean(dim=2).detach().cpu().numpy().reshape(-1),
                    "exact_p1": exact_p1.detach().cpu().numpy().reshape(-1),
                    "slot_joint_js": np.repeat(slot_js.detach().cpu().numpy(), state_count),
                    "different_supporting_slots": np.repeat(
                        different_slots.float().detach().cpu().numpy(), state_count
                    ),
                    "supporting_weight": _selected_weights(weights, target_slot),
                    "initial_supporting_weight": _selected_weights(weights, initial_slot),
                    "opposite_weight": _selected_weights(weights, opposite_slot),
                }
            )
        )
    trials = pd.concat(frames, ignore_index=True)
    summary = _summarize_trials(trials)
    return trials, summary, controlled_gate_report(trials, invariance, stream_count=stream_count)


def normalized_gate_violation(report: dict[str, Any]) -> float:
    metrics = report["metrics"]
    drops = metrics["largest_consistent_weight_drop"]
    concentrations = metrics["final_consistent_supporting_weight"]
    violations = (
        metrics["neutral_absolute_drift"] / 0.05 - 1.0,
        abs(metrics["neutral_signed_drift"]) / 0.01 - 1.0,
        (-0.02 - drops["consistent_zero"]) / 0.02,
        (-0.02 - drops["consistent_one"]) / 0.02,
        (0.90 - concentrations["consistent_zero"]) / 0.90,
        (0.90 - concentrations["consistent_one"]) / 0.90,
        (0.40 - metrics["contradiction_confidence_drop"]) / 0.40,
        (0.70 - metrics["contradiction_reversed_weight"]) / 0.70,
        metrics["final_incoherent_mass"] / 0.10 - 1.0,
        (0.20 - metrics["final_slot_joint_js"]) / 0.20,
        metrics["query_permutation_max_error"] / 1e-5 - 1.0,
        metrics["evidence_order_weight_max_error"] / 1e-5 - 1.0,
        metrics["evidence_order_joint_max_error"] / 1e-5 - 1.0,
        (0.90 - metrics["different_supporting_slots_fraction"]) / 0.90,
    )
    return max(0.0, *(float(value) for value in violations))


def candidate_record(
    name: str,
    evidence_logit_scale: float,
    query_temperature: float,
    trials: pd.DataFrame,
    report: dict[str, Any],
) -> dict[str, Any]:
    final_marginal_js = float(trials[trials.milestone == trials.milestone.max()].marginal_js.mean())
    return {
        "candidate": name,
        "evidence_logit_scale": evidence_logit_scale,
        "query_temperature": query_temperature,
        "passed": bool(report["passed"]),
        "passed_checks": int(sum(report["checks"].values())),
        "total_checks": len(report["checks"]),
        "max_normalized_violation": normalized_gate_violation(report),
        "final_marginal_js": final_marginal_js,
        "identity_distance": (evidence_logit_scale - 1.0) ** 2 + (query_temperature - 1.0) ** 2,
        "neutral_absolute_drift": report["metrics"]["neutral_absolute_drift"],
        "neutral_signed_drift": report["metrics"]["neutral_signed_drift"],
        "final_incoherent_mass": report["metrics"]["final_incoherent_mass"],
        "final_slot_joint_js": report["metrics"]["final_slot_joint_js"],
        "failure_reasons": json.dumps(report["failure_reasons"]),
    }


def rank_candidates(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame = frame.sort_values(
        by=["passed", "passed_checks", "max_normalized_violation", "final_marginal_js", "identity_distance"],
        ascending=(False, False, True, True, True),
        kind="stable",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _invariance_from_source(metadata: dict[str, Any]) -> dict[str, float]:
    metrics = (metadata.get("controlled_gate") or {}).get("metrics", {})
    return {
        "query_permutation_max_error": float(metrics.get("query_permutation_max_error", 0.0)),
        "evidence_order_weight_max_error": float(metrics.get("evidence_order_weight_max_error", 0.0)),
        "evidence_order_joint_max_error": float(metrics.get("evidence_order_joint_max_error", 0.0)),
    }


def _concatenate_cache(cached: list[CachedCondition]) -> tuple[SequentialFilterLogits, torch.Tensor, torch.Tensor]:
    return (
        SequentialFilterLogits(
            torch.cat([item.raw.stream_logits for item in cached]),
            torch.cat([item.raw.query_logits for item in cached]),
            torch.cat([item.raw.slots for item in cached]),
        ),
        torch.cat([item.stream_y for item in cached]),
        torch.cat([item.query_y for item in cached]),
    )


def observed_temperature_loss(
    raw: SequentialFilterLogits,
    stream_y: torch.Tensor,
    query_y: torch.Tensor,
    evidence_logit_scale: torch.Tensor,
    query_temperature: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = filter_sequential_logits(
        raw,
        stream_y,
        evidence_logit_scale=evidence_logit_scale,
        query_temperature=query_temperature,
    )
    prequential = -prediction.prequential_log_likelihood.mean()
    joint = prediction.joint_probabilities()
    indices = _outcome_indices(query_y)
    selected = joint.gather(-1, indices[:, None, None].expand(-1, joint.shape[1], 1)).squeeze(-1)
    trajectory = -selected.clamp_min(1e-12).log().mean()
    marginals = prediction.marginal_probabilities()
    labels = query_y[:, None, :, None].expand(-1, marginals.shape[1], -1, 1)
    marginal = -marginals.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
    total = prequential + trajectory + marginal
    return total, {
        "loss": float(total.detach()),
        "prequential_loss": float(prequential.detach()),
        "trajectory_loss": float(trajectory.detach()),
        "marginal_loss": float(marginal.detach()),
    }


def refine_temperatures(
    cached: list[CachedCondition],
    *,
    initial_evidence_scale: float,
    initial_query_temperature: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, float, pd.DataFrame, BoundedTemperatures]:
    raw, stream_y, query_y = _concatenate_cache(cached)
    calibrator = BoundedTemperatures(initial_evidence_scale, initial_query_temperature).to(stream_y.device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=learning_rate)
    generator = torch.Generator(device=stream_y.device).manual_seed(seed)
    rows = []
    for step in range(1, steps + 1):
        indices = torch.randint(
            0,
            stream_y.shape[0],
            (min(batch_size, stream_y.shape[0]),),
            generator=generator,
            device=stream_y.device,
        )
        batch_raw = SequentialFilterLogits(
            raw.stream_logits.index_select(0, indices),
            raw.query_logits.index_select(0, indices),
            raw.slots.index_select(0, indices),
        )
        evidence_scale, query_temperature = calibrator()
        optimizer.zero_grad()
        loss, metrics = observed_temperature_loss(
            batch_raw,
            stream_y.index_select(0, indices),
            query_y.index_select(0, indices),
            evidence_scale,
            query_temperature,
        )
        loss.backward()
        optimizer.step()
        rows.append(
            {
                "step": step,
                **metrics,
                "evidence_logit_scale": float(evidence_scale.detach()),
                "query_temperature": float(query_temperature.detach()),
            }
        )
    evidence_scale, query_temperature = calibrator()
    return (
        float(evidence_scale.detach()),
        float(query_temperature.detach()),
        pd.DataFrame(rows),
        calibrator,
    )


def _save_candidate_evaluation(
    output_dir: Path,
    name: str,
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    report: dict[str, Any],
    *,
    plots: bool,
) -> None:
    trials.to_csv(output_dir / f"{name}_trajectory_metrics.csv", index=False)
    summary.to_csv(output_dir / f"{name}_summary.csv", index=False)
    (output_dir / f"{name}_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    if plots:
        _save_plots(summary, output_dir, name)


def run(config: CalibrationConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    seeds = split_seed_bases(config.seed)
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Controlled checkpoint does not exist: {checkpoint_path}")
    model, source_metadata = load_sequential_filter_checkpoint(checkpoint_path, map_location=config.device)
    training_config = source_metadata.get("training_config", {})
    if (
        int(training_config.get("initial_support_count", -1)) != 128
        or int(training_config.get("stream_count", -1)) != 128
    ):
        raise ValueError("Calibration source must be a controlled checkpoint trained with 128 prior rows and updates.")
    model.to(config.device).eval()
    source_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    source_invariance = _invariance_from_source(source_metadata)

    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "sequential_latent_filter_calibration" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "calibration_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output_dir / "split_seeds.json").write_text(json.dumps(seeds, indent=2) + "\n")

    grid_cache = cache_controlled_split(model, config, seed_base=seeds["grid"], trials=config.grid_trials)
    grid_records = []
    for evidence_scale in EVIDENCE_LOGIT_SCALES:
        for query_temperature in QUERY_TEMPERATURES:
            trials, _, report = evaluate_cached_split(
                grid_cache,
                evidence_logit_scale=evidence_scale,
                query_temperature=query_temperature,
                stream_count=config.update_count,
                invariance=source_invariance,
            )
            grid_records.append(
                candidate_record(
                    "grid",
                    evidence_scale,
                    query_temperature,
                    trials,
                    report,
                )
            )
    grid = rank_candidates(grid_records)
    grid.to_csv(output_dir / "temperature_grid.csv", index=False)
    grid_best = grid.iloc[0]

    optimization_cache = cache_controlled_split(
        model,
        config,
        seed_base=seeds["optimization"],
        trials=config.optimization_trials,
    )
    learned_evidence, learned_query, learning_curve, calibrator = refine_temperatures(
        optimization_cache,
        initial_evidence_scale=float(grid_best.evidence_logit_scale),
        initial_query_temperature=float(grid_best.query_temperature),
        steps=config.scalar_steps,
        batch_size=config.scalar_batch_size,
        learning_rate=config.scalar_learning_rate,
        seed=seeds["optimization"] + 99,
    )
    learning_curve.to_csv(output_dir / "learned_temperature_curve.csv", index=False)

    selection_cache = cache_controlled_split(
        model,
        config,
        seed_base=seeds["selection"],
        trials=config.selection_trials,
    )
    candidates = {
        "identity": (1.0, 1.0),
        "grid": (float(grid_best.evidence_logit_scale), float(grid_best.query_temperature)),
        "learned": (learned_evidence, learned_query),
    }
    selection_records = []
    selection_results = {}
    for name, (evidence_scale, query_temperature) in candidates.items():
        trials, summary, report = evaluate_cached_split(
            selection_cache,
            evidence_logit_scale=evidence_scale,
            query_temperature=query_temperature,
            stream_count=config.update_count,
            invariance=source_invariance,
        )
        selection_results[name] = (trials, summary, report)
        selection_records.append(candidate_record(name, evidence_scale, query_temperature, trials, report))
        _save_candidate_evaluation(output_dir, f"selection_{name}", trials, summary, report, plots=False)
    selection_ranking = rank_candidates(selection_records)
    selection_ranking.to_csv(output_dir / "selection_ranking.csv", index=False)
    selected_name = str(selection_ranking.iloc[0].candidate)
    selected_evidence = float(selection_ranking.iloc[0].evidence_logit_scale)
    selected_query = float(selection_ranking.iloc[0].query_temperature)

    model.set_temperatures(selected_evidence, selected_query)
    final_config = _episode_config(
        config,
        seed=seeds["final"] - 1000,
        trials=config.final_trials,
    )
    final_trials, final_summary, final_invariance = evaluate_controlled(model, final_config)
    final_gate = controlled_gate_report(final_trials, final_invariance, stream_count=config.update_count)
    _save_candidate_evaluation(
        output_dir,
        "final_selected",
        final_trials,
        final_summary,
        final_gate,
        plots=config.plots,
    )

    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    calibration_metadata = {
        "selected_candidate": selected_name,
        "evidence_logit_scale": selected_evidence,
        "query_temperature": selected_query,
        "grid_candidate": candidates["grid"],
        "learned_candidate": candidates["learned"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_hash,
        "final_gate_passed": bool(final_gate["passed"]),
    }
    (output_dir / "selection.json").write_text(json.dumps(calibration_metadata, indent=2) + "\n")
    save_sequential_filter_checkpoint(
        output_dir / "calibrated_checkpoint.pth",
        model,
        training_config={"calibration": asdict(config), "selection": calibration_metadata},
        source_checkpoint_sha256=source_hash,
        stage="calibrated",
        controlled_gate=final_gate,
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), source_state[name], rtol=0, atol=0)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Frozen model unexpectedly received gradients during scalar calibration.")
    if not all(parameter.grad is not None for parameter in calibrator.parameters()):
        raise AssertionError("Both learned temperature parameters must receive gradients.")

    ordinary_config = _episode_config(
        config,
        seed=seeds["ordinary"],
        trials=config.final_trials,
    )
    ordinary_config = copy.copy(ordinary_config)
    identity_model = copy.deepcopy(model)
    identity_model.set_temperatures(1.0, 1.0)
    grid_model = copy.deepcopy(model)
    grid_model.set_temperatures(*candidates["grid"])
    learned_model = copy.deepcopy(model)
    learned_model.set_temperatures(*candidates["learned"])
    baseline = init_model_from_state_dict_file(config.official_checkpoint)
    ordinary = evaluate_ordinary_accuracy(
        {"identity": identity_model, "grid": grid_model, "learned": learned_model},
        baseline,
        ordinary_config,
    )
    ordinary["selected_candidate"] = selected_name
    ordinary["selected_accuracy"] = ordinary[selected_name]
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary, indent=2) + "\n")

    tabicl_ran = False
    if final_gate["passed"] and config.run_tabicl_on_pass:
        tabicl_config = _episode_config(
            config,
            seed=seeds["ordinary"] + 10_000,
            trials=config.final_trials,
        )
        tabicl_model, history, optimizer = train_head(
            copy.deepcopy(model),
            tabicl_config,
            np.random.default_rng(seeds["ordinary"] + 10_000),
            stage="tabicl",
        )
        pd.DataFrame(history).to_csv(output_dir / "tabicl_learning_curves.csv", index=False)
        save_sequential_filter_checkpoint(
            output_dir / "tabicl_checkpoint.pth",
            tabicl_model,
            training_config={"calibration": asdict(config), "selection": calibration_metadata},
            source_checkpoint_sha256=source_hash,
            stage="tabicl",
            controlled_gate=final_gate,
            optimizer_state=optimizer.state_dict(),
        )
        tabicl_ran = True
    (output_dir / "decision.json").write_text(
        json.dumps(
            {
                "final_gate_passed": bool(final_gate["passed"]),
                "tabicl_eligible": bool(final_gate["passed"]),
                "tabicl_requested": config.run_tabicl_on_pass,
                "tabicl_ran": tabicl_ran,
                "failure_reasons": final_gate["failure_reasons"],
            },
            indent=2,
        )
        + "\n"
    )
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CalibrationConfig.checkpoint)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--prior-count", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--grid-trials", type=int, default=128)
    parser.add_argument("--optimization-trials", type=int, default=128)
    parser.add_argument("--selection-trials", type=int, default=128)
    parser.add_argument("--final-trials", type=int, default=256)
    parser.add_argument("--scalar-steps", type=int, default=200)
    parser.add_argument("--scalar-batch-size", type=int, default=64)
    parser.add_argument("--scalar-learning-rate", type=float, default=0.02)
    parser.add_argument("--ordinary-evaluation-batches", type=int, default=8)
    parser.add_argument("--tabicl-steps", type=int, default=500)
    parser.add_argument("--run-tabicl-on-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    output_dir = run(CalibrationConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote sequential-filter calibration artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
