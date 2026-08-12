"""Train non-leaking cross-fitted hypothesis weights with variational fallback."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.experiments.coherent_hypotheses import outcome_indices, sequence_to_canonical_indices
from tfmplayground.experiments.hypothesis_collapse import (
    NanoTabPFNBinaryPredictor,
    compute_trial_metrics,
    enumerate_chain_joint,
    exact_joint_distribution,
    generate_global_ambiguity_tasks,
    summarize_metrics,
)
from tfmplayground.experiments.train_coherent_hypotheses import (
    TrainingBatch,
    load_consistency_checkpoint,
    make_ordinary_iterator,
    next_ordinary_batch,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.coherent_correction import (
    NanoTabPFNCrossFitHypothesisModel,
    NanoTabPFNVariationalHypothesisModel,
    save_correction_checkpoint,
)
from tfmplayground.models.hypothesis import load_hypothesis_checkpoint
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class CorrectionConfig:
    seed: int = 2402
    evidence_model: str = "crossfit"
    fallback_on_failure: bool = True
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    consistency_checkpoint: str = "runs/coherent_hypotheses/full-run/consistency_checkpoint.pth"
    biased_checkpoint: str | None = "runs/coherent_hypotheses/full-run/hypothesis_checkpoint.pth"
    output_dir: str | None = None
    device: str = "cpu"
    query_count: int = 4
    batch_size: int = 2
    accumulate_gradients: int = 8
    crossfit_frozen_steps: int = 100
    crossfit_unfrozen_steps: int = 300
    variational_frozen_steps: int = 100
    variational_unfrozen_steps: int = 500
    validation_interval: int = 50
    patience: int = 4
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    num_partitions: int = 2
    controlled_only: bool = False
    evaluation_trials: int = 32
    ordinary_evaluation_batches: int = 8


def validate_config(config: CorrectionConfig) -> None:
    if config.evidence_model not in {"crossfit", "variational"}:
        raise ValueError("evidence_model must be crossfit or variational.")
    if config.query_count != 4:
        raise ValueError("Correction acceptance is defined for exactly four queries.")
    if config.num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    numeric = (
        "batch_size",
        "accumulate_gradients",
        "validation_interval",
        "patience",
        "evaluation_trials",
        "ordinary_evaluation_batches",
    )
    if any(getattr(config, name) < 1 for name in numeric):
        raise ValueError("Batch, validation, patience, and evaluation counts must be positive.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def controlled_observation_batch(
    config: CorrectionConfig,
    rng: np.random.Generator,
    *,
    zero_evidence: bool,
    support_size: int | None = None,
) -> TrainingBatch:
    support_size = support_size or int(rng.choice((16, 32, 64, 128)))
    evidence_count = 0 if zero_evidence else int(rng.choice((2, 8, 16, 32, 64, 128)))
    batch = generate_global_ambiguity_tasks(
        trials=config.batch_size,
        query_count=config.query_count,
        evidence_count=evidence_count,
        common_support_size=support_size,
        support_noise=0.1,
        rng=rng,
    )
    device = torch.device(config.device)
    return TrainingBatch(
        support_x=torch.from_numpy(batch.support_x).to(device),
        support_y=torch.from_numpy(batch.support_y).to(device),
        query_x=torch.from_numpy(batch.query_x).to(device),
        query_y=torch.from_numpy(batch.query_y.astype(np.int64)).to(device),
    )


def next_correction_ordinary_batch(ordinary, query_count: int) -> TrainingBatch:
    """Skip TabICL draws too small to leave two stable rows in each fold."""
    for _ in range(100):
        batch = next_ordinary_batch(ordinary, query_count)
        if batch.support_x.shape[1] >= 4:
            return batch
    raise RuntimeError("TabICL did not provide at least four support rows after 100 attempts.")


def _diversity_loss(slot_logits: torch.Tensor, minimum_js: float = 0.05) -> torch.Tensor:
    probabilities = F.softmax(slot_logits, dim=-1)
    left, right = probabilities[:, :, 0], probabilities[:, :, 1]
    midpoint = 0.5 * (left + right)
    js = 0.5 * (
        (left * (left.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
        + (right * (right.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
    )
    return F.relu(minimum_js - js).mean()


def crossfit_loss(model: NanoTabPFNCrossFitHypothesisModel, batch: TrainingBatch):
    full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
    prediction = model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])
    indices = outcome_indices(batch.query_y)
    joint = prediction.joint_probabilities()
    marginals = prediction.marginal_probabilities()
    joint_loss = -joint.gather(1, indices[:, None]).clamp_min(1e-12).log().mean()
    marginal_loss = F.nll_loss(marginals.clamp_min(1e-12).log().reshape(-1, 2), batch.query_y.reshape(-1))
    alignment_loss = prediction.alignment_loss()
    diversity_loss = _diversity_loss(prediction.slot_logits)
    total = joint_loss + marginal_loss + 0.1 * alignment_loss + 0.05 * diversity_loss
    return total, {
        "loss": float(total.detach()),
        "joint_loss": float(joint_loss.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "alignment_loss": float(alignment_loss.detach()),
        "diversity_loss": float(diversity_loss.detach()),
    }


def variational_loss(
    model: NanoTabPFNVariationalHypothesisModel,
    batch: TrainingBatch,
    *,
    beta: float,
):
    full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
    prediction = model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])
    log_probabilities = F.log_softmax(prediction.slot_logits, dim=-1)
    labels = batch.query_y[:, :, None, None].expand(-1, -1, 2, 1)
    trajectory_log_probabilities = log_probabilities.gather(-1, labels).squeeze(-1).sum(dim=1)
    weights = prediction.slot_log_weights.exp()
    expected_nll = -(weights * trajectory_log_probabilities).sum(-1).mean()
    marginals = prediction.marginal_probabilities()
    marginal_loss = F.nll_loss(marginals.clamp_min(1e-12).log().reshape(-1, 2), batch.query_y.reshape(-1))
    kl_loss = (weights * (prediction.slot_log_weights + math.log(2.0))).sum(-1).mean()
    alignment_loss = prediction.alignment_loss()
    diversity_loss = _diversity_loss(prediction.slot_logits)
    total = expected_nll + marginal_loss + beta * kl_loss + 0.1 * alignment_loss + 0.05 * diversity_loss
    return total, {
        "loss": float(total.detach()),
        "expected_nll": float(expected_nll.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "alignment_loss": float(alignment_loss.detach()),
        "diversity_loss": float(diversity_loss.detach()),
        "beta": beta,
    }


def _validation_loss(model, config: CorrectionConfig, model_type: str, step: int) -> float:
    rng = np.random.default_rng(config.seed + 50_000)
    values = []
    model.eval()
    with torch.no_grad():
        for support_size, zero_evidence in ((16, True), (128, True), (128, False)):
            batch = controlled_observation_batch(
                config,
                rng,
                zero_evidence=zero_evidence,
                support_size=support_size,
            )
            if model_type == "crossfit":
                loss, _ = crossfit_loss(model, batch)
            else:
                loss, _ = variational_loss(model, batch, beta=_cyclic_beta(step))
            values.append(float(loss))
    model.train()
    return float(np.mean(values))


def _cyclic_beta(step: int) -> float:
    position = (step - 1) % 100
    return 0.05 * min(1.0, position / 50.0)


def train_correction_model(
    model: NanoTabPFNCrossFitHypothesisModel | NanoTabPFNVariationalHypothesisModel,
    config: CorrectionConfig,
    rng: np.random.Generator,
    *,
    model_type: str,
) -> tuple[torch.nn.Module, list[dict]]:
    if model_type == "crossfit":
        frozen_steps, unfrozen_steps = config.crossfit_frozen_steps, config.crossfit_unfrozen_steps
    else:
        frozen_steps, unfrozen_steps = config.variational_frozen_steps, config.variational_unfrozen_steps
    total_steps = frozen_steps + unfrozen_steps
    model.to(config.device)
    model.freeze_backbone()
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": head_parameters, "lr": config.head_lr},
        ]
    )
    ordinary = make_ordinary_iterator(config, total_steps * config.accumulate_gradients * 4)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale = 0
    microstep = 0
    controlled_index = 0
    for step in range(1, total_steps + 1):
        if step == frozen_steps + 1:
            model.unfreeze_final_backbone_blocks(2)
        model.train()
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _ in range(config.accumulate_gradients):
            use_controlled = ordinary is None or microstep % 2 == 0
            if use_controlled:
                batch = controlled_observation_batch(config, rng, zero_evidence=controlled_index % 2 == 0)
                controlled_index += 1
            else:
                batch = next_correction_ordinary_batch(ordinary, config.query_count)
            if model_type == "crossfit":
                loss, metrics = crossfit_loss(model, batch)
            else:
                loss, metrics = variational_loss(model, batch, beta=_cyclic_beta(step))
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
            microstep += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"stage": model_type, "step": step, **totals}
        if step % config.validation_interval == 0 or step == total_steps:
            row["validation_loss"] = _validation_loss(model, config, model_type, step)
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
    return model, history


@torch.no_grad()
def _direct_distributions(model, support_x, support_y, query_x, device):
    support_x = torch.as_tensor(support_x, device=device)
    support_y = torch.as_tensor(support_y, device=device)
    query_x = torch.as_tensor(query_x, device=device)
    prediction = model(
        (torch.cat((support_x, query_x), dim=1), support_y),
        train_test_split_index=support_x.shape[1],
    )
    return prediction.joint_probabilities().cpu().numpy(), prediction.marginal_probabilities()[..., 1].cpu().numpy()


def evaluate_models(models: dict[str, torch.nn.Module], config: CorrectionConfig):
    rows = []
    seeds = (config.seed + 101, config.seed + 202, config.seed + 303)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for support_size in (16, 64, 128):
            for evidence_count in (0, support_size):
                batch = generate_global_ambiguity_tasks(
                    trials=config.evaluation_trials,
                    query_count=4,
                    evidence_count=evidence_count,
                    common_support_size=support_size,
                    support_noise=0.1,
                    rng=rng,
                )
                bayes_joint = exact_joint_distribution(batch.posterior_b, 4)
                bayes_marginals = np.repeat(batch.posterior_b[:, None], 4, axis=1)
                for name, model in models.items():
                    model.to(config.device).eval()
                    if isinstance(model, NanoTabPFNModel):
                        predictor = NanoTabPFNBinaryPredictor(model, config.device)
                        joint = enumerate_chain_joint(
                            predictor, batch.support_x, batch.support_y, batch.query_x, (0, 1, 2, 3)
                        )
                        reverse = enumerate_chain_joint(
                            predictor, batch.support_x, batch.support_y, batch.query_x, (3, 2, 1, 0)
                        )
                        marginals = predictor.predict_binary_proba(batch.support_x, batch.support_y, batch.query_x)[
                            ..., 1
                        ]
                    else:
                        joint, marginals = _direct_distributions(
                            model, batch.support_x, batch.support_y, batch.query_x, config.device
                        )
                        reverse = joint
                    metrics = compute_trial_metrics(
                        bayes_joint=bayes_joint,
                        predicted_joint=joint,
                        reverse_joint=reverse,
                        bayes_marginals=bayes_marginals,
                        predicted_marginals=marginals,
                    )
                    for trial in range(config.evaluation_trials):
                        row = {
                            "seed": seed,
                            "trial": trial,
                            "query_count": 4,
                            "evidence_count": evidence_count,
                            "common_support_size": support_size,
                            "model": name,
                            "mean_p1": float(marginals[trial].mean()),
                            "extreme_marginal": bool(marginals[trial].mean() < 0.2 or marginals[trial].mean() > 0.8),
                        }
                        row.update({metric: values[trial] for metric, values in metrics.items()})
                        rows.append(row)
    trials = pd.DataFrame(rows)
    summary = summarize_metrics(trials.drop(columns=["seed", "mean_p1", "extreme_marginal"]))
    return trials, summary


@torch.no_grad()
def evaluate_ordinary_accuracy(models: dict[str, torch.nn.Module], config: CorrectionConfig):
    iterator = make_ordinary_iterator(config, config.ordinary_evaluation_batches * 4)
    if iterator is None:
        return {name: None for name in models}
    correct = {name: torch.zeros(2, dtype=torch.long) for name in models}
    total = torch.zeros(2, dtype=torch.long)
    for _ in range(config.ordinary_evaluation_batches):
        batch = next_correction_ordinary_batch(iterator, 4)
        labels = batch.query_y.cpu()
        for class_index in range(2):
            total[class_index] += (labels == class_index).sum()
        full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
        for name, model in models.items():
            model.to(config.device).eval()
            if isinstance(model, NanoTabPFNModel):
                predicted = (
                    model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])[..., :2]
                    .argmax(-1)
                    .cpu()
                )
            else:
                predicted = (
                    model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])
                    .marginal_probabilities()
                    .argmax(-1)
                    .cpu()
                )
            for class_index in range(2):
                correct[name][class_index] += ((predicted == class_index) & (labels == class_index)).sum()
    present = total > 0
    return {name: float((correct[name][present].float() / total[present].float()).mean()) for name in models}


@torch.no_grad()
def permutation_errors(model, config: CorrectionConfig) -> dict[str, float]:
    batch = generate_global_ambiguity_tasks(
        trials=min(8, config.evaluation_trials),
        query_count=4,
        evidence_count=0,
        common_support_size=128,
        support_noise=0.1,
        rng=np.random.default_rng(config.seed + 909),
    )
    canonical, _ = _direct_distributions(model, batch.support_x, batch.support_y, batch.query_x, config.device)
    row_order = np.random.default_rng(config.seed + 910).permutation(batch.support_x.shape[1])
    permuted_support, _ = _direct_distributions(
        model,
        batch.support_x[:, row_order],
        batch.support_y[:, row_order],
        batch.query_x,
        config.device,
    )
    query_order = (3, 1, 0, 2)
    permuted_query, _ = _direct_distributions(
        model,
        batch.support_x,
        batch.support_y,
        batch.query_x[:, query_order],
        config.device,
    )
    mapping = sequence_to_canonical_indices(query_order).numpy()
    remapped_query = permuted_query[:, np.argsort(mapping)]
    return {
        "support_permutation_max_error": float(np.max(np.abs(canonical - permuted_support))),
        "query_permutation_max_error": float(np.max(np.abs(canonical - remapped_query))),
    }


def acceptance_report(
    trials: pd.DataFrame,
    ordinary_accuracy: dict[str, float | None],
    invariance: dict[str, float],
    model_name: str,
) -> dict:
    selected = trials[trials.model == model_name]
    matched = selected[(selected.common_support_size == 128) & (selected.evidence_count == 128)]
    ambiguous = selected[(selected.common_support_size == 128) & (selected.evidence_count == 0)]
    report = {
        "model": model_name,
        "matched_incoherent_mass": float(matched.incoherent_mass.mean()),
        "matched_marginal_js": float(matched.marginal_js.mean()),
        "ambiguous_incoherent_mass": float(ambiguous.incoherent_mass.mean()),
        "ambiguous_marginal_js": float(ambiguous.marginal_js.mean()),
        "ambiguous_mean_p1": float(ambiguous.mean_p1.mean()),
        "ambiguous_extreme_fraction": float(ambiguous.extreme_marginal.mean()),
        "ordinary_balanced_accuracy": ordinary_accuracy.get(model_name),
        "baseline_balanced_accuracy": ordinary_accuracy.get("baseline"),
        **invariance,
    }
    checks = {
        "matched_incoherent_mass": report["matched_incoherent_mass"] <= 0.05,
        "matched_marginal_js": report["matched_marginal_js"] <= 0.01,
        "ambiguous_incoherent_mass": report["ambiguous_incoherent_mass"] <= 0.05,
        "ambiguous_marginal_js": report["ambiguous_marginal_js"] <= 0.01,
        "ambiguous_mean_p1": 0.45 <= report["ambiguous_mean_p1"] <= 0.55,
        "ambiguous_extreme_fraction": report["ambiguous_extreme_fraction"] < 0.05,
        "ordinary_accuracy": (
            report["ordinary_balanced_accuracy"] is not None
            and report["baseline_balanced_accuracy"] is not None
            and report["ordinary_balanced_accuracy"] >= report["baseline_balanced_accuracy"] - 0.01
        ),
        "support_permutation": report["support_permutation_max_error"] <= 1e-5,
        "query_permutation": report["query_permutation_max_error"] <= 1e-5,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["failure_reasons"] = [name for name, passed in checks.items() if not passed]
    return report


def should_launch_variational(
    evidence_model: str,
    fallback_on_failure: bool,
    crossfit_report: dict | None = None,
) -> bool:
    """Return whether the fresh variational fallback must be trained."""
    if evidence_model == "variational":
        return True
    if evidence_model != "crossfit":
        raise ValueError(f"Unknown evidence model: {evidence_model}")
    if crossfit_report is None:
        raise ValueError("A crossfit acceptance report is required after crossfit training.")
    return fallback_on_failure and not bool(crossfit_report["passed"])


def _fresh_initial_backbone(config: CorrectionConfig) -> NanoTabPFNModel:
    """Load either a consistency checkpoint or an unchanged nanoTabPFN checkpoint."""
    checkpoint = torch.load(config.consistency_checkpoint, map_location="cpu")
    if checkpoint.get("model_type") == "nanotabpfn_consistency":
        return load_consistency_checkpoint(config.consistency_checkpoint)
    return init_model_from_state_dict_file(config.consistency_checkpoint)


def run(config: CorrectionConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "coherent_correction" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    source_hash = hashlib.sha256(Path(config.consistency_checkpoint).read_bytes()).hexdigest()
    models: dict[str, torch.nn.Module] = {
        "baseline": init_model_from_state_dict_file(config.checkpoint),
        "initialization": _fresh_initial_backbone(config),
    }
    if config.biased_checkpoint and Path(config.biased_checkpoint).is_file():
        models["biased_slots"] = load_hypothesis_checkpoint(config.biased_checkpoint)[0]
    histories = []
    reports = {}

    selected_name = config.evidence_model
    if config.evidence_model == "crossfit":
        crossfit = NanoTabPFNCrossFitHypothesisModel(
            _fresh_initial_backbone(config), num_partitions=config.num_partitions
        )
        crossfit, history = train_correction_model(crossfit, config, rng, model_type="crossfit")
        histories.extend(history)
        models["crossfit"] = crossfit
        save_correction_checkpoint(
            output_dir / "crossfit_checkpoint.pth",
            crossfit,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="crossfit",
        )
        trials, summary = evaluate_models(models, config)
        ordinary = evaluate_ordinary_accuracy(models, config)
        reports["crossfit"] = acceptance_report(trials, ordinary, permutation_errors(crossfit, config), "crossfit")
        needs_fallback = should_launch_variational(
            config.evidence_model, config.fallback_on_failure, reports["crossfit"]
        )
    else:
        needs_fallback = should_launch_variational(config.evidence_model, config.fallback_on_failure)

    if needs_fallback:
        variational = NanoTabPFNVariationalHypothesisModel(_fresh_initial_backbone(config))
        variational, history = train_correction_model(variational, config, rng, model_type="variational")
        histories.extend(history)
        models["variational"] = variational
        save_correction_checkpoint(
            output_dir / "variational_checkpoint.pth",
            variational,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="variational",
        )
        selected_name = "variational"

    trials, summary = evaluate_models(models, config)
    ordinary = evaluate_ordinary_accuracy(models, config)
    if "variational" in models:
        reports["variational"] = acceptance_report(
            trials, ordinary, permutation_errors(models["variational"], config), "variational"
        )
    selection = {
        "selected_model": selected_name,
        "selected_passed": reports[selected_name]["passed"],
        "reports": reports,
    }
    pd.DataFrame(histories).to_csv(output_dir / "learning_curves.csv", index=False)
    trials.to_csv(output_dir / "evaluation_trial_metrics.csv", index=False)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary, indent=2) + "\n")
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-model", choices=("crossfit", "variational"), default="crossfit")
    parser.add_argument("--fallback-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--consistency-checkpoint",
        default="runs/coherent_hypotheses/full-run/consistency_checkpoint.pth",
    )
    parser.add_argument("--biased-checkpoint", default="runs/coherent_hypotheses/full-run/hypothesis_checkpoint.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--crossfit-frozen-steps", type=int, default=100)
    parser.add_argument("--crossfit-unfrozen-steps", type=int, default=300)
    parser.add_argument("--variational-frozen-steps", type=int, default=100)
    parser.add_argument("--variational-unfrozen-steps", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--num-partitions", type=int, default=2)
    parser.add_argument("--controlled-only", action="store_true")
    parser.add_argument("--evaluation-trials", type=int, default=32)
    parser.add_argument("--ordinary-evaluation-batches", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output_dir = run(CorrectionConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote coherent-correction artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
