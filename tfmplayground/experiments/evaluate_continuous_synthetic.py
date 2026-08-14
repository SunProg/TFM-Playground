"""Synthetic held-out evaluation for slot-free continuous uncertainty.

Every arm is evaluated on identical held-out episodes drawn from the held-out
function families and held-out parameter ranges.  Metrics are reported against
both the exact candidate posterior and the feasible projected teacher, and the
two are always labelled distinctly: only the projected teacher is achievable
while the vanilla mean is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import (
    HELDOUT_REGIME,
    NOISE_LEVELS,
    SUPPORT_SIZES,
    ContinuousEpisode,
    random_label_episode,
    sample_episode,
    sample_paired_episode,
)
from tfmplayground.experiments.train_continuous_bayesian import energy_distance, teacher_targets
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.continuous_posterior import (
    ContextResamplingUncertainty,
    load_continuous_checkpoint,
)

#: Direction in which each reported metric is better.
METRIC_DIRECTION = {
    "energy_distance": "lower",
    "mutual_information_mae": "lower",
    "exact_mutual_information_mae": "lower",
    "epistemic_variance_mae": "lower",
    "exact_epistemic_variance_mae": "lower",
    "covariance_error": "lower",
    "joint_query_nll": "lower",
    "joint_cross_entropy": "lower",
    "posterior_sample_coverage": "higher",
    "candidate_mass_total_variation": "lower",
    "sample_mean_preservation_error": "lower",
    "deployed_probability_difference": "lower",
    "mutual_information": "n/a",
    "expected_conditional_entropy": "n/a",
}

#: Grid searched for the combined practical risk score.  Selection is done on
#: held-out synthetic ordinary episodes only, never with TabArena labels.
RISK_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class SyntheticEvaluationConfig:
    seed: int = 4404
    device: str = "cpu"
    episodes_per_condition: int = 8
    batch_size: int = 1
    support_size: int = 128
    query_count: int = 6
    num_samples: int = 32
    inference_seed: int = 0
    include_scm_families: bool = True
    coverage_quantile: float = 0.05


def _predict(model, episode: ContinuousEpisode, config: SyntheticEvaluationConfig):
    if isinstance(model, ContextResamplingUncertainty):
        return model(episode.support_x, episode.support_y, episode.query_x)
    return model(
        episode.support_x,
        episode.support_y,
        episode.query_x,
        sample_seed=config.inference_seed,
        num_samples=config.num_samples,
    )


def posterior_sample_coverage(prediction, projected: torch.Tensor, weights: torch.Tensor, quantile: float) -> float:
    """Posterior mass of candidates inside the model's central sample interval.

    A candidate counts as covered when *every* one of its query probabilities
    lies inside the model's central interval, so coverage measures the joint
    query vector rather than independent marginals.
    """
    lower = torch.quantile(prediction.sample_positive, quantile, dim=1)
    upper = torch.quantile(prediction.sample_positive, 1.0 - quantile, dim=1)
    inside = (projected >= lower[:, None, :] - 1e-6) & (projected <= upper[:, None, :] + 1e-6)
    covered = inside.all(dim=-1).to(weights.dtype)
    return float((covered * weights).sum(dim=-1).mean())


def candidate_mass_total_variation(prediction, projected: torch.Tensor, weights: torch.Tensor) -> float:
    """Total variation between inferred and exact candidate mass.

    Samples are assigned to their nearest candidate vector *for evaluation
    only*; the model itself never represents candidate identities.
    """
    distance = torch.cdist(prediction.sample_positive, projected)
    nearest = distance.argmin(dim=-1)
    inferred = torch.zeros_like(weights)
    inferred.scatter_add_(1, nearest, torch.ones_like(nearest, dtype=weights.dtype))
    inferred = inferred / inferred.sum(dim=-1, keepdim=True)
    return float(0.5 * (inferred - weights).abs().sum(dim=-1).mean())


@torch.no_grad()
def episode_metrics(model, episode: ContinuousEpisode, config: SyntheticEvaluationConfig) -> dict[str, float]:
    """All synthetic posterior metrics for one episode batch."""
    prediction = _predict(model, episode, config)
    base_positive = prediction.base_positive
    targets = teacher_targets(episode, base_positive)
    outcomes = episode.query_y.to(torch.long)
    model_log_joint = prediction.joint_log_probabilities()
    realized_index = (outcomes * (2 ** torch.arange(episode.query_count - 1, -1, -1, device=outcomes.device))).sum(
        dim=-1
    )
    joint_nll = -model_log_joint.gather(1, realized_index[:, None]).squeeze(1).mean()
    cross_entropy = -(targets.joint * model_log_joint).sum(dim=-1).mean()
    covariance_error = (prediction.epistemic_covariance() - targets.epistemic_covariance).abs().mean()
    vanilla_difference = float(
        (prediction.marginal_probabilities()[..., 1] - base_positive).abs().max()
    )
    return {
        "energy_distance": float(
            energy_distance(prediction.sample_positive, targets.projected_positive, targets.weights)
        ),
        "mutual_information_mae": float(
            (prediction.mutual_information() - targets.mutual_information).abs().mean()
        ),
        "exact_mutual_information_mae": float(
            (prediction.mutual_information() - targets.exact_mutual_information).abs().mean()
        ),
        "epistemic_variance_mae": float(
            (prediction.epistemic_variance() - targets.epistemic_variance).abs().mean()
        ),
        "exact_epistemic_variance_mae": float(
            (prediction.epistemic_variance() - targets.exact_epistemic_variance).abs().mean()
        ),
        "covariance_error": float(covariance_error),
        "joint_query_nll": float(joint_nll),
        "joint_cross_entropy": float(cross_entropy),
        "posterior_sample_coverage": posterior_sample_coverage(
            prediction, targets.projected_positive, targets.weights, config.coverage_quantile
        ),
        "candidate_mass_total_variation": candidate_mass_total_variation(
            prediction, targets.projected_positive, targets.weights
        ),
        "sample_mean_preservation_error": float(prediction.mean_preservation_error().max()),
        "deployed_probability_difference": vanilla_difference,
        "mutual_information": float(prediction.mutual_information().mean()),
        "expected_conditional_entropy": float(prediction.expected_conditional_entropy().mean()),
        "teacher_mutual_information": float(targets.mutual_information.mean()),
        "teacher_safe_scale": float(targets.safe_scale.mean()),
        # Dispersion diagnostics, reported per condition so that a healthy gate
        # with a throttled shape is distinguishable from a collapsed gate.
        "dispersion_gate": float(prediction.dispersion_gate.mean()),
        "dispersion_bound": float(prediction.dispersion_bound.mean()),
        "deviation_shape_ratio": float(prediction.deviation_shape_ratio().mean()),
        "max_deviation": float((prediction.sample_positive - base_positive[:, None, :]).abs().amax()),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


@torch.no_grad()
def evaluate_arm(model, config: SyntheticEvaluationConfig) -> dict[str, Any]:
    """Evaluate one uncertainty arm on the shared held-out episode suite."""
    regime = HELDOUT_REGIME
    report: dict[str, Any] = {"conditions": {}, "evidence": {}, "label_noise": {}}
    for condition in ("ambiguous", "identifiable", "noisy"):
        rng = np.random.default_rng(config.seed)
        rows = []
        for _ in range(config.episodes_per_condition):
            episode = sample_episode(
                rng,
                regime=regime,
                condition=condition,
                batch_size=config.batch_size,
                support_size=config.support_size,
                query_count=config.query_count,
                device=config.device,
            )
            rows.append(episode_metrics(model, episode, config))
        report["conditions"][condition] = _aggregate(rows)

    # Uncertainty as support evidence increases.
    for support_size in SUPPORT_SIZES:
        rng = np.random.default_rng(config.seed + 101)
        rows = []
        for _ in range(max(2, config.episodes_per_condition // 2)):
            episode = sample_episode(
                rng,
                regime=regime,
                condition="ambiguous",
                batch_size=config.batch_size,
                support_size=support_size,
                query_count=config.query_count,
                noise=0.05,
                device=config.device,
            )
            rows.append(episode_metrics(model, episode, config))
        report["evidence"][str(support_size)] = _aggregate(rows)

    # Aleatoric/epistemic separation as controlled label noise increases.
    for noise in NOISE_LEVELS:
        rng = np.random.default_rng(config.seed + 202)
        rows = []
        for _ in range(max(2, config.episodes_per_condition // 2)):
            episode = sample_episode(
                rng,
                regime=regime,
                condition="noisy",
                batch_size=config.batch_size,
                support_size=config.support_size,
                query_count=config.query_count,
                noise=noise,
                device=config.device,
            )
            rows.append(episode_metrics(model, episode, config))
        report["label_noise"][f"{noise:.2f}"] = _aggregate(rows)

    # Paired support-prefix episodes: identifying evidence must not raise MI.
    rng = np.random.default_rng(config.seed + 303)
    drops = []
    for _ in range(max(2, config.episodes_per_condition // 2)):
        short, long = sample_paired_episode(
            rng,
            regime=regime,
            batch_size=config.batch_size,
            support_size=config.support_size,
            query_count=config.query_count,
            noise=0.05,
            device=config.device,
        )
        short_mi = float(_predict(model, short, config).mutual_information().mean())
        long_mi = float(_predict(model, long, config).mutual_information().mean())
        drops.append({"short": short_mi, "long": long_mi, "drop": short_mi - long_mi})
    report["evidence_pairs"] = _aggregate(drops)

    rng = np.random.default_rng(config.seed + 404)
    diagnostic = random_label_episode(
        rng,
        batch_size=config.batch_size,
        support_size=config.support_size,
        query_count=config.query_count,
        device=config.device,
    )
    prediction = _predict(model, diagnostic, config)
    report["random_label_diagnostic"] = {
        "mutual_information": float(prediction.mutual_information().mean()),
        "expected_conditional_entropy": float(prediction.expected_conditional_entropy().mean()),
        "sample_mean_preservation_error": float(prediction.mean_preservation_error().max()),
    }
    report["headline"] = _aggregate([report["conditions"][name] for name in ("ambiguous", "identifiable", "noisy")])
    report["qualitative"] = qualitative_gates(report)
    return report


def qualitative_gates(report: dict[str, Any]) -> dict[str, bool]:
    """Required qualitative behaviour of a slot-free posterior."""
    conditions = report["conditions"]
    noise = report["label_noise"]
    low_noise, high_noise = noise[f"{NOISE_LEVELS[0]:.2f}"], noise[f"{NOISE_LEVELS[-1]:.2f}"]
    diagnostic = report["random_label_diagnostic"]
    return {
        "agreeing_candidates_have_near_zero_mutual_information": (
            conditions["noisy"]["mutual_information"] < 0.02
        ),
        "unresolved_disagreement_is_positive": conditions["ambiguous"]["mutual_information"] > 0.01,
        "identifying_evidence_reduces_mutual_information": report["evidence_pairs"]["drop"] > 0.0,
        "label_noise_raises_expected_conditional_entropy": (
            high_noise["expected_conditional_entropy"] > low_noise["expected_conditional_entropy"]
        ),
        "label_noise_does_not_materially_raise_mutual_information": (
            high_noise["mutual_information"] <= low_noise["mutual_information"] + 0.02
        ),
        "random_labels_are_aleatoric": (
            diagnostic["expected_conditional_entropy"] > 0.3 and diagnostic["mutual_information"] < 0.05
        ),
    }


# --------------------------------------------------------------------------- #
# Combined risk score selection
# --------------------------------------------------------------------------- #
def _error_detection(scores: np.ndarray, errors: np.ndarray) -> tuple[float | None, float]:
    order = np.argsort(scores)
    cumulative = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    aurc = float(cumulative.mean())
    if np.unique(errors).size < 2:
        return None, aurc
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(errors, scores)), aurc


@torch.no_grad()
def select_risk_lambda(model, config: SyntheticEvaluationConfig) -> dict[str, Any]:
    """Pick ``lambda`` in ``risk = predictive_entropy + lambda * mutual_information``.

    Selection uses held-out synthetic *ordinary* episodes only.  TabArena labels
    are never involved.
    """
    rng = np.random.default_rng(config.seed + 909)
    entropies: list[np.ndarray] = []
    informations: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for index in range(max(4, config.episodes_per_condition)):
        condition = ("identifiable", "ambiguous", "noisy")[index % 3]
        episode = sample_episode(
            rng,
            regime=HELDOUT_REGIME,
            condition=condition,
            batch_size=config.batch_size,
            support_size=config.support_size,
            query_count=config.query_count,
            device=config.device,
        )
        prediction = _predict(model, episode, config)
        entropies.append(prediction.predictive_entropy().reshape(-1).cpu().numpy())
        informations.append(prediction.mutual_information().reshape(-1).cpu().numpy())
        predicted = (prediction.base_positive >= 0.5).long().reshape(-1).cpu().numpy()
        errors.append((predicted != episode.query_y.reshape(-1).cpu().numpy()).astype(int))
    entropy = np.concatenate(entropies)
    information = np.concatenate(informations)
    error = np.concatenate(errors)
    table = {}
    for value in RISK_LAMBDAS:
        auroc, aurc = _error_detection(entropy + value * information, error)
        table[str(value)] = {"error_auroc": auroc, "aurc": aurc}
    ranked = sorted(
        table.items(),
        key=lambda item: (-(item[1]["error_auroc"] or 0.0), item[1]["aurc"]),
    )
    return {"selected_lambda": float(ranked[0][0]), "grid": table, "selection_data": "held-out synthetic ordinary"}


# --------------------------------------------------------------------------- #
# Acceptance gates
# --------------------------------------------------------------------------- #
def representation_benefit(adapter: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Gate 2: adapters must beat the frozen encoder without hurting joint NLL."""
    adapter_head, frozen_head = adapter["headline"], frozen["headline"]
    mi_gain = _relative_gain(frozen_head["mutual_information_mae"], adapter_head["mutual_information_mae"])
    variance_gain = _relative_gain(frozen_head["epistemic_variance_mae"], adapter_head["epistemic_variance_mae"])
    joint_change = _relative_gain(frozen_head["joint_query_nll"], adapter_head["joint_query_nll"])
    return {
        "mutual_information_relative_gain": mi_gain,
        "epistemic_variance_relative_gain": variance_gain,
        "joint_nll_relative_change": joint_change,
        "passed": bool(max(mi_gain, variance_gain) >= 0.10 and joint_change >= -0.02),
    }


def _relative_gain(reference: float, candidate: float) -> float:
    """Positive when ``candidate`` improves on ``reference`` for a lower-is-better metric."""
    if not math.isfinite(reference) or abs(reference) < 1e-12:
        return 0.0
    return float((reference - candidate) / abs(reference))


def continuous_posterior_benefit(adapter: dict[str, Any], beta: dict[str, Any]) -> dict[str, Any]:
    """Gate 3: at least two of five metrics must improve on the Beta ablation."""
    metrics = (
        "energy_distance",
        "mutual_information_mae",
        "epistemic_variance_mae",
        "covariance_error",
        "joint_query_nll",
    )
    improvements = {name: _relative_gain(beta["headline"][name], adapter["headline"][name]) for name in metrics}
    improved = sum(1 for value in improvements.values() if value > 0.0)
    return {"relative_gains": improvements, "improved_metric_count": improved, "passed": bool(improved >= 2)}


def build_arms(
    checkpoints: dict[str, str],
    *,
    vanilla_checkpoint: str,
    device: str,
    resampling_subsets: int = 16,
) -> dict[str, Any]:
    """Load every evaluated arm, including the non-learned baseline."""
    arms: dict[str, Any] = {}
    backbone = init_model_from_state_dict_file(vanilla_checkpoint).to(device).eval()
    arms["context_resampling"] = ContextResamplingUncertainty(backbone, num_subsets=resampling_subsets)
    for name, path in checkpoints.items():
        model, _checkpoint = load_continuous_checkpoint(path, map_location=device)
        arms[name] = model.to(device).eval()
    return arms


def run_synthetic_evaluation(
    checkpoints: dict[str, str],
    *,
    vanilla_checkpoint: str,
    output_dir: str,
    config: SyntheticEvaluationConfig,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = build_arms(checkpoints, vanilla_checkpoint=vanilla_checkpoint, device=config.device)
    results: dict[str, Any] = {"metric_direction": METRIC_DIRECTION, "arms": {}}
    for name, model in arms.items():
        report = evaluate_arm(model, config)
        report["risk_lambda"] = select_risk_lambda(model, config)
        results["arms"][name] = report
        print(f"{name}: {json.dumps(report['headline'], sort_keys=True)}", flush=True)
    gates: dict[str, Any] = {}
    if "adapter_continuous" in results["arms"] and "frozen_continuous" in results["arms"]:
        gates["representation_benefit"] = representation_benefit(
            results["arms"]["adapter_continuous"], results["arms"]["frozen_continuous"]
        )
    if "adapter_continuous" in results["arms"] and "beta" in results["arms"]:
        gates["continuous_posterior_benefit"] = continuous_posterior_benefit(
            results["arms"]["adapter_continuous"], results["arms"]["beta"]
        )
    results["gates"] = gates
    (output / "synthetic_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "synthetic_headline.csv").write_text(_headline_csv(results))
    return output.resolve()


def _headline_csv(results: dict[str, Any]) -> str:
    arms = results["arms"]
    keys = sorted({key for report in arms.values() for key in report["headline"]})
    header = "arm," + ",".join(keys)
    body = "\n".join(
        name + "," + ",".join(f"{report['headline'].get(key, float('nan')):.6f}" for key in keys)
        for name, report in arms.items()
    )
    return header + "\n" + body + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named uncertainty checkpoint, for example adapter_continuous=runs/.../best.pth",
    )
    parser.add_argument("--vanilla-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SyntheticEvaluationConfig.seed)
    parser.add_argument("--episodes-per-condition", type=int, default=SyntheticEvaluationConfig.episodes_per_condition)
    parser.add_argument("--support-size", type=int, default=SyntheticEvaluationConfig.support_size)
    parser.add_argument("--query-count", type=int, default=SyntheticEvaluationConfig.query_count)
    parser.add_argument("--num-samples", type=int, default=SyntheticEvaluationConfig.num_samples)
    parser.add_argument("--inference-seed", type=int, default=SyntheticEvaluationConfig.inference_seed)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    named = dict(item.split("=", 1) for item in arguments.checkpoint)
    evaluation_config = SyntheticEvaluationConfig(
        seed=arguments.seed,
        device=arguments.device,
        episodes_per_condition=arguments.episodes_per_condition,
        support_size=arguments.support_size,
        query_count=arguments.query_count,
        num_samples=arguments.num_samples,
        inference_seed=arguments.inference_seed,
    )
    print(
        "Wrote synthetic evaluation to "
        + str(
            run_synthetic_evaluation(
                named,
                vanilla_checkpoint=arguments.vanilla_checkpoint,
                output_dir=arguments.output_dir,
                config=evaluation_config,
            )
        )
    )
