"""Synthetic evaluation of support-resampling encoder dispersion against ground truth.

Every arm from ``tfmplayground.experiments.support_resampling`` is scored on the
same held-out-family synthetic episodes used by
``evaluate_continuous_synthetic.py``, against the *exact* candidate-posterior
mutual information -- no learned head, no sampling, no training. This is stage 1
of the trial described in ``SUPPORT_RESAMPLING_VARIANCE_TRIAL.md``: it validates
each estimator against ground truth and a random-label null before any arm is
allowed to spend TabArena evaluation budget in stage 2.

Query labels never enter estimator construction: dispersion is read from query
*embeddings*, and query performance is scored separately from held-out *support*
rows, joined only through the shared member index sets. See
``support_resampling.py`` for why that residual coupling needs a permutation
null, which this script also reports.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import (
    HELDOUT_REGIME,
    random_label_episode,
    sample_episode,
)
from tfmplayground.experiments.support_resampling import (
    SCHEMES,
    build_ensemble,
    compute_intrinsic_posterior,
    split_fit_and_heldout,
)
from tfmplayground.experiments.train_continuous_bayesian import teacher_targets
from tfmplayground.interface import init_model_from_state_dict_file

#: Floors carried over from the encoder probe (``probe_uncertainty_encoder.py``)
#: so results land in the same table.
MIN_R2 = 0.30
MIN_CONDITION_AUC = 0.85
#: Gate 3: an arm whose random-label mutual information exceeds this is reading
#: resampling noise, not epistemic content, regardless of gates 1 and 2.
MAX_NULL_MUTUAL_INFORMATION = 0.02
#: Gate 4: minimum |Spearman| between episode dispersion and held-out loss
#: variance, required to clear the permutation-null band.
MIN_PERFORMANCE_LINK = 0.4

#: Arms that need a resampled ensemble.
RESAMPLED_ARMS = SCHEMES
#: Arms that read the model's own posterior with no resampling at all.
INTRINSIC_ARMS = ("joint_mi", "self_conditioning")
ARMS = (*RESAMPLED_ARMS, *INTRINSIC_ARMS)


@dataclass(frozen=True)
class ResamplingEvaluationConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    device: str = "cpu"
    episodes_per_condition: int = 40
    support_size: int = 128
    query_count: int = 6
    members: int = 32
    subsample_fraction: float = 0.8
    heldout_size: int = 16
    num_mem_chunks: int = 1
    permutation_trials: int = 200


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    variance = float(((target - target.mean()) ** 2).sum())
    if variance <= 0:
        return float("nan")
    return float(1.0 - ((prediction - target) ** 2).sum() / variance)


def _base_positive(model, episode) -> torch.Tensor:
    with torch.no_grad():
        logits = model(episode.support_x, episode.support_y, episode.query_x)
    return logits[..., :2].softmax(dim=-1)[..., 1]


def episode_arm_scores(model, episode, config: ResamplingEvaluationConfig, seed: int) -> dict[str, dict[str, Any]]:
    """Every arm's per-query mean mutual information for one episode, plus performance data."""
    support_x, support_y, query_x = episode.support_x[0], episode.support_y[0], episode.query_x[0]
    fit_x, fit_y, held_x, held_y = split_fit_and_heldout(support_x, support_y, config.heldout_size, seed)
    scores: dict[str, dict[str, Any]] = {}
    for scheme in RESAMPLED_ARMS:
        ensemble = build_ensemble(
            model,
            fit_x,
            fit_y,
            query_x,
            scheme=scheme,
            members=config.members,
            fraction=config.subsample_fraction,
            seed=seed,
            num_mem_chunks=config.num_mem_chunks,
            compute_gradient=False,
            heldout_x=held_x,
        )
        loss = ensemble.heldout_member_log_loss(held_y)
        scores[scheme] = {
            "mutual_information": ensemble.probability_mutual_information(),
            "heldout_loss_variance": float(loss.var(unbiased=False)),
        }
    posterior = compute_intrinsic_posterior(model, fit_x, fit_y, query_x, num_mem_chunks=config.num_mem_chunks)
    joint = posterior.joint_mutual_information()
    query_count = query_x.shape[0]
    off_diagonal = ~torch.eye(query_count, dtype=torch.bool)
    per_query_joint = torch.nan_to_num(joint, nan=0.0).sum(dim=1) / off_diagonal.sum(dim=1).clamp_min(1)
    scores["joint_mi"] = {"mutual_information": per_query_joint, "heldout_loss_variance": None}
    scores["self_conditioning"] = {
        "mutual_information": posterior.self_conditioning(),
        "heldout_loss_variance": None,
    }
    return scores


def evaluate_condition(model, condition: str, config: ResamplingEvaluationConfig) -> dict[str, Any]:
    """Score every arm on ``episodes_per_condition`` held-out-family episodes of one condition."""
    rng = np.random.default_rng(config.seed)
    per_query_mean: dict[str, list[float]] = {name: [] for name in ARMS}
    performance_pairs: dict[str, list[tuple[float, float]]] = {name: [] for name in RESAMPLED_ARMS}
    teacher_mutual_information: list[float] = []
    for episode_index in range(config.episodes_per_condition):
        episode = sample_episode(
            rng,
            regime=HELDOUT_REGIME,
            condition=condition,
            batch_size=1,
            support_size=config.support_size,
            query_count=config.query_count,
            device=config.device,
        )
        base_positive = _base_positive(model, episode)
        targets = teacher_targets(episode, base_positive)
        teacher_mutual_information.append(float(targets.mutual_information.mean()))
        scores = episode_arm_scores(model, episode, config, seed=config.seed + episode_index)
        for name in ARMS:
            per_query_mean[name].append(float(scores[name]["mutual_information"].mean()))
        for name in RESAMPLED_ARMS:
            performance_pairs[name].append(
                (float(scores[name]["mutual_information"].mean()), scores[name]["heldout_loss_variance"])
            )
    return {
        "per_arm_mean": {name: float(np.mean(values)) for name, values in per_query_mean.items()},
        "per_arm_values": per_query_mean,
        "performance_pairs": performance_pairs,
        "teacher_mutual_information": teacher_mutual_information,
    }


def evaluate_random_label_null(model, config: ResamplingEvaluationConfig) -> dict[str, float]:
    """Gate 3: every arm's mutual information on episodes with zero true epistemic content."""
    rng = np.random.default_rng(config.seed + 909)
    per_arm: dict[str, list[float]] = {name: [] for name in ARMS}
    for trial in range(max(4, config.episodes_per_condition // 4)):
        episode = random_label_episode(
            rng, batch_size=1, support_size=config.support_size, query_count=config.query_count, device=config.device
        )
        scores = episode_arm_scores(model, episode, config, seed=config.seed + 5000 + trial)
        for name in ARMS:
            per_arm[name].append(float(scores[name]["mutual_information"].mean()))
    return {name: float(np.mean(values)) for name, values in per_arm.items()}


def _permutation_band(
    dispersion: np.ndarray, loss_variance: np.ndarray, *, trials: int, seed: int
) -> dict[str, float]:
    """Null band for the dispersion/performance Spearman correlation under a random member pairing."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(trials):
        permuted = rng.permutation(loss_variance)
        draws.append(_spearman(dispersion, permuted))
    finite = np.asarray([value for value in draws if np.isfinite(value)])
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(finite.mean()), "std": float(finite.std())}


def gate_2_ground_truth_fit(condition_reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Gate 2: Spearman and R² of each arm's per-episode MI against the exact teacher, pooled across conditions."""
    teacher = np.concatenate([np.asarray(report["teacher_mutual_information"]) for report in condition_reports.values()])
    result = {}
    for name in ARMS:
        predicted = np.concatenate([np.asarray(report["per_arm_values"][name]) for report in condition_reports.values()])
        result[name] = {
            "spearman": _spearman(predicted, teacher),
            "r2": _r2(predicted, teacher),
            "clears_r2_floor": bool(_r2(predicted, teacher) >= MIN_R2),
        }
    return result


def gate_1_discrimination(condition_reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gate 1: held-out-family ambiguous-vs-identifiable AUC."""
    ambiguous, identifiable = condition_reports["ambiguous"], condition_reports["identifiable"]
    result = {}
    for name in ARMS:
        scores = np.asarray(ambiguous["per_arm_values"][name] + identifiable["per_arm_values"][name])
        labels = np.asarray([1] * len(ambiguous["per_arm_values"][name]) + [0] * len(identifiable["per_arm_values"][name]))
        auc = _auc(scores, labels)
        result[name] = {"auc": auc, "clears_auc_floor": bool(auc is not None and auc >= MIN_CONDITION_AUC)}
    return result


def gate_4_performance_link(
    condition_reports: dict[str, dict[str, Any]], config: ResamplingEvaluationConfig
) -> dict[str, dict[str, Any]]:
    """Gate 4: |Spearman| between episode dispersion and held-out loss variance, outside the permutation null."""
    result = {}
    for name in RESAMPLED_ARMS:
        pairs = [
            pair
            for report in condition_reports.values()
            for pair in report["performance_pairs"][name]
            if np.isfinite(pair[1])
        ]
        if len(pairs) < 4:
            result[name] = {"spearman": None, "null_band": None, "clears_floor": False}
            continue
        dispersion = np.asarray([pair[0] for pair in pairs])
        loss_variance = np.asarray([pair[1] for pair in pairs])
        statistic = _spearman(dispersion, loss_variance)
        band = _permutation_band(dispersion, loss_variance, trials=config.permutation_trials, seed=config.seed)
        result[name] = {
            "spearman": statistic,
            "null_band": band,
            "clears_floor": bool(
                np.isfinite(statistic)
                and abs(statistic) >= MIN_PERFORMANCE_LINK
                and abs(statistic - band["mean"]) >= 2 * (band["std"] or 1.0)
            ),
        }
    return result


def gate_3_null(null_report: dict[str, float]) -> dict[str, dict[str, Any]]:
    return {
        name: {"mutual_information": value, "clears_floor": bool(value <= MAX_NULL_MUTUAL_INFORMATION)}
        for name, value in null_report.items()
    }


def run_resampling_synthetic_evaluation(config: ResamplingEvaluationConfig, output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = init_model_from_state_dict_file(str(Path(config.checkpoint).expanduser().resolve()))
    model = model.to(config.device).requires_grad_(False).eval()

    condition_reports = {}
    for condition in ("ambiguous", "identifiable", "noisy"):
        print(f"evaluating condition={condition}", flush=True)
        condition_reports[condition] = evaluate_condition(model, condition, config)
        print(f"  {condition}: {json.dumps(condition_reports[condition]['per_arm_mean'], sort_keys=True)}", flush=True)

    print("evaluating random-label null", flush=True)
    null_report = evaluate_random_label_null(model, config)
    print(f"  null: {json.dumps(null_report, sort_keys=True)}", flush=True)

    gates = {
        "discrimination": gate_1_discrimination(condition_reports),
        "ground_truth_fit": gate_2_ground_truth_fit(condition_reports),
        "null": gate_3_null(null_report),
        "performance_link": gate_4_performance_link(condition_reports, config),
    }
    decision = {
        name: bool(
            gates["discrimination"][name]["clears_auc_floor"]
            and gates["ground_truth_fit"][name]["clears_r2_floor"]
            and gates["null"][name]["clears_floor"]
        )
        for name in ARMS
    }
    result = {
        "config": asdict(config),
        "conditions": {
            name: {"per_arm_mean": report["per_arm_mean"], "teacher_mutual_information_mean": float(np.mean(report["teacher_mutual_information"]))}
            for name, report in condition_reports.items()
        },
        "random_label_null": null_report,
        "gates": gates,
        "clears_gates_1_2_3": decision,
        "any_arm_proceeds_to_stage_2": bool(any(decision.values())),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "headline.csv").write_text(_headline_csv(condition_reports, ARMS))
    print(json.dumps(decision, indent=2), flush=True)
    return output.resolve()


def _headline_csv(condition_reports: dict[str, dict[str, Any]], arms: tuple[str, ...]) -> str:
    header = "condition,arm,mean_mutual_information\n"
    rows = [
        f"{condition},{arm},{report['per_arm_mean'][arm]:.6f}"
        for condition, report in condition_reports.items()
        for arm in arms
    ]
    return header + "\n".join(rows) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ResamplingEvaluationConfig()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--device", default=defaults.device)
    for name in (
        "seed",
        "episodes_per_condition",
        "support_size",
        "query_count",
        "members",
        "heldout_size",
        "num_mem_chunks",
        "permutation_trials",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--subsample-fraction", type=float, default=defaults.subsample_fraction)
    return parser


if __name__ == "__main__":
    arguments = vars(build_parser().parse_args())
    destination = arguments.pop("output_dir")
    print(f"Wrote resampling synthetic evaluation to {run_resampling_synthetic_evaluation(ResamplingEvaluationConfig(**arguments), destination)}")
