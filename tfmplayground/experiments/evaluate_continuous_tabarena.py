"""TabArena evaluation for slot-free continuous uncertainty (evaluation only).

Every arm receives identical deterministic context rows, builds uncertainty from
labelled context rows only, and is scored on the untouched official test rows.
Query labels never enter model construction.  Because every learned arm returns
the frozen vanilla mean, the predictive metrics must match vanilla to numerical
tolerance; only the uncertainty ranking can differ.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfmplayground.bayesian_interface import BayesianNanoTabPFNClassifier, VanillaNanoTabPFNClassifier
from tfmplayground.continuous_interface import ContextResamplingClassifier, ContinuousUncertaintyClassifier
from tfmplayground.experiments.evaluate_bayesian_tabarena import static_metrics

#: The 20 official binary tasks used by the saved slot trial.
TABARENA_BINARY_TASK_NAMES = (
    "Bank_Customer_Churn",
    "blood-transfusion-service-center",
    "churn",
    "coil2000_insurance_policies",
    "credit-g",
    "diabetes",
    "E-CommereShippingData",
    "Fitness_Club",
    "hazelnut-spread-contaminant-detection",
    "heloc",
    "in_vehicle_coupon_recommendation",
    "Is-this-a-good-customer",
    "Marketing_Campaign",
    "NATICUSdroid",
    "online_shoppers_intention",
    "polish_companies_bankruptcy",
    "qsar-biodeg",
    "seismic-bumps",
    "taiwanese_bankruptcy_prediction",
    "jm1",
)

#: Acceptance margins for the TabArena usefulness gate.
AUROC_MARGIN = 0.01
AURC_MARGIN = 0.005


def error_detection_scores(scores: np.ndarray, errors: np.ndarray) -> dict[str, float | None]:
    """Error-detection AUROC and selective-risk AURC for one ranking."""
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=int)
    if scores.shape != errors.shape or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and aligned with the error indicator.")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    aurc = float(cumulative.mean())
    auroc: float | None = None
    if np.unique(errors).size == 2:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(errors, scores))
    return {"error_auroc": auroc, "aurc": aurc}


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def build_arms(
    *,
    vanilla_checkpoint: str,
    continuous_checkpoints: dict[str, str],
    slot_checkpoint: str | None,
    slot_hypotheses: int,
    context_size: int,
    device: str,
    num_samples: int,
    inference_seed: int,
    resampling_subsets: int,
) -> dict[str, Any]:
    """Instantiate every compared arm with the identical context protocol."""
    arms: dict[str, Any] = {
        "vanilla": VanillaNanoTabPFNClassifier(
            vanilla_checkpoint, context_size=context_size, random_state=0, device=device
        ),
        "context_resampling": ContextResamplingClassifier(
            vanilla_checkpoint,
            context_size=context_size,
            random_state=0,
            device=device,
            num_subsets=resampling_subsets,
        ),
    }
    if slot_checkpoint:
        arms["slot"] = BayesianNanoTabPFNClassifier(
            slot_checkpoint,
            num_hypotheses=slot_hypotheses,
            context_size=context_size,
            random_state=0,
            device=device,
        )
    for name, path in continuous_checkpoints.items():
        arms[name] = ContinuousUncertaintyClassifier(
            path,
            context_size=context_size,
            random_state=0,
            device=device,
            num_samples=num_samples,
            inference_seed=inference_seed,
        )
    return arms


def evaluate_split(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    arms: dict[str, Any],
    risk_lambda: float,
) -> dict[str, Any]:
    """Score every arm on one untouched binary test split."""
    fitted = {name: arm.fit(X_train, y_train) for name, arm in arms.items()}
    probabilities = {name: arm.predict_proba(X_test) for name, arm in fitted.items()}
    encoded = fitted["vanilla"].label_encoder_.transform(np.asarray(y_test))

    vanilla_probabilities = probabilities["vanilla"]
    vanilla_entropy = _binary_entropy(vanilla_probabilities)
    vanilla_errors = (vanilla_probabilities.argmax(1) != encoded).astype(int)
    vanilla_uncertainty = error_detection_scores(vanilla_entropy, vanilla_errors)

    report: dict[str, Any] = {
        "vanilla_predictive_entropy": vanilla_uncertainty,
        "context_indices_identical": True,
        "query_labels_used_for_construction": False,
        "arms": {},
    }
    reference_indices = fitted["vanilla"]._context_indices()
    for name, arm in fitted.items():
        arm_probabilities = probabilities[name]
        errors = (arm_probabilities.argmax(1) != encoded).astype(int)
        entry: dict[str, Any] = {
            "predictive": static_metrics(encoded, arm_probabilities),
            "max_probability_difference_to_vanilla": float(
                np.max(np.abs(arm_probabilities - vanilla_probabilities))
            ),
        }
        if not np.array_equal(arm._context_indices(), reference_indices):
            report["context_indices_identical"] = False
        diagnostics = getattr(arm, "last_diagnostics_", None)
        if diagnostics is not None and "mutual_information" in diagnostics:
            information = np.asarray(diagnostics["mutual_information"], dtype=float)
            entry["raw_epistemic"] = error_detection_scores(information, errors)
            entry["combined_risk"] = error_detection_scores(
                _binary_entropy(arm_probabilities) + risk_lambda * information, errors
            )
            entry["mean_mutual_information"] = float(information.mean())
            if "epistemic_variance" in diagnostics:
                entry["mean_epistemic_variance"] = float(np.mean(diagnostics["epistemic_variance"]))
            if "expected_conditional_entropy" in diagnostics:
                entry["mean_expected_conditional_entropy"] = float(
                    np.mean(diagnostics["expected_conditional_entropy"])
                )
            if "sample_mean_preservation_error" in diagnostics:
                entry["max_sample_mean_preservation_error"] = float(
                    np.max(diagnostics["sample_mean_preservation_error"])
                )
        report["arms"][name] = entry
    return report


def _bootstrap_interval(
    values: list[float], *, iterations: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, float] | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(iterations, finite.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(finite.mean()),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
    }


def _macro(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        node: Any = row
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (int, float)) and np.isfinite(node):
            values.append(float(node))
    return values


def tabarena_gate(arm_summary: dict[str, Any], vanilla_summary: dict[str, Any]) -> dict[str, Any]:
    """Gate 6: usefulness relative to vanilla predictive entropy."""
    result: dict[str, Any] = {}
    base_auroc = vanilla_summary.get("error_auroc")
    base_aurc = vanilla_summary.get("aurc")
    for score_name in ("raw_epistemic", "combined_risk"):
        scores = arm_summary.get(score_name)
        if not scores or base_auroc is None or base_aurc is None:
            result[score_name] = None
            continue
        auroc, aurc = scores.get("error_auroc"), scores.get("aurc")
        if auroc is None or aurc is None:
            result[score_name] = None
            continue
        result[score_name] = {
            "error_auroc": auroc,
            "aurc": aurc,
            "no_material_auroc_harm": bool(auroc >= base_auroc - AUROC_MARGIN),
            "no_material_aurc_harm": bool(aurc <= base_aurc + AURC_MARGIN),
            "improves_auroc": bool(auroc >= base_auroc + AUROC_MARGIN),
            "improves_aurc": bool(aurc <= base_aurc - AURC_MARGIN),
        }
    candidates = [value for value in result.values() if value is not None]
    result["passed"] = bool(
        candidates
        and all(value["no_material_auroc_harm"] and value["no_material_aurc_harm"] for value in candidates)
        and any(value["improves_auroc"] or value["improves_aurc"] for value in candidates)
    )
    return result


def summarize(reports: list[dict[str, Any]], *, arm_names: list[str], seed: int = 0) -> dict[str, Any]:
    """Macro averages, bootstrap intervals, and the acceptance gate."""
    vanilla_entropy = {
        key: _bootstrap_interval(_macro(reports, ("vanilla_predictive_entropy", key)), seed=seed)
        for key in ("error_auroc", "aurc")
    }
    vanilla_point = {key: (value or {}).get("mean") for key, value in vanilla_entropy.items()}
    summary: dict[str, Any] = {
        "task_count": len(reports),
        "vanilla_predictive_entropy": {"bootstrap": vanilla_entropy, "macro": vanilla_point},
        "arms": {},
    }
    for name in arm_names:
        entry: dict[str, Any] = {"predictive": {}, "bootstrap": {}}
        for metric in ("nll", "brier", "roc_auc", "accuracy", "ece"):
            values = _macro(reports, ("arms", name, "predictive", metric))
            entry["predictive"][metric] = float(np.mean(values)) if values else None
        differences = _macro(reports, ("arms", name, "max_probability_difference_to_vanilla"))
        entry["max_probability_difference_to_vanilla"] = float(np.max(differences)) if differences else None
        entry["matches_vanilla_probabilities_at_1e-6"] = bool(differences and max(differences) <= 1e-6)
        for score_name in ("raw_epistemic", "combined_risk"):
            scores = {}
            for metric in ("error_auroc", "aurc"):
                values = _macro(reports, ("arms", name, score_name, metric))
                scores[metric] = float(np.mean(values)) if values else None
                interval = _bootstrap_interval(values, seed=seed)
                if interval is not None:
                    entry["bootstrap"][f"{score_name}.{metric}"] = interval
            entry[score_name] = scores if any(value is not None for value in scores.values()) else None
        for extra in (
            "mean_mutual_information",
            "mean_epistemic_variance",
            "mean_expected_conditional_entropy",
            "max_sample_mean_preservation_error",
        ):
            values = _macro(reports, ("arms", name, extra))
            if values:
                entry[extra] = float(np.mean(values))
        entry["gate"] = tabarena_gate(entry, vanilla_point)
        summary["arms"][name] = entry
    return summary


def run_tabarena_uncertainty(
    *,
    task_ids: list[int],
    output_dir: str,
    vanilla_checkpoint: str = "checkpoints/nanotabpfn.pth",
    continuous_checkpoints: dict[str, str] | None = None,
    slot_checkpoint: str | None = None,
    slot_hypotheses: int = 4,
    context_size: int = 1024,
    device: str = "cpu",
    num_samples: int = 32,
    inference_seeds: tuple[int, ...] = (0, 1, 2),
    resampling_subsets: int = 16,
    risk_lambda: float = 0.5,
) -> Path:
    """Evaluate every arm on the shared official binary tasks."""
    from autogluon.features import AutoMLPipelineFeatureGenerator
    from tabarena.benchmark.task.openml.spec import OpenMLTaskSpec

    continuous_checkpoints = continuous_checkpoints or {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_output = output / "tasks"
    task_output.mkdir(exist_ok=True)
    protocol = {
        "tabarena_role": "evaluation only",
        "context_size": context_size,
        "context_selection": "identical deterministic subset with random_state=0",
        "test_partition": "official untouched test partition",
        "inference_seeds": list(inference_seeds),
        "risk_lambda": risk_lambda,
        "risk_lambda_selection": "held-out synthetic ordinary episodes; never TabArena labels",
        "vanilla_checkpoint_sha256": hashlib.sha256(Path(vanilla_checkpoint).read_bytes()).hexdigest(),
        "checkpoints": continuous_checkpoints,
        "slot_checkpoint": slot_checkpoint,
        "expected_task_names": list(TABARENA_BINARY_TASK_NAMES),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    per_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in inference_seeds}
    datasets: list[str] = []
    for index, task_id in enumerate(sorted(task_ids), start=1):
        cache = task_output / f"{task_id}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            datasets.append(saved["dataset"])
            for seed in inference_seeds:
                per_seed[seed].append(saved["seeds"][str(seed)])
            print(f"[{index}/{len(task_ids)}] {task_id} {saved['dataset']} (cached)", flush=True)
            continue
        task = OpenMLTaskSpec(task_id).load()
        train_indices, test_indices = task.get_split_indices(fold=0, repeat=0, sample=0)
        X_train, X_test = task.X.iloc[train_indices], task.X.iloc[test_indices]
        y_train, y_test = task.y.iloc[train_indices], task.y.iloc[test_indices]
        preprocessor = AutoMLPipelineFeatureGenerator(verbosity=0)
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        seed_reports: dict[str, Any] = {}
        for seed in inference_seeds:
            arms = build_arms(
                vanilla_checkpoint=vanilla_checkpoint,
                continuous_checkpoints=continuous_checkpoints,
                slot_checkpoint=slot_checkpoint,
                slot_hypotheses=slot_hypotheses,
                context_size=context_size,
                device=device,
                num_samples=num_samples,
                inference_seed=seed,
                resampling_subsets=resampling_subsets,
            )
            report = evaluate_split(X_train, y_train, X_test, y_test, arms=arms, risk_lambda=risk_lambda)
            seed_reports[str(seed)] = report
            per_seed[seed].append(report)
        dataset = task.dataset_name or str(task_id)
        datasets.append(dataset)
        cache.write_text(json.dumps({"dataset": dataset, "seeds": seed_reports}, indent=2) + "\n")
        print(f"[{index}/{len(task_ids)}] {task_id} {dataset}", flush=True)
        del task, preprocessor, X_train, X_test, y_train, y_test
        gc.collect()

    arm_names = sorted({name for report in per_seed[inference_seeds[0]] for name in report["arms"]})
    summaries = {seed: summarize(reports, arm_names=arm_names, seed=0) for seed, reports in per_seed.items()}
    variability: dict[str, Any] = {}
    for name in arm_names:
        for score_name in ("raw_epistemic", "combined_risk"):
            for metric in ("error_auroc", "aurc"):
                values = [
                    (summaries[seed]["arms"][name].get(score_name) or {}).get(metric) for seed in inference_seeds
                ]
                finite = [value for value in values if value is not None]
                if finite:
                    variability[f"{name}.{score_name}.{metric}"] = {
                        "per_seed": finite,
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                    }
    result = {
        "protocol": protocol,
        "datasets": datasets,
        "per_seed_summary": {str(seed): summary for seed, summary in summaries.items()},
        "seed_variability": variability,
        "primary_summary": summaries[inference_seeds[0]],
    }
    (output / "tabarena_uncertainty_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    frame = _per_task_frame(per_seed[inference_seeds[0]], datasets, sorted(task_ids))
    frame.to_csv(output / "tabarena_uncertainty_per_task.csv", index=False)
    return output.resolve()


def _per_task_frame(reports: list[dict[str, Any]], datasets: list[str], task_ids: list[int]) -> pd.DataFrame:
    rows = []
    for task_id, dataset, report in zip(task_ids, datasets, reports, strict=False):
        row: dict[str, Any] = {"task_id": task_id, "dataset": dataset}
        row["vanilla_entropy.error_auroc"] = report["vanilla_predictive_entropy"]["error_auroc"]
        row["vanilla_entropy.aurc"] = report["vanilla_predictive_entropy"]["aurc"]
        for name, entry in report["arms"].items():
            for metric, value in entry["predictive"].items():
                row[f"{name}.{metric}"] = value
            row[f"{name}.max_probability_difference"] = entry["max_probability_difference_to_vanilla"]
            for score_name in ("raw_epistemic", "combined_risk"):
                scores = entry.get(score_name) or {}
                for metric, value in scores.items():
                    row[f"{name}.{score_name}.{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", required=True, help="Comma-separated OpenML task ids, or a path to a JSON list.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vanilla-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--slot-checkpoint", default=None)
    parser.add_argument("--slot-hypotheses", type=int, default=4)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--inference-seeds", default="0,1,2")
    parser.add_argument("--resampling-subsets", type=int, default=16)
    parser.add_argument("--risk-lambda", type=float, required=True)
    return parser


def _parse_task_ids(value: str) -> list[int]:
    path = Path(value)
    if path.is_file():
        return [int(item) for item in json.loads(path.read_text())]
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(
        "Wrote TabArena uncertainty metrics to "
        + str(
            run_tabarena_uncertainty(
                task_ids=_parse_task_ids(arguments.task_ids),
                output_dir=arguments.output_dir,
                vanilla_checkpoint=arguments.vanilla_checkpoint,
                continuous_checkpoints=dict(item.split("=", 1) for item in arguments.checkpoint),
                slot_checkpoint=arguments.slot_checkpoint,
                slot_hypotheses=arguments.slot_hypotheses,
                context_size=arguments.context_size,
                device=arguments.device,
                num_samples=arguments.num_samples,
                inference_seeds=tuple(int(item) for item in arguments.inference_seeds.split(",")),
                resampling_subsets=arguments.resampling_subsets,
                risk_lambda=arguments.risk_lambda,
            )
        )
    )
