"""TabArena evaluation for support-resampling and intrinsic-posterior uncertainty.

Stage 2 of the trial described in ``SUPPORT_RESAMPLING_VARIANCE_TRIAL.md``. Only
arms that cleared gates 1-3 in the synthetic stage
(``evaluate_resampling_synthetic.py``) should be pointed at this script; it
follows ``evaluate_continuous_tabarena.py`` exactly -- identical deterministic
context rows, uncertainty built only from labelled context rows, scoring only on
the untouched official test partition, and query labels never entering model
construction.

The comparator that matters is plain vanilla predictive entropy
(``AUROC 0.767 / AURC 0.076`` on these 20 tasks per
``runs/bayesian_nanotabpfn/tracking/current_uncertainty_per_task.csv``): every
uncertainty trial in this repository so far has lost to it.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfmplayground.bayesian_interface import VanillaNanoTabPFNClassifier
from tfmplayground.continuous_interface import ContextResamplingClassifier
from tfmplayground.experiments.evaluate_bayesian_tabarena import static_metrics
from tfmplayground.experiments.evaluate_continuous_tabarena import (
    TABARENA_BINARY_TASK_NAMES,
    _bootstrap_interval,
    _macro,
    error_detection_scores,
    tabarena_gate,
)
from tfmplayground.experiments.support_resampling import SCHEMES
from tfmplayground.support_resampling_interface import IntrinsicPosteriorClassifier, SupportResamplingClassifier

#: Arms cleared by the synthetic stage; overridden per run by ``--arms``.
DEFAULT_ARMS = (*SCHEMES, "joint_mi", "self_conditioning")


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def build_arms(
    *,
    vanilla_checkpoint: str,
    arms: tuple[str, ...],
    context_size: int,
    device: str,
    members: int,
    ensemble_seed: int,
    resampling_subsets: int,
) -> dict[str, Any]:
    """Instantiate every compared arm with the identical context protocol."""
    built: dict[str, Any] = {
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
    for name in arms:
        if name in SCHEMES:
            built[name] = SupportResamplingClassifier(
                vanilla_checkpoint,
                context_size=context_size,
                random_state=0,
                device=device,
                scheme=name,
                members=members,
                ensemble_seed=ensemble_seed,
            )
        elif name in ("joint_mi", "self_conditioning"):
            built[name] = IntrinsicPosteriorClassifier(
                vanilla_checkpoint, context_size=context_size, random_state=0, device=device, score=name
            )
        else:
            raise ValueError(f"Unknown arm {name!r}; expected one of {DEFAULT_ARMS}.")
    return built


def evaluate_split(X_train, y_train, X_test, y_test, *, arms: dict[str, Any]) -> dict[str, Any]:
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
            entry["mean_mutual_information"] = float(information.mean())
        report["arms"][name] = entry
    return report


def summarize(reports: list[dict[str, Any]], *, arm_names: list[str], seed: int = 0) -> dict[str, Any]:
    """Macro averages, bootstrap intervals, and the acceptance gate -- same shape as the continuous trial's."""
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
        scores = {}
        for metric in ("error_auroc", "aurc"):
            values = _macro(reports, ("arms", name, "raw_epistemic", metric))
            scores[metric] = float(np.mean(values)) if values else None
            interval = _bootstrap_interval(values, seed=seed)
            if interval is not None:
                entry["bootstrap"][f"raw_epistemic.{metric}"] = interval
        entry["raw_epistemic"] = scores if any(value is not None for value in scores.values()) else None
        values = _macro(reports, ("arms", name, "mean_mutual_information"))
        if values:
            entry["mean_mutual_information"] = float(np.mean(values))
        entry["gate"] = tabarena_gate(entry, vanilla_point)
        summary["arms"][name] = entry
    return summary


def run_resampling_tabarena_evaluation(
    *,
    task_ids: list[int],
    output_dir: str,
    vanilla_checkpoint: str = "checkpoints/nanotabpfn.pth",
    arms: tuple[str, ...] = DEFAULT_ARMS,
    context_size: int = 1024,
    device: str = "cpu",
    members: int = 32,
    ensemble_seed: int = 0,
    resampling_subsets: int = 16,
) -> Path:
    """Evaluate every arm on the shared official binary tasks."""
    from autogluon.features import AutoMLPipelineFeatureGenerator
    from tabarena.benchmark.task.openml.spec import OpenMLTaskSpec

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_output = output / "tasks"
    task_output.mkdir(exist_ok=True)
    protocol = {
        "tabarena_role": "evaluation only",
        "context_size": context_size,
        "context_selection": "identical deterministic subset with random_state=0",
        "test_partition": "official untouched test partition",
        "arms": list(arms),
        "vanilla_checkpoint": vanilla_checkpoint,
        "expected_task_names": list(TABARENA_BINARY_TASK_NAMES),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    reports: list[dict[str, Any]] = []
    datasets: list[str] = []
    for index, task_id in enumerate(sorted(task_ids), start=1):
        cache = task_output / f"{task_id}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            datasets.append(saved["dataset"])
            reports.append(saved["report"])
            print(f"[{index}/{len(task_ids)}] {task_id} {saved['dataset']} (cached)", flush=True)
            continue
        task = OpenMLTaskSpec(task_id).load()
        train_indices, test_indices = task.get_split_indices(fold=0, repeat=0, sample=0)
        X_train, X_test = task.X.iloc[train_indices], task.X.iloc[test_indices]
        y_train, y_test = task.y.iloc[train_indices], task.y.iloc[test_indices]
        preprocessor = AutoMLPipelineFeatureGenerator(verbosity=0)
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        classifiers = build_arms(
            vanilla_checkpoint=vanilla_checkpoint,
            arms=arms,
            context_size=context_size,
            device=device,
            members=members,
            ensemble_seed=ensemble_seed,
            resampling_subsets=resampling_subsets,
        )
        report = evaluate_split(X_train, y_train, X_test, y_test, arms=classifiers)
        dataset = task.dataset_name or str(task_id)
        datasets.append(dataset)
        reports.append(report)
        cache.write_text(json.dumps({"dataset": dataset, "report": report}, indent=2) + "\n")
        print(f"[{index}/{len(task_ids)}] {task_id} {dataset}", flush=True)
        del task, preprocessor, X_train, X_test, y_train, y_test
        gc.collect()

    arm_names = sorted({name for report in reports for name in report["arms"]})
    summary = summarize(reports, arm_names=arm_names, seed=0)
    result = {"protocol": protocol, "datasets": datasets, "summary": summary}
    (output / "tabarena_resampling_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    frame = _per_task_frame(reports, datasets, sorted(task_ids))
    frame.to_csv(output / "tabarena_resampling_per_task.csv", index=False)
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
            scores = entry.get("raw_epistemic") or {}
            for metric, value in scores.items():
                row[f"{name}.raw_epistemic.{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", required=True, help="Comma-separated OpenML task ids, or a path to a JSON list.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vanilla-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="Comma-separated arm names.")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--ensemble-seed", type=int, default=0)
    parser.add_argument("--resampling-subsets", type=int, default=16)
    return parser


def _parse_task_ids(value: str) -> list[int]:
    path = Path(value)
    if path.is_file():
        return [int(item) for item in json.loads(path.read_text())]
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(
        "Wrote TabArena resampling metrics to "
        + str(
            run_resampling_tabarena_evaluation(
                task_ids=_parse_task_ids(arguments.task_ids),
                output_dir=arguments.output_dir,
                vanilla_checkpoint=arguments.vanilla_checkpoint,
                arms=tuple(arguments.arms.split(",")),
                context_size=arguments.context_size,
                device=arguments.device,
                members=arguments.members,
                ensemble_seed=arguments.ensemble_seed,
                resampling_subsets=arguments.resampling_subsets,
            )
        )
    )
