"""Evaluate Bayesian uncertainty on the exact TabArena-Lite outer splits.

This companion to the official TabArena runner records metrics that TabArena's
standard result schema does not retain (mutual information, ESS, posterior
collapse, error-detection AUROC, and AURC).  It uses the same OpenML split,
model-agnostic preprocessing, deterministic context rows, and untouched test
partition as the official quality evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfmplayground.experiments.evaluate_bayesian_tabarena import evaluate_uncertainty_split


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _flatten_row(task_id: int, dataset: str, report: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"task_id": task_id, "dataset": dataset}
    for section in ("bayesian", "vanilla", "uncertainty_comparison"):
        row.update({f"{section}.{key}": value for key, value in report[section].items()})
    row.update(
        {
            "max_mean_probability_difference": report["max_mean_probability_difference"],
            "context_indices_identical": report["context_indices_identical"],
            "query_labels_used_for_construction": report["query_labels_used_for_construction"],
        }
    )
    return row


def run_uncertainty_evaluation(
    *,
    checkpoint: str,
    results_dir: str,
    output_dir: str,
    vanilla_checkpoint: str = "checkpoints/nanotabpfn.pth",
    context_size: int = 1024,
    device: str = "cpu",
) -> Path:
    from autogluon.features import AutoMLPipelineFeatureGenerator
    from tabarena.benchmark.task.openml.spec import OpenMLTaskSpec

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    vanilla_path = Path(vanilla_checkpoint).expanduser().resolve()
    result_root = Path(results_dir) / "data" / "BayesianNanoTabPFN_c1_default"
    task_ids = sorted(int(path.name) for path in result_root.iterdir() if path.is_dir())
    if not task_ids:
        raise ValueError(f"No completed BayesianNanoTabPFN tasks found under {result_root}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_output = output / "tasks"
    task_output.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for index, task_id in enumerate(task_ids, start=1):
        task_path = task_output / f"{task_id}.json"
        if task_path.exists():
            saved = json.loads(task_path.read_text())
            reports[str(task_id)] = saved
            rows.append(_flatten_row(task_id, saved["dataset"], saved))
            print(f"[{index}/{len(task_ids)}] {task_id} {saved['dataset']} (cached)", flush=True)
            continue
        task = OpenMLTaskSpec(task_id).load()
        train_indices, test_indices = task.get_split_indices(fold=0, repeat=0, sample=0)
        X_train, X_test = task.X.iloc[train_indices], task.X.iloc[test_indices]
        y_train, y_test = task.y.iloc[train_indices], task.y.iloc[test_indices]
        # TabArenaV0pt1ExperimentBundle's ``default`` pipeline is AutoGluon's
        # standard feature generator (not TabArena's optional new pipeline).
        preprocessor = AutoMLPipelineFeatureGenerator(verbosity=0)
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        report = evaluate_uncertainty_split(
            X_train,
            y_train,
            X_test,
            y_test,
            checkpoint=str(checkpoint_path),
            vanilla_checkpoint=str(vanilla_path),
            context_size=context_size,
            random_state=0,
            device=device,
        )
        dataset = task.dataset_name or str(task_id)
        saved = {"dataset": dataset, **report}
        reports[str(task_id)] = saved
        rows.append(_flatten_row(task_id, dataset, report))
        task_path.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"[{index}/{len(task_ids)}] {task_id} {dataset}", flush=True)
        del task, preprocessor, X_train, X_test, y_train, y_test
        gc.collect()

    frame = pd.DataFrame(rows).sort_values("task_id")
    comparison = {
        key: _mean(frame[f"uncertainty_comparison.{key}"].tolist())
        for key in (
            "bayesian_error_auroc",
            "vanilla_entropy_error_auroc",
            "bayesian_aurc",
            "vanilla_entropy_aurc",
        )
    }
    improves_auroc = comparison["bayesian_error_auroc"] > comparison["vanilla_entropy_error_auroc"]
    improves_aurc = comparison["bayesian_aurc"] < comparison["vanilla_entropy_aurc"]
    no_auroc_harm = comparison["bayesian_error_auroc"] >= comparison["vanilla_entropy_error_auroc"] - 0.01
    no_aurc_harm = comparison["bayesian_aurc"] <= comparison["vanilla_entropy_aurc"] + 0.01
    summary = {
        "task_count": len(frame),
        "checkpoint": str(checkpoint_path),
        "vanilla_checkpoint": str(vanilla_path),
        "context_size": context_size,
        "max_mean_probability_difference": float(frame["max_mean_probability_difference"].max()),
        "context_indices_identical": bool(frame["context_indices_identical"].all()),
        "query_labels_used_for_construction": bool(frame["query_labels_used_for_construction"].any()),
        "mean_bayesian_mutual_information": _mean(frame["bayesian.epistemic_uncertainty"].tolist()),
        "mean_bayesian_epistemic_variance": _mean(frame["bayesian.epistemic_variance"].tolist()),
        "mean_effective_sample_size": _mean(frame["bayesian.effective_sample_size"].tolist()),
        "mean_posterior_collapse_fraction": _mean(frame["bayesian.posterior_collapse_fraction"].tolist()),
        "uncertainty_comparison": {
            **comparison,
            "improves_error_auroc": bool(improves_auroc),
            "improves_aurc": bool(improves_aurc),
            "no_material_error_auroc_harm": bool(no_auroc_harm),
            "no_material_aurc_harm": bool(no_aurc_harm),
            "accepted": bool((improves_auroc or improves_aurc) and no_auroc_harm and no_aurc_harm),
        },
    }
    frame.to_csv(output / "uncertainty_per_task.csv", index=False)
    (output / "uncertainty_per_task.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vanilla-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Wrote uncertainty metrics to {run_uncertainty_evaluation(**vars(args))}")
