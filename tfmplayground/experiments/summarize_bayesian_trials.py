"""Build a compact, persistent tracker for Bayesian nanoTabPFN trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _trial_summary(name: str, path: Path) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame[frame["method"].str.contains("NanoTabPFN")].copy()
    frame["roc_auc"] = 1 - frame["metric_error"]
    pivot = frame.pivot(index="dataset", columns="method", values="roc_auc")
    bayesian = next(column for column in pivot if "Bayesian" in column)
    vanilla = next(column for column in pivot if "Vanilla" in column)
    delta = pivot[bayesian] - pivot[vanilla]
    per_task = pd.DataFrame(
        {
            "trial": name,
            "dataset": pivot.index,
            "bayesian_roc_auc": pivot[bayesian],
            "vanilla_roc_auc": pivot[vanilla],
            "roc_auc_delta": delta,
        }
    ).reset_index(drop=True)
    tolerance = 1e-12
    summary = {
        "trial": name,
        "task_count": len(pivot),
        "bayesian_mean_roc_auc": float(pivot[bayesian].mean()),
        "vanilla_mean_roc_auc": float(pivot[vanilla].mean()),
        "mean_roc_auc_delta": float(delta.mean()),
        "max_absolute_roc_auc_delta": float(delta.abs().max()),
        "wins": int((delta > tolerance).sum()),
        "ties": int((delta.abs() <= tolerance).sum()),
        "losses": int((delta < -tolerance).sum()),
        "source": str(path.resolve()),
    }
    return summary, per_task


def build_tracker(
    *,
    trials: list[str],
    output_dir: str,
    selection: str | None = None,
    uncertainty_summary: str | None = None,
    uncertainty_per_task: str | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    per_task_frames = []
    for specification in trials:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("Each trial must use NAME=RESULTS_PER_SPLIT_CSV syntax.")
        summary, per_task = _trial_summary(name, Path(raw_path))
        summaries.append(summary)
        per_task_frames.append(per_task)

    metadata = {
        "trials": summaries,
        "selection": None if selection is None else json.loads(Path(selection).read_text()),
        "uncertainty": (
            None if uncertainty_summary is None else json.loads(Path(uncertainty_summary).read_text())
        ),
    }
    pd.DataFrame(summaries).to_csv(output / "trial_summary.csv", index=False)
    pd.concat(per_task_frames, ignore_index=True).to_csv(output / "trial_per_task.csv", index=False)
    if uncertainty_per_task is not None:
        pd.read_csv(uncertainty_per_task).to_csv(output / "current_uncertainty_per_task.csv", index=False)
    (output / "trial_tracking.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", dest="trials", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--uncertainty-summary")
    parser.add_argument("--uncertainty-per-task")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Wrote trial tracker to {build_tracker(**vars(args))}")
