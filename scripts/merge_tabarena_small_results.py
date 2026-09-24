#!/usr/bin/env python3
"""Merge per-checkpoint evaluate_tabarena_small.py outputs into one report.

Each input directory holds one checkpoint's (or the shared baselines') own
fold_metrics.csv, written by a split array-job task that only ever scored its
own checkpoint plus, in exactly one task, the sklearn/TabPFN baselines. Model
names are disjoint across inputs by construction, so this concatenates rows
and recomputes per_dataset.csv/overall.csv exactly as evaluate_tabarena_small.
run() does over the combined rows -- no result here is a repeat of another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge(input_dirs: list[Path], output_dir: Path) -> None:
    frames = []
    for directory in input_dirs:
        path = directory / "fold_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; that task did not complete.")
        frames.append(pd.read_csv(path))
    metrics = pd.concat(frames, ignore_index=True)
    duplicate = metrics.duplicated(["dataset", "model", "fold"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate (dataset, model, fold) rows across inputs:\n{metrics[duplicate]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    per_dataset = (
        metrics.groupby(["dataset", "model", "labels"], as_index=False)
        .agg(
            predictors=("predictors", "first"),
            roc_auc=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            specificity=("specificity", "mean"),
            auprc=("auprc", "mean"),
            cross_entropy=("cross_entropy", "mean"),
            brier=("brier", "mean"),
            support_positive_pct=("support_positive_pct", "mean"),
            query_positive_pct=("query_positive_pct", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            predict_seconds_total=pd.NamedAgg(column="predict_seconds", aggfunc=lambda s: s.sum(min_count=1)),
            folds=("roc_auc", "size"),
        )
    )
    per_dataset.to_csv(output_dir / "per_dataset.csv", index=False)

    overall = (
        per_dataset.groupby(["model", "labels"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_specificity=("specificity", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_cross_entropy=("cross_entropy", "mean"),
            mean_brier=("brier", "mean"),
            fit_seconds_total=("fit_seconds_total", "sum"),
            predict_seconds_total=pd.NamedAgg(column="predict_seconds_total", aggfunc=lambda s: s.sum(min_count=1)),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    overall.to_csv(output_dir / "overall.csv", index=False)
    print(overall.round(4).to_string(index=False), flush=True)
    print(f"merged {len(input_dirs)} inputs, {len(metrics)} fold rows -> {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="Directory holding task-* subdirectories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = sorted(
        (p for p in args.input_root.glob("task-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not inputs:
        raise ValueError(f"No task-* directories under {args.input_root}")
    merge(inputs, args.output_dir)


if __name__ == "__main__":
    main()
