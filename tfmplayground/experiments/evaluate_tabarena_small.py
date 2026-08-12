"""TabArena evaluation under the small-data protocol.

Protocol (mirrors the reference experimental setup):
  * binary-classification TabArena datasets with **no missing values** and **at most 10
    predictors**;
  * datapoints **subsampled to 200** (stratified);
  * **stratified 5-fold cross-validation with 20 repetitions** (100 fits per dataset);
  * accumulated fit time recorded, prediction time excluded.

This is a materially different regime from `evaluate_integrated_tabarena.py`, which allowed up
to 500 features and used the full training split (contexts of several thousand rows) on a
single fold. With 200 points and 5-fold CV the context is ~160 rows, so the historical
128:128 prior/update split does not fit; the split is derived from the available rows instead.

Deviations from the reference setup, recorded for honesty: scikit-learn is 1.9.0 here (not
1.6.1) and the `tabpfn` package is not installed, so the TabPFN-v2 baseline is absent. The
nanoTabPFN backbone in this repo is used as the transformer baseline instead.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from tfmplayground.evaluation import TABARENA_TASKS
from tfmplayground.experiments.evaluate_integrated_tabarena import (
    predict_adaptive,
    predict_integrated,
    predict_vanilla,
    release_device_memory,
    split_prior_update_counts,
)
from tfmplayground.interface import get_feature_preprocessor, init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import load_adaptive_checkpoint
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class SmallTabArenaConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    integrated_checkpoints: str = ""
    adaptive_checkpoints: str = ""
    output_dir: str | None = None
    device: str = "cpu"
    cache_directory: str | None = None
    max_predictors: int = 10
    subsample: int = 200
    folds: int = 5
    repeats: int = 20
    query_chunk_size: int = 128
    num_mem_chunks: int = 1
    include_sklearn: bool = True
    task_ids: tuple[int, ...] = tuple(TABARENA_TASKS)


def eligible_tasks(config: SmallTabArenaConfig) -> list[dict[str, Any]]:
    """Binary classification, no missing values, at most `max_predictors` predictors."""
    selected = []
    for task_id in config.task_ids:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue
            dataset = task.get_dataset(download_data=False)
            qualities = dataset.qualities
            classes = int(qualities["NumberOfClasses"])
            missing = int(qualities.get("NumberOfMissingValues", 0) or 0)
            # OpenML counts the target in NumberOfFeatures.
            predictors = int(qualities["NumberOfFeatures"]) - 1
            if classes != 2 or missing != 0 or predictors > config.max_predictors:
                continue
            selected.append(
                {
                    "task_id": task_id,
                    "dataset": str(dataset.name),
                    "predictors": predictors,
                    "instances": int(qualities["NumberOfInstances"]),
                }
            )
        except Exception:  # keep the audit trail complete rather than aborting discovery
            continue
    return selected


def _subsample(x: pd.DataFrame, y: np.ndarray, size: int, seed: int):
    if len(y) <= size:
        return x, y
    index, _ = train_test_split(
        np.arange(len(y)), train_size=size, stratify=y, random_state=seed
    )
    return x.iloc[index], y[index]


def run(config: SmallTabArenaConfig) -> Path:
    set_randomness_seed(config.seed)
    if config.cache_directory is not None:
        set_root_cache_directory(config.cache_directory)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "tabarena_small" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    vanilla = init_model_from_state_dict_file(config.official_checkpoint)
    integrated = {}
    for entry in filter(None, config.integrated_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        integrated[name] = load_integrated_checkpoint(path)[0]
    adaptive = {}
    for entry in filter(None, config.adaptive_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        adaptive[name] = load_adaptive_checkpoint(path)[0]

    tasks = eligible_tasks(config)
    pd.DataFrame(tasks).to_csv(output_dir / "eligible_tasks.csv", index=False)
    print(f"eligible datasets: {len(tasks)}", flush=True)

    rows: list[dict[str, Any]] = []
    for task_info in tasks:
        task = openml.tasks.get_task(task_info["task_id"], download_splits=False)
        dataset = task.get_dataset(download_data=False)
        x, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        y = LabelEncoder().fit_transform(y)
        x, y = _subsample(x, y, config.subsample, config.seed)
        splitter = RepeatedStratifiedKFold(
            n_splits=config.folds, n_repeats=config.repeats, random_state=config.seed
        )
        for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y)):
            train_frame, test_frame = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if len(np.unique(y_test)) < 2:
                continue  # ROC AUC undefined
            preprocessor = get_feature_preprocessor(train_frame)
            train_x = preprocessor.fit_transform(train_frame)
            test_x = preprocessor.transform(test_frame)
            prior_count, update_count = split_prior_update_counts(len(y_train), 1, 1)

            def record(model_name: str, probability, fit_seconds: float):
                rows.append(
                    {
                        "dataset": task_info["dataset"],
                        "task_id": task_info["task_id"],
                        "predictors": task_info["predictors"],
                        "fold": fold_index,
                        "model": model_name,
                        "roc_auc": roc_auc_score(y_test, probability),
                        "accuracy": accuracy_score(y_test, probability >= 0.5),
                        "fit_seconds": fit_seconds,
                        "train_rows": len(y_train),
                        "test_rows": len(y_test),
                    }
                )

            start = time.perf_counter()
            vanilla_probability = predict_vanilla(
                vanilla, train_x, y_train, test_x,
                device=config.device,
                query_chunk_size=config.query_chunk_size,
                num_mem_chunks=config.num_mem_chunks,
            )
            record("vanilla", vanilla_probability, time.perf_counter() - start)
            release_device_memory(config.device)

            for name, model in integrated.items():
                start = time.perf_counter()
                probability = predict_integrated(
                    model, train_x, y_train, test_x,
                    prior_count=prior_count, update_count=update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record(name, probability, time.perf_counter() - start)
                release_device_memory(config.device)

            for name, model in adaptive.items():
                start = time.perf_counter()
                probability, _ = predict_adaptive(
                    model, train_x, y_train, test_x, vanilla_probability,
                    prior_count=prior_count, update_count=update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record(name, probability, time.perf_counter() - start)
                release_device_memory(config.device)

            if config.include_sklearn:
                for name, estimator in (
                    # Unscaled features make lbfgs hit its iteration cap and emit convergence warnings,
                    # which would understate the baseline; scale inside the fold to avoid leakage.
                    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                    ("random_forest", RandomForestClassifier(random_state=config.seed)),
                ):
                    start = time.perf_counter()
                    estimator.fit(train_x, y_train)
                    elapsed = time.perf_counter() - start
                    record(name, estimator.predict_proba(test_x)[:, 1], elapsed)

        pd.DataFrame(rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        done = len({row["dataset"] for row in rows})
        print(f"[{done}/{len(tasks)}] {task_info['dataset']} complete", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    per_dataset = (
        metrics.groupby(["dataset", "model"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy=("accuracy", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            folds=("roc_auc", "size"),
        )
    )
    per_dataset.to_csv(output_dir / "per_dataset.csv", index=False)
    overall = (
        per_dataset.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            fit_seconds_total=("fit_seconds_total", "sum"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    overall.to_csv(output_dir / "overall.csv", index=False)
    print(overall.round(4).to_string(index=False), flush=True)
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--integrated-checkpoints",
        default="",
        help="Comma-separated name=path list of integrated (no-gate) latent filter checkpoints.",
    )
    parser.add_argument(
        "--adaptive-checkpoints",
        default="",
        help="Comma-separated name=path list of adaptive (gated) particle filter checkpoints.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--max-predictors", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--num-mem-chunks", type=int, default=1)
    parser.add_argument("--include-sklearn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    output = run(SmallTabArenaConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote small-protocol TabArena artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
