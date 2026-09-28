"""Measure TabICLv2 slot stability, occupancy, entropy, and ablation utility.

This reports observable grouping behavior on canonical multiclass tasks. It
does not assign semantic names to the slots or assume they are true regimes.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import torch
from openml.config import set_root_cache_directory
from sklearn.metrics import accuracy_score, adjusted_rand_score, log_loss

from tfmplayground.evaluation import TABARENA_TASKS
from tfmplayground.models.tabicl_slot_decoder import TabICLSlotDecoderClassifier


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_tasks(task_ids: list[int], max_train_rows: int) -> list[dict]:
    from openml.tasks import TaskType

    eligible = []
    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue
            dataset = task.get_dataset(download_data=False)
            qualities = dataset.qualities
            classes = int(qualities.get("NumberOfClasses", 0))
            features = int(qualities.get("NumberOfFeatures", 1)) - 1
            rows = int(qualities.get("NumberOfInstances", 0))
            if 3 <= classes <= 10 and features <= 500 and rows <= max_train_rows * 2:
                eligible.append({"task_id": int(task_id), "dataset": str(dataset.name), "n_classes": classes})
        except Exception as error:
            print(f"Skipping OpenML task {task_id}: {type(error).__name__}: {error}", flush=True)
    return eligible


def analyze(
    *,
    checkpoints: list[tuple[str, str]],
    output_dir: str,
    task_ids: list[int] | None = None,
    device: str = "cpu",
    cache_directory: str | None = None,
    max_train_rows: int = 10_000,
    limit_tasks: int | None = None,
) -> Path:
    if not checkpoints:
        raise ValueError("Provide at least one --checkpoint NAME=PATH.")
    if cache_directory:
        set_root_cache_directory(cache_directory)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    model_records = []
    for name, checkpoint_name in checkpoints:
        checkpoint_path = Path(checkpoint_name).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("slot_decoder")
        if not isinstance(metadata, dict) or metadata.get("head_kind") != "slot":
            raise ValueError(f"Expected a trained slot-decoder checkpoint, got {checkpoint_path}.")
        model_records.append(
            {
                "name": name,
                "path": checkpoint_path,
                "metadata": metadata,
                "sha256": _digest(checkpoint_path),
            }
        )

    tasks = _eligible_tasks(task_ids or TABARENA_TASKS, max_train_rows)
    if limit_tasks is not None:
        tasks = tasks[:limit_tasks]
    task_rows = []
    ablation_rows = []
    assignment_records: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}

    for task_info in tasks:
        task = openml.tasks.get_task(task_info["task_id"], download_splits=False)
        dataset = task.get_dataset(download_data=False)
        X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)
        if len(train_indices) > max_train_rows:
            continue
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        frequencies = y_train.value_counts(dropna=True)
        imbalance_ratio = float(frequencies.max() / frequencies.min()) if len(frequencies) else float("nan")

        for record in model_records:
            metadata = record["metadata"]
            slot_count = int(metadata["num_slots"])
            classifier = TabICLSlotDecoderClassifier(
                model_path=record["path"],
                expected_num_slots=slot_count,
                device=device,
                random_state=42,
            )
            classifier.fit(X_train, y_train)
            probabilities, diagnostics = classifier.predict_proba_with_slot_diagnostics(X_test)
            probabilities = np.asarray(probabilities)
            predicted = classifier.classes_[probabilities.argmax(axis=1)]
            assignment = diagnostics["support_assignments"].float().mean(dim=0).argmax(dim=-1).cpu().numpy()
            assignment_records.setdefault((task_info["dataset"], slot_count), []).append((record["name"], assignment))
            occupancy = diagnostics["slot_occupancy"].float().mean(dim=0).cpu().numpy()
            task_rows.append(
                {
                    "name": record["name"],
                    "checkpoint": str(record["path"]),
                    "checkpoint_sha256": record["sha256"],
                    "K": slot_count,
                    "seed": int(metadata["seed"]),
                    "task_id": task_info["task_id"],
                    "dataset": task_info["dataset"],
                    "n_classes": task_info["n_classes"],
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "imbalance_ratio": imbalance_ratio,
                    "accuracy": accuracy_score(y_test, predicted),
                    "log_loss": log_loss(y_test, probabilities, labels=classifier.classes_),
                    "slot_occupancy": occupancy.tolist(),
                    "slot_assignment_entropy": float(diagnostics["slot_assignment_entropy"].mean().item()),
                    "query_slot_entropy": float(diagnostics["query_slot_entropy"].mean().item()),
                }
            )

            all_slots = tuple(range(slot_count))
            for slot in all_slots:
                active_slots = [candidate for candidate in all_slots if candidate != slot]
                ablated = np.asarray(classifier.predict_proba_with_slot_mask(X_test, active_slots))
                ablation_rows.append(
                    {
                        "name": record["name"],
                        "K": slot_count,
                        "seed": int(metadata["seed"]),
                        "task_id": task_info["task_id"],
                        "dataset": task_info["dataset"],
                        "ablated_slot": slot,
                        "mean_total_variation": float(np.abs(probabilities - ablated).sum(axis=1).mean() / 2),
                        "log_loss": log_loss(y_test, ablated, labels=classifier.classes_),
                        "accuracy": accuracy_score(y_test, classifier.classes_[ablated.argmax(axis=1)]),
                    }
                )
            print(f"Analyzed {record['name']} on {task_info['dataset']}", flush=True)

    stability_rows = []
    for (dataset_name, slot_count), assignments in assignment_records.items():
        for (name_a, assignment_a), (name_b, assignment_b) in combinations(assignments, 2):
            stability_rows.append(
                {
                    "dataset": dataset_name,
                    "K": slot_count,
                    "model_a": name_a,
                    "model_b": name_b,
                    "support_assignment_ari": adjusted_rand_score(assignment_a, assignment_b),
                }
            )

    pd.DataFrame(task_rows).to_csv(output_path / "slot_diagnostics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(output_path / "slot_ablation.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output_path / "seed_stability.csv", index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", action="append", type=int, help="Limit analysis to selected OpenML tasks.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-directory")
    parser.add_argument("--max-train-rows", type=int, default=10_000)
    parser.add_argument("--limit-tasks", type=int, help="Useful for a quick diagnostics smoke run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = []
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}.")
        checkpoints.append(tuple(item.split("=", maxsplit=1)))
    output = analyze(
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        task_ids=args.task_id,
        device=args.device,
        cache_directory=args.cache_directory,
        max_train_rows=args.max_train_rows,
        limit_tasks=args.limit_tasks,
    )
    print(f"Wrote slot grouping diagnostics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
