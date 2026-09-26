"""Evaluate observed heart-site tags with labelled examples from every site.

For each target site, keep all rows from the other sites in support, draw a
stratified set of labelled target-site shots, and score the remaining target-site
rows. The true-tag and shuffled-support-tag conditions have identical feature
width and site-tag frequencies. Query tags stay true because every query row
belongs to the target site. No model is retrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from tfmplayground.experiments.evaluate_heart_sites_merged import SITE_COLUMN, TARGET, load_merged
from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import (
    UnsupportedFoldError,
    _encode_fold_labels,
    baseline_selections,
    build_external_model,
    discover_v4_checkpoints,
    fold_metrics,
    predict_external_full_fold,
    predict_full_fold,
    prepare_fold_data,
)
from tfmplayground.models.nanotabpfn_native_v4 import NanoTabPFNModel as NativeV4NanoTabPFN

CONDITIONS = ("site_hidden", "site_shuffled", "site_true")
DATA_REVISION = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"


def adaptation_folds(
    frame: pd.DataFrame, *, shots: int = 20, repeats: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray, str, str, int]]:
    """Matched target-site queries with stratified target-site support shots."""

    if shots < 2 or repeats < 1:
        raise ValueError("shots must be at least two and repeats at least one")
    folds = []
    for site_index, site in enumerate(sorted(frame[SITE_COLUMN].unique())):
        site_indices = np.flatnonzero(frame[SITE_COLUMN].to_numpy() == site)
        other_indices = np.flatnonzero(frame[SITE_COLUMN].to_numpy() != site)
        if shots >= len(site_indices):
            raise ValueError(f"shots must be smaller than the {site} site size")
        site_labels = frame.iloc[site_indices][TARGET].to_numpy()
        splitter = StratifiedShuffleSplit(
            n_splits=repeats, train_size=shots, random_state=seed + 1009 * site_index
        )
        for repeat, (shot_positions, query_positions) in enumerate(
            splitter.split(site_indices, site_labels)
        ):
            support = np.sort(np.concatenate((other_indices, site_indices[shot_positions])))
            query = np.sort(site_indices[query_positions])
            if set(support) & set(query):
                raise ValueError(f"Overlapping support/query in {site}, repeat {repeat}")
            if set(frame.iloc[support][SITE_COLUMN]) != set(frame[SITE_COLUMN]):
                raise ValueError(f"Not every site is represented in support for {site}")
            folds.append((support, query, f"adapt-{site}-{repeat}", str(site), repeat))
    return folds


def canonical_model_name(selection: Any) -> str:
    if selection.model_kind in {"tabpfn", "tabicl", "sklearn"}:
        return str(selection.model_identity)
    if selection.checkpoint_path is None:
        raise ValueError("Native selection lacks a checkpoint path")
    return f"native-{Path(selection.checkpoint_path).parent.parent.name}"


def load_native_v4_checkpoint(path: Path, device: str) -> NativeV4NanoTabPFN:
    """Use the archived inference architecture paired with these checkpoints."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = NativeV4NanoTabPFN(**architecture)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def condition_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    condition: str,
    shuffle_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    if condition == "site_hidden":
        return train_frame, test_frame, (SITE_COLUMN,)
    if condition == "site_true":
        return train_frame, test_frame, ()
    if condition != "site_shuffled":
        raise ValueError(f"Unknown condition: {condition}")
    train_frame = train_frame.copy()
    rng = np.random.default_rng(shuffle_seed)
    train_frame[SITE_COLUMN] = rng.permutation(train_frame[SITE_COLUMN].to_numpy())
    return train_frame, test_frame, ()


def evaluate(
    frame: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray, str, str, int]],
    selections: Sequence[Any],
    *,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for selection in selections:
        name = canonical_model_name(selection)
        print(f"Evaluating {name} on {len(folds)} matched folds", flush=True)
        is_external = selection.model_kind in {"tabpfn", "tabicl", "sklearn"}
        native_model = None if is_external else load_native_v4_checkpoint(
            selection.checkpoint_path, device=device
        )
        for support, query, fold_name, target_site, repeat in folds:
            train_frame, test_frame = frame.iloc[support], frame.iloc[query]
            site_index = sorted(frame[SITE_COLUMN].unique()).index(target_site)
            shuffle_seed = seed + 1000003 * site_index + 1009 * repeat
            fold_features: dict[str, int] = {}
            for condition in CONDITIONS:
                conditioned_train, conditioned_test, excluded = condition_frames(
                    train_frame, test_frame, condition=condition, shuffle_seed=shuffle_seed
                )
                try:
                    train_x, test_x, _, _ = prepare_fold_data(
                        conditioned_train,
                        conditioned_test,
                        target_column=TARGET,
                        group_columns=excluded,
                    )
                    fold_features[condition] = int(train_x.shape[1])
                    train_y, test_y = _encode_fold_labels(
                        train_frame[TARGET].tolist(), test_frame[TARGET].tolist(), 2
                    )
                    if is_external:
                        model = build_external_model(selection, device)
                        probabilities = predict_external_full_fold(
                            model, train_x, train_y, test_x,
                            classes=2, query_chunk_size=query_chunk_size,
                        )
                    else:
                        probabilities = predict_full_fold(
                            native_model, train_x, train_y, test_x,
                            classes=2, device=device,
                            query_chunk_size=query_chunk_size,
                            num_mem_chunks=num_mem_chunks,
                        )
                    metrics = fold_metrics(train_y, test_y, probabilities, 2)
                    status, error = "ok", None
                except (UnsupportedFoldError, ValueError) as failure:
                    metrics, status, error = {}, "unsupported", str(failure)
                rows.append({
                    "model": name,
                    "condition": condition,
                    "fold": fold_name,
                    "target_site": target_site,
                    "repeat": repeat,
                    "target_site_shots": int((train_frame[SITE_COLUMN] == target_site).sum()),
                    "support_rows": len(support),
                    "query_rows": len(query),
                    "features": fold_features.get(condition),
                    "status": status,
                    "error": error,
                    **metrics,
                })
            if fold_features.get("site_shuffled") != fold_features.get("site_true"):
                raise ValueError(f"Tag-control feature width differs in {fold_name}")
        del native_model
    return rows


def summarize(rows: Sequence[dict[str, Any]], expected_folds: int) -> list[dict[str, Any]]:
    summary = []
    metrics = ("excess_cross_entropy", "accuracy_gain", "macro_ovr_auc", "model_cross_entropy")
    for name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == name]
        item: dict[str, Any] = {"model": name, "expected_folds": expected_folds}
        for condition in CONDITIONS:
            selected = [row for row in model_rows if row["condition"] == condition and row["status"] == "ok"]
            item[f"{condition}_folds"] = len(selected)
            for metric in metrics:
                values = [float(row[metric]) for row in selected if metric in row and not math.isnan(row[metric])]
                item[f"{condition}_{metric}"] = statistics.fmean(values) if values else float("nan")
        for metric in metrics:
            item[f"true_minus_shuffled_{metric}"] = (
                item[f"site_true_{metric}"] - item[f"site_shuffled_{metric}"]
            )
            item[f"true_minus_hidden_{metric}"] = (
                item[f"site_true_{metric}"] - item[f"site_hidden_{metric}"]
            )
        summary.append(item)
    return summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--run-roots", default=None)
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--num-mem-chunks", type=int, default=8)
    args = parser.parse_args(argv)

    dataset_root = Path(args.datasets_root)
    frame = load_merged(dataset_root)
    folds = adaptation_folds(frame, shots=args.shots, repeats=args.repeats, seed=args.seed)
    selections: list[Any] = []
    if args.run_roots:
        found, _ = discover_v4_checkpoints([Path(root) for root in args.run_roots.split(",")])
        selections.extend(selection for selection in found if selection.checkpoint_policy == "final")
    if args.baselines:
        external, _ = baseline_selections(args.baselines)
        selections.extend(external)
    if not selections:
        raise SystemExit("Pass --run-roots and/or --baselines")
    names = [canonical_model_name(selection) for selection in selections]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names in selection")

    rows = evaluate(
        frame, folds, selections,
        device=args.device,
        query_chunk_size=args.query_chunk_size,
        num_mem_chunks=args.num_mem_chunks,
        seed=args.seed,
    )
    summary = summarize(rows, expected_folds=len(folds))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_fold.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fold_manifest = [
        {
            "fold": fold_name,
            "target_site": target_site,
            "repeat": repeat,
            "support_rows": len(support),
            "query_rows": len(query),
            "support_index_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
            "query_index_sha256": hashlib.sha256(query.tobytes()).hexdigest(),
        }
        for support, query, fold_name, target_site, repeat in folds
    ]
    (output / "fold_manifest.json").write_text(json.dumps(fold_manifest, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "dataset_revision": DATA_REVISION,
        "dataset_rows": len(frame),
        "protocol": "stratified target-site shots; all other sites in support",
        "shots": args.shots,
        "repeats_per_site": args.repeats,
        "seed": args.seed,
        "models": names,
        "evaluation_source_sha256": sha256(Path(__file__)),
        "fold_evaluator_source_sha256": sha256(Path(__file__).resolve().parent / "evaluate_multiregime_v4_beyondarena.py"),
        "preprocessor_source_sha256": sha256(Path(__file__).resolve().parents[1] / "interface.py"),
        "native_model_source_sha256": sha256(Path(__file__).resolve().parents[1] / "models/nanotabpfn_native_v4.py"),
    }, indent=2) + "\n")
    incomplete = [
        (item["model"], condition, item[f"{condition}_folds"])
        for item in summary for condition in CONDITIONS
        if item[f"{condition}_folds"] != len(folds)
    ]
    if incomplete:
        print(f"Incomplete model-condition scores: {incomplete}", flush=True)
        return 2
    print(f"Scored {len(names)} models on {len(folds)} folds x {len(CONDITIONS)} conditions")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
