"""Archive matched heart-site adaptation results from one or more evaluator runs.

Usage: python collect_heart_adaptation_results.py RUN_DIR [RUN_DIR ...]
Each run must use the same fold manifest. The output is read by
build_heart_table.py and keeps the exact per-fold results for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "native"
CONDITIONS = ("site_hidden", "site_shuffled", "site_true")
EXPECTED = {
    *(f"native-{family}-{size}" for family in ("original", "rg_z-fixed", "rg_z-curriculum")
      for size in ("small", "medium", "large")),
    "rf", "logreg", "hgb", "xgboost", "lightgbm", "catboost",
    "tabicl-v1", "tabicl-v2", "tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--checkpoint-file", action="append", default=[],
                        help="Checkpoint identifier=path, repeat for published weights")
    args = parser.parse_args()

    manifests = []
    rows = []
    summaries = []
    runs = []
    for run in args.run_dirs:
        manifests.append(json.loads((run / "fold_manifest.json").read_text()))
        run_rows = json.loads((run / "per_fold.json").read_text())
        rows.extend(run_rows)
        summaries.extend(json.loads((run / "summary.json").read_text()))
        provenance = json.loads((run / "provenance.json").read_text())
        if provenance["dataset_revision"] != "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c":
            raise ValueError(f"Unexpected dataset revision in {run}")
        if (provenance["shots"], provenance["repeats_per_site"], provenance["seed"]) != (20, 5, 0):
            raise ValueError(f"Unexpected protocol parameters in {run}")
        runs.append({"models": provenance["models"], "provenance": provenance})
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("Run fold manifests differ")
    if len(manifests[0]) != 15 or {item["target_site"] for item in manifests[0]} != {
        "cleveland", "hungary", "va_long_beach"
    }:
        raise ValueError("Expected five folds for each of three heart sites")

    found = {item["model"] for item in summaries}
    if found != EXPECTED or len(summaries) != len(EXPECTED):
        raise ValueError(f"Expected 20 unique models; missing={EXPECTED-found}, extra={found-EXPECTED}")
    fold_names = {item["fold"] for item in manifests[0]}
    keys = [(row["model"], row["fold"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)) or len(rows) != 20 * 15 * 3:
        raise ValueError("Duplicate or missing model-fold-condition rows")
    for row in rows:
        if row["status"] != "ok" or row["fold"] not in fold_names or row["target_site_shots"] != 20:
            raise ValueError(f"Invalid row: {row['model']} {row['fold']} {row['condition']}")
        if row["support_rows"] + row["query_rows"] != 797:
            raise ValueError("Support/query sizes do not partition the heart dataset")
    by_key = {(row["model"], row["fold"], row["condition"]): row for row in rows}
    for model in EXPECTED:
        for fold in fold_names:
            widths = [by_key[(model, fold, condition)]["features"] for condition in CONDITIONS]
            if widths[1] != widths[2] or widths[0] >= widths[1]:
                raise ValueError(f"Invalid feature widths for {model} {fold}: {widths}")
    for item in summaries:
        model = item["model"]
        for condition in CONDITIONS:
            if item[f"{condition}_folds"] != 15:
                raise ValueError(f"Incomplete result for {model} {condition}")
            mean = statistics.fmean(
                by_key[(model, fold, condition)]["excess_cross_entropy"]
                for fold in fold_names
            )
            if not math.isclose(mean, item[f"{condition}_excess_cross_entropy"], abs_tol=1e-12):
                raise ValueError(f"Summary mismatch for {model} {condition}")

    checkpoints = {}
    if args.native_root:
        for path in sorted(args.native_root.glob("*/seed-2402/final_checkpoint.pth")):
            checkpoints[f"native-{path.parent.parent.name}"] = {
                "sha256": digest(path), "bytes": path.stat().st_size
            }
    for spec in args.checkpoint_file:
        name, sep, raw_path = spec.partition("=")
        if not sep or name in checkpoints:
            raise ValueError(f"Invalid checkpoint specification: {spec}")
        path = Path(raw_path)
        checkpoints[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    if checkpoints and checkpoints.keys() != EXPECTED - {
        "rf", "logreg", "hgb", "xgboost", "lightgbm", "catboost"
    }:
        raise ValueError(f"Checkpoint hashes missing: {EXPECTED - set(checkpoints)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "heart_sites_adaptation"
    (stem.with_name(stem.name + "_summary.json")).write_text(
        json.dumps(sorted(summaries, key=lambda item: item["model"]), indent=2) + "\n"
    )
    (stem.with_name(stem.name + "_per_fold.json")).write_text(
        json.dumps(sorted(rows, key=lambda item: (item["model"], item["fold"], item["condition"])), indent=2) + "\n"
    )
    (stem.with_name(stem.name + "_fold_manifest.json")).write_text(
        json.dumps(manifests[0], indent=2) + "\n"
    )
    (stem.with_name(stem.name + "_provenance.json")).write_text(
        json.dumps({"runs": runs, "checkpoint_files": checkpoints}, indent=2) + "\n"
    )
    print("Archived 20 models, 15 matched folds, and three site-tag conditions")


if __name__ == "__main__":
    main()
