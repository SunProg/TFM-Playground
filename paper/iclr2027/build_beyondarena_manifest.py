"""Archive the 36 reported BeyondArena tasks and their official fold IDs.

The task list comes from the native result tables. Data Foundry 0.0.5 supplies
the task-to-container mapping. Metadata is downloaded from an immutable
BeyondArena revision. The older 32-task evaluation is used only as a check;
it is not substituted for the 36-task result manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from data_foundry.collections import BEYOND_ARENA
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_native_beyondarena import load_tables  # noqa: E402

REVISION = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"
REPO = "TabArena/BeyondArena"
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "paper/native/native_prior_results.md"
OLD_MANIFEST = ROOT / "results/beyondarena-v4/all-families-37325808/task_manifest.csv"
OLD_FOLDS = ROOT / "results/beyondarena-v4/all-families-37325808/fold_metrics.csv"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_json(prefix: str, filename: str) -> tuple[dict, str]:
    path = Path(
        hf_hub_download(
            repo_id=REPO,
            filename=f"{prefix}/{filename}",
            revision=REVISION,
            repo_type="dataset",
        )
    )
    data = path.read_bytes()
    return json.loads(data), sha256_bytes(data)


def archive_task(item: tuple[str, dict, object, dict]) -> tuple[dict, list[dict]]:
    short, reported, entry, prior = item
    dataset_rows = int(prior["rows"])
    prefix = entry.relative_path.as_posix()
    container, _ = download_json(prefix, "container_metadata.json")
    task, _ = download_json(prefix, "task_metadata.predictive-ml-task-mold-v1.json")
    experiment, split_file_sha256 = download_json(
        prefix, "experiment_metadata.predictive-ml-splits-mold-v1.json"
    )
    if container["uuid"] != str(entry.uuid):
        raise ValueError(f"Container UUID differs for {short}")
    regime = "T" if task["time_on"] else "G" if task["group_on"] else "IID"
    if regime != reported["regime"]:
        raise ValueError(f"Split regime differs for {short}: {regime}")
    expected_type = "binary_classification" if reported["target"] == "bin" else "multiclass_classification"
    if task["problem_type"] != expected_type:
        raise ValueError(f"Problem type differs for {short}: {task['problem_type']}")
    if task["target_column_name"] != prior["target_column"]:
        raise ValueError(f"Target column differs for {short}")

    folds = []
    for repeat, repeat_folds in sorted(experiment["splits"].items(), key=lambda x: int(x[0])):
        for fold, (train, test) in sorted(repeat_folds.items(), key=lambda x: int(x[0])):
            if not train or not test:
                raise ValueError(f"Empty official fold in {short}: {repeat}/{fold}")
            if set(train) & set(test):
                raise ValueError(f"Overlapping train/test indices in {short}: {repeat}/{fold}")
            folds.append(
                {
                    "task_name": entry.unique_name,
                    "task_uuid": str(entry.uuid),
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_index_sha256": sha256_bytes(json.dumps(train, separators=(",", ":")).encode()),
                    "test_index_sha256": sha256_bytes(json.dumps(test, separators=(",", ":")).encode()),
                }
            )
    if any(r["train_rows"] + r["test_rows"] > dataset_rows for r in folds):
        raise ValueError(f"Official fold exceeds dataset row count: {short}")
    if reported["train_rows"] not in {r["train_rows"] for r in folds}:
        raise ValueError(f"Reported training size not found in official folds: {short}")
    summary = {
        "reported_task_name": short,
        "task_name": entry.unique_name,
        "task_uuid": str(entry.uuid),
        "container_checksum": container["checksum"],
        "problem_type": task["problem_type"],
        "split_regime": regime,
        "dataset_rows": dataset_rows,
        "raw_features": int(prior["raw_features"]),
        "classes": int(prior["classes"]),
        "target_column": task["target_column_name"],
        "reported_train_rows": reported["train_rows"],
        "reported_features": reported["features"],
        "official_fold_count": len(folds),
        "official_split_file_sha256": split_file_sha256,
    }
    return summary, folds


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_against_old_run(tasks: list[dict], folds: list[dict]) -> int:
    old = pd.read_csv(OLD_MANIFEST).set_index("task_name")
    old_folds = pd.read_csv(
        OLD_FOLDS,
        usecols=["task_name", "task_uuid", "repeat", "fold", "train_rows", "test_rows"],
    ).drop_duplicates()
    old_folds = old_folds.set_index(["task_name", "repeat", "fold"])
    checked = 0
    for task in tasks:
        name = task["task_name"]
        if name not in old.index:
            continue
        prior = old.loc[name]
        if str(prior["uuid"]) != task["task_uuid"] or str(prior["checksum"]) != task["container_checksum"]:
            raise ValueError(f"Older manifest container differs: {name}")
        if name not in old_folds.index.get_level_values("task_name"):
            continue
        official = [r for r in folds if r["task_name"] == name]
        saved = old_folds.loc[name]
        if len(saved) != len(official):
            raise ValueError(f"Older fold count differs: {name}")
        for row in official:
            key = (row["repeat"], row["fold"])
            if key not in saved.index:
                raise ValueError(f"Older fold ID differs: {name}/{key}")
            prior_fold = saved.loc[key]
            if (int(prior_fold["train_rows"]), int(prior_fold["test_rows"])) != (
                row["train_rows"], row["test_rows"]
            ):
                raise ValueError(f"Older fold sizes differ: {name}/{key}")
        checked += 1
    return checked


def main() -> None:
    if version("data-foundry") != "0.0.5":
        raise RuntimeError("This archive is defined against data-foundry 0.0.5")
    reported = load_tables(RESULTS)
    if len(reported) != 36:
        raise ValueError(f"Expected 36 reported tasks, found {len(reported)}")
    entries = list(BEYOND_ARENA.entries)
    prior_manifest = pd.read_csv(OLD_MANIFEST).set_index("task_name")
    eligible = (
        prior_manifest["problem_type"].astype(str).str.contains("classification", case=False)
        & prior_manifest["rows"].between(1, 11000)
        & prior_manifest["raw_features"].between(1, 30)
        & prior_manifest["classes"].between(2, 5)
        & prior_manifest["text_features"].isna()
        & prior_manifest["high_cardinality_features"].isna()
    )
    selected_names = set(prior_manifest.index[eligible])
    if len(selected_names) != 36:
        raise ValueError(f"Recorded eligibility filter selects {len(selected_names)} tasks, not 36")
    work = []
    for short, details in reported.items():
        matches = [entry for entry in entries if str(entry.unique_name).startswith(short)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous reported task name: {short} ({len(matches)} matches)")
        if matches[0].unique_name not in prior_manifest.index:
            raise ValueError(f"No recorded dataset row count for {short}")
        if matches[0].unique_name not in selected_names:
            raise ValueError(f"Reported task does not pass recorded eligibility filter: {short}")
        work.append((short, details, matches[0], prior_manifest.loc[matches[0].unique_name].to_dict()))
    if {entry.unique_name for _, _, entry, _ in work} != selected_names:
        raise ValueError("Reported task list differs from the 11,000-row eligible set")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(archive_task, work))
    tasks = [task for task, _ in results]
    folds = [fold for _, task_folds in results for fold in task_folds]
    checked = verify_against_old_run(tasks, folds)
    if checked != 36 - 4:
        raise ValueError(f"Expected 32 cross-checked tasks, found {checked}")
    write_csv(HERE / "beyondarena_36_tasks.csv", tasks)
    write_csv(HERE / "beyondarena_36_official_folds.csv", folds)
    print(f"Archived {len(tasks)} tasks and {len(folds)} official fold IDs at {REVISION}")
    print(f"Cross-checked {checked} task containers and fold dimensions against older saved evaluation")


if __name__ == "__main__":
    main()
