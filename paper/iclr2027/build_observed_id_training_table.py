"""Build the exploratory observed-regime-ID appendix table from saved reports.

This script performs no inference or training. It validates the archived report
coverage and checkpoint configs, computes paired observed-minus-original
contrasts on the shared episode IDs, and writes a JSON summary and LaTeX table.
Run from the repository root with:

    python3 paper/iclr2027/build_observed_id_training_table.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORT_ROOT = REPO / "artifacts/cluster_sync/observed_id_training"
BASELINE_ROOT = REPO / "artifacts/cluster_sync/regime_information_checkpoint_compatible"
NATIVE_RUNS = REPO / "artifacts/cluster_sync/tfm_runs/native"
OBSERVED_CONFIGS = REPORT_ROOT / "_provenance"
OUTPUT_JSON = REPO / "paper/native/observed_id_training_comparison.json"
OUTPUT_TEX = HERE / "appendix_observed_id_table.tex"

SIZES = ("small", "medium")
MODELS = ("original", "rg_z-fixed", "rg_z-curriculum", "observed-ID")
CONDITIONS = ("hidden", "shuffled", "true")
LAYERS = {"small": 2, "medium": 4}
EPISODES_PER_CONDITION = 5760
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260926
METRIC_FIELDS = {"query_cross_entropy", "query_accuracy", "query_auc"}


def report_path(size: str, model: str) -> Path:
    if model == "observed-ID":
        return REPORT_ROOT / f"observed-{size}/seed-2402/test.json"
    return BASELINE_ROOT / f"{model}-{size}/seed-2402/test.json"


def config_path(size: str, model: str) -> Path:
    if model == "observed-ID":
        return OBSERVED_CONFIGS / f"observed-{size}-config.json"
    return NATIVE_RUNS / f"{model}-{size}/seed-2402/config.json"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def digest_episode_ids(ids: set[int]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_and_validate(size: str, model: str) -> tuple[dict, dict, dict[str, dict[int, dict]]]:
    report_file = report_path(size, model)
    config_file = config_path(size, model)
    report = load_json(report_file)
    config = load_json(config_file)

    if int(report.get("source_step", -1)) != 10_000:
        raise ValueError(f"{report_file}: expected step 10,000")
    if "/seed-2402/" not in str(report.get("source_checkpoint", "")):
        raise ValueError(f"{report_file}: checkpoint path does not identify seed 2402")
    if int(config.get("seed", -1)) != 2402 or int(config.get("max_steps", -1)) != 10_000:
        raise ValueError(f"{config_file}: expected seed 2402 and max_steps 10,000")
    if int(config.get("num_layers", -1)) != LAYERS[size]:
        raise ValueError(f"{config_file}: unexpected architecture for {size}")
    if model == "observed-ID":
        if config.get("v4_expose_regime_id") is not True:
            raise ValueError(f"{config_file}: training-time regime-ID exposure is not enabled")
    if report.get("information_conditions") != list(CONDITIONS):
        raise ValueError(f"{report_file}: expected hidden, shuffled, and true conditions")
    if report.get("episodes_per_condition") != EPISODES_PER_CONDITION:
        raise ValueError(f"{report_file}: expected {EPISODES_PER_CONDITION} episodes per condition")
    if report.get("multiclass_only") is not True or report.get("multiregime_only") is not True:
        raise ValueError(f"{report_file}: report is not restricted to multiclass multiregime episodes")
    if len(report.get("per_episode", [])) != EPISODES_PER_CONDITION * len(CONDITIONS):
        raise ValueError(f"{report_file}: unexpected per-episode row count")

    by_condition: dict[str, dict[int, dict]] = {condition: {} for condition in CONDITIONS}
    for row in report["per_episode"]:
        condition = row["information"]
        episode_id = int(row["episode_id"])
        if condition not in by_condition:
            raise ValueError(f"{report_file}: unexpected condition {condition!r}")
        if episode_id in by_condition[condition]:
            raise ValueError(f"{report_file}: duplicate episode {episode_id} in {condition}")
        if int(row["num_classes"]) < 3 or int(row["num_regimes"]) < 2 or row["rule_mode"] != "multiregime":
            raise ValueError(f"{report_file}: found an ineligible episode {episode_id}")
        by_condition[condition][episode_id] = row

    expected_ids: set[int] | None = None
    for condition, condition_rows in by_condition.items():
        ids = set(condition_rows)
        if len(ids) != EPISODES_PER_CONDITION:
            raise ValueError(f"{report_file}: {condition} has {len(ids)} unique episodes")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"{report_file}: condition episode IDs differ")

    return report, config, by_condition


def metadata(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in METRIC_FIELDS | {"information"}}


def cell_stratified_difference(candidate: dict[int, dict], reference: dict[int, dict], seed: int) -> dict:
    if set(candidate) != set(reference):
        raise ValueError("Paired reports do not have identical episode IDs")
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for episode_id in sorted(candidate):
        c = candidate[episode_id]
        r = reference[episode_id]
        if int(c["cell_id"]) != int(r["cell_id"]):
            raise ValueError(f"cell mismatch for episode {episode_id}")
        groups[int(c["cell_id"])].append(
            (
                float(c["query_cross_entropy"]) - float(r["query_cross_entropy"]),
                float(c["query_accuracy"]) - float(r["query_accuracy"]),
            )
        )
    cells = [np.asarray(values, dtype=float) for _, values in sorted(groups.items())]
    counts = {len(values) for values in cells}
    if len(cells) != 720 or counts != {8}:
        raise ValueError(f"expected 720 balanced cells of 8 episodes, got {len(cells)} cells and sizes {counts}")

    point = np.mean(np.stack([values.mean(axis=0) for values in cells]), axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAP_REPLICATES, 2), dtype=float)
    cell_values = np.stack(cells)
    for start in range(0, BOOTSTRAP_REPLICATES, 128):
        stop = min(start + 128, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, cell_values.shape[1], size=(stop - start, *cell_values.shape[:2]))
        sampled = np.take_along_axis(
            np.broadcast_to(cell_values, (stop - start, *cell_values.shape)),
            indices[..., None],
            axis=2,
        )
        draws[start:stop] = sampled.mean(axis=2).mean(axis=1)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "episodes": sum(len(values) for values in cells),
        "cells": len(cells),
        "cross_entropy_difference": float(point[0]),
        "cross_entropy_difference_95_interval": [float(x) for x in intervals[:, 0]],
        "accuracy_difference_pp": float(point[1] * 100),
        "accuracy_difference_95_interval_pp": [float(x * 100) for x in intervals[:, 1]],
    }


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_latex(summary: dict) -> None:
    table_rows = []
    labels = {
        "original": "Original",
        "rg_z-fixed": "Fixed mixture",
        "rg_z-curriculum": "Curriculum",
        "observed-ID": "Observed-ID training",
    }
    for size in SIZES:
        for model in MODELS:
            row = summary["scores"][size][model]
            ce = [f"{row[c]['cross_entropy']:.3f}" for c in CONDITIONS]
            acc = [f"{row[c]['accuracy'] * 100:.1f}" for c in CONDITIONS]
            if model == "observed-ID":
                delta = summary["paired_observed_minus_original_hidden"][size]
                ce_delta = delta["cross_entropy_difference"]
                ce_lo, ce_hi = delta["cross_entropy_difference_95_interval"]
                acc_delta = delta["accuracy_difference_pp"]
                acc_lo, acc_hi = delta["accuracy_difference_95_interval_pp"]
                ce_cell = f"{ce_delta:+.3f} [{ce_lo:+.3f}, {ce_hi:+.3f}]"
                acc_cell = f"{acc_delta:+.2f} [{acc_lo:+.2f}, {acc_hi:+.2f}]"
            else:
                ce_cell = acc_cell = "--"
            table_rows.append(
                " & ".join(
                    [size.capitalize(), labels[model], *ce, *acc, ce_cell, acc_cell]
                )
                + r"\\"
            )

    lines = [
        r"\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\caption{Exploratory observed-ID training comparison on 5,760 multiclass multiregime episodes. H = hidden, S = shuffled-tag, and T = true-tag evaluation inputs; the hidden condition is the primary task setting. Cross-model observed-minus-original differences are paired by episode under hidden inputs. Brackets give 95\% within-cell episode-bootstrap intervals (5,000 replicates); they quantify this fixed episode bank and not pretraining-seed variation. The saved model-source hashes differ across model families, so the cross-model comparison is descriptive.}",
        r"\label{tab:observed-id-training}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrll@{}}",
        r"\toprule",
        r"Size & Model & \multicolumn{3}{c}{Cross-entropy (nats)} & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{2}{c}{Observed-ID $-$ original, hidden [95\% interval]}\\",
        r" & & H & S & T & H & S & T & CE & Accuracy (pp)\\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    OUTPUT_TEX.write_text("\n".join(lines) + "\n")


def main() -> None:
    reports: dict[tuple[str, str], dict] = {}
    configs: dict[tuple[str, str], dict] = {}
    rows: dict[tuple[str, str], dict[str, dict[int, dict]]] = {}
    for size in SIZES:
        for model in MODELS:
            report, config, per_condition = read_and_validate(size, model)
            reports[(size, model)] = report
            configs[(size, model)] = config
            rows[(size, model)] = per_condition

    bank_paths = {report.get("bank_path") for report in reports.values()}
    if len(bank_paths) != 1:
        raise ValueError(f"reports do not share one bank path: {bank_paths}")

    reference_metadata: dict[int, dict] = {}
    reference_ids: set[int] | None = None
    for key, per_condition in rows.items():
        hidden_ids = set(per_condition["hidden"])
        if reference_ids is None:
            reference_ids = hidden_ids
        elif hidden_ids != reference_ids:
            raise ValueError(f"episode IDs differ for {key}")
        for condition in CONDITIONS:
            for episode_id, row in per_condition[condition].items():
                row_metadata = metadata(row)
                if episode_id not in reference_metadata:
                    reference_metadata[episode_id] = row_metadata
                elif reference_metadata[episode_id] != row_metadata:
                    raise ValueError(f"episode metadata mismatch for {key}, {condition}, episode {episode_id}")

    cell_counts = Counter(int(row["cell_id"]) for row in reference_metadata.values())
    if len(cell_counts) != 720 or set(cell_counts.values()) != {8}:
        raise ValueError("the shared reports must contain 720 cells with 8 episodes per cell")

    scores: dict[str, dict[str, dict]] = {}
    for size in SIZES:
        scores[size] = {}
        for model in MODELS:
            scores[size][model] = {}
            for condition in CONDITIONS:
                metric_rows = list(rows[(size, model)][condition].values())
                scores[size][model][condition] = {
                    "episodes": len(metric_rows),
                    "cross_entropy": float(np.mean([r["query_cross_entropy"] for r in metric_rows])),
                    "accuracy": float(np.mean([r["query_accuracy"] for r in metric_rows])),
                }

    paired = {
        size: cell_stratified_difference(
            rows[(size, "observed-ID")]["hidden"],
            rows[(size, "original")]["hidden"],
            BOOTSTRAP_SEED + index,
        )
        for index, size in enumerate(SIZES)
    }

    source_hashes = {
        f"{size}:{model}": reports[(size, model)].get("model_source_sha256")
        for size in SIZES
        for model in MODELS
    }
    summary = {
        "scope": "exploratory descriptive comparison from archived full-bank reports; no new inference or training",
        "checkpoint_seed": 2402,
        "checkpoint_step": 10000,
        "sizes": list(SIZES),
        "models": list(MODELS),
        "information_conditions": list(CONDITIONS),
        "primary_condition": "hidden",
        "episodes_per_condition": EPISODES_PER_CONDITION,
        "cells": len(cell_counts),
        "episodes_per_cell": 8,
        "bank_path": next(iter(bank_paths)),
        "bank_content_sha256": None,
        "bank_identity_note": "The saved reports share a bank path, episode IDs, and episode metadata; the bank file content hash is unavailable, so byte-level identity is unverified.",
        "episode_ids_sha256": digest_episode_ids(reference_ids or set()),
        "episode_ids_shared_across_all_reports": True,
        "episode_metadata_shared_across_all_reports": True,
        "model_source_sha256_by_size_and_model": source_hashes,
        "common_model_source_sha256": len(set(source_hashes.values())) == 1,
        "evaluator_source_sha256_recorded_in_reports": False,
        "paired_observed_minus_original_hidden": paired,
        "bootstrap": {
            "method": "paired episode differences; resample episodes with replacement within each factorial cell; average cell means equally",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "scope": "uncertainty over the fixed archived episode bank only; does not estimate training-seed variation",
        },
        "scores": scores,
        "report_files": {
            f"{size}:{model}": str(report_path(size, model).relative_to(REPO))
            for size in SIZES
            for model in MODELS
        },
        "report_file_sha256": {
            f"{size}:{model}": hashlib.sha256(report_path(size, model).read_bytes()).hexdigest()
            for size in SIZES
            for model in MODELS
        },
        "configs": {
            f"{size}:{model}": str(config_path(size, model).relative_to(REPO))
            for size in SIZES
            for model in MODELS
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    write_latex(summary)
    print(f"Wrote {OUTPUT_JSON.relative_to(REPO)}")
    print(f"Wrote {OUTPUT_TEX.relative_to(REPO)}")
    for size in SIZES:
        delta = paired[size]
        print(
            f"{size}: observed-ID − original, hidden: "
            f"CE {delta['cross_entropy_difference']:+.5f} "
            f"[{delta['cross_entropy_difference_95_interval'][0]:+.5f}, "
            f"{delta['cross_entropy_difference_95_interval'][1]:+.5f}], "
            f"accuracy {delta['accuracy_difference_pp']:+.3f} pp "
            f"[{delta['accuracy_difference_95_interval_pp'][0]:+.3f}, "
            f"{delta['accuracy_difference_95_interval_pp'][1]:+.3f}]"
        )
    print(f"Shared report episode IDs: {len(reference_ids or set())}; source hashes common: {summary['common_model_source_sha256']}")


if __name__ == "__main__":
    main()
