#!/usr/bin/env python3
"""Plot fixed-mixture/curriculum comparisons separately for r_z and g_z."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "artifacts/cluster_sync/regime_information_by_mechanism"
DEFAULT_SUMMARY = ROOT / "paper/native/observed_regime_information_model_comparison_by_mechanism_summary.json"
DEFAULT_FIGURE_ROOT = ROOT / "figures/native"

SIZE_ORDER = ("small", "medium", "large")
MODEL_ORDER = ("original", "rg_z-fixed", "rg_z-curriculum")
MODEL_CANDIDATES = ("rg_z-fixed", "rg_z-curriculum")
MODEL_LABELS = {"rg_z-fixed": "Fixed-mixture", "rg_z-curriculum": "Curriculum"}
MODEL_COLORS = {"rg_z-fixed": "#0072B2", "rg_z-curriculum": "#D55E00"}
MODEL_LINESTYLES = {"rg_z-fixed": "-", "rg_z-curriculum": "--"}
MODEL_MARKERS = {"rg_z-fixed": "o", "rg_z-curriculum": "s"}
MECHANISM_ORDER = ("r_z", "g_z")
FAMILY_ORDER = ("persistent", "soft_gate")
FAMILY_LABELS = {"persistent": "Persistent", "soft_gate": "Soft-gate"}
CONDITIONS = ("hidden", "shuffled", "true")
CONDITION_LABELS = ("Hidden\n(no tag)", "Shuffled\n(mismatched tag)", "True\n(aligned tag)")
MODEL_GAP_ROWS = (
    ("persistent", "cross_entropy", "Persistent · cross-entropy", "Cross-entropy difference (nats)"),
    ("soft_gate", "cross_entropy", "Soft-gate · cross-entropy", "Cross-entropy difference (nats)"),
    ("persistent", "accuracy", "Persistent · accuracy", "Accuracy difference (pp)"),
    ("soft_gate", "accuracy", "Soft-gate · accuracy", "Accuracy difference (pp)"),
)
EXPECTED_MODEL_SOURCE_SHA256 = {
    "original": "5ed1a46b2939aa0c8db1e3766a27a9cae9db6e36ac0a8b165471df7bcc14339c",
    "rg_z-fixed": "32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34",
    "rg_z-curriculum": "32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34",
}


def _report_path(root: Path, model: str, size: str) -> Path:
    return root / f"{model}-{size}" / "seed-2402" / "test.json"


def _load_reports(report_root: Path) -> tuple[dict[tuple[str, str], dict], dict[str, Any]]:
    reports: dict[tuple[str, str], dict] = {}
    metadata: dict[str, Any] = {}
    bank_paths: set[str] = set()
    expected_per_stratum = 1440
    expected_cells_per_stratum = 180

    for size in SIZE_ORDER:
        for model in MODEL_ORDER:
            path = _report_path(report_root, model, size)
            report = json.loads(path.read_text())
            if report.get("episodes_per_condition") != 5760:
                raise ValueError(f"{path} does not contain 5,760 episodes per information condition")
            if tuple(report.get("information_conditions", ())) != CONDITIONS:
                raise ValueError(f"{path} has unexpected information conditions")
            expected_hash = EXPECTED_MODEL_SOURCE_SHA256[model]
            if report.get("model_source_sha256") != expected_hash:
                raise ValueError(f"{path} has an unexpected model-source hash")

            probe_path = path.parent / "monitor_probe_seed2402.json"
            probe = json.loads(probe_path.read_text())
            validation = probe.get("hidden_reference_validation", {})
            if not validation.get("passed"):
                raise ValueError(f"{probe_path} does not record a passing hidden-reference validation")
            if probe.get("model_source_sha256") != expected_hash:
                raise ValueError(f"{probe_path} has an unexpected model-source hash")

            bank_paths.add(str(report.get("bank_path")))
            metadata[f"{model}-{size}"] = {
                "path": str(path),
                "source_checkpoint": report.get("source_checkpoint"),
                "source_step": report.get("source_step"),
                "bank_path": report.get("bank_path"),
                "model_source_sha256": report.get("model_source_sha256"),
                "probe_path": str(probe_path),
                "probe_hidden_reference_validation": validation,
            }

            selected: dict[tuple[str, str, str, int, int], dict] = {}
            for row in report["per_episode"]:
                if (
                    str(row["task_family"]) not in FAMILY_ORDER
                    or str(row["information"]) not in CONDITIONS
                    or str(row["mechanism_mode"]) not in MECHANISM_ORDER
                    or str(row["rule_mode"]) != "multiregime"
                    or int(row["num_regimes"]) < 2
                    or int(row["num_classes"]) < 3
                ):
                    continue
                key = (
                    str(row["mechanism_mode"]),
                    str(row["task_family"]),
                    str(row["information"]),
                    int(row["cell_id"]),
                    int(row["episode_id"]),
                )
                if key in selected:
                    raise ValueError(f"Duplicate episode-condition row in {path}: {key}")
                selected[key] = row

            for mechanism in MECHANISM_ORDER:
                for family in FAMILY_ORDER:
                    condition_keys: list[set[tuple[int, int]]] = []
                    for condition in CONDITIONS:
                        stratum = [
                            (key[3], key[4])
                            for key in selected
                            if key[:3] == (mechanism, family, condition)
                        ]
                        cells = {cell_id for cell_id, _ in stratum}
                        if len(stratum) != expected_per_stratum or len(cells) != expected_cells_per_stratum:
                            raise ValueError(
                                f"{path} has {len(stratum)} {mechanism}/{family}/{condition} rows "
                                f"across {len(cells)} cells"
                            )
                        condition_keys.append(set(stratum))
                    if not (condition_keys[0] == condition_keys[1] == condition_keys[2]):
                        raise ValueError(f"{path} has unmatched information-condition episode IDs in {mechanism}/{family}")

            reports[(size, model)] = selected

    if len(bank_paths) != 1:
        raise ValueError(f"Reports do not use one matched bank: {sorted(bank_paths)}")
    return reports, {"reports": metadata, "bank_path": next(iter(bank_paths))}


def _summarize(
    reports: dict[tuple[str, str], dict], *, replicates: int, seed: int
) -> dict[str, Any]:
    summary_rows = []
    for size in SIZE_ORDER:
        baseline = reports[(size, "original")]
        for model_index, model in enumerate(MODEL_CANDIDATES):
            candidate = reports[(size, model)]
            if baseline.keys() != candidate.keys():
                raise ValueError(f"{size}/{model} does not use the same mechanism/family/condition episodes as original")

            for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
                for family_index, family in enumerate(FAMILY_ORDER):
                    for condition_index, condition in enumerate(CONDITIONS):
                        by_cell: dict[int, list[tuple[float, float]]] = defaultdict(list)
                        for key, candidate_row in candidate.items():
                            if key[:3] != (mechanism, family, condition):
                                continue
                            baseline_row = baseline[key]
                            if candidate_row["episode_seed"] != baseline_row["episode_seed"]:
                                raise ValueError(f"Mismatched episode seed for {size}/{model}/{key}")
                            by_cell[key[3]].append(
                                (
                                    float(candidate_row["query_cross_entropy"])
                                    - float(baseline_row["query_cross_entropy"]),
                                    100.0
                                    * (
                                        float(candidate_row["query_accuracy"])
                                        - float(baseline_row["query_accuracy"])
                                    ),
                                )
                            )

                        cells = [np.asarray(rows, dtype=float) for _, rows in sorted(by_cell.items())]
                        if len(cells) != 180:
                            raise ValueError(f"{size}/{model}/{mechanism}/{family}/{condition} has {len(cells)} cells")
                        differences = np.concatenate(cells, axis=0)
                        estimates = differences.mean(axis=0)
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                (
                                    seed,
                                    SIZE_ORDER.index(size),
                                    model_index,
                                    mechanism_index,
                                    family_index,
                                    condition_index,
                                )
                            )
                        )
                        draws = np.zeros((replicates, 2), dtype=float)
                        for cell in cells:
                            sample_indices = rng.integers(0, len(cell), size=(replicates, len(cell)))
                            draws += cell[sample_indices].mean(axis=1)
                        draws /= len(cells)
                        summary_rows.append(
                            {
                                "size": size,
                                "model": model,
                                "mechanism": mechanism,
                                "family": family,
                                "condition": condition,
                                "episodes": len(differences),
                                "cells": len(cells),
                                "cross_entropy_difference": float(estimates[0]),
                                "cross_entropy_difference_95_interval": [
                                    float(np.quantile(draws[:, 0], 0.025)),
                                    float(np.quantile(draws[:, 0], 0.975)),
                                ],
                                "accuracy_difference_pp": float(estimates[1]),
                                "accuracy_difference_95_interval_pp": [
                                    float(np.quantile(draws[:, 1], 0.025)),
                                    float(np.quantile(draws[:, 1], 0.975)),
                                ],
                            }
                        )

    return {
        "description": "Candidate-minus-original episode-paired differences for hidden, shuffled, and true regime-tag inputs, stratified by regime-generating mechanism.",
        "filters": {
            "num_regimes": "at least 2",
            "num_classes": "at least 3",
            "rule_mode": "multiregime",
            "episodes_per_mechanism_routing_family_condition": 1440,
            "cells_per_mechanism_routing_family_condition": 180,
        },
        "bootstrap": {
            "method": "cell-stratified episode bootstrap; resample episodes within each cell and average cells equally",
            "replicates": replicates,
            "seed": seed,
            "scope": "finite evaluation-bank episode variation for these frozen checkpoints; not pretraining-seed uncertainty",
        },
        "rows": summary_rows,
    }


def _plot_mechanism(summary: dict[str, Any], mechanism: str, output: Path) -> None:
    index = {
        (row["size"], row["model"], row["family"], row["condition"]): row
        for row in summary["rows"]
        if row["mechanism"] == mechanism
    }
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 12.5), sharex=True, constrained_layout=False)
    x = list(range(len(CONDITIONS)))
    layer_counts = {"small": 2, "medium": 4, "large": 6}

    for row_idx, (family, metric, panel_title, axis_label) in enumerate(MODEL_GAP_ROWS):
        metric_key = "accuracy_difference_pp" if metric == "accuracy" else "cross_entropy_difference"
        interval_key = (
            "accuracy_difference_95_interval_pp"
            if metric == "accuracy"
            else "cross_entropy_difference_95_interval"
        )
        for size_col, size in enumerate(SIZE_ORDER):
            ax = axes[row_idx, size_col]
            ax.set_title(f"{size.title()} ({layer_counts[size]} layers)\n{panel_title}", fontsize=10, pad=7)
            intervals = [
                row[interval_key]
                for row in summary["rows"]
                if row["size"] == size and row["family"] == family
            ]
            low = min(interval[0] for interval in intervals)
            high = max(interval[1] for interval in intervals)
            pad = max((high - low) * 0.14, 1e-4 if metric == "cross_entropy" else 0.01)
            ax.set_ylim(min(0.0, low - pad), max(0.0, high + pad))

            for model in MODEL_CANDIDATES:
                records = [index[(size, model, family, condition)] for condition in CONDITIONS]
                estimates = np.asarray([record[metric_key] for record in records], dtype=float)
                bounds = np.asarray([record[interval_key] for record in records], dtype=float)
                ax.errorbar(
                    x,
                    estimates,
                    yerr=np.vstack((estimates - bounds[:, 0], bounds[:, 1] - estimates)),
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    markerfacecolor=("white" if model == "rg_z-curriculum" else MODEL_COLORS[model]),
                    markeredgecolor=MODEL_COLORS[model],
                    markersize=5.2,
                    linewidth=1.7,
                    linestyle=MODEL_LINESTYLES[model],
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=1.0,
                    zorder=3 if model == "rg_z-fixed" else 4,
                    label=MODEL_LABELS[model],
                )

            ax.axhline(0, color="#555555", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
            ax.set_xlim(-0.35, 2.35)
            ax.set_xticks(x, CONDITION_LABELS if row_idx == len(MODEL_GAP_ROWS) - 1 else ("", "", ""), fontsize=8.5)
            ax.tick_params(axis="y", labelsize=9)
            if size_col == 0:
                ax.set_ylabel(axis_label, fontsize=9.5, labelpad=9)

    fig.suptitle(
        f"Predictive performance relative to the single-regime baseline · {mechanism}\n"
        "Candidate minus original for hidden, shuffled, and true membership inputs",
        fontsize=14,
        y=0.985,
    )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model],
            marker=MODEL_MARKERS[model],
            markerfacecolor=("white" if model == "rg_z-curriculum" else MODEL_COLORS[model]),
            markeredgecolor=MODEL_COLORS[model], markersize=5, linewidth=1.8,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_CANDIDATES
    ]
    legend_handles.append(Line2D([0], [0], color="#555555", linestyle=(0, (3, 2)), linewidth=0.9,
                                 label="Original baseline (zero)"))
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.048,
        "Whiskers are 95% paired, cell-stratified episode-bootstrap intervals (5,000 replicates; 1,440 episodes per routing family and mechanism).",
        ha="center",
        fontsize=8.4,
        color="#444444",
    )
    fig.text(
        0.5,
        0.023,
        "Intervals capture evaluation-episode variation, not training-seed uncertainty. Matching y-scales are used for corresponding panels across mechanisms.",
        ha="center",
        fontsize=8.4,
        color="#444444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.145, left=0.12, right=0.985, hspace=0.48, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    reports, metadata = _load_reports(args.report_root)
    summary = _summarize(reports, replicates=args.bootstrap_replicates, seed=args.seed)
    summary.update(metadata)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    for mechanism in MECHANISM_ORDER:
        output = args.figure_root / f"regime_information_prediction_comparison_{mechanism}.png"
        _plot_mechanism(summary, mechanism, output)
        print(output)
    print(args.summary)


if __name__ == "__main__":
    main()
