#!/usr/bin/env python3
"""Aggregate and plot hidden/shuffled/true results separately for r_z and g_z.

The evaluator's per-episode reports retain ``mechanism_mode``. This script
filters the matched multiclass multiregime bank, computes cell-stratified
paired intervals, and writes one plot per mechanism.
"""

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
DEFAULT_SUMMARY = ROOT / "paper/native/observed_regime_information_condition_by_mechanism_summary.json"
DEFAULT_FIGURE_ROOT = ROOT / "figures/native"

SIZE_ORDER = ("small", "medium", "large")
MODEL_ORDER = ("original", "rg_z-fixed", "rg_z-curriculum")
MODEL_LABELS = {
    "original": "Original single-regime",
    "rg_z-fixed": "Fixed-mixture",
    "rg_z-curriculum": "Curriculum",
}
MODEL_COLORS = {
    "original": "#4C566A",
    "rg_z-fixed": "#0072B2",
    "rg_z-curriculum": "#D55E00",
}
EXPECTED_MODEL_SOURCE_SHA256 = {
    "original": "5ed1a46b2939aa0c8db1e3766a27a9cae9db6e36ac0a8b165471df7bcc14339c",
    "rg_z-fixed": "32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34",
    "rg_z-curriculum": "32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34",
}
FAMILY_ORDER = ("persistent", "soft_gate")
MECHANISM_ORDER = ("r_z", "g_z")
CONDITIONS = ("hidden", "shuffled", "true")
PAIRS = (("shuffled", "hidden"), ("true", "hidden"), ("true", "shuffled"))
CONTRAST_ROWS = (
    ("shuffled", "hidden", "Cross-entropy change (nats)", "Shuffled − hidden"),
    ("true", "shuffled", "Cross-entropy change (nats)", "True − shuffled"),
    ("shuffled", "hidden", "Accuracy change (pp)", "Shuffled − hidden"),
    ("true", "shuffled", "Accuracy change (pp)", "True − shuffled"),
)


def _report_path(root: Path, model: str, size: str) -> Path:
    return root / f"{model}-{size}" / "seed-2402" / "test.json"


def _load_reports(root: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    report_metadata: dict[str, Any] = {}
    expected_per_condition = 5760

    for size in SIZE_ORDER:
        for model in MODEL_ORDER:
            key = (size, model)
            path = _report_path(root, model, size)
            report = json.loads(path.read_text())
            if "per_episode" not in report:
                raise ValueError(f"{path} does not contain per_episode rows")
            if report.get("episodes_per_condition") != expected_per_condition:
                raise ValueError(
                    f"{path} has episodes_per_condition={report.get('episodes_per_condition')}; "
                    f"expected {expected_per_condition}"
                )
            conditions = tuple(report.get("information_conditions", ()))
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"{path} has information conditions {conditions}, expected {CONDITIONS}")
            expected_source_hash = EXPECTED_MODEL_SOURCE_SHA256[model]
            if report.get("model_source_sha256") != expected_source_hash:
                raise ValueError(
                    f"{path} model source hash is {report.get('model_source_sha256')}, "
                    f"expected {expected_source_hash}"
                )
            probe_path = path.parent / "monitor_probe_seed2402.json"
            probe = json.loads(probe_path.read_text())
            validation = probe.get("hidden_reference_validation", {})
            if not validation.get("passed"):
                raise ValueError(f"{probe_path} does not record a passing hidden-reference validation")
            if probe.get("model_source_sha256") != expected_source_hash:
                raise ValueError(f"{probe_path} does not use the expected checkpoint source")

            rows = [
                row
                for row in report["per_episode"]
                if int(row["num_regimes"]) >= 2
                and int(row["num_classes"]) >= 3
                and str(row["rule_mode"]) == "multiregime"
                and str(row["mechanism_mode"]) in MECHANISM_ORDER
                and str(row["information"]) in CONDITIONS
            ]
            if len(rows) != expected_per_condition * len(CONDITIONS):
                raise ValueError(
                    f"{path} retained {len(rows)} rows after filtering; "
                    f"expected {expected_per_condition * len(CONDITIONS)}"
                )
            selected[key] = rows
            report_metadata[f"{model}-{size}"] = {
                "path": str(path),
                "source_checkpoint": report.get("source_checkpoint"),
                "source_step": report.get("source_step"),
                "bank_path": report.get("bank_path"),
                "model_source_sha256": report.get("model_source_sha256"),
                "hidden_reference_validation": report.get("hidden_reference_validation"),
                "probe_path": str(probe_path),
                "probe_hidden_reference_validation": validation,
                "episodes_per_condition": expected_per_condition,
            }

            by_stratum_condition: dict[tuple[str, str, str], set[int]] = defaultdict(set)
            for row in rows:
                stratum = (str(row["mechanism_mode"]), str(row["task_family"]), str(row["information"]))
                by_stratum_condition[stratum].add(int(row["episode_id"]))
            for mechanism in MECHANISM_ORDER:
                for family in FAMILY_ORDER:
                    sets = [by_stratum_condition[(mechanism, family, condition)] for condition in CONDITIONS]
                    if any(len(values) != 1440 for values in sets):
                        raise ValueError(
                            f"{path} has incomplete {mechanism}/{family} episode counts: "
                            f"{dict(zip(CONDITIONS, map(len, sets), strict=True))}"
                        )
                    if not (sets[0] == sets[1] == sets[2]):
                        raise ValueError(f"{path} has unmatched information-condition episode IDs in {mechanism}/{family}")

    bank_paths = {record["bank_path"] for record in report_metadata.values()}
    if len(bank_paths) != 1:
        raise ValueError(f"Reports do not use one matched bank: {sorted(str(path) for path in bank_paths)}")
    return selected, {"reports": report_metadata, "bank_path": next(iter(bank_paths))}


def _summarize(
    reports: dict[tuple[str, str], list[dict[str, Any]]], *, replicates: int, seed: int
) -> dict[str, Any]:
    means: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (size, model), rows in reports.items():
        for row in rows:
            key = (
                size,
                model,
                str(row["mechanism_mode"]),
                str(row["task_family"]),
                str(row["information"]),
            )
            means[key].append(row)

    mean_rows = []
    for key in sorted(means):
        size, model, mechanism, family, information = key
        group = means[key]
        mean_rows.append(
            {
                "size": size,
                "model": model,
                "mechanism": mechanism,
                "family": family,
                "information": information,
                "episodes": len(group),
                "cross_entropy": float(np.mean([float(row["query_cross_entropy"]) for row in group])),
                "accuracy": float(np.mean([float(row["query_accuracy"]) for row in group])),
            }
        )

    paired_by_episode: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (size, model), rows in reports.items():
        for row in rows:
            paired_by_episode[
                (
                    size,
                    model,
                    str(row["mechanism_mode"]),
                    int(row["episode_id"]),
                )
            ][str(row["information"])] = row

    paired = []
    for size in SIZE_ORDER:
        for model in MODEL_ORDER:
            for mechanism in MECHANISM_ORDER:
                for family in FAMILY_ORDER:
                    selected = {
                        key: conditions
                        for key, conditions in paired_by_episode.items()
                        if key[:3] == (size, model, mechanism)
                        and str(next(iter(conditions.values()))["task_family"]) == family
                    }
                    for information, reference in PAIRS:
                        by_cell: dict[int, list[tuple[float, float]]] = defaultdict(list)
                        for conditions in selected.values():
                            if information not in conditions or reference not in conditions:
                                continue
                            candidate_row, reference_row = conditions[information], conditions[reference]
                            by_cell[int(candidate_row["cell_id"])].append(
                                (
                                    float(candidate_row["query_cross_entropy"])
                                    - float(reference_row["query_cross_entropy"]),
                                    100
                                    * (
                                        float(candidate_row["query_accuracy"])
                                        - float(reference_row["query_accuracy"])
                                    ),
                                )
                            )
                        if not by_cell:
                            raise ValueError(f"No paired episodes for {size}/{model}/{mechanism}/{family}")
                        differences = [value for cell_rows in by_cell.values() for value in cell_rows]
                        values = np.asarray(differences, dtype=float)
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                (seed, SIZE_ORDER.index(size), MODEL_ORDER.index(model), MECHANISM_ORDER.index(mechanism),
                                 FAMILY_ORDER.index(family), PAIRS.index((information, reference)))
                            )
                        )
                        cells = [np.asarray(cell_rows, dtype=float) for _, cell_rows in sorted(by_cell.items())]
                        draws = np.zeros((replicates, 2), dtype=float)
                        for cell in cells:
                            sample_indices = rng.integers(0, len(cell), size=(replicates, len(cell)))
                            draws += cell[sample_indices].mean(axis=1)
                        draws /= len(cells)
                        paired.append(
                            {
                                "size": size,
                                "model": model,
                                "mechanism": mechanism,
                                "family": family,
                                "information": information,
                                "reference": reference,
                                "episodes": len(differences),
                                "cells": len(cells),
                                "cross_entropy_difference": float(values[:, 0].mean()),
                                "cross_entropy_difference_95_interval": [
                                    float(np.quantile(draws[:, 0], 0.025)),
                                    float(np.quantile(draws[:, 0], 0.975)),
                                ],
                                "accuracy_difference_pp": float(values[:, 1].mean()),
                                "accuracy_difference_95_interval_pp": [
                                    float(np.quantile(draws[:, 1], 0.025)),
                                    float(np.quantile(draws[:, 1], 0.975)),
                                ],
                            }
                        )
    return {
        "description": "Per-mechanism means and cell-stratified paired contrasts for hidden, shuffled, and true regime-tag inputs.",
        "filters": {
            "num_regimes": "at least 2",
            "num_classes": "at least 3",
            "rule_mode": "multiregime",
            "episodes_per_mechanism_and_routing_family": 1440,
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "scope": "finite-episode uncertainty only; does not estimate pretraining-seed uncertainty",
        },
        "rows": mean_rows,
        "paired_contrasts": paired,
    }


def _plot_mechanism(summary: dict[str, Any], mechanism: str, output: Path) -> None:
    index = {
        (
            row["size"],
            row["model"],
            row["mechanism"],
            row["family"],
            row["information"],
            row["reference"],
        ): row
        for row in summary["paired_contrasts"]
    }
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(13.2, 12.5), sharex=True, constrained_layout=False)
    family_offsets = {model: offset for model, offset in zip(MODEL_ORDER, (-0.17, 0.0, 0.17))}

    for row_idx, (information, reference, metric_label, contrast_label) in enumerate(CONTRAST_ROWS):
        metric = "cross_entropy_difference" if row_idx < 2 else "accuracy_difference_pp"
        interval_key = (
            "cross_entropy_difference_95_interval"
            if row_idx < 2
            else "accuracy_difference_95_interval_pp"
        )
        interval_bounds = [
            interval
            for row in summary["paired_contrasts"]
            if row["information"] == information and row["reference"] == reference
            for interval in [row[interval_key]]
        ]
        lower = min(interval[0] for interval in interval_bounds)
        upper = max(interval[1] for interval in interval_bounds)
        padding = max((upper - lower) * 0.08, 1e-4)
        ylim = (min(0.0, lower - padding), max(0.0, upper + padding))
        for size_col, size in enumerate(SIZE_ORDER):
            ax = axes[row_idx, size_col]
            ax.set_title(
                f"{size.title()} ({2 if size == 'small' else 4 if size == 'medium' else 6} layers)\n{contrast_label}",
                fontsize=10,
                pad=7,
            )

            for family_idx, family in enumerate(FAMILY_ORDER):
                for model in MODEL_ORDER:
                    record = index[(size, model, mechanism, family, information, reference)]
                    estimate = record[metric]
                    low, high = record[interval_key]
                    ax.errorbar(
                        family_idx + family_offsets[model],
                        estimate,
                        yerr=[[estimate - low], [high - estimate]],
                        fmt="o",
                        color=MODEL_COLORS[model],
                        markersize=5,
                        elinewidth=1.2,
                        capsize=2.5,
                        capthick=1.2,
                        linestyle="none",
                        zorder=3,
                    )

            ax.axhline(0, color="#555555", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
            ax.axvline(0.5, color="#e1e1e1", linewidth=0.8, zorder=0)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.42, 1.42)
            ax.tick_params(axis="y", labelsize=9)
            if size_col == 0:
                ax.set_ylabel(metric_label, fontsize=9.5, labelpad=9)
            if row_idx == len(CONTRAST_ROWS) - 1:
                ax.set_xticks((0, 1), ("Persistent", "Soft-gate"), fontsize=9)
            else:
                ax.set_xticks((0, 1), ("", ""))

    fig.suptitle(
        f"Paired regime-membership contrasts for {mechanism}\n"
        "Negative ΔCE / positive Δaccuracy favor the first-named condition",
        fontsize=14,
        y=0.985,
    )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[model], linestyle="none", markersize=6, label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "N=1,440 matched episodes per routing family and mechanism. Dots are paired mean differences; bars are 95% cell-stratified episode-bootstrap intervals, not training-seed uncertainty.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.83, bottom=0.095, left=0.15, right=0.985, hspace=0.48, wspace=0.28)
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

    reports, source_metadata = _load_reports(args.report_root)
    summary = _summarize(reports, replicates=args.bootstrap_replicates, seed=args.seed)
    summary.update(source_metadata)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    for mechanism in MECHANISM_ORDER:
        output = args.figure_root / f"regime_information_conditions_{mechanism}.png"
        _plot_mechanism(summary, mechanism, output)
        print(output)
    print(args.summary)


if __name__ == "__main__":
    main()
