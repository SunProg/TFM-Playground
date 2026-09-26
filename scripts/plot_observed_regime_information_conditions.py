#!/usr/bin/env python3
"""Plot hidden/shuffled/true regime-tag results from compact full-bank summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "paper/native/observed_regime_information_condition_summary.json"
DEFAULT_OUTPUT = ROOT / "figures/native/regime_information_prediction_comparison_v2.png"
DEFAULT_REPORT_ROOT = ROOT / "artifacts/cluster_sync/regime_information_checkpoint_compatible"
DEFAULT_MODEL_COMPARISON_SUMMARY = ROOT / "paper/native/observed_regime_information_model_comparison_summary.json"
DEFAULT_INPUT_EFFECT_OUTPUT = ROOT / "figures/native/regime_information_input_effects.png"
DEFAULT_MEANS_OUTPUT = ROOT / "figures/native/regime_information_conditions_means.png"

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
FAMILY_ORDER = ("persistent", "soft_gate")
FAMILY_LABELS = {"persistent": "Persistent", "soft_gate": "Soft-gate"}
CONDITIONS = ("hidden", "shuffled", "true")
CONDITION_LABELS = ("Hidden\n(no tag)", "Shuffled\n(mismatched tag)", "True\n(aligned tag)")
MODEL_CANDIDATES = ("rg_z-fixed", "rg_z-curriculum")
MODEL_LINESTYLES = {"rg_z-fixed": "-", "rg_z-curriculum": "--"}
MODEL_MARKERS = {"rg_z-fixed": "o", "rg_z-curriculum": "s"}
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
CONTRAST_ROWS = (
    ("cross_entropy_difference", "shuffled_minus_hidden", "Cross-entropy change (nats)", "Shuffled − hidden", (-0.0025, 0.0065)),
    ("cross_entropy_difference", "true_minus_shuffled", "Cross-entropy change (nats)", "True − shuffled", (-0.19, 0.01)),
    ("accuracy_difference_pp", "shuffled_minus_hidden", "Accuracy change (pp)", "Shuffled − hidden", (-0.45, 0.06)),
    ("accuracy_difference_pp", "true_minus_shuffled", "Accuracy change (pp)", "True − shuffled", (0.0, 12.5)),
)


def summarize_paired_model_differences(
    report_root: Path, *, replicates: int = 5000, seed: int = 20260925
) -> dict:
    """Bootstrap candidate-minus-original differences on matched episodes."""
    reports: dict[tuple[str, str], dict[tuple[str, str, int, int], dict]] = {}
    report_metadata = {}
    bank_paths = set()
    expected_per_family_condition = 2880

    for size in SIZE_ORDER:
        for model in MODEL_ORDER:
            report_path = report_root / f"{model}-{size}" / "seed-2402" / "test.json"
            report = json.loads(report_path.read_text())
            if report.get("episodes_per_condition") != 5760:
                raise ValueError(f"{report_path} does not contain the expected 5,760 episodes per condition")
            if report.get("model_source_sha256") != EXPECTED_MODEL_SOURCE_SHA256[model]:
                raise ValueError(f"{report_path} has an unexpected model-source hash")
            bank_paths.add(report.get("bank_path"))
            report_metadata[f"{model}-{size}"] = {
                "source_checkpoint": report.get("source_checkpoint"),
                "source_step": report.get("source_step"),
                "bank_path": report.get("bank_path"),
                "model_source_sha256": report.get("model_source_sha256"),
            }

            selected: dict[tuple[str, str, int, int], dict] = {}
            for row in report["per_episode"]:
                if (
                    row["task_family"] not in FAMILY_ORDER
                    or row["information"] not in CONDITIONS
                    or row["rule_mode"] != "multiregime"
                    or int(row["num_regimes"]) < 2
                    or int(row["num_classes"]) < 3
                ):
                    continue
                key = (
                    str(row["task_family"]),
                    str(row["information"]),
                    int(row["cell_id"]),
                    int(row["episode_id"]),
                )
                if key in selected:
                    raise ValueError(f"Duplicate episode-condition row in {report_path}: {key}")
                selected[key] = row

            for family in FAMILY_ORDER:
                for condition in CONDITIONS:
                    family_rows = [
                        row for (row_family, row_condition, _, _), row in selected.items()
                        if row_family == family and row_condition == condition
                    ]
                    cells = {int(row["cell_id"]) for row in family_rows}
                    if len(family_rows) != expected_per_family_condition or len(cells) != 360:
                        raise ValueError(
                            f"{report_path} has {len(family_rows)} {family}/{condition} rows in {len(cells)} cells"
                        )
            reports[(size, model)] = selected

    if len(bank_paths) != 1:
        raise ValueError(f"Reports do not use the same evaluation bank: {sorted(str(path) for path in bank_paths)}")

    result_rows = []
    for size in SIZE_ORDER:
        baseline_rows = reports[(size, "original")]
        for model_index, model in enumerate(MODEL_CANDIDATES):
            candidate_rows = reports[(size, model)]
            if baseline_rows.keys() != candidate_rows.keys():
                raise ValueError(f"{size}/{model} does not have the same matched episode-condition keys as original")

            for family_index, family in enumerate(FAMILY_ORDER):
                for condition_index, condition in enumerate(CONDITIONS):
                    by_cell: dict[int, list[tuple[float, float]]] = defaultdict(list)
                    for key, candidate in candidate_rows.items():
                        if key[0] != family or key[1] != condition:
                            continue
                        baseline = baseline_rows[key]
                        if candidate["episode_seed"] != baseline["episode_seed"]:
                            raise ValueError(f"Mismatched episode seed for {size}/{model}/{family}/{condition}/{key}")
                        by_cell[key[2]].append(
                            (
                                float(candidate["query_cross_entropy"])
                                - float(baseline["query_cross_entropy"]),
                                100.0
                                * (
                                    float(candidate["query_accuracy"])
                                    - float(baseline["query_accuracy"])
                                ),
                            )
                        )

                    cells = [np.asarray(by_cell[cell_id], dtype=float) for cell_id in sorted(by_cell)]
                    estimates = np.mean(np.concatenate(cells, axis=0), axis=0)
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            (seed, SIZE_ORDER.index(size), model_index, family_index, condition_index)
                        )
                    )
                    draws = np.zeros((replicates, 2), dtype=float)
                    for cell in cells:
                        sample_indices = rng.integers(0, len(cell), size=(replicates, len(cell)))
                        draws += cell[sample_indices].mean(axis=1)
                    draws /= len(cells)

                    result_rows.append(
                        {
                            "size": size,
                            "model": model,
                            "family": family,
                            "condition": condition,
                            "episodes": sum(len(cell) for cell in cells),
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

    try:
        report_root_reference = str(report_root.resolve().relative_to(ROOT))
    except ValueError:
        report_root_reference = str(report_root.resolve())
    return {
        "description": "Candidate-minus-original episode-paired differences for hidden, shuffled, and true regime-tag inputs.",
        "bootstrap": {
            "method": "5,000 cell-stratified episode-bootstrap replicates; resample episodes within each cell and average cells equally",
            "replicates": replicates,
            "seed": seed,
            "scope": "finite evaluation-bank episode variation for these frozen checkpoints; not pretraining-seed uncertainty",
        },
        "report_root": report_root_reference,
        "reports": report_metadata,
        "rows": result_rows,
    }


def write_summary(summary: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")


def plot_absolute_means(rows: list[dict], output: Path) -> None:
    index = {(row["size"], row["model"], row["family"]): row for row in rows}
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(13.2, 12.2),
        sharex=True,
        constrained_layout=False,
    )
    x = range(len(CONDITIONS))

    for size_col, size in enumerate(SIZE_ORDER):
        for family_row, family in enumerate(FAMILY_ORDER):
            row_index = family_row
            ax_ce = axes[row_index, size_col]
            ax_acc = axes[row_index + 2, size_col]
            for model in MODEL_ORDER:
                record = index[(size, model, family)]
                ce = [record["metrics"][condition]["cross_entropy"] for condition in CONDITIONS]
                acc = [100 * record["metrics"][condition]["accuracy"] for condition in CONDITIONS]
                style = dict(
                    color=MODEL_COLORS[model],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=MODEL_LABELS[model],
                )
                ax_ce.plot(x, ce, **style)
                ax_acc.plot(x, acc, **style)

            ax_ce.set_title(f"{FAMILY_LABELS[family]} · {size.title()} ({2 if size == 'small' else 4 if size == 'medium' else 6} layers)")
            ax_acc.set_title(f"{FAMILY_LABELS[family]} · {size.title()} ({2 if size == 'small' else 4 if size == 'medium' else 6} layers)")
            ax_ce.grid(axis="y", color="#d9d9d9", linewidth=0.7)
            ax_acc.grid(axis="y", color="#d9d9d9", linewidth=0.7)
            ax_acc.set_xticks(list(x), CONDITION_LABELS)

    for ax in axes[0, :]:
        ax.set_ylabel("Cross-entropy (nats)")
    for ax in axes[1, :]:
        ax.set_ylabel("Cross-entropy (nats)")
    for ax in axes[2, :]:
        ax.set_ylabel("Accuracy (%)")
    for ax in axes[3, :]:
        ax.set_ylabel("Accuracy (%)")

    fig.suptitle(
        "Prediction by regime-membership input condition\n"
        "Multiclass, multiregime evaluation; 2,880 matched episodes per routing family",
        fontsize=15,
        y=0.985,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.012,
        "Points are per-episode means on matched episodes. Paired bootstrap intervals for true−shuffled and shuffled−hidden are in the source summary; they quantify finite-bank, not training-seed, variation.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.42, wspace=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def plot_paired_differences(rows: list[dict], output: Path) -> None:
    index = {(row["size"], row["model"], row["family"]): row for row in rows}
    fig, axes = plt.subplots(
        nrows=len(CONTRAST_ROWS),
        ncols=len(SIZE_ORDER),
        figsize=(13.2, 12.5),
        sharex=True,
        constrained_layout=False,
    )
    model_offsets = {model: offset for model, offset in zip(MODEL_ORDER, (-0.17, 0.0, 0.17))}
    layer_counts = {"small": 2, "medium": 4, "large": 6}

    for row_idx, (metric_key, contrast_key, metric_label, contrast_label, ylim) in enumerate(CONTRAST_ROWS):
        for size_col, size in enumerate(SIZE_ORDER):
            ax = axes[row_idx, size_col]
            ax.set_title(
                f"{size.title()} ({layer_counts[size]} layers)\n{contrast_label}",
                fontsize=10,
                pad=7,
            )

            for family_idx, family in enumerate(FAMILY_ORDER):
                for model in MODEL_ORDER:
                    record = index[(size, model, family)]
                    contrast = record["paired_contrasts"][contrast_key]
                    estimate = contrast[metric_key]
                    if metric_key == "cross_entropy_difference":
                        low, high = contrast["cross_entropy_difference_95_interval"]
                    else:
                        low, high = contrast["accuracy_difference_95_interval_pp"]
                    ax.errorbar(
                        family_idx + model_offsets[model],
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
        "Paired changes after changing regime-membership input\n"
        "Each point compares two conditions on the same 2,880 episodes per routing family",
        fontsize=14,
        y=0.985,
    )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[model], linestyle="none", markersize=6,
               label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.012,
        "Dots are paired mean differences; bars are 95% episode-bootstrap intervals, not training-seed uncertainty.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.095, left=0.12, right=0.985, hspace=0.48, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def plot_model_gaps(comparison_summary: dict, output: Path) -> None:
    """Plot paired fixed/curriculum differences from original by condition."""
    index = {
        (row["size"], row["model"], row["family"], row["condition"]): row
        for row in comparison_summary["rows"]
    }
    fig, axes = plt.subplots(
        nrows=len(MODEL_GAP_ROWS),
        ncols=len(SIZE_ORDER),
        figsize=(13.2, 12.5),
        sharex=True,
        constrained_layout=False,
    )
    layer_counts = {"small": 2, "medium": 4, "large": 6}
    x = list(range(len(CONDITIONS)))

    for row_idx, (family, metric, panel_title, axis_label) in enumerate(MODEL_GAP_ROWS):
        for size_col, size in enumerate(SIZE_ORDER):
            ax = axes[row_idx, size_col]
            ax.set_title(
                f"{size.title()} ({layer_counts[size]} layers)\n{panel_title}",
                fontsize=10,
                pad=7,
            )
            panel_values = []
            for model in MODEL_CANDIDATES:
                entries = [index[(size, model, family, condition)] for condition in CONDITIONS]
                metric_key = "accuracy_difference_pp" if metric == "accuracy" else "cross_entropy_difference"
                interval_key = (
                    "accuracy_difference_95_interval_pp"
                    if metric == "accuracy"
                    else "cross_entropy_difference_95_interval"
                )
                estimates = np.asarray([entry[metric_key] for entry in entries], dtype=float)
                intervals = np.asarray([entry[interval_key] for entry in entries], dtype=float)
                yerr = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
                panel_values.extend(intervals.ravel().tolist())
                ax.errorbar(
                    x,
                    estimates,
                    yerr=yerr,
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
                )

            gap_span = max(panel_values) - min(panel_values)
            pad = max(gap_span * 0.14, 1e-4 if metric == "cross_entropy" else 0.01)
            ax.set_ylim(min(0.0, min(panel_values) - pad), max(0.0, max(panel_values) + pad))
            ax.axhline(0, color="#555555", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
            ax.set_xlim(-0.35, 2.35)
            ax.set_xticks(x, CONDITION_LABELS if row_idx == len(MODEL_GAP_ROWS) - 1 else ("", "", ""), fontsize=8.5)
            ax.tick_params(axis="y", labelsize=9)
            if size_col == 0:
                ax.set_ylabel(axis_label, fontsize=9.5, labelpad=9)

    fig.suptitle(
        "Predictive performance relative to the single-regime baseline",
        fontsize=14.5,
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
        "Whiskers are 95% paired, cell-stratified episode-bootstrap intervals (5,000 replicates; 2,880 episodes per family).",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    fig.text(
        0.5,
        0.023,
        "Intervals capture evaluation-episode variation, not training-seed uncertainty. Panel y-scales vary.",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.145, left=0.12, right=0.985, hspace=0.48, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--model-comparison-summary", type=Path, default=DEFAULT_MODEL_COMPARISON_SUMMARY)
    parser.add_argument("--refresh-model-comparison-summary", action="store_true")
    parser.add_argument("--input-effects-output", type=Path, default=DEFAULT_INPUT_EFFECT_OUTPUT)
    parser.add_argument("--means-output", type=Path, default=DEFAULT_MEANS_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())["rows"]
    if args.refresh_model_comparison_summary or not args.model_comparison_summary.exists():
        comparison_summary = summarize_paired_model_differences(args.report_root)
        write_summary(comparison_summary, args.model_comparison_summary)
    else:
        comparison_summary = json.loads(args.model_comparison_summary.read_text())
    plot_model_gaps(comparison_summary, args.output)
    plot_paired_differences(rows, args.input_effects_output)
    plot_absolute_means(rows, args.means_output)


if __name__ == "__main__":
    main()
