#!/usr/bin/env python3
"""Plot paired hidden/shuffled/true regime-tag contrasts with bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "paper/native/observed_regime_information_condition_summary.json"
DEFAULT_OUTPUT = ROOT / "figures/native/regime_information_paired_contrasts.png"

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
FAMILY_LABELS = {"persistent": "Persistent routing", "soft_gate": "Soft-gate routing"}
CONTRASTS = (
    ("shuffled_minus_hidden", "Shuffled − hidden"),
    ("true_minus_shuffled", "True − shuffled"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    index = {(r["size"], r["model"], r["family"]): r for r in data["rows"]}
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 12.3), sharex=True)
    xs = np.arange(len(SIZE_ORDER))

    for family_col, family in enumerate(FAMILY_ORDER):
        for contrast_row, (contrast_key, contrast_label) in enumerate(CONTRASTS):
            for model in MODEL_ORDER:
                records = [index[(size, model, family)]["paired_contrasts"][contrast_key] for size in SIZE_ORDER]
                color = MODEL_COLORS[model]

                ce = np.array([r["cross_entropy_difference"] for r in records])
                ce_ci = np.array([r["cross_entropy_difference_95_interval"] for r in records])
                axes[contrast_row, family_col].errorbar(
                    xs,
                    ce,
                    yerr=np.vstack((ce - ce_ci[:, 0], ce_ci[:, 1] - ce)),
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    capsize=3,
                    label=MODEL_LABELS[model],
                )

                acc = np.array([r["accuracy_difference_pp"] for r in records])
                acc_ci = np.array([r["accuracy_difference_95_interval_pp"] for r in records])
                axes[contrast_row + 2, family_col].errorbar(
                    xs,
                    acc,
                    yerr=np.vstack((acc - acc_ci[:, 0], acc_ci[:, 1] - acc)),
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    capsize=3,
                    label=MODEL_LABELS[model],
                )

        axes[0, family_col].set_title(FAMILY_LABELS[family])

    row_labels = (
        "Δ CE (nats)\nShuffled − hidden",
        "Δ CE (nats)\nTrue − shuffled",
        "Δ accuracy (pp)\nShuffled − hidden",
        "Δ accuracy (pp)\nTrue − shuffled",
    )
    for row, label in enumerate(row_labels):
        for col in range(2):
            ax = axes[row, col]
            ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=0)
            ax.grid(axis="y", color="#dddddd", linewidth=0.7)
            ax.set_ylabel(label)
            ax.set_xticks(xs, [s.title() for s in SIZE_ORDER])

    fig.suptitle(
        "Paired regime-membership contrasts by routing family and model size\n"
        "Negative ΔCE / positive Δaccuracy favor the first-named condition",
        fontsize=14,
        y=0.99,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.015,
        "Points and 95% intervals are paired within episode; 2,880 episodes across 360 factorial cells per routing family. "
        "Intervals use 5,000 cell-stratified bootstrap replicates and quantify finite-bank, not training-seed, variation.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(top=0.87, bottom=0.09, hspace=0.37, wspace=0.27)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
