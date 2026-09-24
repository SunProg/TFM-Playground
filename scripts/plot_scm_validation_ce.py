#!/usr/bin/env python3
"""Plot validation cross-entropy trajectories for realistic SCM runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "decoder_baseline": "Table-slot decoder baseline",
    "decoder_alpha": "Table-slot decoder alpha",
    "blind_decoder": "Table-slot blind decoder",
    "blind_similarity": "Table-slot blind similarity",
    "nanotabpfn": "Plain NanoTabPFN",
    "table_slot_backbone": "Table-slot backbone",
    "table_slot_mufasa": "Table-slot Mufasa",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trajectories = defaultdict(dict)
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            seed = int(row["seed"])
            step = int(row["step"])
            trajectories[model][seed, step] = float(row["validation_cross_entropy"])

    order = (
        "decoder_baseline",
        "nanotabpfn",
        "blind_similarity",
        "table_slot_mufasa",
        "decoder_alpha",
        "blind_decoder",
        "table_slot_backbone",
    )
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for color, model in zip(colors[: len(order)], order, strict=True):
        seeds = sorted({seed for seed, _ in trajectories[model]})
        steps = sorted({step for _, step in trajectories[model]})
        values = np.asarray([[trajectories[model][seed, step] for step in steps] for seed in seeds])
        for seed_values in values:
            ax.plot(steps, seed_values, color=color, alpha=0.22, linewidth=1.2)
        mean = values.mean(axis=0)
        low = values.min(axis=0)
        high = values.max(axis=0)
        ax.fill_between(steps, low, high, color=color, alpha=0.08)
        ax.plot(steps, mean, color=color, linewidth=2.4, label=LABELS[model])

    ax.axhline(np.log(2.0), color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Binary chance CE")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation cross-entropy (lower is better)")
    ax.set_title("Realistic SCM validation trajectories (5,000-step runs)")
    ax.set_xlim(0, 5000)
    ax.set_ylim(bottom=0.38, top=0.72)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
