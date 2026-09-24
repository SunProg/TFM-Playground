"""Per-model-size training-history plots from pretrain_plain_nanotabpfn ``history.jsonl`` files.

For each size writes:

* ``pretraining_loss_<size>.png`` — per-step query cross entropy (rolling mean)
* ``val_loss_<size>.png``         — own-prior validation loss at every validation step

History files are given as ``--history <task>=<path>`` (task = ``<family>-<size>``),
so the script works on a local copy of the cluster files.

    python scripts/plot_pretraining_history.py --output-dir figures/per_size_plots \\
        --history original-small=hist_original-small.jsonl --history r_z-fixed-small=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]
COLORS = {
    "original": "#1f77e6",
    "r_z-fixed": "#ff5a2e",
    "r_z-curriculum": "#12b07a",
    "g_z-fixed": "#f0a000",
    "g_z-curriculum": "#f67ba8",
}
SIZES = ["small", "medium", "large"]


def load(path: str) -> list[dict]:
    """Rows in step order; a resumed run re-logs steps after its checkpoint, so the last record per step wins."""
    by_step: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_step[int(row["step"])] = row
    return [by_step[s] for s in sorted(by_step)]


def rolling(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    out = np.convolve(arr, kernel, mode="valid")
    return np.concatenate([arr[: window - 1], out])  # keep the x-axis aligned with steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", action="append", required=True, metavar="TASK=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=30, help="rolling-mean window for the pretraining loss")
    parser.add_argument("--sizes", nargs="+", default=SIZES)
    parser.add_argument("--families", nargs="+", default=None, help=f"families (lines) to draw; default {FAMILIES}")
    parser.add_argument("--prefix", default="", help="output filename prefix")
    args = parser.parse_args()
    if args.families:
        FAMILIES[:] = args.families
    extra_colors = iter(["#6f42c1", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22"])
    for family in FAMILIES:
        if family not in COLORS:
            COLORS[family] = next(extra_colors)

    histories = {task: load(path) for task, path in (item.split("=", 1) for item in args.history)}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        # pretraining loss
        fig, ax = plt.subplots(figsize=(12, 7.5))
        for family in FAMILIES:
            rows = histories.get(f"{family}-{size}")
            if not rows:
                continue
            steps = [r["step"] for r in rows if "query_cross_entropy" in r]
            losses = [r["query_cross_entropy"] for r in rows if "query_cross_entropy" in r]
            ax.plot(steps, rolling(losses, args.window), lw=1.4, color=COLORS[family], label=f"{family} (step {steps[-1]})")
        ax.set_title(f"{size} tasks — pretraining loss trajectory")
        ax.set_xlabel("step")
        ax.set_ylabel(f"query cross entropy ({args.window}-step rolling mean)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = out_dir / f"{args.prefix}pretraining_loss_{size}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out.resolve())

        # own-prior validation loss
        fig, ax = plt.subplots(figsize=(12, 7.5))
        for family in FAMILIES:
            rows = histories.get(f"{family}-{size}")
            if not rows:
                continue
            val = [(r["step"], r["validation_query_cross_entropy"]) for r in rows if "validation_query_cross_entropy" in r]
            if not val:
                continue
            steps, losses = zip(*val)
            ax.plot(steps, losses, marker="o", ms=4, lw=1.5, color=COLORS[family], label=f"{family} (step {steps[-1]})")
        ax.set_title(f"{size} tasks — own-prior validation loss (NOT shared across families)")
        ax.set_xlabel("step")
        ax.set_ylabel("own-prior validation query cross entropy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = out_dir / f"{args.prefix}val_loss_{size}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out.resolve())


if __name__ == "__main__":
    main()
