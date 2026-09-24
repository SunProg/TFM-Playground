"""Within-episode regime breakdown with the regimes ranked by a reference model's cross entropy.

Input: ``scratchpad/agg_regime_ranked.py`` output (``native_regime_ranked_<size>_<split>.json``), which
buckets every K>=2 episode's query rows by their regime id, then orders the regimes of each episode by the
reference run's per-regime CE (rank 1 = the regime that run fits best) and applies the same permutation to
every other run.  Ranking makes the positions comparable across episodes, which the raw regime index is not:
``soft_gate`` ids are ordered rank bins of x[:, 0], but ``persistent`` ids are a fresh random relabelling per
episode.

    python scripts/plot_native_regime_ranked.py --data native_regime_ranked_large_test.json \\
        --output-dir figures/native
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"original": "#1f77e6", "rg_z-fixed": "#7b3fe4", "rg_z-curriculum": "#00a3c4"}
GROUPS = [
    ("soft_gate_binary", "soft_gate, binary"), ("soft_gate_multiclass", "soft_gate, multiclass"),
    ("persistent_binary", "persistent, binary"), ("persistent_multiclass", "persistent, multiclass"),
]
METRICS = [("ce", "cross entropy (lower better)"), ("acc", "accuracy"), ("auc", "macro OvR AUC")]


def family(run: str, size: str) -> str:
    return run[: -len(size) - 1] if run.endswith(size) else run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--style", default="bar", choices=["bar", "line", "delta"],
                    help="grouped bars (default), lines, or paired difference vs the reference run with 95 % CI")
    a = ap.parse_args()
    payload = json.load(open(a.data))
    size, split, runs, ref = payload["size"], payload["split"], payload["runs"], payload["reference"]
    out = Path(a.output_dir) / "regime_ranked"; out.mkdir(parents=True, exist_ok=True)
    groups = payload["groups"]
    ks = sorted({k for group in groups.values() for k in group}, key=lambda s: int(s[1:]))

    for metric, label in METRICS:
        fig, axes = plt.subplots(len(GROUPS), len(ks), figsize=(4.6 * len(ks), 3.6 * len(GROUPS)), squeeze=False)
        for i, (key, title) in enumerate(GROUPS):
            for j, k in enumerate(ks):
                ax = axes[i][j]
                cell = groups.get(key, {}).get(k)
                if not cell:
                    ax.set_axis_off(); continue
                n_ranks = int(k[1:])
                ranks = list(range(1, n_ranks + 1))
                values = {run: cell[run][metric] for run in runs}
                if a.style == "delta":
                    others = [run for run in runs if run != ref]
                    width = 0.8 / max(1, len(others))
                    for offset, run in enumerate(others):
                        delta = cell[run].get("delta_vs_ref", {}).get(metric)
                        if not delta:
                            continue
                        positions = [r + (offset - (len(others) - 1) / 2) * width for r in ranks]
                        errors = [1.96 * e for e in delta["se"]]
                        ax.bar(positions, delta["mean"], width=width * 0.9, yerr=errors, capsize=3,
                               color=COLORS.get(family(run, size), "gray"), label=f"{family(run, size)} - {family(ref, size)}",
                               edgecolor="white", linewidth=0.6, error_kw={"lw": 1.0})
                        for r, value, error in zip(positions, delta["mean"], errors):
                            significant = abs(value) > error
                            ax.annotate(f"{value:+.3f}{'*' if significant else ''}", (r, value),
                                        ha="center", va="bottom" if value >= 0 else "top", fontsize=7,
                                        fontweight="bold" if significant else "normal",
                                        color=COLORS.get(family(run, size), "gray"))
                    ax.axhline(0, color="black", lw=1.0)
                elif a.style == "line":
                    for run in runs:
                        ax.plot(ranks, values[run], marker="o", ms=6, lw=2,
                                color=COLORS.get(family(run, size), "gray"), label=family(run, size))
                else:
                    width = 0.8 / len(runs)
                    for offset, run in enumerate(runs):
                        positions = [r + (offset - (len(runs) - 1) / 2) * width for r in ranks]
                        ax.bar(positions, values[run], width=width * 0.92,
                               color=COLORS.get(family(run, size), "gray"), label=family(run, size),
                               edgecolor="white", linewidth=0.6)
                    # zoom the y-axis to the bars and label each one
                    flat = [v for run in runs for v in values[run]]
                    lo, hi = min(flat), max(flat)
                    pad = 0.25 * (hi - lo or 0.01)
                    base = 0.5 if metric == "auc" else max(0.0, lo - pad)
                    ax.set_ylim(min(base, lo - pad), hi + pad)
                    for offset, run in enumerate(runs):
                        for r, v in zip(ranks, values[run]):
                            ax.annotate(f"{v:.3f}", (r + (offset - (len(runs) - 1) / 2) * width, v),
                                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                                        color=COLORS.get(family(run, size), "gray"))
                if metric == "auc" and a.style != "delta":
                    ax.axhline(0.5, color="black", lw=1.0, ls=":")
                ax.set_title(f"{title}, {k} ({cell['episodes']} episodes)", fontsize=10)
                ax.set_xticks(ranks); ax.grid(alpha=0.3, axis="y")
                if j == 0: ax.set_ylabel(("Δ " if a.style == "delta" else "") + label, fontsize=9)
                if i == len(GROUPS) - 1: ax.set_xlabel(f"regime rank (1 = best for {ref})")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(runs), fontsize=10, frameon=True)
        if a.style == "delta":
            fig.suptitle(f"{size} runs, {split.upper()}: paired change in {label} vs {ref} per regime "
                         f"(negative = better for CE; bars = mean per-episode difference, whiskers = 95 % CI, "
                         f"* = CI excludes 0); regimes ranked by {ref} (rank 1 = easiest)", fontsize=12)
        else:
            fig.suptitle(f"{size} runs, {split.upper()} {label} per regime — regimes ranked within each episode by "
                         f"{ref} (rank 1 = easiest)", fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.965))
        suffix = {"bar": "", "line": "_line", "delta": "_delta"}[a.style]
        path = out / f"native_{size}_{split}_regime_ranked_{metric}{suffix}.png"
        fig.savefig(path, dpi=110); plt.close(fig); print(path)


if __name__ == "__main__":
    main()
