"""BeyondArena v4 figures from ``summarize_beyondarena_v4.py --json-output`` files.

* ``beyondarena_scores_<condition>.png`` — three panels (excess CE, accuracy
  gain, macro AUC): one bar per model, grouped by size, coloured by prior
  family, hatched for z-exposed training; published models as horizontal
  reference lines.
* ``beyondarena_rank_<condition>.png`` — the same layout for cross-model mean
  ranks (1 = best), with the published models as bars too.
* ``beyondarena_rank_scatter_<condition>.png`` — excess-CE rank vs AUC rank
  per model (calibration vs ranking quality).

    python scripts/plot_beyondarena_v4.py --scores ba/scores.json --rank ba/rank32.json \\
        --conditions all binary multi --output-dir figures/beyondarena_v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FAMILIES = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]
COLORS = {
    "original": "#1f77e6",
    "r_z-fixed": "#ff5a2e",
    "r_z-curriculum": "#12b07a",
    "g_z-fixed": "#f0a000",
    "g_z-curriculum": "#f67ba8",
}
SIZES = ["small", "medium", "large"]
PUBLISHED = ["tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3", "tabicl-v1", "tabicl-v2"]
CONVENTIONAL = ["logreg", "rf", "hgb", "xgboost", "lightgbm", "catboost"]
GROUP_COLOR = {"published": "gray", "conventional": "silver"}
PUB_STYLE = {"tabpfn-v2.2": "-.", "tabpfn-v2.6": ":", "tabpfn-v3": (0, (5, 1)), "tabicl-v1": "-.", "tabicl-v2": ":"}
PUB_COLOR = {"tabpfn-v2.2": "black", "tabpfn-v2.6": "black", "tabpfn-v3": "black", "tabicl-v1": "dimgray", "tabicl-v2": "dimgray"}
METRICS = [("excess_cross_entropy", "excess CE (lower better)"), ("accuracy_gain", "accuracy gain (higher better)"), ("macro_ovr_auc", "macro OvR AUC (higher better)")]


def parse(model: str) -> tuple[str, str, bool] | None:
    """-> (family, size, z_exposed) for NanoTabPFN runs, None for published models."""
    zx = model.startswith("zx-")
    name = model[3:] if zx else model
    for fam in FAMILIES:
        if name.startswith(fam + "-"):
            return fam, name[len(fam) + 1 :], zx
    return None


def group(model: str) -> str:
    return "conventional" if model in CONVENTIONAL else "published"


def order(models: list[str]) -> list[str]:
    def key(m):
        p = parse(m)
        if p is None:
            return ((8, 0, 0, PUBLISHED.index(m)) if m in PUBLISHED else (9, 0, 0, CONVENTIONAL.index(m)))
        fam, size, zx = p
        return (SIZES.index(size), FAMILIES.index(fam), int(zx), m)
    return sorted(models, key=key)


def bars(ax, models: list[str], values: dict[str, float], ylabel: str, published_lines: dict[str, float] | None, published_bars: bool):
    xs, ticks = [], []
    x = 0.0
    last_size = None
    for m in models:
        p = parse(m)
        if p is None and not published_bars:
            continue
        if p is not None and p[1] != last_size:
            x += 0.8  # gap between size groups
            last_size = p[1]
        elif p is None and last_size != group(m):
            x += 0.8
            last_size = group(m)
        color = COLORS[p[0]] if p else GROUP_COLOR[group(m)]
        ax.bar(x, values[m], width=0.8, color=color, edgecolor="black", linewidth=0.4, hatch="//" if (p and p[2]) else None)
        xs.append(x)
        ticks.append(m)
        x += 1.0
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, rotation=90, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if published_lines:
        for name, v in published_lines.items():
            ax.axhline(v, color=PUB_COLOR[name], ls=PUB_STYLE[name], lw=1.1, alpha=0.85)
    if "AUC" in ylabel and "rank" not in ylabel:
        vals = [values[m] for m in models if m in values] + list((published_lines or {}).values())
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.01)
    elif "rank" not in ylabel:
        ax.axhline(0, color="black", lw=0.8)


def legend(fig, published: list[str]):
    handles = [Patch(facecolor=COLORS[f], edgecolor="black", label=f) for f in FAMILIES]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="z-exposed training (zx-)"))
    handles += [plt.Line2D([], [], color=PUB_COLOR[p], ls=PUB_STYLE[p], label=p) for p in published]
    handles.append(Patch(facecolor=GROUP_COLOR["published"], edgecolor="black", label="published model (bar)"))
    handles.append(Patch(facecolor=GROUP_COLOR["conventional"], edgecolor="black", label="conventional ML (bar)"))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--rank", required=True, help="rank JSON including the published models")
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heatmap-rank", default=None, help="rank JSON (NanoTabPFN runs only) for family x size heatmaps")
    parser.add_argument("--heatmap-grid", nargs="+", default=["all", "binary__IID", "binary__Grouped", "multi__IID", "multi__Grouped"])
    parser.add_argument("--heatmap-rows", nargs="*", default=None,
                        help="row families for the heatmap (default: the 5 v4 families + their zx- twins); "
                             "e.g. nat-original nat-rg_z-fixed nat-rg_z-curriculum")
    parser.add_argument("--heatmap-tag", default="", help="suffix for the heatmap file name, e.g. _native")
    args = parser.parse_args()
    if args.heatmap_rank:
        heatmaps(args)
    scores = json.load(open(args.scores))
    ranks = json.load(open(args.rank))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        if cond not in scores or cond not in ranks:
            print(f"skip {cond}: not in both inputs")
            continue
        # scores: keys "model|policy"; take final / published_default
        sc = {k.split("|")[0]: v for k, v in scores[cond].items() if k.split("|")[1] in ("final", "published_default")}
        models = order([m for m in sc if parse(m) is not None])
        published = [p for p in PUBLISHED if p in sc]
        tasks = next(iter(sc.values()))["tasks"]

        smodels = models + published + [c for c in CONVENTIONAL if c in sc]
        fig, axes = plt.subplots(3, 1, figsize=(0.42 * len(smodels) + 4, 18))
        for ax, (metric, label) in zip(axes, METRICS):
            bars(ax, smodels, {m: sc[m][metric] for m in smodels}, label, {p: sc[p][metric] for p in published}, published_bars=True)
        fig.suptitle(f"BeyondArena v4 — final checkpoints, condition = {cond} ({tasks} tasks); coloured bars = NanoTabPFN runs, gray = published, silver = conventional ML (lines = published)")
        legend(fig, published)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(out / f"beyondarena_scores_{cond}.png", dpi=110)
        plt.close(fig)

        rk = ranks[cond]
        rmodels = order([m for m in rk if parse(m) is not None]) + [p for p in PUBLISHED if p in rk] + [c for c in CONVENTIONAL if c in rk]
        ntasks = next(iter(rk.values()))["tasks"]
        fig, axes = plt.subplots(3, 1, figsize=(0.42 * len(rmodels) + 4, 18))
        for ax, (metric, label) in zip(axes, METRICS):
            bars(ax, rmodels, {m: rk[m][metric] for m in rmodels}, f"mean rank of {label.split(' (')[0]} (1 = best)", None, published_bars=True)
            ax.invert_yaxis()
            ax.axhline(len(rmodels) / 2 + 0.5, color="gray", lw=0.8, ls="--")
        fig.suptitle(f"BeyondArena v4 — cross-model mean rank, condition = {cond} ({ntasks} tasks, {len(rmodels)} models)")
        legend(fig, [])
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(out / f"beyondarena_rank_{cond}.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 12))
        for m in rmodels:
            p = parse(m)
            color = COLORS[p[0]] if p else ("black" if group(m) == "published" else "dimgray")
            marker = {"small": "o", "medium": "s", "large": "^"}[p[1]] if p else ("*" if group(m) == "published" else "P")
            ax.scatter(rk[m]["excess_cross_entropy"], rk[m]["macro_ovr_auc"], color=color, marker=marker, s=90 if p else 160,
                       facecolors="none" if (p and p[2]) else color, linewidths=1.5, edgecolors=color, zorder=3)
            ax.annotate(m, (rk[m]["excess_cross_entropy"], rk[m]["macro_ovr_auc"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("mean rank, excess CE (1 = best)")
        ax.set_ylabel("mean rank, macro AUC (1 = best)")
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.grid(alpha=0.3)
        lim = max(len(rmodels), 1)
        ax.plot([1, lim], [1, lim], color="gray", lw=0.8, ls="--")
        ax.set_title(f"BeyondArena v4 — calibration rank vs ranking rank, condition = {cond}\n(circle small, square medium, triangle large; hollow = z-exposed; star = published; plus = conventional ML)")
        fig.tight_layout()
        fig.savefig(out / f"beyondarena_rank_scatter_{cond}.png", dpi=110)
        plt.close(fig)
        print(cond, "->", out / f"beyondarena_{{scores,rank,rank_scatter}}_{cond}.png")


def heatmaps(args) -> None:
    """Family x size heatmaps of mean rank (among the NanoTabPFN runs), one column per task grid, one row per
    metric. Rows = the 9 model families incl. the z-exposed (zx-) variants, so all 27 runs are in one figure."""
    ranks = json.load(open(args.heatmap_rank))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grids = [g for g in args.heatmap_grid if g in ranks]
    nmodels = len(next(iter(ranks.values())))
    rows = args.heatmap_rows or (FAMILIES + [f"zx-{f}" for f in FAMILIES[1:]])
    separator = len(FAMILIES) - 0.5 if not args.heatmap_rows else None
    fig, axes = plt.subplots(len(METRICS), len(grids), figsize=(5.2 * len(grids), 5.4 * len(METRICS)), squeeze=False)
    for i, (metric, label) in enumerate(METRICS):
        for j, g in enumerate(grids):
            ax = axes[i][j]
            mat = [[float("nan")] * len(SIZES) for _ in rows]
            for a, fam in enumerate(rows):
                for b, size in enumerate(SIZES):
                    name = f"{fam}-{size}"
                    if name in ranks[g]:
                        mat[a][b] = ranks[g][name][metric]
            im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=nmodels, aspect="auto")
            for a in range(len(rows)):
                for b in range(len(SIZES)):
                    v = mat[a][b]
                    if v == v:
                        ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=8.5)
            if separator is not None:
                ax.axhline(separator, color="black", lw=1.2)  # separates blind from z-exposed rows
            ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows if j == 0 else [""] * len(rows), fontsize=8.5)
            ntasks = next(iter(ranks[g].values()))["tasks"]
            title = "ALL tasks" if g == "all" else g.replace("__", " × ")
            ax.set_title(f"{title} ({ntasks} tasks)", fontsize=10, fontweight="bold" if g == "all" else "normal")
            if g == "all":  # mark the overall column
                for spine in ax.spines.values():
                    spine.set_linewidth(2.0)
            if j == 0:
                ax.set_ylabel(label.split(" (")[0] + "\nmean rank (1 = best)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label=f"mean rank among {nmodels} runs")
    subtitle = "(rows below the line: z-exposed training)" if separator is not None else "(native-TabICL-prior runs)"
    fig.suptitle(f"BeyondArena v4 — family × size mean rank by task grid, final checkpoints {subtitle}")
    path = out / f"beyondarena_heatmap_rank{args.heatmap_tag}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
