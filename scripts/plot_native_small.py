"""Native-prior runs: per-checkpoint excess CE on the native banks, per cell group, blind vs z-exposed bank.

Input: scratchpad/native_small_cells.json produced on the cluster (keys 'bank/split/run/tag' ->
{'excess_ce': {cell: value}, 'acc_gain': {...}, 'overall': {...}}).

    python scripts/plot_native_small.py --cells native_small_cells.json --output-dir figures/native --size small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {"original": "#1f77e6", "rg_z-fixed": "#7b3fe4", "rg_z-curriculum": "#00a3c4"}


REF_STYLE = {  # reference models: distinct colours, all dashed so they can't be confused with the run curves
    "tabpfn-v2.2": ("#d62728", "--"), "tabpfn-v2.6": ("#ff7f0e", "--"), "tabpfn-v3": ("#8b0000", "--"),
    "tabicl-v1": ("#2ca02c", "--"), "tabicl-v1.1": ("#98df8a", "--"), "tabicl-v2": ("#006400", "--"),
    "logreg": ("#bcbd22", ":"), "rf": ("#8c564b", ":"), "hgb": ("#e377c2", ":"), "xgboost": ("#17becf", ":"),
    "lightgbm": ("#7f7f7f", ":"), "catboost": ("#9467bd", ":"),
}


def draw_refs(ax, refs, cell, metric, ylo, yhi, split="validation"):
    """Horizontal reference lines (given split), labelled at the right edge; off-scale values are pinned to
    the edge with an arrow and the value."""
    if not refs:
        return
    span = yhi - ylo
    step = 0.055 * span                      # minimum vertical gap between right-edge labels
    on_scale, above, below = [], [], []
    for name, (color, ls) in REF_STYLE.items():
        v = refs.get(f"{split}/{name}", {}).get(metric, {}).get(cell)
        if v is None or v != v:
            continue
        if ylo <= v <= yhi:
            ax.axhline(v, color=color, ls=ls, lw=1.6, alpha=0.95)
            on_scale.append((v, name, color))
        elif v > yhi:
            above.append((v, name, color))
        else:
            below.append((v, name, color))
    # On-scale labels: walk from the top down, nudging each label below the previous one and below the ▲ column.
    y_last = yhi - (len(above) - 0.5) * step if above else None
    floor = ylo + (len(below) - 0.5) * step if below else ylo - step   # keep clear of the ▼ column too
    for v, name, color in sorted(on_scale, reverse=True):
        y = v if y_last is None else min(v, y_last - step)
        y = max(y, floor)
        ax.annotate(f"{name} {v:+.3f}", (10600, y), fontsize=7, color=color, ha="left", va="center", annotation_clip=False)
        y_last = y
    # Off-scale labels: a column pinned to the top (▲) / bottom (▼) edge, one per row, closest value first.
    for i, (v, name, color) in enumerate(sorted(above)):
        ax.annotate(f"▲ {name} {v:+.2f}", (10600, yhi - i * step), fontsize=7, color=color, ha="left", va="center",
                    annotation_clip=False, fontweight="bold")
    for i, (v, name, color) in enumerate(sorted(below, reverse=True)):
        ax.annotate(f"▼ {name} {v:+.2f}", (10600, ylo + i * step), fontsize=7, color=color, ha="left", va="center",
                    annotation_clip=False, fontweight="bold")
    ax.set_xlim(0, 10500)


def ref_handles(refs, split="validation"):
    return [Line2D([], [], color=c, ls=ls, lw=2, label=f"{n} ({split})") for n, (c, ls) in REF_STYLE.items()
            if f"{split}/{n}" in (refs or {})]


def legend_handles(fams, split="validation"):
    """Colour = prior family; line style / marker = which bank the checkpoint was scored on."""
    h = [Line2D([], [], color=COLORS.get(f, "gray"), lw=3, label=f"{f}") for f in fams]
    h += [Line2D([], [], color="black", ls="-", marker="o", ms=5, label=f"{split}, z-BLIND bank (solid, circles)"),
          Line2D([], [], color="black", ls="--", marker="s", ms=5, label=f"{split}, z-EXPOSED bank (dashed, squares)"),
          Line2D([], [], color="black", ls="none", marker="D", ms=7, label="final TEST, z-blind bank (large diamond at x=10250)"),
          Line2D([], [], color="black", ls="none", marker="d", ms=7, label="final TEST, z-exposed bank (thin diamond at x=10250)")]
    return h
def draw_cell(ax, data, runs, size, metric, cell, split, refs, ms=5, lw=1.8):
    """One panel: per-run curves on both banks for the given split (validation: every 500 steps from the trainer;
    test: every 2000 steps from the checkpoint evals), final-checkpoint TEST diamonds, reference lines."""
    for run in runs:
        fam = run[: -len(size) - 1]
        for bank, ls, mk in (("blind", "-", "o"), ("zx", "--", "s")):
            pts = sorted((int(t.split("-")[1]), v[metric][cell]) for key, v in data.items() for b, sp, r, t in [key.split("/")]
                         if b == bank and r == run and sp == split and t != "final" and cell in v[metric])
            if pts:
                ax.plot(*zip(*pts), ls=ls, marker=mk, ms=ms, lw=lw, color=COLORS.get(fam, "gray"))
            tk = f"{bank}/test/{run}/final"
            if tk in data and cell in data[tk][metric]:
                ax.plot([10250], [data[tk][metric][cell]], marker="D" if bank == "blind" else "d", ms=ms + 4, color=COLORS.get(fam, "gray"), ls="none", markeredgecolor="black")
    ax.axhline(0, color="black", lw=0.8); ax.grid(alpha=0.3)
    ys = [y for line in ax.get_lines() for y in line.get_ydata() if y == y]
    if ys:
        lo, hi = min(ys), max(ys); pad = 0.6 * (hi - lo or 0.01)
        draw_refs(ax, refs, cell, metric, lo - pad, hi + pad, split); ax.set_ylim(lo - pad, hi + pad)


PANELS = [
    ("all", "all cells"), ("c2r0.1", "binary, ratio 0.1"), ("c2r0.3", "binary, ratio 0.3"), ("c2r0.5", "binary, ratio 0.5"),
    ("c3", "3 classes"), ("c5", "5 classes"), ("K1", "K = 1"), ("shared_bin", "shared rule, K≥2, binary"),
    ("multiregime_bin", "multiregime, K≥2, binary"), ("multiregime_multi", "multiregime, K≥2, multiclass"),
    ("multiregime_soft_gate", "multiregime, soft_gate"), ("multiregime_persistent", "multiregime, persistent"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--metric", default="excess_ce", choices=["excess_ce", "acc_gain"])
    ap.add_argument("--grid", action="store_true", help="regime (K) x classes grid instead of the cell panels")
    ap.add_argument("--family", default=None, help="with --grid: restrict to one routing family (soft_gate | persistent), multiregime rule only, K>=2")
    ap.add_argument("--split", default="validation", choices=["validation", "test"],
                    help="validation: trainer's 500-step reports; test: checkpoint evals every 2000 steps (output *_test.png)")
    ap.add_argument("--refs", default=None, help="native_refs_cells.json: published + conventional reference models drawn as horizontal lines")
    ap.add_argument("--mechanism", default=None, choices=["r_z", "g_z"],
                    help="restrict to one SCM mechanism (r_z = per-regime score column, g_z = independent rules on one score)")
    ap.add_argument("--support-x-features", metavar="CONDITION", default=None,
                    help="support size (rows) x feature count (columns) grid for ONE condition, one file per condition: "
                         "all | bin | multi | K1 | multiregime | shared | multiregime_bin | multiregime_multi | "
                         "multiregime_soft_gate | multiregime_persistent | c2r0.1 | c3 | ...")
    a = ap.parse_args()
    a.refs = json.load(open(a.refs)) if a.refs else None
    if a.support_x_features:
        return support_x_features(a)
    if a.grid:
        return grid(a)
    data = json.load(open(a.cells))
    out = Path(a.output_dir) / "cells"; out.mkdir(parents=True, exist_ok=True)
    runs = sorted({k.split("/")[2] for k in data if k.split("/")[2].endswith(a.size)})
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for ax, (cell, title) in zip(axes.ravel(), PANELS):
        draw_cell(ax, data, runs, a.size, a.metric, cell, a.split, a.refs)
        ax.set_title(title, fontsize=11); ax.set_xlabel("step"); ax.set_ylabel("excess CE (lower better)" if a.metric == "excess_ce" else "accuracy gain")
    fams = [r[: -len(a.size) - 1] for r in runs]
    fig.legend(handles=legend_handles(fams, a.split) + ref_handles(a.refs, a.split), loc="lower center", ncol=5, fontsize=9, frameon=True)
    fig.suptitle(f"Native-prior {a.size} runs — {a.split.upper()} {'excess CE (lower = better; 0 = class prior)' if a.metric == 'excess_ce' else 'accuracy gain over majority class'} per checkpoint{' (every 2000 steps, checkpoint evals)' if a.split == 'test' else ''}; colour = prior family, line style = evaluation bank; horizontal lines = reference models ({a.split})", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    p = out / f"native_{a.size}_{a.metric}_cells{'_test' if a.split == 'test' else ''}.png"
    fig.savefig(p, dpi=110); print(p)



SUPPORT_SIZES = [64, 128, 256, 512]
FEATURE_COUNTS = [2, 4, 8, 12, 16, 24]


def support_x_features(a) -> None:
    """One file per condition; rows = support-set size, columns = number of features."""
    data = json.load(open(a.cells))
    out = Path(a.output_dir) / "support_x_features"; out.mkdir(parents=True, exist_ok=True)
    runs = sorted({k.split("/")[2] for k in data if k.split("/")[2].endswith(a.size)})
    cond = a.support_x_features
    parts = [] if cond == "all" else [cond]
    if a.mechanism:
        parts = (parts[:1] if parts and parts[0] in ("K1", "shared", "multiregime") else []) + [a.mechanism] + \
                ([p for p in parts if p not in ("K1", "shared", "multiregime")])
    prefix = ("_".join(parts) + "_") if parts else ""
    fig, axes = plt.subplots(len(SUPPORT_SIZES), len(FEATURE_COUNTS),
                             figsize=(4.6 * len(FEATURE_COUNTS), 3.8 * len(SUPPORT_SIZES)), squeeze=False)
    drawn = 0
    for i, sup in enumerate(SUPPORT_SIZES):
        for j, nfeat in enumerate(FEATURE_COUNTS):
            ax = axes[i][j]; cell = f"{prefix}s{sup}f{nfeat}"
            if not any(cell in v[a.metric] for v in data.values()):
                ax.set_axis_off(); continue
            draw_cell(ax, data, runs, a.size, a.metric, cell, a.split, a.refs, ms=4, lw=1.6)
            drawn += 1
            ax.set_title(f"support {sup} × {nfeat} features", fontsize=11)
            if j == 0: ax.set_ylabel(f"support {sup}\n" + ("excess CE" if a.metric == "excess_ce" else "accuracy gain"))
            if i == len(SUPPORT_SIZES) - 1: ax.set_xlabel("step")
    if not drawn:
        raise SystemExit(f"no cells for condition {cond!r} (looked for keys like {prefix}s64f2)")
    fams = [r[: -len(a.size) - 1] for r in runs]
    fig.legend(handles=legend_handles(fams, a.split) + ref_handles(a.refs, a.split), loc="lower center", ncol=5, fontsize=10, frameon=True)
    fig.suptitle(f"Native-prior {a.size} runs — {a.split.upper()} "
                 f"{'excess CE (lower = better; 0 = class prior)' if a.metric == 'excess_ce' else 'accuracy gain'} per checkpoint, "
                 f"support-set size (rows) × feature count (columns), condition: {cond}{(', mechanism ' + a.mechanism) if a.mechanism else ''}; "
                 f"colour = prior family, line style = evaluation bank; horizontal lines = reference models ({a.split})", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    p = out / f"native_{a.size}_{a.metric}_support_x_features_{cond}{('_' + a.mechanism) if a.mechanism else ''}{'_test' if a.split == 'test' else ''}.png"
    fig.savefig(p, dpi=110); print(p)


def grid(a) -> None:
    """Rows = regime count K (1 = single rule; 2-4 = multiregime rule), columns = class condition."""
    data = json.load(open(a.cells))
    out = Path(a.output_dir) / "regime_x_classes"; out.mkdir(parents=True, exist_ok=True)
    runs = sorted({k.split("/")[2] for k in data if k.split("/")[2].endswith(a.size)})
    cols = [("c2r0.1", "binary, ratio 0.1"), ("c2r0.3", "binary, ratio 0.3"), ("c2r0.5", "binary, ratio 0.5"), ("c3", "3 classes"), ("c4", "4 classes"), ("c5", "5 classes")]
    rows = [(1, "K1"), (2, "multiregime"), (3, "multiregime"), (4, "multiregime")]
    if a.family:
        rows = [(2, f"multiregime_{a.family}"), (3, f"multiregime_{a.family}"), (4, f"multiregime_{a.family}")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(5.0 * len(cols), 4.0 * len(rows)), squeeze=False)
    for i, (k, rm) in enumerate(rows):
        for j, (c, title) in enumerate(cols):
            ax = axes[i][j]; cell = f"K{k}_{rm}_{a.mechanism}_{c}" if a.mechanism else f"K{k}_{rm}_{c}"
            draw_cell(ax, data, runs, a.size, a.metric, cell, a.split, a.refs, ms=4, lw=1.6)
            ax.set_title(f"K={k}{(' ' + a.family + ', multiregime rule') if a.family else (' (multiregime rule)' if k > 1 else '')} × {title}", fontsize=11)
            if j == 0: ax.set_ylabel("excess CE" if a.metric == "excess_ce" else "accuracy gain")
            if i == len(rows) - 1: ax.set_xlabel("step")
    fams = [r[: -len(a.size) - 1] for r in runs]
    fig.legend(handles=legend_handles(fams, a.split) + ref_handles(a.refs, a.split), loc="lower center", ncol=5, fontsize=10, frameon=True)
    fam_txt = (f", {a.family} routing only" if a.family else "") + (f", {a.mechanism} mechanism only" if a.mechanism else "")
    fig.suptitle(f"Native-prior {a.size} runs — {a.split.upper()} {'excess CE (lower = better; 0 = class prior)' if a.metric == 'excess_ce' else 'accuracy gain'} per checkpoint{' (every 2000 steps, checkpoint evals)' if a.split == 'test' else ''}, regime count (rows) × class condition (columns){fam_txt}; colour = prior family, line style = evaluation bank; horizontal lines = reference models ({a.split})", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    p = out / f"native_{a.size}_{a.metric}_regime_x_classes{('_' + a.family) if a.family else ''}{('_' + a.mechanism) if a.mechanism else ''}{'_test' if a.split == 'test' else ''}.png"; fig.savefig(p, dpi=110); print(p)


if __name__ == "__main__":
    main()
