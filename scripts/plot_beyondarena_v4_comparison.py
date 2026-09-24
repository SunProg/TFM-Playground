"""BeyondArena v4 comparison figures (rank-based, per-dataset) from a per-task JSON.

Input: ``summarize_beyondarena_v4.py --per-task-json`` output
(``metrics[metric][task][model]`` = fold-mean; ``tasks[task]`` = metadata incl.
``majority`` class fraction and ``classes``/``regime``). For each requested metric:

1. ``cd_<metric>.png``            average-rank / critical-difference diagram (Friedman + Nemenyi, α = 0.05)
2. ``delta_heatmap_<metric>_vs_<ref>.png``  dataset × model Δ vs a reference model (tasks ordered by problem, then
                                    majority class fraction); one file per reference
3. ``pairwise_<metric>.png``      our model vs reference scatter, one point per task, y = x diagonal
3b. ``pairwise_all_<metric>_vs_<ref>.png`` the same scatter for every NanoTabPFN run (rows = size, columns = family, blind and zx-)
4. ``delta_dist_<metric>.png``    distribution of Δ per model (box + points), wins / ties / losses, sign test
5. ``main_table_<metric>.md``     mean / median / average rank per model (+ wins vs each reference)

Δ is always oriented so that positive = the model is better than the reference
(for excess CE: Δ = ref − model).

    python scripts/plot_beyondarena_v4_comparison.py --per-task ba/per_task.json \\
        --panel tabpfn-v3 tabicl-v2 tabpfn-v2.2 tabicl-v1 logreg xgboost original-medium original-large \\
                r_z-fixed-medium r_z-curriculum-large g_z-fixed-medium zx-r_z-fixed-medium \\
        --references tabpfn-v3 original blind --ours r_z-fixed-medium r_z-curriculum-large \\
        --metrics macro_ovr_auc excess_cross_entropy accuracy_gain --output-dir figures/beyondarena_v4/comparison
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HIGHER_BETTER = {"excess_cross_entropy": False, "accuracy_gain": True, "macro_ovr_auc": True}
LABEL = {"excess_cross_entropy": "excess CE", "accuracy_gain": "accuracy gain", "macro_ovr_auc": "macro OvR AUC"}
FAMILIES = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum", "rg_z-fixed", "rg_z-curriculum"]
SIZES = ["small", "medium", "large"]
VARIANTS = ["", "zx", "nat"]  # run-name prefixes: plain old-prior run, z-exposed twin, native-TabICL-prior run
COLORS = {"original": "#1f77e6", "r_z-fixed": "#ff5a2e", "r_z-curriculum": "#12b07a", "g_z-fixed": "#f0a000", "g_z-curriculum": "#f67ba8",
          "rg_z-fixed": "#7b3fe4", "rg_z-curriculum": "#00a3c4"}
PUBLISHED = ("tabpfn", "tabicl")
CONVENTIONAL = ("logreg", "rf", "hgb", "xgboost", "lightgbm", "catboost")


def kind(model: str) -> str:
    if model.startswith(PUBLISHED):
        return "published"
    if model.startswith(CONVENTIONAL):
        return "conventional"
    return "ours"


def parse(model: str):
    """-> (family, size, variant) with variant in VARIANTS ('' = plain old-prior run, 'zx', 'nat'), or None."""
    variant = ""
    n = model
    for v in ("zx", "nat"):
        if model.startswith(v + "-"):
            variant, n = v, model[len(v) + 1 :]
    for f in FAMILIES:
        if n.startswith(f + "-"):
            return f, n[len(f) + 1 :], variant
    return None


def color(model: str) -> str:
    p = parse(model)
    if p:
        return COLORS[p[0]]
    return "black" if kind(model) == "published" else "dimgray"


def size_of(model: str) -> str | None:
    p = parse(model)
    return p[1] if p else None


def ref_for(model: str, reference: str) -> str:
    """'original' means the same-size original run; 'blind' means the z-blind twin of a zx- run;
    anything else is a fixed model name."""
    if reference == "original":
        p = parse(model)
        if not p:
            return reference
        return f"{'nat-' if p[2] == 'nat' else ''}original-{p[1]}"  # the same-size original of the same prior
    if reference == "blind":
        return model[3:] if model.startswith("zx-") else reference
    return reference


# ---------------------------------------------------------------- statistics
def average_ranks(values: dict[str, dict[str, float]], models: list[str], higher_better: bool) -> tuple[dict[str, float], int]:
    """values[task][model] -> mean rank per model over tasks scored by all models (ties get average rank)."""
    ranks = {m: [] for m in models}
    n = 0
    for task, row in values.items():
        if not all(m in row for m in models):
            continue
        n += 1
        vals = [(-row[m] if higher_better else row[m]) for m in models]
        order = np.argsort(vals, kind="mergesort")
        r = np.empty(len(models))
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2 + 1
            i = j + 1
        for m, rk in zip(models, r):
            ranks[m].append(float(rk))
    return {m: mean(v) for m, v in ranks.items()}, n


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    try:
        from scipy.stats import studentized_range

        q = studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    except Exception:  # fallback: Demšar (2006) table, alpha = 0.05
        table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q = table.get(k, 3.164 + 0.06 * (k - 10))
    return q * math.sqrt(k * (k + 1) / (6 * n))


def friedman_p(values: dict[str, dict[str, float]], models: list[str], higher_better: bool) -> float | None:
    try:
        from scipy.stats import friedmanchisquare
    except Exception:
        return None
    cols = {m: [] for m in models}
    for row in values.values():
        if all(m in row for m in models):
            for m in models:
                cols[m].append(row[m] if higher_better else -row[m])
    return float(friedmanchisquare(*[cols[m] for m in models]).pvalue)


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


# ---------------------------------------------------------------- figures
def cd_diagram(avg: dict[str, float], cd: float, n: int, metric: str, p: float | None, out: Path):
    models = sorted(avg, key=avg.get)
    k = len(models)
    half = (k + 1) // 2
    # cliques: maximal groups whose spread <= cd (computed first: the clique bars need vertical room)
    ranks = [avg[m] for m in models]
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and not any(i >= a and j <= b for a, b in cliques):
            cliques.append((i, j))
        i += 1
    levels = max(1, len(cliques))
    bar_zone = 0.12 * levels + 0.1  # vertical space under the axis used by the clique bars
    fig_h = 2.2 + 0.32 * half + 0.25 * levels
    margin = 2.5 + 0.12 * k  # label margins grow with the number of models (labels are in axis units)
    fig, ax = plt.subplots(figsize=(max(11, 0.5 * k + 6), fig_h))
    ax.set_xlim(k + 1 + margin, -margin)  # reversed: rank 1 on the right; margins hold the labels
    ax.set_ylim(-bar_zone - 0.6 - 0.5 * half - 0.3, 1.9)
    ax.axis("off")
    ax.plot([0.5, k + 0.5], [0, 0], color="black", lw=1)
    for i in range(1, k + 1):
        ax.plot([i, i], [0, 0.15], color="black", lw=1)
        ax.text(i, 0.25, str(i), ha="center", va="bottom", fontsize=8)
    ax.plot([k + 0.4, k + 0.4 - cd], [1.2, 1.2], color="black", lw=2)
    ax.text(k + 0.4 - cd / 2, 1.35, f"CD = {cd:.2f} (Nemenyi, α = 0.05, N = {n})", ha="center", fontsize=8)
    for idx, m in enumerate(models):
        r = avg[m]
        best_side = idx < half  # best-ranked half of the models: labels on the right (rank 1 is at the right end)
        y = -bar_zone - 0.6 - 0.5 * (idx if best_side else idx - half)
        xt = 0.45 if best_side else k + 0.55
        ax.plot([r, r], [0, y], color=color(m), lw=1)
        ax.plot([r, xt], [y, y], color=color(m), lw=1)
        ax.text(xt - 0.08 if best_side else xt + 0.08, y, f"{m} ({r:.2f})", ha="left" if best_side else "right",
                va="center", fontsize=8, color=color(m))
    for level, (a, b) in enumerate(cliques):  # one row per clique so overlapping bars stay distinguishable
        y = 0.07 + 0.12 * level
        ax.plot([ranks[a] + 0.08, ranks[b] - 0.08], [-y, -y], color="black", lw=3, solid_capstyle="butt")
    ptxt = f", Friedman p = {p:.2e}" if p is not None else ""
    ax.set_title(f"BeyondArena v4 — average rank, {LABEL[metric]} (1 = best; bars join models within one CD{ptxt})", fontsize=10)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def task_label(t: str, tasks_meta) -> str:
    """task (b|m, regime, maj x.xx, n=<train rows> x d=<features>)."""
    mt = tasks_meta[t]
    return f"{t} ({mt['classes'][0]}, {mt['regime']}, maj {mt['majority']:.2f}, n={mt['train_rows']} d={mt['features']})"


def delta_heatmap(values, tasks_meta, models: list[str], reference: str, metric: str, out: Path):
    hb = HIGHER_BETTER[metric]
    tasks = [t for t in values if all(ref_for(m, reference) in values[t] and m in values[t] for m in models)]
    # binary then multiclass; within a block IID first, Grouped/Temporal last, each by majority fraction
    tasks.sort(key=lambda t: (tasks_meta[t]["classes"], tasks_meta[t]["regime"] != "IID", -(tasks_meta[t]["majority"] or 0)))
    mat = np.array([[((values[t][m] - values[t][ref_for(m, reference)]) if hb else (values[t][ref_for(m, reference)] - values[t][m])) for m in models] for t in tasks])
    vmax = float(np.nanpercentile(np.abs(mat), 95)) or 1e-3
    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 4, 0.28 * len(tasks) + 2))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(models)):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6, color="black" if abs(v) < 0.6 * vmax else "white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([task_label(t, tasks_meta) for t in tasks], fontsize=7)
    for lab in ax.get_yticklabels():  # non-IID (Grouped / Temporal) rows in bold
        if ", IID," not in lab.get_text():
            lab.set_fontweight("bold")
    nb = sum(1 for t in tasks if tasks_meta[t]["classes"] == "binary")
    ax.axhline(nb - 0.5, color="black", lw=1)
    for start, block in ((0, tasks[:nb]), (nb, tasks[nb:])):  # dashed line before the non-IID tasks of each block
        n_iid = sum(1 for t in block if tasks_meta[t]["regime"] == "IID")
        if 0 < n_iid < len(block):
            ax.axhline(start + n_iid - 0.5, color="black", lw=0.8, ls="--")
    refname = {"original": "same-size original", "blind": "z-blind twin (same family & size)"}.get(reference, reference)
    fig.colorbar(im, ax=ax, shrink=0.6, label=f"Δ {LABEL[metric]} vs {refname} (positive = better than reference)")
    ax.set_title(f"BeyondArena v4 — per-dataset Δ {LABEL[metric]} vs {refname} (final checkpoints; row label: classes, regime, majority fraction, n = train rows, d = features)", fontsize=10)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _pairwise_axis(ax, values, tasks_meta, m: str, r: str, metric: str, compact: bool = False) -> None:
    """One model-vs-reference scatter (one point per task, diagonal = tie)."""
    pts = [(values[t][r], values[t][m], t) for t in values if r in values[t] and m in values[t]]
    if not pts:
        ax.set_axis_off()
        return
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    for x, y, t in pts:
        mk = "o" if tasks_meta[t]["classes"] == "binary" else "s"
        maj = tasks_meta[t]["majority"] or 0.5
        ax.scatter(x, y, marker=mk, s=(12 + 60 * (maj - 0.4)) if compact else (25 + 120 * (maj - 0.4)), color=color(m),
                   edgecolors="black", linewidths=0.4, alpha=0.85, zorder=3)
        if not compact and abs(y - x) > 0.05 * (max(abs(x), abs(y), 0.05)) + 0.02:
            ax.annotate(f"{t[:22]} (n={tasks_meta[t]['train_rows']}, d={tasks_meta[t]['features']})", (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    lo, hi = min(xs + ys), max(xs + ys)
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", lw=0.8, ls="--")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    wins = sum(((y > x) if HIGHER_BETTER[metric] else (y < x)) for x, y, _ in pts)
    side = "above" if HIGHER_BETTER[metric] else "below"
    if compact:
        ax.set_title(f"{m}\nbetter on {wins}/{len(pts)} ({side})", fontsize=7.5)
        ax.tick_params(labelsize=6)
    else:
        ax.set_xlabel(f"{r} — {LABEL[metric]}")
        ax.set_ylabel(f"{m} — {LABEL[metric]}")
        ax.set_title(f"{m} vs {r}: {m.split('-')[0]} better on {wins}/{len(pts)} tasks ({side} diagonal)", fontsize=9)
    ax.grid(alpha=0.3)


def pairwise(values, tasks_meta, ours: list[str], references: list[str], metric: str, out: Path):
    fig, axes = plt.subplots(len(references), len(ours), figsize=(5.2 * len(ours), 5 * len(references)), squeeze=False)
    for i, ref in enumerate(references):
        for j, m in enumerate(ours):
            _pairwise_axis(axes[i][j], values, tasks_meta, m, ref_for(m, ref), metric)
    fig.suptitle(f"BeyondArena v4 — pairwise per-dataset comparison, {LABEL[metric]} (circle = binary, square = multiclass; marker size ∝ majority class fraction)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def pairwise_all(values, tasks_meta, reference: str, metric: str, out: Path):
    """Every NanoTabPFN run vs one reference: rows = model size, columns = family (blind and zx-)."""
    all_models = sorted({m for t in values for m in values[t]})
    runs = [m for m in all_models if parse(m) and ref_for(m, reference) != m and ref_for(m, reference) in all_models]
    cols = sorted({(parse(m)[0], parse(m)[2]) for m in runs}, key=lambda c: (VARIANTS.index(c[1]) == 2, FAMILIES.index(c[0]), c[1]))
    rows = [sz for sz in SIZES if any(size_of(m) == sz for m in runs)]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.1 * len(cols), 3.2 * len(rows)), squeeze=False)
    for i, sz in enumerate(rows):
        for j, (fam, variant) in enumerate(cols):
            m = f"{variant + '-' if variant else ''}{fam}-{sz}"
            ax = axes[i][j]
            if m not in runs:
                ax.set_axis_off()
                continue
            _pairwise_axis(ax, values, tasks_meta, m, ref_for(m, reference), metric, compact=True)
            if j == 0:
                ax.set_ylabel(f"{sz}\nmodel {LABEL[metric]}", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel(f"{ref_for(m, reference)} {LABEL[metric]}", fontsize=7)
    refname = {"original": "same-size original", "blind": "z-blind twin"}.get(reference, reference)
    fig.suptitle(f"BeyondArena v4 — every run vs {refname}, {LABEL[metric]} (x = reference, y = model; circle = binary, square = multiclass; "
                 f"{'below' if not HIGHER_BETTER[metric] else 'above'} the diagonal = model better)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def delta_distribution(values, models: list[str], references: list[str], metric: str, out: Path) -> dict:
    hb = HIGHER_BETTER[metric]
    stats = {}
    fig, axes = plt.subplots(len(references), 1, figsize=(1.0 * len(models) + 3, 4.2 * len(references)), squeeze=False)
    for i, ref in enumerate(references):
        ax = axes[i][0]
        data, labels = [], []
        for m in models:
            r = ref_for(m, ref)
            if r == m:
                continue
            d = [((values[t][m] - values[t][r]) if hb else (values[t][r] - values[t][m])) for t in values if r in values[t] and m in values[t]]
            if not d:
                continue
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            stats[(m, ref)] = {"median": median(d), "mean": mean(d), "wins": w, "losses": l, "ties": len(d) - w - l, "n": len(d), "p_sign": sign_test_p(w, l)}
            data.append(d); labels.append(m)
        pos = range(1, len(data) + 1)
        ax.boxplot(data, positions=list(pos), widths=0.55, showfliers=False, medianprops={"color": "black"})
        for x, d, m in zip(pos, data, labels):
            ax.scatter(np.random.default_rng(0).normal(x, 0.06, len(d)), d, s=12, color=color(m), alpha=0.7, zorder=3)
            st = stats[(m, ref)]
            ax.text(x, ax.get_ylim()[1] if False else max(d) + 0.02 * (max(map(max, data)) - min(map(min, data)) or 1),
                    f"med {st['median']:+.3f}\n{st['wins']}/{st['n']} wins\np={st['p_sign']:.2g}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(list(pos)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
        refname = "same-size original" if ref == "original" else ref
        ax.set_ylabel(f"Δ {LABEL[metric]} vs {refname}\n(positive = model better)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"BeyondArena v4 — per-dataset Δ {LABEL[metric]} distribution (box = quartiles, points = tasks; sign test p)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return stats


def main_table(values, models: list[str], avg: dict[str, float], stats: dict, references: list[str], metric: str, n: int) -> str:
    rows = []
    for m in models:
        v = [values[t][m] for t in values if m in values[t]]
        cells = [m, f"{mean(v):+.4f}" if metric != "macro_ovr_auc" else f"{mean(v):.4f}",
                 f"{median(v):+.4f}" if metric != "macro_ovr_auc" else f"{median(v):.4f}", f"{avg[m]:.2f}"]
        for ref in references:
            st = stats.get((m, ref))
            cells.append(f"{st['wins']}/{st['n']} (p={st['p_sign']:.2g})" if st else "–")
        rows.append(cells)
    rows.sort(key=lambda c: float(c[3]))
    hdr = ["model", f"mean {LABEL[metric]}", "median", f"avg rank (N={n})"] + [f"wins vs {('same-size original' if r == 'original' else r)}" for r in references]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(c) + " |" for c in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-task", required=True)
    parser.add_argument("--panel", nargs="+", required=True, help="models for the CD diagram, heatmaps, distributions and table")
    parser.add_argument("--references", nargs="+", default=["tabpfn-v3", "original"], help="reference models; 'original' = same-size original run")
    parser.add_argument("--ours", nargs="+", default=["r_z-fixed-medium", "r_z-curriculum-large"], help="models for the pairwise scatter")
    parser.add_argument("--metrics", nargs="+", default=["macro_ovr_auc", "excess_cross_entropy", "accuracy_gain"])
    parser.add_argument("--exclude-regime", nargs="*", default=["Temporal"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="", help="suffix for output names (e.g. _all32)")
    args = parser.parse_args()
    data = json.load(open(args.per_task))
    tasks_meta = {t: md for t, md in data["tasks"].items() if md["regime"] not in args.exclude_regime}
    out = Path(args.output_dir)
    # one subdirectory per figure kind, created on demand
    def sub(kind: str) -> Path:
        directory = out / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    out.mkdir(parents=True, exist_ok=True)
    panel = [m for m in args.panel if all(m in data["metrics"][mt].get(t, {}) for mt in args.metrics for t in tasks_meta)]
    missing = [m for m in args.panel if m not in panel]
    if missing:
        print("dropped (not scored on every task):", missing)
    for metric in args.metrics:
        values = {t: data["metrics"][metric][t] for t in tasks_meta}
        avg, n = average_ranks(values, panel, HIGHER_BETTER[metric])
        cd = nemenyi_cd(len(panel), n)
        p = friedman_p(values, panel, HIGHER_BETTER[metric])
        cd_diagram(avg, cd, n, metric, p, sub("cd") / f"cd_{metric}{args.tag}.png")
        for ref in args.references:
            all_models = sorted({m for t in values for m in values[t]})
            if ref == "original":
                # pretraining-effect figure: every multiregime run (blind + zx, all sizes) vs its same-size original
                cols = [m for m in all_models if parse(m) and parse(m)[0] != "original" and f"original-{size_of(m)}" in all_models]
                cols.sort(key=lambda m: (VARIANTS.index(parse(m)[2]) == 2, SIZES.index(parse(m)[1]), FAMILIES.index(parse(m)[0]), parse(m)[2]))
            elif ref == "blind":
                # z-exposure effect: every zx- run vs the same family/size trained z-blind
                cols = [m for m in all_models if m.startswith("zx-") and m[3:] in all_models]
                cols.sort(key=lambda m: (SIZES.index(parse(m)[1]), FAMILIES.index(parse(m)[0])))
            elif kind(ref) == "conventional":
                # vs a conventional ML baseline: every NanoTabPFN run plus the published models
                cols = [m for m in all_models if (parse(m) or kind(m) == "published") and m != ref]
                cols.sort(key=lambda m: (VARIANTS.index(parse(m)[2]) == 2, SIZES.index(parse(m)[1]), FAMILIES.index(parse(m)[0]), parse(m)[2]) if parse(m) else (True, 9, 0, "", m))
            else:
                cols = [m for m in panel if ref_for(m, ref) != m and ref_for(m, ref) in values[next(iter(values))]]
            delta_heatmap(values, tasks_meta, cols, ref, metric, sub("delta_heatmap") / f"delta_heatmap_{metric}_vs_{ref}{args.tag}.png")
        ours = [m for m in args.ours if m in panel]
        for ref in args.references:
            pairwise_all(values, tasks_meta, ref, metric, sub("pairwise") / f"pairwise_all_{metric}_vs_{ref}{args.tag}.png")
        refs = [r for r in args.references if r != "blind"]  # 'blind' only makes sense for the zx- figures
        pairwise(values, tasks_meta, ours, refs, metric, sub("pairwise") / f"pairwise_{metric}{args.tag}.png")
        stats = delta_distribution(values, [m for m in panel if kind(m) == "ours"], refs, metric, sub("delta_dist") / f"delta_dist_{metric}{args.tag}.png")
        table = main_table(values, panel, avg, stats, refs, metric, n)
        (sub("tables") / f"main_table_{metric}{args.tag}.md").write_text(f"### {LABEL[metric]} — {len(panel)} models, {n} tasks, CD = {cd:.2f}\n\n{table}\n")
        print(f"{metric}: N={n}, k={len(panel)}, CD={cd:.2f}, Friedman p={p}")
        print(table)


if __name__ == "__main__":
    main()
