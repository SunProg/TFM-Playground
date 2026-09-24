"""Per-model-size v4 validation history by num_regimes, one metric per figure.

Input is the cell file from ``extract_v4_validation_cells.py`` (optionally
with ``--prior-baselines`` so the excess / gain metrics exist). For each model
size one figure is written with six panels: rule_mode=multiregime at
num_regimes=1..4, rule_mode=shared (all regimes pooled), and everything
pooled. Every panel shows the five prior families; cells are pooled with an
episode-weighted mean (``episodes`` counts from the extraction), never a mean
of cell means. If a test cell file is given, the final-checkpoint test value
is drawn as a star at the last step of each family.

    python scripts/plot_v4_per_regime_history.py \\
        --history cells_history.json --test cells_test.json \\
        --metric excess_cross_entropy --output-dir figures/per_size_plots
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]
COLORS = {
    "original": "#1f77e6",
    "r_z-fixed": "#ff5a2e",
    "r_z-curriculum": "#12b07a",
    "g_z-fixed": "#f0a000",
    "g_z-curriculum": "#f67ba8",
}
SIZES = ["small", "medium", "large"]
METRIC_LABELS = {
    "query_cross_entropy": ("v4 val query cross entropy", "lower is better"),
    "excess_cross_entropy": ("excess cross entropy (CE − class-prior CE)", "≥0: no better than the class prior"),
    "query_accuracy": ("v4 val query accuracy", "higher is better"),
    "accuracy_gain": ("accuracy gain (acc − majority acc)", "≤0: no better than the majority class"),
    "query_auc": ("v4 val OvR macro AUC", "0.5 = chance"),
}
PANELS = [
    ("multiregime, num_regimes=1", lambda r: r["rule_mode"] == "multiregime" and r["num_regimes"] == 1),
    ("multiregime, num_regimes=2", lambda r: r["rule_mode"] == "multiregime" and r["num_regimes"] == 2),
    ("multiregime, num_regimes=3", lambda r: r["rule_mode"] == "multiregime" and r["num_regimes"] == 3),
    ("multiregime, num_regimes=4", lambda r: r["rule_mode"] == "multiregime" and r["num_regimes"] == 4),
    ("shared (all regimes)", lambda r: r["rule_mode"] == "shared"),
    ("all cells", lambda r: True),
]
PANEL_KEYS = ("mr1", "mr2", "mr3", "mr4", "shared", "all")  # summarize_v4_report_excess.py names
REGIMES, CLASSES = (1, 2, 3, 4), (2, 3, 4, 5)
FEATURE_GROUPS = {"small": (2, 4), "medium": (8, 12), "large": (16, 24)}  # bank num_features levels
TEST_OFFSET = 250  # steps; x-offset of the final-checkpoint test marker past the last training step
GRID_PANELS = [
    (
        f"num_regimes={r}, num_classes={c}",
        (lambda rec, r=r, c=c: rec["rule_mode"] == "multiregime" and rec["num_regimes"] == r and rec["num_classes"] == c),
        f"mr{r}c{c}",
    )
    for r in REGIMES
    for c in CLASSES
]
SUPPORT_PANELS = [
    (f"support_size={s}, multiregime (all regimes)", (lambda r, s=s: r["rule_mode"] == "multiregime" and r["support_size"] == s))
    for s in (64, 128, 256, 512)
]
# (linestyle, color) per reference line; TabPFN family in black, TabICL family in gray
REFERENCE_STYLES = [
    ("-.", "black"), (":", "black"), ((0, (5, 1)), "black"),
    ("-.", "dimgray"), (":", "dimgray"), ((0, (5, 1)), "dimgray"),
]


def weighted_by_step(records: list[dict], metric: str, keep) -> dict[int, float]:
    """Episode-weighted mean of ``metric`` per step over the cells that pass ``keep``."""
    num: dict[int, float] = defaultdict(float)
    den: dict[int, int] = defaultdict(int)
    for r in records:
        if metric not in r or not keep(r):
            continue
        num[r["step"]] += r[metric] * r["episodes"]
        den[r["step"]] += r["episodes"]
    return {s: num[s] / den[s] for s in sorted(den)}


def weighted(records: list[dict], metric: str, keep) -> float | None:
    num = sum(r[metric] * r["episodes"] for r in records if metric in r and keep(r))
    den = sum(r["episodes"] for r in records if metric in r and keep(r))
    return num / den if den else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True)
    parser.add_argument("--test", default=None, help="final-checkpoint test cells (extract --test-output)")
    parser.add_argument(
        "--best-val-test", default=None, help="best-own-val checkpoint test cells (extract --best-val-test-output)"
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="summarize_v4_report_excess.py --json-output file; each entry is drawn as a horizontal reference line",
    )
    parser.add_argument("--metric", default="excess_cross_entropy", choices=sorted(METRIC_LABELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", nargs="+", default=SIZES)
    parser.add_argument("--families", nargs="+", default=None, help=f"families (lines) to draw; default {FAMILIES}")
    parser.add_argument("--title-prefix", default=None, help="text before the size in the figure title")
    parser.add_argument(
        "--panels",
        choices=("regime", "support", "regime_x_classes"),
        default="regime",
        help="regime: 6 panels (multiregime regimes 1-4, shared, all); support: 4 panels by support_size (multiregime); "
        "regime_x_classes: 4x4 grid rows=num_regimes 1-4, cols=num_classes 2-5 (multiregime)",
    )
    parser.add_argument("--mechanism", choices=("r_z", "g_z"), default=None, help="restrict to one mechanism_mode")
    parser.add_argument("--min-classes", type=int, default=None, help="drop cells with num_classes below this (e.g. 3 = no binary)")
    parser.add_argument(
        "--feature-group",
        choices=sorted(FEATURE_GROUPS),
        default=None,
        help="restrict to a num_features group: small={2,4}, medium={8,12}, large={16,24}",
    )
    parser.add_argument("--basename", default=None, help="override the output file stem (default v4_<metric>_per_<panels>)")
    parser.add_argument(
        "--support-size",
        type=int,
        default=None,
        help="restrict every panel to cells with this support_size (default: pool all support sizes)",
    )
    args = parser.parse_args()

    if args.families:
        FAMILIES[:] = args.families
    extra_colors = iter(["#6f42c1", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22"])
    for family in FAMILIES:
        if family not in COLORS:
            COLORS[family] = next(extra_colors)
    history = json.load(open(args.history))
    test = json.load(open(args.test)) if args.test else {}
    best_val = json.load(open(args.best_val_test)) if args.best_val_test else {}
    reference = json.load(open(args.reference)) if args.reference else {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ylabel, note = METRIC_LABELS[args.metric]
    support = args.support_size
    mechanism = args.mechanism
    min_classes = args.min_classes
    feature_set = set(FEATURE_GROUPS[args.feature_group]) if args.feature_group else None
    condition = (
        (f"support_size={support}; " if support is not None else "")
        + (f"num_features in {sorted(feature_set)}; " if feature_set else "")
        + (f"mechanism={mechanism}; " if mechanism else "")
        + (f"num_classes>={min_classes}; " if min_classes else "")
    )
    stem = (
        (f"support{support}_" if support is not None else "")
        + (f"features_{args.feature_group}_" if args.feature_group else "")
        + (f"{mechanism}_" if mechanism else "")
        + (f"classes{min_classes}plus_" if min_classes else "")
    )

    if args.panels == "support":
        panels, panel_keys, layout = SUPPORT_PANELS, ("mr", "mr", "mr", "mr"), (2, 2, (14, 10))
    elif args.panels == "regime_x_classes":
        panels = [(t, k) for t, k, _ in GRID_PANELS]
        panel_keys, layout = [key for _, _, key in GRID_PANELS], (4, 4, (22, 18))
    else:
        panels, panel_keys, layout = PANELS, PANEL_KEYS, (2, 3, (18, 10))

    def restrict(keep):
        return lambda r: (
            keep(r)
            and (support is None or r["support_size"] == support)
            and (mechanism is None or r["mechanism_mode"] == mechanism)
            and (min_classes is None or r["num_classes"] >= min_classes)
            and (feature_set is None or r["num_features"] in feature_set)
        )

    for size in args.sizes:
        fig, axes = plt.subplots(layout[0], layout[1], figsize=layout[2])
        for ax, (title, panel_keep), panel_key in zip(axes.ravel(), panels, panel_keys):
            keep = restrict(panel_keep)
            shown: list[float] = []  # values that set the y-range (everything except the step-0 init point)
            init_value = None
            for family in FAMILIES:
                task = f"{family}-{size}"
                if task not in history:
                    continue
                series = weighted_by_step(history[task], args.metric, keep)
                if not series:
                    continue
                steps, values = list(series), list(series.values())
                ax.plot(steps, values, marker="o", ms=3.5, lw=1.5, color=COLORS[family], label=family)
                shown += [v for st, v in series.items() if st > 0]
                if 0 in series:
                    init_value = series[0]  # identical init for every family of a size
                if task in test:
                    t = weighted(test[task], args.metric, keep)
                    if t is not None:
                        # test markers sit just past the last step so they don't cover the curve's end point
                        ax.plot([steps[-1] + TEST_OFFSET], [t], marker="D", ms=5, color=COLORS[family], mec="black", mew=0.5, ls="none", zorder=5)
                        shown.append(t)
                if task in best_val:
                    t = weighted(best_val[task], args.metric, keep)
                    if t is not None:
                        ax.plot([best_val[task][0]["step"]], [t], marker="s", ms=5, mfc="white", mec=COLORS[family], mew=1.2, ls="none", zorder=5)
                        shown.append(t)
            for (name, values), (style, color) in zip(reference.items(), REFERENCE_STYLES):
                value = values.get(args.metric, {}).get(panel_key)
                if value is not None:
                    ax.axhline(value, color=color, lw=1.2, ls=style, alpha=0.85)
                    shown.append(value)
            if shown:
                lo, hi = min(shown), max(shown)
                if args.metric in ("excess_cross_entropy", "accuracy_gain"):
                    lo, hi = min(lo, 0.0), max(hi, 0.0)
                pad = 0.08 * (hi - lo or 1.0)
                ax.set_ylim(lo - pad, hi + pad)  # the step-0 point is deliberately clipped; its value is annotated
            if init_value is not None:
                ax.annotate(f"step 0 (init): {init_value:+.3f}", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9, color="dimgray")
            if args.metric in ("excess_cross_entropy", "accuracy_gain"):
                ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        stars = [
            plt.Line2D([], [], marker="D", ms=5, color="gray", mec="black", mew=0.5, ls="none", label="test (final checkpoint)"),
            plt.Line2D([], [], marker="s", ms=5, mfc="white", mec="gray", mew=1.2, ls="none", label="test (best own-val checkpoint)"),
        ]
        refs = [
            plt.Line2D([], [], color=color, lw=1.2, ls=style, label=name)
            for name, (style, color) in zip(reference, REFERENCE_STYLES)
        ]
        fig.legend(
            handles + stars + refs, labels + [s.get_label() for s in stars] + [r.get_label() for r in refs],
            loc="lower center", ncol=6,
        )
        by = {"support": "support_size", "regime_x_classes": "num_regimes x num_classes"}.get(args.panels, "num_regimes")
        prefix = f"{args.title_prefix} " if args.title_prefix else ""
        fig.suptitle(f"{prefix}{size} tasks — {ylabel} by {by} ({condition}{note}; episode-weighted)")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        suffix = {"support": "per_support", "regime_x_classes": "regime_x_classes"}.get(args.panels, "per_regime")
        base = args.basename or f"v4_{stem}{args.metric}_{suffix}"
        out = out_dir / f"{base}_{size}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out.resolve())


if __name__ == "__main__":
    main()
