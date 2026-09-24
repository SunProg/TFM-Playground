"""Plot v4 validation loss/accuracy trajectories conditioned on a specific
(mechanism_mode, rule_mode, support_size, num_features, effective_is_causal)
cell slice.

Consumes the JSON produced by ``extract_v4_validation_cells.py``. For every
(support_size, num_features) combination it writes, per requested
mechanism_mode, per model size, and per effective_is_causal value, a grid of
small multiples (rows = num_regimes, columns = num_classes), each panel
showing one line per prior family traced across training steps. If a
test-set JSON is supplied, the final test-set value for a completed task is
overlaid as a star at its last step.

    python scripts/plot_v4_mechanism_rule_grid.py \\
        --history cells_history.json --test cells_test.json \\
        --outdir figures/v4_mechanism_rule_all_sizes --metric both
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY_COLOR = {
    "original": "#2a78d6",
    "r_z-fixed": "#eb6834",
    "r_z-curriculum": "#1baf7a",
    "g_z-fixed": "#eda100",
    "g_z-curriculum": "#e87ba4",
}
REGIMES = [1, 2, 3, 4]
CLASSES = [2, 3, 4, 5]
METRIC_LABEL = {"query_cross_entropy": "v4 query cross entropy", "query_accuracy": "v4 query accuracy"}


def _cell_key(r: dict) -> tuple:
    return (
        r["mechanism_mode"],
        r["rule_mode"],
        r["support_size"],
        r["num_features"],
        r["effective_is_causal"],
        r["num_regimes"],
        r["num_classes"],
    )


def _index_history(records: dict[str, list[dict]]) -> dict[str, dict]:
    """task -> cell key -> {step: [rows]}"""
    idx: dict[str, dict] = {}
    for task, rows in records.items():
        by_key: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_key[_cell_key(r)][r["step"]].append(r)
        idx[task] = by_key
    return idx


def _index_test(records: dict[str, list[dict]]) -> dict[str, dict]:
    """task -> cell key -> [rows] (test reports have no step)"""
    idx: dict[str, dict] = {}
    for task, rows in records.items():
        by_key: dict = defaultdict(list)
        for r in rows:
            by_key[_cell_key(r)].append(r)
        idx[task] = by_key
    return idx


def plot_one(
    history_idx: dict,
    test_idx: dict | None,
    families: list[str],
    size: str,
    mech: str,
    rule: str,
    support: int,
    nfeat: int,
    causal: bool,
    metric: str,
    outpath: str,
) -> bool:
    fig, axes = plt.subplots(len(REGIMES), len(CLASSES), figsize=(18, 13), sharex=True)
    any_data = False
    for i, regime in enumerate(REGIMES):
        for j, ncls in enumerate(CLASSES):
            ax = axes[i, j]
            for fam in families:
                task = f"{fam}-{size}"
                color = FAMILY_COLOR[fam]
                bystep = history_idx.get(task, {}).get((mech, rule, support, nfeat, causal, regime, ncls), {})
                pts = sorted((step, sum(r[metric] for r in rows) / len(rows)) for step, rows in bystep.items())
                if not pts:
                    continue
                any_data = True
                steps, values = zip(*pts)
                ax.plot(steps, values, color=color, marker="o", markersize=4, linewidth=1.6, label=fam)
                if test_idx is not None:
                    test_rows = test_idx.get(task, {}).get((mech, rule, support, nfeat, causal, regime, ncls))
                    if test_rows:
                        test_val = sum(r[metric] for r in test_rows) / len(test_rows)
                        ax.scatter([steps[-1]], [test_val], color=color, marker="*", s=220,
                                   edgecolor="black", zorder=6)
            if i == 0:
                ax.set_title(f"num_classes={ncls}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"num_regimes={regime}\n{METRIC_LABEL[metric]}", fontsize=9)
            if i == len(REGIMES) - 1:
                ax.set_xlabel("step")
            ax.grid(alpha=0.3)

    if not any_data:
        plt.close(fig)
        return False

    handles, labels = [], []
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    fig.legend(handles, labels, loc="lower center", ncol=len(families), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    star_note = " (★ = final test)" if test_idx is not None else ""
    fig.suptitle(
        f"{size} tasks — {METRIC_LABEL[metric]}{star_note}, mechanism_mode={mech}, rule_mode={rule}, "
        f"support_size={support}, num_features={nfeat}, effective_is_causal={causal}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True, help="Output of extract_v4_validation_cells.py --output")
    parser.add_argument("--test", default=None, help="Output of extract_v4_validation_cells.py --test-output")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--mechanisms", nargs="+", default=["r_z", "g_z"])
    parser.add_argument("--rule-mode", default="multiregime")
    parser.add_argument("--support-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--feature-counts", nargs="+", type=int, default=[2, 4, 8, 12, 16, 24])
    parser.add_argument(
        "--causal-values",
        nargs="+",
        type=lambda v: v.lower() in ("true", "1"),
        default=[True, False],
        metavar="true|false",
        help="effective_is_causal values to condition on (default: both)",
    )
    parser.add_argument("--metric", choices=["loss", "accuracy", "both"], default="both")
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)
    test_idx = None
    if args.test:
        with open(args.test) as f:
            test_idx = _index_test(json.load(f))
    history_idx = _index_history(history)

    metrics = {"loss": ["query_cross_entropy"], "accuracy": ["query_accuracy"], "both": ["query_cross_entropy", "query_accuracy"]}[args.metric]
    families = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]

    count = 0
    for support in args.support_sizes:
        for nfeat in args.feature_counts:
            outdir = os.path.join(args.outdir, f"support_{support}__features_{nfeat}")
            os.makedirs(outdir, exist_ok=True)
            for mech in args.mechanisms:
                for size in args.sizes:
                    for causal in args.causal_values:
                        causal_suffix = "_causal" if causal else "_noncausal"
                        for metric in metrics:
                            suffix = "" if metric == "query_cross_entropy" else "_accuracy"
                            outpath = os.path.join(outdir, f"{mech}_regime_{size}{causal_suffix}{suffix}.png")
                            if plot_one(
                                history_idx, test_idx, families, size, mech, args.rule_mode,
                                support, nfeat, causal, metric, outpath,
                            ):
                                count += 1
    print(f"wrote {count} files under {args.outdir}")


if __name__ == "__main__":
    main()
