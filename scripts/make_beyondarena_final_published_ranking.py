"""Recompute BeyondArena ranks for final v4 checkpoints plus published baselines."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


METRICS = ("excess_cross_entropy", "macro_ovr_auc", "accuracy_gain")
METRIC_LABELS = {
    "excess_cross_entropy": "Excess CE (lower is better)",
    "macro_ovr_auc": "Macro OVR AUC (higher is better)",
    "accuracy_gain": "Accuracy gain (higher is better)",
}
COLORS = {
    "canonical": "#4c78a8",
    "r_z-fixed": "#f58518",
    "r_z-curriculum": "#ffbf79",
    "g_z-fixed": "#54a24b",
    "g_z-curriculum": "#8cd17d",
    "tabpfn": "#e45756",
    "tabicl": "#b279a2",
}
SIZE_HATCHES = {
    "small": "///",
    "medium": "...",
    "large": "xxx",
    "published": "",
}


def family_group(row: pd.Series) -> str:
    family = str(row["family"])
    if row["model_kind"] == "nanotabpfn" and family in {"r_z", "g_z"}:
        identity = str(row["model_identity"]).lower()
        for branch in ("fixed", "curriculum"):
            if f"/{family}-{branch}-" in identity:
                return f"{family}-{branch}"
    return family


def short_label(row: pd.Series) -> str:
    exposure = " z-expose" if bool(row.get("z_expose", False)) else ""
    if row["model_kind"] != "nanotabpfn":
        return f"{row['model_identity']}{exposure}"
    identity = str(row["model_identity"])
    match = re.search(r"/(\d{8})/([^/]+)/seed-[^/]+$", identity)
    if match:
        run_id, branch = match.groups()
        family_prefix = f"{row['family_group']}-"
        display_branch = branch.removeprefix(family_prefix)
        return f"{row['family_group']}:{display_branch}{exposure}\n{run_id}"
    return f"{row['family_group']}:{row['size']}{exposure}"


def recompute_ranks(rankings: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = rankings[
        ((rankings["model_kind"] == "nanotabpfn") & (rankings["checkpoint_policy"] == "final"))
        | ((rankings["model_kind"].isin(["tabpfn", "tabicl"])) & (rankings["checkpoint_policy"] == "published_default"))
    ].copy()
    if "z_expose" not in selected.columns:
        selected["z_expose"] = False
    selected["family_group"] = selected.apply(family_group, axis=1)
    selected["size_bucket"] = selected["size"].map(
        lambda value: "small" if str(value).startswith("e64-") else
        "medium" if str(value).startswith("e128-") else
        "large" if str(value).startswith("e192-") else "published"
    )
    selected["comparison_set"] = "final_v4_plus_published"

    selected_keys = selected[["model_identity", "checkpoint_policy"]].drop_duplicates()
    valid = fold_metrics[fold_metrics["status"] == "evaluated"].merge(
        selected_keys,
        on=["model_identity", "checkpoint_policy"],
        how="inner",
    )
    rank_sums: defaultdict[tuple[str, str, str, str], float] = defaultdict(float)
    rank_counts: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    for metric in METRICS:
        task_scores = (
            valid.groupby(
                ["condition", "task_name", "model_identity", "checkpoint_policy"],
                as_index=False,
            )[metric]
            .mean()
        )
        for (condition, _task_name), task_group in task_scores.groupby(
            ["condition", "task_name"], sort=False
        ):
            task_ranks = task_group.set_index(["model_identity", "checkpoint_policy"])[metric].rank(
                method="min", ascending=metric == "excess_cross_entropy"
            )
            for (model_identity, checkpoint_policy), task_rank in task_ranks.items():
                key = (condition, metric, model_identity, checkpoint_policy)
                rank_sums[key] += float(task_rank)
                rank_counts[key] += 1

    selected["rank"] = [
        rank_sums[key] / rank_counts[key]
        if rank_counts.get(key, 0)
        else np.nan
        for key in zip(
            selected["condition"],
            selected["metric"],
            selected["model_identity"],
            selected["checkpoint_policy"],
            strict=True,
        )
    ]
    selected["rank_task_count"] = [
        rank_counts.get(key, 0)
        for key in zip(
            selected["condition"],
            selected["metric"],
            selected["model_identity"],
            selected["checkpoint_policy"],
            strict=True,
        )
    ]
    selected["label"] = selected.apply(short_label, axis=1)
    return selected.sort_values(["condition", "metric", "rank", "model_identity"]).reset_index(drop=True)


def write_summary(ranks: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BeyondArena ranking: final v4 plus published models",
        "",
        "Comparison set: all v4 `final` selections plus published TabPFN v2.2/v2.6/v3 and TabICL v1/v2.",
        "Models are ranked independently within each task after fold averaging; "
        "the reported rank is the mean task rank within this comparison set. "
        "Lower mean rank is better.",
        "",
        f"- Models: {ranks.model_identity.nunique()}",
        f"- Conditions: {ranks.condition.nunique()}",
        f"- Ranking rows: {len(ranks)}",
        "",
        "## Best model by condition and metric",
        "",
    ]
    for condition in sorted(ranks.condition.unique()):
        lines.append(f"### `{condition}`")
        lines.append("")
        for metric in METRICS:
            subset = ranks[(ranks.condition == condition) & (ranks.metric == metric)]
            best = subset[subset["rank"] == subset["rank"].min()]
            names = ", ".join(best.model_identity.astype(str))
            best_rank = float(best["rank"].min())
            lines.append(f"- {metric}: `{names}` (mean task rank `{best_rank:.4g}`)")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_plot(ranks: pd.DataFrame, path: Path) -> None:
    data = ranks[ranks.condition == "all"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 14))
    for ax, metric in zip(axes, METRICS):
        subset = data[data.metric == metric].sort_values(["rank", "model_identity"])
        y = np.arange(len(subset))
        for position, (_, row) in zip(y, subset.iterrows()):
            ax.barh(
                position,
                row["rank"],
                color=COLORS.get(row["family_group"], "#777777"),
                hatch=SIZE_HATCHES[row["size_bucket"]],
                edgecolor="#444444",
                linewidth=0.25,
            )
        ax.set_yticks(y, labels=subset["label"], fontsize=7)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xlabel("Mean task rank (1 = best)")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="x", alpha=0.25)
        ax.axvline(1, color="#444444", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        "BeyondArena mean task ranks: final v4 checkpoints plus published TabPFN/TabICL",
        fontsize=16,
        fontweight="bold",
    )
    size_handles = [
        Patch(facecolor="white", edgecolor="#444444", hatch=pattern, label=label)
        for label, pattern in (
            ("v4 small", SIZE_HATCHES["small"]),
            ("v4 medium", SIZE_HATCHES["medium"]),
            ("v4 large", SIZE_HATCHES["large"]),
            ("published baseline", SIZE_HATCHES["published"]),
        )
    ]
    family_handles = [
        Patch(facecolor=color, edgecolor="#444444", label=family)
        for family, color in COLORS.items()
    ]
    fig.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=7, frameon=False)
    fig.legend(handles=size_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings",
        default="results/beyondarena-v4/all-families-37325808/rankings.csv",
    )
    parser.add_argument(
        "--fold-metrics",
        default="results/beyondarena-v4/all-families-37325808/fold_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/beyondarena-v4/all-families-37325808/final_published",
    )
    parser.add_argument("--figures-dir", default="figures/beyondarena")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ranks = recompute_ranks(pd.read_csv(args.rankings), pd.read_csv(args.fold_metrics))
    ranks.to_csv(output_dir / "rankings.csv", index=False)
    write_summary(ranks, output_dir / "ranking_summary.md")
    write_plot(ranks, figures_dir / "beyondarena_final_published_rank_comparison.png")
    print(f"Wrote {len(ranks)} rows to {output_dir}")


if __name__ == "__main__":
    main()
