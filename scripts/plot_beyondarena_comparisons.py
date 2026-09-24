"""Create comparison plots from the BeyondArena ranking table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


METRIC_LABELS = {
    "excess_cross_entropy": "Excess cross-entropy (lower is better)",
    "macro_ovr_auc": "Macro one-vs-rest AUC (higher is better)",
    "accuracy_gain": "Accuracy gain (higher is better)",
}
METRICS = tuple(METRIC_LABELS)
CONDITIONS = (
    "all",
    "IID",
    "Temporal",
    "Grouped",
    "small_rows",
    "medium_rows",
    "large_rows",
    "small_feat",
    "medium_feat",
    "large_feat",
)
FAMILY_COLORS = {
    "canonical": "#4c78a8",
    "r_z": "#f58518",
    "g_z": "#54a24b",
    "tabpfn": "#e45756",
    "tabicl": "#b279a2",
}


def short_label(row: pd.Series) -> str:
    exposure = " z-expose" if bool(row.get("z_expose", False)) else ""
    if row["model_kind"] != "nanotabpfn":
        return f"{row['model_identity']}{exposure}"
    identity = str(row["model_identity"])
    match = re.search(r"/(\d{8})/([^/]+)/seed-[^/]+$", identity)
    if match:
        run_id, branch = match.groups()
        return f"{row['family']}:{branch}{exposure}\n{run_id} {row['checkpoint_policy']}"
    return f"{row['family']}:{row['size']}{exposure}\n{row['checkpoint_policy']}"


def family_legend() -> list[Patch]:
    return [Patch(facecolor=color, edgecolor="none", label=family) for family, color in FAMILY_COLORS.items()]


def direction(metric: str) -> bool:
    """Return True when lower values are better."""

    return metric == "excess_cross_entropy"


def recompute_subset_ranks(rankings: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Recompute mean task ranks within exactly the models being plotted."""

    subset = rankings.copy()
    keys = subset[["model_identity", "checkpoint_policy"]].drop_duplicates()
    valid = fold_metrics[fold_metrics["status"] == "evaluated"].merge(
        keys,
        on=["model_identity", "checkpoint_policy"],
        how="inner",
    )
    rank_sums: dict[tuple[str, str, str, str], float] = {}
    rank_counts: dict[tuple[str, str, str, str], int] = {}
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
                method="min", ascending=direction(metric)
            )
            for (model_identity, checkpoint_policy), task_rank in task_ranks.items():
                key = (condition, metric, model_identity, checkpoint_policy)
                rank_sums[key] = rank_sums.get(key, 0.0) + float(task_rank)
                rank_counts[key] = rank_counts.get(key, 0) + 1

    keys_in_order = zip(
        subset["condition"],
        subset["metric"],
        subset["model_identity"],
        subset["checkpoint_policy"],
        strict=True,
    )
    subset["rank"] = [
        rank_sums[key] / rank_counts[key] if rank_counts.get(key, 0) else np.nan
        for key in keys_in_order
    ]
    keys_in_order = zip(
        subset["condition"],
        subset["metric"],
        subset["model_identity"],
        subset["checkpoint_policy"],
        strict=True,
    )
    subset["rank_task_count"] = [rank_counts.get(key, 0) for key in keys_in_order]
    return subset


def prepare(rankings: pd.DataFrame) -> pd.DataFrame:
    rankings = rankings.copy()
    if "z_expose" not in rankings.columns:
        rankings["z_expose"] = False
    rankings["label"] = rankings.apply(short_label, axis=1)
    rankings["family"] = rankings["family"].replace({"mr_only": "mr-only"})
    return rankings


def v4_family_group(row: pd.Series) -> str:
    """Split r_z/g_z branches while keeping other families unchanged."""

    family = str(row["family"])
    if family in {"r_z", "g_z"}:
        identity = str(row["model_identity"]).lower()
        for branch in ("fixed", "curriculum"):
            if f"/{family}-{branch}-" in identity:
                return f"{family}-{branch}"
    return family


def plot_overall(rankings: pd.DataFrame, output: Path) -> None:
    data = rankings[rankings["condition"] == "all"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 18), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        subset = data[data["metric"] == metric].sort_values("score", ascending=direction(metric))
        y = np.arange(len(subset))
        colors = [FAMILY_COLORS.get(family, "#777777") for family in subset["family"]]
        ax.barh(y, subset["score"], color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        if metric == "excess_cross_entropy":
            ax.axvline(0, color="#444444", linewidth=0.8)
        for tick, identity in zip(ax.get_yticklabels(), subset["model_identity"]):
            if identity in {"tabicl-v2", "tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3"}:
                tick.set_fontweight("bold")
    fig.suptitle("BeyondArena overall comparison", fontsize=16, fontweight="bold")
    fig.legend(handles=family_legend(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overall_ranks(rankings: pd.DataFrame, output: Path) -> None:
    """Compare mean per-task rank positions; rank 1 is best in every panel."""

    data = rankings[rankings["condition"] == "all"].copy()
    pivot = data.pivot(index=["model_identity", "family", "label"], columns="metric", values="rank")
    pivot["mean_rank"] = pivot[list(METRICS)].mean(axis=1)
    pivot = pivot.sort_values("mean_rank")
    labels = pivot.index.get_level_values("label")
    families = pivot.index.get_level_values("family")
    fig, axes = plt.subplots(1, 3, figsize=(19, 18), constrained_layout=True)
    y = np.arange(len(pivot))
    for ax, metric in zip(axes, METRICS):
        values = pivot[metric].to_numpy()
        colors = [FAMILY_COLORS.get(family, "#777777") for family in families]
        ax.barh(y, values, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xlabel("Mean task rank (1 = best)")
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.axvline(1, color="#444444", linewidth=0.8)
    fig.suptitle("BeyondArena overall mean task-rank comparison", fontsize=16, fontweight="bold")
    fig.legend(handles=family_legend(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_rank_family_size(
    rankings: pd.DataFrame, fold_metrics: pd.DataFrame, output: Path
) -> None:
    """Show mean rank by checkpoint policy, family, and size bucket."""

    policies = ("final", "best_own_val")
    data = pd.concat(
        [
            recompute_subset_ranks(
                rankings[
                    (rankings["model_kind"] == "nanotabpfn")
                    & (rankings["checkpoint_policy"] == policy)
                ],
                fold_metrics,
            )
            for policy in policies
        ],
        ignore_index=True,
    )
    data["size_bucket"] = data["size"].map(
        lambda value: "small" if str(value).startswith("e64-") else
        "medium" if str(value).startswith("e128-") else
        "large" if str(value).startswith("e192-") else np.nan
    )
    data = data.dropna(subset=["size_bucket"])
    data["family_group"] = data.apply(v4_family_group, axis=1)
    grouped = (
        data.groupby(["checkpoint_policy", "family_group", "size_bucket", "metric"], as_index=False, dropna=False)["rank"]
        .mean()
        .rename(columns={"rank": "mean_rank"})
    )
    families = [
        family
        for family in ("canonical", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum", "mr-only")
        if family in set(grouped["family_group"])
    ]
    sizes = ["small", "medium", "large"]
    all_values = grouped["mean_rank"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row, policy in enumerate(policies):
        for column, metric in enumerate(METRICS):
            ax = axes[row, column]
            table = grouped[
                (grouped["checkpoint_policy"] == policy) & (grouped["metric"] == metric)
            ].pivot(index="family_group", columns="size_bucket", values="mean_rank")
            table = table.reindex(index=families, columns=sizes)
            values = table.to_numpy(dtype=float)
            image = ax.imshow(
                values,
                aspect="auto",
                cmap="RdYlGn_r",
                vmin=float(np.nanmin(all_values)),
                vmax=float(np.nanmax(all_values)),
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(len(sizes)), labels=sizes, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(families)), labels=families)
            ax.set_xlabel("Model size bucket")
            ax.set_title(f"{policy}: {METRIC_LABELS[metric]}", fontsize=10)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Mean rank (1 = best)")
    fig.suptitle(
        "BeyondArena v4 mean rank by checkpoint, family, and model size\n"
        "Mean task rank across all conditions; rank 1 = best",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_grid_mean_ranks(
    rankings: pd.DataFrame, fold_metrics: pd.DataFrame, output_dir: Path
) -> None:
    """Split family x size mean ranks by problem type and regime."""

    policies = ("final", "best_own_val")
    task_grid = (
        ("binary__IID", "Binary × IID"),
        ("binary__Grouped", "Binary × Grouped"),
        ("multi__IID", "Multi × IID"),
        ("multi__Grouped", "Multi × Grouped"),
    )
    data = pd.concat(
        [
            recompute_subset_ranks(
                rankings[
                    (rankings["model_kind"] == "nanotabpfn")
                    & (rankings["checkpoint_policy"] == policy)
                    & rankings["condition"].isin([condition for condition, _ in task_grid])
                ],
                fold_metrics,
            )
            for policy in policies
        ],
        ignore_index=True,
    )
    data["size_bucket"] = data["size"].map(
        lambda value: "small" if str(value).startswith("e64-") else
        "medium" if str(value).startswith("e128-") else
        "large" if str(value).startswith("e192-") else np.nan
    )
    data = data.dropna(subset=["size_bucket"])
    data["family_group"] = data.apply(v4_family_group, axis=1)
    grouped = (
        data.groupby(
            ["checkpoint_policy", "condition", "family_group", "size_bucket", "metric"],
            as_index=False,
            dropna=False,
        )["rank"]
        .mean()
        .rename(columns={"rank": "mean_rank"})
    )
    families = [
        family
        for family in ("canonical", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum", "mr-only")
        if family in set(grouped["family_group"])
    ]
    sizes = ["small", "medium", "large"]
    for policy in policies:
        policy_data = grouped[grouped["checkpoint_policy"] == policy]
        if policy_data.empty:
            continue
        vmin = float(policy_data["mean_rank"].min())
        vmax = float(policy_data["mean_rank"].max())
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
        for row, metric in enumerate(METRICS):
            row_image = None
            for column, (condition, condition_label) in enumerate(task_grid):
                ax = axes[row, column]
                table = policy_data[
                    (policy_data["condition"] == condition) & (policy_data["metric"] == metric)
                ].pivot(index="family_group", columns="size_bucket", values="mean_rank")
                table = table.reindex(index=families, columns=sizes)
                values = table.to_numpy(dtype=float)
                row_image = ax.imshow(
                    values,
                    aspect="auto",
                    cmap="RdYlGn_r",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks(np.arange(len(sizes)), labels=sizes, rotation=35, ha="right")
                ax.set_yticks(np.arange(len(families)), labels=families)
                ax.set_xlabel("Size")
                ax.set_title(condition_label, fontsize=10)
                if column == 0:
                    ax.set_ylabel(METRIC_LABELS[metric])
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        if np.isfinite(values[i, j]):
                            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8)
            if row_image is not None:
                fig.colorbar(row_image, ax=axes[row, :], fraction=0.012, pad=0.01, label="Mean rank")
        fig.suptitle(
            f"BeyondArena v4 mean rank by task grid — {policy}\n"
            "Family × size; rank 1 = best; mean task rank across model identities in each grid",
            fontsize=15,
            fontweight="bold",
        )
        fig.savefig(output_dir / f"beyondarena_mean_rank_task_grid_{policy}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_task_grid_all_models(
    rankings: pd.DataFrame, fold_metrics: pd.DataFrame, output: Path
) -> None:
    """Compare final v4 groups with published TabPFN/TabICL baselines."""

    task_grid = (
        ("binary__IID", "Binary × IID"),
        ("binary__Grouped", "Binary × Grouped"),
        ("multi__IID", "Multi × IID"),
        ("multi__Grouped", "Multi × Grouped"),
    )
    v4 = rankings[
        (rankings["model_kind"] == "nanotabpfn") & (rankings["checkpoint_policy"] == "final")
    ].copy()
    v4["size_bucket"] = v4["size"].map(
        lambda value: "small" if str(value).startswith("e64-") else
        "medium" if str(value).startswith("e128-") else
        "large" if str(value).startswith("e192-") else np.nan
    )
    v4 = v4.dropna(subset=["size_bucket"])
    v4["family_group"] = v4.apply(v4_family_group, axis=1)
    v4["model_group"] = v4["family_group"].astype(str) + " / " + v4["size_bucket"]
    baselines = rankings[
        (rankings["model_kind"].isin(["tabpfn", "tabicl"]))
        & (rankings["checkpoint_policy"] == "published_default")
    ].copy()
    baselines["model_group"] = baselines["model_identity"]
    data = recompute_subset_ranks(pd.concat([v4, baselines], ignore_index=True), fold_metrics)
    data = data[data["condition"].isin([condition for condition, _ in task_grid])]
    grouped = (
        data.groupby(["model_group", "condition", "metric"], as_index=False, dropna=False)["rank"]
        .mean()
        .rename(columns={"rank": "mean_rank"})
    )
    v4_order = [
        f"{family} / {size}"
        for family in ("canonical", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum", "mr-only")
        for size in ("small", "medium", "large")
        if f"{family} / {size}" in set(grouped["model_group"])
    ]
    baseline_order = [
        name for name in ("tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3", "tabicl-v1", "tabicl-v2")
        if name in set(grouped["model_group"])
    ]
    model_order = v4_order + baseline_order
    fig, axes = plt.subplots(1, 3, figsize=(18, 13), constrained_layout=True)
    vmin = float(grouped["mean_rank"].min())
    vmax = float(grouped["mean_rank"].max())
    for ax, metric in zip(axes, METRICS):
        table = grouped[grouped["metric"] == metric].pivot(index="model_group", columns="condition", values="mean_rank")
        table = table.reindex(index=model_order, columns=[condition for condition, _ in task_grid])
        values = table.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(len(task_grid)), labels=[label for _, label in task_grid], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(model_order)), labels=model_order)
        ax.set_xlabel("Task grid")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Mean rank")
    fig.suptitle(
        "BeyondArena mean rank across task grids\n"
        "Final v4 groups versus published TabPFN/TabICL baselines; rank 1 = best",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_by_family(rankings: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:
    data = rankings[rankings["condition"].isin(conditions)].copy()
    rows: list[pd.Series] = []
    for (condition, metric, family), group in data.groupby(["condition", "metric", "family"], sort=False):
        index = group["score"].idxmin() if direction(metric) else group["score"].idxmax()
        rows.append(group.loc[index])
    return pd.DataFrame(rows)


def plot_family_heatmaps(rankings: pd.DataFrame, output: Path) -> None:
    conditions = tuple(condition for condition in CONDITIONS if condition in set(rankings["condition"]))
    data = best_by_family(rankings, conditions)
    families = [family for family in FAMILY_COLORS if family in set(data["family"])]
    fig, axes = plt.subplots(3, 1, figsize=(17, 12), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        table = data[data["metric"] == metric].pivot(index="family", columns="condition", values="score")
        table = table.reindex(index=families, columns=conditions)
        values = table.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(np.arange(len(conditions)), labels=conditions, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(families)), labels=families)
        ax.set_title(f"Best selection within each family: {METRIC_LABELS[metric]}", fontsize=11)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.suptitle("BeyondArena family comparison across conditions", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_baselines(rankings: pd.DataFrame, output: Path) -> None:
    conditions = tuple(condition for condition in CONDITIONS if condition in {"all", "IID", "Temporal", "Grouped"})
    data = rankings[rankings["condition"].isin(conditions)].copy()
    external = data[data["model_kind"].isin(["tabpfn", "tabicl"])].copy()
    v4 = data[data["model_kind"] == "nanotabpfn"].copy()
    rows: list[pd.Series] = []
    for (condition, metric), group in v4.groupby(["condition", "metric"], sort=False):
        index = group["score"].idxmin() if direction(metric) else group["score"].idxmax()
        row = group.loc[index].copy()
        row["model_identity"] = "Best v4"
        row["family"] = "v4"
        rows.append(row)
    best_v4 = pd.DataFrame(rows)
    plot_data = pd.concat([external, best_v4], ignore_index=True)
    models = ["Best v4", "tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3", "tabicl-v1", "tabicl-v2"]
    model_colors = {
        "Best v4": "#4c78a8",
        "tabpfn-v2.2": "#e45756",
        "tabpfn-v2.6": "#f58518",
        "tabpfn-v3": "#ff9da6",
        "tabicl-v1": "#b279a2",
        "tabicl-v2": "#79706e",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    x = np.arange(len(conditions))
    width = 0.13
    for ax, metric in zip(axes, METRICS):
        for offset, model in enumerate(models):
            subset = plot_data[(plot_data["metric"] == metric) & (plot_data["model_identity"] == model)].set_index("condition")
            values = [subset["score"].get(condition, np.nan) for condition in conditions]
            ax.bar(x + (offset - (len(models) - 1) / 2) * width, values, width, label=model, color=model_colors[model])
        ax.set_xticks(x, conditions)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if metric == "excess_cross_entropy":
            ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_ylabel("Score")
    fig.suptitle("Published baselines versus the best v4 selection", fontsize=16, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rankings",
        default="results/beyondarena-v4/all-families-37325808/rankings.csv",
        help="Path to rankings.csv",
    )
    parser.add_argument(
        "--fold-metrics",
        default="results/beyondarena-v4/all-families-37325808/fold_metrics.csv",
        help="Path to fold_metrics.csv used for subset-specific task ranks",
    )
    parser.add_argument("--output-dir", default="figures/beyondarena")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = prepare(pd.read_csv(args.rankings))
    fold_metrics = pd.read_csv(args.fold_metrics)
    plot_overall(rankings, output_dir / "beyondarena_overall_comparison.png")
    plot_overall_ranks(rankings, output_dir / "beyondarena_rank_comparison.png")
    plot_mean_rank_family_size(rankings, fold_metrics, output_dir / "beyondarena_mean_rank_family_size.png")
    plot_task_grid_mean_ranks(rankings, fold_metrics, output_dir)
    plot_task_grid_all_models(rankings, fold_metrics, output_dir / "beyondarena_mean_rank_task_grid_all_models.png")
    plot_family_heatmaps(rankings, output_dir / "beyondarena_family_heatmaps.png")
    plot_baselines(rankings, output_dir / "beyondarena_baseline_comparison.png")
    print(f"Wrote plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
