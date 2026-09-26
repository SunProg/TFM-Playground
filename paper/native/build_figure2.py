"""Build Figure 2: paired multiregime effects against matched native controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_REPORT_ROOT = ROOT / "artifacts/cluster_sync/tfm_eval/native_regime_split/blind"
SIZES = (("small", 2), ("medium", 4), ("large", 6))
SCHEDULES = (
    ("rg_z-fixed", "Fixed mixture", "#cc7722", "o", -0.12),
    ("rg_z-curriculum", "Curriculum", "#167f86", "s", 0.12),
)
FACETS = (
    ("soft_gate", "binary", "Soft gate · Binary"),
    ("soft_gate", "multiclass", "Soft gate · Multiclass"),
    ("persistent", "binary", "Persistent · Binary"),
    ("persistent", "multiclass", "Persistent · Multiclass"),
)

# Fields that define the matched test episode and its factorial condition.
IDENTITY_FIELDS = (
    "episode_id",
    "episode_seed",
    "cell_id",
    "episode_in_cell",
    "support_size",
    "query_size",
    "num_features",
    "input_width",
    "z_column_index",
    "num_regimes",
    "num_classes",
    "class_ratio",
    "task_family",
    "mechanism_mode",
    "rule_mode",
    "prior_type",
    "sampled_is_causal",
    "effective_is_causal",
    "scm_num_layers",
    "scm_hidden_dim",
    "scm_num_causes",
)
CELL_FIELDS = (
    "support_size",
    "num_features",
    "num_regimes",
    "num_classes",
    "class_ratio",
    "task_family",
    "mechanism_mode",
    "rule_mode",
)
METRICS = ("query_cross_entropy", "query_accuracy")


def _load_report(path: Path) -> dict[str, Any]:
    # The evaluator writes NaN for undefined AUC values; CE and accuracy are checked below.
    report = json.loads(path.read_text())
    if report.get("split") != "test":
        raise ValueError(f"{path}: expected a TEST report, got {report.get('split')!r}.")
    rows = report.get("per_episode")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: missing per_episode rows.")
    if int(report.get("episodes", -1)) != len(rows):
        raise ValueError(f"{path}: declared episode count disagrees with per_episode rows.")
    return report


def _episode_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["episode_id"]), int(row["episode_seed"]), int(row["cell_id"])


def _cell_metadata(report: dict[str, Any], label: str) -> dict[int, tuple[Any, ...]]:
    metadata: dict[int, tuple[Any, ...]] = {}
    for row in report["per_episode"]:
        cell_id = int(row["cell_id"])
        current = tuple(row[field] for field in CELL_FIELDS)
        previous = metadata.setdefault(cell_id, current)
        if previous != current:
            raise ValueError(f"{label}: cell {cell_id} has inconsistent factorial metadata.")
    return metadata


def _aligned_rows(
    reference: dict[str, Any], candidate: dict[str, Any], *, candidate_label: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    reference_by_key = {_episode_key(row): row for row in reference["per_episode"]}
    candidate_by_key = {_episode_key(row): row for row in candidate["per_episode"]}
    if len(reference_by_key) != len(reference["per_episode"]):
        raise ValueError("Control report has duplicate episode identities.")
    if len(candidate_by_key) != len(candidate["per_episode"]):
        raise ValueError(f"{candidate_label}: report has duplicate episode identities.")
    if reference_by_key.keys() != candidate_by_key.keys():
        missing = len(reference_by_key.keys() - candidate_by_key.keys())
        extra = len(candidate_by_key.keys() - reference_by_key.keys())
        raise ValueError(
            f"{candidate_label}: episode identities differ from the matched control (missing={missing}, extra={extra})."
        )

    reference_cells = _cell_metadata(reference, "control")
    candidate_cells = _cell_metadata(candidate, candidate_label)
    if reference_cells != candidate_cells:
        raise ValueError(f"{candidate_label}: factorial cell metadata differ from the matched control.")

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key in sorted(reference_by_key):
        base = reference_by_key[key]
        other = candidate_by_key[key]
        for field in IDENTITY_FIELDS:
            if base.get(field) != other.get(field):
                raise ValueError(
                    f"{candidate_label}: episode {key} differs in {field}: {base.get(field)!r} != {other.get(field)!r}."
                )
        for metric in METRICS:
            if not np.isfinite(float(base[metric])) or not np.isfinite(float(other[metric])):
                raise ValueError(f"{candidate_label}: non-finite {metric} for episode {key}.")
        pairs.append((base, other))
    return pairs


def _select_facet(row: dict[str, Any], task_family: str, class_type: str) -> bool:
    return (
        2 <= int(row["num_regimes"]) <= 4
        and row["rule_mode"] == "multiregime"
        and row["task_family"] == task_family
        and ((int(row["num_classes"]) == 2) == (class_type == "binary"))
    )


def _facet_values(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    task_family: str,
    class_type: str,
) -> tuple[np.ndarray, int, int]:
    selected = [(base, candidate) for base, candidate in pairs if _select_facet(base, task_family, class_type)]
    by_cell: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in selected:
        by_cell[int(pair[0]["cell_id"])].append(pair)
    if len(selected) != 2_880 or len(by_cell) != 360:
        raise ValueError(
            f"{task_family}/{class_type}: expected 2,880 episodes in 360 factorial cells; "
            f"found {len(selected)} episodes in {len(by_cell)} cells."
        )

    cell_sizes = {len(rows) for rows in by_cell.values()}
    if cell_sizes != {8}:
        raise ValueError(f"{task_family}/{class_type}: expected 8 episodes per cell, found {sorted(cell_sizes)}.")

    # One row per cell, episode, and paired metric difference.
    cell_values = np.asarray(
        [
            [
                [
                    float(candidate["query_cross_entropy"]) - float(base["query_cross_entropy"]),
                    100.0 * (float(candidate["query_accuracy"]) - float(base["query_accuracy"])),
                ]
                for base, candidate in by_cell[cell_id]
            ]
            for cell_id in sorted(by_cell)
        ],
        dtype=np.float64,
    )
    return cell_values, len(selected), len(by_cell)


def _summarize(
    cell_values: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    # Equal factorial-cell weighting is applied after resampling episodes within each cell.
    point = cell_values.mean(axis=1).mean(axis=0)
    draws = np.empty((replicates, len(METRICS)), dtype=np.float64)
    cell_count, episodes_per_cell, _ = cell_values.shape
    chunk_size = 256
    cells = np.arange(cell_count)[None, :, None]
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(episodes_per_cell, size=(stop - start, cell_count, episodes_per_cell))
        sampled = cell_values[cells, indices]
        draws[start:stop] = sampled.mean(axis=2).mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha), axis=0)
    return [
        {
            "estimate": float(point[index]),
            "confidence_low": float(lower[index]),
            "confidence_high": float(upper[index]),
        }
        for index in range(len(METRICS))
    ]


def _analyze(report_root: Path, *, replicates: int, confidence: float, seed: int) -> dict[str, Any]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for size, _layers in SIZES:
        control_path = report_root / f"original-{size}/seed-2402/v4_test/final.json"
        control = _load_report(control_path)
        reports[("original", size)] = control
        report_paths[f"original-{size}"] = str(control_path)
        for model_name, _display, _color, _marker, _offset in SCHEDULES:
            candidate_path = report_root / f"{model_name}-{size}/seed-2402/v4_test/final.json"
            candidate = _load_report(candidate_path)
            reports[(model_name, size)] = candidate
            report_paths[f"{model_name}-{size}"] = str(candidate_path)
            _aligned_rows(control, candidate, candidate_label=f"{model_name}-{size}")

    # Also require the same test identities and factorial metadata across all size banks.
    reference = reports[("original", "small")]
    reference_keys = {_episode_key(row): row for row in reference["per_episode"]}
    reference_cells = _cell_metadata(reference, "original-small")
    for (model_name, size), report in reports.items():
        keys = {_episode_key(row): row for row in report["per_episode"]}
        if keys.keys() != reference_keys.keys():
            raise ValueError(f"{model_name}-{size}: report differs from the common test bank identities.")
        if _cell_metadata(report, f"{model_name}-{size}") != reference_cells:
            raise ValueError(f"{model_name}-{size}: report differs from the common factorial cell metadata.")

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for task_family, class_type, facet_title in FACETS:
        for size, layers in SIZES:
            control = reports[("original", size)]
            for model_name, schedule, _color, _marker, _offset in SCHEDULES:
                candidate = reports[(model_name, size)]
                pairs = _aligned_rows(control, candidate, candidate_label=f"{model_name}-{size}")
                cell_values, episodes, cells = _facet_values(pairs, task_family=task_family, class_type=class_type)
                summaries = _summarize(
                    cell_values,
                    replicates=replicates,
                    confidence=confidence,
                    rng=rng,
                )
                results.append(
                    {
                        "facet": facet_title,
                        "task_family": task_family,
                        "class_type": class_type,
                        "size": size,
                        "layers": layers,
                        "candidate": model_name,
                        "schedule": schedule,
                        "episodes": episodes,
                        "cells": cells,
                        "excess_cross_entropy_nats": summaries[0],
                        "accuracy_gain_percentage_points": summaries[1],
                    }
                )

    return {
        "schema_version": 1,
        "analysis": "paired_matched_control_effects",
        "evaluation_subset": {
            "bank": "standard-input blind TEST",
            "rule_mode": "multiregime",
            "num_regimes": [2, 3, 4],
            "facets": [title for _family, _classes, title in FACETS],
            "episodes_per_facet_and_size": 2_880,
            "factorial_cells_per_facet_and_size": 360,
            "episode_counts_per_factorial_cell": 8,
        },
        "estimand": "candidate minus same-size native single-regime control; equal mean of factorial-cell means",
        "bootstrap": {
            "unit": "episode",
            "stratification": "factorial evaluation cell",
            "replicates": replicates,
            "confidence": confidence,
            "interval": "percentile",
            "seed": seed,
            "scope": "evaluation-episode variation; not variation across training seeds",
        },
        "model_sizes": {size: {"layers": layers} for size, layers in SIZES},
        "inputs": report_paths,
        "results": results,
    }


def _plot(report: dict[str, Any], output_dir: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 4.35),
        sharex=True,
        sharey="row",
        gridspec_kw={"left": 0.085, "right": 0.99, "top": 0.755, "bottom": 0.185, "wspace": 0.14, "hspace": 0.20},
    )
    figure.text(0.085, 0.955, "Matched-control effects across routing and class type", fontsize=10.6, weight="bold")
    figure.text(
        0.085,
        0.915,
        "Standard-input blind TEST · multiregime tasks (K = 2–4) · 2,880 episodes / 360 cells per panel and size",
        fontsize=7.8,
        color="#526574",
    )
    legend_handles = [
        Line2D([0], [0], color=color, marker=marker, lw=1.35, markersize=3.6, label=label)
        for _key, label, color, marker, _offset in SCHEDULES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.89),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
        fontsize=7.6,
    )

    metric_specs = (
        ("excess_cross_entropy_nats", "Δ excess cross-entropy\n(nats)", "{x:.3f}"),
        ("accuracy_gain_percentage_points", "Δ accuracy gain\n(percentage points)", "{x:.1f}"),
    )
    data_lookup = {
        (row["task_family"], row["class_type"], row["size"], row["candidate"]): row for row in report["results"]
    }
    y_limits: dict[str, tuple[float, float]] = {}
    for row_index, (metric_key, _ylabel, _formatter) in enumerate(metric_specs):
        bounds = [0.0]
        for result in report["results"]:
            summary = result[metric_key]
            bounds.extend((summary["confidence_low"], summary["confidence_high"]))
        span = max(bounds) - min(bounds)
        padding = max(span * 0.10, 0.0005 if row_index == 0 else 0.04)
        y_limits[metric_key] = (min(bounds) - padding, max(bounds) + padding)

    for column, (task_family, class_type, facet_title) in enumerate(FACETS):
        axes[0, column].set_title(facet_title, pad=5, color="#26384a", weight="bold")
        for row_index, (metric_key, ylabel, formatter) in enumerate(metric_specs):
            axis = axes[row_index, column]
            for size, layers in SIZES:
                for model_name, _schedule, color, marker, offset in SCHEDULES:
                    result = data_lookup[(task_family, class_type, size, model_name)]
                    summary = result[metric_key]
                    x = layers + offset
                    estimate = summary["estimate"]
                    lower = summary["confidence_low"]
                    upper = summary["confidence_high"]
                    axis.errorbar(
                        [x],
                        [estimate],
                        yerr=[[estimate - lower], [upper - estimate]],
                        color=color,
                        marker=marker,
                        markersize=3.3,
                        linewidth=1.1,
                        elinewidth=0.85,
                        capsize=1.8,
                        capthick=0.85,
                        zorder=3,
                    )
            # Connect each schedule's estimates over the three tested configurations.
            for _model_name, _schedule, color, _marker, offset in SCHEDULES:
                xs, ys = [], []
                for size, layers in SIZES:
                    summary = data_lookup[(task_family, class_type, size, _model_name)][metric_key]
                    xs.append(layers + offset)
                    ys.append(summary["estimate"])
                axis.plot(xs, ys, color=color, linewidth=1.05, alpha=0.88, zorder=2)

            axis.set_ylim(*y_limits[metric_key])
            axis.axhline(0.0, color="#606f7b", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
            axis.grid(axis="y", color="#e4e9ed", linewidth=0.55, zorder=0)
            axis.set_xlim(1.45, 6.55)
            axis.set_xticks((2, 4, 6))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.yaxis.set_major_formatter(lambda value, _position, fmt=formatter: fmt.format(x=value))
            axis.tick_params(axis="both", length=2.2, width=0.55, colors="#415464", pad=2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#b7c2ca")
            axis.spines["bottom"].set_color("#b7c2ca")
            axis.spines["left"].set_linewidth(0.55)
            axis.spines["bottom"].set_linewidth(0.55)
            if column == 0:
                axis.set_ylabel(ylabel, labelpad=5, color="#26384a")
            if row_index == 0:
                axis.tick_params(labelbottom=False)

    figure.text(0.54, 0.085, "Layers in configuration", ha="center", color="#26384a", fontsize=8.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = "figure2_native_matched_effects"
    figure.savefig(output_dir / f"{basename}.svg", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(output_dir / f"{basename}.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.replicates < 1_000:
        parser.error("--replicates must be at least 1,000.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be strictly between zero and one.")

    report = _analyze(
        args.report_root,
        replicates=args.replicates,
        confidence=args.confidence,
        seed=args.seed,
    )
    _plot(report, args.output_dir)
    json_path = args.output_dir / "figure2_native_matched_effects.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Validated {len(report['inputs'])} TEST reports; every size/facet has 2,880 episodes "
        f"in 360 cells. Wrote Figure 2 SVG, PDF, PNG, and summary JSON to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
