"""Descriptive paired analysis of native-prior BeyondArena results.

The manuscript's complete 36-task result tables contain one task-equal score
per final checkpoint.  This script compares the matched six-layer native
single-regime and curriculum-pretrained NanoTabPFN rows task by task, rather
than treating the small grouped subset as a separate benchmark leaderboard.

It reads the three Markdown tables in ``native_prior_results.md`` and writes a
compact report.  Thus values have the precision of the printed source tables;
the output is descriptive, not a significance test or an estimate of variation
across pretraining seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


TABLE_NAMES = {
    "### A1 — excess_cross_entropy": "excess_cross_entropy",
    "### A1 — accuracy_gain": "accuracy_gain",
    "### A1 — macro_ovr_auc": "macro_ovr_auc",
}
METRIC_DIRECTIONS = {
    "excess_cross_entropy": -1,
    "accuracy_gain": 1,
    "macro_ovr_auc": 1,
}


def _plain(cell: str) -> str:
    return cell.replace("**", "").strip()


def _parse_table(lines: list[str], start: int) -> tuple[list[dict[str, str]], int]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    index = start + 1
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("### ") or line.startswith("## "):
            break
        if line.startswith("|"):
            cells = [_plain(cell) for cell in line.strip("|").split("|")]
            if header is None:
                header = cells
            elif set(cells) == {"---"}:
                pass
            elif len(cells) == len(header):
                rows.append(dict(zip(header, cells, strict=True)))
        index += 1
    if header is None:
        raise ValueError("Table is missing a header")
    return rows, index


def load_tables(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load task metadata and model scores from the three appendix tables."""
    lines = Path(path).read_text().splitlines()
    parsed: dict[str, list[dict[str, str]]] = {}
    for heading, metric in TABLE_NAMES.items():
        try:
            start = lines.index(heading)
        except ValueError as error:
            raise ValueError(f"Missing required table heading: {heading}") from error
        parsed[metric], _ = _parse_table(lines, start)
    task_names = [row["task"] for row in parsed["excess_cross_entropy"]]
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("Expected exactly one excess-CE row per task")
    if any([row["task"] for row in rows] != task_names for rows in parsed.values()):
        raise ValueError("The task ordering differs between metric tables")

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for task_index, task in enumerate(task_names):
        source = parsed["excess_cross_entropy"][task_index]
        entry: dict[str, Any] = {
            "task": task,
            "target": source["cls"],
            "regime": source["regime"],
            "train_rows": int(source["n"]),
            "features": int(source["d"]),
            "majority_fraction": float(source["maj"]),
            "metrics": {},
        }
        for metric, rows in parsed.items():
            row = rows[task_index]
            entry["metrics"][metric] = {
                "canonical": float(row["nat-o-L"]),
                "curriculum": float(row["nat-rgC-L"]),
            }
        result[task] = entry
    return result


def _bucket(task: dict[str, Any]) -> tuple[str, ...]:
    target = "binary" if task["target"] == "bin" else "multiclass"
    regime = {"IID": "IID", "G": "grouped", "T": "temporal"}[task["regime"]]
    return ("all", target, regime, f"{target} × {regime}")


def _difference(task: dict[str, Any], metric: str) -> float:
    values = task["metrics"][metric]
    return float(values["curriculum"] - values["canonical"])


def _win_tie_loss(differences: list[float], direction: int, tolerance: float) -> tuple[int, int, int]:
    oriented = np.asarray(differences) * direction
    return (
        int(np.sum(oriented > tolerance)),
        int(np.sum(np.abs(oriented) <= tolerance)),
        int(np.sum(oriented < -tolerance)),
    )


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def analyse(tasks: dict[str, dict[str, Any]], *, tolerance: float = 0.0005) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks.values():
        for bucket in _bucket(task):
            groups[bucket].append(task)
    summaries: list[dict[str, Any]] = []
    for bucket, members in sorted(groups.items(), key=lambda item: (item[0] != "all", item[0])):
        entry: dict[str, Any] = {"slice": bucket, "tasks": len(members), "metrics": {}}
        for metric, direction in METRIC_DIRECTIONS.items():
            differences = [_difference(task, metric) for task in members]
            wins, ties, losses = _win_tie_loss(differences, direction, tolerance)
            entry["metrics"][metric] = {
                "curriculum_minus_canonical_mean": mean(differences),
                "curriculum_minus_canonical_median": median(differences),
                "curriculum_favorable_tasks": wins,
                "ties_at_printed_precision": ties,
                "canonical_favorable_tasks": losses,
            }
        summaries.append(entry)

    all_tasks = list(tasks.values())
    correlations: dict[str, dict[str, float]] = {}
    covariates = {
        "log_train_rows": [math.log(task["train_rows"]) for task in all_tasks],
        "features": [float(task["features"]) for task in all_tasks],
        "majority_fraction": [task["majority_fraction"] for task in all_tasks],
    }
    for metric in METRIC_DIRECTIONS:
        deltas = [_difference(task, metric) for task in all_tasks]
        correlations[metric] = {name: _spearman(values, deltas) for name, values in covariates.items()}
    return {
        "tasks": list(tasks.values()),
        "tolerance": tolerance,
        "summaries": summaries,
        "spearman_task_level": correlations,
        "notes": [
            "Each task has one task-equal, final-checkpoint score from the printed appendix table.",
            "Differences are curriculum minus canonical; negative excess CE and positive accuracy/AUC favor curriculum.",
            "Ties use a 0.0005 tolerance because the source tables are printed to three decimals.",
            "These comparisons are descriptive: the 36 tasks are heterogeneous and this source has one pretraining seed.",
        ],
    }


def markdown(report: dict[str, Any], *, source: str) -> str:
    lines = [
        "# BeyondArena heterogeneity analysis",
        "",
        f"Source: [{Path(source).name}]({Path(source).name}). This report compares the final six-layer native "
        "canonical and curriculum NanoTabPFN checkpoints task by task. It does not re-evaluate the benchmark.",
        "",
        "The source table contains 36 tasks: 27 binary (26 IID, one grouped) and nine multiclass "
        "(six IID, two grouped, one temporal). Grouped and temporal rows are too few for inferential claims.",
        "",
        "Differences are **curriculum minus canonical**. For excess CE, negative favors curriculum; for accuracy "
        "gain and AUC, positive favors curriculum. The values are rounded to 0.001 in the source table; tie counts "
        "use an absolute tolerance of 0.0005.",
        "",
        "| Slice | Tasks | Δ excess CE | CE W/T/L | Δ accuracy gain (pp) | Accuracy W/T/L | Δ AUC | AUC W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wanted = ("all", "binary", "multiclass", "IID", "grouped", "temporal", "binary × IID", "binary × grouped", "multiclass × IID", "multiclass × grouped", "multiclass × temporal")
    summary = {row["slice"]: row for row in report["summaries"]}
    for name in wanted:
        row = summary.get(name)
        if row is None:
            continue
        ce = row["metrics"]["excess_cross_entropy"]
        acc = row["metrics"]["accuracy_gain"]
        auc = row["metrics"]["macro_ovr_auc"]
        lines.append(
            f"| {name} | {row['tasks']} | {ce['curriculum_minus_canonical_mean']:+.4f} | "
            f"{ce['curriculum_favorable_tasks']}/{ce['ties_at_printed_precision']}/{ce['canonical_favorable_tasks']} | "
            f"{100 * acc['curriculum_minus_canonical_mean']:+.2f} | "
            f"{acc['curriculum_favorable_tasks']}/{acc['ties_at_printed_precision']}/{acc['canonical_favorable_tasks']} | "
            f"{auc['curriculum_minus_canonical_mean']:+.4f} | "
            f"{auc['curriculum_favorable_tasks']}/{auc['ties_at_printed_precision']}/{auc['canonical_favorable_tasks']} |"
        )
    lines.extend(
        [
            "",
            "W/T/L is the number of tasks favoring curriculum / tied at printed precision / favoring canonical.",
            "",
            "## Interpretation",
            "",
            "The all-task comparison tests whether the synthetic-prior change produces a broad real-data advantage. "
            "The IID/grouped/temporal rows instead describe where the existing task-level differences occur; they "
            "do not establish an effect of group shift because the groups contain only 32, three, and one tasks, "
            "respectively.",
            "",
            "The task-level Spearman correlations below are exploratory and should not be used to select a favorable "
            "subgroup. They test only whether the observed curriculum-minus-canonical difference covaries with coarse "
            "dataset descriptors already present in the appendix.",
            "",
            "| Metric difference | log(training rows) | features | majority fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, values in report["spearman_task_level"].items():
        lines.append(
            f"| {metric} | {values['log_train_rows']:+.3f} | {values['features']:+.3f} | "
            f"{values['majority_fraction']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "The complementary merged-heart-site experiment is the stronger real-data group-information diagnostic: "
            "the hospital is highly recoverable from features in IID folds, adding its label changes little, and "
            "multiregime pretraining does not consistently alter that pattern. See "
            "[heart_sites_group_feature.md](heart_sites_group_feature.md).",
            "",
            "## Scope",
            "",
            "This report is descriptive rather than confirmatory. It uses one trained model per condition and values "
            "rounded in the source table; it cannot estimate training-seed uncertainty or prove that a benchmark split "
            "regime corresponds to latent regime-dependent label mechanisms.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="paper/native/native_prior_results.md")
    parser.add_argument("--markdown-output", default="paper/native/beyondarena_heterogeneity_analysis.md")
    parser.add_argument("--json-output", default="paper/native/beyondarena_heterogeneity_analysis.json")
    args = parser.parse_args()
    tasks = load_tables(args.source)
    report = analyse(tasks)
    markdown_path = Path(args.markdown_output)
    json_path = Path(args.json_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(report, source=args.source))
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(tasks)} tasks -> {markdown_path} and {json_path}")


if __name__ == "__main__":
    main()
