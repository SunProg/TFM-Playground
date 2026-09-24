"""Paired episode-bootstrap uncertainty for native-prior NanoTabPFN bank results.

This script compares final checkpoints that were evaluated on the *same*
synthetic bank. It resamples episodes independently within each factorial
cell, then averages cell means. Thus every cell retains equal weight and the
interval quantifies finite evaluation-episode uncertainty only; it does not
estimate variation from independent pretraining seeds.

Example:

    python scripts/report_native_paired_bootstrap.py \
      --reference artifacts/.../original-large/.../v4_test/final.json \
      --candidate fixed=artifacts/.../rg_z-fixed-large/.../v4_test/final.json \
      --candidate curriculum=artifacts/.../rg_z-curriculum-large/.../v4_test/final.json \
      --json-output paper/native/paired_bootstrap.json \
      --markdown-output paper/native/paired_bootstrap.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("query_cross_entropy", "query_accuracy")
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


def _is_multiregime(row: dict[str, Any]) -> bool:
    return int(row["num_regimes"]) >= 2 and row["rule_mode"] == "multiregime"


def _is_multiclass(row: dict[str, Any]) -> bool:
    return int(row["num_classes"]) >= 3


SLICE_PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "multiregime_multiclass": lambda row: _is_multiregime(row) and _is_multiclass(row),
    "soft_gate_multiregime_multiclass": lambda row: (
        _is_multiregime(row) and _is_multiclass(row) and row["task_family"] == "soft_gate"
    ),
    "persistent_multiregime_multiclass": lambda row: (
        _is_multiregime(row) and _is_multiclass(row) and row["task_family"] == "persistent"
    ),
    "multiregime_binary": lambda row: _is_multiregime(row) and int(row["num_classes"]) == 2,
}
DEFAULT_SLICES = tuple(SLICE_PREDICATES)


def _episode_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["episode_id"]), int(row["episode_seed"]), int(row["cell_id"]))


def _load_report(path: Path) -> dict[str, Any]:
    # Python's JSON decoder intentionally accepts the NaN tokens emitted by
    # the existing evaluator for undefined per-episode AUC. This analysis
    # uses only finite cross-entropy and accuracy fields.
    report = json.loads(path.read_text())
    if report.get("split") != "test":
        raise ValueError(f"{path}: expected a test report, got {report.get('split')!r}.")
    rows = report.get("per_episode")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: no per_episode rows.")
    if int(report.get("episodes", -1)) != len(rows):
        raise ValueError(f"{path}: declared episode count disagrees with per_episode rows.")
    return report


def _aligned_rows(
    reference: dict[str, Any], candidate: dict[str, Any], *, candidate_label: str
) -> list[tuple[dict, dict]]:
    reference_by_key = {_episode_key(row): row for row in reference["per_episode"]}
    candidate_by_key = {_episode_key(row): row for row in candidate["per_episode"]}
    if len(reference_by_key) != len(reference["per_episode"]):
        raise ValueError("Reference report has duplicate episode identities.")
    if len(candidate_by_key) != len(candidate["per_episode"]):
        raise ValueError(f"{candidate_label}: report has duplicate episode identities.")
    if reference_by_key.keys() != candidate_by_key.keys():
        missing = len(reference_by_key.keys() - candidate_by_key.keys())
        extra = len(candidate_by_key.keys() - reference_by_key.keys())
        raise ValueError(
            f"{candidate_label}: reports do not evaluate the same episodes " f"(missing={missing}, extra={extra})."
        )

    pairs = []
    for key in sorted(reference_by_key):
        base = reference_by_key[key]
        other = candidate_by_key[key]
        for field in IDENTITY_FIELDS:
            if base.get(field) != other.get(field):
                raise ValueError(
                    f"{candidate_label}: episode {key} differs in {field}: "
                    f"{base.get(field)!r} != {other.get(field)!r}."
                )
        for metric in METRICS:
            if not np.isfinite(float(base[metric])) or not np.isfinite(float(other[metric])):
                raise ValueError(f"{candidate_label}: non-finite {metric} for episode {key}.")
        pairs.append((base, other))
    return pairs


def _cell_weighted_mean(rows: list[tuple[dict, dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Return exact equal-cell means for reference and candidate metrics."""
    by_cell: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in rows:
        by_cell[int(pair[0]["cell_id"])].append(pair)
    base_cell_means, candidate_cell_means = [], []
    for cell_rows in by_cell.values():
        base_cell_means.append([np.mean([float(base[metric]) for base, _ in cell_rows]) for metric in METRICS])
        candidate_cell_means.append([np.mean([float(other[metric]) for _, other in cell_rows]) for metric in METRICS])
    return np.mean(base_cell_means, axis=0), np.mean(candidate_cell_means, axis=0)


def _stratified_bootstrap(
    rows: list[tuple[dict, dict]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return [metric, replicate] draws of candidate-minus-reference means.

    The resampling unit is an episode. Sampling within cell preserves the
    intended equal-cell factorial estimand even if future banks contain a
    nonuniform number of episodes per cell.
    """
    by_cell: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in rows:
        by_cell[int(pair[0]["cell_id"])].append(pair)
    if not by_cell:
        raise ValueError("Cannot bootstrap an empty slice.")

    means = np.zeros((len(METRICS), replicates), dtype=np.float64)
    for cell_rows in by_cell.values():
        differences = np.asarray(
            [[float(other[metric]) - float(base[metric]) for base, other in cell_rows] for metric in METRICS],
            dtype=np.float64,
        )
        sample_size = differences.shape[1]
        indices = rng.integers(sample_size, size=(replicates, sample_size))
        # One equal-weighted bootstrap sample from this cell for every
        # replicate, accumulated before taking the mean over cells.
        means += differences[:, indices].sum(axis=2) / sample_size
    return means / len(by_cell)


def _summarize(
    pairs: list[tuple[dict, dict]],
    *,
    slice_name: str,
    candidate_label: str,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected = [pair for pair in pairs if SLICE_PREDICATES[slice_name](pair[0])]
    if len(selected) < 2:
        raise ValueError(f"{candidate_label}/{slice_name}: fewer than two selected episodes.")
    cell_count = len({int(base["cell_id"]) for base, _ in selected})
    base_means, candidate_means = _cell_weighted_mean(selected)
    draws = _stratified_bootstrap(selected, replicates=replicates, rng=rng)
    alpha = (1.0 - confidence) / 2.0
    result = []
    for metric_index, metric in enumerate(METRICS):
        mean_delta = float(candidate_means[metric_index] - base_means[metric_index])
        lower, upper = (float(value) for value in np.quantile(draws[metric_index], (alpha, 1.0 - alpha)))
        # Candidate-minus-canonical: lower CE is favorable, while greater
        # accuracy (and therefore accuracy gain) is favorable.
        favors_candidate = draws[metric_index] < 0 if metric == "query_cross_entropy" else draws[metric_index] > 0
        result.append(
            {
                "candidate": candidate_label,
                "slice": slice_name,
                "metric": metric,
                "episodes": len(selected),
                "cells": cell_count,
                "reference_mean": float(base_means[metric_index]),
                "candidate_mean": float(candidate_means[metric_index]),
                "candidate_minus_reference": mean_delta,
                "confidence_low": lower,
                "confidence_high": upper,
                "bootstrap_probability_favors_candidate": float(np.mean(favors_candidate)),
            }
        )
    return result


def _parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--candidate must have the form label=path.")
    return label, Path(path)


def _format_value(value: float, metric: str) -> str:
    if metric == "query_accuracy":
        return f"{100 * value:+.2f} pp"
    return f"{value:+.4f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired episode-bootstrap analysis",
        "",
        "This analysis compares final checkpoints on the same score-hidden native synthetic **test** bank. "
        "Every bootstrap replicate resamples episodes independently within each factorial cell and then averages "
        "cell means. `candidate − canonical` is reported: negative cross-entropy and positive accuracy "
        "difference (equivalently, accuracy-gain difference) favor the candidate.",
        "",
        f"Bootstrap configuration: {report['bootstrap']['replicates']:,} percentile replicates, "
        f"{100 * report['bootstrap']['confidence']:.0f}% intervals, random seed {report['bootstrap']['seed']}.",
        "",
        "These intervals quantify variation across the fixed evaluation episodes only. They do **not** quantify "
        "variation across independent pretraining seeds, and their per-slice interpretation is descriptive rather "
        "than adjusted for multiple comparisons.",
        "",
        "## Results",
        "",
        "| Candidate | Slice | Episodes / cells | Metric | Canonical | Candidate | "
        "Candidate − canonical (CI) | Bootstrap fraction favoring candidate |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    display_names = {
        "multiregime_multiclass": "Multiregime multiclass",
        "soft_gate_multiregime_multiclass": "Soft-gate multiregime multiclass",
        "persistent_multiregime_multiclass": "Persistent multiregime multiclass",
        "multiregime_binary": "Multiregime binary",
    }
    metric_names = {
        "query_cross_entropy": "Cross-entropy (nats)",
        "query_accuracy": "Accuracy (pp)",
    }
    for row in report["results"]:
        metric = row["metric"]
        delta = _format_value(row["candidate_minus_reference"], metric)
        lower = _format_value(row["confidence_low"], metric)
        upper = _format_value(row["confidence_high"], metric)
        lines.append(
            f"| {row['candidate']} | {display_names[row['slice']]} | {row['episodes']:,} / {row['cells']:,} | "
            f"{metric_names[metric]} | {_format_value(row['reference_mean'], metric)} | "
            f"{_format_value(row['candidate_mean'], metric)} | {delta} [{lower}, {upper}] | "
            f"{100 * row['bootstrap_probability_favors_candidate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Inputs:",
            "",
            f"- Canonical: `{report['inputs']['reference']}`",
            *[f"- {label}: `{path}`" for label, path in report["inputs"]["candidates"].items()],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path, help="Canonical model's final test report.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_parse_candidate,
        metavar="LABEL=PATH",
        help="Candidate final test report; may be supplied more than once.",
    )
    parser.add_argument(
        "--slice",
        action="append",
        choices=tuple(SLICE_PREDICATES),
        help="Slice to report; defaults to four central synthetic slices.",
    )
    parser.add_argument("--replicates", type=int, default=5_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    if args.replicates < 1_000:
        parser.error("--replicates must be at least 1,000.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between zero and one.")
    candidate_paths = dict(args.candidate)
    if len(candidate_paths) != len(args.candidate):
        parser.error("Candidate labels must be unique.")
    slices = tuple(args.slice) if args.slice else DEFAULT_SLICES

    reference = _load_report(args.reference)
    candidate_reports = {label: _load_report(path) for label, path in candidate_paths.items()}
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    for candidate_label, candidate in candidate_reports.items():
        pairs = _aligned_rows(reference, candidate, candidate_label=candidate_label)
        for slice_name in slices:
            results.extend(
                _summarize(
                    pairs,
                    slice_name=slice_name,
                    candidate_label=candidate_label,
                    replicates=args.replicates,
                    confidence=args.confidence,
                    rng=rng,
                )
            )

    report = {
        "schema_version": 1,
        "analysis": "paired_episode_bootstrap",
        "inputs": {
            "reference": str(args.reference),
            "candidates": {label: str(path) for label, path in candidate_paths.items()},
            "split": reference["split"],
        },
        "bootstrap": {
            "unit": "episode",
            "stratification": "factorial evaluation cell",
            "cell_estimand": "equal-weighted mean of cell means",
            "replicates": args.replicates,
            "confidence": args.confidence,
            "seed": args.seed,
            "interval": "percentile",
            "scope": "finite evaluation-episode uncertainty; not independent-pretraining-seed uncertainty",
        },
        "results": results,
    }
    for output in (args.json_output, args.markdown_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.markdown_output.write_text(_markdown(report))
    print(f"Wrote {len(results)} summaries to {args.json_output} and {args.markdown_output}.")


if __name__ == "__main__":
    main()
