"""Merged UCI Heart Disease sites: does naming the hospital as a feature help?

BeyondArena carries three sites of the same 1988 study (Cleveland 303, Hungary 294, VA Long Beach 200) with an
identical 13-attribute schema and the same binary target.  Merging them gives one real dataset with a known
latent regime (the site), i.e. the real-data counterpart of the v4 synthetic multiregime setting: the sites
differ in class balance (0.54 / 0.64 / 0.74 majority) and in missingness (``ca``/``thal`` are almost entirely
absent outside Cleveland), so the label rule is plausibly site-dependent.

Two feature conditions, scored on the same folds:

``site_hidden``   the 13 attributes only (the model must infer the regime, as in a z-blind v4 episode)
``site_given``    the 13 attributes plus a ``site`` column (the z-exposed twin)

Two fold protocols:

``iid``   repeated stratified K-fold over the pooled rows (stratified on target x site) — the sites are mixed
          across support and query, so ``site`` is a legitimately usable feature
``loso``  leave-one-site-out — the query site never appears in the support fold, so ``site_given`` carries a
          value the model has never seen with a label: a pure distribution-shift test

Every model is scored on the identical folds with the existing BeyondArena machinery (same preprocessor,
same full-fold prediction path, same metrics).

    python -m tfmplayground.experiments.evaluate_heart_sites_merged \\
        --datasets-root <BeyondArena snapshot dir> --run-roots runs_exp/native \\
        --baselines tabpfn-v2.2 tabicl-v2 rf logreg --output-dir runs_eval/heart_sites --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import (
    UnsupportedFoldError,
    _encode_fold_labels,
    baseline_selections,
    build_external_model,
    discover_v4_checkpoints,
    fold_metrics,
    load_checkpoint_for_inference,
    predict_external_full_fold,
    predict_full_fold,
    prepare_fold_data,
)

SITES = ("heart_disease_cleveland", "heart_disease_hungary", "heart_disease_va_long_beach")
TARGET = "heart_disease_diagnosis"
SITE_COLUMN = "site"
CONDITIONS = ("site_hidden", "site_given")


def load_merged(datasets_root: Path, sites: Sequence[str] = SITES) -> pd.DataFrame:
    """Concatenate the site tables, coercing the per-site dtypes to one numeric schema."""

    frames = []
    for name in sites:
        matches = sorted(datasets_root.glob(f"{name}/*/dataset.parquet"))
        if not matches:
            raise FileNotFoundError(f"No dataset.parquet under {datasets_root / name}")
        frame = pd.read_parquet(matches[0])
        missing = [column for column in frames[0].columns
                   if column not in frame.columns and column != SITE_COLUMN] if frames else []
        if missing:
            raise ValueError(f"{name} lacks columns {missing}; the sites must share one schema.")
        frame = frame.copy()
        for column in frame.columns:
            if column != TARGET:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[SITE_COLUMN] = name.removeprefix("heart_disease_")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged[TARGET] = merged[TARGET].astype(int)
    return merged


def iid_folds(frame: pd.DataFrame, *, splits: int, repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    from sklearn.model_selection import RepeatedStratifiedKFold

    strata = frame[SITE_COLUMN].astype(str) + "|" + frame[TARGET].astype(str)
    splitter = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    return [
        (train, test, f"iid-{index // splits}-{index % splits}")
        for index, (train, test) in enumerate(splitter.split(frame, strata))
    ]


def loso_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    folds = []
    for site in sorted(frame[SITE_COLUMN].unique()):
        test = np.flatnonzero((frame[SITE_COLUMN] == site).to_numpy())
        train = np.flatnonzero((frame[SITE_COLUMN] != site).to_numpy())
        folds.append((train, test, f"loso-{site}"))
    return folds


def evaluate(
    frame: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray, str]],
    selections: Sequence[Any],
    *,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
    checkpoint_policy: str,
) -> list[dict[str, Any]]:
    """One row per (model, condition, fold); the same folds are reused for every model and condition."""

    rows: list[dict[str, Any]] = []
    for selection in selections:
        is_external = selection.model_kind in {"tabpfn", "tabicl", "sklearn"}
        model_cache: Any = None
        if not is_external:
            if selection.checkpoint_policy != checkpoint_policy or selection.checkpoint_path is None:
                continue
            model_cache = load_checkpoint_for_inference(selection.checkpoint_path, device=device)
        for condition in CONDITIONS:
            group_columns = () if condition == "site_given" else (SITE_COLUMN,)
            for train_index, test_index, fold_name in folds:
                train_frame, test_frame = frame.iloc[train_index], frame.iloc[test_index]
                train_x = None
                try:
                    train_x, test_x, _, _ = prepare_fold_data(
                        train_frame, test_frame, target_column=TARGET, group_columns=group_columns
                    )
                    train_y, test_y = _encode_fold_labels(train_frame[TARGET].tolist(), test_frame[TARGET].tolist(), 2)
                    if is_external:
                        model = build_external_model(selection, device)
                        probabilities = predict_external_full_fold(
                            model, train_x, train_y, test_x, classes=2, query_chunk_size=query_chunk_size
                        )
                    else:
                        probabilities = predict_full_fold(
                            model_cache, train_x, train_y, test_x, classes=2, device=device,
                            query_chunk_size=query_chunk_size, num_mem_chunks=num_mem_chunks,
                        )
                    metrics = fold_metrics(train_y, test_y, probabilities, 2)
                    per_site = {}
                    query_sites = test_frame[SITE_COLUMN].to_numpy()
                    for site in sorted(set(query_sites.tolist())):
                        mask = query_sites == site
                        if mask.sum() >= 2 and len(np.unique(test_y[mask])) > 0:
                            per_site[site] = fold_metrics(train_y, test_y[mask], probabilities[mask], 2)
                    status, error = "ok", None
                except (UnsupportedFoldError, ValueError) as failure:
                    metrics, per_site, status, error = {}, {}, "unsupported", str(failure)
                base = {
                    "model": selection.model_identity,
                    "condition": condition,
                    "fold": fold_name,
                    "protocol": fold_name.split("-")[0],
                    "features": int(train_x.shape[1]) if train_x is not None else None,
                    "support_rows": int(len(train_index)),
                    "query_rows": int(len(test_index)),
                    "status": status,
                    "error": error,
                }
                rows.append({**base, "query_site": "all", **metrics})
                for site, site_metrics in per_site.items():
                    rows.append({**base, "query_site": site,
                                 "query_rows": int((test_frame[SITE_COLUMN] == site).sum()), **site_metrics})
        del model_cache
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean per (model, protocol, condition) plus the site_given - site_hidden delta."""

    rows = [row for row in rows if row.get("query_site", "all") == "all"]
    keys = sorted({(row["model"], row["protocol"]) for row in rows})
    metrics = ("excess_cross_entropy", "accuracy_gain", "macro_ovr_auc", "model_cross_entropy")
    summary = []
    for model, protocol in keys:
        entry: dict[str, Any] = {"model": model, "protocol": protocol}
        for condition in CONDITIONS:
            selected = [r for r in rows if r["model"] == model and r["protocol"] == protocol
                        and r["condition"] == condition and r["status"] == "ok"]
            entry[f"{condition}_folds"] = len(selected)
            for metric in metrics:
                values = [r[metric] for r in selected if r.get(metric) is not None and not math.isnan(r[metric])]
                entry[f"{condition}_{metric}"] = statistics.fmean(values) if values else float("nan")
        for metric in metrics:
            entry[f"delta_{metric}"] = entry[f"site_given_{metric}"] - entry[f"site_hidden_{metric}"]
        summary.append(entry)
    return summary


def markdown(summary: Sequence[dict[str, Any]]) -> str:
    lines = []
    for protocol, title in (("iid", "Repeated stratified K-fold (sites mixed across folds)"),
                            ("loso", "Leave-one-site-out (query site unseen in support)")):
        selected = [s for s in summary if s["protocol"] == protocol]
        if not selected:
            continue
        selected.sort(key=lambda s: s["site_hidden_excess_cross_entropy"])
        lines += [f"\n### {title}\n",
                  "| model | excess CE hidden | excess CE given | Δ CE | acc gain hidden | acc gain given | AUC hidden | AUC given |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in selected:
            lines.append(
                f"| {s['model']} | {s['site_hidden_excess_cross_entropy']:+.4f} | {s['site_given_excess_cross_entropy']:+.4f} | "
                f"{s['delta_excess_cross_entropy']:+.4f} | {s['site_hidden_accuracy_gain']:+.4f} | {s['site_given_accuracy_gain']:+.4f} | "
                f"{s['site_hidden_macro_ovr_auc']:.4f} | {s['site_given_macro_ovr_auc']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def markdown_per_site(rows: Sequence[dict[str, Any]], protocol: str = "iid") -> str:
    """Mean per (model, condition, query site) over the folds of one protocol."""

    selected = [r for r in rows if r["protocol"] == protocol and r["status"] == "ok" and r.get("query_site", "all") != "all"]
    if not selected:
        return ""
    sites = sorted({r["query_site"] for r in selected})
    models = sorted({r["model"] for r in selected})
    lines = [f"\n### {protocol.upper()} folds, query rows split by site (hidden / given)\n",
             "| model | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites)]
    for model in models:
        cells = []
        for site in sites:
            values = {c: [r["excess_cross_entropy"] for r in selected
                          if r["model"] == model and r["query_site"] == site and r["condition"] == c]
                      for c in CONDITIONS}
            cells.append(f"{statistics.fmean(values['site_hidden']):+.4f} / {statistics.fmean(values['site_given']):+.4f}"
                         if all(values.values()) else "—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", required=True, help="BeyondArena snapshot directory holding the site folders")
    parser.add_argument("--run-roots", default=None, help="Comma-separated run roots with v4 .pth checkpoints")
    parser.add_argument("--baselines", nargs="*", default=[], help="Published/conventional models, e.g. tabpfn-v3 rf logreg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-policy", default="final", choices=("final", "best_own_val"))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--num-mem-chunks", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_merged(Path(args.datasets_root))
    folds = iid_folds(frame, splits=args.splits, repeats=args.repeats, seed=args.seed) + loso_folds(frame)
    selections: list[Any] = []
    if args.run_roots:
        found, _ = discover_v4_checkpoints([Path(root) for root in args.run_roots.split(",")])
        selections += list(found)
    if args.baselines:
        external, _ = baseline_selections(args.baselines)
        selections += external
    if not selections:
        raise SystemExit("Nothing to evaluate: pass --run-roots and/or --baselines.")
    print(f"merged {len(frame)} rows x {frame.shape[1] - 2} features from {frame[SITE_COLUMN].nunique()} sites; "
          f"{len(folds)} folds; {len(selections)} models", flush=True)
    rows = evaluate(
        frame, folds, selections, device=args.device, query_chunk_size=args.query_chunk_size,
        num_mem_chunks=args.num_mem_chunks, checkpoint_policy=args.checkpoint_policy,
    )
    summary = summarize(rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "per_fold.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = markdown(summary) + markdown_per_site(rows, "iid") + markdown_per_site(rows, "loso")
    (out / "summary.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
