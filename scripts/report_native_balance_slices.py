"""Describe paired native-model effects by support-label balance on the fixed test bank.

This reads labels and existing evaluation reports only. It performs no model
inference. Balance bins are defined within each requested multiclass count,
without looking at model outcomes. Effects are candidate minus native control
and average episode differences within each occupied factorial cell. The
pooled routing summaries give the three class counts equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts/cluster_sync/tfm_eval/native_regime_split/blind"
DEFAULT_BANK = ROOT / "artifacts/local_mps/data/tabicl_mix_scm_evaluation/test.h5"
LABELS = ("low", "middle", "high")
ROUTINGS = ("soft_gate", "persistent")
MODELS = ("fixed", "curriculum")


def _report(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    rows = document["per_episode"]
    if document.get("split") != "test" or len(rows) != 32256:
        raise ValueError(f"Unexpected test report size or split: {path}")
    if any(row["episode_id"] != index for index, row in enumerate(rows)):
        raise ValueError(f"Report episode order is not bank order: {path}")
    return rows


def _verify_alignment(handle: h5py.File, reports: dict[str, list[dict]]) -> None:
    if handle["evaluation_split"][()] != b"test" or len(handle["y"]) != 32256:
        raise ValueError("Expected the 32,256-episode standard-input test bank")
    names = {
        "task_family": ("task_family_index", "families"),
        "mechanism_mode": ("mechanism_mode_index", "mechanism_modes"),
        "rule_mode": ("rule_mode_index", "rule_modes"),
        "prior_type": ("prior_type_index", "prior_types"),
    }
    scalar_fields = (
        "cell_id", "episode_in_cell", "episode_seed", "support_size",
        "query_size", "num_features", "input_width", "z_column_index",
        "num_regimes", "num_classes",
    )
    scalar_values = {field: handle[field][:] for field in scalar_fields}
    enum_indices = {field: handle[indices][:] for field, (indices, _) in names.items()}
    enum_values = {
        field: [value.decode() for value in handle[values][:]]
        for field, (_, values) in names.items()
    }
    for label, rows in reports.items():
        for index, row in enumerate(rows):
            for field in scalar_fields:
                if row[field] != int(scalar_values[field][index]):
                    raise ValueError(f"{label}: episode {index} differs in {field}")
            for field, (indices, _) in names.items():
                expected = enum_values[field][int(enum_indices[field][index])]
                if row[field] != expected:
                    raise ValueError(f"{label}: episode {index} differs in {field}")
    # The reports retain binary class frequencies. Check those label summaries
    # as an additional bank-content fingerprint beyond episode metadata.
    for index, row in enumerate(reports["native"]):
        if row["num_classes"] != 2:
            continue
        support_size, query_size = row["support_size"], row["query_size"]
        labels = handle["y"][index, : support_size + query_size]
        support_rate = float(np.mean(labels[:support_size] == 1))
        query_rate = float(np.mean(labels[support_size:] == 1))
        if support_rate != row["support_positive_rate"] or query_rate != row["query_positive_rate"]:
            raise ValueError(f"Binary label frequencies differ in episode {index}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_mean(rows: list[dict], candidate: str, metric: str) -> float:
    cells: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        cells[row["cell_id"]].append(row[candidate][metric] - row["native"][metric])
    return float(np.mean([np.mean(values) for values in cells.values()]))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty balance slice")
    result = {
        "episodes": len(rows),
        "cells": len({row["cell_id"] for row in rows}),
        "mean_support_majority": float(np.mean([row["majority"] for row in rows])),
    }
    for candidate in MODELS:
        result[candidate] = {
            "ce": _cell_mean(rows, candidate, "query_cross_entropy"),
            "accuracy_pp": 100 * _cell_mean(rows, candidate, "query_accuracy"),
        }
    return result


def _tex_table(rows: list[dict], *, caption: str, label: str, include_class: bool) -> str:
    first = "Classes & " if include_class else "Routing & "
    second = "Routing & " if include_class else ""
    lines = [
        r"\begin{table}[ht]" if include_class else r"\begin{table}[t]", r"\centering\footnotesize",
        "\\caption{" + caption + "}\\label{" + label + "}",
        r"\begin{tabular}{@{}" + ("ll" if include_class else "l") + "lrrrrrr@{}}\\toprule",
        first + second + r"Maj. bin & $n$ & Cells & Fixed $\Delta$CE & Fixed $\Delta$Acc & Curr. $\Delta$CE & Curr. $\Delta$Acc\\\midrule",
    ]
    for row in rows:
        prefix = (f"{row['classes']} & " if include_class else f"{row['routing'].replace('_', ' ')} & ")
        if include_class:
            prefix += f"{row['routing'].replace('_', ' ')} & "
        fixed, curriculum = row["fixed"], row["curriculum"]
        lines.append(
            prefix + f"{row['balance']} & {row['episodes']:,} & {row['cells']} & "
            + f"{fixed['ce']:+.4f} & {fixed['accuracy_pp']:+.2f} & "
            + f"{curriculum['ce']:+.4f} & {curriculum['accuracy_pp']:+.2f}\\\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _mechanism_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Matched six-layer candidate-minus-native effects by routing and rule mechanism on multiregime multiclass episodes. Each row has 1,440 episodes in 180 cells. Negative CE and positive accuracy differences (pp) favor the candidate.}",
        r"\label{tab:mechanism}",
        r"\begin{tabular}{@{}llrrrr@{}}\toprule",
        r"Routing & Rule & Fixed $\Delta$CE & Fixed $\Delta$Acc & Curr. $\Delta$CE & Curr. $\Delta$Acc\\\midrule",
    ]
    for row in rows:
        fixed, curriculum = row["fixed"], row["curriculum"]
        lines.append(
            row["routing"].replace("_", " ") + f" & ${row['mechanism'][0]}_z$ & "
            + f"{fixed['ce']:+.4f} & {fixed['accuracy_pp']:+.2f} & "
            + f"{curriculum['ce']:+.4f} & {curriculum['accuracy_pp']:+.2f}\\\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--json-output", type=Path, default=ROOT / "paper/native/balance_slices.json")
    parser.add_argument("--main-tex", type=Path, default=ROOT / "paper/iclr2027/main_balance_table.tex")
    parser.add_argument("--mechanism-tex", type=Path, default=ROOT / "paper/iclr2027/main_mechanism_table.tex")
    parser.add_argument("--appendix-tex", type=Path, default=ROOT / "paper/iclr2027/appendix_c_balance_tables.tex")
    args = parser.parse_args()
    reports = {
        "native": _report(args.report_root / "original-large/seed-2402/v4_test/final.json"),
        "fixed": _report(args.report_root / "rg_z-fixed-large/seed-2402/v4_test/final.json"),
        "curriculum": _report(args.report_root / "rg_z-curriculum-large/seed-2402/v4_test/final.json"),
    }
    with h5py.File(args.bank) as handle:
        _verify_alignment(handle, reports)
        selected = []
        for index, base in enumerate(reports["native"]):
            if base["num_regimes"] < 2 or base["rule_mode"] != "multiregime" or base["num_classes"] < 3:
                continue
            labels = np.asarray(handle["y"][index, : base["support_size"]], dtype=np.int64)
            if np.any(labels < 0) or np.any(labels >= base["num_classes"]):
                raise ValueError(f"Invalid support label in episode {index}")
            majority = np.bincount(labels, minlength=base["num_classes"]).max() / len(labels)
            selected.append({
                "episode_id": index, "cell_id": base["cell_id"], "classes": base["num_classes"],
                "routing": base["task_family"], "mechanism": base["mechanism_mode"],
                "majority": float(majority),
                **{name: reports[name][index] for name in reports},
            })
    if len(selected) != 5760:
        raise ValueError(f"Expected 5,760 matched multiregime multiclass episodes, found {len(selected)}")
    cutoffs = {}
    for classes in (3, 4, 5):
        values = [row["majority"] for row in selected if row["classes"] == classes]
        cutoffs[classes] = [float(value) for value in np.quantile(values, [1 / 3, 2 / 3])]
    for row in selected:
        low, high = cutoffs[row["classes"]]
        row["balance"] = LABELS[0 if row["majority"] <= low else 1 if row["majority"] <= high else 2]

    class_slices = []
    for routing in ROUTINGS:
        for classes in (3, 4, 5):
            group = [row for row in selected if row["routing"] == routing and row["classes"] == classes]
            summary = {"routing": routing, "classes": classes, **_summarize(group)}
            if summary["episodes"] != 960 or summary["cells"] != 120:
                raise ValueError(f"Unexpected class-count slice: {routing}/{classes}")
            class_slices.append(summary)

    mechanism_slices = []
    for routing in ROUTINGS:
        for mechanism in ("r_z", "g_z"):
            group = [row for row in selected if row["routing"] == routing and row["mechanism"] == mechanism]
            summary = {"routing": routing, "mechanism": mechanism, **_summarize(group)}
            if summary["episodes"] != 1440 or summary["cells"] != 180:
                raise ValueError(f"Unexpected mechanism slice: {routing}/{mechanism}")
            mechanism_slices.append(summary)

    detailed = []
    pooled = []
    for routing in ROUTINGS:
        for balance in LABELS:
            class_summaries = []
            for classes in (3, 4, 5):
                group = [row for row in selected if row["routing"] == routing and row["balance"] == balance and row["classes"] == classes]
                summary = {"classes": classes, "routing": routing, "balance": balance, **_summarize(group)}
                detailed.append(summary)
                class_summaries.append(summary)
            pooled.append({
                "routing": routing, "balance": balance,
                "episodes": sum(row["episodes"] for row in class_summaries),
                "cells": sum(row["cells"] for row in class_summaries),
                **{candidate: {
                    metric: float(np.mean([row[candidate][metric] for row in class_summaries]))
                    for metric in ("ce", "accuracy_pp")
                } for candidate in MODELS},
            })
    output = {
        "bank": str(args.bank.relative_to(ROOT) if args.bank.is_relative_to(ROOT) else args.bank),
        "bank_sha256": _sha256(args.bank),
        "training_lock_sha256": _sha256(ROOT / "uv.lock"),
        "alignment_check": "all report episode metadata and all binary support/query label rates match the local bank",
        "definition": "within-class-count support-majority tertiles; tied cutoff values remain in the lower bin",
        "estimand": "candidate-minus-native within-cell means; pooled routing rows average the three class counts equally",
        "cutoffs": {str(key): value for key, value in cutoffs.items()},
        "detailed": detailed, "pooled": pooled,
        "class_slices": class_slices, "mechanism_slices": mechanism_slices,
    }
    args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    args.main_tex.write_text(_tex_table(
        pooled,
        caption="Evaluation-balance sensitivity on the existing standard-input bank. Six-layer candidate minus native-control paired effects; lower CE and higher accuracy differences (pp) favor the candidate. Bins are defined by support majority fraction within each class count, without using outcomes; pooled rows give class counts equal weight. Descriptive fixed-checkpoint contrasts, not a control for training-label balance.",
        label="tab:balance", include_class=False,
    ))
    args.mechanism_tex.write_text(_mechanism_tex(mechanism_slices))
    parts = []
    for routing in ROUTINGS:
        parts.append(_tex_table(
            [row for row in detailed if row["routing"] == routing],
            caption=f"Detailed {routing.replace('_', ' ')} evaluation-balance slices. Candidate minus same-size native control; accuracy differences are percentage points. Cell means are weighted equally within each class-count slice. Support-majority cutoffs are shared across routing families within class count.",
            label=f"tab:balance-{routing}", include_class=True,
        ))
    args.appendix_tex.write_text("\n".join(parts))
    print(f"Verified {len(selected):,} paired episodes against {args.bank}")
    for row in pooled:
        print(row["routing"], row["balance"], row["episodes"], row["fixed"], row["curriculum"])


if __name__ == "__main__":
    main()
