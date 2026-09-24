#!/usr/bin/env python3
"""Verify a CREATE synthetic-baseline pull and write a small original-family comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_MODELS = (
    "logreg",
    "random_forest",
    "tabpfn-v2.2",
    "tabpfn-v2.2-finetuned",
    "tabpfn-v2.6",
    "tabpfn-v2.6-finetuned",
    "tabpfn-v3",
    "tabpfn-v3-finetuned",
)
EXPECTED_TABICL_VERSION = "2.1.1"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_digest(value: str) -> str:
    path = Path(value)
    digest = path.read_text().strip().split()[0] if path.exists() else value.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError(f"{value} is not a SHA256 digest or checksum-file path")
    return digest.lower()


def finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def verify(args) -> str:
    baseline = read_json(args.baseline)
    pilot = read_json(args.pilot_result)
    ours_rows = json.loads(args.ours_accuracy_auc.read_text())

    if baseline.get("complete") is not True:
        raise ValueError("baseline JSON is not marked complete")
    if baseline.get("pilot_source_hash") != pilot.get("source_hash"):
        raise ValueError("baseline did not use the pilot source hash recorded by CREATE")
    if baseline.get("package_versions", {}).get("tabicl") != EXPECTED_TABICL_VERSION:
        raise ValueError(
            f"baseline used tabicl=={baseline.get('package_versions', {}).get('tabicl')}; "
            f"expected tabicl=={EXPECTED_TABICL_VERSION}"
        )

    expected_hashes = pilot.get("evaluation_bank_hashes", {}).get("original")
    generated_hashes = baseline.get("episode_hashes", {}).get("original")
    if generated_hashes != expected_hashes or baseline.get("episode_hashes_match") is not True:
        raise ValueError("generated original episode hashes do not match the CREATE pilot")

    rows = [row for row in baseline.get("rows", []) if row.get("family") == "original"]
    expected_keys = {(model, episode) for model in EXPECTED_MODELS for episode in range(len(expected_hashes))}
    actual_keys = {(row.get("model"), row.get("episode")) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(f"expected {len(expected_keys)} unique model/episode rows, found {len(rows)}")
    if any("error" in row for row in rows):
        failures = [row for row in rows if "error" in row]
        raise ValueError(f"baseline contains model errors: {failures[:3]}")
    if any(row.get("episode_hash") != expected_hashes[row["episode"]] for row in rows):
        raise ValueError("one or more per-row episode hashes differ from the pilot")

    for model in EXPECTED_MODELS:
        summary = baseline.get("summary", {}).get(model, {}).get("original")
        if not summary:
            raise ValueError(f"missing summary for {model}/original")
        if summary.get("episodes") != len(expected_hashes) or summary.get("query_rows") != len(expected_hashes) * 32:
            raise ValueError(f"incomplete summary for {model}/original: {summary}")
        finite(summary.get("pooled_auc"), f"{model}/original pooled_auc")
        finite(summary.get("mean_episode_auc"), f"{model}/original mean_episode_auc")

    if args.stderr.exists():
        stderr = args.stderr.read_text()
        if "Traceback (most recent call last)" in stderr or "srun: error" in stderr:
            raise ValueError("Slurm stderr contains a job error")
    if args.stdout.exists() and "finished_at=" not in args.stdout.read_text():
        raise ValueError("Slurm stdout does not contain the completion marker")

    local_digest = sha256(args.baseline)
    if args.remote_sha256:
        expected_digest = remote_digest(args.remote_sha256)
        if local_digest != expected_digest:
            raise ValueError(f"pulled-file SHA256 mismatch: local={local_digest}, remote={expected_digest}")

    ours = next((row for row in ours_rows if row.get("cell") == args.ours_cell), None)
    if ours is None:
        raise ValueError(f"cell {args.ours_cell} not found in {args.ours_accuracy_auc}")
    ours_cell_result = read_json(args.ours_cell_result)
    ours_validation = ours_cell_result.get("validation", {}).get("original", {})

    lines = [
        "# CREATE original-family baseline verification",
        "",
        f"- Baseline JSON SHA256: `{local_digest}`",
        f"- Pilot episode hashes: exact match ({len(expected_hashes)}/{len(expected_hashes)})",
        f"- Model/episode rows: {len(rows)}/{len(expected_keys)}; model errors: 0",
        "",
        "| model | log_loss | accuracy | pooled AUC | mean episode AUC |",
        "|---|---:|---:|---:|---:|",
        (
            f"| OURS: plain/original | {finite(ours_validation['log_loss'], 'OURS log_loss'):.4f} | "
            f"{finite(ours_validation['accuracy'], 'OURS accuracy'):.3f} | "
            f"{finite(ours['original_auc'], 'OURS AUC'):.4f} | — |"
        ),
    ]
    for model in EXPECTED_MODELS:
        summary = baseline["summary"][model]["original"]
        lines.append(
            f"| {model} | {finite(summary['log_loss'], model + ' log_loss'):.4f} | "
            f"{finite(summary['accuracy'], model + ' accuracy'):.3f} | "
            f"{finite(summary['pooled_auc'], model + ' pooled_auc'):.4f} | "
            f"{finite(summary['mean_episode_auc'], model + ' mean_episode_auc'):.4f} |"
        )
    lines.extend(
        [
            "",
            "AUC is pooled over all 256 query rows per model; mean episode AUC is retained as a diagnostic.",
            "This is a comparison artifact only; it does not rewrite the paper table.",
            "",
        ]
    )
    args.comparison.write_text("\n".join(lines))
    return "\n".join(
        [
            "verification passed",
            f"rows={len(rows)} models={len(EXPECTED_MODELS)} episodes={len(expected_hashes)}",
            f"pilot_hashes=match sha256={local_digest}",
            f"comparison={args.comparison}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--ours-cell-result", type=Path, required=True)
    parser.add_argument("--ours-accuracy-auc", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--remote-sha256")
    parser.add_argument("--ours-cell", type=int, default=0)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    print(verify(args))


if __name__ == "__main__":
    main()
