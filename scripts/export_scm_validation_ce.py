#!/usr/bin/env python3
"""Export validation trajectories from the saved realistic SCM runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def model_name(data: dict, path: Path) -> str:
    if "condition" in data:
        return str(data["condition"])
    if "model_type" in data:
        return str(data["model_type"])
    return path.parent.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = (
        args.results_root / "scm_table_slot_head_routing_realistic_5000_slurm" / "37062078",
        args.results_root / "scm_nanotabpfn_realistic_5000_slurm" / "37050453",
        args.results_root / "scm_table_slot_variants_realistic_5000_slurm" / "37050501",
    )
    rows = []
    for root in roots:
        for path in sorted(root.glob("**/*.json")):
            data = json.loads(path.read_text())
            if "history" not in data:
                continue
            model = model_name(data, path)
            seed = int(data["seed"])
            for item in data["history"]:
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "step": int(item["step"]),
                        "validation_cross_entropy": float(item["cross_entropy"]["mean"]),
                        "validation_accuracy": float(item["accuracy"]["mean"]),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} trajectory rows to {args.output}")


if __name__ == "__main__":
    main()
