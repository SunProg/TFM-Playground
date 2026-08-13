"""Rank static Bayesian nanoTabPFN Slurm-array trials by gated validation loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def summarize_sweep(sweep_dir: str | Path) -> Path:
    root = Path(sweep_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for selection_path in sorted(root.glob("trial-*/training/selection.json")):
        training_dir = selection_path.parent
        selection = json.loads(selection_path.read_text())
        config = json.loads((training_dir / "config.json").read_text())
        acceptance = selection["acceptance"]
        validation_loss = float(selection["validation_loss"])
        required_gates = (
            bool(acceptance.get("finite_validation_loss")),
            bool(acceptance.get("backbone_unchanged")),
            bool(acceptance.get("mean_preserved_at_1e-6")),
            bool(acceptance.get("ordinary_accuracy_identical")),
        )
        eligible = all(required_gates) and math.isfinite(validation_loss)
        rows.append(
            {
                "trial": training_dir.parent.name,
                "num_hypotheses": int(config["num_hypotheses"]),
                "learning_rate": float(config["learning_rate"]),
                "likelihood_temperature": float(config["likelihood_temperature"]),
                "validation_loss": validation_loss,
                "eligible": eligible,
                "all_acceptance_gates": bool(acceptance.get("passed")),
                "mean_preserved_at_1e-6": bool(acceptance.get("mean_preserved_at_1e-6")),
                "backbone_unchanged": bool(acceptance.get("backbone_unchanged")),
                "ordinary_accuracy_identical": bool(acceptance.get("ordinary_accuracy_identical")),
                "checkpoint": str((training_dir / "mean_preserving_frozen.pth").resolve()),
                "selection": str(selection_path.resolve()),
            }
        )
    if not rows:
        raise ValueError(f"No completed trial selection files found under {root}.")

    rows.sort(key=lambda row: (not row["eligible"], row["validation_loss"], row["trial"]))
    fieldnames = list(rows[0])
    with (root / "sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    selected = next((row for row in rows if row["eligible"]), None)
    report = {
        "completed_trials": len(rows),
        "eligible_trials": sum(bool(row["eligible"]) for row in rows),
        "selection_rule": (
            "minimum entropy- and cardinality-normalized validation loss after "
            "frozen-backbone, mean-preservation, ordinary-accuracy, and finite-loss gates"
        ),
        "selected": selected,
        "trials": rows,
    }
    (root / "sweep_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return root / "sweep_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Wrote sweep summary to {summarize_sweep(args.sweep_dir)}")
