"""Select the v2 auxiliary weight from held-out predictive log loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tfmplayground.experiments.pretrain_multiregime_v2 import select_auxiliary_weight

SEEDS = (2402, 2403, 2404)
ARMS = {
    0.01: "mufasa-supervised-lambda0.01",
    0.1: "mufasa-supervised-lambda0.1",
    1.0: "mufasa-supervised-lambda1",
}


def select(evaluation_root: Path, output: Path) -> dict:
    losses: dict[float, float] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for weight, arm in ARMS.items():
        seed_losses = []
        for seed in SEEDS:
            path = evaluation_root / arm / f"seed-{seed}" / "summary.csv"
            with path.open(newline="") as source:
                rows = list(csv.DictReader(source))
            values = [
                float(row["mean"])
                for row in rows
                if row["model"] == "tabpfn"
                and row["scope"] == "overall"
                and row["metric"] == "log_loss"
                and row["backend"] == "analytic"
                and row["mean"]
            ]
            if not values:
                raise ValueError(f"No held-out predictive log losses in {path}.")
            mean = float(np.mean(values))
            seed_losses.append(mean)
            per_seed.setdefault(str(weight), {})[str(seed)] = mean
        losses[weight] = float(np.mean(seed_losses))
    selected = select_auxiliary_weight(losses)
    result = {
        "selected_aux_regime_weight": selected,
        "mean_prediction_log_loss": {str(key): value for key, value in losses.items()},
        "per_seed_prediction_log_loss": per_seed,
        "tie_break": "smaller_weight",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    return select(arguments.evaluation_root, arguments.output)


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
