#!/usr/bin/env python3
"""Accuracy + ROC AUC per family for a completed v3 pilot run.

The pilot's own evaluate() only recorded log_loss/brier/accuracy/ECE, not
AUC. This reloads each cell's checkpoint into a freshly built model and
rebuilds the same deterministic evaluation bank (same seed/config), so no
retraining is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v3 import FAMILIES, OriginalPrior
from tfmplayground.experiments.pretrain_multiregime_v3 import (
    PilotConfig,
    build_model,
    evaluation_bank,
    log_predictions,
)

CELL_KIND = {0: "plain", 1: "slot", 2: "plain", 3: "slot", 4: "plain", 5: "slot"}
CELL_MODE = {0: "original", 1: "original", 2: "fixed", 3: "fixed", 4: "curriculum", 5: "curriculum"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = []
    for index in range(6):
        cell_dir = args.run_root / f"cell-{index}"
        config_data = json.loads((cell_dir / "config.json").read_text())["config"]
        config = PilotConfig(**{**config_data, "device": args.device})
        kind, mode = CELL_KIND[index], CELL_MODE[index]

        model, _ = build_model(config, kind)
        state = torch.load(cell_dir / "checkpoint.pth", map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        original = OriginalPrior(config.generator())
        bank = evaluation_bank(config, original)

        row = {"cell": index, "kind": kind, "mode": mode}
        with torch.no_grad():
            for family in FAMILIES:
                labels, probabilities = [], []
                for episode in bank[family]:
                    logs, _ = log_predictions(model, [episode], args.device)
                    p = logs[0, :, 1].exp().cpu().numpy()
                    y = episode.query_y.numpy().reshape(-1)
                    labels.append(y)
                    probabilities.append(p)
                y = np.concatenate(labels)
                p = np.concatenate(probabilities)
                row[f"{family}_accuracy"] = float(np.mean((p >= 0.5) == y))
                row[f"{family}_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = args.run_root / "accuracy_auc.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
