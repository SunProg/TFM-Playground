"""Summarize the six v3 cells with episode-paired uncertainty and match checks."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(study: Path):
    cells = {}
    missing = []
    for index in range(6):
        path = study / f"cell-{index}" / "result.json"
        if path.is_file():
            cells[index] = json.loads(path.read_text())
        else:
            missing.append(index)
    checks = {}
    for left in (0, 2, 4):
        if left in cells and left + 1 in cells:
            for key in ("source_hash", "initial_backbone_hash", "evaluation_bank_hashes", "training_stream_hash"):
                checks[f"pair_{left}_{left + 1}_{key}"] = cells[left][key] == cells[left + 1][key]
    if cells:
        anchor = next(iter(cells.values()))
        checks["all_cells_evaluation_bank"] = all(
            c["evaluation_bank_hashes"] == anchor["evaluation_bank_hashes"] for c in cells.values()
        )
    records = {}
    for index, cell in cells.items():
        rows = [json.loads(line) for line in (study / f"cell-{index}" / "evaluation.jsonl").read_text().splitlines()]
        records[index] = {(r["family"], r["seed"]): r for r in rows if r["step"] == cell["config"]["steps"]}
    rng = np.random.default_rng(204)
    comparisons = []
    if all(checks.values()):
        # Negative differences mean the right-hand cell has lower loss.
        for left, right in ((0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (1, 3), (1, 5)):
            if left not in records or right not in records:
                continue
            for family in cells[left]["validation"]:
                keys = sorted(k for k in records[left] if k[0] == family)
                delta = np.array([records[right][k]["log_loss"] - records[left][k]["log_loss"] for k in keys])
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(-1)
                comparisons.append(
                    {
                        "left": left,
                        "right": right,
                        "family": family,
                        "delta_log_loss": float(delta.mean()),
                        "paired_episode_ci95": np.quantile(boot, (0.025, 0.975)).tolist(),
                    }
                )
    result = {
        "complete": not missing,
        "missing_cells": missing,
        "matching_checks": checks,
        "comparisons": comparisons,
        "cells": {
            str(i): {
                "kind": c["kind"],
                "mode": c["mode"],
                "validation": c["validation"],
                "composition": c["composition"],
                "elapsed_seconds": c["elapsed_seconds"],
            }
            for i, c in cells.items()
        },
    }
    (study / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Multi-regime v3 pilot",
        "",
        f"Completed cells: {len(cells)}/6.",
        "Single training seed; intervals resample evaluation episodes, not training seeds.",
        "",
        "| Cell | Model | Prior | Original NLL | Shared rule | Soft gate | Independent | Persistent |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, cell in cells.items():
        values = " | ".join(
            f"{cell['validation'][family]['log_loss']:.4f}"
            for family in ("original", "shared_rule", "soft_gate", "independent", "persistent")
        )
        lines.append(f"| {index} | {cell['kind']} | {cell['mode']} | {values} |")
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Negative delta favors the right cell.",
        "",
        "| Left → right | Family | Delta NLL | Episode bootstrap 95% interval |",
        "|---|---|---:|---|",
    ]
    for comparison in comparisons:
        lo, hi = comparison["paired_episode_ci95"]
        lines.append(
            f"| {comparison['left']} → {comparison['right']} | {comparison['family']} | "
            f"{comparison['delta_log_loss']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += [
        "",
        "## Matching checks",
        "",
        *[f"- {name}: {passed}" for name, passed in checks.items()],
        "",
        "Synthetic pilot results do not establish real-world transfer. Full metrics, grouping controls, "
        "and slot-recovery references are in each cell's result.json and evaluation.jsonl.",
        "",
    ]
    (study / "report.md").write_text("\n".join(lines))
    print(json.dumps({"complete": result["complete"], "missing_cells": missing, "matching_checks": checks}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    summarize(parser.parse_args().study)
