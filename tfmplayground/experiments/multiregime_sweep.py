"""Screening grid and selection for the multiregime backbone fine-tune.

Same structure as `continuous_sweep.py`: the array index to configuration
mapping lives here (not in the Slurm script) so it is testable and
reproducible, and `summarize_sweep` ranks completed runs by held-out
cross-entropy to pick finalists for a three-seed final run.

Grid: learning rate x multiregime curriculum probability x support size,
18 base configurations, single screening seed each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LEARNING_RATES = (1e-5, 3e-5, 1e-4)
MULTIREGIME_PROBABILITIES = (0.2, 0.3, 0.5)
SUPPORT_SIZES = (64, 128)

#: 100,000 steps, no early stopping: patience larger than max_steps / validation
#: interval means the "stale >= patience" check in finetune() can never trigger
#: within the run, so every task uses its full step budget.
SCREENING_STEPS = 100_000
SCREENING_VALIDATION_INTERVAL = 100
SCREENING_PATIENCE = 10_000_000
FINAL_STEPS = 3000
FINAL_VALIDATION_INTERVAL = 100
FINAL_PATIENCE = 10
MIN_DELTA = 1e-4
SCREENING_SEED = 2402
#: Seeds used for the three-seed final training of each selected configuration.
FINAL_SEEDS = (2402, 2403, 2404)


def screening_configurations() -> list[dict[str, Any]]:
    """Every screened configuration in a fixed, documented order."""
    return [
        {"learning_rate": learning_rate, "multiregime_probability": probability, "support_size": support_size}
        for learning_rate in LEARNING_RATES
        for probability in MULTIREGIME_PROBABILITIES
        for support_size in SUPPORT_SIZES
    ]


def configuration_label(configuration: dict[str, Any]) -> str:
    return (
        f"lr{configuration['learning_rate']:g}"
        f"-p{configuration['multiregime_probability']:g}"
        f"-s{configuration['support_size']}"
    )


def configuration_flags(index: int, *, final: bool = False, seed: int | None = None) -> str:
    """Command-line flags for one screening (or final) array task."""
    configurations = screening_configurations()
    if not 0 <= index < len(configurations):
        raise IndexError(f"Array index {index} is outside 0..{len(configurations) - 1}.")
    configuration = configurations[index]
    steps = FINAL_STEPS if final else SCREENING_STEPS
    patience = FINAL_PATIENCE if final else SCREENING_PATIENCE
    flags = [
        f"--learning-rate {configuration['learning_rate']:g}",
        f"--multiregime-probability {configuration['multiregime_probability']:g}",
        f"--support-size {configuration['support_size']}",
        f"--max-steps {steps}",
        f"--validation-interval {FINAL_VALIDATION_INTERVAL if final else SCREENING_VALIDATION_INTERVAL}",
        f"--patience {patience}",
        f"--min-delta {MIN_DELTA:g}",
        f"--seed {seed if seed is not None else SCREENING_SEED}",
    ]
    return " ".join(flags)


def _read_run(directory: Path) -> dict[str, Any] | None:
    selection_path = directory / "selection.json"
    config_path = directory / "config.json"
    baseline_path = directory / "baseline.json"
    if not selection_path.is_file() or not config_path.is_file():
        return None
    selection = json.loads(selection_path.read_text())
    configuration = json.loads(config_path.read_text())
    baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else selection.get("baseline")
    return {
        "run_dir": str(directory),
        "learning_rate": configuration["learning_rate"],
        "multiregime_probability": configuration["multiregime_probability"],
        "support_size": configuration["support_size"],
        "seed": configuration["seed"],
        "best_step": selection.get("best_step"),
        "best_validation_cross_entropy": selection.get("best_validation_cross_entropy"),
        "baseline_multiregime_error_detection_auroc": (baseline or {}).get("multiregime_error_detection_auroc"),
    }


def summarize_sweep(sweep_dir: str | Path, *, top_k: int = 2) -> dict[str, Any]:
    """Rank screening runs by held-out cross-entropy; lower is better."""
    root = Path(sweep_dir)
    runs = [
        run
        for run in (_read_run(path.parent) for path in sorted(root.rglob("config.json")))
        if run and run["best_validation_cross_entropy"] is not None
    ]
    ranked = sorted(runs, key=lambda item: item["best_validation_cross_entropy"])
    return {"runs": runs, "selected": ranked[:top_k], "top_k": top_k, "final_seeds": list(FINAL_SEEDS)}


def sweep_csv(summary: dict[str, Any]) -> str:
    keys = (
        "learning_rate",
        "multiregime_probability",
        "support_size",
        "seed",
        "best_step",
        "best_validation_cross_entropy",
        "baseline_multiregime_error_detection_auroc",
        "run_dir",
    )
    rows = sorted(summary["runs"], key=lambda item: item["best_validation_cross_entropy"])
    body = "\n".join(",".join(str(row.get(key, "")) for key in keys) for row in rows)
    return ",".join(keys) + "\n" + body + ("\n" if rows else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, help="Print the flags for one screening array index.")
    parser.add_argument("--final", action="store_true", help="Use the final training budget for --index.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", action="store_true", help="Print the number of screening configurations.")
    parser.add_argument("--label", type=int, help="Print the label of one screening array index.")
    parser.add_argument("--summarize", help="Rank the runs beneath a sweep directory.")
    parser.add_argument("--output-dir", default=None)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.count:
        print(len(screening_configurations()))
    elif arguments.label is not None:
        print(configuration_label(screening_configurations()[arguments.label]))
    elif arguments.index is not None:
        print(configuration_flags(arguments.index, final=arguments.final, seed=arguments.seed))
    elif arguments.summarize:
        result = summarize_sweep(arguments.summarize)
        if arguments.output_dir:
            output = Path(arguments.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "sweep_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "sweep_runs.csv").write_text(sweep_csv(result))
        print(json.dumps(result["selected"], indent=2))
    else:
        raise SystemExit("Choose one of --index, --label, --count, or --summarize.")
