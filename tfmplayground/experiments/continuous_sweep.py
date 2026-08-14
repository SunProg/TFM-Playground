"""Screening grid and selection for the continuous-uncertainty sweep.

The grid is defined here rather than in the Slurm script so that the array
index to configuration mapping is testable and reproducible.

Base totals: 18 adapter-continuous, 6 frozen-continuous, 4 full-copy
continuous, and 6 Beta-adapter configurations.  Every base configuration is
run at both moment weights, giving 68 screening tasks.  The moment weight is
applied to all four architectures rather than only the primary one so that the
adapter-versus-Beta and adapter-versus-frozen comparisons stay like for like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ADAPTER_BOTTLENECKS = (16, 32, 64)
LATENT_DIMENSIONS = (16, 32, 64)
HEAD_LEARNING_RATES = (1e-4, 3e-4)
FULL_LATENT_DIMENSIONS = (32, 64)
FULL_LEARNING_RATES = (1e-5, 3e-5)
TRAINING_SAMPLES = 32
#: Multiplier on the mutual-information and variance loss weights.
MOMENT_WEIGHTS = (1.0, 4.0)

SCREENING_STEPS = 1500
SCREENING_VALIDATION_INTERVAL = 100
SCREENING_PATIENCE = 8
FINAL_STEPS = 5000
FINAL_VALIDATION_INTERVAL = 100
FINAL_PATIENCE = 10
MIN_DELTA = 1e-4
#: Seeds used for the three-seed final training of each selected configuration.
FINAL_SEEDS = (2402, 2403, 2404)


def screening_configurations() -> list[dict[str, Any]]:
    """Every screened configuration in a fixed, documented order."""
    return [
        {**configuration, "moment_weight": moment_weight}
        for configuration in _base_configurations()
        for moment_weight in MOMENT_WEIGHTS
    ]


def _base_configurations() -> list[dict[str, Any]]:
    """The architecture grid before the moment-weight axis is applied."""
    configurations: list[dict[str, Any]] = []
    for bottleneck in ADAPTER_BOTTLENECKS:
        for latent in LATENT_DIMENSIONS:
            for learning_rate in HEAD_LEARNING_RATES:
                configurations.append(
                    {
                        "architecture": "adapter_continuous",
                        "model_type": "continuous",
                        "uncertainty_mode": "adapters",
                        "adapter_bottleneck": bottleneck,
                        "latent_dim": latent,
                        "learning_rate": learning_rate,
                        "num_samples": TRAINING_SAMPLES,
                    }
                )
    for latent in LATENT_DIMENSIONS:
        for learning_rate in HEAD_LEARNING_RATES:
            configurations.append(
                {
                    "architecture": "frozen_continuous",
                    "model_type": "continuous",
                    "uncertainty_mode": "frozen",
                    "adapter_bottleneck": 32,
                    "latent_dim": latent,
                    "learning_rate": learning_rate,
                    "num_samples": TRAINING_SAMPLES,
                }
            )
    for latent in FULL_LATENT_DIMENSIONS:
        for learning_rate in FULL_LEARNING_RATES:
            configurations.append(
                {
                    "architecture": "full_continuous",
                    "model_type": "continuous",
                    "uncertainty_mode": "full",
                    "adapter_bottleneck": 32,
                    "latent_dim": latent,
                    "learning_rate": learning_rate,
                    "num_samples": TRAINING_SAMPLES,
                }
            )
    for bottleneck in ADAPTER_BOTTLENECKS:
        for learning_rate in HEAD_LEARNING_RATES:
            configurations.append(
                {
                    "architecture": "beta_adapter",
                    "model_type": "beta",
                    "uncertainty_mode": "adapters",
                    "adapter_bottleneck": bottleneck,
                    "latent_dim": 32,
                    "learning_rate": learning_rate,
                    "num_samples": TRAINING_SAMPLES,
                }
            )
    return configurations


def configuration_label(configuration: dict[str, Any]) -> str:
    return (
        f"{configuration['architecture']}"
        f"-b{configuration['adapter_bottleneck']}"
        f"-z{configuration['latent_dim']}"
        f"-lr{configuration['learning_rate']:g}"
        f"-m{configuration['moment_weight']:g}"
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
        f"--model-type {configuration['model_type']}",
        f"--uncertainty-mode {configuration['uncertainty_mode']}",
        f"--adapter-bottleneck {configuration['adapter_bottleneck']}",
        f"--latent-dim {configuration['latent_dim']}",
        f"--num-samples {configuration['num_samples']}",
        f"--learning-rate {configuration['learning_rate']:g}",
        f"--moment-weight {configuration['moment_weight']:g}",
        f"--max-steps {steps}",
        f"--validation-interval {FINAL_VALIDATION_INTERVAL if final else SCREENING_VALIDATION_INTERVAL}",
        f"--patience {patience}",
        f"--min-delta {MIN_DELTA:g}",
    ]
    if seed is not None:
        flags.append(f"--seed {seed}")
    return " ".join(flags)


def _read_run(directory: Path) -> dict[str, Any] | None:
    selection_path = directory / "selection.json"
    config_path = directory / "config.json"
    if not selection_path.is_file() or not config_path.is_file():
        return None
    selection = json.loads(selection_path.read_text())
    configuration = json.loads(config_path.read_text())
    architecture = {
        ("continuous", "adapters"): "adapter_continuous",
        ("continuous", "frozen"): "frozen_continuous",
        ("continuous", "full"): "full_continuous",
        ("beta", "adapters"): "beta_adapter",
    }.get((configuration["model_type"], configuration["uncertainty_mode"]))
    return {
        "run_dir": str(directory),
        "architecture": architecture,
        "adapter_bottleneck": configuration["adapter_bottleneck"],
        "latent_dim": configuration["latent_dim"],
        "learning_rate": configuration["learning_rate"],
        "moment_weight": configuration.get("moment_weight", 1.0),
        "seed": configuration["seed"],
        "best_step": selection.get("best_step"),
        "validation_selection_loss": selection.get("best_validation_selection_loss"),
        "checkpoint": selection.get("selected_checkpoint"),
    }


def summarize_sweep(sweep_dir: str | Path, *, top_k: int = 2) -> dict[str, Any]:
    """Rank screening runs by held-out-family validation loss.

    Selection uses only held-out *families*, so a configuration cannot win by
    memorizing the training generators.
    """
    root = Path(sweep_dir)
    # ``rglob`` finds the config file; ``_read_run`` receives its directory.
    runs = [
        run
        for run in (_read_run(path.parent) for path in sorted(root.rglob("config.json")))
        if run and run["validation_selection_loss"] is not None
    ]
    by_architecture: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_architecture.setdefault(run["architecture"], []).append(run)
    selected = {
        architecture: sorted(entries, key=lambda item: item["validation_selection_loss"])[:top_k]
        for architecture, entries in by_architecture.items()
    }
    return {"runs": runs, "selected": selected, "top_k": top_k, "final_seeds": list(FINAL_SEEDS)}


def sweep_csv(summary: dict[str, Any]) -> str:
    keys = (
        "architecture",
        "adapter_bottleneck",
        "latent_dim",
        "learning_rate",
        "moment_weight",
        "seed",
        "best_step",
        "validation_selection_loss",
        "run_dir",
    )
    rows = sorted(
        summary["runs"], key=lambda item: (item["architecture"] or "", item["validation_selection_loss"])
    )
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
