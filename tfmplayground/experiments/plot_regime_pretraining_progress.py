"""Plot matched plain versus multiregime pretraining trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_epoch_rows(run_directories: list[str]) -> dict[str, dict[int, list[float]]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for directory in run_directories:
        history = Path(directory) / "history.jsonl"
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if "tabarena_mean_roc_auc" not in row:
                continue
            epoch = int(row["epoch"])
            for metric in ("query_cross_entropy", "tabarena_mean_roc_auc", "tabarena_mean_accuracy"):
                values[metric][epoch].append(float(row[metric]))
    return values


def _plot_metric(axis, plain, multiregime, metric: str, label: str) -> None:
    for name, values, color in (("plain", plain, "C0"), ("multiregime", multiregime, "C1")):
        epochs = sorted(values[metric])
        mean = np.asarray([np.mean(values[metric][epoch]) for epoch in epochs])
        sem = np.asarray(
            [
                np.std(values[metric][epoch], ddof=1) / np.sqrt(len(values[metric][epoch]))
                if len(values[metric][epoch]) > 1
                else 0.0
                for epoch in epochs
            ]
        )
        axis.plot(epochs, mean, label=name, color=color)
        axis.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.2)
    axis.set(xlabel="epoch (1,000 updates)", ylabel=label)
    axis.legend()
    axis.grid(alpha=0.25)


def plot(plain_runs: list[str], multiregime_runs: list[str], output: str | Path) -> Path:
    plain = _load_epoch_rows(plain_runs)
    multiregime = _load_epoch_rows(multiregime_runs)
    required = {"query_cross_entropy", "tabarena_mean_roc_auc", "tabarena_mean_accuracy"}
    if set(plain) != required or set(multiregime) != required:
        raise ValueError("Each run must contain epoch-level TabArena metrics from the pretraining runner.")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    _plot_metric(axes[0], plain, multiregime, "query_cross_entropy", "training query cross-entropy")
    _plot_metric(axes[1], plain, multiregime, "tabarena_mean_roc_auc", "TabArena-small mean ROC-AUC")
    _plot_metric(axes[2], plain, multiregime, "tabarena_mean_accuracy", "TabArena-small mean accuracy")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-run", action="append", required=True)
    parser.add_argument("--multiregime-run", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(plot(args.plain_run, args.multiregime_run, args.output))
