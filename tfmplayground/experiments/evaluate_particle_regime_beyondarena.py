"""Evaluate trained comparison arms on the derived BeyondArena protocol.

This is deliberately not the standard BeyondArena leaderboard protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfmplayground.experiments.environment_adaptation import (
    BEYOND_ARENA_GROUPED,
    BEYOND_ARENA_TEMPORAL,
    GroupedClassificationData,
    TemporalClassificationData,
    evaluate_grouped_few_shot,
    evaluate_temporal_delayed,
    evaluate_temporal_few_shot,
    load_beyondarena_slice,
)
from tfmplayground.experiments.particle_benchmark import real_data_promotion
from tfmplayground.experiments.train_particle_regime_comparison import ComparisonConfig, _fresh_model
from tfmplayground.models.particle_online import BatchParticleOnlineClassifier, NanoTabPFNContextOnlineClassifier


def load_comparison_model(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device)
    config = ComparisonConfig(**{**payload["config"], "device": device})
    model, identity = _fresh_model(config, payload["initialization"])
    model.load_state_dict(payload["model"])
    if payload["identity"] != identity:
        raise ValueError("Official backbone or initialization identity changed since training.")
    return model.eval(), config, payload


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    model, config, payload = load_comparison_model(checkpoint, device)
    output = Path(output_dir) if output_dir else Path(checkpoint).parent / "beyondarena"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    identities = []
    for name in (*BEYOND_ARENA_TEMPORAL, *BEYOND_ARENA_GROUPED):
        sliced = load_beyondarena_slice(name)
        identities.append(
            {
                "dataset": name,
                "uuid": sliced.dataset_uuid,
                "checksum": sliced.checksum,
                "repeat": sliced.repeat,
                "fold": sliced.fold,
                "protocol": sliced.protocol,
            }
        )

        def particle_factory(temporal: bool = True):
            return BatchParticleOnlineClassifier(copy.deepcopy(model), device=device, temporal=temporal)

        def vanilla_factory(temporal: bool = True):
            return NanoTabPFNContextOnlineClassifier(
                copy.deepcopy(model.vanilla_backbone),
                device=device,
                context_limit=config.context_limit,
                temporal=temporal,
            )

        if isinstance(sliced.data, TemporalClassificationData):
            delayed = evaluate_temporal_delayed(
                sliced.data,
                {"particle": particle_factory(), "cumulative_vanilla": vanilla_factory()},
                batch_size=32,
            )
            rows.extend(
                {
                    "dataset": name,
                    "slice": "temporal_delayed",
                    "shots": "",
                    **summary,
                }
                for summary in delayed.summary()
            )
            for method, factory in (("particle", particle_factory), ("cumulative_vanilla", vanilla_factory)):
                rows.extend(
                    {"method": method, **row}
                    for row in evaluate_temporal_few_shot(sliced.data, factory, shots=(0, 8, 32, 128))
                )
        elif isinstance(sliced.data, GroupedClassificationData):
            for method, factory in (
                ("particle", lambda: particle_factory(False)),
                ("cumulative_vanilla", lambda: vanilla_factory(False)),
            ):
                rows.extend(
                    {"method": method, **row}
                    for row in evaluate_grouped_few_shot(
                        sliced.data,
                        factory,
                        shots=(0, 8, 32, 128),
                        seed=int(payload["seed"]),
                    )
                )
    _write(output / "metrics.csv", rows)
    (output / "dataset_identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    improvements: dict[str, list[float]] = {"temporal": [], "grouped": []}
    for name in (*BEYOND_ARENA_TEMPORAL, *BEYOND_ARENA_GROUPED):
        subset = [row for row in rows if row["dataset"] == name]
        slice_name = "temporal" if name in BEYOND_ARENA_TEMPORAL else "grouped"
        methods = {
            method: [float(row["auc"]) for row in subset if row["method"] == method and np.isfinite(float(row["auc"]))]
            for method in ("particle", "cumulative_vanilla")
        }
        if all(methods.values()):
            difference = np.mean(methods["particle"]) - np.mean(methods["cumulative_vanilla"])
            improvements[slice_name].append(float(difference))
    promotion = (
        real_data_promotion(improvements)
        if all(improvements.values())
        else {
            "passed": False,
            "classification": "incomplete_derived_protocol",
            "reason": "At least one temporal/grouped slice produced no finite paired AUC.",
        }
    )
    (output / "promotion.json").write_text(json.dumps(promotion, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(evaluate_checkpoint(args.checkpoint, output_dir=args.output_dir, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
