"""Deterministic sharded prior dumps for the v2 training curriculum."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v2 import (
    RegimeEpisode,
    RegimeGeneratorConfig,
    episode_metadata_json,
    sample_regime_episode,
)

_TENSOR_FIELDS = (
    "support_x",
    "support_y",
    "query_x",
    "query_y",
    "support_z",
    "query_z",
    "active_regime_mask",
    "support_gate_probabilities",
    "query_gate_probabilities",
    "counterfactual_support_probabilities",
    "counterfactual_query_probabilities",
)


def _episode_from_payload(payload: dict[str, Any], index: int) -> RegimeEpisode:
    return RegimeEpisode(
        **{name: payload[name][index : index + 1] for name in _TENSOR_FIELDS},
        metadata=payload["metadata"][index],
    )


class PriorDumpReader:
    """Sequential reader for a sharded dump, loading one shard at a time."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.shards = [self.directory / name for name in self.manifest["shards"]]
        self._shard_index = 0
        self._offset = 0
        self._payload: dict[str, Any] | None = None
        self._global_index = 0

    @property
    def index(self) -> int:
        return self._global_index

    @property
    def total_episodes(self) -> int:
        return int(self.manifest["total_episodes"])

    def seek(self, index: int) -> None:
        if not 0 <= index <= self.total_episodes:
            raise ValueError(f"prior dump index must lie in [0, {self.total_episodes}].")
        self._shard_index = 0
        self._offset = 0
        self._payload = None
        self._global_index = 0
        while self._global_index < index:
            self.next_episode()

    def _load_shard(self) -> None:
        if self._shard_index >= len(self.shards):
            raise StopIteration
        self._payload = torch.load(self.shards[self._shard_index], map_location="cpu", weights_only=False)
        self._offset = 0

    def next_episode(self) -> tuple[RegimeEpisode, dict[str, Any]]:
        if self._payload is None:
            self._load_shard()
        assert self._payload is not None
        count = len(self._payload["metadata"])
        if self._offset >= count:
            self._shard_index += 1
            self._payload = None
            return self.next_episode()
        episode = _episode_from_payload(self._payload, self._offset)
        metadata = self._payload["metadata"][self._offset]
        self._offset += 1
        self._global_index += 1
        return episode, metadata


def write_prior_dump(
    output: str | Path,
    *,
    seed: int,
    max_steps: int = 10_000,
    micro_batch_size: int = 8,
    accumulate_gradients: int = 4,
    support_size: int = 128,
    query_size: int = 32,
    min_features: int = 2,
    max_features: int = 12,
    backend: str = "analytic",
    difference_components: tuple[str, ...] = ("coefficients",),
    label_noise: float = 0.0,
    gate_strength: float = 1.0,
    curriculum_mode: str = "mixed",
    shard_episodes: int = 4096,
) -> Path:
    """Generate exactly the curriculum stream consumed by one training seed."""
    from tfmplayground.experiments.pretrain_multiregime_v2 import curriculum_cell

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior dump {output_path}.")
    output_path.mkdir(parents=True)
    generator_config = RegimeGeneratorConfig(
        backend=backend,
        max_regimes=4,
        difference_components=difference_components,
        support_size=support_size,
        query_size=query_size,
        min_features=min_features,
        max_features=max_features,
        label_noise=label_noise,
        gate_strength=gate_strength,
        seed=seed,
        single_regime_source="matched",
    )
    total = max_steps * micro_batch_size * accumulate_gradients
    rng = np.random.default_rng(seed + 1)
    shard_names: list[str] = []
    metadata_path = output_path / "metadata.jsonl"
    metadata_file = metadata_path.open("w")
    try:
        records: list[RegimeEpisode] = []
        for step in range(1, max_steps + 1):
            for position in range(micro_batch_size * accumulate_gradients):
                cell = curriculum_cell(step, max_steps, rng, mode=curriculum_mode)
                episode_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
                config = generator_config.__class__(
                    **{
                        **asdict(generator_config),
                        "regime_separation": cell.alpha,
                        "imbalance_ratio": cell.imbalance_ratio,
                        "seed": episode_seed,
                    }
                )
                episode = sample_regime_episode(
                    config,
                    num_regimes=cell.num_regimes,
                    seed=episode_seed,
                    episode_id=f"train-{seed}-{step:06d}-{position + 1:03d}",
                )
                records.append(episode)
                metadata_file.write(episode_metadata_json(episode) + "\n")
                if len(records) >= shard_episodes:
                    shard_names.append(_write_shard(output_path, len(shard_names), records))
                    records = []
        if records:
            shard_names.append(_write_shard(output_path, len(shard_names), records))
    finally:
        metadata_file.close()
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "total_episodes": total,
        "max_steps": max_steps,
        "micro_batch_size": micro_batch_size,
        "accumulate_gradients": accumulate_gradients,
        "curriculum_mode": curriculum_mode,
        "shard_episodes": shard_episodes,
        "generator": asdict(generator_config),
        "shards": shard_names,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_path


def _write_shard(output: Path, index: int, episodes: list[RegimeEpisode]) -> str:
    name = f"shard-{index:04d}.pt"
    payload = {name: torch.cat([getattr(episode, name) for episode in episodes], dim=0) for name in _TENSOR_FIELDS}
    payload["metadata"] = [episode.metadata for episode in episodes]
    torch.save(payload, output / name)
    return name


def main(argv: list[str] | None = None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--accumulate-gradients", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--backend", choices=("analytic", "tabicl_scm"), default="analytic")
    parser.add_argument("--difference-components", nargs="+", default=("coefficients",))
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--gate-strength", type=float, default=1.0)
    parser.add_argument("--curriculum-mode", choices=("mixed", "k1_only"), default="mixed")
    parser.add_argument("--shard-episodes", type=int, default=4096)
    args = parser.parse_args(argv)
    return write_prior_dump(
        args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        accumulate_gradients=args.accumulate_gradients,
        support_size=args.support_size,
        query_size=args.query_size,
        min_features=args.min_features,
        max_features=args.max_features,
        backend=args.backend,
        difference_components=tuple(args.difference_components),
        label_noise=args.label_noise,
        gate_strength=args.gate_strength,
        curriculum_mode=args.curriculum_mode,
        shard_episodes=args.shard_episodes,
    )


if __name__ == "__main__":
    print(main())


__all__ = ["PriorDumpReader", "write_prior_dump"]
