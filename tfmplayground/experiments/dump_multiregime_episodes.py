"""Pre-generate multiregime episodes into an HDF5 dump.

Generating SCM multiregime episodes is expensive and happens on the CPU, which
made the multiregime training arm run at roughly half the rate of the
single-regime one -- measured at 39 against 85 steps/min, i.e. about 0.83 s per
step spent building 8 episodes, or ~11.5 h of a 21.5 h run.  Dumping the
episodes once and streaming them from disk removes essentially all of that, the
same trick the repository already uses for the ordinary prior
(``external_priors.PriorDumpDataLoader``).

The layout follows ``external_priors.base.dump_prior_to_h5`` -- one padded table
per episode with ``num_features`` / ``num_datapoints`` / ``train_test_split_index``
-- and adds what multiregime work needs and the generic dump has no place for:

``regime_source``
    Per-row 0/1 tag over the concatenated support+query table, saying which of
    the two label functions actually produced each row's label.  Diagnostic
    only; it must never be fed to a model.  Without it the slot-binding scores
    cannot be computed, which is the whole reason a plain prior dump will not do.
``candidate_positive`` / ``posterior``
    The two candidates' class-1 probabilities and the exact Bayes posterior.
    Unused by the slot mixture loss, stored so the dump can also serve the
    teacher-based objectives rather than having to be regenerated for them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import (
    SCM_FAMILIES,
    TRAIN_REGIME,
    ContinuousEpisode,
    sample_scm_multiregime_episode,
)

#: The balanced within- and cross-family mixture `multiregime_batch` draws from.
MULTIREGIME_SOURCES: tuple[str | tuple[str, str], ...] = (*SCM_FAMILIES, ("mlp_scm", "tree_scm"))


@dataclass(frozen=True)
class DumpConfig:
    output: str
    episodes: int = 100_000
    batch_size: int = 8
    support_size: int = 128
    query_count: int = 32
    contamination: float = 0.3
    noise: float = 0.0
    max_features: int = 12
    seed: int = 2402
    shard_index: int = 0
    num_shards: int = 1

    @property
    def rows(self) -> int:
        return self.support_size + self.query_count


def _shard_episodes(config: DumpConfig) -> int:
    """Episodes this shard is responsible for, splitting any remainder evenly."""
    base, extra = divmod(config.episodes, config.num_shards)
    return base + (1 if config.shard_index < extra else 0)


def dump_multiregime_episodes(config: DumpConfig) -> Path:
    """Generate this shard's episodes and write them to ``config.output``."""
    if config.num_shards < 1 or not 0 <= config.shard_index < config.num_shards:
        raise ValueError("shard_index must lie in [0, num_shards).")
    if config.episodes < 1:
        raise ValueError("episodes must be positive.")
    path = Path(config.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = _shard_episodes(config)
    rows, features = config.rows, config.max_features
    # Each shard owns a disjoint seed stream, so shards are independent and the
    # union is reproducible from (seed, num_shards).
    rng = np.random.default_rng([config.seed, config.shard_index])

    with h5py.File(path, "w") as handle:
        chunk = min(config.batch_size, target)
        dump_x = handle.create_dataset(
            "X",
            shape=(0, rows, features),
            maxshape=(None, rows, features),
            chunks=(chunk, rows, features),
            compression="lzf",
            dtype="f4",
        )
        dump_y = handle.create_dataset(
            "y", shape=(0, rows), maxshape=(None, rows), chunks=(chunk, rows), compression="lzf", dtype="f4"
        )
        dump_regime = handle.create_dataset(
            "regime_source",
            shape=(0, rows),
            maxshape=(None, rows),
            chunks=(chunk, rows),
            compression="lzf",
            dtype="i1",
        )
        dump_candidate = handle.create_dataset(
            "candidate_positive",
            shape=(0, 2, rows),
            maxshape=(None, 2, rows),
            chunks=(chunk, 2, rows),
            compression="lzf",
            dtype="f4",
        )
        dump_posterior = handle.create_dataset(
            "posterior", shape=(0, 2), maxshape=(None, 2), chunks=(chunk, 2), dtype="f4"
        )
        dump_features = handle.create_dataset("num_features", shape=(0,), maxshape=(None,), dtype="i4")
        handle.create_dataset("num_datapoints", data=np.array((rows,)))
        handle.create_dataset("train_test_split_index", data=np.array((config.support_size,)))
        handle.create_dataset("contamination", data=np.array((config.contamination,)))
        handle.create_dataset("problem_type", data="classification", dtype=h5py.string_dtype())

        written = 0
        while written < target:
            source = MULTIREGIME_SOURCES[int(rng.integers(len(MULTIREGIME_SOURCES)))]
            episode = sample_scm_multiregime_episode(
                rng,
                regime=TRAIN_REGIME,
                family=source,
                batch_size=min(config.batch_size, target - written),
                support_size=config.support_size,
                query_count=config.query_count,
                noise=config.noise,
                contamination=config.contamination,
                device="cpu",
            )
            count = episode.support_x.shape[0]
            observed = episode.support_x.shape[2]
            if observed > features:
                raise ValueError(f"Episode has {observed} features, exceeding max_features={features}.")

            x = torch.cat((episode.support_x, episode.query_x), dim=1).numpy()
            x = np.pad(x, ((0, 0), (0, 0), (0, features - observed)))
            y = torch.cat((episode.support_y, episode.query_y.float()), dim=1).numpy()
            regime = torch.cat((episode.support_regime_source, episode.query_regime_source), dim=1).numpy()
            candidate = torch.cat((episode.candidate_support_positive, episode.candidate_query_positive), dim=2).numpy()

            for dataset, value in (
                (dump_x, x),
                (dump_y, y),
                (dump_regime, regime),
                (dump_candidate, candidate),
                (dump_posterior, episode.posterior.numpy()),
                (dump_features, np.full(count, observed, dtype="i4")),
            ):
                dataset.resize(dataset.shape[0] + count, axis=0)
                dataset[-count:] = value
            written += count
            if written % (config.batch_size * 250) < config.batch_size:
                print(f"{written}/{target} episodes", flush=True)
    print(f"wrote {written} episodes to {path}", flush=True)
    return path


class MultiregimeDumpLoader:
    """Stream pre-generated multiregime episodes as ``ContinuousEpisode`` batches.

    Accepts a single ``.h5`` file or a directory of shards, which are read in
    sorted order and then cycled.  Cycling means an arm sees each episode more
    than once over a long run -- the diversity that on-the-fly generation buys is
    what a dump trades away for speed, so size the dump accordingly.
    """

    def __init__(self, path: str | Path, *, batch_size: int = 8, device: str | torch.device = "cpu", seed: int = 0):
        self.paths = self._resolve(Path(path))
        self.batch_size = batch_size
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._file_index = 0
        self._pointer = 0
        self._handle: h5py.File | None = None
        self._open(self.paths[0])

    @staticmethod
    def _resolve(path: Path) -> list[Path]:
        if path.is_dir():
            shards = sorted(path.glob("*.h5"))
            if not shards:
                raise FileNotFoundError(f"No .h5 shards under {path}.")
            return shards
        if not path.is_file():
            raise FileNotFoundError(f"No multiregime dump at {path}.")
        return [path]

    def _open(self, path: Path) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = h5py.File(path, "r")
        self._pointer = 0
        self._split = int(self._handle["train_test_split_index"][0])
        self._count = self._handle["X"].shape[0]

    def _advance_file(self) -> None:
        self._file_index = (self._file_index + 1) % len(self.paths)
        self._open(self.paths[self._file_index])

    def sample(self) -> ContinuousEpisode:
        """Return the next batch, moving to the next shard when one is exhausted."""
        assert self._handle is not None
        if self._pointer + self.batch_size > self._count:
            self._advance_file()
        stop = self._pointer + self.batch_size
        handle, split = self._handle, self._split
        features = int(handle["num_features"][self._pointer : stop].max())
        x = torch.from_numpy(handle["X"][self._pointer : stop, :, :features])
        y = torch.from_numpy(handle["y"][self._pointer : stop])
        regime = torch.from_numpy(handle["regime_source"][self._pointer : stop]).long()
        candidate = torch.from_numpy(handle["candidate_positive"][self._pointer : stop])
        posterior = torch.from_numpy(handle["posterior"][self._pointer : stop])
        contamination = float(handle["contamination"][0])
        self._pointer = stop

        device = self.device
        return ContinuousEpisode(
            x[:, :split].to(device),
            y[:, :split].to(device),
            x[:, split:].to(device),
            y[:, split:].long().to(device),
            candidate[:, :, split:].to(device),
            candidate[:, :, :split].to(device),
            posterior.to(device),
            0.0,
            "multiregime",
            "dump",
            {"contamination": contamination, "source": str(self.paths[self._file_index])},
            regime[:, split:].to(device),
            regime[:, :split].to(device),
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def build_parser() -> argparse.ArgumentParser:
    defaults = DumpConfig(output="")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for name in (
        "episodes",
        "batch_size",
        "support_size",
        "query_count",
        "max_features",
        "seed",
        "shard_index",
        "num_shards",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("contamination", "noise"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    return parser


def main(argv: list[str] | None = None) -> Path:
    return dump_multiregime_episodes(DumpConfig(**vars(build_parser().parse_args(argv))))


if __name__ == "__main__":
    print(main())
