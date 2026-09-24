"""Pre-generate grouped, variable-size multiregime-v4 episodes into HDF5.

Each contiguous ``generation_group`` shares a TabICL-style table length,
support/query split, feature count, SCM backend, and sampled SCM
hyperparameters. Its member episodes remain independent draws. Tables are
padded to the configured maxima on disk; ``num_datapoints`` and
``train_test_split_index`` are stored per episode and the loader only forms a
batch from one generation group, so padding never reaches a model as rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch

from tfmplayground.experiments.multiregime_v4 import (
    FAMILIES,
    HP_PROFILES,
    MECHANISM_MODES,
    sample_episode_v4,
    sample_generation_group_v4,
)


@dataclass(frozen=True)
class DumpConfig:
    output: str
    episodes: int = 20_000
    generation_group_size: int = 8
    min_features: int = 4
    max_features: int = 10
    #: Target cardinality follows TabICL's binary-or-uniform mixture on this
    #: inclusive range. Production labels use one controlled class vector per
    #: cardinality in every regime; the paper setting uses 2--5.
    min_classes: int = 2
    max_classes: int = 2
    min_instances: int = 64
    max_instances: int = 256
    min_train_fraction: float = 0.1
    max_train_fraction: float = 0.9
    #: Set both for an exact fixed train/query geometry, e.g. the native
    #: TabICL 128-support / 32-query training setting.
    support_size: int | None = None
    query_size: int | None = None
    min_regimes: int = 2
    max_regimes: int = 4
    min_samples_per_regime: int = 32
    #: Compatibility-only; controlled labels use the complete generated
    #: episode as their native-style rank reference.
    calibration_size: int = 256
    #: Padded width must accommodate the widest sampled table plus expose_z's
    #: optional extra routing-score column.
    pad_features: int | None = None
    mlp_probability: float = 0.7
    seed: int = 2402
    shard_index: int = 0
    num_shards: int = 1
    #: Probability of the shared-rule control condition.
    expose_z_probability: float = 0.0
    #: Whether labels use one shared candidate rule or each row's latent
    #: regime-selected candidate. Re-running with the same seed and only this
    #: field changed produces paired X/metadata and different final labels.
    rule_mode: Literal["shared", "multiregime"] = "multiregime"
    #: ``r_z`` changes the SCM mechanism; ``g_z`` keeps one SCM score and
    #: selects a regime-specific label rule.
    mechanism_mode: Literal["r_z", "g_z"] = "r_z"
    #: ``native`` (default) = TabICL's prior with native Reg2Cls labels;
    #: ``production`` = the deprecated calibrated-label profile of the 2026-09 dumps.
    hp_profile: Literal["native", "production", "tabicl_test"] = "native"

    @property
    def mix_probs(self) -> tuple[float, float]:
        return self.mlp_probability, 1.0 - self.mlp_probability


def _shard_episodes(config: DumpConfig) -> int:
    base, extra = divmod(config.episodes, config.num_shards)
    return base + (1 if config.shard_index < extra else 0)


def _validate(config: DumpConfig) -> None:
    if config.episodes < 1:
        raise ValueError("episodes must be positive.")
    if config.generation_group_size < 1:
        raise ValueError("generation_group_size must be positive.")
    if config.num_shards < 1 or not 0 <= config.shard_index < config.num_shards:
        raise ValueError("shard_index must lie in [0, num_shards).")
    if not 0 <= config.mlp_probability <= 1:
        raise ValueError("mlp_probability must lie in [0, 1].")
    if not 0 <= config.expose_z_probability <= 1:
        raise ValueError("expose_z_probability must lie in [0, 1].")
    if config.rule_mode not in {"shared", "multiregime"}:
        raise ValueError("rule_mode must be shared or multiregime.")
    if config.mechanism_mode not in MECHANISM_MODES:
        raise ValueError(f"mechanism_mode must be one of {MECHANISM_MODES}.")
    if config.hp_profile not in HP_PROFILES:
        raise ValueError(f"hp_profile must be one of {HP_PROFILES}.")
    if not 1 <= config.min_regimes <= config.max_regimes:
        raise ValueError("Regime bounds must satisfy 1 <= min_regimes <= max_regimes.")
    if config.min_classes != 2 or config.max_classes < config.min_classes:
        raise ValueError("Class bounds must satisfy min_classes == 2 <= max_classes.")
    if (config.support_size is None) != (config.query_size is None):
        raise ValueError("support_size and query_size must be supplied together.")
    if config.support_size is not None:
        if config.support_size < 1 or config.query_size is None or config.query_size < 1:
            raise ValueError("support_size and query_size must both be positive.")
        if not config.min_instances <= config.support_size + config.query_size <= config.max_instances:
            raise ValueError("support_size plus query_size must lie within instance bounds.")
    if config.min_samples_per_regime < 1:
        raise ValueError("min_samples_per_regime must be positive.")
    if config.pad_features is not None and config.pad_features < config.max_features + 1:
        raise ValueError("pad_features must fit max_features plus expose_z's extra column.")


def dump_multiregime_v4_episodes(config: DumpConfig) -> Path:
    """Generate this shard's grouped episodes and write them to ``output``."""
    _validate(config)
    path = Path(config.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = _shard_episodes(config)
    pad_features = config.max_features + 1 if config.pad_features is None else config.pad_features
    rng = np.random.default_rng([config.seed, config.shard_index])

    with h5py.File(path, "w") as handle:
        chunk = min(64, target)
        dump_x = handle.create_dataset(
            "X",
            shape=(0, config.max_instances, pad_features),
            maxshape=(None, config.max_instances, pad_features),
            chunks=(chunk, config.max_instances, pad_features),
            compression="lzf",
            dtype="f4",
        )
        dump_y = handle.create_dataset(
            "y",
            shape=(0, config.max_instances),
            maxshape=(None, config.max_instances),
            chunks=(chunk, config.max_instances),
            compression="lzf",
            dtype="f4",
        )
        dump_regime = handle.create_dataset(
            "regime",
            shape=(0, config.max_instances),
            maxshape=(None, config.max_instances),
            chunks=(chunk, config.max_instances),
            compression="lzf",
            dtype="i1",
        )
        dump_family = handle.create_dataset("family_index", shape=(0,), maxshape=(None,), dtype="i1")
        dump_mechanism = handle.create_dataset("mechanism_mode_index", shape=(0,), maxshape=(None,), dtype="i1")
        dump_expose_z = handle.create_dataset("expose_z", shape=(0,), maxshape=(None,), dtype="i1")
        dump_z_column_index = handle.create_dataset("z_column_index", shape=(0,), maxshape=(None,), dtype="i4")
        dump_features = handle.create_dataset("num_features", shape=(0,), maxshape=(None,), dtype="i4")
        dump_rows = handle.create_dataset("num_datapoints", shape=(0,), maxshape=(None,), dtype="i4")
        dump_split = handle.create_dataset("train_test_split_index", shape=(0,), maxshape=(None,), dtype="i4")
        dump_group = handle.create_dataset("generation_group", shape=(0,), maxshape=(None,), dtype="i4")
        handle.create_dataset("generation_group_size", data=np.array((config.generation_group_size,), dtype="i4"))
        dump_num_regimes = handle.create_dataset("num_regimes", shape=(0,), maxshape=(None,), dtype="i1")
        dump_num_classes = handle.create_dataset("num_classes", shape=(0,), maxshape=(None,), dtype="i1")
        dump_prior_type = handle.create_dataset("prior_type_index", shape=(0,), maxshape=(None,), dtype="i1")
        dump_sampled_is_causal = handle.create_dataset(
            "sampled_is_causal", shape=(0,), maxshape=(None,), dtype="i1"
        )
        dump_effective_is_causal = handle.create_dataset(
            "effective_is_causal", shape=(0,), maxshape=(None,), dtype="i1"
        )
        dump_scm_num_layers = handle.create_dataset("scm_num_layers", shape=(0,), maxshape=(None,), dtype="i1")
        dump_scm_hidden_dim = handle.create_dataset("scm_hidden_dim", shape=(0,), maxshape=(None,), dtype="i4")
        dump_scm_num_causes = handle.create_dataset("scm_num_causes", shape=(0,), maxshape=(None,), dtype="i4")
        dump_scm_metadata = handle.create_dataset(
            "scm_metadata_json", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        handle.create_dataset("families", data=list(FAMILIES), dtype=h5py.string_dtype())
        handle.create_dataset("mechanism_modes", data=list(MECHANISM_MODES), dtype=h5py.string_dtype())
        handle.create_dataset("prior_types", data=("mlp_scm", "tree_scm"), dtype=h5py.string_dtype())
        handle.create_dataset("problem_type", data="classification", dtype=h5py.string_dtype())
        handle.create_dataset("min_classes", data=config.min_classes, dtype="i1")
        handle.create_dataset("max_classes", data=config.max_classes, dtype="i1")
        handle.create_dataset(
            "prior_family", data=f"tabicl_mix_scm_{config.mechanism_mode}_{config.rule_mode}", dtype=h5py.string_dtype()
        )
        handle.create_dataset("rule_mode", data=config.rule_mode, dtype=h5py.string_dtype())
        handle.create_dataset("mechanism_mode", data=config.mechanism_mode, dtype=h5py.string_dtype())
        handle.create_dataset("hp_profile", data=config.hp_profile, dtype=h5py.string_dtype())
        handle.create_dataset("metadata_schema_version", data=4, dtype="i4")

        written = 0
        group_id = 0
        while written < target:
            group = sample_generation_group_v4(
                rng,
                min_features=config.min_features,
                max_features=config.max_features,
                min_instances=config.min_instances,
                max_instances=config.max_instances,
                min_train_fraction=config.min_train_fraction,
                max_train_fraction=config.max_train_fraction,
                min_regimes=config.min_regimes,
                max_regimes=config.max_regimes,
                min_samples_per_regime=config.min_samples_per_regime,
                mix_probs=config.mix_probs,
                hp_profile=config.hp_profile,
                fixed_support_size=config.support_size,
                fixed_query_size=config.query_size,
            )
            members = min(config.generation_group_size, target - written)
            for _ in range(members):
                family_index = int(rng.integers(len(FAMILIES)))
                expose_z = bool(rng.random() < config.expose_z_probability)
                episode = sample_episode_v4(
                    int(rng.integers(2**31)),
                    family=FAMILIES[family_index],
                    min_features=config.min_features,
                    max_features=config.max_features,
                    num_regimes=group.num_regimes,
                    calibration_size=config.calibration_size,
                    pad_features=pad_features,
                    max_classes=config.max_classes,
                    expose_z=expose_z,
                    rule_mode=config.rule_mode,
                    mechanism_mode=config.mechanism_mode,
                    group=group,
                )
                x = torch.cat((episode.support_x, episode.query_x), dim=1).numpy()[0]
                y = torch.cat((episode.support_y, episode.query_y), dim=1).numpy()[0]
                regime = np.concatenate((episode.support_z, episode.query_z))
                rows = group.rows

                padded_x = np.zeros((config.max_instances, pad_features), dtype=np.float32)
                padded_y = np.zeros(config.max_instances, dtype=np.float32)
                padded_regime = np.zeros(config.max_instances, dtype=np.int8)
                padded_x[:rows] = x
                padded_y[:rows] = y
                padded_regime[:rows] = regime
                for dataset, value in (
                    (dump_x, padded_x[None]),
                    (dump_y, padded_y[None]),
                    (dump_regime, padded_regime[None]),
                    (dump_family, np.array([family_index], dtype="i1")),
                    (dump_mechanism, np.array([MECHANISM_MODES.index(config.mechanism_mode)], dtype="i1")),
                    (dump_expose_z, np.array([expose_z], dtype="i1")),
                    (dump_z_column_index, np.array([episode.z_column_index if expose_z else -1], dtype="i4")),
                    (dump_features, np.array([episode.d], dtype="i4")),
                    (dump_rows, np.array([rows], dtype="i4")),
                    (dump_split, np.array([group.support_size], dtype="i4")),
                    (dump_group, np.array([group_id], dtype="i4")),
                    (dump_num_regimes, np.array([group.num_regimes], dtype="i1")),
                    (dump_num_classes, np.array([episode.num_classes], dtype="i1")),
                    (dump_prior_type, np.array([episode.prior_type == "tree_scm"], dtype="i1")),
                    (
                        dump_sampled_is_causal,
                        np.array([episode.scm_metadata["sampled_is_causal"]], dtype="i1"),
                    ),
                    (
                        dump_effective_is_causal,
                        np.array([episode.scm_metadata["effective_is_causal"]], dtype="i1"),
                    ),
                    (dump_scm_num_layers, np.array([episode.scm_metadata["effective_num_layers"]], dtype="i1")),
                    (dump_scm_hidden_dim, np.array([episode.scm_metadata["effective_hidden_dim"]], dtype="i4")),
                    (dump_scm_num_causes, np.array([episode.scm_metadata["effective_num_causes"]], dtype="i4")),
                ):
                    dataset.resize(dataset.shape[0] + 1, axis=0)
                    dataset[-1:] = value
                dump_scm_metadata.resize(dump_scm_metadata.shape[0] + 1, axis=0)
                dump_scm_metadata[-1:] = [json.dumps(episode.scm_metadata, sort_keys=True)]
                written += 1
            group_id += 1
            if written % 250 == 0 or written == target:
                print(f"{written}/{target} episodes", flush=True)
    print(f"wrote {target} episodes to {path}", flush=True)
    return path


class MultiregimeV4DumpLoader:
    """Stream grouped v4 episodes without passing padding or diagnostics to a model."""

    def __init__(self, path: str | Path, *, batch_size: int = 8, device: str | torch.device = "cpu", seed: int = 0,
                 family: str | None = None, min_classes: int | None = None):
        """``family`` / ``min_classes`` restrict which episodes of the dump are served (ablations).

        Episode geometry (rows, split, regime count) is fixed per generation group, so selecting a subset of a
        group's episodes still yields a rectangular, same-distribution batch; only the batch's episodes change.
        """
        del seed  # Kept for API compatibility; dumping fixes the episode order.
        self.paths = self._resolve(Path(path))
        self.batch_size = batch_size
        self.device = device
        if family is not None and family not in FAMILIES:
            raise ValueError(f"Unknown routing family {family!r}; choose from {FAMILIES}.")
        self.family = family
        self.min_classes = min_classes
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
            raise FileNotFoundError(f"No multiregime_v4 dump at {path}.")
        return [path]

    def _metadata(self, name: str) -> np.ndarray:
        assert self._handle is not None
        values = np.asarray(self._handle[name][:], dtype=np.int64)
        if len(values) == self._count:
            return values
        if len(values) == 1:  # Compatibility with the original fixed-size v4 dump layout.
            return np.repeat(values, self._count)
        raise ValueError(f"{name} must contain one value per episode or one shared value.")

    def _open(self, path: Path) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = h5py.File(path, "r")
        self._pointer = 0
        self._count = self._handle["X"].shape[0]
        self.families = [f.decode("utf-8") for f in self._handle["families"][:]]
        self._rows = self._metadata("num_datapoints")
        self._splits = self._metadata("train_test_split_index")
        self._num_regimes = self._metadata("num_regimes")
        self._num_classes = self._metadata("num_classes") if "num_classes" in self._handle else None
        self._mechanism_modes = (
            self._metadata("mechanism_mode_index") if "mechanism_mode_index" in self._handle else None
        )
        self.num_regimes = int(self._num_regimes.max())
        self._groups = self._metadata("generation_group") if "generation_group" in self._handle else None
        self._mask = self._episode_filter()
        self._buckets: list[np.ndarray] | None = None
        self._bucket_index = 0
        if self._mask is not None:
            if not self._mask.any():
                raise ValueError(f"No episodes in {path} match family={self.family} / min_classes={self.min_classes}.")
            # Generation groups hold only a handful of episodes, so a filter rarely leaves a full batch inside
            # one group. Batches are therefore assembled across groups that share the episode geometry
            # (rows, train/test split, regime count) — the only properties a rectangular batch requires.
            selected = np.flatnonzero(self._mask)
            keys = {}
            for episode in selected.tolist():
                keys.setdefault((int(self._rows[episode]), int(self._splits[episode]),
                                 int(self._num_regimes[episode])), []).append(episode)
            self._buckets = [np.asarray(v, dtype=np.int64) for v in keys.values() if len(v) >= self.batch_size]
            if not self._buckets:
                raise ValueError(
                    f"No (rows, split, regimes) bucket in {path} has {self.batch_size} episodes matching "
                    f"family={self.family} / min_classes={self.min_classes}."
                )
            self._bucket_offsets = [0] * len(self._buckets)
        if self._groups is not None and self.batch_size > 1:
            group_counts = np.unique(self._groups, return_counts=True)[1]
            if group_counts.size == 0 or group_counts.max() < self.batch_size:
                raise ValueError("generation groups are smaller than the requested batch size.")

    def _advance_file(self) -> None:
        self._file_index = (self._file_index + 1) % len(self.paths)
        self._open(self.paths[self._file_index])

    def _episode_filter(self) -> np.ndarray | None:
        """Boolean mask of the currently open file's episodes that pass the family / class filters."""
        if self.family is None and self.min_classes is None:
            return None
        assert self._handle is not None
        mask = np.ones(self._count, dtype=bool)
        if self.family is not None:
            wanted = self.families.index(self.family)
            mask &= np.asarray(self._handle["family_index"][:], dtype=np.int64) == wanted
        if self.min_classes is not None and self._num_classes is not None:
            mask &= self._num_classes >= self.min_classes
        return mask

    def _next_slice(self) -> slice | np.ndarray:
        """Return one full batch: a contiguous slice from one generation group, or — when a filter is active —
        episode indices drawn from one (rows, split, regimes) bucket of the filtered episodes."""
        if self._buckets is not None:
            for _ in range(len(self._buckets) + 1):
                bucket = self._buckets[self._bucket_index % len(self._buckets)]
                offset = self._bucket_offsets[self._bucket_index % len(self._buckets)]
                if offset + self.batch_size > len(bucket):
                    self._bucket_offsets[self._bucket_index % len(self._buckets)] = 0
                    self._bucket_index += 1
                    continue
                self._bucket_offsets[self._bucket_index % len(self._buckets)] = offset + self.batch_size
                self._bucket_index += 1
                return bucket[offset : offset + self.batch_size]
            self._advance_file()
            return self._next_slice()
        for _ in range(len(self.paths) + 1):
            if self._pointer >= self._count:
                self._advance_file()
                continue
            if self._groups is None:
                if self._pointer + self.batch_size <= self._count:
                    start = self._pointer
                    self._pointer += self.batch_size
                    return slice(start, start + self.batch_size)
                self._advance_file()
                continue

            start = self._pointer
            group_id = self._groups[start]
            stop = start + 1
            while stop < self._count and self._groups[stop] == group_id:
                stop += 1
            if stop - start < self.batch_size:
                # A group remainder cannot form a rectangular, same-distribution
                # batch. Skip it and start at the next complete group.
                self._pointer = stop
                continue
            self._pointer = start + self.batch_size
            return slice(start, start + self.batch_size)
        raise RuntimeError("No generation group can supply the requested batch size.")

    def sample(self) -> dict[str, torch.Tensor]:
        """Return one same-group batch, cropped to its real rows and features."""
        assert self._handle is not None
        index = self._next_slice()
        rows = self._rows[index]
        splits = self._splits[index]
        regime_counts = self._num_regimes[index]
        same_geometry = np.all(rows == rows[0]) and np.all(splits == splits[0])
        same_regime_count = np.all(regime_counts == regime_counts[0])
        if not same_geometry or not same_regime_count:
            raise RuntimeError("A v4 training batch must share table length, split, and regime count.")
        row_count, split = int(rows[0]), int(splits[0])
        if not 0 < split < row_count:
            raise RuntimeError("Invalid train_test_split_index in v4 dump.")

        handle = self._handle
        features_per_episode = np.asarray(handle["num_features"][index], dtype=np.int64)
        features = int(features_per_episode.max())
        x = torch.from_numpy(handle["X"][index, :row_count, :features])
        y = torch.from_numpy(handle["y"][index, :row_count])
        regime = torch.from_numpy(handle["regime"][index, :row_count]).long()
        family_index = torch.from_numpy(handle["family_index"][index]).long()
        expose_z = torch.from_numpy(handle["expose_z"][index]).bool()
        z_column_index = torch.from_numpy(handle["z_column_index"][index]).long()
        mechanism_mode_index = (
            torch.from_numpy(self._mechanism_modes[index]).long()
            if self._mechanism_modes is not None
            else torch.full((len(x),), -1, dtype=torch.long)
        )
        num_classes = (
            torch.from_numpy(self._num_classes[index]).long()
            if self._num_classes is not None
            else torch.full((len(x),), 2, dtype=torch.long)
        )
        group = (
            torch.from_numpy(np.asarray(handle["generation_group"][index], dtype=np.int64))
            if "generation_group" in handle
            else torch.full((len(x),), -1, dtype=torch.long)
        )

        return {
            "support_x": x[:, :split].to(self.device),
            "support_y": y[:, :split].to(self.device),
            "query_x": x[:, split:].to(self.device),
            "query_y": y[:, split:].long().to(self.device),
            "support_regime": regime[:, :split].to(self.device),
            "query_regime": regime[:, split:].to(self.device),
            "family_index": family_index.to(self.device),
            "mechanism_mode_index": mechanism_mode_index.to(self.device),
            "expose_z": expose_z.to(self.device),
            "z_column_index": z_column_index.to(self.device),
            "generation_group": group.to(self.device),
            "num_datapoints": torch.full((len(x),), row_count, dtype=torch.long, device=self.device),
            "train_test_split_index": torch.full((len(x),), split, dtype=torch.long, device=self.device),
            "num_regimes": torch.full(
                (len(x),), int(regime_counts[0]), dtype=torch.long, device=self.device
            ),
            "num_classes": num_classes.to(self.device),
        }

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
        "generation_group_size",
        "min_features",
        "max_features",
        "min_classes",
        "max_classes",
        "min_instances",
        "max_instances",
        "support_size",
        "query_size",
        "min_regimes",
        "max_regimes",
        "min_samples_per_regime",
        "calibration_size",
        "seed",
        "shard_index",
        "num_shards",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("min_train_fraction", "max_train_fraction", "mlp_probability", "expose_z_probability"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--pad-features", type=int, default=defaults.pad_features)
    parser.add_argument("--rule-mode", choices=("shared", "multiregime"), default=defaults.rule_mode)
    parser.add_argument("--mechanism-mode", choices=MECHANISM_MODES, default=defaults.mechanism_mode)
    parser.add_argument("--hp-profile", choices=HP_PROFILES, default=defaults.hp_profile)
    return parser


def main(argv: list[str] | None = None) -> Path:
    return dump_multiregime_v4_episodes(DumpConfig(**vars(build_parser().parse_args(argv))))


if __name__ == "__main__":
    print(main())
