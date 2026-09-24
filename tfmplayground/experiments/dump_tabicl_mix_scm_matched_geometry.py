"""Dump exact TabICL ``mix_scm`` tasks on a v4 dump's sampled geometry.

This is the ordinary control for a multiregime-v4 comparison.  Rather than
recreating TabICL's SCM hyperprior, it invokes TabICL's own ``mix_scm`` prior
for every contiguous v4 generation group.  Each ordinary batch therefore has
the same number of tables, rows, support/query split, and requested feature
width as its matching v4 group, while retaining TabICL's native SCM and
``Reg2Cls`` label generation.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.external_priors.base import dump_prior_to_h5


@dataclass(frozen=True)
class MatchedMixSCMDumpConfig:
    """Configuration for an exact TabICL control matched to a v4 dump."""

    output: str
    geometry_source: str
    batch_size: int = 8
    max_classes: int = 2
    prior_type: str = "mix_scm"
    seed: int = 2402


@dataclass(frozen=True)
class GeometryGroup:
    """One contiguous, homogeneous v4 generation group."""

    group_id: int
    members: int
    rows: int
    support_size: int
    requested_num_features: int
    num_regimes: int


def _metadata(handle: h5py.File, name: str, count: int) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"Geometry source is missing required dataset {name!r}.")
    values = np.asarray(handle[name][:], dtype=np.int64)
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.repeat(values, count)
    raise ValueError(f"Geometry source {name!r} must have one value per episode or one shared value.")


def read_geometry_groups(path: str | Path, *, batch_size: int) -> tuple[list[GeometryGroup], int, int]:
    """Read and validate v4's group boundaries and per-group geometry."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"No v4 geometry source at {source}.")

    with h5py.File(source, "r") as handle:
        if "X" not in handle:
            raise ValueError("Geometry source is missing X.")
        count, max_rows, max_features = handle["X"].shape
        if count < 1:
            raise ValueError("Geometry source has no episodes.")
        groups = _metadata(handle, "generation_group", count)
        rows = _metadata(handle, "num_datapoints", count)
        splits = _metadata(handle, "train_test_split_index", count)
        features = _metadata(handle, "num_features", count)
        regimes = _metadata(handle, "num_regimes", count)

    result: list[GeometryGroup] = []
    start = 0
    seen: set[int] = set()
    while start < count:
        group_id = int(groups[start])
        stop = start + 1
        while stop < count and groups[stop] == group_id:
            stop += 1
        if group_id in seen:
            raise ValueError(f"generation_group {group_id} is not contiguous.")
        seen.add(group_id)
        if stop - start != batch_size:
            raise ValueError(
                f"generation_group {group_id} has {stop - start} episodes; expected exactly {batch_size}."
            )
        members = slice(start, stop)
        values = (rows[members], splits[members], features[members], regimes[members])
        if any(np.any(value != value[0]) for value in values):
            raise ValueError(f"generation_group {group_id} does not have homogeneous geometry.")
        row_count, split, width, num_regimes = (int(value[0]) for value in values)
        if not 0 < split < row_count <= max_rows:
            raise ValueError(f"generation_group {group_id} has invalid rows/split: {row_count}/{split}.")
        if not 1 <= width <= max_features:
            raise ValueError(f"generation_group {group_id} has invalid requested width: {width}.")
        if num_regimes < 1:
            raise ValueError(f"generation_group {group_id} has invalid regime count: {num_regimes}.")
        result.append(GeometryGroup(group_id, stop - start, row_count, split, width, num_regimes))
        start = stop
    return result, max_rows, max_features


class _MatchedMixSCMIterable:
    """Generate one native TabICL batch for each fixed v4 geometry group."""

    def __init__(self, config: MatchedMixSCMDumpConfig, groups: list[GeometryGroup]):
        self.config = config
        self.groups = groups

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        for geometry in self.groups:
            loader = TabICLPriorDataLoader(
                num_steps=1,
                batch_size=geometry.members,
                # TabICL uses an exclusive max bound, so these fix table rows.
                num_datapoints_min=geometry.rows,
                num_datapoints_max=geometry.rows + 1,
                min_features=geometry.requested_num_features,
                max_features=geometry.requested_num_features,
                max_num_classes=self.config.max_classes,
                device=torch.device("cpu"),
                prior_type=self.config.prior_type,
                # Likewise, fix the support/query split exactly.
                min_train_size=geometry.support_size,
                max_train_size=geometry.support_size + 1,
            )
            # TabICL mix_scm's nested sampler closures are not pickleable.
            loader.pd.prior.n_jobs = 1
            batch = next(iter(loader))
            x = batch["x"]
            if x.shape[:2] != (geometry.members, geometry.rows):
                raise RuntimeError(
                    f"TabICL did not honour group {geometry.group_id}'s table geometry: {tuple(x.shape[:2])}."
                )
            if int(batch["train_test_split_index"]) != geometry.support_size:
                raise RuntimeError(f"TabICL did not honour group {geometry.group_id}'s support split.")
            yield batch

    def __len__(self) -> int:
        return len(self.groups)


def _validate(config: MatchedMixSCMDumpConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if config.max_classes < 2:
        raise ValueError("max_classes must be at least 2 for classification.")
    if config.prior_type != "mix_scm":
        raise ValueError("The matched ordinary control must use TabICL's exact mix_scm prior.")
    output = Path(config.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite matched mix_scm dump: {output}.")
    if output.resolve() == Path(config.geometry_source).resolve():
        raise ValueError("output and geometry_source must be different files.")


def dump_tabicl_mix_scm_matched_geometry(config: MatchedMixSCMDumpConfig) -> Path:
    """Write native TabICL ``mix_scm`` episodes matched to a v4 HDF5 dump."""
    _validate(config)
    groups, max_rows, max_features = read_geometry_groups(config.geometry_source, batch_size=config.batch_size)
    path = Path(config.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dump_prior_to_h5(
        _MatchedMixSCMIterable(config, groups),
        config.max_classes,
        config.batch_size,
        str(path),
        "classification",
        max_rows,
        max_features,
    )

    group_values = np.repeat(np.array([group.group_id for group in groups], dtype="i4"), config.batch_size)
    requested_widths = np.repeat(
        np.array([group.requested_num_features for group in groups], dtype="i4"), config.batch_size
    )
    regime_values = np.repeat(np.array([group.num_regimes for group in groups], dtype="i1"), config.batch_size)
    with h5py.File(path, "a") as handle:
        if handle["X"].shape[0] != len(group_values):
            raise RuntimeError("Matched dump wrote an unexpected number of episodes.")
        handle.create_dataset("generation_group", data=group_values)
        handle.create_dataset("geometry_requested_num_features", data=requested_widths)
        handle.create_dataset("geometry_num_regimes", data=regime_values)
        handle.create_dataset("prior_family", data="tabicl_mix_scm_matched_geometry", dtype=h5py.string_dtype())
        handle.create_dataset("geometry_source", data=str(Path(config.geometry_source)), dtype=h5py.string_dtype())
        handle.create_dataset("seed", data=np.array((config.seed,), dtype="i8"))
    return path


def build_parser() -> argparse.ArgumentParser:
    defaults = MatchedMixSCMDumpConfig(output="", geometry_source="")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-source", required=True)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-classes", type=int, default=defaults.max_classes)
    parser.add_argument("--prior-type", default=defaults.prior_type)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main(argv: list[str] | None = None) -> Path:
    return dump_tabicl_mix_scm_matched_geometry(MatchedMixSCMDumpConfig(**vars(build_parser().parse_args(argv))))


if __name__ == "__main__":
    print(main())
