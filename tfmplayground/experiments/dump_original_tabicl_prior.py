"""Pre-generate the ordinary TabICL classification prior for fast pretraining.

The HDF5 layout is the repository's standard ``PriorDumpDataLoader`` format,
so loading a dumped batch is interchangeable with a dynamically sampled
``TabICLPriorDataLoader`` batch. Validation intentionally remains dynamic and
therefore never consumes this training bank.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.external_priors.base import dump_prior_to_h5


@dataclass(frozen=True)
class OriginalPriorDumpConfig:
    output: str
    episodes: int = 20_000
    batch_size: int = 8
    #: Keep one complete TabICL group in each stored training batch.
    batch_size_per_gp: int = 8
    #: Legacy exact support/query geometry. Used unless the native TabICL
    #: row/split controls below are supplied.
    support_size: int = 128
    query_size: int = 32
    #: Inclusive total-row bounds and fractional/absolute support bounds.
    #: ``1024, 1024, 0.1, 0.9`` reproduces TabICL's default table/split setup.
    min_rows: int | None = None
    max_rows: int | None = None
    min_train_size: int | float | None = None
    max_train_size: int | float | None = None
    min_features: int = 2
    max_features: int = 12
    max_classes: int = 2
    prior_type: str = "mix_scm"
    seed: int = 2402

    @property
    def rows(self) -> int:
        if self.max_rows is not None:
            return self.max_rows
        return self.support_size + self.query_size

    @property
    def uses_native_row_split_sampling(self) -> bool:
        return self.min_rows is not None

    @property
    def batches(self) -> int:
        return self.episodes // self.batch_size


def _validate(config: OriginalPriorDumpConfig) -> None:
    if config.episodes < 1 or config.batch_size < 1 or config.batch_size_per_gp < 1:
        raise ValueError("episodes, batch_size, and batch_size_per_gp must be positive.")
    if config.batch_size_per_gp != config.batch_size:
        raise ValueError("Store one complete TabICL group per dump batch: batch_size_per_gp == batch_size.")
    if config.episodes % config.batch_size:
        raise ValueError("episodes must be divisible by batch_size so every stored batch is complete.")
    if config.support_size < 1 or config.query_size < 1:
        raise ValueError("support_size and query_size must be positive.")
    if config.uses_native_row_split_sampling:
        if config.max_rows is None or config.min_train_size is None or config.max_train_size is None:
            raise ValueError("min_rows, max_rows, min_train_size, and max_train_size must be supplied together.")
        if not 2 <= config.min_rows <= config.max_rows:
            raise ValueError("Native row bounds must satisfy 2 <= min_rows <= max_rows.")
        if type(config.min_train_size) is not type(config.max_train_size):
            raise ValueError("Native train-size bounds must have the same type.")
        if isinstance(config.min_train_size, float):
            if not 0 < config.min_train_size < config.max_train_size < 1:
                raise ValueError("Fractional train-size bounds must satisfy 0 < min < max < 1.")
        elif not 0 < config.min_train_size < config.max_train_size <= config.min_rows:
            raise ValueError("Integer train-size bounds must leave at least one query row.")
    elif config.max_rows is not None or config.min_train_size is not None or config.max_train_size is not None:
        raise ValueError("Native row/split controls must be supplied together, including min_rows.")
    if not 1 <= config.min_features <= config.max_features:
        raise ValueError("Feature bounds must satisfy 1 <= min_features <= max_features.")
    if config.max_classes < 2:
        raise ValueError("max_classes must be at least 2 for classification.")
    if config.prior_type not in {"mlp_scm", "tree_scm", "mix_scm"}:
        raise ValueError("prior_type must be mlp_scm, tree_scm, or mix_scm.")
    if Path(config.output).exists():
        raise FileExistsError(f"Refusing to overwrite original-prior dump: {config.output}.")


def dump_original_tabicl_prior(config: OriginalPriorDumpConfig) -> Path:
    """Write a finite ordinary-prior training bank with the requested geometry."""
    _validate(config)
    path = Path(config.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.uses_native_row_split_sampling:
        assert config.min_rows is not None and config.max_rows is not None
        assert config.min_train_size is not None and config.max_train_size is not None
        min_rows, max_rows = config.min_rows, config.max_rows
        min_train_size, max_train_size = config.min_train_size, config.max_train_size
    else:
        min_rows = max_rows = config.rows
        min_train_size, max_train_size = config.support_size, config.support_size + 1
    loader = TabICLPriorDataLoader(
        num_steps=config.batches,
        batch_size=config.batch_size,
        num_datapoints_min=min_rows,
        num_datapoints_max=max_rows + 1,
        min_features=config.min_features,
        max_features=config.max_features,
        max_num_classes=config.max_classes,
        device=torch.device("cpu"),
        prior_type=config.prior_type,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        batch_size_per_gp=config.batch_size_per_gp,
    )
    # TabICL's mixed sampler holds nested closures which cannot be pickled by
    # the dataset's multiprocessing path. Dumping is a CPU batch job, so one
    # worker is deterministic and avoids that implementation limitation.
    loader.pd.prior.n_jobs = 1
    dump_prior_to_h5(
        loader,
        config.max_classes,
        config.batch_size,
        str(path),
        "classification",
        config.rows,
        config.max_features,
    )
    # Keep provenance in the ordinary HDF5 schema without changing its loader
    # contract.
    with h5py.File(path, "a") as handle:
        handle.create_dataset("prior_family", data="original", dtype=h5py.string_dtype())
        handle.create_dataset("seed", data=np.array((config.seed,), dtype="i8"))
    return path


def build_parser() -> argparse.ArgumentParser:
    defaults = OriginalPriorDumpConfig(output="")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for name in (
        "episodes",
        "batch_size",
        "batch_size_per_gp",
        "support_size",
        "query_size",
        "min_rows",
        "max_rows",
        "min_features",
        "max_features",
        "max_classes",
        "seed",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--min-train-size", type=float, default=defaults.min_train_size)
    parser.add_argument("--max-train-size", type=float, default=defaults.max_train_size)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm"), default=defaults.prior_type)
    return parser


def main(argv: list[str] | None = None) -> Path:
    return dump_original_tabicl_prior(OriginalPriorDumpConfig(**vars(build_parser().parse_args(argv))))


if __name__ == "__main__":
    print(main())
