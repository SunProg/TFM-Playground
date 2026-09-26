"""Matched plain and multiregime, from-scratch nanoTabPFN pretraining."""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.continuous_episodes import (
    SCM_FAMILIES,
    TRAIN_REGIME,
    sample_scm_multiregime_episode,
)
from tfmplayground.experiments.dump_multiregime_v4_episodes import MultiregimeV4DumpLoader
from tfmplayground.experiments.multiregime_v4_evaluation import evaluate_multiregime_v4_bank
from tfmplayground.external_priors import PriorDumpDataLoader, TabICLPriorDataLoader
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed

_MAX_NON_FINITE_BATCH_RETRIES = 32  # matches continuous_episodes.py's SCM candidate retry budget


@dataclass(frozen=True)
class PlainPretrainingConfig:
    """Configuration for one reproducible ordinary-prior training seed."""

    seed: int = 2402
    device: str = "cuda"
    require_cuda: bool = False
    max_steps: int = 10_000
    micro_batch_size: int = 8
    #: Native TabICL group size. Training uses one complete group per model
    #: batch so every batch has one sampled table length and support split.
    batch_size_per_gp: int | None = None
    accumulate_gradients: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 2_000
    #: Warmup as a proportion of max_steps (e.g. 0.2 = 20 %). -1 (default) keeps
    #: the fixed ``warmup_steps`` above; any value in (0, 1) overrides it.
    warmup_proportion: float = -1.0
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    validation_interval: int = 5_000
    validation_batches: int = 16
    #: Fixed factorial v4 validation bank, evaluated independently of the
    #: ordinary-prior validation stream above. Each report includes every cell
    #: and every requested aggregate slice.
    v4_validation_bank_path: str | None = None
    v4_validation_interval: int = 1_000
    #: Optional second validation bank scored on the same cadence (e.g. the z-blind
    #: bank for a z-exposed run); its numbers are recorded under
    #: ``v4_validation_<tag>_*`` in history.jsonl and ``v4_validation_<tag>/`` reports.
    v4_extra_validation_bank_path: str | None = None
    v4_extra_validation_tag: str = "blind"
    #: Cap v4-bank inference batches independently of the number of episodes
    #: per factorial cell, so large support/feature stress cells fit on GPU.
    v4_evaluation_batch_size: int = 1
    #: An optional fixed held-out bank, scored exactly once after training.
    v4_test_bank_path: str | None = None
    #: Lightweight held-out regime-information probe during training. This uses
    #: one episode from a deterministic subset of factorial cells; the complete
    #: bank remains reserved for the final evaluation.
    v4_regime_info_bank_path: str | None = None
    v4_regime_info_interval: int = 1_000
    v4_regime_info_cell_subsample_modulus: int = 16
    checkpoint_interval: int = 10_000
    #: Additionally overwrite ``latest_checkpoint.pth`` every this many steps
    #: (0 = off) so an interrupted run can be resumed with little lost work.
    latest_checkpoint_interval: int = 0
    #: Legacy exact geometry. Used unless the native TabICL row/split controls
    #: below are supplied.
    support_size: int = 128
    query_size: int = 32
    #: Optional inclusive total-row range. Set both bounds together. A pair of
    #: 1024 values reproduces TabICL's default fixed table length.
    min_rows: int | None = None
    max_rows: int | None = None
    #: Optional TabICL support-split bounds. Floats are fractions of the
    #: realised row count, e.g. 0.1 and 0.9. Set both with min_rows/max_rows.
    min_train_size: int | float | None = None
    max_train_size: int | float | None = None
    min_features: int = 2
    max_features: int = 12
    max_classes: int = 2
    prior_type: str = "mix_scm"
    #: ``original`` is ordinary TabICL only; ``fixed`` mixes a constant,
    #: configurable multiregime share; ``curriculum`` ramps to that share.
    #: ``plain`` and ``multiregime`` are retained as backward-compatible
    #: aliases for 0% and 100% multiregime respectively.
    prior_mode: Literal["original", "fixed", "curriculum", "plain", "multiregime"] = "plain"
    #: The fixed share and curriculum endpoint. It has no effect for original/
    #: plain or the legacy all-multiregime mode.
    multiregime_ratio: float = 0.5
    #: Which generator backs the multiregime share of prior_mode="fixed"/
    #: "curriculum" (and legacy "multiregime") -- "legacy" is
    #: continuous_episodes.py's naive hand-built
    #: generator (unchanged default, byte-identical to every existing run);
    #: "v4" sources episodes from multiregime_v4.py's real mlp_scm/tree_scm-
    #: grounded prior via a pre-generated MultiregimeV4DumpLoader dump
    #: (required: v4_dump_path). This only changes *where* multiregime
    #: episodes come from, not *how much* -- the selected prior mode controls
    #: the mixture share.
    multiregime_source: Literal["legacy", "v4"] = "legacy"
    #: One dump (file or shard directory) or several separated by commas, e.g.
    #: "r_z-multiregime.h5,g_z-multiregime.h5": several dumps are drawn from in
    #: round-robin order per micro-batch, so a mixed r_z+g_z prior is an even mix.
    v4_dump_path: str | None = None
    #: Ablations on the MULTIREGIME branch only: serve just one routing family
    #: (``soft_gate``/``persistent``) and/or only episodes with at least this
    #: many classes. The ordinary (K=1) branch is untouched.
    v4_dump_family: str | None = None
    v4_dump_min_classes: int | None = None
    #: Optional paired shared-rule V4 dump.  When supplied, ordinary training
    #: draws come from this file while multiregime draws come from
    #: ``v4_dump_path``.  This isolates r_z/g_z label selection: the two
    #: files have the same generated base tasks and differ only in Y_0 vs Y_Z.
    v4_shared_dump_path: str | None = None
    #: Append the realized v4 regime identity as an observed context feature.
    #: The IDs are independently permuted within every episode before entering
    #: the model, so the model can use support/query membership correspondence
    #: but cannot attach a global meaning to an integer label.  This is a
    #: distinct intervention from ``expose_z``, which exposes a feature-derived
    #: routing score in the dump itself.
    v4_expose_regime_id: bool = False
    #: Per-episode Bernoulli(p) alternative to ``v4_expose_regime_id``: each
    #: multiregime episode independently gets the ID column with this
    #: probability instead of every episode in a run getting it uniformly.
    #: Mutually exclusive with ``v4_expose_regime_id`` and
    #: ``v4_expose_regime_id_paired``. Unlike ``v4_expose_regime_id_paired``,
    #: the exposed and hidden copies of a given draw are *different*
    #: episodes, so any measured exposed-vs-hidden gap is confounded with
    #: which episodes happened to land in which condition.
    v4_expose_regime_id_probability: float | None = None
    #: Train on BOTH hidden and exposed copies of each multiregime episode.
    #: They are concatenated on the batch axis for one model call; the hidden
    #: copy gets a -1 sentinel in the appended regime-ID feature.
    #: Mutually exclusive with ``v4_expose_regime_id`` and
    #: ``v4_expose_regime_id_probability``.
    v4_expose_regime_id_paired: bool = False
    #: The ordinary branch may stay dynamic or stream a standard TabICL HDF5
    #: training dump. Validation always uses freshly generated ordinary tasks.
    original_source: Literal["dynamic", "dump"] = "dynamic"
    original_dump_path: str | None = None
    multiregime_contamination: float = 0.3
    #: How feature-coherent the contaminated group is.  0.0 is the original
    #: design, where the relabelled rows are a uniform random subset and the
    #: regime tag is independent of the features; above zero they concentrate on
    #: one side of a per-episode hyperplane.  See
    #: ``continuous_episodes._contaminated_positions``.
    regime_coherence: float = 0.0
    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 6
    epoch_steps: int = 500
    tabarena_every_epoch: bool = False
    tabarena_folds: int = 5
    tabarena_repeats: int = 10
    tabarena_subsample: int = 2_048
    tabarena_cache_directory: str | None = None
    tensorboard: bool = True

    @property
    def rows(self) -> int:
        if self.max_rows is not None:
            return self.max_rows
        return self.support_size + self.query_size

    @property
    def uses_native_row_split_sampling(self) -> bool:
        return self.min_rows is not None

    def architecture(self) -> dict[str, int]:
        return {
            "embedding_size": self.embedding_size,
            "num_attention_heads": self.num_attention_heads,
            "mlp_hidden_size": self.mlp_hidden_size,
            "num_layers": self.num_layers,
            "num_outputs": self.max_classes,
        }


def make_prior(config: PlainPretrainingConfig, *, batches: int, device: str | torch.device | None = None):
    """Build the ordinary TabICL prior with fixed or native-sampled geometry."""
    if config.uses_native_row_split_sampling:
        assert config.min_rows is not None and config.max_rows is not None
        assert config.min_train_size is not None and config.max_train_size is not None
        min_rows, max_rows = config.min_rows, config.max_rows
        min_train_size, max_train_size = config.min_train_size, config.max_train_size
    else:
        min_rows = max_rows = config.rows
        min_train_size, max_train_size = config.support_size, config.support_size + 1
    loader = TabICLPriorDataLoader(
        num_steps=batches,
        batch_size=config.micro_batch_size,
        # TabICL samples ``randint(min, max)``; public row bounds are inclusive.
        num_datapoints_min=min_rows,
        num_datapoints_max=max_rows + 1,
        min_features=config.min_features,
        max_features=config.max_features,
        max_num_classes=config.max_classes,
        device=torch.device(config.device if device is None else device),
        prior_type=config.prior_type,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        batch_size_per_gp=config.micro_batch_size if config.batch_size_per_gp is None else config.batch_size_per_gp,
    )
    # mix_scm's HpSampler builds a nested-closure sampler
    # (setup_meta_choice_mixed_sampler) that Python's default multiprocessing
    # pickling can't serialize, so TabICLPriorDataLoader's default n_jobs>1
    # crashes on the first batch. multiregime_v3.py's OriginalPrior already
    # works around this the same way; matched here for parity, not a change
    # in what gets generated -- n_jobs is a parallelism knob, not a content
    # knob, so this doesn't touch the actual prior data.
    loader.pd.prior.n_jobs = 1
    return loader


def query_loss(model: NanoTabPFNModel, batch) -> torch.Tensor:
    """Cross entropy on query labels only; support labels are model inputs."""
    if not isinstance(batch, dict):
        logits = model(batch.support_x, batch.support_y, batch.query_x)
        target = batch.query_y
    else:
        split = int(batch["train_test_split_index"])
        x, y = batch["x"], batch["y"]
        logits = model(x[:, :split], y[:, :split], x[:, split:])
        target = y[:, split:]
    # The model's own output width, not a hardcoded 2: with three classes a
    # `[..., :2]` slice would drop a class and renormalize over the rest, which
    # trains fine and reports nonsense.
    classes = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, classes), target.reshape(-1).long())


def multiregime_batch(
    config: PlainPretrainingConfig,
    v4_loader: "MultiregimeV4DumpLoader | RoundRobinV4DumpLoader | None",
    rng: np.random.Generator,
) -> list:
    """Draw one multiregime episode, from whichever generator config.multiregime_source selects."""
    if config.multiregime_source == "v4":
        if v4_loader is None:
            raise RuntimeError("multiregime_source='v4' requires a constructed MultiregimeV4DumpLoader.")
        return _v4_dump_batch(
            v4_loader,
            expose_regime_id=config.v4_expose_regime_id,
            expose_regime_id_probability=config.v4_expose_regime_id_probability,
            expose_regime_id_paired=config.v4_expose_regime_id_paired,
            rng=rng,
        )

    sources: tuple[str | tuple[str, str], ...] = (
        *SCM_FAMILIES,
        ("mlp_scm", "tree_scm"),
    )
    source = sources[int(rng.integers(len(sources)))]
    return [sample_scm_multiregime_episode(
        rng,
        regime=TRAIN_REGIME,
        family=source,
        batch_size=config.micro_batch_size,
        support_size=config.support_size,
        query_count=config.query_size,
        noise=0.0,
        contamination=config.multiregime_contamination,
        regime_coherence=config.regime_coherence,
        num_classes=config.max_classes,
        device=config.device,
    )]


class RoundRobinV4DumpLoader:
    """Alternate ``sample()`` calls between several MultiregimeV4DumpLoaders (r_z + g_z mixtures)."""

    def __init__(self, loaders: Sequence[MultiregimeV4DumpLoader]):
        if not loaders:
            raise ValueError("RoundRobinV4DumpLoader needs at least one loader.")
        self.loaders = list(loaders)
        self._turn = 0

    def sample(self) -> dict[str, torch.Tensor]:
        loader = self.loaders[self._turn % len(self.loaders)]
        self._turn += 1
        return loader.sample()

    def close(self) -> None:
        for loader in self.loaders:
            loader.close()


def _dump_paths(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def make_v4_dump_loader(spec: str, *, batch_size: int, device: str, family: str | None = None,
                        min_classes: int | None = None) -> MultiregimeV4DumpLoader | RoundRobinV4DumpLoader:
    """Build the loader for a comma-separated dump spec (a single path keeps the plain loader).

    ``family`` / ``min_classes`` are ablation filters applied to every dump in the spec (see
    ``MultiregimeV4DumpLoader``); they are meant for the multiregime branch only."""
    paths = _dump_paths(spec)
    kwargs = {"batch_size": batch_size, "device": device, "family": family, "min_classes": min_classes}
    if len(paths) == 1:
        return MultiregimeV4DumpLoader(paths[0], **kwargs)
    return RoundRobinV4DumpLoader([MultiregimeV4DumpLoader(path, **kwargs) for path in paths])


def _regime_tag_columns(
    support_x: torch.Tensor,
    batch: dict,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Per-episode-permuted realized regime IDs, one column per support+query row.

    A fresh per-episode permutation makes the values categorical
    correspondence markers, rather than globally meaningful ordered feature
    values.
    """
    support_regime, query_regime = batch["support_regime"], batch["query_regime"]
    num_regimes = batch["num_regimes"]
    if support_regime.shape[0] != query_regime.shape[0] or support_regime.shape[0] != num_regimes.shape[0]:
        raise ValueError("V4 regime metadata must have one row per episode.")
    tags = torch.empty(
        (support_regime.shape[0], support_regime.shape[1] + query_regime.shape[1]),
        dtype=support_x.dtype,
        device=support_x.device,
    )
    for episode_index, regime_count in enumerate(num_regimes.detach().cpu().tolist()):
        regime_count = int(regime_count)
        if regime_count < 1:
            raise ValueError("A v4 episode must contain at least one regime.")
        permutation = torch.as_tensor(
            rng.permutation(regime_count), dtype=support_regime.dtype, device=support_regime.device
        )
        regime = torch.cat((support_regime[episode_index], query_regime[episode_index]))
        if bool(((regime < 0) | (regime >= regime_count)).any()):
            raise ValueError("V4 regime IDs must lie in [0, num_regimes).")
        tags[episode_index] = permutation[regime].to(dtype=support_x.dtype)
    return tags


def _expose_regime_tag(support_x: torch.Tensor, query_x: torch.Tensor, batch: dict, rng: np.random.Generator):
    """Append the (permuted) realized regime ID as one input column to every episode."""
    tags = _regime_tag_columns(support_x, batch, rng)
    support_width = support_x.shape[1]
    support_x = torch.cat((support_x, tags[:, :support_width, None]), dim=-1)
    query_x = torch.cat((query_x, tags[:, support_width:, None]), dim=-1)
    return support_x, query_x


def _index_v4_batch(batch: dict, index: np.ndarray) -> dict:
    """Select a subset of episodes from a raw ``loader.sample()`` batch dict."""
    return {key: value[index] for key, value in batch.items()}


def _batch_namespace(support_x: torch.Tensor, query_x: torch.Tensor, batch: dict) -> SimpleNamespace:
    return SimpleNamespace(
        support_x=support_x,
        support_y=batch["support_y"],
        query_x=query_x,
        query_y=batch["query_y"],
    )


def _v4_dump_batch(
    loader: MultiregimeV4DumpLoader | RoundRobinV4DumpLoader,
    *,
    expose_regime_id: bool = False,
    expose_regime_id_probability: float | None = None,
    expose_regime_id_paired: bool = False,
    rng: np.random.Generator | None = None,
) -> list[SimpleNamespace]:
    """Convert a diagnostic-rich V4 dump batch into one or two model batches.

    Exactly one of three (mutually exclusive) regime-ID interventions may be
    active:

    * ``expose_regime_id``: every episode in the draw gets the tag column.
    * ``expose_regime_id_probability``: each episode independently gets the
      tag column with this probability; exposed and hidden episodes in the
      same draw are different underlying episodes.
    * ``expose_regime_id_paired``: every episode in the draw is trained BOTH
      with and without the tag (two copies of the same episode), returned as
      one concatenated batch. The hidden copy gets -1 in the appended tag
      column; valid regime IDs are nonnegative.

    Diagnostics such as mechanism mode remain excluded from model inputs
    regardless of mode. With none of the three set, every episode is hidden.
    """
    modes = [expose_regime_id, expose_regime_id_probability is not None, expose_regime_id_paired]
    if sum(modes) > 1:
        raise ValueError(
            "Choose at most one of expose_regime_id / expose_regime_id_probability / expose_regime_id_paired."
        )
    batch = loader.sample()
    support_x, query_x = batch["support_x"], batch["query_x"]

    if expose_regime_id_paired:
        if rng is None:
            raise ValueError("A NumPy generator is required when exposing v4 regime IDs.")
        exposed_support_x, exposed_query_x = _expose_regime_tag(support_x, query_x, batch, rng)
        # Pad the hidden copy with an out-of-range missing-ID sentinel. Equal
        # feature widths let both paired conditions share one model call.
        hidden_support_x = torch.cat(
            (support_x, support_x.new_full((*support_x.shape[:-1], 1), -1.0)), dim=-1
        )
        hidden_query_x = torch.cat(
            (query_x, query_x.new_full((*query_x.shape[:-1], 1), -1.0)), dim=-1
        )
        combined = SimpleNamespace(
            support_x=torch.cat((hidden_support_x, exposed_support_x), dim=0),
            support_y=torch.cat((batch["support_y"], batch["support_y"]), dim=0),
            query_x=torch.cat((hidden_query_x, exposed_query_x), dim=0),
            query_y=torch.cat((batch["query_y"], batch["query_y"]), dim=0),
        )
        return [combined]

    if expose_regime_id_probability is not None:
        if rng is None:
            raise ValueError("A NumPy generator is required when exposing v4 regime IDs.")
        if not 0.0 <= expose_regime_id_probability <= 1.0:
            raise ValueError("expose_regime_id_probability must lie in [0, 1].")
        mask = rng.random(support_x.shape[0]) < expose_regime_id_probability
        if not mask.any():
            return [_batch_namespace(support_x, query_x, batch)]
        if mask.all():
            exposed_support_x, exposed_query_x = _expose_regime_tag(support_x, query_x, batch, rng)
            return [_batch_namespace(exposed_support_x, exposed_query_x, batch)]
        exposed_index = np.flatnonzero(mask)
        hidden_index = np.flatnonzero(~mask)
        exposed_batch = _index_v4_batch(batch, exposed_index)
        exposed_support_x, exposed_query_x = _expose_regime_tag(
            exposed_batch["support_x"], exposed_batch["query_x"], exposed_batch, rng
        )
        hidden_batch = _index_v4_batch(batch, hidden_index)
        return [
            _batch_namespace(hidden_batch["support_x"], hidden_batch["query_x"], hidden_batch),
            _batch_namespace(exposed_support_x, exposed_query_x, exposed_batch),
        ]

    if expose_regime_id:
        if rng is None:
            raise ValueError("A NumPy generator is required when exposing v4 regime IDs.")
        support_x, query_x = _expose_regime_tag(support_x, query_x, batch, rng)

    return [_batch_namespace(support_x, query_x, batch)]


def multiregime_probability(config: PlainPretrainingConfig, step: int) -> float:
    """Return the multiregime share for this optimizer step.

    ``original`` (and the legacy name ``plain``) always returns 0. ``fixed``
    always returns ``multiregime_ratio``. ``curriculum`` presents only
    ordinary TabICL ``mix_scm`` tables for the first 10% of the update budget,
    then linearly ramps the *episode* mixture to ``multiregime_ratio`` between
    10% and 50%, retaining that share for the rest of training. It is
    intentionally independent of the learning-rate schedule so that the
    curriculum remains explicit in run metadata.
    """
    ratio = float(getattr(config, "multiregime_ratio", 0.5))
    if config.prior_mode in {"original", "plain"}:
        return 0.0
    if config.prior_mode == "multiregime":
        return 1.0
    if config.prior_mode == "fixed":
        return ratio
    if config.prior_mode != "curriculum":
        raise ValueError(f"Unknown prior_mode {config.prior_mode!r}.")
    plain_end = 0.10 * config.max_steps
    ramp_end = 0.50 * config.max_steps
    if step <= plain_end:
        return 0.0
    if step >= ramp_end:
        return ratio
    return ratio * (step - plain_end) / (ramp_end - plain_end)


def _needs_ordinary_training_prior(config: PlainPretrainingConfig) -> bool:
    """Whether any training update can draw the ordinary-prior branch."""
    return multiregime_probability(config, config.max_steps) < 1.0


def training_batch(
    config: PlainPretrainingConfig,
    prior,
    v4_loader: "MultiregimeV4DumpLoader | RoundRobinV4DumpLoader | None",
    v4_shared_loader: "MultiregimeV4DumpLoader | RoundRobinV4DumpLoader | None",
    episode_rng: np.random.Generator,
    step: int,
) -> list:
    """Draw one training batch (as a list of one or two model batches — see ``_v4_dump_batch``)
    under the configured ordinary/multiregime curriculum."""
    probability = multiregime_probability(config, step)
    if probability == 0.0 or episode_rng.random() >= probability:
        if v4_shared_loader is not None:
            return _v4_dump_batch(
                v4_shared_loader,
                expose_regime_id=config.v4_expose_regime_id,
                expose_regime_id_probability=config.v4_expose_regime_id_probability,
                expose_regime_id_paired=config.v4_expose_regime_id_paired,
                rng=episode_rng,
            )
        if prior is None:
            raise RuntimeError("The ordinary TabICL prior is required for this curriculum batch.")
        return [next(iter(prior))]
    return multiregime_batch(config, v4_loader, episode_rng)


def effective_warmup_steps(config: PlainPretrainingConfig) -> int:
    """warmup_proportion * max_steps when a proportion is set, else the fixed warmup_steps."""
    if config.warmup_proportion == -1:
        return config.warmup_steps
    return int(round(config.warmup_proportion * config.max_steps))


def _scheduler_lambda(config: PlainPretrainingConfig):
    warmup = effective_warmup_steps(config)

    def schedule(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, config.max_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        floor = config.min_learning_rate / config.learning_rate
        return floor + (1 - floor) * cosine

    return schedule


@contextmanager
def _preserved_rng_state() -> Iterator[None]:
    """Make periodic validation deterministic without perturbing training draws."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def validate(model: NanoTabPFNModel, config: PlainPretrainingConfig) -> dict[str, float]:
    """Evaluate a fixed, independent ordinary-prior validation stream."""
    with _preserved_rng_state():
        set_randomness_seed(config.seed + 100_000)
        model.eval()
        batches = iter(make_prior(config, batches=config.validation_batches * _MAX_NON_FINITE_BATCH_RETRIES))
        losses = []
        for _ in range(config.validation_batches):
            for _attempt in range(_MAX_NON_FINITE_BATCH_RETRIES):
                loss = query_loss(model, next(batches))
                if torch.isfinite(loss):
                    losses.append(float(loss))
                    break
            else:
                raise RuntimeError(
                    "Could not draw a finite validation batch within "
                    f"{_MAX_NON_FINITE_BATCH_RETRIES} attempts."
                )
    model.train()
    return {"query_cross_entropy": float(np.mean(losses)), "validation_batches": len(losses)}


def evaluate_v4_bank(
    model: NanoTabPFNModel,
    bank_path: str | Path,
    *,
    output: Path,
    split: str,
    tag: str,
    device: str | torch.device,
    max_episodes_per_forward: int | None = None,
) -> tuple[dict[str, float], Path]:
    """Evaluate and persist the full factorial v4 report for one checkpoint."""
    report = evaluate_multiregime_v4_bank(
        model, bank_path, device=device, max_episodes_per_forward=max_episodes_per_forward
    )
    destination = output / f"v4_{split}"
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{tag}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return (
        {
            "query_cross_entropy": float(report["overall"]["query_cross_entropy"]),
            "query_accuracy": float(report["overall"]["query_accuracy"]),
            "query_auc": float(report["overall"]["query_auc"]),
            "episodes": float(report["episodes"]),
        },
        report_path,
    )


def _checkpoint(
    model: NanoTabPFNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: PlainPretrainingConfig,
    step: int,
    validation: dict[str, float] | None,
    episode_rng: np.random.Generator,
    v4_validation: dict[str, float] | None = None,
    v4_test: dict[str, float] | None = None,
    original_dump_pointer: int | None = None,
) -> dict:
    return {
        "model_type": f"nanotabpfn_{config.prior_mode}_scm_pretraining",
        "architecture": config.architecture(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_config": asdict(config),
        "seed": config.seed,
        "step": step,
        "validation": validation,
        "v4_validation": v4_validation,
        "v4_test": v4_test,
        # Dynamic TabICL sampling is already governed by the captured RNG
        # state. A finite ordinary dump has an independent file cursor, which
        # must be recorded explicitly for a resumed mixed run to retain its
        # episode order.
        "original_dump_pointer": original_dump_pointer,
        "rng_state": _serializable_rng_state(),
        "episode_rng_state": episode_rng.bit_generator.state,
    }


def _original_dump_pointer(prior) -> int | None:
    """Return the finite ordinary-prior cursor, if this run uses one."""
    pointer = getattr(prior, "pointer", None)
    return None if pointer is None else int(pointer)


def _serializable_rng_state() -> dict:
    """Return an RNG snapshot accepted by ``torch.load(weights_only=True)``."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_kind": numpy_state[0],
        "numpy_keys": numpy_state[1].tolist(),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _as_rng_tensor(state) -> torch.Tensor:
    """A CPU uint8 tensor, whatever device or dtype the checkpoint carried.

    Resume loads the checkpoint with ``map_location=config.device``, which puts
    every tensor in it -- the RNG state included -- on the GPU.  ``set_rng_state``
    accepts only a CPU ByteTensor and raises ``TypeError`` otherwise, so a GPU
    resume failed on its first line before this coercion existed.
    """
    return torch.as_tensor(state).detach().cpu().to(torch.uint8)


def _restore_rng_state(rng_state: dict) -> None:
    random.setstate(rng_state["python"])
    np.random.set_state(
        (
            rng_state["numpy_kind"],
            np.asarray(rng_state["numpy_keys"], dtype=np.uint32),
            int(rng_state["numpy_position"]),
            int(rng_state["numpy_has_gauss"]),
            float(rng_state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(_as_rng_tensor(rng_state["torch"]))
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_rng_tensor(state) for state in rng_state["cuda"]])


def _make_tensorboard_writer(output: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError("TensorBoard logging requires `uv sync --extra tensorboard`.") from error
    return SummaryWriter(log_dir=output / "tensorboard")


class _NullWriter:
    """No-op writer used by CPU tests and installations without TensorBoard."""

    def add_scalar(self, *_args, **_kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def evaluate_tabarena_epoch(
    checkpoint: Path, output: Path, config: PlainPretrainingConfig, epoch: int
) -> dict[str, float]:
    """Run a progress-only TabArena-small evaluation without affecting training state."""
    from tfmplayground.experiments.evaluate_tabarena_small import SmallTabArenaConfig, run

    destination = output / "tabarena" / f"epoch-{epoch:03d}"
    with _preserved_rng_state():
        run(
            SmallTabArenaConfig(
                standalone_checkpoints=f"current={checkpoint}",
                include_vanilla=False,
                output_dir=str(destination),
                device=config.device,
                cache_directory=config.tabarena_cache_directory,
                subsample=config.tabarena_subsample,
                folds=config.tabarena_folds,
                repeats=config.tabarena_repeats,
                include_sklearn=False,
                include_tabpfn=False,
                label_source="real",
                contamination=0.0,
                seed=config.seed,
            )
        )
    with (destination / "overall.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    current = next(row for row in rows if row["model"] == "current")
    return {
        "tabarena_mean_roc_auc": float(current["mean_roc_auc"]),
        "tabarena_mean_accuracy": float(current["mean_accuracy"]),
    }


def run_pretraining(
    config: PlainPretrainingConfig,
    output_dir: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Run one fixed-budget seed and write resumable, inference-compatible artifacts."""
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available.")
    if (
        config.max_steps <= 0
        or config.micro_batch_size <= 0
        or config.accumulate_gradients <= 0
        or config.v4_evaluation_batch_size <= 0
    ):
        raise ValueError("max_steps, batch sizes, and accumulate_gradients must be positive.")
    if config.warmup_proportion != -1 and not 0 <= config.warmup_proportion < 1:
        raise ValueError("warmup_proportion must be -1 (use warmup_steps) or in [0, 1).")
    if config.batch_size_per_gp is not None and config.batch_size_per_gp != config.micro_batch_size:
        raise ValueError("Use one complete TabICL group per model batch: batch_size_per_gp == micro_batch_size.")
    intervals = (
        config.validation_interval,
        config.v4_validation_interval,
        config.v4_regime_info_interval,
        config.checkpoint_interval,
        config.epoch_steps,
    )
    if min(intervals) <= 0:
        raise ValueError("validation intervals, checkpoint_interval, and epoch_steps must be positive.")
    if config.uses_native_row_split_sampling:
        if config.max_rows is None or config.min_train_size is None or config.max_train_size is None:
            raise ValueError("min_rows, max_rows, min_train_size, and max_train_size must be supplied together.")
        if not 2 <= config.min_rows <= config.max_rows:
            raise ValueError("Native row bounds must satisfy 2 <= min_rows <= max_rows.")
        if isinstance(config.min_train_size, float) != isinstance(config.max_train_size, float):
            raise ValueError("Native train-size bounds must have the same type.")
        if isinstance(config.min_train_size, float):
            if not 0 < config.min_train_size < config.max_train_size < 1:
                raise ValueError("Fractional train-size bounds must satisfy 0 < min < max < 1.")
        elif not 0 < config.min_train_size < config.max_train_size <= config.min_rows:
            raise ValueError("Integer train-size bounds must leave at least one query row.")
    elif config.max_rows is not None or config.min_train_size is not None or config.max_train_size is not None:
        raise ValueError("Native row/split controls must be supplied together, including min_rows.")
    elif not 0 < config.support_size < config.rows:
        raise ValueError("support_size must leave at least one query row.")
    if config.prior_mode not in {"original", "fixed", "curriculum", "plain", "multiregime"}:
        raise ValueError("prior_mode must be original, fixed, curriculum, plain, or multiregime.")
    if not 0 <= config.multiregime_ratio <= 1:
        raise ValueError("multiregime_ratio must lie in [0, 1].")
    if config.multiregime_source not in {"legacy", "v4"}:
        raise ValueError("multiregime_source must be 'legacy' or 'v4'.")
    if config.v4_expose_regime_id and config.multiregime_source != "v4":
        raise ValueError("v4_expose_regime_id requires multiregime_source='v4'.")
    if config.v4_expose_regime_id_probability is not None and config.multiregime_source != "v4":
        raise ValueError("v4_expose_regime_id_probability requires multiregime_source='v4'.")
    if config.v4_expose_regime_id_paired and config.multiregime_source != "v4":
        raise ValueError("v4_expose_regime_id_paired requires multiregime_source='v4'.")
    if sum([
        config.v4_expose_regime_id,
        config.v4_expose_regime_id_probability is not None,
        config.v4_expose_regime_id_paired,
    ]) > 1:
        raise ValueError(
            "Choose at most one of v4_expose_regime_id / v4_expose_regime_id_probability / "
            "v4_expose_regime_id_paired."
        )
    if config.v4_expose_regime_id_probability is not None and not 0.0 <= config.v4_expose_regime_id_probability <= 1.0:
        raise ValueError("v4_expose_regime_id_probability must lie in [0, 1].")
    if config.original_source not in {"dynamic", "dump"}:
        raise ValueError("original_source must be 'dynamic' or 'dump'.")
    for name, bank_path in (
        ("v4_validation_bank_path", config.v4_validation_bank_path),
        ("v4_extra_validation_bank_path", config.v4_extra_validation_bank_path),
        ("v4_test_bank_path", config.v4_test_bank_path),
        ("v4_regime_info_bank_path", config.v4_regime_info_bank_path),
    ):
        if bank_path is not None and not Path(bank_path).is_file():
            raise FileNotFoundError(f"{name} does not point to a file: {bank_path}.")
    if config.v4_shared_dump_path is not None:
        for path in _dump_paths(config.v4_shared_dump_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"v4_shared_dump_path does not point to a file: {path}.")
    if (
        config.original_source == "dump"
        and _needs_ordinary_training_prior(config)
        and config.v4_shared_dump_path is None
        and (config.original_dump_path is None or not Path(config.original_dump_path).is_file())
    ):
        raise FileNotFoundError("original_source='dump' requires an existing original_dump_path.")
    if (
        config.multiregime_source == "v4"
        and multiregime_probability(config, config.max_steps) > 0
        and not config.v4_dump_path
    ):
        raise ValueError(
            "multiregime_source='v4' requires v4_dump_path when the selected prior mode draws multiregime episodes."
        )
    if config.v4_shared_dump_path is not None and config.multiregime_source != "v4":
        raise ValueError("v4_shared_dump_path requires multiregime_source='v4'.")

    output = Path(output_dir)
    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    elif not output.is_dir():
        raise ValueError("A resumed run must use the existing output directory.")

    set_randomness_seed(config.seed)
    model = NanoTabPFNModel(**config.architecture()).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scheduler_lambda(config))
    start_step = 0
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=config.device, weights_only=False)
        if state.get("architecture") != config.architecture():
            raise ValueError("Resume checkpoint architecture does not match the requested configuration.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        rng_state = state.get("rng_state")
        if rng_state is None:
            raise ValueError("Resume checkpoint is missing RNG state and cannot resume reproducibly.")
        _restore_rng_state(rng_state)

    if _needs_ordinary_training_prior(config) and config.v4_shared_dump_path is None:
        prior = (
            PriorDumpDataLoader(
                config.original_dump_path,
                num_steps=1,
                batch_size=config.micro_batch_size,
                device=torch.device(config.device),
            )
            if config.original_source == "dump"
            else make_prior(config, batches=1)
        )
    else:
        prior = None
    if resume_checkpoint is not None and config.original_source == "dump" and prior is not None:
        prior.pointer = int(state.get("original_dump_pointer", 0))
    v4_loader = (
        make_v4_dump_loader(
            config.v4_dump_path,
            batch_size=config.micro_batch_size,
            device=config.device,
            family=config.v4_dump_family,
            min_classes=config.v4_dump_min_classes,
        )
        if config.multiregime_source == "v4" and multiregime_probability(config, config.max_steps) > 0
        else None
    )
    v4_shared_loader = (
        make_v4_dump_loader(
            config.v4_shared_dump_path,
            batch_size=config.micro_batch_size,
            device=config.device,
        )
        if config.v4_shared_dump_path is not None and _needs_ordinary_training_prior(config)
        else None
    )
    episode_rng = np.random.default_rng(config.seed + 1)
    if resume_checkpoint is not None:
        episode_rng.bit_generator.state = state["episode_rng_state"]
    last_v4_validation = state.get("v4_validation") if resume_checkpoint is not None else None
    history_path = output / "history.jsonl"
    mode = "a" if resume_checkpoint is not None else "w"
    writer = _make_tensorboard_writer(output) if config.tensorboard else _NullWriter()
    with history_path.open(mode) as history:
        epoch_started_at = time.perf_counter()
        for step in range(start_step + 1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for _ in range(config.accumulate_gradients):
                for _attempt in range(_MAX_NON_FINITE_BATCH_RETRIES):
                    # A probability split can yield separate hidden and exposed
                    # subsets. Paired mode concatenates its same-episode variants
                    # along the batch axis, so it uses one model call here.
                    batches = training_batch(config, prior, v4_loader, v4_shared_loader, episode_rng, step)
                    losses = [query_loss(model, batch) for batch in batches]
                    if all(torch.isfinite(loss) for loss in losses):
                        break
                else:
                    raise RuntimeError(
                        "Could not draw a finite training batch within "
                        f"{_MAX_NON_FINITE_BATCH_RETRIES} attempts at step {step}."
                    )
                for loss in losses:
                    (loss / (config.accumulate_gradients * len(losses))).backward()
                    loss_total += float(loss.detach()) / (config.accumulate_gradients * len(losses))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step}.")
            optimizer.step()
            scheduler.step()

            validation = None
            if step % config.validation_interval == 0 or step == config.max_steps:
                validation = validate(model, config)
            v4_validation = None
            if config.v4_validation_bank_path is not None and (
                step % config.v4_validation_interval == 0 or step == config.max_steps
            ):
                v4_validation, _ = evaluate_v4_bank(
                    model,
                    config.v4_validation_bank_path,
                    output=output,
                    split="validation",
                    tag=f"step-{step:06d}",
                    device=config.device,
                    max_episodes_per_forward=config.v4_evaluation_batch_size,
                )
                last_v4_validation = v4_validation
            v4_extra = None
            if config.v4_extra_validation_bank_path is not None and (
                step % config.v4_validation_interval == 0 or step == config.max_steps
            ):
                v4_extra, _ = evaluate_v4_bank(
                    model,
                    config.v4_extra_validation_bank_path,
                    output=output,
                    split=f"validation_{config.v4_extra_validation_tag}",
                    tag=f"step-{step:06d}",
                    device=config.device,
                    max_episodes_per_forward=config.v4_evaluation_batch_size,
                )
            v4_regime_info = None
            if config.v4_regime_info_bank_path is not None and (
                step % config.v4_regime_info_interval == 0 or step == config.max_steps
            ):
                # Import lazily: ordinary pretraining does not need the diagnostic
                # script or its additional evaluation dependencies.
                from scripts.evaluate_v4_regime_information import evaluate as evaluate_regime_information

                v4_regime_info = evaluate_regime_information(
                    model,
                    config.v4_regime_info_bank_path,
                    device=config.device,
                    information=("hidden", "shuffled", "true"),
                    multiclass_only=True,
                    multiregime_only=True,
                    episode_in_cell=0,
                    cell_subsample_modulus=config.v4_regime_info_cell_subsample_modulus,
                    cell_subsample_remainder=0,
                    max_episodes=None,
                    seed=config.seed,
                    max_episodes_per_forward=config.v4_evaluation_batch_size,
                    with_auc=False,
                    bootstrap_replicates=200,
                )
                v4_regime_info["step"] = step
                progress_path = output / "regime_information_progress.jsonl"
                with progress_path.open("a") as progress:
                    progress.write(json.dumps(v4_regime_info, sort_keys=True) + "\n")
            epoch = step // config.epoch_steps
            tabarena = None
            epoch_seconds = None
            if step % config.epoch_steps == 0:
                if config.tabarena_every_epoch:
                    epoch_checkpoint = output / f"epoch-{epoch:03d}-checkpoint.pth"
                    torch.save(
                        _checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            config,
                            step,
                            validation,
                            episode_rng,
                            last_v4_validation,
                            original_dump_pointer=_original_dump_pointer(prior),
                        ),
                        epoch_checkpoint,
                    )
                    tabarena = evaluate_tabarena_epoch(epoch_checkpoint, output, config, epoch)
                    # The checkpoint is needed to evaluate this epoch, but retaining all
                    # twenty per-seed copies is unnecessary. Keep the documented 10k
                    # resumable milestones (and the final checkpoint) instead.
                    if step % config.checkpoint_interval != 0:
                        epoch_checkpoint.unlink()
                epoch_seconds = time.perf_counter() - epoch_started_at
                epoch_started_at = time.perf_counter()
            row = {
                "step": step,
                "epoch": epoch,
                "query_cross_entropy": loss_total,
                "gradient_norm": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "multiregime_probability": multiregime_probability(config, step),
                **({"epoch_seconds": epoch_seconds} if epoch_seconds is not None else {}),
                **({f"validation_{key}": value for key, value in validation.items()} if validation else {}),
                **({f"v4_validation_{key}": value for key, value in v4_validation.items()} if v4_validation else {}),
                **({f"v4_validation_{config.v4_extra_validation_tag}_{key}": value for key, value in v4_extra.items()} if v4_extra else {}),
                **({
                    "regime_info_probe_episodes": v4_regime_info["episodes_per_condition"],
                    "regime_info_probe_true_minus_hidden_ce": float(np.average(
                        [item["cross_entropy_difference"] for item in v4_regime_info["paired_differences"]
                         if item["information"] == "true" and item["reference"] == "hidden"],
                        weights=[item["episodes"] for item in v4_regime_info["paired_differences"]
                                 if item["information"] == "true" and item["reference"] == "hidden"],
                    )),
                    "regime_info_probe_true_minus_hidden_accuracy": float(np.average(
                        [item["accuracy_difference"] for item in v4_regime_info["paired_differences"]
                         if item["information"] == "true" and item["reference"] == "hidden"],
                        weights=[item["episodes"] for item in v4_regime_info["paired_differences"]
                                 if item["information"] == "true" and item["reference"] == "hidden"],
                    )),
                } if v4_regime_info else {}),
                **(tabarena or {}),
            }
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            if validation is not None or v4_validation is not None or v4_extra is not None or v4_regime_info is not None or epoch_seconds is not None:
                print(json.dumps(row, sort_keys=True), flush=True)
            writer.add_scalar("train/query_cross_entropy", loss_total, step)
            writer.add_scalar("train/gradient_norm", gradient_norm, step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/multiregime_probability", multiregime_probability(config, step), step)
            if epoch_seconds is not None:
                writer.add_scalar("train/epoch_seconds", epoch_seconds, step)
            if validation is not None:
                writer.add_scalar("validation/query_cross_entropy", validation["query_cross_entropy"], step)
            if v4_validation is not None:
                writer.add_scalar("v4_validation/query_cross_entropy", v4_validation["query_cross_entropy"], step)
                writer.add_scalar("v4_validation/query_accuracy", v4_validation["query_accuracy"], step)
                writer.add_scalar("v4_validation/query_auc", v4_validation["query_auc"], step)
            if tabarena is not None:
                writer.add_scalar("tabarena/mean_roc_auc", tabarena["tabarena_mean_roc_auc"], step)
                writer.add_scalar("tabarena/mean_accuracy", tabarena["tabarena_mean_accuracy"], step)
            writer.flush()
            if step % config.checkpoint_interval == 0:
                state = _checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    validation,
                    episode_rng,
                    last_v4_validation,
                    original_dump_pointer=_original_dump_pointer(prior),
                )
                torch.save(state, output / f"checkpoint-{step:06d}.pth")
            if config.latest_checkpoint_interval and step % config.latest_checkpoint_interval == 0:
                state = _checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    validation,
                    episode_rng,
                    last_v4_validation,
                    original_dump_pointer=_original_dump_pointer(prior),
                )
                # write-then-rename so a kill mid-save never leaves a truncated latest checkpoint
                tmp = output / "latest_checkpoint.pth.tmp"
                torch.save(state, tmp)
                os.replace(tmp, output / "latest_checkpoint.pth")

    final_validation = validate(model, config)
    v4_test = None
    if config.v4_test_bank_path is not None:
        v4_test, _ = evaluate_v4_bank(
            model,
            config.v4_test_bank_path,
            output=output,
            split="test",
            tag="final",
            device=config.device,
            max_episodes_per_forward=config.v4_evaluation_batch_size,
        )
    final_state = _checkpoint(
        model,
        optimizer,
        scheduler,
        config,
        config.max_steps,
        final_validation,
        episode_rng,
        last_v4_validation,
        v4_test,
        original_dump_pointer=_original_dump_pointer(prior),
    )
    torch.save(final_state, output / "final_checkpoint.pth")
    (output / "final_validation.json").write_text(json.dumps(final_validation, indent=2) + "\n")
    if v4_loader is not None:
        v4_loader.close()
    if v4_shared_loader is not None:
        v4_shared_loader.close()
    writer.close()
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PlainPretrainingConfig()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--require-cuda", action="store_true")
    for name in (
        "seed",
        "max_steps",
        "micro_batch_size",
        "batch_size_per_gp",
        "accumulate_gradients",
        "warmup_steps",
        "validation_interval",
        "validation_batches",
        "v4_validation_interval",
        "v4_evaluation_batch_size",
        "checkpoint_interval",
        "latest_checkpoint_interval",
        "epoch_steps",
        "support_size",
        "query_size",
        "min_rows",
        "max_rows",
        "min_features",
        "max_features",
        "max_classes",
        "embedding_size",
        "num_attention_heads",
        "mlp_hidden_size",
        "num_layers",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in (
        "learning_rate",
        "min_learning_rate",
        "warmup_proportion",
        "weight_decay",
        "gradient_clip",
        "multiregime_ratio",
        "multiregime_contamination",
        "regime_coherence",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--prior-type", default=defaults.prior_type, choices=("mlp_scm", "tree_scm", "mix_scm"))
    parser.add_argument("--min-train-size", type=float, default=defaults.min_train_size)
    parser.add_argument("--max-train-size", type=float, default=defaults.max_train_size)
    parser.add_argument(
        "--prior-mode",
        choices=("original", "fixed", "curriculum", "plain", "multiregime"),
        default=defaults.prior_mode,
    )
    parser.add_argument("--multiregime-source", choices=("legacy", "v4"), default=defaults.multiregime_source)
    parser.add_argument("--v4-dump-path", default=defaults.v4_dump_path)
    parser.add_argument("--v4-dump-family", default=defaults.v4_dump_family, choices=("soft_gate", "persistent"),
                        help="ablation: serve only this routing family from the multiregime dump(s)")
    parser.add_argument("--v4-dump-min-classes", type=int, default=defaults.v4_dump_min_classes,
                        help="ablation: serve only multiregime episodes with at least this many classes (e.g. 3)")
    parser.add_argument("--v4-shared-dump-path", default=defaults.v4_shared_dump_path)
    parser.add_argument(
        "--v4-expose-regime-id",
        action=argparse.BooleanOptionalAction,
        default=defaults.v4_expose_regime_id,
        help="append randomly relabelled realized v4 regime IDs as an observed context feature",
    )
    parser.add_argument(
        "--v4-expose-regime-id-probability",
        type=float,
        default=defaults.v4_expose_regime_id_probability,
        help="per-episode Bernoulli(p) alternative to --v4-expose-regime-id: exposed and hidden "
             "episodes in the same draw are different episodes (mutually exclusive with "
             "--v4-expose-regime-id and --v4-expose-regime-id-paired)",
    )
    parser.add_argument(
        "--v4-expose-regime-id-paired",
        action=argparse.BooleanOptionalAction,
        default=defaults.v4_expose_regime_id_paired,
        help="train on both a hidden and an exposed copy of every multiregime episode every step "
             "(mutually exclusive with --v4-expose-regime-id and --v4-expose-regime-id-probability)",
    )
    parser.add_argument("--original-source", choices=("dynamic", "dump"), default=defaults.original_source)
    parser.add_argument("--original-dump-path", default=defaults.original_dump_path)
    parser.add_argument("--v4-validation-bank-path", default=defaults.v4_validation_bank_path)
    parser.add_argument("--v4-extra-validation-bank-path", default=defaults.v4_extra_validation_bank_path)
    parser.add_argument("--v4-extra-validation-tag", default=defaults.v4_extra_validation_tag)
    parser.add_argument("--v4-test-bank-path", default=defaults.v4_test_bank_path)
    parser.add_argument("--v4-regime-info-bank-path", default=defaults.v4_regime_info_bank_path)
    parser.add_argument("--v4-regime-info-interval", type=int, default=defaults.v4_regime_info_interval)
    parser.add_argument(
        "--v4-regime-info-cell-subsample-modulus",
        type=int,
        default=defaults.v4_regime_info_cell_subsample_modulus,
        help="sample one held-out episode from every Nth deterministic multiclass multiregime cell per probe",
    )
    parser.add_argument(
        "--tabarena-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=defaults.tabarena_every_epoch,
    )
    parser.add_argument("--tabarena-folds", type=int, default=defaults.tabarena_folds)
    parser.add_argument("--tabarena-repeats", type=int, default=defaults.tabarena_repeats)
    parser.add_argument("--tabarena-subsample", type=int, default=defaults.tabarena_subsample)
    parser.add_argument("--tabarena-cache-directory", default=defaults.tabarena_cache_directory)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=defaults.tensorboard)
    return parser


if __name__ == "__main__":
    args = vars(build_parser().parse_args())
    output_dir = args.pop("output_dir")
    resume_checkpoint = args.pop("resume_checkpoint")
    print(run_pretraining(PlainPretrainingConfig(**args), output_dir, resume_checkpoint=resume_checkpoint))
