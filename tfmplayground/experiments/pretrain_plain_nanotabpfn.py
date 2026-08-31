"""Matched plain and multiregime, from-scratch nanoTabPFN pretraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.continuous_episodes import (
    SCM_FAMILIES,
    TRAIN_REGIME,
    sample_scm_multiregime_episode,
)
from tfmplayground.external_priors import TabICLPriorDataLoader
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
    accumulate_gradients: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 2_000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    validation_interval: int = 5_000
    validation_batches: int = 16
    checkpoint_interval: int = 10_000
    support_size: int = 128
    query_size: int = 32
    min_features: int = 2
    max_features: int = 12
    max_classes: int = 2
    prior_type: str = "mix_scm"
    prior_mode: Literal["plain", "multiregime", "curriculum"] = "plain"
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
        return self.support_size + self.query_size

    def architecture(self) -> dict[str, int]:
        return {
            "embedding_size": self.embedding_size,
            "num_attention_heads": self.num_attention_heads,
            "mlp_hidden_size": self.mlp_hidden_size,
            "num_layers": self.num_layers,
            "num_outputs": self.max_classes,
        }


def make_prior(config: PlainPretrainingConfig, *, batches: int, device: str | torch.device | None = None):
    """Build the ordinary TabICL prior with the experiment's exact table split."""
    return TabICLPriorDataLoader(
        num_steps=batches,
        batch_size=config.micro_batch_size,
        # TabICL samples ``randint(min, max)``; this pair yields exactly rows.
        num_datapoints_min=config.rows,
        num_datapoints_max=config.rows + 1,
        min_features=config.min_features,
        max_features=config.max_features,
        max_num_classes=config.max_classes,
        device=torch.device(config.device if device is None else device),
        prior_type=config.prior_type,
        min_train_size=config.support_size,
        max_train_size=config.support_size + 1,
    )


def query_loss(model: NanoTabPFNModel, batch) -> torch.Tensor:
    """Cross entropy on query labels only; support labels are model inputs."""
    if not isinstance(batch, dict):
        logits = model(batch.support_x, batch.support_y, batch.query_x)[..., :2]
        return F.cross_entropy(logits.reshape(-1, 2), batch.query_y.reshape(-1).long())
    split = int(batch["train_test_split_index"])
    x, y = batch["x"], batch["y"]
    logits = model(x[:, :split], y[:, :split], x[:, split:])[..., :2]
    target = y[:, split:].reshape(-1).long()
    return F.cross_entropy(logits.reshape(-1, 2), target)


def multiregime_batch(config: PlainPretrainingConfig, rng: np.random.Generator):
    """Draw one equally weighted within- and cross-family SCM mixture episode."""
    sources: tuple[str | tuple[str, str], ...] = (
        *SCM_FAMILIES,
        ("mlp_scm", "tree_scm"),
    )
    source = sources[int(rng.integers(len(sources)))]
    return sample_scm_multiregime_episode(
        rng,
        regime=TRAIN_REGIME,
        family=source,
        batch_size=config.micro_batch_size,
        support_size=config.support_size,
        query_count=config.query_size,
        noise=0.0,
        contamination=config.multiregime_contamination,
        regime_coherence=config.regime_coherence,
        device=config.device,
    )


def multiregime_probability(config: PlainPretrainingConfig, step: int) -> float:
    """Return the multiregime share for this optimizer step.

    ``curriculum`` presents only ordinary TabICL ``mix_scm`` tables for the
    first 10% of the update budget, then linearly ramps the *episode* mixture
    from 0% to 50% between 10% and 50%, and retains a 50% multiregime share
    for the rest of training.  It is intentionally independent of the
    learning-rate schedule so that the curriculum remains explicit in run
    metadata.
    """
    if config.prior_mode == "plain":
        return 0.0
    if config.prior_mode == "multiregime":
        return 1.0
    plain_end = 0.10 * config.max_steps
    ramp_end = 0.50 * config.max_steps
    if step <= plain_end:
        return 0.0
    if step >= ramp_end:
        return 0.5
    return 0.5 * (step - plain_end) / (ramp_end - plain_end)


def training_batch(
    config: PlainPretrainingConfig,
    prior,
    episode_rng: np.random.Generator,
    step: int,
):
    """Draw one training batch under the configured ordinary/multiregime curriculum."""
    probability = multiregime_probability(config, step)
    if probability == 0.0 or episode_rng.random() >= probability:
        if prior is None:
            raise RuntimeError("The ordinary TabICL prior is required for this curriculum batch.")
        return next(iter(prior))
    return multiregime_batch(config, episode_rng)


def _scheduler_lambda(config: PlainPretrainingConfig):
    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
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


def _checkpoint(
    model: NanoTabPFNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: PlainPretrainingConfig,
    step: int,
    validation: dict[str, float] | None,
    episode_rng: np.random.Generator,
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
        "rng_state": _serializable_rng_state(),
        "episode_rng_state": episode_rng.bit_generator.state,
    }


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
    torch.set_rng_state(rng_state["torch"])
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


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
    if config.max_steps <= 0 or config.micro_batch_size <= 0 or config.accumulate_gradients <= 0:
        raise ValueError("max_steps, micro_batch_size, and accumulate_gradients must be positive.")
    if min(config.validation_interval, config.checkpoint_interval, config.epoch_steps) <= 0:
        raise ValueError("validation_interval, checkpoint_interval, and epoch_steps must be positive.")
    if not 0 < config.support_size < config.rows:
        raise ValueError("support_size must leave at least one query row.")
    if config.prior_mode not in {"plain", "multiregime", "curriculum"}:
        raise ValueError("prior_mode must be 'plain', 'multiregime', or 'curriculum'.")

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

    prior = make_prior(config, batches=1) if config.prior_mode != "multiregime" else None
    episode_rng = np.random.default_rng(config.seed + 1)
    if resume_checkpoint is not None:
        episode_rng.bit_generator.state = state["episode_rng_state"]
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
                    batch = training_batch(config, prior, episode_rng, step)
                    loss = query_loss(model, batch)
                    if torch.isfinite(loss):
                        break
                else:
                    raise RuntimeError(
                        "Could not draw a finite training batch within "
                        f"{_MAX_NON_FINITE_BATCH_RETRIES} attempts at step {step}."
                    )
                (loss / config.accumulate_gradients).backward()
                loss_total += float(loss.detach()) / config.accumulate_gradients
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step}.")
            optimizer.step()
            scheduler.step()

            validation = None
            if step % config.validation_interval == 0 or step == config.max_steps:
                validation = validate(model, config)
            epoch = step // config.epoch_steps
            tabarena = None
            epoch_seconds = None
            if step % config.epoch_steps == 0:
                epoch_checkpoint = output / f"epoch-{epoch:03d}-checkpoint.pth"
                torch.save(
                    _checkpoint(model, optimizer, scheduler, config, step, validation, episode_rng), epoch_checkpoint
                )
                if config.tabarena_every_epoch:
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
                **(tabarena or {}),
            }
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            if validation is not None or epoch_seconds is not None:
                print(json.dumps(row, sort_keys=True), flush=True)
            writer.add_scalar("train/query_cross_entropy", loss_total, step)
            writer.add_scalar("train/gradient_norm", gradient_norm, step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/multiregime_probability", multiregime_probability(config, step), step)
            if epoch_seconds is not None:
                writer.add_scalar("train/epoch_seconds", epoch_seconds, step)
            if validation is not None:
                writer.add_scalar("validation/query_cross_entropy", validation["query_cross_entropy"], step)
            if tabarena is not None:
                writer.add_scalar("tabarena/mean_roc_auc", tabarena["tabarena_mean_roc_auc"], step)
                writer.add_scalar("tabarena/mean_accuracy", tabarena["tabarena_mean_accuracy"], step)
            writer.flush()
            if step % config.checkpoint_interval == 0:
                state = _checkpoint(model, optimizer, scheduler, config, step, validation, episode_rng)
                torch.save(state, output / f"checkpoint-{step:06d}.pth")

    final_validation = validate(model, config)
    final_state = _checkpoint(model, optimizer, scheduler, config, config.max_steps, final_validation, episode_rng)
    torch.save(final_state, output / "final_checkpoint.pth")
    (output / "final_validation.json").write_text(json.dumps(final_validation, indent=2) + "\n")
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
        "accumulate_gradients",
        "warmup_steps",
        "validation_interval",
        "validation_batches",
        "checkpoint_interval",
        "epoch_steps",
        "support_size",
        "query_size",
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
        "weight_decay",
        "gradient_clip",
        "multiregime_contamination",
        "regime_coherence",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--prior-type", default=defaults.prior_type, choices=("mlp_scm", "tree_scm", "mix_scm"))
    parser.add_argument("--prior-mode", choices=("plain", "multiregime", "curriculum"), default=defaults.prior_mode)
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
