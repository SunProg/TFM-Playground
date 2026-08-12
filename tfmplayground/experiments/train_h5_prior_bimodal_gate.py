"""Add an adaptive-K ambiguity gate on top of an h5-prior bimodal (K=2) latent filter.

The h5-prior bimodal family (`train_h5_prior_bimodal_filter.py`) produces a
two-hypothesis `NanoTabPFNIntegratedLatentFilter`, which has no ambiguity gate:
its output is used directly, with no vanilla fallback. This script expands such a
checkpoint into an `AdaptiveKParticleFilter` (K particles + a learned ambiguity
gate) and trains *only* the gate, on episodes drawn from the same HDF5 prior dump
the source filter was trained on -- so the gate is calibrated in the same data
distribution as the filter it gates, rather than on the unrelated 1-feature
four-mode task used by `train_four_mode_particle_filter.py`.

The gate blends `(1 - alpha) * vanilla + alpha * particle`. Note the practical
consequence: the gate can only ever recover vanilla's performance unless the
underlying filter's predictions actually beat vanilla on the target
distribution. Train the source filter properly before expecting a gate to help.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.evaluate_integrated_tabarena import release_device_memory
from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    generate_h5_prior_bimodal_episodes,
)
from tfmplayground.experiments.train_adaptive_particle_filter import (
    _full_support_vanilla_query,
    adaptive_loss,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import (
    AdaptiveKParticleFilter,
    checkpoint_sha256,
    expand_two_to_k_particles,
    save_adaptive_checkpoint,
)
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed

# The HDF5 episode adapter hard-requires this layout.
H5_SUPPORT_COUNT = 32
H5_STREAM_COUNT = 32
H5_QUERY_COUNT = 4


@dataclass(frozen=True)
class H5GateConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    source_checkpoint: str = "runs/h5_prior_bimodal_filter/k2-eval1000-smoke-v2/selected_checkpoint.pth"
    prior_dump: str = "300k_150x5_2.h5"
    output_dir: str | None = None
    device: str = "cpu"
    particle_count: int = 4
    min_features: int = 1
    max_features: int = 16
    # Pairing an episode needs ~70 attempts on average for this dump, so the
    # PriorBimodalConfig default of 64 is too low; the source filter run used 5000.
    max_pair_attempts: int = 5000
    support_disagreement_max: float = 0.20
    stream_disagreement_min: float = 0.25
    query_disagreement_min: float = 0.25
    batch_size: int = 2
    accumulate_gradients: int = 8
    steps: int = 500
    gate_learning_rate: float = 1e-3
    initial_particle_jitter: float = 0.05
    evaluation_trials: int = 64


def validate_config(config: H5GateConfig) -> None:
    if config.particle_count < 2:
        raise ValueError("particle_count must be at least 2 (expansion needs a 2-hypothesis source).")
    if min(config.batch_size, config.accumulate_gradients, config.evaluation_trials) < 1:
        raise ValueError("Batch, accumulation and evaluation counts must be positive.")
    if config.steps < 1:
        raise ValueError("Training steps must be positive.")
    if config.initial_particle_jitter <= 0:
        raise ValueError("Positive symmetry-breaking jitter is required to differentiate particles.")
    if not Path(config.prior_dump).exists():
        raise FileNotFoundError(f"Prior dump not found: {config.prior_dump}")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def episode_config(config: H5GateConfig) -> PriorBimodalConfig:
    return PriorBimodalConfig(
        initial_support_count=H5_SUPPORT_COUNT,
        stream_count=H5_STREAM_COUNT,
        query_count=H5_QUERY_COUNT,
        min_features=config.min_features,
        max_features=config.max_features,
        support_disagreement_max=config.support_disagreement_max,
        stream_disagreement_min=config.stream_disagreement_min,
        query_disagreement_min=config.query_disagreement_min,
        max_pair_attempts=config.max_pair_attempts,
        device=config.device,
    )


def _episode_is_finite(batch) -> bool:
    return all(
        torch.isfinite(tensor).all()
        for tensor in (
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
            batch.query_y,
        )
    )


def finite_episode(
    config: H5GateConfig,
    prior_config: PriorBimodalConfig,
    rng: np.random.Generator,
    *,
    batch_size: int,
    max_resamples: int = 20,
):
    """Draw an episode, resampling past records with non-finite features.

    The shipped prior dump `300k_150x5_2.h5` contains a small number of corrupted
    records (5 of 300000, each with ~300 of 750 non-finite cells). Drawing one makes the
    whole forward/backward NaN, which gradient clipping cannot recover -- so the run dies
    permanently the first time the sampler happens to hit one. Note that
    `train_prior_bimodal_filter.py` consumes the same generator and has the same exposure.
    """
    for _ in range(max_resamples):
        batch = generate_h5_prior_bimodal_episodes(config.prior_dump, prior_config, rng, batch_size=batch_size)
        if _episode_is_finite(batch):
            return batch
    raise RuntimeError(
        f"Could not draw a finite episode in {max_resamples} attempts; the prior dump may be "
        "more badly corrupted than expected."
    )


def build_model(config: H5GateConfig) -> AdaptiveKParticleFilter:
    """Expand the two-hypothesis h5 filter into K particles plus an ambiguity gate."""
    source, _ = load_integrated_checkpoint(config.source_checkpoint)
    if source.num_hypotheses != 2:
        raise ValueError(f"Source must be a two-hypothesis filter, got num_hypotheses={source.num_hypotheses}.")
    vanilla = init_model_from_state_dict_file(config.official_checkpoint)
    model = expand_two_to_k_particles(source, vanilla, particle_count=config.particle_count)
    # expand_two_to_k_particles duplicates the two source latents cyclically, so particles
    # beyond the first two start identical; jitter breaks that symmetry.
    if config.particle_count > 2:
        with torch.no_grad():
            latents = model.particle_model.initial_latents
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(config.seed + 17)
            latents[2:].add_(
                torch.randn(latents[2:].shape, generator=generator, device=latents.device)
                * config.initial_particle_jitter
            )
    return model


def train_gate(
    model: AdaptiveKParticleFilter, config: H5GateConfig
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    """Train only the ambiguity gate; the particle filter and vanilla backbone stay frozen."""
    model.particle_model.requires_grad_(False)
    model.vanilla_backbone.requires_grad_(False)
    model.ambiguity_gate.requires_grad_(True)
    model.to(config.device).train()
    optimizer = torch.optim.AdamW(model.ambiguity_gate.parameters(), lr=config.gate_learning_rate)

    prior_config = episode_config(config)
    rng = np.random.default_rng(config.seed + 100_000)
    history: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _ in range(config.accumulate_gradients):
            batch = finite_episode(config, prior_config, rng, batch_size=config.batch_size)
            prediction = model(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.stream_y,
                batch.query_x,
            )
            # These are empirical prior tasks, not hand-designed trajectories, so supervise the
            # final filter state only (controlled=False) and give vanilla the full support+stream
            # context so the fallback it is measured against is a fair one.
            prediction = type(prediction)(
                particle=prediction.particle,
                vanilla_stream_logits=prediction.vanilla_stream_logits,
                vanilla_query_logits=_full_support_vanilla_query(model, batch),
                ambiguity_probability=prediction.ambiguity_probability,
            )
            loss, metrics = adaptive_loss(prediction, batch, controlled=False)
            (loss / config.accumulate_gradients).backward()
            release_device_memory(config.device)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(model.ambiguity_gate.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step, **totals})
    return history, optimizer


@torch.no_grad()
def evaluate_gate(model: AdaptiveKParticleFilter, config: H5GateConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Report how decisive the gate is, and whether trusting the particles beats vanilla."""
    model.to(config.device).eval()
    prior_config = episode_config(config)
    rng = np.random.default_rng(config.seed + 500_000)
    rows = []
    remaining = config.evaluation_trials
    while remaining > 0:
        size = min(config.batch_size, remaining)
        remaining -= size
        batch = finite_episode(config, prior_config, rng, batch_size=size)
        prediction = model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        vanilla_logits = _full_support_vanilla_query(model, batch)
        vanilla_probability = vanilla_logits.softmax(-1)
        particle = prediction.particle_marginal_probabilities()[:, -1]
        labels = batch.query_y.long()
        vanilla_correct = (vanilla_probability.argmax(-1) == labels).float().mean(-1)
        particle_correct = (particle.argmax(-1) == labels).float().mean(-1)
        alpha = prediction.ambiguity_probability
        for index in range(size):
            rows.append(
                {
                    "alpha": float(alpha[index]),
                    "vanilla_accuracy": float(vanilla_correct[index]),
                    "particle_accuracy": float(particle_correct[index]),
                }
            )
        release_device_memory(config.device)

    trials = pd.DataFrame(rows)
    particle_edge = float((trials.particle_accuracy - trials.vanilla_accuracy).mean())
    metrics = {
        "mean_alpha": float(trials.alpha.mean()),
        "min_alpha": float(trials.alpha.min()),
        "max_alpha": float(trials.alpha.max()),
        "mean_vanilla_accuracy": float(trials.vanilla_accuracy.mean()),
        "mean_particle_accuracy": float(trials.particle_accuracy.mean()),
        "particle_edge_over_vanilla": particle_edge,
        "episodes_particle_wins_fraction": float((trials.particle_accuracy > trials.vanilla_accuracy).mean()),
    }
    # A gate is only useful if the particles it gates toward actually beat vanilla somewhere.
    checks = {
        "gate_is_not_saturated": 0.02 <= metrics["mean_alpha"] <= 0.98,
        "particles_beat_vanilla_somewhere": metrics["episodes_particle_wins_fraction"] >= 0.10,
    }
    report = {
        "threshold_profile": "h5_prior_bimodal_gate_v1",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "failure_reasons": [name for name, ok in checks.items() if not ok],
    }
    return trials, report


def run(config: H5GateConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "h5_prior_bimodal_gate" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    model = build_model(config)
    history, optimizer = train_gate(model, config)
    pd.DataFrame(history).to_csv(output_dir / "learning_curves.csv", index=False)
    trials, report = evaluate_gate(model, config)
    trials.to_csv(output_dir / "gate_metrics.csv", index=False)
    (output_dir / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    save_adaptive_checkpoint(
        output_dir / "adaptive_checkpoint.pth",
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=checkpoint_sha256(config.source_checkpoint),
        controlled_gate=report,
        optimizer_state=optimizer.state_dict(),
    )
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--source-checkpoint",
        default="runs/h5_prior_bimodal_filter/k2-eval1000-smoke-v2/selected_checkpoint.pth",
    )
    parser.add_argument("--prior-dump", default="300k_150x5_2.h5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--particle-count", type=int, default=4)
    parser.add_argument("--min-features", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--max-pair-attempts", type=int, default=5000)
    parser.add_argument("--support-disagreement-max", type=float, default=0.20)
    parser.add_argument("--stream-disagreement-min", type=float, default=0.25)
    parser.add_argument("--query-disagreement-min", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-particle-jitter", type=float, default=0.05)
    parser.add_argument("--evaluation-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    output = run(H5GateConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote h5-prior bimodal gate artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
