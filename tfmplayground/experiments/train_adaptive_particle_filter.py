"""Train and evaluate an exact-fallback adaptive K-particle gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.train_sequential_latent_filter import (
    CONDITIONS,
    SequentialEpisodeBatch,
    SequentialFilterConfig,
    _canonical_permutation_indices,
    _effective_milestones,
    _js_divergence,
    _outcome_indices,
    controlled_gate_report,
    generate_controlled_episodes,
    make_tabicl_iterator,
    next_tabicl_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import (
    AdaptiveKParticleFilter,
    AdaptiveParticlePrediction,
    checkpoint_sha256,
    expand_two_to_k_particles,
    save_adaptive_checkpoint,
)
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class AdaptiveParticleConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    source_checkpoint: str = "runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth"
    output_dir: str | None = None
    device: str = "cpu"
    particle_count: int = 4
    prior_count: int = 128
    update_count: int = 128
    query_count: int = 4
    batch_size: int = 2
    accumulate_gradients: int = 8
    steps: int = 500
    learning_rate: float = 1e-3
    evaluation_trials: int = 256


def _episode_config(config: AdaptiveParticleConfig, *, seed: int, trials: int) -> SequentialFilterConfig:
    return SequentialFilterConfig(
        seed=seed,
        checkpoint=config.official_checkpoint,
        device=config.device,
        initial_support_count=config.prior_count,
        stream_count=config.update_count,
        query_count=config.query_count,
        batch_size=config.batch_size,
        controlled_steps=0,
        tabicl_steps=config.steps,
        evaluation_trials=trials,
        ordinary_evaluation_batches=0,
        plots=False,
    )


def adaptive_loss(
    prediction: AdaptiveParticlePrediction,
    batch: SequentialEpisodeBatch,
    *,
    controlled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    prequential = -prediction.prequential_log_likelihood_for(batch.stream_y).mean()
    joint = prediction.joint_probabilities()
    marginals = prediction.marginal_probabilities()
    if not controlled:
        joint = joint[:, -1:]
        marginals = marginals[:, -1:]
    indices = _outcome_indices(batch.query_y)
    selected = joint.gather(-1, indices[:, None, None].expand(-1, joint.shape[1], 1)).squeeze(-1)
    trajectory = -selected.clamp_min(1e-12).log().mean()
    labels = batch.query_y[:, None, :, None].expand(-1, marginals.shape[1], -1, 1)
    marginal = -marginals.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
    total = prequential + trajectory + marginal
    return total, {
        "loss": float(total.detach()),
        "prequential_loss": float(prequential.detach()),
        "trajectory_loss": float(trajectory.detach()),
        "marginal_loss": float(marginal.detach()),
        "ambiguity_probability": float(prediction.ambiguity_probability.mean().detach()),
        "effective_particles_final": float(prediction.effective_particle_count()[:, -1].mean().detach()),
    }


@torch.no_grad()
def _full_support_vanilla_query(
    model: AdaptiveKParticleFilter,
    batch: SequentialEpisodeBatch,
) -> torch.Tensor:
    support_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
    support_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1)
    return model.vanilla_backbone(
        (torch.cat((support_x, batch.query_x), dim=1), support_y),
        train_test_split_index=support_x.shape[1],
    )[..., :2].detach()


def train_gate(
    model: AdaptiveKParticleFilter, config: AdaptiveParticleConfig
) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    model.particle_model.requires_grad_(False)
    model.vanilla_backbone.requires_grad_(False)
    model.ambiguity_gate.requires_grad_(True)
    model.to(config.device).train()
    optimizer = torch.optim.AdamW(model.ambiguity_gate.parameters(), lr=config.learning_rate)
    episode_config = _episode_config(config, seed=config.seed + 100_000, trials=config.batch_size)
    iterator = make_tabicl_iterator(episode_config, config.steps * config.accumulate_gradients * 10)
    rng = np.random.default_rng(config.seed + 100_000)
    history = []
    for step in range(1, config.steps + 1):
        controlled = bool(step % 2)
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _ in range(config.accumulate_gradients):
            batch = (
                generate_controlled_episodes(episode_config, rng, batch_size=config.batch_size)
                if controlled
                else next_tabicl_episode(iterator, episode_config, rng)
            )
            prediction = model(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.stream_y,
                batch.query_x,
            )
            if not controlled:
                prediction = replace(
                    prediction,
                    vanilla_query_logits=_full_support_vanilla_query(model, batch),
                )
            loss, metrics = adaptive_loss(prediction, batch, controlled=controlled)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(model.ambiguity_gate.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step, "source": "controlled" if controlled else "tabicl", **totals})
    return history, optimizer


def _pairwise_max_js(slot_joint: torch.Tensor) -> torch.Tensor:
    values = []
    for left in range(slot_joint.shape[1]):
        for right in range(left + 1, slot_joint.shape[1]):
            values.append(_js_divergence(slot_joint[:, left], slot_joint[:, right]))
    return torch.stack(values, dim=1).max(dim=1).values


@torch.no_grad()
def _invariance(
    model: AdaptiveKParticleFilter, batch: SequentialEpisodeBatch, rng: np.random.Generator
) -> dict[str, float]:
    original = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x,
    )
    query_order = (3, 1, 0, 2)
    query = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x[:, query_order],
    )
    mapping = _canonical_permutation_indices(query_order, batch.query_x.device)
    query_error = (original.joint_probabilities() - query.joint_probabilities()[:, :, mapping]).abs().max()
    order = torch.as_tensor(rng.permutation(batch.stream_x.shape[1]), device=batch.stream_x.device)
    reordered = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x[:, order],
        batch.stream_y[:, order],
        batch.query_x,
    )
    return {
        "query_permutation_max_error": float(query_error),
        "evidence_order_weight_max_error": float(
            (original.log_weights[:, -1].exp() - reordered.log_weights[:, -1].exp()).abs().max()
        ),
        "evidence_order_joint_max_error": float(
            (original.joint_probabilities()[:, -1] - reordered.joint_probabilities()[:, -1]).abs().max()
        ),
    }


@torch.no_grad()
def evaluate_controlled_k(
    model: AdaptiveKParticleFilter, config: AdaptiveParticleConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.to(config.device).eval()
    episode_config = _episode_config(config, seed=config.seed + 500_000, trials=config.evaluation_trials)
    milestones = _effective_milestones(config.update_count)
    states = torch.as_tensor(milestones, device=config.device)
    rows = []
    invariance = {
        name: 0.0
        for name in (
            "query_permutation_max_error",
            "evidence_order_weight_max_error",
            "evidence_order_joint_max_error",
        )
    }
    for condition_index, condition in enumerate(CONDITIONS):
        rng = np.random.default_rng(config.seed + 500_000 + condition_index)
        batch = generate_controlled_episodes(
            episode_config, rng, condition=condition, batch_size=config.evaluation_trials
        )
        prediction = model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        weights = prediction.log_weights.index_select(1, states).exp()
        joint = prediction.joint_probabilities(states)
        marginals = prediction.marginal_probabilities(states)
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        class_one_mask = slot_joint[:, :, -1] > slot_joint[:, :, 0]
        class_zero_mask = ~class_one_mask
        class_zero_weight = (weights * class_zero_mask[:, None]).sum(-1)
        class_one_weight = (weights * class_one_mask[:, None]).sum(-1)
        best_zero = slot_joint[:, :, 0].argmax(1)
        best_one = slot_joint[:, :, -1].argmax(1)
        exact_p1 = batch.exact_p1.index_select(1, states)
        exact_binary = torch.stack((1 - exact_p1, exact_p1), dim=-1)
        marginal_js = _js_divergence(exact_binary, marginals.mean(2))
        errors = _invariance(
            model,
            SequentialEpisodeBatch(
                initial_support_x=batch.initial_support_x[:16],
                initial_support_y=batch.initial_support_y[:16],
                stream_x=batch.stream_x[:16],
                stream_y=batch.stream_y[:16],
                query_x=batch.query_x[:16],
                query_y=batch.query_y[:16],
            ),
            rng,
        )
        for name, value in errors.items():
            invariance[name] = max(invariance[name], value)
        for trial in range(config.evaluation_trials):
            target_class = int(batch.latent_class[trial])
            initial_class = int(batch.initial_evidence_class[trial])
            target = class_one_weight if target_class else class_zero_weight
            initial = class_one_weight if initial_class == 1 else class_zero_weight if initial_class == 0 else target
            opposite = class_zero_weight if initial_class == 1 else class_one_weight
            for state_index, milestone in enumerate(milestones):
                rows.append(
                    {
                        "condition": condition,
                        "trial": trial,
                        "milestone": milestone,
                        "weight_0": float(class_zero_weight[trial, state_index]),
                        "weight_1": float(class_one_weight[trial, state_index]),
                        "supporting_weight": float(target[trial, state_index]),
                        "initial_supporting_weight": float(initial[trial, state_index]),
                        "opposite_weight": float(opposite[trial, state_index]),
                        "incoherent_mass": float(
                            (1 - joint[trial, state_index, 0] - joint[trial, state_index, -1]).clamp_min(0)
                        ),
                        "marginal_js": float(marginal_js[trial, state_index]),
                        "mean_p1": float(marginals[trial, state_index, :, 1].mean()),
                        "exact_p1": float(exact_p1[trial, state_index]),
                        "slot_joint_js": float(_pairwise_max_js(slot_joint)[trial]),
                        "different_supporting_slots": float(best_zero[trial] != best_one[trial]),
                        "ambiguity_probability": float(prediction.ambiguity_probability[trial]),
                        "effective_particles": float(prediction.effective_particle_count(states)[trial, state_index]),
                    }
                )
    trials = pd.DataFrame(rows)
    return trials, controlled_gate_report(trials, invariance, stream_count=config.update_count)


def run(config: AdaptiveParticleConfig) -> Path:
    if config.update_count > config.prior_count or config.particle_count < 2:
        raise ValueError("Require K >= 2 and updates <= prior rows.")
    set_randomness_seed(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "adaptive_particle_filter" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    source, _ = load_integrated_checkpoint(config.source_checkpoint)
    vanilla = init_model_from_state_dict_file(config.official_checkpoint)
    model = expand_two_to_k_particles(source, vanilla, particle_count=config.particle_count)
    history, optimizer = train_gate(model, config)
    pd.DataFrame(history).to_csv(output_dir / "learning_curves.csv", index=False)
    trials, gate = evaluate_controlled_k(model, config)
    trials.to_csv(output_dir / "controlled_metrics.csv", index=False)
    (output_dir / "controlled_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    save_adaptive_checkpoint(
        output_dir / "adaptive_checkpoint.pth",
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=checkpoint_sha256(config.source_checkpoint),
        controlled_gate=gate,
        optimizer_state=optimizer.state_dict(),
    )
    summary = trials.groupby(["condition", "milestone"], as_index=False).mean(numeric_only=True)
    summary.to_csv(output_dir / "controlled_summary.csv", index=False)
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--source-checkpoint",
        default="runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--particle-count", type=int, default=4)
    parser.add_argument("--prior-count", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-trials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    output = run(AdaptiveParticleConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote adaptive particle-filter artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
