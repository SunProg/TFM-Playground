"""Sweep label-routing variants of the table-slot head on the SCM prior.

The four cells keep the support reconstruction and slot-balance objectives
fixed while changing only the query routing/compositing path:

``decoder_baseline``
    learned decoder gate, attention-weighted support reconstruction.
``decoder_alpha``
    learned decoder gate, decoder-alpha-weighted support reconstruction.
``blind_decoder``
    learned decoder gate evaluated on the label-blind pass.
``blind_similarity``
    cosine-similarity gate between blind query/support representations.

The model in every cell is ``TableSlotModel(mode="head")`` from
``tfmplayground.models.table_slot``.  The support reconstruction term is
enabled for all cells; without it the ``alpha`` and baseline settings would
be identical during training because the mixture choice only affects that
auxiliary objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.multiregime_v2 import RegimeEpisode, stack_regime_episodes
from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import (
    SlotRegimePrediction,
    slot_mi_loss,
    slot_regime_loss,
    support_reconstruction_loss,
)
from tfmplayground.models.table_slot import TableSlotModel
from tfmplayground.utils import set_randomness_seed

CONDITIONS = {
    "decoder_baseline": {"query_routing_mode": "decoder", "reconstruction_mixture": "attention"},
    "decoder_alpha": {"query_routing_mode": "decoder", "reconstruction_mixture": "alpha"},
    "blind_decoder": {"query_routing_mode": "blind_decoder", "reconstruction_mixture": "attention"},
    "blind_similarity": {"query_routing_mode": "blind_similarity", "reconstruction_mixture": "attention"},
}

# ``control`` reproduces the original positive-control pilot.  ``realistic``
# keeps the same identifiable SCM mechanism family but makes the episode less
# forgiving: more covariates, overlapping regime cues, an imbalanced regime
# mixture, noisy labels, and deeper mechanisms.  It is still synthetic; the
# profile is a stress test before moving to an actual tabular dataset.
PRIOR_PROFILES = {
    "control": SCMRegimeConfig(),
    "realistic": SCMRegimeConfig(
        # Keep the 128/256 episode size used by the control so this stress
        # test changes data realism rather than silently changing compute.
        support_size=128,
        query_size=256,
        task_features=4,
        cue_separation=1.5,
        regime_probability=0.65,
        num_layers=2,
        hidden_dim=12,
        init_std=0.5,
        label_noise=0.05,
        calibration_size=1024,
        min_disagreement=0.20,
        max_disagreement=0.80,
        max_attempts=128,
    ),
}


def summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95": [mean - half, mean + half], "n_episodes": len(values)}


def episode_batch(prior: SCMRegimeConfig, seed: int, batch_size: int, *, query_size: int) -> RegimeEpisode:
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, batch_size)
    episodes = [sample_scm_regime_episode(replace(prior, query_size=query_size), seed=int(child)) for child in seeds]
    return stack_regime_episodes(episodes)


def build_model(config: V2TrainingConfig, condition: str) -> TableSlotModel:
    """Build one table-slot head with the requested routing/compositing cell."""
    settings = CONDITIONS[condition]
    backbone = NanoTabPFNModel(**config.architecture())
    return TableSlotModel(
        backbone,
        mode="head",
        num_slots=config.num_slots,
        num_slot_iterations=config.num_slot_iterations,
        max_classes=config.num_outputs,
        scope=config.table_slot_scope,
        query_routing_mode=settings["query_routing_mode"],
        reconstruction_mixture=settings["reconstruction_mixture"],
    ).to(config.device)


def _prediction_log_probabilities(model, episode, *, shuffle_seed: int | None = None):
    if shuffle_seed is not None:
        generator = np.random.default_rng(shuffle_seed)
        support_y = episode.support_y.detach().cpu().numpy().copy()
        for row in support_y:
            generator.shuffle(row)
        episode = replace(episode, support_y=torch.from_numpy(support_y).to(episode.support_y))
    prediction = model(*episode.latent_inputs())
    if isinstance(prediction, SlotRegimePrediction):
        return prediction.marginal_log_probabilities(), episode
    return F.log_softmax(prediction, dim=-1), episode


@torch.no_grad()
def evaluate(model, prior: SCMRegimeConfig, *, episodes: int, namespace: int, shuffle: bool = False):
    model.eval()
    device = next(model.parameters()).device
    accuracies, losses = [], []
    for start in range(0, episodes, 8):
        count = min(8, episodes - start)
        episode = episode_batch(prior, namespace + start, count, query_size=prior.query_size)
        episode = episode.to(device)
        log_prob, episode = _prediction_log_probabilities(
            model, episode, shuffle_seed=namespace + start if shuffle else None
        )
        target = episode.query_y.long()
        accuracies.extend((log_prob.argmax(-1) == target).float().mean(-1).tolist())
        losses.extend(F.nll_loss(log_prob.transpose(1, 2), target, reduction="none").mean(-1).tolist())
    return {"accuracy": summary(accuracies), "cross_entropy": summary(losses)}


def _training_loss(model, episode, *, support_reconstruction_weight: float, slot_mi_weight: float):
    prediction = model(*episode.latent_inputs(), reconstruct_support=True)
    target_loss = slot_regime_loss(prediction, episode.query_y)
    reconstruction = support_reconstruction_loss(prediction, episode.support_y)
    total = target_loss + support_reconstruction_weight * reconstruction
    mi = slot_mi_loss(prediction.support_attention)
    total = total + slot_mi_weight * mi
    return total, {
        "target_loss": float(target_loss.detach()),
        "reconstruction_nll": float(reconstruction.detach()),
        "mi_loss": float(mi.detach()),
    }


def train_one(
    condition: str,
    seed: int,
    prior: SCMRegimeConfig,
    *,
    steps: int,
    validation_episodes: int,
    test_episodes: int,
    support_reconstruction_weight: float,
    slot_mi_weight: float,
    device: str,
    output: Path,
    embedding_size: int,
    num_attention_heads: int,
    mlp_hidden_size: int,
    num_layers: int,
    num_slots: int,
    num_slot_iterations: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
):
    set_randomness_seed(seed)
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        max_steps=steps,
        micro_batch_size=8,
        accumulate_gradients=1,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        gradient_clip=1.0,
        model_type="table_slot_head",
        input_mode="latent",
        curriculum_mode="multiregime",
        num_slots=num_slots,
        num_slot_iterations=num_slot_iterations,
        slot_layer_index=0,
        slot_layer_indices=(0, 1),
        table_slot_scope="cell_and_data",
        support_size=prior.support_size,
        query_size=32,
        min_features=prior.task_features + 1,
        max_features=prior.task_features + 1,
        embedding_size=embedding_size,
        num_attention_heads=num_attention_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=2,
    )
    model = build_model(config, condition)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    if min_learning_rate <= 0 or min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be positive and no greater than learning_rate")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    def schedule(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return float(step_index + 1) / warmup_steps
        progress = (step_index - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_learning_rate / learning_rate + (1.0 - min_learning_rate / learning_rate) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    name = f"{condition}-seed-{seed}"
    checkpoint = output / f"{name}.pth"
    history, best = [], float("inf")
    started = time.monotonic()
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            metrics = evaluate(model, prior, episodes=validation_episodes, namespace=2_000_000)
            row = {"step": step, "seconds": time.monotonic() - started, "learning_rate": scheduler.get_last_lr()[0], **metrics}
            history.append(row)
            if metrics["cross_entropy"]["mean"] < best:
                best = metrics["cross_entropy"]["mean"]
                torch.save({"state_dict": model.state_dict(), "step": step}, checkpoint)
            print(json.dumps({"run": name, **row}), flush=True)
        if step == steps:
            break
        model.train()
        episode = episode_batch(prior, seed * 100_000 + step, 8, query_size=32).to(device)
        loss, _ = _training_loss(
            model,
            episode,
            support_reconstruction_weight=support_reconstruction_weight,
            slot_mi_weight=slot_mi_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    result = {
        "condition": condition,
        "seed": seed,
        "architecture": {
            "embedding_size": embedding_size,
            "num_attention_heads": num_attention_heads,
            "mlp_hidden_size": mlp_hidden_size,
            "num_layers": num_layers,
            "num_outputs": 2,
            "num_slots": num_slots,
            "num_slot_iterations": num_slot_iterations,
        },
        "selected_step": saved["step"],
        "history": history,
        "test": evaluate(model, prior, episodes=test_episodes, namespace=3_000_000),
        "test_shuffled_support_labels": evaluate(
            model, prior, episodes=test_episodes, namespace=3_000_000, shuffle=True
        ),
    }
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": name, "test": result["test"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-profile", choices=tuple(PRIOR_PROFILES), default="control")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=tuple(CONDITIONS))
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--support-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--slot-mi-weight", type=float, default=0.05)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument("--num-slot-iterations", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()
    if args.support_reconstruction_weight <= 0:
        parser.error("--support-reconstruction-weight must be positive for this sweep")
    if args.num_slots < 1:
        parser.error("--num-slots must be positive")
    if args.num_slot_iterations < 1:
        parser.error("--num-slot-iterations must be positive")
    if args.min_learning_rate is None:
        args.min_learning_rate = args.learning_rate
    if args.require_cuda and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("--require-cuda requested, but CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("Output already contains results; choose a fresh directory.")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    prior = PRIOR_PROFILES[args.prior_profile]
    results = {
        "config": {
            **vars(args),
            "output": str(args.output),
            "prior": asdict(prior),
            "conditions": {name: CONDITIONS[name] for name in args.conditions},
        },
        "runs": [],
    }
    for condition in args.conditions:
        for seed in args.seeds:
            results["runs"].append(
                train_one(
                    condition,
                    seed,
                    prior,
                    steps=args.steps,
                    validation_episodes=args.validation_episodes,
                    test_episodes=args.test_episodes,
                    support_reconstruction_weight=args.support_reconstruction_weight,
                    slot_mi_weight=args.slot_mi_weight,
                    device=args.device,
                    output=args.output,
                    embedding_size=args.embedding_size,
                    num_attention_heads=args.num_attention_heads,
                    mlp_hidden_size=args.mlp_hidden_size,
                    num_layers=args.num_layers,
                    num_slots=args.num_slots,
                    num_slot_iterations=args.num_slot_iterations,
                    learning_rate=args.learning_rate,
                    min_learning_rate=args.min_learning_rate,
                    warmup_steps=args.warmup_steps,
                )
            )
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
