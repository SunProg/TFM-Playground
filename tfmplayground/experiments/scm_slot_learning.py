"""Pilot slot-attention training on the constrained TabICL SCM prior."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.multiregime_v2 import RegimeEpisode, stack_regime_episodes
from tfmplayground.experiments.pretrain_multiregime_v2 import (
    V2TrainingConfig,
    build_model,
    episode_loss,
)
from tfmplayground.experiments.scm_prior_learning import summary
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.slot_regime import SlotRegimePrediction
from tfmplayground.utils import set_randomness_seed


def episode_batch(prior: SCMRegimeConfig, seed: int, batch_size: int, *, query_size: int) -> RegimeEpisode:
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, batch_size)
    episodes = [sample_scm_regime_episode(replace(prior, query_size=query_size), seed=int(child)) for child in seeds]
    return stack_regime_episodes(episodes)


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
    accuracies, losses = [], []
    for start in range(0, episodes, 8):
        count = min(8, episodes - start)
        episode = episode_batch(prior, namespace + start, count, query_size=256)
        log_prob, episode = _prediction_log_probabilities(
            model, episode, shuffle_seed=namespace + start if shuffle else None
        )
        target = episode.query_y.long()
        accuracies.extend((log_prob.argmax(-1) == target).float().mean(-1).tolist())
        losses.extend(F.nll_loss(log_prob.transpose(1, 2), target, reduction="none").mean(-1).tolist())
    return {"accuracy": summary(accuracies), "cross_entropy": summary(losses)}


def train_one(
    model_type: str,
    seed: int,
    prior: SCMRegimeConfig,
    *,
    steps: int,
    validation_episodes: int,
    test_episodes: int,
    output: Path,
):
    set_randomness_seed(seed)
    config = V2TrainingConfig(
        seed=seed,
        device="cpu",
        max_steps=steps,
        micro_batch_size=8,
        accumulate_gradients=1,
        learning_rate=1e-3,
        min_learning_rate=1e-3,
        warmup_steps=1,
        weight_decay=0.01,
        gradient_clip=1.0,
        model_type=model_type,
        input_mode="latent",
        curriculum_mode="multiregime",
        num_slots=2,
        num_slot_iterations=3,
        slot_layer_index=0,
        slot_layer_indices=(0, 1),
        support_size=prior.support_size,
        query_size=32,
        min_features=prior.task_features + 1,
        max_features=prior.task_features + 1,
        embedding_size=32,
        num_attention_heads=4,
        mlp_hidden_size=64,
        num_layers=2,
        num_outputs=2,
    )
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    name = f"{model_type}-seed-{seed}"
    checkpoint = output / f"{name}.pth"
    history, best = [], float("inf")
    started = time.monotonic()
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            metrics = evaluate(model, prior, episodes=validation_episodes, namespace=2_000_000)
            row = {"step": step, "seconds": time.monotonic() - started, **metrics}
            history.append(row)
            if metrics["cross_entropy"]["mean"] < best:
                best = metrics["cross_entropy"]["mean"]
                torch.save({"state_dict": model.state_dict(), "step": step}, checkpoint)
            print(json.dumps({"run": name, **row}), flush=True)
        if step == steps:
            break
        model.train()
        episode = episode_batch(prior, seed * 100_000 + step, 8, query_size=32)
        loss, _ = episode_loss(model, episode, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    result = {
        "model_type": model_type,
        "seed": seed,
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
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["table_slot_head", "table_slot_backbone"],
        choices=("table_slot_head", "table_slot_backbone", "table_slot_mufasa"),
    )
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("Output already contains results; choose a fresh directory.")
    torch.set_num_threads(4)
    prior = SCMRegimeConfig()
    results = {"config": {**vars(args), "output": str(args.output), "prior": asdict(prior)}, "runs": []}
    for model_type in args.models:
        for seed in args.seeds:
            results["runs"].append(
                train_one(
                    model_type,
                    seed,
                    prior,
                    steps=args.steps,
                    validation_episodes=args.validation_episodes,
                    test_episodes=args.test_episodes,
                    output=args.output,
                )
            )
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
