"""Evaluate saved realistic SCM checkpoints overall and by latent regime."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.multiregime_v2 import RegimeEpisode, stack_regime_episodes
from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig
from tfmplayground.experiments.pretrain_multiregime_v2 import build_model as build_v2_model
from tfmplayground.experiments.scm_regime_prior import sample_scm_regime_episode
from tfmplayground.experiments.scm_table_slot_head_sweep import CONDITIONS, PRIOR_PROFILES, build_model
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import SlotRegimePrediction
from tfmplayground.utils import set_randomness_seed

HEAD_CONDITIONS = tuple(CONDITIONS)
VARIANT_TYPES = ("table_slot_backbone", "table_slot_mufasa")


def episode_batch(prior, seed: int, batch_size: int) -> RegimeEpisode:
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, batch_size)
    episodes = [sample_scm_regime_episode(prior, seed=int(child)) for child in seeds]
    return stack_regime_episodes(episodes)


def build_nano(prior, seed: int, device: str):
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        model_type="tabpfn",
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
    return NanoTabPFNModel(**config.architecture()).to(device)


def build_variant(model_type: str, prior, seed: int, device: str):
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        model_type=model_type,
        num_slots=2,
        num_slot_iterations=3,
        slot_layer_indices=(0, 1),
        table_slot_scope="cell_and_data",
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
    return build_v2_model(config).to(device)


def build_head(condition: str, prior, seed: int, device: str):
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        model_type="table_slot_head",
        input_mode="latent",
        curriculum_mode="multiregime",
        num_slots=2,
        num_slot_iterations=3,
        slot_layer_index=0,
        slot_layer_indices=(0, 1),
        table_slot_scope="cell_and_data",
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
    return build_model(config, condition)


def log_probabilities(model, episode):
    prediction = model(*episode.latent_inputs())
    if isinstance(prediction, SlotRegimePrediction):
        return prediction.marginal_log_probabilities()
    return F.log_softmax(prediction, dim=-1)


def summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    return {"mean": mean, "n_episodes": int(len(values))}


def binary_auc(scores, targets):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=np.int64)
    positives = targets == 1
    negatives = targets == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (ranks[start] + ranks[end - 1]) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = float(original_ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


@torch.no_grad()
def evaluate(model, prior, *, episodes: int, namespace: int):
    model.eval()
    device = next(model.parameters()).device
    values = {
        "overall": {"accuracy": [], "cross_entropy": []},
        "majority_z1": {"accuracy": [], "cross_entropy": []},
        "minority_z0": {"accuracy": [], "cross_entropy": []},
    }
    pooled_scores = {name: [] for name in values}
    pooled_targets = {name: [] for name in values}
    regime_fractions = []
    for start in range(0, episodes, 8):
        count = min(8, episodes - start)
        episode = episode_batch(prior, namespace + start, count).to(device)
        log_prob = log_probabilities(model, episode)
        target = episode.query_y.long()
        correct = log_prob.argmax(-1).eq(target)
        losses = F.nll_loss(log_prob.transpose(1, 2), target, reduction="none")
        positive_scores = log_prob[..., 1].exp()
        for row in range(count):
            values["overall"]["accuracy"].append(correct[row].float().mean().item())
            values["overall"]["cross_entropy"].append(losses[row].mean().item())
            pooled_scores["overall"].append(positive_scores[row].cpu().numpy())
            pooled_targets["overall"].append(target[row].cpu().numpy())
            z = episode.query_z[row]
            regime_fractions.append(float(z.float().mean().item()))
            for name, regime in (("majority_z1", 1), ("minority_z0", 0)):
                mask = z == regime
                values[name]["accuracy"].append(correct[row][mask].float().mean().item())
                values[name]["cross_entropy"].append(losses[row][mask].mean().item())
                pooled_scores[name].append(positive_scores[row][mask].cpu().numpy())
                pooled_targets[name].append(target[row][mask].cpu().numpy())
    result = {}
    for name, metrics in values.items():
        result[name] = {metric: summary(items) for metric, items in metrics.items()}
        result[name]["auc"] = binary_auc(
            np.concatenate(pooled_scores[name]), np.concatenate(pooled_targets[name])
        )
    result["mean_query_fraction_z1"] = float(np.mean(regime_fractions))
    return result


def load_checkpoint(model, path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return int(saved["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--nano-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--episodes", type=int, default=128)
    args = parser.parse_args()
    if args.require_cuda and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("--require-cuda requested, but CUDA is unavailable")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    prior = PRIOR_PROFILES["realistic"]
    runs = []

    specs = []
    for condition in HEAD_CONDITIONS:
        for seed in (11, 12):
            checkpoint = args.head_root / condition / f"seed-{seed}" / f"{condition}-seed-{seed}.pth"
            specs.append((condition, seed, checkpoint))
    for seed in (11, 12):
        checkpoint = args.nano_root / f"seed-{seed}" / f"nanotabpfn-seed-{seed}.pth"
        specs.append(("nanotabpfn", seed, checkpoint))
    for model_type in VARIANT_TYPES:
        for seed in (11, 12):
            checkpoint = args.variant_root / model_type / f"seed-{seed}" / f"{model_type}-seed-{seed}.pth"
            specs.append((model_type, seed, checkpoint))

    for model_name, seed, checkpoint in specs:
        set_randomness_seed(seed)
        if model_name in HEAD_CONDITIONS:
            model = build_head(model_name, prior, seed, args.device)
        elif model_name == "nanotabpfn":
            model = build_nano(prior, seed, args.device)
        else:
            model = build_variant(model_name, prior, seed, args.device)
        selected_step = load_checkpoint(model, checkpoint)
        metrics = evaluate(model, prior, episodes=args.episodes, namespace=3_000_000)
        row = {"model": model_name, "seed": seed, "selected_step": selected_step, "metrics": metrics}
        runs.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"prior": asdict(prior), "episodes": args.episodes, "runs": runs}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
