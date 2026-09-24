"""Train a plain NanoTabPFN baseline on the SCM routing-sweep profile.

This is the matched non-slot reference for
``scm_table_slot_head_sweep``: the prior, architecture, optimizer, episode
stream, validation schedule and shuffled-support control are identical.  The
only difference is that the model is the ordinary ``NanoTabPFNModel`` and is
optimized with query cross-entropy, without slot reconstruction or MI terms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig
from tfmplayground.experiments.scm_table_slot_head_sweep import (
    PRIOR_PROFILES,
    episode_batch,
    evaluate,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed


def build_model(
    prior: SCMRegimeConfig,
    device: str,
    *,
    embedding_size: int = 32,
    num_attention_heads: int = 4,
    mlp_hidden_size: int = 64,
    num_layers: int = 2,
) -> NanoTabPFNModel:
    config = V2TrainingConfig(
        device=device,
        model_type="tabpfn",
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
    return NanoTabPFNModel(**config.architecture()).to(config.device)


def train_one(
    seed: int,
    prior: SCMRegimeConfig,
    *,
    steps: int,
    validation_episodes: int,
    test_episodes: int,
    device: str,
    output: Path,
    embedding_size: int,
    num_attention_heads: int,
    mlp_hidden_size: int,
    num_layers: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
):
    set_randomness_seed(seed)
    model = build_model(
        prior,
        device,
        embedding_size=embedding_size,
        num_attention_heads=num_attention_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
    )
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
    name = f"nanotabpfn-seed-{seed}"
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
        logits = model(*episode.latent_inputs())
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), episode.query_y.reshape(-1).long())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    result = {
        "model_type": "nanotabpfn",
        "architecture": {
            "embedding_size": embedding_size,
            "num_attention_heads": num_attention_heads,
            "mlp_hidden_size": mlp_hidden_size,
            "num_layers": num_layers,
            "num_outputs": 2,
        },
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
    parser.add_argument("--prior-profile", choices=tuple(PRIOR_PROFILES), default="realistic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.min_learning_rate is None:
        args.min_learning_rate = args.learning_rate
    if (args.output / "results.json").exists():
        parser.error("Output already contains results; choose a fresh directory.")
    if args.require_cuda and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("--require-cuda requested, but CUDA is unavailable")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    prior = PRIOR_PROFILES[args.prior_profile]
    results = {
        "config": {**vars(args), "output": str(args.output), "prior": asdict(prior)},
        "runs": [],
    }
    for seed in args.seeds:
        results["runs"].append(
            train_one(
                seed,
                prior,
                steps=args.steps,
                validation_episodes=args.validation_episodes,
                test_episodes=args.test_episodes,
                device=args.device,
                output=args.output,
                embedding_size=args.embedding_size,
                num_attention_heads=args.num_attention_heads,
                mlp_hidden_size=args.mlp_hidden_size,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
            )
        )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
