"""Train the non-head table-slot placements on the realistic SCM profile.

This runner covers ``table_slot_backbone`` and ``table_slot_mufasa`` under the
same 5,000-step protocol as the head-routing and plain NanoTabPFN jobs. These
placements use the standard query-mixture objective; blind routing and support
reconstruction are head-only settings in ``table_slot.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tfmplayground.experiments.pretrain_multiregime_v2 import (
    V2TrainingConfig,
    episode_loss,
)
from tfmplayground.experiments.pretrain_multiregime_v2 import (
    build_model as build_v2_model,
)
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig
from tfmplayground.experiments.scm_table_slot_head_sweep import PRIOR_PROFILES, episode_batch, evaluate
from tfmplayground.utils import set_randomness_seed

MODEL_TYPES = ("table_slot_backbone", "table_slot_mufasa")


def build_model(model_type: str, prior: SCMRegimeConfig, device: str):
    config = V2TrainingConfig(
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
    return build_v2_model(config)


def train_one(
    model_type: str,
    seed: int,
    prior: SCMRegimeConfig,
    *,
    steps: int,
    validation_episodes: int,
    test_episodes: int,
    device: str,
    output: Path,
):
    set_randomness_seed(seed)
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        max_steps=steps,
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
    model = build_v2_model(config)
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
        episode = episode_batch(prior, seed * 100_000 + step, 8, query_size=32).to(device)
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
    parser.add_argument("--prior-profile", choices=tuple(PRIOR_PROFILES), default="realistic")
    parser.add_argument("--model-types", nargs="+", choices=MODEL_TYPES, default=list(MODEL_TYPES))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
    for model_type in args.model_types:
        for seed in args.seeds:
            results["runs"].append(
                train_one(
                    model_type,
                    seed,
                    prior,
                    steps=args.steps,
                    validation_episodes=args.validation_episodes,
                    test_episodes=args.test_episodes,
                    device=args.device,
                    output=args.output,
                )
            )
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
