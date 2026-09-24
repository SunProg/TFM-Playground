"""Regime-stratified evaluation for the four SCM table-slot routing heads."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig
from tfmplayground.experiments.scm_regime_stratified_eval import evaluate, load_checkpoint
from tfmplayground.experiments.scm_table_slot_head_sweep import CONDITIONS, PRIOR_PROFILES, build_model
from tfmplayground.utils import set_randomness_seed


def make_config(prior, seed: int, device: str, architecture: dict[str, int]) -> V2TrainingConfig:
    return V2TrainingConfig(
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
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_layers=architecture["num_layers"],
        num_outputs=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=tuple(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("--require-cuda requested, but CUDA is unavailable")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    prior = PRIOR_PROFILES["realistic"]
    architecture = {
        "embedding_size": args.embedding_size,
        "num_attention_heads": args.num_attention_heads,
        "mlp_hidden_size": args.mlp_hidden_size,
        "num_layers": args.num_layers,
    }
    runs = []
    for condition in args.conditions:
        for seed in args.seeds:
            checkpoint = args.root / condition / f"seed-{seed}" / f"{condition}-seed-{seed}.pth"
            set_randomness_seed(seed)
            config = make_config(prior, seed, args.device, architecture)
            model = build_model(config, condition)
            selected_step = load_checkpoint(model, checkpoint)
            metrics = evaluate(model, prior, episodes=args.episodes, namespace=3_000_000)
            row = {
                "model": condition,
                "seed": seed,
                "selected_step": selected_step,
                "architecture": architecture,
                "metrics": metrics,
            }
            runs.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"prior": asdict(prior), "episodes": args.episodes, "architecture": architecture, "runs": runs}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
