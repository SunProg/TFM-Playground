"""Evaluate the large SCM capacity-control checkpoints by latent regime."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig
from tfmplayground.experiments.scm_regime_stratified_eval import evaluate, load_checkpoint
from tfmplayground.experiments.scm_table_slot_head_sweep import PRIOR_PROFILES, build_model as build_head
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed


def build_nano(prior, seed: int, device: str, architecture: dict[str, int]):
    config = V2TrainingConfig(
        seed=seed,
        device=device,
        model_type="tabpfn",
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
    return NanoTabPFNModel(**config.architecture()).to(device)


def build_large_head(prior, seed: int, device: str, architecture: dict[str, int]):
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
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_layers=architecture["num_layers"],
        num_outputs=2,
    )
    return build_head(config, "decoder_baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=192)
    parser.add_argument("--num-attention-heads", type=int, default=6)
    parser.add_argument("--mlp-hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--skip-head", action="store_true", help="Evaluate only the NanoTabPFN checkpoint")
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
    specs = []
    for seed in (11, 12):
        specs.append(("nanotabpfn_large", seed, args.nano_root / f"seed-{seed}" / f"nanotabpfn-seed-{seed}.pth"))
        if not args.skip_head:
            specs.append(("table_slot_decoder_baseline_large", seed, args.head_root / "decoder_baseline" / f"seed-{seed}" / f"decoder_baseline-seed-{seed}.pth"))
    for model_name, seed, checkpoint in specs:
        set_randomness_seed(seed)
        model = (
            build_nano(prior, seed, args.device, architecture)
            if model_name == "nanotabpfn_large"
            else build_large_head(prior, seed, args.device, architecture)
        )
        selected_step = load_checkpoint(model, checkpoint)
        metrics = evaluate(model, prior, episodes=args.episodes, namespace=3_000_000)
        row = {
            "model": model_name,
            "seed": seed,
            "selected_step": selected_step,
            "architecture": architecture,
            "metrics": metrics,
        }
        runs.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"prior": asdict(prior), "episodes": args.episodes, "architecture": architecture, "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
