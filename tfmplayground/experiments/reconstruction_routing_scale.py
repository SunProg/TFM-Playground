"""Larger, resumable GPU comparison on the same SCM family as the CPU pilots."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from tfmplayground.experiments.multiregime_v2 import stack_regime_episodes
from tfmplayground.experiments.reconstruction_routing_pilot import PlainControl, PreviousControl, evaluate
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.reconstruction_routing import ReconstructionRouter
from tfmplayground.models.slot_regime import slot_regime_loss


def configurations():
    cells = []
    for seed in (11, 12):
        for scope in ("data", "cell_and_data", "cell"):
            cells.append(dict(seed=seed, scope=scope, arm="label_alpha", embedding_weight=0.0))
            for weight in (0.1, 1.0):
                if scope != "cell":
                    cells.append(dict(seed=seed, scope=scope, arm="embedding_mse", embedding_weight=weight))
                cells.append(dict(seed=seed, scope=scope, arm="values", embedding_weight=weight))
    for seed in (11, 12):
        cells.append(dict(seed=seed, scope="vanilla", arm="vanilla", embedding_weight=0.0))
    return cells


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.stack([g.detach().float().square().sum() for g in gradients if g is not None]).sum().sqrt())


def run(args):
    torch.set_num_threads(1)  # Tiny CPU SCM generators do not benefit from BLAS thread pools.
    cell = configurations()[args.index]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    architecture = dict(width=args.width, hidden=args.hidden, layers=args.layers, heads=args.heads, num_slots=4)
    prior_profile = getattr(args, "prior_profile", "scale")
    if prior_profile == "realistic":
        # Match the realistic SCM protocol used in the earlier 5,000-step
        # report.  The model architecture remains controlled by the CLI.
        prior = SCMRegimeConfig(
            support_size=args.support_size,
            query_size=args.query_size,
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
        )
    else:
        prior = SCMRegimeConfig(
            support_size=args.support_size,
            query_size=args.query_size,
            task_features=2,
            cue_separation=1.5,
            regime_probability=0.65,
            label_noise=0.05,
            calibration_size=128,
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        cell=cell,
        architecture=architecture,
        prior=asdict(prior),
        training={k: v for k, v in vars(args).items() if k not in ("output", "index")},
        prior_profile=prior_profile,
        prior_family=(
            "realistic SCM protocol from scm_realistic_5000_step_results.md"
            if prior_profile == "realistic"
            else "same noisy, overlapping-cue SCM family as the small pilots"
        ),
        alpha_loss="query NLL + support-label NLL + 0.05 MI",
        embedding_loss="query NLL + embedding_weight * embedding MSE",
        cell_alignment="support-row-zero Hungarian; augmented cell retrieval includes cell attention",
    )
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains a different configuration")
    config_path.write_text(json.dumps(config, indent=2))
    torch.manual_seed(cell["seed"])
    if cell["arm"] == "vanilla":
        model = PlainControl(**{k: architecture[k] for k in ("width", "hidden", "layers", "heads")})
        backbone = model.backbone
    elif cell["arm"] == "values":
        model = ReconstructionRouter(scope=cell["scope"], variant="values", **architecture)
        backbone = model.backbone
    else:
        model = PreviousControl(cell["scope"], cell["arm"], **architecture)
        backbone = model.model.backbone
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    latest = output / "latest.pth"
    start, best = 0, math.inf
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_nll"]
    # Disjoint episode seeds; fixed validation tasks shared across every arm.
    validation_query_size = getattr(args, "validation_query_size", None) or prior.query_size
    validation_episode_start = getattr(args, "validation_episode_start", 900000)
    heldout = [
        sample_scm_regime_episode(
            replace(prior, query_size=validation_query_size), seed=validation_episode_start + i
        ).to(args.device)
        for i in range(args.validation_episodes)
    ]
    parameters = list(backbone.parameters())
    start_time = time.monotonic()
    print(
        json.dumps({"config": config, "parameters": sum(p.numel() for p in model.parameters()), "resume_step": start}),
        flush=True,
    )
    for step in range(start, args.steps):
        model.train()
        progress = max(0.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        multiplier = (
            (step + 1) / max(1, args.warmup_steps)
            if step < args.warmup_steps
            else (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        optimizer.zero_grad()
        means = dict(query_nll=0.0, auxiliary=0.0)
        for micro in range(args.accumulate):
            first = cell["seed"] * 10**9 + (step * args.accumulate + micro) * args.batch_size
            batch = stack_regime_episodes(
                [sample_scm_regime_episode(prior, seed=first + j) for j in range(args.batch_size)]
            ).to(args.device)
            torch.manual_seed(first)
            prediction, auxiliary = model(*batch.latent_inputs())
            query_loss = slot_regime_loss(prediction, batch.query_y)
            weight = cell["embedding_weight"] if cell["arm"] != "label_alpha" else 1.0
            total = query_loss + weight * auxiliary
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            if micro == 0 and (step == start or (step + 1) % args.validation_interval == 0):
                means["query_backbone_gradient_norm"] = gradient_norm(query_loss, parameters)
                means["auxiliary_backbone_gradient_norm"] = gradient_norm(weight * auxiliary, parameters)
            (total / args.accumulate).backward()
            means["query_nll"] += float(query_loss.detach()) / args.accumulate
            means["auxiliary"] += float(auxiliary.detach()) / args.accumulate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = dict(step=step + 1, elapsed_seconds=time.monotonic() - start_time, **means)
        if (step + 1) % 50 == 0 or step == start:
            if args.device == "cuda":
                metrics["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps(metrics), flush=True)
        if (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            validation, per_episode = evaluate(model, heldout)
            metrics.update(validation=validation)
            improved = validation["nll"] < best
            best = min(best, validation["nll"])
            payload = dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, best_nll=best, config=config
            )
            atomic_save(payload, latest)
            if improved:
                atomic_save(payload, output / "best.pth")
            (output / "validation-latest.json").write_text(json.dumps(per_episode, indent=2))
            print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
    (output / "complete.json").write_text(json.dumps({"steps": args.steps, "best_validation_nll": best}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(len(configurations())), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prior-profile", choices=("scale", "realistic"), default="scale")
    for name, default in dict(
        steps=20000,
        width=192,
        hidden=768,
        layers=6,
        heads=6,
        support_size=128,
        query_size=32,
        batch_size=8,
        accumulate=4,
        warmup_steps=2000,
        validation_interval=500,
        validation_episodes=32,
    ).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument(
        "--validation-query-size",
        type=int,
        default=None,
        help="Query rows per validation episode; defaults to the training prior query size.",
    )
    parser.add_argument("--validation-episode-start", type=int, default=900000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    run(parser.parse_args())
