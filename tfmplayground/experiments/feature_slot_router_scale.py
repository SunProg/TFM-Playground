"""Train FeatureSlotRouter on the same budget/prior conventions as
attention_slot_router_scale.py. No scope axis -- FeatureSlotRouter has no
TableSlotAdapter and therefore no scope choice; only seed varies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tfmplayground.experiments.multiregime_v3 import OriginalPrior, V3Config
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.feature_slot_router import FeatureSlotRouter, feature_slot_router_loss


@dataclass
class DeviceEpisode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    def latent_inputs(self):
        return self.support_x, self.support_y, self.query_x


def build_prior(args):
    if args.prior == "scm":
        prior = SCMRegimeConfig(
            support_size=args.support_size,
            query_size=args.query_size,
            task_features=2,
            cue_separation=1.5,
            regime_probability=0.65,
            label_noise=0.05,
            calibration_size=128,
        )
        return (lambda seed: sample_scm_regime_episode(prior, seed=seed)), asdict(prior)
    if args.prior == "v3_original":
        v3_config = V3Config(
            support_size=args.support_size, query_size=args.query_size, min_features=2, max_features=12, num_groups=5
        )
        original = OriginalPrior(v3_config)
        return (lambda seed: original.sample(seed=seed)), asdict(v3_config)
    raise ValueError(f"Unknown prior: {args.prior!r}")


def configurations():
    return [dict(seed=seed) for seed in (11, 12)]


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    finite = [g.detach().float().square().sum() for g in gradients if g is not None]
    return float(torch.stack(finite).sum().sqrt()) if finite else 0.0


@torch.no_grad()
def evaluate(model, episodes):
    model.eval()
    measurements = []
    for episode in episodes:
        prediction, responsibilities = model(*episode.latent_inputs())
        loss = feature_slot_router_loss(prediction, episode.query_y)
        row = {k: float(v) for k, v in model.last.items()}
        row.update(nll=float(loss), accuracy=float((prediction.argmax(-1) == episode.query_y).float().mean()))
        measurements.append(row)
    return {k: float(sum(m[k] for m in measurements) / len(measurements)) for k in measurements[0]}, measurements


def run(args):
    torch.set_num_threads(1)
    cell = configurations()[args.index]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    architecture = dict(width=args.width, hidden=args.hidden, layers=args.layers, heads=args.heads, num_slots=4)
    sample_episode, prior_config = build_prior(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        cell=cell,
        architecture=architecture,
        prior=args.prior,
        prior_config=prior_config,
        training={k: v for k, v in vars(args).items() if k not in ("output", "index")},
        model="FeatureSlotRouter: query -> column-attention slot -> slot_value(real support labels) retrieval",
    )
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains a different configuration")
    config_path.write_text(json.dumps(config, indent=2))
    torch.manual_seed(cell["seed"])
    model = FeatureSlotRouter(**architecture)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    latest = output / "latest.pth"
    start, best = 0, math.inf
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_nll"]
    if args.prior == "v3_original":
        validation_seeds = [
            (10_000_000_000 + cell["seed"] * 10_000 + 0 * 1000 + i) % (2**32 - 1) for i in range(args.validation_episodes)
        ]
    else:
        validation_episode_start = getattr(args, "validation_episode_start", 900000)
        validation_seeds = [validation_episode_start + i for i in range(args.validation_episodes)]
    heldout = []
    for seed in validation_seeds:
        episode = sample_episode(seed)
        heldout.append(
            DeviceEpisode(
                episode.support_x.to(args.device),
                episode.support_y.to(args.device),
                episode.query_x.to(args.device),
                episode.query_y.to(args.device),
            )
        )
    parameters = list(model.parameters())
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
        means = dict(query_nll=0.0)
        for micro in range(args.accumulate):
            first = cell["seed"] * 10**9 + (step * args.accumulate + micro) * args.batch_size
            episodes = [sample_episode(first + j) for j in range(args.batch_size)]
            support_x = torch.cat([e.support_x for e in episodes]).to(args.device)
            support_y = torch.cat([e.support_y for e in episodes]).to(args.device)
            query_x = torch.cat([e.query_x for e in episodes]).to(args.device)
            query_y = torch.cat([e.query_y for e in episodes]).to(args.device)
            torch.manual_seed(first)
            prediction, responsibilities = model(support_x, support_y, query_x)
            query_loss = feature_slot_router_loss(prediction, query_y)
            if not torch.isfinite(query_loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            if micro == 0 and (step == start or (step + 1) % args.validation_interval == 0):
                means["query_backbone_gradient_norm"] = gradient_norm(query_loss, parameters)
            (query_loss / args.accumulate).backward()
            means["query_nll"] += float(query_loss.detach()) / args.accumulate
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
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--prior", choices=("scm", "v3_original"), default="scm")
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
    parser.add_argument("--validation-episode-start", type=int, default=900000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    run(parser.parse_args())
