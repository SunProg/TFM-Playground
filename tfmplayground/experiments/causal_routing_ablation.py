"""Causal seven-combination sweep for the three routing fixes."""

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
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v2 import stack_regime_episodes
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.causal_routing import CausalRoutingModel
from tfmplayground.models.slot_regime import slot_regime_loss


def configurations():
    # Empty tuple is the unchanged baseline; the remaining seven are exactly
    # {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}.
    combinations = (
        (),
        (1,),
        (2,),
        (3,),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 2, 3),
    )
    return [
        dict(seed=seed, fixes=fixes, name="none" if not fixes else "+".join(map(str, fixes)))
        for seed in (11, 12)
        for fixes in combinations
    ]


def _prior(args):
    return SCMRegimeConfig(
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


def _metrics(prediction, episode):
    logp = prediction.marginal_log_probabilities()[0]
    y = episode.query_y[0]
    losses = -logp.gather(-1, y[:, None].long()).squeeze(-1)
    result = {
        "nll": float(losses.mean()),
        "accuracy": float((logp.argmax(-1) == y).float().mean()),
        "query_entropy": float(prediction.gate_entropy().mean()),
        "query_confidence": float(prediction.gate().max(-1).values.mean()),
    }
    for name, mask in (("minor", episode.query_z[0] == 0), ("major", episode.query_z[0] == 1)):
        if int(mask.sum()) == 0:
            continue
        probability = logp[:, 1].detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        selected = mask.detach().cpu().numpy()
        result[f"{name}_nll"] = float(losses[mask].mean())
        result[f"{name}_accuracy"] = float((logp[mask].argmax(-1) == y[mask]).float().mean())
        result[f"{name}_auc"] = float(roc_auc_score(labels[selected], probability[selected]))
    return result


@torch.no_grad()
def evaluate(model, episodes, device):
    model.eval()
    rows = []
    for episode in episodes:
        episode = episode.to(device)
        prediction = model(*episode.latent_inputs())
        row = _metrics(prediction, episode)
        row.update({k: float(v) for k, v in model.last.items()})
        rows.append(row)
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args):
    torch.set_num_threads(1)
    cell = configurations()[args.index]
    prior = _prior(args)
    fixes = set(cell["fixes"])
    config = {
        "cell": cell,
        "fixes": {
            "1_query_support_retrieval": 1 in fixes,
            "2_slot_aware_loss": 2 in fixes,
            "3_blind_routing_no_label_bypass": 3 in fixes,
        },
        "architecture": {
            "width": args.width,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "slots": 4,
        },
        "prior": asdict(prior),
        "training": vars(args),
        "validation": {"episodes": args.validation_episodes, "seed_start": args.validation_episode_start},
        "test": {"episodes": args.test_episodes, "seed_start": args.test_episode_start},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(cell["seed"])
    model = CausalRoutingModel(
        query_retrieval=1 in fixes,
        slot_loss=2 in fixes,
        blind_routing=3 in fixes,
        width=args.width,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    validation = [
        sample_scm_regime_episode(
            replace(prior, query_size=args.validation_query_size),
            seed=args.validation_episode_start + i,
        )
        for i in range(args.validation_episodes)
    ]
    best = math.inf
    latest = output / "latest.pth"
    started = time.monotonic()
    for step in range(args.steps):
        model.train()
        progress = max(0.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        multiplier = (
            (step + 1) / max(1, args.warmup_steps)
            if step < args.warmup_steps
            else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        optimizer.zero_grad()
        first = cell["seed"] * 10**9 + step * args.batch_size
        batch = stack_regime_episodes(
            [sample_scm_regime_episode(prior, seed=first + j) for j in range(args.batch_size)]
        ).to(args.device)
        prediction = model(*batch.latent_inputs())
        query_loss = slot_regime_loss(prediction, batch.query_y)
        auxiliary = model.slot_loss(prediction, batch.support_y)
        total = query_loss + auxiliary
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "step": step + 1,
            "query_nll": float(query_loss.detach()),
            "slot_loss": float(auxiliary.detach()),
            "elapsed_seconds": time.monotonic() - started,
        }
        if (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            val = evaluate(model, validation, args.device)
            metrics["validation"] = val
            if val["nll"] < best:
                best = val["nll"]
                atomic_save(
                    {"model": model.state_dict(), "step": step + 1, "best_nll": best, "config": config},
                    output / "best.pth",
                )
            atomic_save({"model": model.state_dict(), "step": step + 1, "best_nll": best, "config": config}, latest)
            print(json.dumps(metrics), flush=True)
        if (step + 1) % 50 == 0 or step == 0:
            print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")

    state = torch.load(output / "best.pth", map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    test = [
        sample_scm_regime_episode(replace(prior, query_size=args.test_query_size), seed=args.test_episode_start + i)
        for i in range(args.test_episodes)
    ]
    test_metrics = evaluate(model, test, args.device)
    (output / "test.json").write_text(json.dumps({"checkpoint_step": state["step"], "metrics": test_metrics}, indent=2))
    (output / "complete.json").write_text(
        json.dumps(
            {"steps": args.steps, "best_validation_nll": best, "test": test_metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(len(configurations())), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-query-size", type=int, default=256)
    parser.add_argument("--validation-episode-start", type=int, default=2000000)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--test-query-size", type=int, default=256)
    parser.add_argument("--test-episode-start", type=int, default=3000000)
    run(parser.parse_args())
