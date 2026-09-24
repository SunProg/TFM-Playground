"""Eight-arm causal routing sweep on the multi-regime v3 fixed mixture."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    # Running the file directly avoids importing the broad top-level package
    # registry, which eagerly initializes unrelated prior backends.
    _root = Path(__file__).resolve().parents[2]
    _package = ModuleType("tfmplayground")
    _package.__path__ = [str(_root / "tfmplayground")]
    sys.modules.setdefault("tfmplayground", _package)
    for _name in ("experiments", "models"):
        _submodule = ModuleType(f"tfmplayground.{_name}")
        _submodule.__path__ = [str(_root / "tfmplayground" / _name)]
        sys.modules.setdefault(f"tfmplayground.{_name}", _submodule)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v3 import (
    FAMILIES,
    OriginalPrior,
    V3Config,
    sample_episode,
)
from tfmplayground.models.causal_routing import CausalRoutingModel
from tfmplayground.models.slot_regime import slot_regime_loss


def configurations():
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


def _v3_config(args):
    return V3Config(
        support_size=args.support_size,
        query_size=args.query_size,
        min_features=2,
        max_features=args.max_features,
        num_groups=args.num_groups,
        calibration_size=256,
        num_regimes=2,
        separation=1.0,
        gate_strength=1.0,
        imbalance_ratio=0.5,
    )


def _task_config(config, seed):
    rng = np.random.default_rng(seed)
    generator = replace(
        config,
        num_regimes=int(rng.choice((2, 3, 4), p=(0.5, 0.3, 0.2))),
        separation=float(rng.choice((0.0, 0.5, 1.0, 2.0, 3.0), p=(0.05, 0.15, 0.3, 0.3, 0.2))),
        gate_strength=float(rng.choice((0.25, 1.0, 2.5))),
        imbalance_ratio=float(rng.choice((0.15, 0.5, 1.0))),
    )
    active_groups = int(rng.integers(1, config.num_groups + 1))
    return generator, active_groups


def _mixture_probability(mode, step, steps):
    """Probability of sampling a synthetic v3 episode at a training step."""
    if mode == "original":
        return 0.0
    if mode == "fixed":
        return 0.5
    if mode == "curriculum":
        # Match the canonical v3 curriculum: original-only for the first 10%
        # of training, a linear ramp to 50% v3 by 40%, then hold at 50%.
        fraction = step / max(1, steps)
        return 0.5 * float(np.clip((fraction - 0.1) / 0.3, 0.0, 1.0))
    raise ValueError(f"Unknown prior mode: {mode}")


def _sample(config, original, seed, *, mode, step, steps):
    rng = np.random.default_rng(seed)
    if rng.random() >= _mixture_probability(mode, step, steps):
        # Pure original-mode training omits nuisance group columns, matching
        # the canonical v3 harness. Mixed modes retain them so episode family
        # identity cannot be leaked by feature width.
        return original.sample(seed % (2**32 - 1), pad_groups=mode != "original")
    family = str(rng.choice(FAMILIES[1:], p=(0.2, 0.3, 0.2, 0.3)))
    generator, active_groups = _task_config(config, seed)
    return sample_episode(generator, family=family, seed=seed, active_groups=active_groups)


def _stack(episodes, device):
    return (
        torch.cat([episode.support_x for episode in episodes], dim=0).to(device),
        torch.cat([episode.support_y for episode in episodes], dim=0).to(device),
        torch.cat([episode.query_x for episode in episodes], dim=0).to(device),
        torch.cat([episode.query_y for episode in episodes], dim=0).to(device),
    )


def _metrics(prediction, episode):
    logp = prediction.marginal_log_probabilities()[0]
    y = episode.query_y.reshape(-1).long().to(logp.device)
    losses = -logp.gather(-1, y[:, None]).squeeze(-1)
    probability = logp[:, 1].detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    result = {
        "nll": float(losses.mean()),
        "accuracy": float((logp.argmax(-1) == y).float().mean()),
        "auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else float("nan"),
    }
    result.update({key: float(value) for key, value in prediction_model_stats(prediction).items()})
    return result


def prediction_model_stats(prediction):
    # The model records these values on itself; this helper is replaced by the
    # caller after each forward pass to keep episode metrics self-contained.
    return {}


@torch.no_grad()
def evaluate(model, bank, device):
    model.eval()
    output = {}
    for family, episodes in bank.items():
        rows = []
        for episode in episodes:
            support_x, support_y, query_x, _ = _stack([episode], device)
            prediction = model(support_x, support_y, query_x)
            row = _metrics(prediction, episode)
            row.update({key: float(value) for key, value in model.last.items()})
            rows.append(row)
        keys = rows[0].keys()
        output[family] = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
    all_rows = [row for family in output.values() for row in [family]]
    output["overall"] = {
        key: float(np.nanmean([row[key] for row in all_rows]))
        for key in all_rows[0]
        if key in ("nll", "accuracy", "auc")
    }
    return output


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args):
    torch.set_num_threads(1)
    cell = configurations()[args.index]
    fixes = set(cell["fixes"])
    prior = _v3_config(args)
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
        "prior": {
            "name": "v3_fixed_mixture",
            "mode": args.prior_mode,
            "generator": asdict(prior),
            "curriculum": "original 0-10%, linear ramp to 50% v3 by 40%, then fixed"
            if args.prior_mode == "curriculum"
            else None,
        },
        "training": vars(args),
        "evaluation": {"families": list(FAMILIES), "episodes_per_family": args.test_episodes},
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
    original = OriginalPrior(prior)
    validation = {}
    for family_index, family in enumerate(FAMILIES):
        validation[family] = []
        for episode_index in range(args.validation_episodes):
            seed = 2_000_000 + cell["seed"] * 10_000 + family_index * 1_000 + episode_index
            if family == "original":
                episode = original.sample(seed % (2**32 - 1))
            else:
                generator, active_groups = _task_config(prior, seed)
                episode = sample_episode(generator, family=family, seed=seed, active_groups=active_groups)
            validation[family].append(episode)

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
        episodes = [
            _sample(
                prior,
                original,
                cell["seed"] * 10**9 + step * args.batch_size + j,
                mode=args.prior_mode,
                step=step,
                steps=args.steps,
            )
            for j in range(args.batch_size)
        ]
        support_x, support_y, query_x, query_y = _stack(episodes, args.device)
        prediction = model(support_x, support_y, query_x)
        query_loss = slot_regime_loss(prediction, query_y)
        auxiliary = model.slot_loss(prediction, support_y)
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
            if val["overall"]["nll"] < best:
                best = val["overall"]["nll"]
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
    test = {}
    for family_index, family in enumerate(FAMILIES):
        test[family] = []
        for episode_index in range(args.test_episodes):
            seed = 3_000_000 + cell["seed"] * 10_000 + family_index * 1_000 + episode_index
            if family == "original":
                episode = original.sample(seed % (2**32 - 1))
            else:
                generator, active_groups = _task_config(prior, seed)
                episode = sample_episode(generator, family=family, seed=seed, active_groups=active_groups)
            test[family].append(episode)
    test_metrics = evaluate(model, test, args.device)
    (output / "test.json").write_text(json.dumps({"checkpoint_step": state["step"], "metrics": test_metrics}, indent=2))
    (output / "complete.json").write_text(
        json.dumps({"steps": args.steps, "best_validation_nll": best, "test": test_metrics}, indent=2)
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--prior-mode", choices=("original", "fixed", "curriculum"), default="fixed")
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=32)
    run(parser.parse_args())
