"""The same routing-arm comparison as ``reconstruction_routing_scale.py``, on
the multiregime_v3 prior instead of the two-regime SCM family.

Mirrors that script's grid, architecture, and training budget exactly so the
two studies are directly comparable; only the episode source changes. Trains
on a fixed mixture of the four non-"original" v3 families (matching
``pretrain_multiregime_v3.py``'s established ``p=(0.2, 0.3, 0.2, 0.3)`` over
``shared_rule, soft_gate, independent, persistent``), one family per
micro-batch, with no group-count curriculum (``active_groups=None``, the
"fixed" condition that pilot also validated). "original" is held out of
training and is not sampled for validation either, since it is a diagnostic
control family elsewhere in this repo, not part of the routing comparison.

Every v3 episode's tensors are already fixed-width regardless of family or
the per-episode random mechanism feature count (``_model_x`` pads to
``max_features + num_groups`` unconditionally), so batches across different
families or micro-batches concatenate safely -- no ``stack_regime_episodes``
analogue is needed here, just ``torch.cat`` the way
``pretrain_multiregime_v3.py`` already batches this same episode type.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v3 import FAMILIES, V3Config, sample_episode
from tfmplayground.experiments.reconstruction_routing_pilot import PlainControl, PreviousControl, evaluate
from tfmplayground.models.reconstruction_routing import ReconstructionRouter
from tfmplayground.models.slot_regime import slot_regime_loss

#: Matches pretrain_multiregime_v3.py's training-time family mixture exactly
#: (FAMILIES[1:] == shared_rule, soft_gate, independent, persistent).
TRAINING_FAMILIES = FAMILIES[1:]
TRAINING_FAMILY_WEIGHTS = (0.2, 0.3, 0.2, 0.3)


@dataclass
class DeviceEpisode:
    """A v3 episode's tensors already moved to a device; matches the
    ``.latent_inputs()``/``.query_y`` interface ``evaluate()`` needs, the same
    interface ``scm_regime_prior``'s episode type exposes. ``V3Episode`` has
    no ``.to()`` of its own -- ``pretrain_multiregime_v3.py`` moves tensors at
    the batched-tensor level too, not per-episode."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    def latent_inputs(self):
        return self.support_x, self.support_y, self.query_x


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


def sample_batch(prior: V3Config, *, family: str, seed: int, batch_size: int) -> DeviceEpisode:
    episodes = [sample_episode(prior, family=family, seed=seed + j) for j in range(batch_size)]
    return DeviceEpisode(
        torch.cat([e.support_x for e in episodes]),
        torch.cat([e.support_y for e in episodes]),
        torch.cat([e.query_x for e in episodes]),
        torch.cat([e.query_y for e in episodes]),
    )


def run(args):
    torch.set_num_threads(1)
    cell = configurations()[args.index]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    architecture = dict(width=args.width, hidden=args.hidden, layers=args.layers, heads=args.heads, num_slots=4)
    prior = V3Config(
        support_size=args.support_size,
        query_size=args.query_size,
        max_features=args.max_features,
        num_groups=args.num_groups,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        cell=cell,
        architecture=architecture,
        prior=asdict(prior),
        training={k: v for k, v in vars(args).items() if k not in ("output", "index")},
        prior_family="multiregime_v3, mixture over shared_rule/soft_gate/independent/persistent",
        training_families=TRAINING_FAMILIES,
        training_family_weights=TRAINING_FAMILY_WEIGHTS,
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
    # Fixed validation tasks shared across every arm and seed: cycle evenly
    # through the four trained families rather than drawing them randomly, so
    # a held-out family imbalance cannot itself explain a validation-NLL gap
    # between arms.
    validation_episode_start = getattr(args, "validation_episode_start", 900000)
    heldout = []
    for i in range(args.validation_episodes):
        family = TRAINING_FAMILIES[i % len(TRAINING_FAMILIES)]
        episode = sample_episode(prior, family=family, seed=validation_episode_start + i)
        heldout.append(
            DeviceEpisode(
                episode.support_x.to(args.device),
                episode.support_y.to(args.device),
                episode.query_x.to(args.device),
                episode.query_y.to(args.device),
            )
        )
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
            family_rng = np.random.default_rng(first)
            family = str(family_rng.choice(TRAINING_FAMILIES, p=TRAINING_FAMILY_WEIGHTS))
            batch = sample_batch(prior, family=family, seed=first, batch_size=args.batch_size)
            batch = DeviceEpisode(
                batch.support_x.to(args.device),
                batch.support_y.to(args.device),
                batch.query_x.to(args.device),
                batch.query_y.to(args.device),
            )
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
    for name, default in dict(
        steps=20000,
        width=192,
        hidden=768,
        layers=6,
        heads=6,
        support_size=128,
        query_size=32,
        max_features=12,
        num_groups=5,
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
