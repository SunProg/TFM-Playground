"""Train AttentionSlotRouterFrozenTabPFN on the same SCM/v3 families and
budget as attention_slot_router_scale.py, swapping the jointly-trained
NanoTabPFNModel backbone for a frozen real TabPFN model (see
tfmplayground/models/tabpfn_frozen.py).

Cost note: every training step now performs batch_size * accumulate separate
real-TabPFN .fit()+.predict_proba() calls (one per episode), each running
TabPFN's own preprocessing pipeline and a full transformer forward pass
through every block, even though only one block's intermediate output is
kept. This is substantially slower per step than the NanoTabPFNModel
backbone. Start with a smaller --batch-size/--accumulate than this script's
NanoTabPFN counterpart to measure wall-clock cost before scaling up.

No arm/weight axis, same as attention_slot_router_scale.py: the only axes
are scope and seed (plus, here, --tabpfn-layer-index for backbone-layer
ablations).
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

from tfmplayground.experiments.multiregime_v3 import FAMILIES, OriginalPrior, V3Config
from tfmplayground.experiments.multiregime_v3 import sample_episode as sample_family_episode
from tfmplayground.experiments.pretrain_multiregime_v3 import PilotConfig, task_config
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.attention_slot_router import AttentionSlotRouterFrozenTabPFN, attention_slot_router_loss


@dataclass
class DeviceEpisode:
    """Tensors already on-device; matches the .latent_inputs()/.query_y interface
    evaluate() needs, uniformly for either prior (V3Episode has no .to() of its
    own, unlike RegimeEpisode)."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    def latent_inputs(self):
        return self.support_x, self.support_y, self.query_x


def to_device_episode(episode, device):
    return DeviceEpisode(
        episode.support_x.to(device),
        episode.support_y.to(device),
        episode.query_x.to(device),
        episode.query_y.to(device),
    )


def _without_constant_support_columns(sample_fn, seed, *, max_attempts=32):
    """Retry with a perturbed seed until no support column is exactly constant.

    Real TabPFN's preprocessing (RemoveConstantFeaturesStep, unconditionally
    part of its pipeline -- not something PREPROCESS_TRANSFORMS can turn off)
    drops a column outright if every row has the exact same value, checked by
    exact equality, not a variance threshold. Fixing min_features ==
    max_features (see build_prior below) only rules out the *padding*-induced
    version of this; a binary/categorical root cause in the underlying SCM
    can still, by chance, land on the same value for every row of a finite
    episode -- expected, low-probability data, not a bug in the prior. Left
    unhandled, that episode gets a different post-preprocessing column count
    than the rest of its batch and crashes
    FrozenTabPFNBackbone.encode_table's consistency guard (observed on the
    very first training step of the first real run: episode 1 got 9 columns,
    episode 0 got 8 after one of its features was dropped as constant).
    """
    for attempt in range(max_attempts):
        episode = sample_fn(seed + attempt * 7_919)  # arbitrary prime stride, distinct from OriginalPrior's own retry stride
        support = episode.support_x
        if not (support[:, :1, :] == support).all(dim=1).any():
            return episode
    raise RuntimeError(f"Could not sample an episode without a constant support column after {max_attempts} attempts (seed={seed}).")


def build_prior(args):
    """Returns (sample_fn(seed) -> episode, prior_config_dict, query_size, original_or_none).

    original_or_none is the OriginalPrior instance backing v3_original --
    family_bank() needs it directly to build the non-"original" families'
    held-out banks. None for scm, which has no family concept.
    """
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
        return (lambda seed: sample_scm_regime_episode(prior, seed=seed)), asdict(prior), prior.query_size, None
    if args.prior == "v3_original":
        # num_groups=5 matches PilotConfig.generator() in pretrain_multiregime_v3.py
        # (its own override of V3Config's default of 8) -- "their setting", not an
        # arbitrary choice of ours.
        #
        # min_features == max_features (fixed, not the 2-12 range
        # attention_slot_router_scale.py uses under NanoTabPFNModel): real
        # TabPFN's preprocessing drops constant (all-zero) columns per
        # episode, and _model_x zero-pads each episode's actual feature
        # count up to max_features -- with a real range, episodes with
        # fewer actual features than another episode in the same batch would
        # have a different post-preprocessing column count, tripping
        # FrozenTabPFNBackbone.encode_table's column-count-consistency
        # guard. Fixing min==max only rules out this *padding*-induced
        # version of the problem -- a real feature can still, by chance,
        # come out exactly constant (see _without_constant_support_columns).
        # NanoTabPFNModel does no such preprocessing at all, which is why
        # this was never an issue for attention_slot_router_scale.py.
        v3_config = V3Config(
            support_size=args.support_size,
            query_size=args.query_size,
            min_features=args.features,
            max_features=args.features,
            num_groups=5,
        )
        original = OriginalPrior(v3_config)
        return (
            (lambda seed: _without_constant_support_columns(lambda s: original.sample(seed=s), seed)),
            asdict(v3_config),
            v3_config.query_size,
            original,
        )
    if args.prior == "mix_scm":
        # Literal, unmodified mix_scm episodes (TabICL's own prior: randomly
        # mlp_scm or tree_scm per episode) -- pad_groups=False drops
        # v3_original's nuisance one-hot group-code block, which has no
        # purpose here since nothing else is ever interleaved with mix_scm
        # in this branch. Same fixed min_features==max_features requirement,
        # and same residual constant-support-column risk, as v3_original above.
        v3_config = V3Config(
            support_size=args.support_size,
            query_size=args.query_size,
            min_features=args.features,
            max_features=args.features,
        )
        original = OriginalPrior(v3_config)
        return (
            (lambda seed: _without_constant_support_columns(lambda s: original.sample(seed=s, pad_groups=False), seed)),
            asdict(v3_config),
            v3_config.query_size,
            None,
        )
    raise ValueError(f"Unknown prior: {args.prior!r}")


def family_bank(*, seed_base, episodes, offset, args, original):
    """Mirrors evaluation_bank()'s exact seed formula and per-family branching
    (pretrain_multiregime_v3.py:252-262), parameterized by `offset` so
    validation and test sets can use disjoint seed ranges. The
    ``% (2**32 - 1)`` modulo applies only to the "original" branch
    (TabICLPriorDataLoader's legacy numpy uint32 seed requirement) -- not
    uniformly, matching the real implementation exactly rather than a
    cleaned-up guess at it.
    """
    pilot_config = PilotConfig(
        seed=seed_base,
        support_size=args.support_size,
        query_size=args.query_size,
        max_features=12,
        num_groups=5,
        validation_episodes=episodes,
    )
    bank = {}
    for family_index, family in enumerate(FAMILIES):
        bank[family] = []
        for index in range(episodes):
            seed = offset + seed_base * 10_000 + family_index * 1000 + index
            if family == "original":
                episode = original.sample(seed % (2**32 - 1))
            else:
                generator, active_groups = task_config(pilot_config, seed)
                episode = sample_family_episode(generator, family=family, seed=seed, active_groups=active_groups)
            bank[family].append(to_device_episode(episode, args.device))
    return bank


def mixture_probability(mode, step, steps):
    """Verbatim port of pretrain_multiregime_v3.py's own function: how much of
    training at this step should be non-original families, by mode."""
    if mode == "original":
        return 0.0
    if mode == "fixed":
        return 0.5
    if mode != "curriculum":
        raise ValueError(f"Unknown mode: {mode!r}")
    fraction = step / steps
    return 0.5 * float(np.clip((fraction - 0.1) / 0.3, 0, 1))


def training_family_episode(seed, *, mode, step, steps, original, task_config_source):
    """Mirrors training_episode()'s mixture logic (pretrain_multiregime_v3.py)
    exactly, given a seed already computed by this script's own per-
    (step, micro, within_batch) formula. mode="original" always takes the
    `original.sample(seed, pad_groups=False)` branch (mixture_probability
    returns 0, and `rng.random() >= 0.0` is always true) -- unpadded, since
    nothing else is ever interleaved with it in that mode, matching
    OriginalPrior.sample's own pad_groups docstring. "fixed"/"curriculum"
    keep the nuisance block (pad_groups=True) because its absence would
    itself leak that an episode is "original" once other families are mixed
    in.
    """
    rng = np.random.default_rng(seed)
    if rng.random() >= mixture_probability(mode, step, steps):
        return original.sample(seed, pad_groups=mode != "original")
    family = str(rng.choice(FAMILIES[1:], p=(0.2, 0.3, 0.2, 0.3)))
    generator, active_groups = task_config(task_config_source, seed)
    return sample_family_episode(generator, family=family, seed=seed, active_groups=active_groups)


def configurations():
    # seed=2402 matches PilotConfig()'s own default, same as
    # attention_slot_router_scale.py's configurations() -- comparable to that
    # script's cells against the identical held-out episode bank.
    return [
        dict(seed=2402, scope=scope, mode=mode)
        for scope in ("data", "cell_and_data", "cell")
        for mode in ("original", "fixed", "curriculum")
    ]


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    finite = [g.detach().float().square().sum() for g in gradients if g is not None]
    if not finite:
        return 0.0
    return float(torch.stack(finite).sum().sqrt())


@torch.no_grad()
def evaluate(model, episodes):
    model.eval()
    measurements = []
    for episode in episodes:
        prediction, responsibilities = model(*episode.latent_inputs())
        loss = attention_slot_router_loss(prediction, episode.query_y)
        row = {k: float(v) for k, v in model.last.items()}
        row.update(
            nll=float(loss),
            accuracy=float((prediction.argmax(-1) == episode.query_y).float().mean()),
        )
        measurements.append(row)
    return {k: float(sum(m[k] for m in measurements) / len(measurements)) for k in measurements[0]}, measurements


@torch.no_grad()
def evaluate_families(model, bank):
    """Family-keyed counterpart of evaluate(): one (aggregate, per_episode)
    pair per family (bank is {family: [DeviceEpisode, ...]}), plus an
    unweighted mean-of-per-family-means as "overall" -- mirrors
    pretrain_multiregime_v3.py's own evaluate(), which keys its summary by
    family rather than pooling episodes across families (so a family with
    more/fewer episodes can't silently dominate the aggregate).
    """
    summary, per_episode = {}, {}
    for family, episodes in bank.items():
        aggregate, measurements = evaluate(model, episodes)
        summary[family] = aggregate
        per_episode[family] = measurements
    summary["overall"] = {
        key: float(sum(summary[family][key] for family in summary if family != "overall") / len(bank))
        for key in next(iter(summary.values()))
    }
    return summary, per_episode


def run(args):
    torch.set_num_threads(1)
    cell = configurations()[args.index]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA")
    sample_episode, prior_config, prior_query_size, original = build_prior(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        cell=cell,
        architecture=dict(
            hidden=args.hidden,
            num_slots=4,
            tabpfn_layer_index=args.tabpfn_layer_index,
            tabpfn_model_path=args.tabpfn_model_path,
        ),
        prior=args.prior,
        prior_config=prior_config,
        training={k: v for k, v in vars(args).items() if k not in ("output", "index")},
        model="AttentionSlotRouterFrozenTabPFN: query -> slot -> slot_value(real support labels) "
        "retrieval over a frozen real-TabPFN backbone, no MLP",
    )
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains a different configuration")
    config_path.write_text(json.dumps(config, indent=2))
    torch.manual_seed(cell["seed"])
    model = AttentionSlotRouterFrozenTabPFN(
        scope=cell["scope"],
        hidden=args.hidden,
        num_slots=4,
        layer_index=args.tabpfn_layer_index,
        tabpfn_device=args.tabpfn_device,
        tabpfn_model_path=args.tabpfn_model_path,
    )
    model.to(args.device)
    # model.parameters() already excludes every real-TabPFN weight:
    # FrozenTabPFNBackbone is a plain object, not an nn.Module submodule.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    latest = output / "latest.pth"
    start, best = 0, math.inf
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best_nll"]
    # Validation query size always matches training's (an override per prior
    # is easy for SCMRegimeConfig via dataclasses.replace, but OriginalPrior
    # bakes support/query size into its TabICLPriorDataLoader at construction
    # time -- not worth a second, differently-configured instance for a pilot).
    test_bank = None
    if args.prior == "v3_original":
        # All 5 FAMILIES, not just "original" -- evaluation_bank()'s own
        # definition of "validation" in pretrain_multiregime_v3.py.
        validation_bank = family_bank(
            seed_base=cell["seed"], episodes=args.validation_episodes, offset=10_000_000_000, args=args, original=original
        )
        # Disjoint seed range, never touched during training or for
        # checkpoint selection; evaluated once at the end against whichever
        # checkpoint validation actually picked.
        test_bank = family_bank(
            seed_base=cell["seed"], episodes=args.validation_episodes, offset=20_000_000_000, args=args, original=original
        )
        # task_config() only reads num_groups/generator() off this, not seed
        # -- shared across every mixture draw in the training loop below.
        training_task_config = PilotConfig(
            seed=cell["seed"], support_size=args.support_size, query_size=args.query_size, max_features=12, num_groups=5
        )
    else:
        validation_episode_start = getattr(args, "validation_episode_start", 900000)
        validation_seeds = [validation_episode_start + i for i in range(args.validation_episodes)]
        heldout = [to_device_episode(sample_episode(seed), args.device) for seed in validation_seeds]
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
            if args.prior == "v3_original":
                episodes = [
                    training_family_episode(
                        first + j,
                        mode=cell["mode"],
                        step=step,
                        steps=args.steps,
                        original=original,
                        task_config_source=training_task_config,
                    )
                    for j in range(args.batch_size)
                ]
            else:
                episodes = [sample_episode(first + j) for j in range(args.batch_size)]
            support_x = torch.cat([e.support_x for e in episodes]).to(args.device)
            support_y = torch.cat([e.support_y for e in episodes]).to(args.device)
            query_x = torch.cat([e.query_x for e in episodes]).to(args.device)
            query_y = torch.cat([e.query_y for e in episodes]).to(args.device)
            torch.manual_seed(first)
            prediction, responsibilities = model(support_x, support_y, query_x)
            query_loss = attention_slot_router_loss(prediction, query_y)
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
            if args.prior == "v3_original":
                # Checkpoint selection stays on "original"'s own NLL, matching
                # attention_slot_router_scale.py's reasoning: training only
                # ever samples original-family episodes here too.
                validation, per_episode = evaluate_families(model, validation_bank)
                selection_nll = validation["original"]["nll"]
            else:
                validation, per_episode = evaluate(model, heldout)
                selection_nll = validation["nll"]
            metrics.update(validation=validation)
            improved = selection_nll < best
            best = min(best, selection_nll)
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
    if test_bank is not None:
        # Evaluate the checkpoint validation actually selected, not the final
        # (possibly worse, since training keeps moving after the best point)
        # weights still sitting in `model`.
        best_state = torch.load(output / "best.pth", map_location=args.device, weights_only=False)
        model.load_state_dict(best_state["model"])
        test_summary, _ = evaluate_families(model, test_bank)
        (output / "test_metrics.json").write_text(json.dumps(test_summary, indent=2))
    (output / "complete.json").write_text(json.dumps({"steps": args.steps, "best_validation_nll": best}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=range(len(configurations())), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--prior", choices=("scm", "v3_original", "mix_scm"), default="scm")
    parser.add_argument("--tabpfn-device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--tabpfn-model-path", default=None)
    parser.add_argument("--tabpfn-layer-index", type=int, default=-1)
    # Fixed (not a range): real TabPFN's preprocessing drops constant columns
    # per episode, so v3_original/mix_scm need every episode to have exactly
    # this many genuinely-varying feature columns -- see build_prior().
    parser.add_argument("--features", type=int, default=8)
    for name, default in dict(
        steps=20000,
        hidden=768,
        # Smaller than attention_slot_router_scale.py's 8/4 default: every
        # step here pays for batch_size * accumulate real-TabPFN fit+predict
        # calls, not a single cheap NanoTabPFNModel forward pass.
        support_size=128,
        query_size=32,
        batch_size=4,
        accumulate=2,
        warmup_steps=2000,
        validation_interval=500,
        validation_episodes=32,
    ).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--validation-query-size", type=int, default=None)
    parser.add_argument("--validation-episode-start", type=int, default=900000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    run(parser.parse_args())
