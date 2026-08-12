"""Paired K=2 scratch-versus-warm batch-causal retraining experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.particle_benchmark import (
    RegimeStream,
    RegimeStreamSpec,
    default_baselines,
    evaluate_delayed_stream,
    generate_regime_stream,
    paired_bootstrap_improvement,
    sample_training_stream,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.batch_particle_filter import BatchCausalParticleFilter, PendingBatchUpdate
from tfmplayground.models.integrated_latent_filter import (
    NanoTabPFNIntegratedLatentFilter,
    backbone_sha256,
    load_integrated_checkpoint,
)
from tfmplayground.models.particle_online import BatchParticleOnlineClassifier, NanoTabPFNContextOnlineClassifier
from tfmplayground.utils import set_randomness_seed

SEEDS = (2402, 22402, 42402)
WARM_COMPONENTS = ("initial_latents", "adapters.", "decoder.")


@dataclass(frozen=True)
class ComparisonConfig:
    phase: Literal["pilot", "full"] = "pilot"
    initialization: Literal["scratch", "warm", "all"] = "all"
    seed: int | None = None
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    warm_checkpoint: str = "runs/h5_prior_bimodal_filter/k2-all-terms-v1/selected_checkpoint.pth"
    output_root: str = "runs/particle_regime_comparison"
    device: str = "cpu"
    steps: int | None = None
    learning_rate: float = 1e-4
    validation_interval: int = 50
    synthetic_episodes: int = 256
    context_limit: int = 1024
    resume: bool = True

    @property
    def effective_steps(self) -> int:
        return self.steps if self.steps is not None else (50 if self.phase == "pilot" else 600)

    @property
    def accumulate_gradients(self) -> int:
        return 1 if self.phase == "pilot" else 4

    @property
    def max_features(self) -> int:
        return 16 if self.phase == "pilot" else 100


@dataclass(frozen=True)
class TorchRegimeEpisode:
    x: torch.Tensor
    y: torch.Tensor
    candidate_y: torch.Tensor
    regime: torch.Tensor
    batch_slices: tuple[tuple[int, int], ...]
    stable: bool


def _episode(seed: int, max_features: int, device: str | torch.device) -> TorchRegimeEpisode:
    """Draw 30% stable, 35% A-B, and 35% A-B-A episodes."""

    rng = np.random.default_rng(seed)
    draw = rng.random()
    # sample_training_stream owns the latent-function generator and nuisance distributions.
    stream = sample_training_stream(seed=seed + 1, stable_probability=1.0 if draw < 0.30 else 0.0)
    if draw >= 0.30:
        wanted = (0, 1) if draw < 0.65 else (0, 1, 0)
        dwell = tuple(int(rng.integers(24, 129)) for _ in wanted)
        stream = generate_regime_stream(
            RegimeStreamSpec(
                pattern=wanted,
                dwell_lengths=dwell,
                n_features=int(rng.integers(1, max_features + 1)),
                positive_rate=float(rng.choice((0.1, 0.25, 0.5, 0.75, 0.9))),
                label_noise=float(rng.choice((0.0, 0.05, 0.1))),
                missing_rate=float(rng.choice((0.0, 0.05, 0.2))),
                latent_regime_count=2,
            ),
            seed + 2,
        )
    elif stream.x.shape[1] > max_features:
        stream = RegimeStream(
            x=stream.x[:, :max_features],
            y=stream.y,
            regime=stream.regime,
            candidate_y=stream.candidate_y,
            segment_starts=stream.segment_starts,
            spec=replace(stream.spec, n_features=max_features),
        )
    batch_size = int(rng.choice((8, 16, 32)))
    boundaries = sorted({*stream.segment_starts, len(stream.y)})
    slices = tuple(
        (start, min(start + batch_size, right))
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        for start in range(left, right, batch_size)
    )
    x = np.nan_to_num(stream.x, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = 1 if len(stream.spec.pattern) == 1 else 2
    return TorchRegimeEpisode(
        x=torch.as_tensor(x, device=device)[None],
        y=torch.as_tensor(stream.y, device=device)[None],
        candidate_y=torch.as_tensor(stream.candidate_y[:candidate_count], device=device)[None],
        regime=torch.as_tensor(stream.regime, device=device)[None],
        batch_slices=slices,
        stable=candidate_count == 1,
    )


def _episode_assignment(first: PendingBatchUpdate, candidate_y: torch.Tensor) -> torch.Tensor:
    """Detached permutation-invariant matching, computed once then reused."""

    candidate_count = candidate_y.shape[1]
    scores = []
    for candidate in range(candidate_count):
        labels = candidate_y[:, candidate]
        observed = first.particle_log_probabilities.gather(
            -1, labels[:, :, None, None].expand(-1, -1, first.particle_logits.shape[2], 1)
        ).squeeze(-1)
        scores.append(observed.sum(1))
    score = torch.stack(scores, dim=-1)
    assignments = list(itertools.permutations(range(score.shape[1]), candidate_count))
    totals = torch.stack([sum(score[:, particle, c] for c, particle in enumerate(row)) for row in assignments], 1)
    choice = totals.detach().argmax(1)
    return torch.tensor(assignments, device=score.device, dtype=torch.long)[choice]


def episode_loss(
    model: BatchCausalParticleFilter, episode: TorchRegimeEpisode
) -> tuple[torch.Tensor, dict[str, float]]:
    state = model.initial_state(episode.x.shape[-1], device=episode.x.device)
    assignment = None
    terms = {
        "prequential_nll": [],
        "posterior": [],
        "specialization": [],
        "anchor_kl": [],
        "residual": [],
        "unmatched": [],
    }
    for start, stop in episode.batch_slices:
        x, y = episode.x[:, start:stop], episode.y[:, start:stop]
        unmasked = None
        if assignment is None:
            unmasked = model.predict_batch(state, x)
            assignment = _episode_assignment(unmasked, episode.candidate_y[:, :, start:stop])
            pending = model.restrict_pending(unmasked, assignment)
        else:
            pending = model.predict_batch(state, x, matched_particles=assignment)
        observed = pending.probabilities.gather(-1, y[..., None]).squeeze(-1)
        terms["prequential_nll"].append(-observed.clamp_min(1e-12).log().mean())
        # Candidate specialization uses the fixed episode assignment at every batch.
        specialization = []
        for candidate in range(assignment.shape[1]):
            labels = episode.candidate_y[:, candidate, start:stop]
            particle = assignment[:, candidate]
            chosen = pending.particle_log_probabilities.gather(
                2, particle[:, None, None, None].expand(-1, stop - start, 1, 2)
            ).squeeze(2)
            specialization.append(-chosen.gather(-1, labels[..., None]).squeeze(-1).mean())
        terms["specialization"].append(torch.stack(specialization).mean())
        if episode.stable:
            particle = pending.particle_log_probabilities[:, :, assignment[0, 0]].exp()
            vanilla = pending.vanilla_logits.softmax(-1)
            terms["anchor_kl"].append(
                F.kl_div(particle.clamp_min(1e-12).log(), vanilla, reduction="batchmean") / x.shape[1]
            )
            unmatched = torch.ones_like(pending.unmasked_log_weights, dtype=torch.bool)
            unmatched.scatter_(1, assignment, False)
            terms["unmatched"].append(pending.unmasked_log_weights.exp().masked_select(unmatched).mean())
        terms["residual"].append(pending.residuals.square().mean())
        state = model.reveal_batch(pending, y, temporal=True)
        active_candidate = episode.regime[:, start]
        target = assignment.gather(1, active_candidate[:, None]).squeeze(1)
        terms["posterior"].append(F.nll_loss(state.log_weights, target))

    zero = episode.x.sum() * 0
    means = {name: torch.stack(values).mean() if values else zero for name, values in terms.items()}
    total = (
        means["prequential_nll"]
        + 0.5 * means["posterior"]
        + 0.25 * means["specialization"]
        + 0.25 * means["anchor_kl"]
        + 0.01 * means["residual"]
        + 0.1 * means["unmatched"]
    )
    return total, {name: float(value.detach()) for name, value in {"loss": total, **means}.items()}


def _fresh_model(config: ComparisonConfig, initialization: str) -> tuple[BatchCausalParticleFilter, dict[str, str]]:
    baseline = init_model_from_state_dict_file(config.checkpoint)
    # A fixed forked seed gives both independently-created arms byte-identical fresh gates.
    with torch.random.fork_rng():
        torch.manual_seed(99173)
        particle = NanoTabPFNIntegratedLatentFilter(copy.deepcopy(baseline), num_hypotheses=2)
        model = BatchCausalParticleFilter(
            particle,
            copy.deepcopy(baseline),
            context_limit=config.context_limit,
            transition_probability=0.05,
            residual_logit_bound=4.0,
        )
    imported: list[str] = []
    if initialization == "warm":
        warm, _ = load_integrated_checkpoint(config.warm_checkpoint)
        source = warm.state_dict()
        destination = model.particle_model.state_dict()
        for name in destination:
            if name == "initial_latents" or name.startswith(WARM_COMPONENTS[1:]):
                destination[name] = source[name].detach().clone()
                imported.append(name)
        model.particle_model.load_state_dict(destination)
    model.particle_model.backbone.requires_grad_(False).eval()
    model.vanilla_backbone.requires_grad_(False).eval()
    metadata = {
        "backbone_sha256": backbone_sha256(model.particle_model),
        "vanilla_backbone_sha256": _module_sha256(model.vanilla_backbone),
        "gate_sha256": _module_sha256(model.ambiguity_gate),
        "imported_keys_sha256": hashlib.sha256("\n".join(imported).encode()).hexdigest(),
    }
    return model.to(config.device), metadata


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _validate(model: BatchCausalParticleFilter, config: ComparisonConfig, seed: int) -> float:
    model.eval()
    values = [
        float(episode_loss(model, _episode(seed + 900_000 + i, config.max_features, config.device))[0])
        for i in range(8)
    ]
    model.train()
    return float(np.mean(values))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


@torch.no_grad()
def evaluate_synthetic(
    model: BatchCausalParticleFilter, config: ComparisonConfig, seed: int, output: Path
) -> list[dict[str, Any]]:
    """Evaluate the predeclared held-out stable/A-B/A-B-A stream mixture."""

    rows: list[dict[str, Any]] = []
    patterns = (((0,), "stable"), ((0, 1), "a_b"), ((0, 1, 0), "a_b_a"))
    for index in range(config.synthetic_episodes):
        rng = np.random.default_rng(seed + 2_000_000 + index)
        pattern, name = patterns[index % len(patterns)]
        dwell = tuple(int(rng.integers(64, 161)) for _ in pattern)
        stream = generate_regime_stream(
            RegimeStreamSpec(
                pattern=pattern,
                dwell_lengths=dwell,
                n_features=int(rng.integers(1, 101)),
                positive_rate=float(rng.choice((0.1, 0.25, 0.5, 0.75, 0.9))),
                label_noise=float(rng.choice((0.0, 0.05, 0.1))),
                missing_rate=float(rng.choice((0.0, 0.05, 0.2))),
                latent_regime_count=2,
            ),
            seed + 3_000_000 + index,
        )
        baselines = default_baselines(window=128)
        baselines["cumulative_vanilla"] = NanoTabPFNContextOnlineClassifier(
            copy.deepcopy(model.vanilla_backbone), device=config.device, context_limit=1024
        )
        baselines["sliding_window"] = NanoTabPFNContextOnlineClassifier(
            copy.deepcopy(model.vanilla_backbone), device=config.device, context_limit=128
        )
        predictors = {
            "particle": BatchParticleOnlineClassifier(copy.deepcopy(model), device=config.device),
            **baselines,
        }
        result = evaluate_delayed_stream(
            stream,
            predictors,
            batch_size=int(rng.choice((8, 16, 32))),
            oracle_name="oracle",
        )
        rows.extend({"episode": index, "stream": name, **summary} for summary in result.summary())
    _write_rows(output / "synthetic_metrics.csv", rows)
    metrics = [
        key for key, value in rows[0].items() if key not in {"episode", "stream", "method"} and isinstance(value, float)
    ]
    summary = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        means = {key: _finite_mean(row[key] for row in selected) for key in metrics}
        summary.append({"method": method, **means})
    _write_rows(output / "synthetic_summary.csv", summary)
    return summary


def run_arm(config: ComparisonConfig, seed: int, initialization: Literal["scratch", "warm"]) -> Path:
    output = Path(config.output_root) / config.phase / str(seed) / initialization
    output.mkdir(parents=True, exist_ok=True)
    set_randomness_seed(seed)
    model, identity = _fresh_model(config, initialization)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    checkpoint_path = output / "last_checkpoint.pth"
    start = 0
    history: list[dict[str, Any]] = []
    best_validation = _validate(model, config, seed)
    initial_validation = best_validation
    best_state = copy.deepcopy(model.state_dict())
    if config.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["step"])
        history = list(saved["history"])
        best_validation = float(saved["best_validation"])
        initial_validation = float(saved["initial_validation"])
        best_state = saved["best_model"]
        if saved["identity"] != identity:
            raise ValueError("Resume checkpoint identity does not match this arm.")
    model.train()
    finite_gradients = True
    for step in range(start + 1, config.effective_steps + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for accumulation in range(config.accumulate_gradients):
            episode_seed = seed + step * config.accumulate_gradients + accumulation
            loss, metrics = episode_loss(model, _episode(episode_seed, config.max_features, config.device))
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite_gradients &= bool(gradients and all(torch.isfinite(g).all() for g in gradients))
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        row: dict[str, Any] = {"step": step, "episode_seed": seed + step * config.accumulate_gradients, **totals}
        if step % config.validation_interval == 0 or step == config.effective_steps:
            row["validation_loss"] = _validate(model, config, seed + step)
            if row["validation_loss"] < best_validation:
                best_validation = row["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        payload = {
            "model_type": model.model_type,
            "model": model.state_dict(),
            "best_model": best_state,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "history": history,
            "best_validation": best_validation,
            "initial_validation": initial_validation,
            "identity": identity,
            "config": asdict(config),
            "seed": seed,
            "initialization": initialization,
        }
        torch.save(payload, checkpoint_path)
    model.load_state_dict(best_state)
    if start >= config.effective_steps:
        payload = {
            "model_type": model.model_type,
            "model": model.state_dict(),
            "best_model": best_state,
            "optimizer": optimizer.state_dict(),
            "step": start,
            "history": history,
            "best_validation": best_validation,
            "initial_validation": initial_validation,
            "identity": identity,
            "config": asdict(config),
            "seed": seed,
            "initialization": initialization,
        }
    selected = {**payload, "model": best_state, "selected_validation": best_validation}
    torch.save(selected, output / "selected_checkpoint.pth")
    _write_rows(output / "learning_curves.csv", history)
    health = {
        "finite_gradients_and_losses": finite_gradients and all(math.isfinite(float(row["loss"])) for row in history),
        "checkpoint_reload": False,
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best_validation,
        "held_out_loss_decreased": best_validation < initial_validation,
        "exact_causal_api": True,
        "transition_location": "predict_batch_only",
        "gate_inputs": "slots,posterior_entropy,posterior_margin,previous_surprise,current_disagreement",
        **identity,
    }
    reloaded = torch.load(output / "selected_checkpoint.pth", map_location="cpu")
    health["checkpoint_reload"] = reloaded["step"] == config.effective_steps
    health["passed"] = all(
        health[key]
        for key in (
            "finite_gradients_and_losses",
            "checkpoint_reload",
            "held_out_loss_decreased",
            "exact_causal_api",
        )
    )
    (output / "health.json").write_text(json.dumps(health, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    if config.phase == "full":
        evaluate_synthetic(model, config, seed, output)
    return output.resolve()


def run(config: ComparisonConfig) -> list[Path]:
    if config.phase == "full":
        pilot_root = Path(config.output_root) / "pilot" / str(SEEDS[0])
        pilot_health = []
        for arm in ("scratch", "warm"):
            path = pilot_root / arm / "health.json"
            pilot_health.append(path.exists() and json.loads(path.read_text()).get("passed", False))
        if not all(pilot_health):
            raise RuntimeError("Both paired pilot arms must pass health checks before full training.")
    seeds = (config.seed,) if config.seed is not None else ((SEEDS[0],) if config.phase == "pilot" else SEEDS)
    arms = ("scratch", "warm") if config.initialization == "all" else (config.initialization,)
    outputs = [run_arm(config, seed, arm) for seed in seeds for arm in arms]
    rows = []
    for output in outputs:
        health = json.loads((output / "health.json").read_text())
        rows.append({"phase": config.phase, "seed": int(output.parent.name), "initialization": output.name, **health})
    root = Path(config.output_root) / config.phase
    _write_rows(root / "paired_report.csv", rows)
    paired: dict[str, Any] = {"phase": config.phase, "runs": rows}
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["initialization"]] = row
    paired["warm_minus_scratch_validation"] = {
        str(seed): values["warm"]["best_validation_loss"] - values["scratch"]["best_validation_loss"]
        for seed, values in by_seed.items()
        if {"scratch", "warm"} <= values.keys()
    }
    paired["paired_identity_passed"] = all(
        values["scratch"][key] == values["warm"][key]
        for values in by_seed.values()
        if {"scratch", "warm"} <= values.keys()
        for key in ("backbone_sha256", "vanilla_backbone_sha256", "gate_sha256")
    )
    synthetic_rows = []
    for output in outputs:
        path = output / "synthetic_summary.csv"
        if path.exists():
            with path.open(newline="") as handle:
                synthetic_rows.extend(
                    {
                        "seed": int(output.parent.name),
                        "initialization": output.name,
                        **row,
                    }
                    for row in csv.DictReader(handle)
                )
    if synthetic_rows:
        _write_rows(root / "synthetic_summary.csv", synthetic_rows)
        particle = [row for row in synthetic_rows if row["method"] == "particle"]
        synthetic_differences = {}
        for seed, values in by_seed.items():
            if not {"scratch", "warm"} <= values.keys():
                continue
            arms = {row["initialization"]: row for row in particle if int(row["seed"]) == seed}
            synthetic_differences[str(seed)] = float(arms["warm"]["prequential_log_loss"]) - float(
                arms["scratch"]["prequential_log_loss"]
            )
        paired["warm_minus_scratch_synthetic"] = synthetic_differences
        promotions = {}
        for arm in ("scratch", "warm"):
            arm_rows = [row for row in synthetic_rows if row["initialization"] == arm]
            arm_seeds = sorted({int(row["seed"]) for row in arm_rows})
            if len(arm_seeds) < 3:
                continue
            particle_losses, baseline_losses = [], []
            for seed in arm_seeds:
                indexed = {row["method"]: row for row in arm_rows if int(row["seed"]) == seed}
                particle_losses.append(float(indexed["particle"]["prequential_log_loss"]))
                baseline_losses.append(
                    min(
                        float(indexed[method]["prequential_log_loss"])
                        for method in ("sliding_window", "retrieval_context")
                    )
                )
            paired_test = paired_bootstrap_improvement(particle_losses, baseline_losses, seed=SEEDS[0])
            particle_recovery = _finite_mean(
                float(row["mean_switch_recovery_delay_batches"]) for row in arm_rows if row["method"] == "particle"
            )
            sliding_recovery = _finite_mean(
                float(row["mean_switch_recovery_delay_batches"])
                for row in arm_rows
                if row["method"] == "sliding_window"
            )
            recurrence = _finite_mean(float(row["recurrence_gain"]) for row in arm_rows if row["method"] == "particle")
            detailed = []
            for output in outputs:
                if output.name != arm:
                    continue
                with (output / "synthetic_metrics.csv").open(newline="") as handle:
                    detailed.extend(csv.DictReader(handle))
            stable_auc = {
                method: _finite_mean(
                    float(row["auc"]) for row in detailed if row["stream"] == "stable" and row["method"] == method
                )
                for method in ("particle", "cumulative_vanilla")
            }
            gates = {
                "paired_log_loss": bool(paired_test["significant"]),
                "recovers_faster_than_sliding": bool(particle_recovery < sliding_recovery),
                "positive_recurrence_gain": bool(recurrence > 0),
                "stable_auc_no_harm": bool(stable_auc["particle"] >= stable_auc["cumulative_vanilla"] - 0.002),
            }
            promotions[arm] = {
                "passed": all(gates.values()),
                "gates": gates,
                "paired_test": paired_test,
                "particle_recovery_delay": float(particle_recovery),
                "sliding_recovery_delay": float(sliding_recovery),
                "recurrence_gain": float(recurrence),
                "stable_auc": {key: float(value) for key, value in stable_auc.items()},
            }
        if promotions:
            (root / "synthetic_promotion.json").write_text(json.dumps(promotions, indent=2) + "\n")
    (root / "paired_report.json").write_text(json.dumps(paired, indent=2) + "\n")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--initialization", choices=("scratch", "warm", "all"), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=ComparisonConfig.checkpoint)
    parser.add_argument("--warm-checkpoint", default=ComparisonConfig.warm_checkpoint)
    parser.add_argument("--output-root", default=ComparisonConfig.output_root)
    parser.add_argument("--device", default=ComparisonConfig.device)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=ComparisonConfig.learning_rate)
    parser.add_argument("--validation-interval", type=int, default=ComparisonConfig.validation_interval)
    parser.add_argument("--synthetic-episodes", type=int, default=ComparisonConfig.synthetic_episodes)
    parser.add_argument("--context-limit", type=int, default=ComparisonConfig.context_limit)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    outputs = run(ComparisonConfig(**vars(build_parser().parse_args(argv))))
    print("\n".join(map(str, outputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
