"""Train and evaluate the task-posterior adapter on held-out HDF5 prior episodes.

This is a bounded representation/no-harm experiment, not TabArena evidence.
It exists to decide whether an official TabArena-Lite run is warranted.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from tfmplayground.experiments.evaluate_task_posterior_tabarena import TABARENA_COMMIT
from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    generate_h5_prior_bimodal_episodes,
    generate_prior_bimodal_episodes,
)
from tfmplayground.experiments.structural_latents import StructuralLatentSchema, probe_r2
from tfmplayground.experiments.task_posterior_acceptance import (
    no_harm_gate,
    paired_bootstrap_gate,
)
from tfmplayground.experiments.train_task_posterior_adapter import (
    TaskPosteriorTrainingConfig,
    choose_ordinary_episode,
    contrastive_episode_objective,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    load_task_posterior_checkpoint,
    match_regimes_to_particles,
    save_task_posterior_checkpoint,
)
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class LocalEvaluationConfig:
    prior_dump: str = "300k_150x5_2.h5"
    backbone_checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str = "runs/task_posterior/local-evaluation"
    device: str = "cpu"
    seeds: tuple[int, ...] = (2402, 22402, 42402)
    steps: int = 300
    batch_size: int = 4
    learning_rate: float = 3e-4
    paired_evaluation_episodes: int = 256
    ordinary_evaluation_episodes: int = 256
    context_permutations: int = 4
    ordinary_context_rows: int = 64
    ordinary_query_rows: int = 16
    resume: bool = False
    # Structural-latent and slot-proposal arms.  The defaults reproduce the
    # original experiment exactly.
    episode_source: str = "h5"
    initial_support_count: int = 32
    stream_count: int = 32
    slot_mode: str = "deterministic"
    structural_probe: bool = False
    structural_weight: float = 0.0
    structural_detach: bool = False
    kl_weight: float = 0.0

    def validate(self) -> None:
        if self.episode_source not in {"h5", "tabicl"}:
            raise ValueError("episode_source must be 'h5' or 'tabicl'.")
        # Both sources supply structure.  'h5' loses only the generating family,
        # whose block is then marked unknown and dropped from the objective.
        if self.structural_weight > 0 and not self.structural_probe:
            raise ValueError("structural_weight is only meaningful with structural_probe enabled.")
        if self.structural_detach and not self.structural_probe:
            raise ValueError("structural_detach is only meaningful with structural_probe enabled.")
        if self.structural_probe and self.structural_weight <= 0:
            # Detaching stops the gradient at the slots, not at the probe.  With
            # zero weight the probe is never trained at all and its R2 describes
            # a random head, which is easy to mistake for "slots encode nothing".
            raise ValueError("structural_probe requires structural_weight > 0; the probe needs a loss to fit.")


def _episode_config(config: LocalEvaluationConfig) -> PriorBimodalConfig:
    return PriorBimodalConfig(
        initial_support_count=config.initial_support_count,
        stream_count=config.stream_count,
        query_count=4,
        max_features=5,
        max_pair_attempts=512,
        device=config.device,
        compute_structural_latents=config.structural_probe,
    )


def _schema(config: LocalEvaluationConfig) -> StructuralLatentSchema:
    return StructuralLatentSchema(max_features=_episode_config(config).max_features)


def _ordinary_version(batch):
    return replace(
        batch,
        candidate_task=None,
        candidate_support_y=None,
        candidate_stream_y=None,
        candidate_query_y=None,
        candidate_structural_z=None,
        structural_feature_mask=None,
    )


def _paired_batch(config, episode_config, rng, *, batch_size):
    last_error = None
    for _ in range(20):
        try:
            if config.episode_source == "tabicl":
                return generate_prior_bimodal_episodes(episode_config, rng, batch_size=batch_size)
            return generate_h5_prior_bimodal_episodes(config.prior_dump, episode_config, rng, batch_size=batch_size)
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("Paired generator failed after 20 advanced retries.") from last_error


def _permutations(rows: int, count: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    return [torch.as_tensor(rng.permutation(rows), device=device) for _ in range(count)]


@torch.no_grad()
def _ensemble_probabilities(model, context_x, context_y, query_x, *, permutations: int, seed: int):
    predictions = []
    vanilla_predictions = []
    for order in _permutations(context_x.shape[1], permutations, seed, context_x.device):
        prediction = model(
            context_x[:, order],
            context_y[:, order].long(),
            query_x,
            class_count=2,
            num_mem_chunks=1,
        )
        predictions.append(prediction.marginal_probabilities())
        vanilla_predictions.append(prediction.vanilla_logits.softmax(-1))
    stacked = torch.stack(predictions)
    vanilla = torch.stack(vanilla_predictions)
    return stacked.mean(0), vanilla.mean(0), float((stacked - stacked[0]).abs().max().cpu())


def _task_identification(probabilities: torch.Tensor, candidate_y: torch.Tensor) -> torch.Tensor:
    log_probabilities = probabilities.clamp_min(1e-12).log()
    candidates = candidate_y.long().permute(0, 2, 1)
    expanded = log_probabilities[:, :, None, :].expand(-1, -1, candidates.shape[-1], -1)
    likelihood = expanded.gather(-1, candidates[..., None]).squeeze(-1).sum(1)
    return likelihood.argmax(-1)


@torch.no_grad()
def _particle_diagnostics(model, batch) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """One canonical-order forward for collapse and structure diagnostics.

    The matched probe output uses the same label-space assignment the training
    loss uses, so the reported R2 answers "does the particle that reproduces
    this task's labels also describe its structure", not "does some particle
    happen to fit".  ``effective_particle_count`` sees weight collapse and
    ``slot_dispersion`` sees representation collapse, which weights cannot.
    """
    context_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
    context_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1).long()
    prediction = model(context_x, context_y, batch.query_x, class_count=2)
    matched = None
    if prediction.structural is not None and batch.candidate_structural_z is not None:
        assignment = match_regimes_to_particles(prediction, batch.candidate_query_y.long())
        latent_dim = prediction.structural.shape[-1]
        index = assignment.particle_for_regime[:, :, None].expand(-1, -1, latent_dim)
        matched = prediction.structural.gather(1, index)
    return matched, prediction.effective_particle_count(), prediction.slot_dispersion()


def _probe_r2(
    predictions: list[torch.Tensor], targets: list[torch.Tensor], schema: StructuralLatentSchema
) -> dict[str, float | None]:
    """In-loop telemetry only.

    The authoritative measurement is
    ``tfmplayground.experiments.structural_probe_analysis``, which fits probes of
    several capacities to convergence against controls and an upper bound.  The
    probe scored here shares the adapter's optimizer and step budget.
    """
    return probe_r2(predictions, targets, schema)


def _parameter_groups(model) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split trainable parameters into (adapter, probe).

    They must be clipped separately.  Clipping one global norm over both lets
    the probe's gradients scale the adapter's down, so a detached probe would
    still change the adapter it is supposed to be measuring.
    """
    adapter, probe = [], []
    for name, parameter in model.named_parameters():
        if name.startswith("backbone."):
            continue
        (probe if name.startswith("structural_probe.") else adapter).append(parameter)
    return adapter, probe


def _sample_ordinary_h5(config: LocalEvaluationConfig, rng: np.random.Generator):
    examples = []
    with h5py.File(config.prior_dump, "r") as handle:
        while len(examples) < config.ordinary_evaluation_episodes:
            index = int(rng.integers(0, handle["X"].shape[0]))
            x = np.asarray(handle["X"][index], dtype=np.float32)
            y = np.asarray(handle["y"][index], dtype=np.int64)
            required = config.ordinary_context_rows + config.ordinary_query_rows
            if required <= len(x) and np.isfinite(x[:required]).all() and len(np.unique(y[:required])) == 2:
                examples.append((x[:required], y[:required]))
    return examples


def _evaluate(model, config: LocalEvaluationConfig, seed: int) -> tuple[dict, list[dict]]:
    model.eval()
    paired_rng = np.random.default_rng(seed + 1_000_000)
    batch_size = config.batch_size
    paired_rows = []
    max_permutation_delta = 0.0
    structural_predictions: list[torch.Tensor] = []
    structural_targets: list[torch.Tensor] = []
    effective_counts: list[float] = []
    dispersions: list[float] = []
    paired_batches = (config.paired_evaluation_episodes + batch_size - 1) // batch_size
    for batch_index in range(paired_batches):
        current = min(batch_size, config.paired_evaluation_episodes - len(paired_rows))
        batch = _paired_batch(config, _episode_config(config), paired_rng, batch_size=current)
        matched, effective, dispersion = _particle_diagnostics(model, batch)
        if matched is not None:
            structural_predictions.append(matched.cpu())
            structural_targets.append(batch.candidate_structural_z.cpu())
        effective_counts.extend(effective.cpu().tolist())
        dispersions.extend(dispersion.cpu().tolist())
        context_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
        context_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1)
        adapter, vanilla, delta = _ensemble_probabilities(
            model,
            context_x,
            context_y,
            batch.query_x,
            permutations=config.context_permutations,
            seed=seed + batch_index,
        )
        max_permutation_delta = max(max_permutation_delta, delta)
        adapter_task = _task_identification(adapter, batch.candidate_query_y)
        vanilla_task = _task_identification(vanilla, batch.candidate_query_y)
        adapter_class = adapter.argmax(-1)
        vanilla_class = vanilla.argmax(-1)
        for row in range(current):
            paired_rows.append(
                {
                    "seed": seed,
                    "episode": len(paired_rows),
                    "true_task": int(batch.candidate_task[row].cpu()),
                    "adapter_task": int(adapter_task[row].cpu()),
                    "vanilla_task": int(vanilla_task[row].cpu()),
                    "adapter_query_accuracy": float((adapter_class[row] == batch.query_y[row]).float().mean().cpu()),
                    "vanilla_query_accuracy": float((vanilla_class[row] == batch.query_y[row]).float().mean().cpu()),
                }
            )

    ordinary_rows = []
    ordinary = _sample_ordinary_h5(config, np.random.default_rng(seed + 2_000_000))
    for episode, (x, y) in enumerate(ordinary):
        context_x = torch.as_tensor(x[: config.ordinary_context_rows], device=config.device).float()[None]
        context_y = torch.as_tensor(y[: config.ordinary_context_rows], device=config.device).long()[None]
        query_x = torch.as_tensor(
            x[config.ordinary_context_rows : config.ordinary_context_rows + config.ordinary_query_rows],
            device=config.device,
        ).float()[None]
        query_y = y[config.ordinary_context_rows : config.ordinary_context_rows + config.ordinary_query_rows]
        adapter, vanilla, delta = _ensemble_probabilities(
            model,
            context_x,
            context_y,
            query_x,
            permutations=config.context_permutations,
            seed=seed + 10_000 + episode,
        )
        max_permutation_delta = max(max_permutation_delta, delta)
        adapter_p = adapter[0, :, 1].cpu().numpy()
        vanilla_p = vanilla[0, :, 1].cpu().numpy()
        if len(np.unique(query_y)) == 2:
            ordinary_rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "adapter_auc": roc_auc_score(query_y, adapter_p),
                    "vanilla_auc": roc_auc_score(query_y, vanilla_p),
                    "adapter_accuracy": accuracy_score(query_y, adapter_p >= 0.5),
                    "vanilla_accuracy": accuracy_score(query_y, vanilla_p >= 0.5),
                }
            )

    paired = pd.DataFrame(paired_rows)
    ordinary_frame = pd.DataFrame(ordinary_rows)
    paired_gate = paired_bootstrap_gate(
        (paired.adapter_task == paired.true_task).astype(float),
        (paired.vanilla_task == paired.true_task).astype(float),
        bootstrap_samples=10_000,
        random_state=seed,
    )
    summary = {
        "seed": seed,
        "paired_episodes": len(paired),
        "adapter_task_identification": float((paired.adapter_task == paired.true_task).mean()),
        "vanilla_task_identification": float((paired.vanilla_task == paired.true_task).mean()),
        "adapter_query_accuracy": float(paired.adapter_query_accuracy.mean()),
        "vanilla_query_accuracy": float(paired.vanilla_query_accuracy.mean()),
        "representation_gate": asdict(paired_gate),
        "ordinary_auc": float(ordinary_frame.adapter_auc.mean()),
        "vanilla_ordinary_auc": float(ordinary_frame.vanilla_auc.mean()),
        "ordinary_auc_delta": float((ordinary_frame.adapter_auc - ordinary_frame.vanilla_auc).mean()),
        "no_harm_gate": no_harm_gate(ordinary_frame.adapter_auc, ordinary_frame.vanilla_auc),
        "max_context_permutation_probability_delta": max_permutation_delta,
        "mean_effective_particle_count": float(np.mean(effective_counts)) if effective_counts else float("nan"),
        "mean_slot_dispersion": float(np.mean(dispersions)) if dispersions else float("nan"),
        "structural_probe": _probe_r2(structural_predictions, structural_targets, _schema(config)),
    }
    detail = paired_rows + [dict(row, kind="ordinary") for row in ordinary_rows]
    return summary, detail


def run(config: LocalEvaluationConfig) -> Path:
    config.validate()
    output = Path(config.output_dir)
    if output.exists() and not config.resume:
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=config.resume)
    if not (output / "config.json").exists():
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    schema = _schema(config)
    training_config = TaskPosteriorTrainingConfig(
        structural_weight=config.structural_weight,
        structural_family_count=schema.family_count,
        kl_weight=config.kl_weight,
    )
    summaries = []
    details = []
    history_path = output / "training_history.csv"
    history = (
        pd.read_csv(history_path).to_dict(orient="records")
        if config.resume and history_path.exists() and history_path.stat().st_size > 1
        else []
    )
    started = time.perf_counter()
    for seed in config.seeds:
        set_randomness_seed(seed)
        checkpoint_path = output / f"adapter_seed_{seed}.pth"
        if config.resume and checkpoint_path.exists():
            model, _ = load_task_posterior_checkpoint(checkpoint_path)
            model.to(config.device).eval()
        else:
            backbone = init_model_from_state_dict_file(config.backbone_checkpoint)
            model = NanoTabPFNTaskPosteriorAdapter(
                backbone,
                particle_count=4,
                slot_mode=config.slot_mode,
                structural_latent_dim=schema.latent_dim if config.structural_probe else None,
                structural_family_count=schema.family_count,
                structural_detach=config.structural_detach,
            ).to(config.device)
            adapter_parameters, probe_parameters = _parameter_groups(model)
            optimizer = torch.optim.AdamW(
                adapter_parameters + probe_parameters, lr=config.learning_rate, weight_decay=1e-2
            )
            model.train()
            rng = np.random.default_rng(seed)
            for step in range(1, config.steps + 1):
                batch = _paired_batch(
                    config,
                    _episode_config(config),
                    rng,
                    batch_size=config.batch_size,
                )
                ordinary = choose_ordinary_episode(
                    step=step,
                    seed=seed,
                    ordinary_episode_fraction=training_config.ordinary_episode_fraction,
                )
                if ordinary:
                    batch = _ordinary_version(batch)
                optimizer.zero_grad(set_to_none=True)
                objective = contrastive_episode_objective(model, batch, training_config)
                objective.total.backward()
                torch.nn.utils.clip_grad_norm_(adapter_parameters, 1.0)
                if probe_parameters:
                    torch.nn.utils.clip_grad_norm_(probe_parameters, 1.0)
                optimizer.step()
                history.append(
                    {
                        "seed": seed,
                        "step": step,
                        "ordinary": ordinary,
                        "loss": float(objective.total.detach().cpu()),
                        "prior_mixture": float(objective.prior_only.mixture.detach().cpu()),
                        "updated_mixture": float(objective.updated.mixture.detach().cpu()),
                        "prior_structural": float(objective.prior_only.structural.detach().cpu()),
                        "prior_kl": float(objective.prior_only.kl.detach().cpu()),
                    }
                )
            save_task_posterior_checkpoint(
                checkpoint_path,
                model.eval(),
                training_config={**asdict(training_config), **asdict(config), "training_seed": seed},
                lineage={"backbone_checkpoint": str(Path(config.backbone_checkpoint).resolve())},
                data_provenance={
                    "real_meta_training_datasets": [],
                    "synthetic_prior_families": ["TabICL-HDF5 mix_scm/tree_scm/mlp_scm"],
                    "tabarena_overlap": [],
                    "tabarena_checked_against_commit": TABARENA_COMMIT,
                },
            )
            # Persist training before evaluation so a generator/evaluator
            # failure cannot discard the completed optimization trajectory.
            pd.DataFrame(history).to_csv(history_path, index=False)
        summary, seed_details = _evaluate(model, config, seed)
        summaries.append(summary)
        details.extend(seed_details)
        pd.DataFrame(history).to_csv(history_path, index=False)
        pd.DataFrame(summaries).to_json(output / "seed_summaries.json", orient="records", indent=2)

    frame = pd.DataFrame(summaries)
    aggregate = {
        "protocol": "held-out synthetic HDF5 diagnostic; not TabArena",
        "elapsed_seconds": time.perf_counter() - started,
        "training_seeds": list(config.seeds),
        "mean_adapter_task_identification": float(frame.adapter_task_identification.mean()),
        "mean_vanilla_task_identification": float(frame.vanilla_task_identification.mean()),
        "mean_task_identification_delta": float(
            (frame.adapter_task_identification - frame.vanilla_task_identification).mean()
        ),
        "all_representation_gates_pass": bool(all(row["representation_gate"]["passes"] for row in summaries)),
        "mean_ordinary_auc_delta": float(frame.ordinary_auc_delta.mean()),
        "all_no_harm_gates_pass": bool(frame.no_harm_gate.all()),
        "eligible_for_tabarena_lite": bool(
            all(row["representation_gate"]["passes"] for row in summaries) and frame.no_harm_gate.all()
        ),
        "max_context_permutation_probability_delta": float(frame.max_context_permutation_probability_delta.max()),
        "slot_mode": config.slot_mode,
        "mean_effective_particle_count": float(frame.mean_effective_particle_count.mean()),
        "mean_slot_dispersion": float(frame.mean_slot_dispersion.mean()),
        "structural_probe_by_seed": [row["structural_probe"] for row in summaries],
    }
    pd.DataFrame(details).to_csv(output / "evaluation_details.csv", index=False)
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dump", default="300k_150x5_2.h5")
    parser.add_argument("--backbone-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--paired-evaluation-episodes", type=int, default=256)
    parser.add_argument("--ordinary-evaluation-episodes", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--episode-source",
        choices=("h5", "tabicl"),
        default="h5",
        help="HDF5 dumps carry no structure; structural supervision needs 'tabicl'.",
    )
    parser.add_argument("--initial-support-count", type=int, default=32)
    parser.add_argument("--stream-count", type=int, default=32)
    parser.add_argument("--slot-mode", choices=("deterministic", "gaussian"), default="deterministic")
    parser.add_argument("--structural-probe", action="store_true")
    parser.add_argument(
        "--structural-detach",
        action="store_true",
        help="Stage A: measure whether existing slots encode structure without shaping them.",
    )
    parser.add_argument("--structural-weight", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LocalEvaluationConfig(
        prior_dump=args.prior_dump,
        backbone_checkpoint=args.backbone_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        paired_evaluation_episodes=args.paired_evaluation_episodes,
        ordinary_evaluation_episodes=args.ordinary_evaluation_episodes,
        resume=args.resume,
        episode_source=args.episode_source,
        initial_support_count=args.initial_support_count,
        stream_count=args.stream_count,
        slot_mode=args.slot_mode,
        structural_probe=args.structural_probe,
        structural_detach=args.structural_detach,
        structural_weight=args.structural_weight,
        kl_weight=args.kl_weight,
    )
    print(run(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
