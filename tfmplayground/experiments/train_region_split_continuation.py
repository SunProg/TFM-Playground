"""Continue task-posterior training on region-split episodes.

The generator runs on CPU while the adapter/backbone run on the requested
accelerator.  Validation is printed and written every ``--report-every``
steps, making the script suitable for a Slurm batch job.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.region_split_episodes import RegionSplitConfig, generate_region_split_episodes
from tfmplayground.experiments.train_task_posterior_adapter import (
    TaskPosteriorTrainingConfig,
    choose_ordinary_episode,
    contrastive_episode_objective,
)
from tfmplayground.models.task_posterior_adapter import load_task_posterior_checkpoint, save_task_posterior_checkpoint
from tfmplayground.utils import set_randomness_seed


def _move_batch(batch, device: torch.device):
    updates = {
        field.name: getattr(batch, field.name).to(device)
        for field in fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    }
    return replace(batch, **updates)


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


def _identify(prediction, candidates: torch.Tensor) -> torch.Tensor:
    log_probabilities = prediction.marginal_probabilities().clamp_min(1e-12).log()
    scores = [
        log_probabilities.gather(-1, candidates[:, index, :, None].long()).squeeze(-1).sum(-1)
        for index in range(candidates.shape[1])
    ]
    return torch.stack(scores, dim=-1).argmax(-1)


@torch.no_grad()
def _evaluate(model, validation_batches, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "prior_task_id": 0.0,
        "updated_task_id": 0.0,
        "prior_query_accuracy": 0.0,
        "updated_query_accuracy": 0.0,
        "updated_effective_particles": 0.0,
        "updated_ambiguity": 0.0,
    }
    count = 0
    for raw_batch in validation_batches:
        batch = _move_batch(raw_batch, device)
        prior = model(batch.initial_support_x, batch.initial_support_y.long(), batch.query_x, class_count=2)
        updated = model(
            torch.cat((batch.initial_support_x, batch.stream_x), dim=1),
            torch.cat((batch.initial_support_y, batch.stream_y), dim=1).long(),
            batch.query_x,
            class_count=2,
        )
        prior_probabilities = prior.marginal_probabilities()
        updated_probabilities = updated.marginal_probabilities()
        prior_task = _identify(prior, batch.candidate_query_y)
        updated_task = _identify(updated, batch.candidate_query_y)
        for index in range(batch.initial_support_x.shape[0]):
            totals["prior_task_id"] += float((prior_task[index] == batch.candidate_task[index]).cpu())
            totals["updated_task_id"] += float((updated_task[index] == batch.candidate_task[index]).cpu())
            totals["prior_query_accuracy"] += float(
                (prior_probabilities[index].argmax(-1) == batch.query_y[index].long()).float().mean().cpu()
            )
            totals["updated_query_accuracy"] += float(
                (updated_probabilities[index].argmax(-1) == batch.query_y[index].long()).float().mean().cpu()
            )
            totals["updated_effective_particles"] += float(updated.effective_particle_count()[index].cpu())
            totals["updated_ambiguity"] += float(updated.ambiguity()[index].cpu())
            count += 1
    model.train()
    return {key: value / count for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--start-step", type=int, default=400)
    parser.add_argument("--end-step", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=64)
    parser.add_argument("--ordinary-episode-fraction", type=float, default=0.35)
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.end_step <= args.start_step:
        raise ValueError("end-step must be greater than start-step.")
    if args.report_every < 1 or args.batch_size < 1:
        raise ValueError("report-every and batch-size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    episode_config = RegionSplitConfig(
        alternative_mode="region_flip",
        source="tabicl",
        min_features=5,
        max_features=5,
        device="cpu",
        compute_structural_latents=False,
    )
    training_config = TaskPosteriorTrainingConfig(
        structural_weight=0.0,
        ordinary_episode_fraction=args.ordinary_episode_fraction,
    )
    (output / "config.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
                "start_step": args.start_step,
                "end_step": args.end_step,
                "device": str(device),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "report_every": args.report_every,
                "validation_episodes": args.validation_episodes,
                "episode_config": asdict(episode_config),
                "training_config": asdict(training_config),
                "optimizer_state": "restarted; source checkpoints store model weights only",
            },
            indent=2,
        )
        + "\n"
    )

    # Fixed validation batches keep all report points comparable.
    set_randomness_seed(args.seed + 9_000_000)
    validation_batches = [
        generate_region_split_episodes(
            episode_config,
            np.random.default_rng(args.seed + 9_100_000 + index),
            batch_size=args.batch_size,
        )
        for index in range(max(1, args.validation_episodes // args.batch_size))
    ]

    set_randomness_seed(args.seed)
    model, _ = load_task_posterior_checkpoint(args.source_checkpoint, map_location="cpu")
    model.to(device)
    model.train()
    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-2)
    rng = np.random.default_rng(args.seed + args.start_step + 700_000)
    rows: list[dict[str, float | int | bool]] = []
    started = time.perf_counter()

    for step in range(args.start_step + 1, args.end_step + 1):
        ordinary = choose_ordinary_episode(
            step=step,
            seed=args.seed,
            ordinary_episode_fraction=training_config.ordinary_episode_fraction,
        )
        batch = _move_batch(
            generate_region_split_episodes(episode_config, rng, batch_size=args.batch_size), device
        )
        if ordinary:
            batch = _ordinary_version(batch)
        optimizer.zero_grad(set_to_none=True)
        objective = contrastive_episode_objective(model, batch, training_config)
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % args.report_every != 0 and step != args.end_step:
            continue
        metrics = _evaluate(model, validation_batches, device)
        row = {
            "step": step,
            "loss": float(objective.total.detach().cpu()),
            "ordinary": ordinary,
            **metrics,
        }
        row["stream_identification_gain"] = row["updated_task_id"] - row["prior_task_id"]
        rows.append(row)
        print(
            " ".join(
                [
                    f"step={step}",
                    f"loss={row['loss']:.4f}",
                    f"updated_id={row['updated_task_id']:.4f}",
                    f"gain={row['stream_identification_gain']:.4f}",
                    f"query_acc={row['updated_query_accuracy']:.4f}",
                ]
            ),
            flush=True,
        )

    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(output / "trajectory.csv", index=False)
    final_checkpoint = output / f"adapter_step_{args.end_step}.pth"
    save_task_posterior_checkpoint(
        final_checkpoint,
        model.eval(),
        training_config={
            **asdict(training_config),
            "learning_rate": args.learning_rate,
            "start_step": args.start_step,
            "end_step": args.end_step,
            "training_seed": args.seed,
        },
        lineage={"source_checkpoint": str(Path(args.source_checkpoint).resolve())},
        data_provenance={
            "episode_family": "region_split",
            "alternative_mode": "region_flip",
            "synthetic_prior_families": ["TabICL mix_scm"],
        },
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "trajectory": rows,
                "final_checkpoint": str(final_checkpoint.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    return output.resolve()


def main() -> int:
    print(run(build_parser().parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
