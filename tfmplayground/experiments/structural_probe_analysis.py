"""Does a particle slot encode the structure of the task it was matched to?

The in-loop probe in ``run_task_posterior_local_evaluation`` shares the adapter's
optimizer and step budget and has no control and no upper bound, so its ``R2 <=
0`` cannot be attributed.  It is equally consistent with four different
explanations: the slots encode no structure, the probe is too small, the probe is
undertrained, or this structure is not recoverable from this backbone at all.

This module separates them.  Representations are extracted once from frozen
models and cached; probes of several capacities are then fit to convergence with
early stopping.  Four representations are compared:

``trained_slots``      the matched particle's slot - the measurement
``untrained_slots``    same architecture at random init - does *training* add it?
``observed_context``   pooled context states under the episode's real labels
``candidate_context``  pooled context states re-encoded under that candidate's
                       own labels - the upper bound

``observed_context`` is one vector per episode while targets differ per
candidate, so it cannot discriminate the two candidates and its R2 is
structurally capped.  ``candidate_context`` is the per-candidate ceiling and is
the honest comparison for ``trained_slots``.

Scores are additionally split by whether the candidate was the episode's true
task.  Structure recovered for the true candidate but not the false one means the
particles collapsed onto the observed labeling, which is a different and more
actionable failure than encoding nothing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    generate_h5_prior_bimodal_episodes,
)
from tfmplayground.experiments.structural_latents import StructuralLatentSchema, probe_r2
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    load_task_posterior_checkpoint,
    match_regimes_to_particles,
)
from tfmplayground.utils import get_default_device, set_randomness_seed

REPRESENTATIONS = ("trained_slots", "untrained_slots", "observed_context", "candidate_context")
CAPACITIES = ("linear", "mlp1", "mlp2")


@dataclass(frozen=True)
class StructuralProbeConfig:
    checkpoint: str
    prior_dump: str = "300k_150x5_2.h5"
    output_dir: str = "runs/structural_probe"
    device: str = "cpu"
    seed: int = 2402
    episodes: int = 4_000
    batch_size: int = 8
    initial_support_count: int = 32
    stream_count: int = 32
    max_features: int = 5
    hidden_size: int = 256
    probe_epochs: int = 400
    patience: int = 40
    learning_rates: tuple[float, ...] = (1e-2, 3e-3, 1e-3)
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def validate(self) -> None:
        if self.episodes < 32:
            raise ValueError("At least 32 episodes are needed to fit and score a probe.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not self.learning_rates:
            raise ValueError("At least one learning rate is required.")
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("Split fractions must lie strictly between 0 and 1.")
        if self.validation_fraction + self.test_fraction >= 0.8:
            raise ValueError("Training must keep at least 20 percent of the rows.")
        if self.patience < 1 or self.probe_epochs < 1:
            raise ValueError("probe_epochs and patience must be positive.")


def _episode_config(config: StructuralProbeConfig) -> PriorBimodalConfig:
    return PriorBimodalConfig(
        initial_support_count=config.initial_support_count,
        stream_count=config.stream_count,
        query_count=4,
        max_features=config.max_features,
        max_pair_attempts=512,
        device=config.device,
        compute_structural_latents=True,
    )


def _schema(config: StructuralProbeConfig) -> StructuralLatentSchema:
    return StructuralLatentSchema(max_features=config.max_features)


def _pool(states: torch.Tensor) -> torch.Tensor:
    """Mean and standard deviation over context rows, concatenated."""
    return torch.cat((states.mean(1), states.std(1)), dim=-1)


@torch.no_grad()
def _pooled_context(
    backbone: NanoTabPFNModel, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
) -> torch.Tensor:
    """Pool the target-column context states the slots are themselves built from.

    The query rows are kept in the encode so the call shape matches the adapter's
    exactly - the backbone splits train from test by position, and an empty test
    block is not a shape it is ever asked to handle.  Only the labeled prefix is
    pooled, which is the same block the slot attention reads.
    """
    encoded = backbone.encode_table(
        (torch.cat((context_x, query_x), dim=1), context_y.float()),
        train_test_split_index=context_x.shape[1],
    )
    return _pool(encoded[:, : context_x.shape[1], -1, :])


def _paired_batch(config: StructuralProbeConfig, episode_config: PriorBimodalConfig, rng, *, batch_size: int):
    """Draw one paired batch, retrying exhausted rejection-sampling budgets.

    Pairing is rejection sampling against disagreement thresholds, so a single
    call can legitimately run out of attempts; advancing the generator and
    retrying is what ``run_task_posterior_local_evaluation`` does too.
    """
    last_error = None
    for _ in range(20):
        try:
            return generate_h5_prior_bimodal_episodes(config.prior_dump, episode_config, rng, batch_size=batch_size)
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("Paired generator failed after 20 advanced retries.") from last_error


@torch.no_grad()
def _matched_slots(model: NanoTabPFNTaskPosteriorAdapter, batch) -> torch.Tensor:
    """Slot of the particle each candidate was matched to, ``(batch, candidates, E)``."""
    context_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
    context_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1).long()
    prediction = model(context_x, context_y, batch.query_x, class_count=2)
    assignment = match_regimes_to_particles(prediction, batch.candidate_query_y.long())
    index = assignment.particle_for_regime[:, :, None].expand(-1, -1, prediction.slots.shape[-1])
    return prediction.slots.gather(1, index)


def extract_representations(
    config: StructuralProbeConfig,
    trained: NanoTabPFNTaskPosteriorAdapter,
    untrained: NanoTabPFNTaskPosteriorAdapter,
) -> dict[str, torch.Tensor]:
    """Run every frozen model once per episode and cache what the probes need.

    Returns flat ``(episodes * candidates, ...)`` tensors so probes can be fit
    without touching a backbone again.
    """
    config.validate()
    trained.eval()
    untrained.eval()
    episode_config = _episode_config(config)
    rng = np.random.default_rng(config.seed)
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in REPRESENTATIONS}
    targets: list[torch.Tensor] = []
    is_true: list[torch.Tensor] = []

    remaining = config.episodes
    while remaining > 0:
        current = min(config.batch_size, remaining)
        batch = _paired_batch(config, episode_config, rng, batch_size=current)
        remaining -= current
        context_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
        context_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1).long()
        candidates = batch.candidate_structural_z.shape[1]

        collected["trained_slots"].append(_matched_slots(trained, batch).cpu())
        collected["untrained_slots"].append(_matched_slots(untrained, batch).cpu())
        # Observed labels are shared by both candidates, hence the expand: this
        # representation is per episode and cannot separate them by construction.
        observed = _pooled_context(trained.backbone, context_x, context_y, batch.query_x)
        collected["observed_context"].append(observed[:, None].expand(-1, candidates, -1).cpu())
        per_candidate = torch.stack(
            [
                _pooled_context(
                    trained.backbone,
                    context_x,
                    torch.cat(
                        (batch.candidate_support_y[:, candidate], batch.candidate_stream_y[:, candidate]), dim=1
                    ).long(),
                    batch.query_x,
                )
                for candidate in range(candidates)
            ],
            dim=1,
        )
        collected["candidate_context"].append(per_candidate.cpu())
        targets.append(batch.candidate_structural_z.cpu())
        task = batch.candidate_task.cpu()[:, None]
        is_true.append(torch.arange(candidates)[None, :] == task)

    cached = {name: torch.cat(values).flatten(0, 1).float() for name, values in collected.items()}
    cached["targets"] = torch.cat(targets).flatten(0, 1).float()
    cached["is_true_candidate"] = torch.cat(is_true).flatten().bool()
    return cached


def _build_probe(input_size: int, output_size: int, capacity: str, hidden_size: int) -> nn.Module:
    """Build a probe head.

    Deliberately no input LayerNorm.  It normalizes across the feature axis of
    each row, which discards that row's mean and scale - information a probe is
    supposed to be measuring, not deleting.  Conditioning is handled instead by
    standardizing each dimension with training-split statistics in
    :func:`fit_probe`, which preserves every per-row difference.
    """
    if capacity == "linear":
        return nn.Linear(input_size, output_size)
    if capacity == "mlp1":
        return nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, output_size))
    if capacity == "mlp2":
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    raise ValueError(f"Unknown capacity {capacity!r}; expected one of {CAPACITIES}.")


def _splits(rows: int, config: StructuralProbeConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(rows, generator=generator)
    test = int(rows * config.test_fraction)
    validation = int(rows * config.validation_fraction)
    return order[test + validation :], order[:validation], order[validation : validation + test]


def fit_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    config: StructuralProbeConfig,
    *,
    capacity: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit one probe to convergence, sweeping learning rates on the validation split.

    Returns test-split predictions and the fit's diagnostics.  Early stopping on
    validation loss is what distinguishes this from the in-loop probe: capacity
    is only a fair test of the representation once the probe has converged.
    """
    train_index, validation_index, test_index = _splits(features.shape[0], config)
    # Standardize per dimension using training rows only, so the optimizer sees a
    # well-conditioned problem without any information crossing the split.
    mean = features[train_index].mean(0, keepdim=True)
    std = features[train_index].std(0, keepdim=True).clamp_min(1e-6)
    features = (features - mean) / std
    best_state, best_loss, best_rate, best_epoch = None, float("inf"), None, 0
    for rate in config.learning_rates:
        torch.manual_seed(config.seed)
        probe = _build_probe(features.shape[-1], targets.shape[-1], capacity, config.hidden_size)
        optimizer = torch.optim.Adam(probe.parameters(), lr=rate, weight_decay=config.weight_decay)
        rate_state, rate_loss, stale = None, float("inf"), 0
        for epoch in range(config.probe_epochs):
            probe.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(probe(features[train_index]), targets[train_index])
            loss.backward()
            optimizer.step()
            probe.eval()
            with torch.no_grad():
                validation_loss = float(
                    nn.functional.mse_loss(probe(features[validation_index]), targets[validation_index])
                )
            if validation_loss < rate_loss - 1e-7:
                rate_loss, stale = validation_loss, 0
                rate_state = {key: value.detach().clone() for key, value in probe.state_dict().items()}
                if validation_loss < best_loss:
                    best_loss, best_rate, best_epoch = validation_loss, rate, epoch
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if rate_loss <= best_loss and rate_state is not None:
            best_state = rate_state
    probe = _build_probe(features.shape[-1], targets.shape[-1], capacity, config.hidden_size)
    assert best_state is not None
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        predictions = probe(features[test_index])
    diagnostics = {
        "validation_mse": best_loss,
        "learning_rate": float(best_rate) if best_rate is not None else float("nan"),
        "best_epoch": float(best_epoch),
        "train_rows": float(len(train_index)),
        "test_rows": float(len(test_index)),
    }
    return predictions, diagnostics


def score_representations(cached: dict[str, torch.Tensor], config: StructuralProbeConfig) -> pd.DataFrame:
    """Fit every representation x capacity, plus a shuffled-target null."""
    schema = _schema(config)
    targets = cached["targets"]
    _, _, test_index = _splits(targets.shape[0], config)
    test_truth = cached["is_true_candidate"][test_index]
    generator = torch.Generator().manual_seed(config.seed + 1)
    shuffled = targets[torch.randperm(targets.shape[0], generator=generator)]

    rows = []
    for name in REPRESENTATIONS:
        features = cached[name]
        for capacity in CAPACITIES:
            for label, target in (("real", targets), ("shuffled_null", shuffled)):
                predictions, diagnostics = fit_probe(features, target, config, capacity=capacity)
                actual = target[test_index]
                for subset, mask in (
                    ("all", torch.ones_like(test_truth)),
                    ("true_candidate", test_truth),
                    ("false_candidate", ~test_truth),
                ):
                    if not bool(mask.any()):
                        continue
                    report = probe_r2(predictions[mask], actual[mask], schema)
                    rows.append(
                        {
                            "representation": name,
                            "capacity": capacity,
                            "targets": label,
                            "subset": subset,
                            "feature_size": features.shape[-1],
                            **{key: value for key, value in report.items()},
                            **diagnostics,
                        }
                    )
    return pd.DataFrame(rows)


def _load_models(
    config: StructuralProbeConfig,
) -> tuple[NanoTabPFNTaskPosteriorAdapter, NanoTabPFNTaskPosteriorAdapter]:
    trained, checkpoint = load_task_posterior_checkpoint(config.checkpoint, map_location=config.device)
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    # The control shares the *pretrained* backbone and differs only in that its
    # adapter was never trained, isolating what adapter training contributes.
    backbone.load_state_dict(trained.backbone.state_dict())
    untrained = NanoTabPFNTaskPosteriorAdapter(
        backbone,
        particle_count=architecture["particle_count"],
        max_classes=architecture["max_classes"],
        context_mode=architecture.get("context_mode", "iid_set"),
        residual_logit_bound=architecture.get("residual_logit_bound"),
        slot_mode=architecture.get("slot_mode", "deterministic"),
        slot_sample_seed=architecture.get("slot_sample_seed", 0),
    )
    return trained.to(config.device).eval(), untrained.to(config.device).eval()


def run(config: StructuralProbeConfig) -> Path:
    config.validate()
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    set_randomness_seed(config.seed)
    trained, untrained = _load_models(config)
    cached = extract_representations(config, trained, untrained)
    torch.save(cached, output / "representations.pt")

    scores = score_representations(cached, config)
    scores.to_csv(output / "probe_scores.csv", index=False)

    real = scores[(scores.targets == "real") & (scores.subset == "all")]
    best = real.loc[real.groupby("representation").mean_continuous_r2.idxmax()]
    null = scores[scores.targets == "shuffled_null"]
    summary = {
        "protocol": "frozen-representation structural probing; not TabArena",
        "checkpoint": config.checkpoint,
        "episodes": config.episodes,
        "rows": int(cached["targets"].shape[0]),
        "true_candidate_fraction": float(cached["is_true_candidate"].double().mean()),
        "best_r2_by_representation": {
            row.representation: {"capacity": row.capacity, "mean_continuous_r2": float(row.mean_continuous_r2)}
            for row in best.itertuples()
        },
        "max_shuffled_null_r2": float(null.mean_continuous_r2.max()),
        "upper_bound_minus_trained_slots": float(
            best.set_index("representation").loc["candidate_context"].mean_continuous_r2
            - best.set_index("representation").loc["trained_slots"].mean_continuous_r2
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior-dump", default="300k_150x5_2.h5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--episodes", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-support-count", type=int, default=32)
    parser.add_argument("--stream-count", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StructuralProbeConfig(
        checkpoint=args.checkpoint,
        prior_dump=args.prior_dump,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        episodes=args.episodes,
        batch_size=args.batch_size,
        initial_support_count=args.initial_support_count,
        stream_count=args.stream_count,
        hidden_size=args.hidden_size,
        probe_epochs=args.probe_epochs,
    )
    print(run(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
