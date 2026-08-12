"""Train and evaluate a minimal sequential latent Bayes filter.

The vanilla nanoTabPFN checkpoint is frozen and used exactly once per episode
to embed an initial support, an unlabeled chronological stream, and four query
rows. Only a two-slot head is optimized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.sequential_latent_filter import (
    NanoTabPFNSequentialLatentFilter,
    SequentialFilterPrediction,
    load_sequential_filter_checkpoint,
    save_sequential_filter_checkpoint,
)
from tfmplayground.utils import get_default_device, set_randomness_seed

CONDITIONS = ("neutral", "consistent_zero", "consistent_one", "contradictory", "noisy")
MILESTONES = (0, 1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class SequentialFilterConfig:
    seed: int = 2402
    stage: str = "controlled"
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    controlled_checkpoint: str | None = None
    output_dir: str | None = None
    device: str = "cpu"
    initial_support_count: int = 16
    stream_count: int = 128
    query_count: int = 4
    batch_size: int = 16
    controlled_steps: int = 1000
    tabicl_steps: int = 500
    validation_interval: int = 50
    patience: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    diversity_weight: float = 0.05
    diversity_target: float = 0.20
    evaluation_trials: int = 256
    ordinary_evaluation_batches: int = 8
    plots: bool = True


@dataclass
class SequentialEpisodeBatch:
    initial_support_x: torch.Tensor
    initial_support_y: torch.Tensor
    stream_x: torch.Tensor
    stream_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    # The remaining fields are evaluator-only generator truth. They are never
    # accepted by the model or used by sequential_filter_loss.
    conditions: tuple[str, ...] = ()
    latent_class: torch.Tensor | None = None
    initial_evidence_class: torch.Tensor | None = None
    exact_p1: torch.Tensor | None = None
    # Optional paired-task metadata used by prior-generated ambiguity episodes.
    # These fields are evaluator-only and are never passed to a model.
    candidate_task: torch.Tensor | None = None
    candidate_support_y: torch.Tensor | None = None
    candidate_stream_y: torch.Tensor | None = None
    candidate_query_y: torch.Tensor | None = None
    support_disagreement: torch.Tensor | None = None
    stream_disagreement: torch.Tensor | None = None
    query_disagreement: torch.Tensor | None = None
    pair_attempts: torch.Tensor | None = None
    # Structural summaries of each candidate task, laid out by
    # tfmplayground.experiments.structural_latents.StructuralLatentSchema.  Like
    # every field above they are generator truth: a loss may consume them, a
    # model's forward pass never does.  HDF5 prior dumps store no structure, so
    # dump-backed generators leave both fields None.
    candidate_structural_z: torch.Tensor | None = None
    structural_feature_mask: torch.Tensor | None = None


def validate_config(config: SequentialFilterConfig) -> None:
    if config.stage not in {"controlled", "tabicl", "all"}:
        raise ValueError("stage must be controlled, tabicl, or all.")
    if config.query_count != 4:
        raise ValueError("The proof of concept requires exactly four final queries.")
    if config.initial_support_count < 4 or config.initial_support_count % 2:
        raise ValueError("initial_support_count must be an even number of at least four.")
    if config.stream_count < 1:
        raise ValueError("stream_count must be positive.")
    for name in ("batch_size", "validation_interval", "patience", "evaluation_trials"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if min(config.controlled_steps, config.tabicl_steps, config.ordinary_evaluation_batches) < 0:
        raise ValueError("Training and evaluation counts cannot be negative.")
    if config.stage == "tabicl" and not config.controlled_checkpoint:
        raise ValueError("tabicl stage requires --controlled-checkpoint.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def _condition_schedule(batch_size: int, rng: np.random.Generator, condition: str | None) -> list[str]:
    if condition is not None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown controlled condition: {condition}")
        return [condition] * batch_size
    offset = int(rng.integers(0, len(CONDITIONS)))
    values = [CONDITIONS[(offset + index) % len(CONDITIONS)] for index in range(batch_size)]
    rng.shuffle(values)
    return values


def _normalized_binary_log_weights(log_weights: np.ndarray, row_log_likelihood: np.ndarray) -> np.ndarray:
    updated = log_weights + row_log_likelihood
    maximum = updated.max()
    return updated - (maximum + math.log(np.exp(updated - maximum).sum()))


def generate_controlled_episodes(
    config: SequentialFilterConfig,
    rng: np.random.Generator,
    *,
    condition: str | None = None,
    batch_size: int | None = None,
) -> SequentialEpisodeBatch:
    """Generate observed episodes plus evaluator-only exact posterior paths."""
    batch_size = batch_size or config.batch_size
    support_count = config.initial_support_count
    stream_count = config.stream_count
    query_count = config.query_count
    half = support_count // 2
    conditions = _condition_schedule(batch_size, rng, condition)

    support_x = np.empty((batch_size, support_count, 1), dtype=np.float32)
    support_y = np.empty((batch_size, support_count), dtype=np.float32)
    stream_x = np.empty((batch_size, stream_count, 1), dtype=np.float32)
    stream_y = np.empty((batch_size, stream_count), dtype=np.float32)
    query_x = np.broadcast_to(
        np.linspace(0.25, 1.0, query_count, dtype=np.float32)[None, :, None],
        (batch_size, query_count, 1),
    ).copy()
    query_y = np.empty((batch_size, query_count), dtype=np.int64)
    latent_class = np.empty(batch_size, dtype=np.int64)
    initial_evidence_class = np.full(batch_size, -1, dtype=np.int64)
    exact_p1 = np.empty((batch_size, stream_count + 1), dtype=np.float32)

    for batch_index, name in enumerate(conditions):
        negative_one = rng.uniform(-2.0, -1.25, size=(half, 1))
        negative_zero = rng.uniform(-0.75, -0.25, size=(half, 1))
        rows = np.concatenate((negative_one, negative_zero), axis=0)
        labels = np.concatenate((np.ones(half), np.zeros(half)))
        order = rng.permutation(support_count)
        support_x[batch_index] = rows[order]
        support_y[batch_index] = labels[order]

        latent = int(rng.integers(0, 2))
        noise_rate = 0.1
        if name == "neutral":
            neutral_classes = np.arange(stream_count) % 2
            rng.shuffle(neutral_classes)
            one_mask = neutral_classes == 1
            values = np.empty(stream_count)
            values[one_mask] = rng.uniform(-2.0, -1.25, size=one_mask.sum())
            values[~one_mask] = rng.uniform(-0.75, -0.25, size=(~one_mask).sum())
            stream_x[batch_index, :, 0] = values
            stream_y[batch_index] = neutral_classes
            row_labels = neutral_classes
            # Both latent tasks define the same likelihood in the neutral region.
            row_log_likelihood = np.zeros((stream_count, 2), dtype=np.float64)
        else:
            stream_x[batch_index, :, 0] = rng.uniform(0.05, 1.25, size=stream_count)
            if name == "consistent_zero":
                latent = 0
                row_labels = np.zeros(stream_count, dtype=np.int64)
            elif name == "consistent_one":
                latent = 1
                row_labels = np.ones(stream_count, dtype=np.int64)
            elif name == "contradictory":
                first = int(rng.integers(0, 2))
                initial_evidence_class[batch_index] = first
                latent = 1 - first
                row_labels = np.full(stream_count, latent, dtype=np.int64)
                row_labels[: min(16, stream_count)] = first
            else:
                noise_rate = float(rng.choice((0.10, 0.25, 0.40)))
                row_labels = np.full(stream_count, latent, dtype=np.int64)
                flips = rng.random(stream_count) < noise_rate
                row_labels = np.logical_xor(row_labels, flips).astype(np.int64)
            stream_y[batch_index] = row_labels
            correct = math.log1p(-noise_rate)
            flipped = math.log(noise_rate)
            row_log_likelihood = np.stack(
                (
                    np.where(row_labels == 0, correct, flipped),
                    np.where(row_labels == 1, correct, flipped),
                ),
                axis=-1,
            )

        latent_class[batch_index] = latent
        query_y[batch_index] = latent
        exact_log_weights = np.array([-math.log(2.0), -math.log(2.0)])
        exact_p1[batch_index, 0] = 0.5
        for row_index in range(stream_count):
            exact_log_weights = _normalized_binary_log_weights(exact_log_weights, row_log_likelihood[row_index])
            exact_p1[batch_index, row_index + 1] = math.exp(exact_log_weights[1])

    device = torch.device(config.device)
    return SequentialEpisodeBatch(
        initial_support_x=torch.from_numpy(support_x).to(device),
        initial_support_y=torch.from_numpy(support_y).to(device),
        stream_x=torch.from_numpy(stream_x).to(device),
        stream_y=torch.from_numpy(stream_y).to(device),
        query_x=torch.from_numpy(query_x).to(device),
        query_y=torch.from_numpy(query_y).to(device),
        conditions=tuple(conditions),
        latent_class=torch.from_numpy(latent_class).to(device),
        initial_evidence_class=torch.from_numpy(initial_evidence_class).to(device),
        exact_p1=torch.from_numpy(exact_p1).to(device),
    )


def predict_episode(
    model: NanoTabPFNSequentialLatentFilter, batch: SequentialEpisodeBatch
) -> SequentialFilterPrediction:
    return model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x,
    )


def _outcome_indices(labels: torch.Tensor) -> torch.Tensor:
    powers = 2 ** torch.arange(labels.shape[1] - 1, -1, -1, device=labels.device)
    return (labels.long() * powers).sum(dim=1)


def _js_divergence(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    midpoint = 0.5 * (left + right)

    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.where(
            p > 0,
            p * (p.clamp_min(epsilon).log() - q.clamp_min(epsilon).log()),
            0.0,
        ).sum(-1)

    return 0.5 * (kl(left, midpoint) + kl(right, midpoint))


def sequential_filter_loss(
    model: NanoTabPFNSequentialLatentFilter,
    batch: SequentialEpisodeBatch,
    *,
    diversity_weight: float = 0.05,
    diversity_target: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = predict_episode(model, batch)
    prequential_loss = -prediction.prequential_log_likelihood.mean()
    joint = prediction.joint_probabilities()
    indices = _outcome_indices(batch.query_y)
    selected_joint = joint.gather(-1, indices[:, None, None].expand(-1, joint.shape[1], 1)).squeeze(-1)
    trajectory_loss = -selected_joint.clamp_min(1e-12).log().mean()
    marginals = prediction.marginal_probabilities()
    labels = batch.query_y[:, None, :, None].expand(-1, marginals.shape[1], -1, 1)
    marginal_loss = -marginals.gather(-1, labels).squeeze(-1).clamp_min(1e-12).log().mean()
    slot_joint = prediction.slot_joint_log_probabilities().exp()
    slot_js = _js_divergence(slot_joint[:, 0], slot_joint[:, 1])
    diversity_loss = F.relu(diversity_target - slot_js).mean()
    total = prequential_loss + trajectory_loss + marginal_loss + diversity_weight * diversity_loss
    metrics = {
        "loss": float(total.detach()),
        "prequential_loss": float(prequential_loss.detach()),
        "trajectory_loss": float(trajectory_loss.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "diversity_loss": float(diversity_loss.detach()),
        "slot_joint_js": float(slot_js.mean().detach()),
    }
    return total, metrics


def make_tabicl_iterator(config: SequentialFilterConfig, num_steps: int) -> Iterator[dict]:
    from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader

    # TabICL samples its train/test split over a wide range. Request tables
    # large enough that a draw can contain the configured prior plus queries;
    # next_tabicl_episode separately caps updates at the prior-set size.
    minimum_datapoints = max(32, 2 * config.initial_support_count + config.query_count)
    maximum_datapoints = max(128, 4 * config.initial_support_count + config.query_count + 1)
    loader = TabICLPriorDataLoader(
        num_steps=num_steps,
        batch_size=config.batch_size,
        num_datapoints_min=minimum_datapoints,
        num_datapoints_max=maximum_datapoints,
        min_features=1,
        max_features=3,
        max_num_classes=2,
        prior_type="mix_scm",
        device=torch.device(config.device),
    )
    return iter(loader)


def next_tabicl_episode(
    iterator: Iterator[dict], config: SequentialFilterConfig, rng: np.random.Generator
) -> SequentialEpisodeBatch:
    for _ in range(100):
        try:
            raw = next(iterator)
        except StopIteration:
            break
        split = int(raw["train_test_split_index"])
        if split <= config.initial_support_count or raw["x"].shape[1] - split < config.query_count:
            continue
        support_x = raw["x"][:, :split].float()
        support_y = raw["y"][:, :split].float()
        batch_size = support_x.shape[0]
        permutations = torch.stack(
            [torch.as_tensor(rng.permutation(split), device=support_x.device) for _ in range(batch_size)]
        )
        gather_x = permutations[:, :, None].expand(-1, -1, support_x.shape[2])
        shuffled_x = support_x.gather(1, gather_x)
        shuffled_y = support_y.gather(1, permutations)
        update_count = min(
            split - config.initial_support_count,
            config.initial_support_count,
            config.stream_count,
        )
        return SequentialEpisodeBatch(
            initial_support_x=shuffled_x[:, : config.initial_support_count],
            initial_support_y=shuffled_y[:, : config.initial_support_count],
            stream_x=shuffled_x[:, config.initial_support_count : config.initial_support_count + update_count],
            stream_y=shuffled_y[:, config.initial_support_count : config.initial_support_count + update_count],
            query_x=raw["x"][:, split : split + config.query_count].float(),
            query_y=raw["target_y"][:, split : split + config.query_count].long(),
        )
    raise RuntimeError("TabICL did not yield a sufficiently large binary episode after 100 attempts.")


def _validation_loss(model: NanoTabPFNSequentialLatentFilter, config: SequentialFilterConfig) -> float:
    rng = np.random.default_rng(config.seed + 50_000)
    model.eval()
    values = []
    with torch.no_grad():
        for condition in CONDITIONS:
            batch = generate_controlled_episodes(config, rng, condition=condition, batch_size=min(config.batch_size, 5))
            loss, _ = sequential_filter_loss(
                model,
                batch,
                diversity_weight=config.diversity_weight,
                diversity_target=config.diversity_target,
            )
            values.append(float(loss))
    model.train()
    return float(np.mean(values))


def train_head(
    model: NanoTabPFNSequentialLatentFilter,
    config: SequentialFilterConfig,
    rng: np.random.Generator,
    *,
    stage: str,
) -> tuple[NanoTabPFNSequentialLatentFilter, list[dict[str, Any]], torch.optim.Optimizer]:
    if stage not in {"controlled", "tabicl"}:
        raise ValueError("Training stage must be controlled or tabicl.")
    steps = config.controlled_steps if stage == "controlled" else config.tabicl_steps
    model.to(config.device).train()
    head_parameters = list(model.head_parameters())
    optimizer = torch.optim.AdamW(head_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    ordinary = make_tabicl_iterator(config, max(1, steps * 100)) if stage == "tabicl" and steps else None
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale = 0
    for step in range(1, steps + 1):
        if stage == "tabicl" and step % 2 == 0:
            batch = next_tabicl_episode(ordinary, config, rng)
        else:
            batch = generate_controlled_episodes(config, rng)
        optimizer.zero_grad()
        loss, metrics = sequential_filter_loss(
            model,
            batch,
            diversity_weight=config.diversity_weight,
            diversity_target=config.diversity_target,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
        optimizer.step()
        row: dict[str, Any] = {"stage": stage, "step": step, **metrics}
        if step % config.validation_interval == 0 or step == steps:
            row["validation_loss"] = _validation_loss(model, config)
            if row["validation_loss"] < best_validation:
                best_validation = row["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history, optimizer


def _effective_milestones(stream_count: int) -> tuple[int, ...]:
    values = tuple(value for value in MILESTONES if value <= stream_count)
    return values if values[-1] == stream_count else (*values, stream_count)


def _canonical_permutation_indices(order: Sequence[int], device: torch.device) -> torch.Tensor:
    outcomes = torch.arange(2 ** len(order), device=device)
    bits = (outcomes[:, None] >> torch.arange(len(order) - 1, -1, -1, device=device)) & 1
    permuted_bits = bits[:, torch.as_tensor(order, device=device)]
    powers = 2 ** torch.arange(len(order) - 1, -1, -1, device=device)
    return (permuted_bits * powers).sum(-1)


@torch.no_grad()
def _invariance_errors(
    model: NanoTabPFNSequentialLatentFilter,
    batch: SequentialEpisodeBatch,
    rng: np.random.Generator,
) -> dict[str, float]:
    original = predict_episode(model, batch)
    query_order = (3, 1, 0, 2)
    permuted_query_batch = copy.copy(batch)
    permuted_query_batch.query_x = batch.query_x[:, query_order]
    query_prediction = predict_episode(model, permuted_query_batch)
    mapping = _canonical_permutation_indices(query_order, batch.query_x.device)
    remapped_query = query_prediction.joint_probabilities()[:, :, mapping]
    query_error = (original.joint_probabilities() - remapped_query).abs().max()

    order = torch.as_tensor(rng.permutation(batch.stream_x.shape[1]), device=batch.stream_x.device)
    reordered_batch = copy.copy(batch)
    reordered_batch.stream_x = batch.stream_x[:, order]
    reordered_batch.stream_y = batch.stream_y[:, order]
    reordered = predict_episode(model, reordered_batch)
    weight_error = (original.log_weights[:, -1].exp() - reordered.log_weights[:, -1].exp()).abs().max()
    joint_error = (original.joint_probabilities()[:, -1] - reordered.joint_probabilities()[:, -1]).abs().max()
    return {
        "query_permutation_max_error": float(query_error),
        "evidence_order_weight_max_error": float(weight_error),
        "evidence_order_joint_max_error": float(joint_error),
    }


@torch.no_grad()
def evaluate_controlled(
    model: NanoTabPFNSequentialLatentFilter,
    config: SequentialFilterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model.to(config.device).eval()
    milestones = _effective_milestones(config.stream_count)
    state_indices = torch.as_tensor(milestones, device=config.device)
    rows: list[dict[str, Any]] = []
    invariance = {
        "query_permutation_max_error": 0.0,
        "evidence_order_weight_max_error": 0.0,
        "evidence_order_joint_max_error": 0.0,
    }
    for condition_index, condition in enumerate(CONDITIONS):
        rng = np.random.default_rng(config.seed + 1000 + condition_index)
        batch = generate_controlled_episodes(config, rng, condition=condition, batch_size=config.evaluation_trials)
        prediction = predict_episode(model, batch)
        weights = prediction.log_weights.index_select(1, state_indices).exp()
        joint = prediction.joint_probabilities(state_indices)
        marginals = prediction.marginal_probabilities(state_indices)
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        slot_js = _js_divergence(slot_joint[:, 0], slot_joint[:, 1])
        class_zero_slot = slot_joint[:, :, 0].argmax(dim=1)
        class_one_slot = slot_joint[:, :, -1].argmax(dim=1)
        different_slots = class_zero_slot != class_one_slot
        exact_p1 = batch.exact_p1.index_select(1, state_indices)
        exact_binary = torch.stack((1.0 - exact_p1, exact_p1), dim=-1)
        predicted_binary = marginals.mean(dim=2)
        marginal_js = _js_divergence(exact_binary, predicted_binary)

        subset = SequentialEpisodeBatch(
            initial_support_x=batch.initial_support_x[: min(16, config.evaluation_trials)],
            initial_support_y=batch.initial_support_y[: min(16, config.evaluation_trials)],
            stream_x=batch.stream_x[: min(16, config.evaluation_trials)],
            stream_y=batch.stream_y[: min(16, config.evaluation_trials)],
            query_x=batch.query_x[: min(16, config.evaluation_trials)],
            query_y=batch.query_y[: min(16, config.evaluation_trials)],
        )
        errors = _invariance_errors(model, subset, rng)
        for name, value in errors.items():
            invariance[name] = max(invariance[name], value)

        for trial in range(config.evaluation_trials):
            target_class = int(batch.latent_class[trial])
            target_slot = class_one_slot[trial] if target_class else class_zero_slot[trial]
            initial_class = int(batch.initial_evidence_class[trial])
            initial_slot = (
                class_one_slot[trial]
                if initial_class == 1
                else class_zero_slot[trial]
                if initial_class == 0
                else target_slot
            )
            opposite_slot = class_zero_slot[trial] if initial_class == 1 else class_one_slot[trial]
            for state_position, milestone in enumerate(milestones):
                row = {
                    "condition": condition,
                    "trial": trial,
                    "milestone": milestone,
                    "weight_0": float(weights[trial, state_position, 0]),
                    "weight_1": float(weights[trial, state_position, 1]),
                    "incoherent_mass": float(
                        (1.0 - joint[trial, state_position, 0] - joint[trial, state_position, -1]).clamp_min(0)
                    ),
                    "marginal_js": float(marginal_js[trial, state_position]),
                    "mean_p1": float(marginals[trial, state_position, :, 1].mean()),
                    "exact_p1": float(exact_p1[trial, state_position]),
                    "slot_joint_js": float(slot_js[trial]),
                    "different_supporting_slots": float(different_slots[trial]),
                    "supporting_weight": float(weights[trial, state_position, target_slot]),
                    "initial_supporting_weight": float(weights[trial, state_position, initial_slot]),
                    "opposite_weight": float(weights[trial, state_position, opposite_slot]),
                }
                rows.append(row)
    trials = pd.DataFrame(rows)
    metric_names = (
        "weight_0",
        "weight_1",
        "incoherent_mass",
        "marginal_js",
        "mean_p1",
        "exact_p1",
        "slot_joint_js",
        "different_supporting_slots",
        "supporting_weight",
        "initial_supporting_weight",
        "opposite_weight",
    )
    summaries = []
    for (condition, milestone), group in trials.groupby(["condition", "milestone"], sort=False):
        for metric in metric_names:
            values = group[metric]
            count = int(values.count())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            sem = std / math.sqrt(count) if count else math.nan
            mean = float(values.mean())
            summaries.append(
                {
                    "condition": condition,
                    "milestone": milestone,
                    "metric": metric,
                    "count": count,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                }
            )
    return trials, pd.DataFrame(summaries), invariance


def controlled_gate_report(
    trials: pd.DataFrame,
    invariance: dict[str, float],
    *,
    stream_count: int,
) -> dict[str, Any]:
    final = stream_count
    neutral = trials[(trials.condition == "neutral") & (trials.milestone == final)]
    neutral_signed = float((neutral.weight_0 - 0.5).mean())
    neutral_absolute = float((neutral.weight_0 - 0.5).abs().mean())

    monotonic_drops = {}
    final_consistent = {}
    for condition in ("consistent_zero", "consistent_one"):
        curve = trials[trials.condition == condition].groupby("milestone").supporting_weight.mean()
        monotonic_drops[condition] = float(curve.diff().min())
        final_consistent[condition] = float(curve.loc[final])

    contradictory = trials[trials.condition == "contradictory"]
    switch = min(16, stream_count)
    at_switch = contradictory[contradictory.milestone == switch].set_index("trial")
    at_final = contradictory[contradictory.milestone == final].set_index("trial")
    confidence_drop = float((at_switch.initial_supporting_weight - at_final.initial_supporting_weight).mean())
    reversed_weight = float(at_final.opposite_weight.mean())
    final_rows = trials[trials.milestone == final]
    incoherent_mass = float(final_rows.incoherent_mass.mean())
    slot_joint_js = float(final_rows.slot_joint_js.mean())
    different_slots = float(final_rows.different_supporting_slots.mean())
    checks = {
        "neutral_absolute_drift": neutral_absolute <= 0.05,
        "neutral_signed_drift": abs(neutral_signed) <= 0.01,
        "consistent_zero_monotonic": monotonic_drops["consistent_zero"] >= -0.02,
        "consistent_one_monotonic": monotonic_drops["consistent_one"] >= -0.02,
        "consistent_zero_concentration": final_consistent["consistent_zero"] >= 0.90,
        "consistent_one_concentration": final_consistent["consistent_one"] >= 0.90,
        "contradiction_confidence_drop": confidence_drop >= 0.40,
        "contradiction_reversal": reversed_weight >= 0.70,
        "incoherent_mass": incoherent_mass <= 0.10,
        "slot_joint_diversity": slot_joint_js >= 0.20,
        "query_permutation": invariance["query_permutation_max_error"] <= 1e-5,
        "evidence_order_weights": invariance["evidence_order_weight_max_error"] <= 1e-5,
        "evidence_order_joint": invariance["evidence_order_joint_max_error"] <= 1e-5,
        "different_supporting_slots": different_slots >= 0.90,
    }
    return {
        "threshold_profile": "pragmatic_poc_v1",
        "metrics": {
            "neutral_absolute_drift": neutral_absolute,
            "neutral_signed_drift": neutral_signed,
            "largest_consistent_weight_drop": monotonic_drops,
            "final_consistent_supporting_weight": final_consistent,
            "contradiction_confidence_drop": confidence_drop,
            "contradiction_reversed_weight": reversed_weight,
            "final_incoherent_mass": incoherent_mass,
            "final_slot_joint_js": slot_joint_js,
            "different_supporting_slots_fraction": different_slots,
            **invariance,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
    }


def _save_plots(summary: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    conditions = ("neutral", "consistent_zero", "consistent_one", "contradictory", "noisy")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for condition in conditions:
        selected = summary[(summary.condition == condition) & (summary.metric == "weight_1")]
        axes[0].plot(selected.milestone, selected["mean"], marker="o", label=condition)
        incoherent = summary[(summary.condition == condition) & (summary.metric == "incoherent_mass")]
        axes[1].plot(incoherent.milestone, incoherent["mean"], marker="o", label=condition)
        p1 = summary[(summary.condition == condition) & (summary.metric == "mean_p1")]
        axes[2].plot(p1.milestone, p1["mean"], marker="o", label=condition)
    axes[0].set_title("Latent weight 1")
    axes[1].set_title("Four-query incoherent mass")
    axes[2].set_title("Mean P(y=1)")
    for axis in axes:
        axis.set_xlabel("Arriving labelled samples")
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Probability")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_trajectories.png", dpi=180)
    fig.savefig(output_dir / f"{prefix}_trajectories.pdf")
    plt.close(fig)


@torch.no_grad()
def evaluate_ordinary_accuracy(
    models: dict[str, NanoTabPFNSequentialLatentFilter],
    baseline: NanoTabPFNModel,
    config: SequentialFilterConfig,
) -> dict[str, float | None]:
    if config.ordinary_evaluation_batches == 0:
        return {"vanilla": None, **{name: None for name in models}}
    iterator = make_tabicl_iterator(config, config.ordinary_evaluation_batches * 100)
    rng = np.random.default_rng(config.seed + 80_000)
    correct = {"vanilla": torch.zeros(2, dtype=torch.long)}
    correct.update({name: torch.zeros(2, dtype=torch.long) for name in models})
    total = torch.zeros(2, dtype=torch.long)
    baseline.to(config.device).eval()
    for _ in range(config.ordinary_evaluation_batches):
        batch = next_tabicl_episode(iterator, config, rng)
        labels = batch.query_y.cpu()
        for class_index in range(2):
            total[class_index] += (labels == class_index).sum()
        full_support_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
        full_support_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1)
        vanilla_logits = baseline(
            (torch.cat((full_support_x, batch.query_x), dim=1), full_support_y),
            train_test_split_index=full_support_x.shape[1],
        )[..., :2]
        predictions = {"vanilla": vanilla_logits.argmax(-1).cpu()}
        for name, model in models.items():
            model.to(config.device).eval()
            predictions[name] = predict_episode(model, batch).marginal_probabilities()[:, -1].argmax(-1).cpu()
        for name, predicted in predictions.items():
            for class_index in range(2):
                correct[name][class_index] += ((predicted == class_index) & (labels == class_index)).sum()
    present = total > 0
    return {name: float((values[present].float() / total[present].float()).mean()) for name, values in correct.items()}


def _write_evaluation(
    model: NanoTabPFNSequentialLatentFilter,
    config: SequentialFilterConfig,
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trials, summary, invariance = evaluate_controlled(model, config)
    gate = controlled_gate_report(trials, invariance, stream_count=config.stream_count)
    trials.to_csv(output_dir / f"{prefix}_trajectory_metrics.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    (output_dir / f"{prefix}_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    if config.plots:
        _save_plots(summary, output_dir, prefix)
    return gate, trials, summary


def _load_passed_controlled_checkpoint(
    path: str | Path,
) -> tuple[NanoTabPFNSequentialLatentFilter, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Controlled checkpoint does not exist: {resolved}")
    model, metadata = load_sequential_filter_checkpoint(resolved)
    gate = metadata.get("controlled_gate")
    if not gate or not gate.get("passed", False):
        raise ValueError("TabICL stage requires a controlled checkpoint with a passing saved gate.")
    return model, metadata


def run(config: SequentialFilterConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "sequential_latent_filter" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    histories: list[dict[str, Any]] = []
    evaluated_models: dict[str, NanoTabPFNSequentialLatentFilter] = {}

    if config.stage == "tabicl":
        model, source_metadata = _load_passed_controlled_checkpoint(config.controlled_checkpoint)
        source_hash = source_metadata["source_checkpoint_sha256"]
        controlled_gate = source_metadata["controlled_gate"]
        baseline = init_model_from_state_dict_file(config.checkpoint)
        evaluated_models["controlled"] = copy.deepcopy(model)
    else:
        checkpoint_path = Path(config.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Official checkpoint does not exist: {checkpoint_path}")
        source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        baseline = init_model_from_state_dict_file(str(checkpoint_path))
        model = NanoTabPFNSequentialLatentFilter(copy.deepcopy(baseline))
        model, history, optimizer = train_head(model, config, rng, stage="controlled")
        histories.extend(history)
        controlled_gate, _, _ = _write_evaluation(model, config, output_dir, "controlled")
        save_sequential_filter_checkpoint(
            output_dir / "controlled_checkpoint.pth",
            model,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="controlled",
            controlled_gate=controlled_gate,
            optimizer_state=optimizer.state_dict(),
        )
        evaluated_models["controlled"] = copy.deepcopy(model)

    (output_dir / "source_checkpoint.json").write_text(
        json.dumps(
            {
                "path": str(Path(config.checkpoint).expanduser().resolve()),
                "sha256": source_hash,
            },
            indent=2,
        )
        + "\n"
    )

    ran_tabicl = False
    if config.stage == "tabicl" or (config.stage == "all" and controlled_gate["passed"]):
        model, history, optimizer = train_head(model, config, rng, stage="tabicl")
        histories.extend(history)
        tabicl_gate, _, _ = _write_evaluation(model, config, output_dir, "tabicl")
        save_sequential_filter_checkpoint(
            output_dir / "tabicl_checkpoint.pth",
            model,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="tabicl",
            controlled_gate=controlled_gate,
            optimizer_state=optimizer.state_dict(),
        )
        evaluated_models["tabicl"] = model
        ran_tabicl = True
        (output_dir / "tabicl_gate.json").write_text(json.dumps(tabicl_gate, indent=2) + "\n")

    pd.DataFrame(histories).to_csv(output_dir / "learning_curves.csv", index=False)
    ordinary = evaluate_ordinary_accuracy(evaluated_models, baseline, config)
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary, indent=2) + "\n")
    selection = {
        "controlled_gate_passed": bool(controlled_gate["passed"]),
        "tabicl_requested": config.stage in {"tabicl", "all"},
        "tabicl_ran": ran_tabicl,
        "tabicl_skipped_reason": (
            "controlled_gate_failed" if config.stage == "all" and not controlled_gate["passed"] else None
        ),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("controlled", "tabicl", "all"), default="controlled")
    parser.add_argument("--checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--controlled-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--initial-support-count", type=int, default=16)
    parser.add_argument("--stream-count", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--controlled-steps", type=int, default=1000)
    parser.add_argument("--tabicl-steps", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--diversity-target", type=float, default=0.20)
    parser.add_argument("--evaluation-trials", type=int, default=256)
    parser.add_argument("--ordinary-evaluation-batches", type=int, default=8)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output_dir = run(SequentialFilterConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote sequential latent-filter artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
