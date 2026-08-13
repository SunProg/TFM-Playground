"""Train the static Bayesian nanoTabPFN hypothesis head.

This experiment is deliberately batch-only.  A labelled support table and an
unlabelled query table are encoded once; the head represents uncertainty over
latent task hypotheses.  It has no stream, transition, filtering, update, or
reveal operation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.hypothesis_collapse import all_binary_vectors
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.hypothesis import (
    NanoTabPFNBayesianModel,
    save_bayesian_checkpoint,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class StaticBayesianTrainingConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    num_hypotheses: int = 2
    query_count: int = 4
    batch_size: int = 2
    frozen_steps: int = 300
    full_steps: int = 0
    validation_interval: int = 25
    patience: int = 4
    min_delta: float = 0.0
    learning_rate: float = 1e-4
    accumulate_gradients: int = 1
    support_size: int = 32
    evidence_count: int = 8
    ambiguous_probability: float = 0.5
    identifiable_probability: float = 0.3
    noisy_probability: float = 0.2
    min_features: int = 1
    max_features: int = 16
    prior_type: str = "mix_scm"
    label_noise: float = 0.1
    noisy_label_rate: float = 0.25
    candidate_pool_multiplier: int = 4
    max_candidate_attempts: int = 32
    num_partitions: int = 2
    likelihood_temperature: float = 0.1
    validation_episodes: int = 12
    ordinary_evaluation_batches: int = 8


@dataclass
class StaticEpisodeBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    posterior: torch.Tensor | None = None
    hypothesis_labels: torch.Tensor | None = None
    hypothesis_probabilities: torch.Tensor | None = None
    condition: str = "controlled"

    @property
    def controlled(self) -> bool:
        return self.posterior is not None and (
            self.hypothesis_labels is not None or self.hypothesis_probabilities is not None
        )


def validate_static_config(config: StaticBayesianTrainingConfig) -> None:
    if config.num_hypotheses < 2:
        raise ValueError("num_hypotheses must be at least two.")
    if config.query_count < 1:
        raise ValueError("query_count must be positive.")
    if config.num_hypotheses > 2**config.query_count:
        raise ValueError(
            f"num_hypotheses={config.num_hypotheses} cannot represent distinct binary query hypotheses "
            f"with query_count={config.query_count}; require num_hypotheses <= 2**query_count."
        )
    for name in ("batch_size", "validation_interval", "patience", "accumulate_gradients"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be at least one.")
    if min(config.frozen_steps, config.full_steps) < 0:
        raise ValueError("training steps cannot be negative.")
    if config.min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    probabilities = (config.ambiguous_probability, config.identifiable_probability, config.noisy_probability)
    if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0):
        raise ValueError("curriculum probabilities must be non-negative and sum to one.")
    if config.full_steps != 0:
        raise ValueError("mean-preserving training freezes nanoTabPFN; full_steps must be zero.")
    if not 0 < config.label_noise < 0.5 or not 0 < config.noisy_label_rate < 0.5:
        raise ValueError("label noise rates must lie strictly between zero and 0.5.")
    if config.min_features < 1 or config.max_features < config.min_features:
        raise ValueError("feature bounds are invalid.")
    if config.candidate_pool_multiplier < 2:
        raise ValueError("candidate_pool_multiplier must be at least two.")
    if config.max_candidate_attempts < 1:
        raise ValueError("max_candidate_attempts must be positive.")


def hypothesis_codebook(num_hypotheses: int, query_count: int, *, device=None) -> torch.Tensor:
    """Return K distinct binary query vectors in deterministic canonical order."""
    if num_hypotheses < 1 or num_hypotheses > 2**query_count:
        raise ValueError("num_hypotheses must satisfy 1 <= K <= 2**query_count.")
    return torch.tensor(
        list(itertools.islice(itertools.product((0, 1), repeat=query_count), num_hypotheses)),
        dtype=torch.long,
        device=device,
    )


def _log_posterior_from_evidence(
    evidence_bits: torch.Tensor,
    evidence_indices: torch.Tensor,
    codebook: torch.Tensor,
    noise: float,
) -> torch.Tensor:
    # evidence_bits: B,E; evidence_indices: B,E; codebook: K,Q
    if evidence_bits.shape[1] == 0:
        return torch.zeros(evidence_bits.shape[0], codebook.shape[0], device=evidence_bits.device)
    # Transposing makes the first indexed dimension the query-bit dimension,
    # yielding one selected bit for every (batch, evidence, hypothesis).
    selected = codebook.transpose(0, 1)[evidence_indices.long()]
    # selected is B,E,K.  Each observed bit has a symmetric flip likelihood.
    correct = selected == evidence_bits[:, :, None].long()
    log_correct = math.log1p(-noise)
    log_flip = math.log(noise)
    return torch.where(correct, log_correct, log_flip).sum(dim=1)


def controlled_static_batch(
    config: StaticBayesianTrainingConfig,
    rng: np.random.Generator,
    *,
    support_size: int | None = None,
    evidence_count: int | None = None,
    noise: float = 0.1,
) -> StaticEpisodeBatch:
    """Create episodes with an exact posterior over K latent binary tasks."""
    support_size = config.support_size if support_size is None else support_size
    evidence_count = config.evidence_count if evidence_count is None else evidence_count
    if support_size < 2 or evidence_count < 0 or not 0 <= noise < 0.5:
        raise ValueError("support_size >= 2, evidence_count >= 0, and noise in [0, .5) are required.")
    device = torch.device(config.device)
    codebook = hypothesis_codebook(config.num_hypotheses, config.query_count, device=device)
    latent_index = torch.from_numpy(rng.integers(config.num_hypotheses, size=config.batch_size)).to(device)
    evidence_indices_np = rng.integers(config.query_count, size=(config.batch_size, evidence_count))
    evidence_indices = torch.from_numpy(evidence_indices_np).to(device)
    latent_bits = codebook[latent_index]
    evidence_y_np = latent_bits.gather(1, evidence_indices).cpu().numpy()
    flips = rng.random(evidence_y_np.shape) < noise
    evidence_y_np = np.logical_xor(evidence_y_np, flips).astype(np.float32)

    # The balanced common rows keep the ordinary binary context well formed;
    # evidence rows have feature values that identify which latent query bit
    # was observed.  The posterior uses only support labels and this known
    # likelihood, never query labels.
    common_count = support_size
    common_x = rng.normal(size=(config.batch_size, common_count, 1)).astype(np.float32)
    common_y = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (config.batch_size, (common_count + 1) // 2))[
        :, :common_count
    ]
    evidence_x = ((evidence_indices_np + 1) / max(config.query_count, 1)).astype(np.float32)[..., None]
    support_x = torch.from_numpy(np.concatenate((common_x, evidence_x), axis=1)).to(device)
    support_y = torch.from_numpy(np.concatenate((common_y, evidence_y_np), axis=1)).to(device)
    query_values = np.linspace(0.25, 1.0, config.query_count, dtype=np.float32)
    query_x = torch.from_numpy(
        np.broadcast_to(query_values, (config.batch_size, config.query_count)).copy()[..., None]
    ).to(device)
    query_y = latent_bits
    log_likelihood = _log_posterior_from_evidence(
        torch.from_numpy(evidence_y_np).to(device), evidence_indices, codebook, noise
    )
    posterior = log_likelihood.log_softmax(dim=-1).exp()
    hypothesis_labels = codebook[None].expand(config.batch_size, -1, -1)
    return StaticEpisodeBatch(
        support_x,
        support_y,
        query_x,
        query_y,
        posterior,
        hypothesis_labels,
        hypothesis_labels.float(),
        "codebook",
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _tabicl_candidate_pool(
    config: StaticBayesianTrainingConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample K structured SCM tasks on shared feature rows for each episode."""
    from tabicl.prior._dataset import SCMPrior
    from tabicl.prior._mlp_scm import MLPSCM
    from tabicl.prior._reg2cls import Reg2Cls
    from tabicl.prior._tree_scm import TreeSCM

    from tfmplayground.experiments.prior_bimodal_episodes import (
        PriorBimodalConfig,
        _rng_state,
        _sample_params,
        _set_rng_state,
    )

    pool_rows = max(
        128,
        config.candidate_pool_multiplier * (config.support_size + config.query_count),
    )
    features = int(rng.integers(config.min_features, config.max_features + 1))
    episode_config = PriorBimodalConfig(
        initial_support_count=config.support_size,
        stream_count=0,
        query_count=config.query_count,
        min_features=features,
        max_features=features,
        device=config.device,
        prior_type=config.prior_type,
    )
    prior = SCMPrior(
        batch_size=1,
        min_features=features,
        max_features=features,
        max_classes=2,
        min_seq_len=pool_rows,
        max_seq_len=pool_rows + 1,
        min_train_size=config.support_size,
        max_train_size=config.support_size + 1,
        prior_type=config.prior_type,
        n_jobs=1,
        device=config.device,
    )
    outer_state = _rng_state()
    examples_x: list[torch.Tensor] = []
    examples_y: list[torch.Tensor] = []
    try:
        for _batch in range(config.batch_size):
            for _attempt in range(config.max_candidate_attempts):
                _seed_all(int(rng.integers(0, 2**31 - 1)))
                params = _sample_params(prior, episode_config, pool_rows, features)
                prior_cls = MLPSCM if params["prior_type"] == "mlp_scm" else TreeSCM
                candidate_models = []
                for _candidate in range(config.num_hypotheses):
                    _seed_all(int(rng.integers(0, 2**31 - 1)))
                    with torch.no_grad():
                        candidate_models.append(prior_cls(**params))

                # Sample covariates exactly once.  Each independently initialized
                # SCM below is a different latent task evaluated on this same
                # table, rather than a separate dataset whose normalized feature
                # values merely happen to compare equal.
                _seed_all(int(rng.integers(0, 2**31 - 1)))
                with torch.no_grad():
                    shared_raw_x = candidate_models[0].xsampler.sample()

                processed_x = None
                candidate_y = []
                for model in candidate_models:
                    _seed_all(int(rng.integers(0, 2**31 - 1)))
                    with torch.no_grad():
                        y_value = model.layers(shared_raw_x)
                        if y_value.shape[-1] == 1:
                            y_value = y_value.squeeze(-1)
                        x_value, y_value = Reg2Cls(params)(shared_raw_x.clone(), y_value)
                    if processed_x is None:
                        processed_x = x_value.float()
                    candidate_y.append(y_value)
                if processed_x is not None and torch.isfinite(processed_x).all() and all(
                    torch.isfinite(value).all() and value.unique().numel() == 2 for value in candidate_y
                ):
                    examples_x.append(processed_x)
                    examples_y.append(torch.stack(candidate_y))
                    break
            else:
                raise RuntimeError(
                    "Could not draw finite binary TabICL candidates within max_candidate_attempts."
                )
    finally:
        _set_rng_state(outer_state)
    return torch.stack(examples_x), torch.stack(examples_y)


def _exact_candidate_posterior(
    observed: torch.Tensor,
    candidate_labels: torch.Tensor,
    noise: float,
) -> torch.Tensor:
    matches = candidate_labels == observed[:, None, :].long()
    log_likelihood = torch.where(
        matches,
        torch.full_like(candidate_labels, math.log1p(-noise), dtype=torch.float32),
        torch.full_like(candidate_labels, math.log(noise), dtype=torch.float32),
    ).sum(dim=-1)
    return log_likelihood.softmax(dim=-1)


def structured_static_batch(
    config: StaticBayesianTrainingConfig,
    rng: np.random.Generator,
    *,
    condition: str,
) -> StaticEpisodeBatch:
    """Create structured ambiguous, identifiable, or noisy TabICL episodes."""
    if condition not in {"ambiguous", "identifiable", "noisy"}:
        raise ValueError("condition must be ambiguous, identifiable, or noisy.")
    pool_x, candidates = _tabicl_candidate_pool(config, rng)
    device = torch.device(config.device)
    support_x_values = []
    query_x_values = []
    support_candidates = []
    query_candidates = []
    for batch in range(config.batch_size):
        labels = candidates[batch]
        if condition == "noisy":
            labels = labels[:1].expand(config.num_hypotheses, -1).clone()
        disagreement = labels.float().var(dim=0, unbiased=False)
        if condition == "ambiguous":
            order = disagreement.argsort(descending=False)
            support_indices = order[: config.support_size]
            remaining = order[config.support_size :]
            query_indices = remaining[disagreement[remaining].argsort(descending=True)[: config.query_count]]
        elif condition == "identifiable":
            order = disagreement.argsort(descending=True)
            support_indices = order[: config.support_size]
            query_indices = order[config.support_size : config.support_size + config.query_count]
        else:
            order = torch.randperm(pool_x.shape[1], generator=torch.Generator().manual_seed(int(rng.integers(2**31))))
            support_indices = order[: config.support_size]
            query_indices = order[config.support_size : config.support_size + config.query_count]
        support_x_values.append(pool_x[batch, support_indices])
        query_x_values.append(pool_x[batch, query_indices])
        support_candidates.append(labels[:, support_indices])
        query_candidates.append(labels[:, query_indices])

    support_x = torch.stack(support_x_values).to(device).float()
    query_x = torch.stack(query_x_values).to(device).float()
    candidate_support = torch.stack(support_candidates).to(device).long()
    candidate_query = torch.stack(query_candidates).to(device).long()
    true_candidate = torch.from_numpy(rng.integers(config.num_hypotheses, size=config.batch_size)).to(device)
    noise = config.noisy_label_rate if condition == "noisy" else config.label_noise
    support_clean = candidate_support[torch.arange(config.batch_size, device=device), true_candidate]
    query_clean = candidate_query[torch.arange(config.batch_size, device=device), true_candidate]
    support_flips = torch.from_numpy(rng.random(support_clean.shape) < noise).to(device)
    query_flips = torch.from_numpy(rng.random(query_clean.shape) < noise).to(device)
    support_y = support_clean.logical_xor(support_flips).float()
    query_y = query_clean.logical_xor(query_flips).long()
    posterior = _exact_candidate_posterior(support_y, candidate_support, noise)
    candidate_probabilities = noise + (1.0 - 2.0 * noise) * candidate_query.float()
    return StaticEpisodeBatch(
        support_x,
        support_y,
        query_x,
        query_y,
        posterior,
        candidate_query,
        candidate_probabilities,
        condition,
    )


def ordinary_static_batch(config: StaticBayesianTrainingConfig, rng: np.random.Generator) -> StaticEpisodeBatch:
    """Compatibility name for an ordinary structured, identifiable SCM batch."""
    return structured_static_batch(config, rng, condition="identifiable")


def random_label_diagnostic_batch(
    config: StaticBayesianTrainingConfig,
    rng: np.random.Generator,
) -> StaticEpisodeBatch:
    """No-signal labels retained solely for aleatoric diagnostic tests."""
    device = torch.device(config.device)
    support_x = torch.from_numpy(
        rng.normal(size=(config.batch_size, config.support_size, 2)).astype(np.float32)
    ).to(device)
    query_x = torch.from_numpy(
        rng.normal(size=(config.batch_size, config.query_count, 2)).astype(np.float32)
    ).to(device)
    support_y = torch.from_numpy(rng.integers(0, 2, size=support_x.shape[:2]).astype(np.float32)).to(device)
    query_y = torch.from_numpy(rng.integers(0, 2, size=query_x.shape[:2])).to(device)
    return StaticEpisodeBatch(support_x, support_y, query_x, query_y, condition="random_diagnostic")


def _target_joint(posterior: torch.Tensor, hypothesis_probabilities: torch.Tensor) -> torch.Tensor:
    batch, hypotheses, query_count = hypothesis_probabilities.shape
    outcomes = all_binary_vectors(query_count)
    outcome_tensor = torch.as_tensor(outcomes, device=posterior.device, dtype=posterior.dtype)
    probabilities = hypothesis_probabilities[:, :, None, :].clamp(1e-6, 1 - 1e-6)
    selected = outcome_tensor[None, None] * probabilities + (1 - outcome_tensor[None, None]) * (1 - probabilities)
    per_hypothesis = selected.prod(dim=-1)
    return (posterior[:, :, None] * per_hypothesis).sum(dim=1)


def _assignment(prediction, target_probabilities: torch.Tensor) -> torch.Tensor:
    """Match predicted slots to target hypotheses using a permutation-invariant cost."""
    # Cost uses query-label cross entropy.  Assignment is intentionally a
    # discrete operation; gradients flow through the selected matched loss.
    raw = getattr(prediction, "raw_query_probabilities", prediction.query_probabilities)
    predicted = raw[..., 1].detach().transpose(1, 2)
    target = target_probabilities.float()[:, :, None, :]
    slot = predicted[:, None, :, :]
    cost = (slot - target).square().mean(dim=-1)  # B,target,predicted
    assignments = []
    for matrix in cost.cpu().numpy():
        try:
            from scipy.optimize import linear_sum_assignment

            _, columns = linear_sum_assignment(matrix)
            assignments.append(columns)
        except ImportError:
            # SciPy is transitively available with scikit-learn, but retain a
            # tiny deterministic fallback for minimal unit-test environments.
            assignments.append(
                np.asarray(
                    min(
                        itertools.permutations(range(matrix.shape[1])),
                        key=lambda p: sum(matrix[i, p[i]] for i in range(matrix.shape[0])),
                    )
                )
            )
    return torch.as_tensor(np.asarray(assignments), device=target_probabilities.device, dtype=torch.long)


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1 - 1e-6)
    return -(probability * probability.log() + (1 - probability) * (1 - probability).log())


def static_bayesian_loss(
    model: NanoTabPFNBayesianModel,
    batch: StaticEpisodeBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
    prediction = model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])
    query_count = batch.query_x.shape[1]
    if batch.controlled:
        if batch.hypothesis_probabilities is not None:
            candidate_positive = batch.hypothesis_probabilities.float()
        else:
            candidate_positive = batch.hypothesis_labels.float()
        assignment = _assignment(prediction, candidate_positive)
        target_joint = _target_joint(batch.posterior, candidate_positive)
        joint_loss = -(target_joint * prediction.joint_probabilities().clamp_min(1e-12).log()).sum(-1).mean()
        target_joint_entropy = -(target_joint * target_joint.clamp_min(1e-12).log()).sum(-1).mean()
        joint_kl = (joint_loss - target_joint_entropy).clamp_min(0)
        target_binary = torch.stack((1 - candidate_positive, candidate_positive), dim=-1).permute(0, 2, 1, 3)
        raw_logits = getattr(prediction, "raw_slot_logits", prediction.slot_logits)
        matched_logits = raw_logits.gather(
            2,
            assignment[:, None, :, None].expand(-1, query_count, -1, 2),
        )
        # Average over batch, queries, and hypotheses. A sum over hypotheses
        # would make both training scale and validation ranking depend on K.
        slot_loss = -(target_binary * F.log_softmax(matched_logits, dim=-1)).sum(dim=-1).mean()
        target_slot_entropy = -(target_binary * target_binary.clamp_min(1e-12).log()).sum(dim=-1).mean()
        slot_kl = (slot_loss - target_slot_entropy).clamp_min(0)
        matched_weights = prediction.slot_log_weights.gather(1, assignment)
        weight_loss = -(batch.posterior * matched_weights).sum(-1).mean()
        target_weight_entropy = -(batch.posterior * batch.posterior.clamp_min(1e-12).log()).sum(-1).mean()
        weight_kl = (weight_loss - target_weight_entropy).clamp_min(0)
        candidate_mean = (batch.posterior[:, :, None] * candidate_positive).sum(dim=1)
        target_expected_entropy = (
            batch.posterior[:, :, None] * _binary_entropy(candidate_positive)
        ).sum(dim=1)
        target_mi = (_binary_entropy(candidate_mean) - target_expected_entropy).clamp_min(0)
        log_two = math.log(2.0)
        mi_loss = F.mse_loss(prediction.mutual_information() / log_two, target_mi / log_two)
        target_variance = (
            batch.posterior[:, :, None] * (candidate_positive - candidate_mean[:, None]).square()
        ).sum(dim=1)
        variance_loss = F.mse_loss(prediction.epistemic_variance(), target_variance)
    else:
        raise ValueError("No-signal diagnostic batches are not valid training batches.")
    total = joint_loss + 0.5 * weight_loss + 0.5 * slot_loss + 0.25 * mi_loss + 0.25 * variance_loss
    # Cross-entropies include target entropy, which increases mechanically with
    # K. Subtract it for validation/model selection so different hypothesis
    # counts are ranked by approximation error rather than target complexity.
    log_two = math.log(2.0)
    joint_kl_normalized = joint_kl / (query_count * log_two)
    weight_kl_normalized = weight_kl / math.log(candidate_positive.shape[1])
    slot_kl_normalized = slot_kl / log_two
    selection_loss = (
        joint_kl_normalized
        + 0.5 * weight_kl_normalized
        + 0.5 * slot_kl_normalized
        + 0.25 * mi_loss
        + 0.25 * variance_loss
    )
    return total, {
        "loss": float(total.detach()),
        "selection_loss": float(selection_loss.detach()),
        "joint_nll": float(joint_loss.detach()),
        "joint_kl": float(joint_kl.detach()),
        "joint_kl_normalized": float(joint_kl_normalized.detach()),
        "weight_loss": float(weight_loss.detach()),
        "weight_kl": float(weight_kl.detach()),
        "weight_kl_normalized": float(weight_kl_normalized.detach()),
        "slot_loss": float(slot_loss.detach()),
        "slot_kl": float(slot_kl.detach()),
        "slot_kl_normalized": float(slot_kl_normalized.detach()),
        "mi_loss": float(mi_loss.detach()),
        "variance_loss": float(variance_loss.detach()),
        "mean_preservation_error": float(prediction.mean_preservation_error().max().detach()),
    }


def _validation_loss(model: NanoTabPFNBayesianModel, config: StaticBayesianTrainingConfig) -> float:
    model.eval()
    rng = np.random.default_rng(config.seed + 10_001)
    losses = []
    with torch.no_grad():
        for index in range(config.validation_episodes):
            condition = ("ambiguous", "identifiable", "noisy")[index % 3]
            batch = structured_static_batch(config, rng, condition=condition)
            losses.append(static_bayesian_loss(model, batch)[1]["selection_loss"])
    model.train()
    return float(np.mean(losses))


def _train_stage(
    model: NanoTabPFNBayesianModel,
    config: StaticBayesianTrainingConfig,
    steps: int,
) -> tuple[NanoTabPFNBayesianModel, list[dict[str, Any]], float]:
    model.freeze_backbone()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    rng = np.random.default_rng(config.seed + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    model.to(config.device)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _micro in range(config.accumulate_gradients):
            draw = rng.random()
            if draw < config.ambiguous_probability:
                condition = "ambiguous"
            elif draw < config.ambiguous_probability + config.identifiable_probability:
                condition = "identifiable"
            else:
                condition = "noisy"
            batch = structured_static_batch(config, rng, condition=condition)
            loss, metrics = static_bayesian_loss(model, batch)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        row = {"stage": "mean_preserving_frozen", "step": step, **totals}
        if step % config.validation_interval == 0 or step == steps:
            row["validation_normalized_loss"] = _validation_loss(model, config)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["validation_normalized_loss"] < best_validation - config.min_delta:
                best_validation = row["validation_normalized_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_validation


def train_frozen_stage(model: NanoTabPFNBayesianModel, config: StaticBayesianTrainingConfig):
    """Stage 1: only Bayesian hypothesis-head parameters receive gradients."""
    return _train_stage(model, config, config.frozen_steps)


def train_full_stage(model: NanoTabPFNBayesianModel, config: StaticBayesianTrainingConfig):
    """Rejected compatibility entry point: this model must preserve the frozen mean."""
    del model, config
    raise RuntimeError("Full fine-tuning is disabled for mean-preserving Bayesian uncertainty training.")


@torch.no_grad()
def _ordinary_accuracy(model: NanoTabPFNModel | NanoTabPFNBayesianModel, config: StaticBayesianTrainingConfig) -> float:
    rng = np.random.default_rng(config.seed + 20_001)
    correct = np.zeros(2, dtype=np.int64)
    total = np.zeros(2, dtype=np.int64)
    model.to(config.device)
    model.eval()
    for _ in range(config.ordinary_evaluation_batches):
        batch = ordinary_static_batch(config, rng)
        if isinstance(model, NanoTabPFNBayesianModel):
            probabilities = model(batch.support_x, batch.support_y, batch.query_x).marginal_probabilities()
        else:
            probabilities = model(batch.support_x, batch.support_y, batch.query_x)[..., :2].softmax(-1)
        predicted = probabilities.argmax(-1).cpu().numpy()
        labels = batch.query_y.cpu().numpy()
        for class_index in range(2):
            total[class_index] += int((labels == class_index).sum())
            correct[class_index] += int(((predicted == class_index) & (labels == class_index)).sum())
    present = total > 0
    return float(np.mean(correct[present] / total[present])) if present.any() else 0.0


@torch.no_grad()
def _uncertainty_report(
    model: NanoTabPFNBayesianModel,
    config: StaticBayesianTrainingConfig,
) -> dict[str, dict[str, float]]:
    model.eval()
    rng = np.random.default_rng(config.seed + 30_001)
    report: dict[str, dict[str, float]] = {}
    for condition in ("ambiguous", "identifiable", "noisy"):
        values: dict[str, list[float]] = {
            "mutual_information": [],
            "expected_hypothesis_entropy": [],
            "epistemic_variance": [],
            "effective_sample_size": [],
            "posterior_nll": [],
            "mean_preservation_error": [],
        }
        for _ in range(max(1, config.ordinary_evaluation_batches // 2)):
            batch = structured_static_batch(config, rng, condition=condition)
            prediction = model(batch.support_x, batch.support_y, batch.query_x)
            assignment = _assignment(prediction, batch.hypothesis_probabilities)
            matched_weights = prediction.slot_log_weights.gather(1, assignment)
            values["mutual_information"].append(float(prediction.mutual_information().mean()))
            values["expected_hypothesis_entropy"].append(
                float(prediction.expected_hypothesis_entropy().mean())
            )
            values["epistemic_variance"].append(float(prediction.epistemic_variance().mean()))
            values["effective_sample_size"].append(float(prediction.effective_sample_size().mean()))
            values["posterior_nll"].append(float(-(batch.posterior * matched_weights).sum(-1).mean()))
            values["mean_preservation_error"].append(float(prediction.mean_preservation_error().max()))
        report[condition] = {name: float(np.mean(metric_values)) for name, metric_values in values.items()}

    diagnostic = random_label_diagnostic_batch(config, rng)
    prediction = model(diagnostic.support_x, diagnostic.support_y, diagnostic.query_x)
    report["random_label_diagnostic"] = {
        "mutual_information": float(prediction.mutual_information().mean()),
        "expected_hypothesis_entropy": float(prediction.expected_hypothesis_entropy().mean()),
        "epistemic_variance": float(prediction.epistemic_variance().mean()),
        "effective_sample_size": float(prediction.effective_sample_size().mean()),
        "mean_preservation_error": float(prediction.mean_preservation_error().max()),
    }
    return report


def run_training(config: StaticBayesianTrainingConfig) -> Path:
    validate_static_config(config)
    set_randomness_seed(config.seed)
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    output = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "bayesian_nanotabpfn" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    baseline = init_model_from_state_dict_file(str(checkpoint_path))
    vanilla_reference = copy.deepcopy(baseline)
    model = NanoTabPFNBayesianModel(
        baseline,
        num_hypotheses=config.num_hypotheses,
        num_partitions=config.num_partitions,
        likelihood_temperature=config.likelihood_temperature,
    )
    source_backbone = {name: value.detach().cpu().clone() for name, value in model.backbone.state_dict().items()}
    model, frozen_history, frozen_nll = train_frozen_stage(model, config)
    selected_path = output / "mean_preserving_frozen.pth"
    save_bayesian_checkpoint(
        selected_path,
        model,
        training_config=asdict(config),
        source_checkpoint_sha256=source_hash,
        stage="mean_preserving_frozen",
    )
    frozen_ordinary_accuracy = _ordinary_accuracy(model, config)
    baseline_ordinary_accuracy = _ordinary_accuracy(vanilla_reference, config)
    uncertainty = _uncertainty_report(model, config)
    backbone_unchanged = all(
        torch.equal(source_backbone[name], value.detach().cpu())
        for name, value in model.backbone.state_dict().items()
    )
    max_mean_error = max(
        metrics["mean_preservation_error"] for metrics in uncertainty.values()
    )
    acceptance = {
        "finite_validation_loss": bool(np.isfinite(frozen_nll)),
        "backbone_unchanged": backbone_unchanged,
        "mean_preserved_at_1e-6": max_mean_error <= 1e-6,
        "ordinary_accuracy_identical": abs(frozen_ordinary_accuracy - baseline_ordinary_accuracy) <= 1e-6,
        "ambiguous_mi_exceeds_identifiable": (
            uncertainty["ambiguous"]["mutual_information"]
            > uncertainty["identifiable"]["mutual_information"]
        ),
        "ambiguous_mi_exceeds_noisy": (
            uncertainty["ambiguous"]["mutual_information"] > uncertainty["noisy"]["mutual_information"]
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    (output / "learning_curves.csv").write_text(_history_csv(frozen_history))
    (output / "source_checkpoint.json").write_text(
        json.dumps({"path": str(checkpoint_path), "sha256": source_hash}, indent=2) + "\n"
    )
    (output / "selection.json").write_text(
        json.dumps(
            {
                "selected_stage": "mean_preserving_frozen",
                "selected_checkpoint": str(selected_path),
                "validation_loss": frozen_nll,
                "full_stage_run": False,
                "ordinary_balanced_accuracy": {
                    "baseline": baseline_ordinary_accuracy,
                    "frozen": frozen_ordinary_accuracy,
                },
                "uncertainty": uncertainty,
                "acceptance": acceptance,
            },
            indent=2,
        )
        + "\n"
    )
    return output.resolve()


def _history_csv(rows: list[dict[str, Any]]) -> str:
    keys = sorted({key for row in rows for key in row})
    return (
        ",".join(keys)
        + "\n"
        + "\n".join(",".join(str(row.get(key, "")) for key in keys) for row in rows)
        + ("\n" if rows else "")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = StaticBayesianTrainingConfig()
    for field in (
        "seed",
        "num_hypotheses",
        "query_count",
        "batch_size",
        "frozen_steps",
        "full_steps",
        "validation_interval",
        "patience",
        "accumulate_gradients",
        "support_size",
        "evidence_count",
        "min_features",
        "max_features",
        "candidate_pool_multiplier",
        "max_candidate_attempts",
        "num_partitions",
        "validation_episodes",
        "ordinary_evaluation_batches",
    ):
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=int, default=getattr(defaults, field)
        )
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--ambiguous-probability", type=float, default=defaults.ambiguous_probability)
    parser.add_argument("--identifiable-probability", type=float, default=defaults.identifiable_probability)
    parser.add_argument("--noisy-probability", type=float, default=defaults.noisy_probability)
    parser.add_argument("--prior-type", default=defaults.prior_type)
    parser.add_argument("--label-noise", type=float, default=defaults.label_noise)
    parser.add_argument("--noisy-label-rate", type=float, default=defaults.noisy_label_rate)
    parser.add_argument("--likelihood-temperature", type=float, default=defaults.likelihood_temperature)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Wrote static Bayesian training artifacts to {run_training(StaticBayesianTrainingConfig(**vars(args)))}")
