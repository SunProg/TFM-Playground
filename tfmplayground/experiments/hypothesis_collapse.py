"""Diagnose whether nanoTabPFN preserves coherent latent-task hypotheses.

Run with::

    python -m tfmplayground.experiments.hypothesis_collapse
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.interface import NanoTabPFNClassifier, init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device

PROBABILITY_EPSILON = 1e-12
SUPPORTED_QUERY_COUNTS = (2, 3, 4)
METRIC_NAMES = (
    "marginal_cross_entropy",
    "marginal_js",
    "joint_cross_entropy",
    "joint_js",
    "incoherent_mass",
    "order_inconsistency",
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2402
    trials: int = 32
    query_counts: tuple[int, ...] = (2, 3, 4)
    evidence_counts: tuple[int, ...] = (0, 1, 2, 4, 8)
    common_support_size: int = 16
    common_support_sizes: tuple[int, ...] | None = None
    support_noise: float = 0.1
    models: tuple[str, ...] = ("exact", "independent", "nano")
    device: str = "cpu"
    num_mem_chunks: int = 8
    checkpoint: str | None = None
    output_dir: str | None = None
    plots: bool = True


@dataclass(frozen=True)
class SyntheticTaskBatch:
    """A batch of tasks with equal support/query shapes."""

    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    latent_task: np.ndarray
    posterior_b: np.ndarray
    evidence_y: np.ndarray


class NanoTabPFNBinaryPredictor:
    """Experiment-only raw predictor that always retains the first two logits."""

    def __init__(self, model: NanoTabPFNModel, device: str | torch.device, num_mem_chunks: int = 8):
        if model.num_outputs < 2:
            raise ValueError(f"Checkpoint must expose at least two outputs, found {model.num_outputs}.")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.num_mem_chunks = num_mem_chunks

    @torch.no_grad()
    def predict_binary_proba(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray,
    ) -> np.ndarray:
        """Predict both binary classes for a batch of independent ICL contexts."""
        support_x = np.asarray(support_x, dtype=np.float32)
        support_y = np.asarray(support_y, dtype=np.float32)
        query_x = np.asarray(query_x, dtype=np.float32)
        if support_x.ndim != 3 or query_x.ndim != 3 or support_y.ndim != 2:
            raise ValueError("Expected support_x/query_x with 3 dimensions and support_y with 2 dimensions.")
        if support_x.shape[0] != support_y.shape[0] or support_x.shape[0] != query_x.shape[0]:
            raise ValueError("Support and query batches must have the same batch dimension.")
        if support_x.shape[1] != support_y.shape[1]:
            raise ValueError("support_x and support_y must contain the same number of rows.")
        if support_x.shape[2] != query_x.shape[2]:
            raise ValueError("Support and query feature counts must match.")

        x = torch.from_numpy(np.concatenate((support_x, query_x), axis=1)).to(self.device)
        y = torch.from_numpy(support_y).to(self.device)
        logits = self.model(
            (x, y),
            train_test_split_index=support_x.shape[1],
            num_mem_chunks=self.num_mem_chunks,
        )
        probabilities = F.softmax(logits[..., :2], dim=-1)
        return probabilities.cpu().numpy()


def validate_config(config: ExperimentConfig) -> None:
    if config.trials < 1:
        raise ValueError("--trials must be at least 1.")
    support_sizes = effective_common_support_sizes(config)
    if len(set(support_sizes)) != len(support_sizes):
        raise ValueError("--common-support-sizes cannot contain duplicates.")
    if any(size < 2 or size % 2 for size in support_sizes):
        raise ValueError("Common support sizes must be positive even numbers of at least 2.")
    if not 0 <= config.support_noise < 0.5:
        raise ValueError("--support-noise must be in [0, 0.5).")
    if config.num_mem_chunks < 1:
        raise ValueError("--num-mem-chunks must be at least 1.")
    if len(set(config.query_counts)) != len(config.query_counts):
        raise ValueError("--query-counts cannot contain duplicates.")
    unsupported = sorted(set(config.query_counts) - set(SUPPORTED_QUERY_COUNTS))
    if unsupported:
        raise ValueError(f"Unsupported query counts {unsupported}; choose from {SUPPORTED_QUERY_COUNTS}.")
    if len(set(config.evidence_counts)) != len(config.evidence_counts):
        raise ValueError("--evidence-counts cannot contain duplicates.")
    if any(value < 0 for value in config.evidence_counts):
        raise ValueError("--evidence-counts must be non-negative.")
    if not config.models:
        raise ValueError("--models must contain at least one model.")
    unknown_models = sorted(set(config.models) - {"exact", "independent", "nano"})
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def effective_common_support_sizes(config: ExperimentConfig) -> tuple[int, ...]:
    """Return the support-size sweep, falling back to the original scalar setting."""
    if config.common_support_sizes is None:
        return (config.common_support_size,)
    if not config.common_support_sizes:
        raise ValueError("--common-support-sizes must contain at least one value.")
    return config.common_support_sizes


def all_binary_vectors(query_count: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0, 1), repeat=query_count)), dtype=np.int8)


def _posterior_b_from_evidence(evidence_y: np.ndarray, support_noise: float) -> np.ndarray:
    """Compute P(z=B | evidence) under equal priors in log space."""
    evidence_y = np.asarray(evidence_y, dtype=np.int8)
    ones = evidence_y.sum(axis=1)
    zeros = evidence_y.shape[1] - ones

    if evidence_y.shape[1] == 0:
        return np.full(evidence_y.shape[0], 0.5)
    if support_noise == 0:
        if np.any((ones > 0) & (zeros > 0)):
            raise ValueError("Mixed evidence has zero likelihood under both noiseless latent tasks.")
        return (ones > 0).astype(np.float64)

    log_correct = math.log1p(-support_noise)
    log_flip = math.log(support_noise)
    log_a = math.log(0.5) + zeros * log_correct + ones * log_flip
    log_b = math.log(0.5) + ones * log_correct + zeros * log_flip
    maximum = np.maximum(log_a, log_b)
    weight_a = np.exp(log_a - maximum)
    weight_b = np.exp(log_b - maximum)
    return weight_b / (weight_a + weight_b)


def generate_global_ambiguity_tasks(
    *,
    trials: int,
    query_count: int,
    evidence_count: int,
    common_support_size: int,
    support_noise: float,
    rng: np.random.Generator,
) -> SyntheticTaskBatch:
    """Generate tasks whose query labels are all zero (A) or all one (B)."""
    if query_count not in SUPPORTED_QUERY_COUNTS:
        raise ValueError(f"query_count must be one of {SUPPORTED_QUERY_COUNTS}.")
    if common_support_size < 2 or common_support_size % 2:
        raise ValueError("common_support_size must be an even number of at least 2.")
    if evidence_count < 0:
        raise ValueError("evidence_count must be non-negative.")
    if not 0 <= support_noise < 0.5:
        raise ValueError("support_noise must be in [0, 0.5).")

    half = common_support_size // 2
    negative_class_one = rng.uniform(-2.0, -1.25, size=(trials, half, 1))
    negative_class_zero = rng.uniform(-0.75, -0.25, size=(trials, half, 1))
    common_x = np.concatenate((negative_class_one, negative_class_zero), axis=1)
    common_y = np.concatenate((np.ones((trials, half)), np.zeros((trials, half))), axis=1)

    # Shuffle common rows independently without changing their guaranteed class balance.
    for trial_index in range(trials):
        permutation = rng.permutation(common_support_size)
        common_x[trial_index] = common_x[trial_index, permutation]
        common_y[trial_index] = common_y[trial_index, permutation]

    latent_task = rng.integers(0, 2, size=trials, dtype=np.int8)
    evidence_x = rng.uniform(0.05, 1.25, size=(trials, evidence_count, 1))
    evidence_y = np.repeat(latent_task[:, None], evidence_count, axis=1)
    if evidence_count:
        flips = rng.random((trials, evidence_count)) < support_noise
        evidence_y = np.logical_xor(evidence_y, flips).astype(np.int8)

    support_x = np.concatenate((common_x, evidence_x), axis=1).astype(np.float32)
    support_y = np.concatenate((common_y, evidence_y), axis=1).astype(np.float32)
    query_values = np.linspace(0.25, 1.0, query_count, dtype=np.float32)
    query_x = np.broadcast_to(query_values[None, :, None], (trials, query_count, 1)).copy()
    query_y = np.repeat(latent_task[:, None], query_count, axis=1).astype(np.int8)
    posterior_b = _posterior_b_from_evidence(evidence_y, support_noise)
    return SyntheticTaskBatch(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        latent_task=latent_task,
        posterior_b=posterior_b,
        evidence_y=evidence_y,
    )


def exact_joint_distribution(posterior_b: np.ndarray, query_count: int) -> np.ndarray:
    posterior_b = np.asarray(posterior_b, dtype=np.float64)
    joint = np.zeros((posterior_b.shape[0], 2**query_count), dtype=np.float64)
    joint[:, 0] = 1.0 - posterior_b
    joint[:, -1] = posterior_b
    return joint


def independent_joint_distribution(posterior_b: np.ndarray, query_count: int) -> np.ndarray:
    posterior_b = np.asarray(posterior_b, dtype=np.float64)
    outcomes = all_binary_vectors(query_count)
    ones = outcomes.sum(axis=1)[None, :]
    probabilities = posterior_b[:, None]
    return probabilities**ones * (1.0 - probabilities) ** (query_count - ones)


def _sequence_to_canonical_indices(order: Sequence[int]) -> np.ndarray:
    query_count = len(order)
    sequence_vectors = all_binary_vectors(query_count)
    canonical_indices = np.empty(2**query_count, dtype=np.int64)
    powers = 2 ** np.arange(query_count - 1, -1, -1)
    for sequence_index, sequence_bits in enumerate(sequence_vectors):
        canonical_bits = np.zeros(query_count, dtype=np.int8)
        canonical_bits[np.asarray(order)] = sequence_bits
        canonical_indices[sequence_index] = int(canonical_bits @ powers)
    return canonical_indices


def enumerate_chain_joint(
    predictor,
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    order: Sequence[int],
    *,
    context: str = "chain enumeration",
) -> np.ndarray:
    """Enumerate a chain-rule joint while batching all prefixes at each depth."""
    support_x = np.asarray(support_x, dtype=np.float32)
    support_y = np.asarray(support_y, dtype=np.float32)
    query_x = np.asarray(query_x, dtype=np.float32)
    trials, query_count, feature_count = query_x.shape
    order = tuple(order)
    if sorted(order) != list(range(query_count)):
        raise ValueError(f"order must be a permutation of 0..{query_count - 1}, got {order}.")

    branch_probabilities = np.ones((trials, 1), dtype=np.float64)
    for depth, query_index in enumerate(order):
        branch_count = 2**depth
        prefix_vectors = all_binary_vectors(depth) if depth else np.zeros((1, 0), dtype=np.int8)
        repeated_support_x = np.repeat(support_x[:, None], branch_count, axis=1)
        repeated_support_y = np.repeat(support_y[:, None], branch_count, axis=1)

        if depth:
            conditioned_x = query_x[:, np.asarray(order[:depth]), :]
            conditioned_x = np.repeat(conditioned_x[:, None], branch_count, axis=1)
            conditioned_y = np.broadcast_to(prefix_vectors[None, :, :], (trials, branch_count, depth))
            context_x = np.concatenate((repeated_support_x, conditioned_x), axis=2)
            context_y = np.concatenate((repeated_support_y, conditioned_y), axis=2)
        else:
            context_x = repeated_support_x
            context_y = repeated_support_y

        next_query = query_x[:, query_index : query_index + 1, :]
        next_query = np.repeat(next_query[:, None], branch_count, axis=1)
        flat_context_x = context_x.reshape(trials * branch_count, context_x.shape[2], feature_count)
        flat_context_y = context_y.reshape(trials * branch_count, context_y.shape[2])
        flat_next_query = next_query.reshape(trials * branch_count, 1, feature_count)
        conditional = predictor.predict_binary_proba(flat_context_x, flat_context_y, flat_next_query)
        conditional = np.asarray(conditional, dtype=np.float64)
        if conditional.shape != (trials * branch_count, 1, 2):
            raise ValueError(
                f"{context}: predictor returned {conditional.shape}, expected "
                f"{(trials * branch_count, 1, 2)} at depth {depth}."
            )
        conditional = conditional[:, 0, :].reshape(trials, branch_count, 2)
        _validate_probability_rows(conditional.reshape(-1, 2), f"{context}, depth={depth}")
        branch_probabilities = (branch_probabilities[:, :, None] * conditional).reshape(trials, branch_count * 2)

    canonical = np.empty_like(branch_probabilities)
    canonical[:, _sequence_to_canonical_indices(order)] = branch_probabilities
    validate_joint_distribution(canonical, context)
    return canonical


def _validate_probability_rows(probabilities: np.ndarray, context: str) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {context}.")
    if (probabilities < -1e-7).any() or (probabilities > 1 + 1e-7).any():
        raise ValueError(f"Probabilities outside [0, 1] in {context}.")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"Probabilities do not sum to one in {context}; range={sums.min()}..{sums.max()}.")


def validate_joint_distribution(distribution: np.ndarray, context: str) -> None:
    distribution = np.asarray(distribution)
    if distribution.ndim != 2:
        raise ValueError(f"Joint distribution must be two-dimensional in {context}.")
    _validate_probability_rows(distribution, context)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(left)
        mask = left > 0
        terms[mask] = left[mask] * (np.log(left[mask]) - np.log(np.clip(right[mask], PROBABILITY_EPSILON, None)))
        return terms.sum(axis=-1)

    return np.maximum(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint), 0.0)


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(q, PROBABILITY_EPSILON, 1.0))).sum(axis=-1)


def compute_trial_metrics(
    *,
    bayes_joint: np.ndarray,
    predicted_joint: np.ndarray,
    reverse_joint: np.ndarray,
    bayes_marginals: np.ndarray,
    predicted_marginals: np.ndarray,
) -> dict[str, np.ndarray]:
    validate_joint_distribution(bayes_joint, "Bayes metric input")
    validate_joint_distribution(predicted_joint, "predicted metric input")
    validate_joint_distribution(reverse_joint, "reverse metric input")
    bayes_marginals = np.asarray(bayes_marginals, dtype=np.float64)
    predicted_marginals = np.asarray(predicted_marginals, dtype=np.float64)
    bayes_binary = np.stack((1.0 - bayes_marginals, bayes_marginals), axis=-1)
    predicted_binary = np.stack((1.0 - predicted_marginals, predicted_marginals), axis=-1)
    return {
        "marginal_cross_entropy": _cross_entropy(bayes_binary, predicted_binary).mean(axis=1),
        "marginal_js": _js_divergence(bayes_binary, predicted_binary).mean(axis=1),
        "joint_cross_entropy": _cross_entropy(bayes_joint, predicted_joint),
        "joint_js": _js_divergence(bayes_joint, predicted_joint),
        "incoherent_mass": np.clip(1.0 - predicted_joint[:, 0] - predicted_joint[:, -1], 0.0, 1.0),
        "order_inconsistency": _js_divergence(predicted_joint, reverse_joint),
    }


def _load_nano_predictor(config: ExperimentConfig) -> tuple[NanoTabPFNBinaryPredictor, dict]:
    resolved_checkpoint: Path
    try:
        if config.checkpoint is not None:
            resolved_checkpoint = Path(config.checkpoint).expanduser().resolve()
            if not resolved_checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")
            model = init_model_from_state_dict_file(str(resolved_checkpoint))
        else:
            classifier = NanoTabPFNClassifier(model=None, device=config.device, num_mem_chunks=config.num_mem_chunks)
            model = classifier.model
            resolved_checkpoint = Path("checkpoints/nanotabpfn.pth").resolve()
    except Exception as error:
        raise RuntimeError(
            "Unable to load the nanoTabPFN classifier checkpoint. Ensure network access is available for the "
            "official checkpoint, or pass a local file with --checkpoint PATH."
        ) from error

    checkpoint_hash = hashlib.sha256(resolved_checkpoint.read_bytes()).hexdigest()
    metadata = {
        "path": str(resolved_checkpoint),
        "sha256": checkpoint_hash,
        "architecture": {
            "num_layers": model.num_layers,
            "embedding_size": model.embedding_size,
            "num_attention_heads": model.num_attention_heads,
            "mlp_hidden_size": model.mlp_hidden_size,
            "num_outputs": model.num_outputs,
        },
    }
    return NanoTabPFNBinaryPredictor(model, config.device, config.num_mem_chunks), metadata


def _metric_rows(
    *,
    task_batch: SyntheticTaskBatch,
    query_count: int,
    evidence_count: int,
    common_support_size: int,
    model_name: str,
    metrics: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for trial_index in range(task_batch.support_x.shape[0]):
        row = {
            "trial": trial_index,
            "query_count": query_count,
            "evidence_count": evidence_count,
            "common_support_size": common_support_size,
            "model": model_name,
            "latent_task": "B" if task_batch.latent_task[trial_index] else "A",
            "posterior_b": task_batch.posterior_b[trial_index],
            "evidence_ones": int(task_batch.evidence_y[trial_index].sum()),
        }
        row.update({name: values[trial_index] for name, values in metrics.items()})
        rows.append(row)
    return rows


def _joint_rows(
    *,
    task_batch: SyntheticTaskBatch,
    query_count: int,
    evidence_count: int,
    common_support_size: int,
    model_name: str,
    order_name: str,
    predicted_joint: np.ndarray,
    bayes_joint: np.ndarray,
) -> list[dict]:
    outcomes = all_binary_vectors(query_count)
    rows = []
    for trial_index in range(task_batch.support_x.shape[0]):
        for outcome_index, outcome in enumerate(outcomes):
            rows.append(
                {
                    "trial": trial_index,
                    "query_count": query_count,
                    "evidence_count": evidence_count,
                    "common_support_size": common_support_size,
                    "model": model_name,
                    "order": order_name,
                    "outcome": "".join(map(str, outcome)),
                    "probability": predicted_joint[trial_index, outcome_index],
                    "bayes_probability": bayes_joint[trial_index, outcome_index],
                }
            )
    return rows


def summarize_metrics(trial_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["query_count", "evidence_count", "common_support_size", "model"]
    for group_values, group in trial_metrics.groupby(group_columns, sort=True):
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=float)
            count = len(values)
            standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
            sem = standard_deviation / math.sqrt(count)
            mean = float(values.mean())
            rows.append(
                {
                    **dict(zip(group_columns, group_values, strict=True)),
                    "metric": metric,
                    "count": count,
                    "mean": mean,
                    "std": standard_deviation,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                }
            )
    return pd.DataFrame(rows)


def run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    validate_config(config)
    predictor = None
    checkpoint_metadata = None
    if "nano" in config.models:
        predictor, checkpoint_metadata = _load_nano_predictor(config)

    rng = np.random.default_rng(config.seed)
    metric_rows = []
    joint_rows = []
    for common_support_size in effective_common_support_sizes(config):
        for query_count in config.query_counts:
            canonical_order = tuple(range(query_count))
            reverse_order = tuple(reversed(canonical_order))
            for evidence_count in config.evidence_counts:
                task_batch = generate_global_ambiguity_tasks(
                    trials=config.trials,
                    query_count=query_count,
                    evidence_count=evidence_count,
                    common_support_size=common_support_size,
                    support_noise=config.support_noise,
                    rng=rng,
                )
                bayes_joint = exact_joint_distribution(task_batch.posterior_b, query_count)
                bayes_marginals = np.repeat(task_batch.posterior_b[:, None], query_count, axis=1)

                model_distributions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                if "exact" in config.models:
                    model_distributions["exact"] = (bayes_joint, bayes_joint, bayes_marginals)
                if "independent" in config.models:
                    independent_joint = independent_joint_distribution(task_batch.posterior_b, query_count)
                    model_distributions["independent"] = (
                        independent_joint,
                        independent_joint,
                        bayes_marginals,
                    )
                if "nano" in config.models:
                    canonical_joint = enumerate_chain_joint(
                        predictor,
                        task_batch.support_x,
                        task_batch.support_y,
                        task_batch.query_x,
                        canonical_order,
                        context=f"nano, m={query_count}, r={evidence_count}, canonical",
                    )
                    reverse_joint = enumerate_chain_joint(
                        predictor,
                        task_batch.support_x,
                        task_batch.support_y,
                        task_batch.query_x,
                        reverse_order,
                        context=f"nano, m={query_count}, r={evidence_count}, reverse",
                    )
                    initial_probabilities = predictor.predict_binary_proba(
                        task_batch.support_x, task_batch.support_y, task_batch.query_x
                    )
                    _validate_probability_rows(
                        initial_probabilities.reshape(-1, 2),
                        f"nano initial marginals, m={query_count}, r={evidence_count}",
                    )
                    model_distributions["nano"] = (canonical_joint, reverse_joint, initial_probabilities[..., 1])

                for model_name, (canonical_joint, reverse_joint, marginals) in model_distributions.items():
                    metrics = compute_trial_metrics(
                        bayes_joint=bayes_joint,
                        predicted_joint=canonical_joint,
                        reverse_joint=reverse_joint,
                        bayes_marginals=bayes_marginals,
                        predicted_marginals=marginals,
                    )
                    metric_rows.extend(
                        _metric_rows(
                            task_batch=task_batch,
                            query_count=query_count,
                            evidence_count=evidence_count,
                            common_support_size=common_support_size,
                            model_name=model_name,
                            metrics=metrics,
                        )
                    )
                    for order_name, predicted_joint in (
                        ("canonical", canonical_joint),
                        ("reverse", reverse_joint),
                    ):
                        joint_rows.extend(
                            _joint_rows(
                                task_batch=task_batch,
                                query_count=query_count,
                                evidence_count=evidence_count,
                                common_support_size=common_support_size,
                                model_name=model_name,
                                order_name=order_name,
                                predicted_joint=predicted_joint,
                                bayes_joint=bayes_joint,
                            )
                        )

    trial_metrics = pd.DataFrame(metric_rows)
    joint_probabilities = pd.DataFrame(joint_rows)
    summary = summarize_metrics(trial_metrics)
    metadata = {"checkpoint": checkpoint_metadata, "outcome_encoding": "canonical binary, most significant bit first"}
    return trial_metrics, joint_probabilities, summary, metadata


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install the experiment extra with "
            "`pip install -e '.[experiments]'`, or pass --no-plots."
        ) from error
    return plt


def create_figures(summary: pd.DataFrame, joint_probabilities: pd.DataFrame, output_dir: Path) -> None:
    plt = _import_pyplot()
    colours = {"exact": "#1b9e77", "independent": "#d95f02", "nano": "#7570b3"}
    display_metrics = (
        ("marginal_cross_entropy", "Marginal cross-entropy"),
        ("joint_js", "Joint JS divergence"),
        ("incoherent_mass", "Incoherent mass"),
        ("order_inconsistency", "Order inconsistency"),
    )
    support_sizes = sorted(summary["common_support_size"].unique())
    multiple_support_sizes = len(support_sizes) > 1
    for query_count in sorted(summary["query_count"].unique()):
        for common_support_size in support_sizes:
            figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            subset = summary[
                (summary["query_count"] == query_count) & (summary["common_support_size"] == common_support_size)
            ]
            for axis, (metric, title) in zip(axes.flat, display_metrics, strict=True):
                metric_subset = subset[subset["metric"] == metric]
                for model_name in metric_subset["model"].unique():
                    model_data = metric_subset[metric_subset["model"] == model_name].sort_values("evidence_count")
                    axis.plot(
                        model_data["evidence_count"],
                        model_data["mean"],
                        marker="o",
                        label=model_name,
                        color=colours.get(model_name),
                    )
                    axis.fill_between(
                        model_data["evidence_count"],
                        np.maximum(model_data["ci95_low"], 0.0),
                        model_data["ci95_high"],
                        alpha=0.18,
                        color=colours.get(model_name),
                    )
                axis.set_title(title)
                axis.set_xlabel("Disambiguating support rows (r)")
                axis.set_ylim(bottom=0)
                axis.grid(alpha=0.25)
            axes[0, 0].legend(frameon=False)
            figure.suptitle(f"Hypothesis-collapse diagnostics (m={query_count}, support={common_support_size})")
            support_suffix = f"_n{common_support_size}" if multiple_support_sizes else ""
            for suffix in ("png", "pdf"):
                figure.savefig(output_dir / f"metric_sweep_m{query_count}{support_suffix}.{suffix}", dpi=300)
            plt.close(figure)

            representative = joint_probabilities[
                (joint_probabilities["query_count"] == query_count)
                & (joint_probabilities["common_support_size"] == common_support_size)
                & (joint_probabilities["evidence_count"] == 0)
                & (joint_probabilities["trial"] == 0)
                & (joint_probabilities["order"] == "canonical")
            ]
            if representative.empty:
                continue
            outcomes = list(dict.fromkeys(representative["outcome"]))
            model_names = list(dict.fromkeys(representative["model"]))
            x_positions = np.arange(len(outcomes))
            width = 0.8 / len(model_names)
            figure, axis = plt.subplots(figsize=(max(8, len(outcomes) * 0.6), 4.8), constrained_layout=True)
            for model_index, model_name in enumerate(model_names):
                model_data = representative[representative["model"] == model_name].set_index("outcome")
                values = [model_data.loc[outcome, "probability"] for outcome in outcomes]
                offset = (model_index - (len(model_names) - 1) / 2) * width
                axis.bar(
                    x_positions + offset,
                    values,
                    width=width,
                    label=model_name,
                    color=colours.get(model_name),
                )
            axis.set_xticks(x_positions, outcomes, rotation=45 if query_count >= 4 else 0)
            axis.set_ylabel("Probability")
            axis.set_xlabel("Canonical query-label vector")
            axis.set_title(
                f"Representative ambiguous joint prediction (m={query_count}, r=0, support={common_support_size})"
            )
            axis.set_ylim(0, 1)
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.25)
            for suffix in ("png", "pdf"):
                figure.savefig(output_dir / f"joint_distribution_m{query_count}_r0{support_suffix}.{suffix}", dpi=300)
            plt.close(figure)


def write_artifacts(
    *,
    config: ExperimentConfig,
    trial_metrics: pd.DataFrame,
    joint_probabilities: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict,
) -> Path:
    if config.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / "hypothesis_collapse" / timestamp
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    serializable_config = asdict(config)
    serializable_config.update(metadata)
    (output_dir / "config.json").write_text(json.dumps(serializable_config, indent=2) + "\n", encoding="utf-8")
    trial_metrics.to_csv(output_dir / "trial_metrics.csv", index=False)
    joint_probabilities.to_csv(output_dir / "joint_probabilities.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    if config.plots:
        create_figures(summary, joint_probabilities, output_dir)
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Local nanoTabPFN classifier checkpoint.")
    parser.add_argument("--device", type=str, default=str(get_default_device()), help="Torch device (cpu, mps, cuda).")
    parser.add_argument("--output-dir", type=str, default=None, help="New directory for CSVs, config, and figures.")
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--query-counts", type=int, nargs="+", default=list(SUPPORTED_QUERY_COUNTS))
    parser.add_argument("--evidence-counts", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--common-support-size", type=int, default=16)
    parser.add_argument(
        "--common-support-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sweep several common support-set sizes; overrides --common-support-size.",
    )
    parser.add_argument("--support-noise", type=float, default=0.1)
    parser.add_argument("--num-mem-chunks", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("exact", "independent", "nano"),
        default=["exact", "independent", "nano"],
    )
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON artifacts without figures.")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        seed=args.seed,
        trials=args.trials,
        query_counts=tuple(args.query_counts),
        evidence_counts=tuple(args.evidence_counts),
        common_support_size=args.common_support_size,
        common_support_sizes=None if args.common_support_sizes is None else tuple(args.common_support_sizes),
        support_noise=args.support_noise,
        models=tuple(args.models),
        device=args.device,
        num_mem_chunks=args.num_mem_chunks,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trial_metrics, joint_probabilities, summary, metadata = run_experiment(config)
    output_dir = write_artifacts(
        config=config,
        trial_metrics=trial_metrics,
        joint_probabilities=joint_probabilities,
        summary=summary,
        metadata=metadata,
    )
    print(f"Wrote hypothesis-collapse experiment artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
