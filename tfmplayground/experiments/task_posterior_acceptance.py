"""Predeclared statistical gates for promoting the task-posterior adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_delta: float
    confidence_low: float
    confidence_high: float
    samples: int
    passes: bool


def paired_bootstrap_gate(
    adapter_scores,
    baseline_scores,
    *,
    minimum_baseline: float = 0.621,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    random_state: int = 0,
) -> PairedBootstrapResult:
    """Require a positive paired CI and adapter mean above the known baseline."""
    adapter = np.asarray(adapter_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if adapter.shape != baseline.shape or adapter.ndim != 1 or len(adapter) < 2:
        raise ValueError("Scores must be paired one-dimensional arrays with at least two items.")
    if not np.isfinite(adapter).all() or not np.isfinite(baseline).all():
        raise ValueError("Scores must be finite.")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    difference = adapter - baseline
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))
    means = difference[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, (alpha, 1 - alpha))
    passes = bool(low > 0 and adapter.mean() > minimum_baseline)
    return PairedBootstrapResult(
        mean_delta=float(difference.mean()),
        confidence_low=float(low),
        confidence_high=float(high),
        samples=len(difference),
        passes=passes,
    )


def no_harm_gate(adapter_auc, vanilla_auc, *, tolerance: float = 0.002) -> bool:
    """Ordinary-prior macro AUC may trail vanilla by at most ``tolerance``."""
    adapter = np.asarray(adapter_auc, dtype=float)
    vanilla = np.asarray(vanilla_auc, dtype=float)
    if adapter.shape != vanilla.shape or adapter.size == 0:
        raise ValueError("AUC arrays must be non-empty and paired.")
    return bool(np.mean(adapter - vanilla) >= -tolerance)
