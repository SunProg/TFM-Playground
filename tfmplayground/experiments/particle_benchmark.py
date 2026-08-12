"""Causal benchmarks for particle models under changing environments.

This module deliberately does not depend on BeyondArena or a particular model
class.  Models implement :class:`OnlineClassifier`; the same evaluator is then
used for synthetic streams and externally loaded temporal/grouped datasets.
IID TabArena remains a separate no-harm benchmark.
"""

from __future__ import annotations

import copy
import math
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

Array = np.ndarray


class OnlineClassifier(Protocol):
    """Minimal predict-then-reveal interface used by every benchmark method."""

    def predict_proba(self, x: Array, *, regime_hint: int | None = None) -> Array: ...

    def update(self, x: Array, y: Array, *, regime: int | None = None) -> None: ...

    def diagnostics(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RegimeStreamSpec:
    pattern: tuple[int, ...] = (0, 1, 0)
    dwell_lengths: tuple[int, ...] = (128, 128, 128)
    n_features: int = 8
    positive_rate: float = 0.5
    label_noise: float = 0.05
    missing_rate: float = 0.0
    nonlinear: bool = True
    latent_regime_count: int | None = None

    def __post_init__(self) -> None:
        if len(self.pattern) != len(self.dwell_lengths) or not self.pattern:
            raise ValueError("pattern and dwell_lengths must be non-empty and have equal length.")
        if min(self.pattern) < 0 or not 1 <= self.n_features <= 100:
            raise ValueError("regimes must be non-negative and n_features must be in [1, 100].")
        if self.latent_regime_count is not None and self.latent_regime_count <= max(self.pattern):
            raise ValueError("latent_regime_count must include every regime named in pattern.")
        if min(self.dwell_lengths) < 1:
            raise ValueError("Every dwell length must be positive.")
        if not 0 < self.positive_rate < 1:
            raise ValueError("positive_rate must be in (0, 1).")
        if not 0 <= self.label_noise < 0.5 or not 0 <= self.missing_rate < 1:
            raise ValueError("label_noise must be in [0, .5) and missing_rate in [0, 1).")


@dataclass(frozen=True)
class RegimeStream:
    x: Array
    y: Array
    regime: Array
    candidate_y: Array
    segment_starts: tuple[int, ...]
    spec: RegimeStreamSpec

    def __post_init__(self) -> None:
        rows = self.x.shape[0]
        if self.x.ndim != 2 or self.y.shape != (rows,) or self.regime.shape != (rows,):
            raise ValueError("Inconsistent stream array shapes.")
        if self.candidate_y.ndim != 2 or self.candidate_y.shape[1] != rows:
            raise ValueError("candidate_y must have shape (regimes, rows).")


def generate_regime_stream(spec: RegimeStreamSpec, seed: int) -> RegimeStream:
    """Generate distinct latent classifiers evaluated on exactly the same X rows."""

    rng = np.random.default_rng(seed)
    row_count = sum(spec.dwell_lengths)
    regime_count = spec.latent_regime_count or (max(spec.pattern) + 1)
    x_complete = rng.normal(size=(row_count, spec.n_features)).astype(np.float32)
    weights = rng.normal(size=(regime_count, spec.n_features))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(1e-12)
    quadratic = rng.normal(scale=0.35, size=(regime_count, spec.n_features))
    scores = weights @ x_complete.T
    if spec.nonlinear:
        scores += quadratic @ (np.square(x_complete) - 1).T
    thresholds = np.quantile(scores, 1 - spec.positive_rate, axis=1)
    candidate_y = (scores > thresholds[:, None]).astype(np.int64)
    if spec.label_noise:
        flips = rng.random(candidate_y.shape) < spec.label_noise
        candidate_y = np.logical_xor(candidate_y, flips).astype(np.int64)

    regime = np.concatenate(
        [np.full(length, mode, dtype=np.int64) for mode, length in zip(spec.pattern, spec.dwell_lengths, strict=True)]
    )
    y = candidate_y[regime, np.arange(row_count)]
    x = x_complete.copy()
    if spec.missing_rate:
        x[rng.random(x.shape) < spec.missing_rate] = np.nan
    starts = tuple(np.cumsum((0, *spec.dwell_lengths[:-1])).tolist())
    return RegimeStream(x=x, y=y, regime=regime, candidate_y=candidate_y, segment_starts=starts, spec=spec)


def standard_stream_suite(seed: int = 0) -> dict[str, RegimeStream]:
    """Small deterministic suite spanning stable, switching, and recurrent tasks."""

    patterns = {
        "stable_a": ((0,), (384,)),
        "a_b": ((0, 1), (128, 256)),
        "a_b_a": ((0, 1, 0), (96, 160, 128)),
        "a_b_c_a": ((0, 1, 2, 0), (64, 128, 96, 96)),
    }
    return {
        name: generate_regime_stream(
            RegimeStreamSpec(
                pattern=pattern,
                dwell_lengths=dwell,
                n_features=(1, 8, 32, 100)[index],
                positive_rate=(0.5, 0.25, 0.5, 0.1)[index],
                label_noise=(0.0, 0.05, 0.1, 0.05)[index],
                missing_rate=(0.0, 0.05, 0.1, 0.2)[index],
            ),
            seed + index,
        )
        for index, (name, (pattern, dwell)) in enumerate(patterns.items())
    }


def sample_training_stream(
    *,
    seed: int,
    particle_count: int = 2,
    stable_probability: float = 0.25,
    dwell_range: tuple[int, int] = (32, 128),
) -> RegimeStream:
    """Sample switching and genuine stable episodes from one training mixture.

    Every episode exposes exactly ``particle_count`` candidate functions, so K
    is never increased beyond the number of supervised regimes. Stable episodes
    keep one candidate active throughout while retaining the same supervised
    candidate set for episode-level matching.
    """

    if particle_count < 2 or particle_count > 4:
        raise ValueError("Training streams support K in [2, 4].")
    if not 0 <= stable_probability <= 1:
        raise ValueError("stable_probability must be in [0, 1].")
    low, high = dwell_range
    if low < 1 or high < low:
        raise ValueError("dwell_range must satisfy 1 <= low <= high.")
    rng = np.random.default_rng(seed)
    if rng.random() < stable_probability:
        pattern = (0,)
    elif particle_count == 2:
        pattern = (0, 1, 0) if rng.random() < 0.5 else (0, 1)
    else:
        middle = tuple(range(1, particle_count))
        pattern = (0, *middle, 0)
    dwell = tuple(int(rng.integers(low, high + 1)) for _ in pattern)
    n_features = int(rng.integers(1, 101))
    return generate_regime_stream(
        RegimeStreamSpec(
            pattern=pattern,
            dwell_lengths=dwell,
            n_features=n_features,
            positive_rate=float(rng.choice((0.1, 0.25, 0.5))),
            label_noise=float(rng.choice((0.0, 0.05, 0.1))),
            missing_rate=float(rng.choice((0.0, 0.05, 0.2))),
            latent_regime_count=particle_count,
        ),
        seed + 1,
    )


def _positive_probability(probabilities: Array) -> Array:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        result = probabilities
    elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
        result = probabilities[:, 1]
    else:
        raise ValueError("predict_proba must return (rows,), or (rows, 2) for binary classification.")
    if result.shape[0] == 0 or not np.isfinite(result).all():
        raise ValueError("Predictions must be finite and non-empty.")
    return result.clip(1e-7, 1 - 1e-7)


def expected_calibration_error(y: Array, probability: Array, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    memberships = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    value = 0.0
    for index in range(bins):
        selected = memberships == index
        if selected.any():
            value += selected.mean() * abs(float(y[selected].mean()) - float(probability[selected].mean()))
    return float(value)


@dataclass(frozen=True)
class BatchResult:
    method: str
    batch: int
    start: int
    stop: int
    regime: int
    log_loss: float
    brier: float
    auc: float
    balanced_accuracy: float
    calibration_error: float
    oracle_log_loss: float | None
    predicted_regime: int | None
    predict_seconds: float
    update_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    batches: tuple[BatchResult, ...]
    y_true: Array = field(repr=False)
    probabilities: Mapping[str, Array] = field(repr=False)
    regimes: Array = field(repr=False)

    def summary(self) -> list[dict[str, float | str]]:
        rows = []
        for method in sorted(self.probabilities):
            selected = [row for row in self.batches if row.method == method]
            probability = self.probabilities[method]
            labels = self.y_true
            auc = roc_auc_score(labels, probability) if np.unique(labels).size == 2 else math.nan
            oracle = [row.oracle_log_loss for row in selected if row.oracle_log_loss is not None]
            identified = [row.predicted_regime == row.regime for row in selected if row.predicted_regime is not None]
            delays = recovery_delays(self, method)
            rows.append(
                {
                    "method": method,
                    "prequential_log_loss": float(log_loss(labels, probability, labels=[0, 1])),
                    "auc": float(auc),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= 0.5)),
                    "brier": float(np.mean((probability - labels) ** 2)),
                    "calibration_error": expected_calibration_error(labels, probability),
                    "regime_identification_accuracy": float(np.mean(identified)) if identified else math.nan,
                    "mean_switch_recovery_delay_batches": float(np.mean(delays)) if delays else math.nan,
                    "recurrence_gain": recurrence_gain(self, method),
                    "regret_vs_oracle": (
                        float(np.mean([row.log_loss for row in selected]) - np.mean(oracle)) if oracle else math.nan
                    ),
                    "runtime_seconds": float(sum(row.predict_seconds + row.update_seconds for row in selected)),
                    "peak_memory_bytes": float(max(row.peak_memory_bytes for row in selected)),
                }
            )
        return rows


def evaluate_delayed_stream(
    stream: RegimeStream,
    predictors: Mapping[str, OnlineClassifier],
    *,
    batch_size: int = 16,
    oracle_name: str = "oracle",
) -> BenchmarkResult:
    """Score each batch before its labels are passed to ``update``.

    The API makes the causal boundary structural: ``predict_proba`` receives X
    only.  The active regime hint is provided exclusively to the named oracle.
    A batch may not cross a regime boundary, which keeps recovery measurements
    and oracle diagnostics unambiguous.
    """

    if batch_size < 1 or not predictors:
        raise ValueError("batch_size and predictors must be non-empty/positive.")
    boundaries = sorted({*stream.segment_starts, len(stream.y)})
    slices: list[tuple[int, int]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        slices.extend((start, min(start + batch_size, right)) for start in range(left, right, batch_size))

    probability_chunks: dict[str, list[Array]] = {name: [] for name in predictors}
    results: list[BatchResult] = []
    for batch_index, (start, stop) in enumerate(slices):
        x_batch, y_batch = stream.x[start:stop], stream.y[start:stop]
        active = int(stream.regime[start])
        pending: dict[str, tuple[Array, float, int, int | None]] = {}
        for name, predictor in predictors.items():
            tracemalloc.start()
            before = time.perf_counter()
            raw = predictor.predict_proba(x_batch, regime_hint=active if name == oracle_name else None)
            elapsed = time.perf_counter() - before
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            probability = _positive_probability(raw)
            diagnostics = predictor.diagnostics()
            predicted_regime = diagnostics.get("predicted_regime")
            pending[name] = (probability.copy(), elapsed, peak, predicted_regime)

        # Labels are revealed only after every method has committed predictions.
        for name, predictor in predictors.items():
            before = time.perf_counter()
            predictor.update(x_batch, y_batch.copy(), regime=active if name == oracle_name else None)
            update_seconds = time.perf_counter() - before
            probability, predict_seconds, peak, predicted_regime = pending[name]
            probability_chunks[name].append(probability)
            batch_auc = roc_auc_score(y_batch, probability) if np.unique(y_batch).size == 2 else math.nan
            results.append(
                BatchResult(
                    method=name,
                    batch=batch_index,
                    start=start,
                    stop=stop,
                    regime=active,
                    log_loss=float(log_loss(y_batch, probability, labels=[0, 1])),
                    brier=float(np.mean((probability - y_batch) ** 2)),
                    auc=float(batch_auc),
                    balanced_accuracy=float(balanced_accuracy_score(y_batch, probability >= 0.5)),
                    calibration_error=expected_calibration_error(y_batch, probability),
                    oracle_log_loss=None,
                    predicted_regime=None if predicted_regime is None else int(predicted_regime),
                    predict_seconds=predict_seconds,
                    update_seconds=update_seconds,
                    peak_memory_bytes=peak,
                )
            )

    probabilities = {name: np.concatenate(parts) for name, parts in probability_chunks.items()}
    if oracle_name in predictors:
        oracle_rows = {row.batch: row.log_loss for row in results if row.method == oracle_name}
        results = [BatchResult(**{**row.__dict__, "oracle_log_loss": oracle_rows[row.batch]}) for row in results]
    return BenchmarkResult(tuple(results), stream.y.copy(), probabilities, stream.regime.copy())


def recovery_delays(result: BenchmarkResult, method: str, *, tolerance: float = 0.05) -> list[int]:
    """Batches after each switch until loss returns near the prior segment level."""

    rows = [row for row in result.batches if row.method == method]
    delays: list[int] = []
    for index in range(1, len(rows)):
        if rows[index].regime == rows[index - 1].regime:
            continue
        previous = [row.log_loss for row in rows[:index] if row.regime == rows[index - 1].regime]
        target = float(np.mean(previous[-3:])) + tolerance
        delay = 0
        for row in rows[index:]:
            if row.regime != rows[index].regime:
                break
            if row.log_loss <= target:
                break
            delay += 1
        delays.append(delay)
    return delays


def recurrence_gain(result: BenchmarkResult, method: str, regime: int = 0) -> float:
    """Positive values mean a recurrent regime recovers faster on its second visit."""

    rows = [row for row in result.batches if row.method == method]
    runs: list[list[BatchResult]] = []
    for row in rows:
        if row.regime == regime:
            if not runs or runs[-1][-1].batch + 1 != row.batch:
                runs.append([])
            runs[-1].append(row)
    if len(runs) < 2:
        return math.nan
    return float(runs[0][0].log_loss - runs[1][0].log_loss)


class RefitContextClassifier:
    """Cumulative, sliding-window, or exponentially weighted context baseline."""

    def __init__(
        self,
        estimator_factory: Callable[[], ClassifierMixin] | None = None,
        *,
        window: int | None = None,
        decay: float | None = None,
    ):
        if window is not None and window < 1:
            raise ValueError("window must be positive or None.")
        if decay is not None and not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1].")
        self.estimator_factory = estimator_factory or (lambda: LogisticRegression(max_iter=500))
        self.window = window
        self.decay = decay
        self.x = np.empty((0, 0), dtype=np.float32)
        self.y = np.empty(0, dtype=np.int64)

    @staticmethod
    def _clean(x: Array) -> Array:
        return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, x: Array, *, regime_hint: int | None = None) -> Array:
        del regime_hint
        if self.y.size == 0 or np.unique(self.y).size < 2:
            positive = float(self.y.mean()) if self.y.size else 0.5
            return np.column_stack((np.full(len(x), 1 - positive), np.full(len(x), positive)))
        estimator = self.estimator_factory()
        kwargs: dict[str, Array] = {}
        if self.decay is not None:
            kwargs["sample_weight"] = self.decay ** np.arange(self.y.size - 1, -1, -1)
        try:
            estimator.fit(self.x, self.y, **kwargs)
        except TypeError:
            estimator.fit(self.x, self.y)
        return estimator.predict_proba(self._clean(x))

    def update(self, x: Array, y: Array, *, regime: int | None = None) -> None:
        del regime
        cleaned = self._clean(x)
        if self.x.shape[0] == 0:
            self.x = cleaned.copy()
        else:
            self.x = np.concatenate((self.x, cleaned))
        self.y = np.concatenate((self.y, np.asarray(y, dtype=np.int64)))
        if self.window is not None and self.y.size > self.window:
            self.x, self.y = self.x[-self.window :], self.y[-self.window :]

    def diagnostics(self) -> Mapping[str, Any]:
        return {}


class RetrievalContextClassifier(RefitContextClassifier):
    """Nearest labelled rows with an exponential recency tie-break."""

    def __init__(
        self,
        estimator_factory: Callable[[], ClassifierMixin] | None = None,
        *,
        context_size: int = 128,
        recency_weight: float = 0.1,
    ):
        super().__init__(estimator_factory)
        if context_size < 1 or recency_weight < 0:
            raise ValueError("context_size must be positive and recency_weight non-negative.")
        self.context_size = context_size
        self.recency_weight = recency_weight

    def predict_proba(self, x: Array, *, regime_hint: int | None = None) -> Array:
        del regime_hint
        if self.y.size <= self.context_size:
            return super().predict_proba(x)
        query = self._clean(x).mean(axis=0)
        scale = self.x.std(axis=0).clip(1e-6)
        distance = np.mean(((self.x - query) / scale) ** 2, axis=1)
        age = np.arange(self.y.size - 1, -1, -1) / self.y.size
        selected = np.argpartition(distance + self.recency_weight * age, self.context_size)[: self.context_size]
        original_x, original_y = self.x, self.y
        self.x, self.y = self.x[selected], self.y[selected]
        try:
            return super().predict_proba(x)
        finally:
            self.x, self.y = original_x, original_y


class OracleRegimeClassifier:
    """Diagnostic headroom: a distinct cumulative context for each known regime."""

    def __init__(self, estimator_factory: Callable[[], ClassifierMixin] | None = None):
        self.estimator_factory = estimator_factory
        self.models: dict[int, RefitContextClassifier] = {}

    def _model(self, regime: int) -> RefitContextClassifier:
        if regime not in self.models:
            self.models[regime] = RefitContextClassifier(self.estimator_factory)
        return self.models[regime]

    def predict_proba(self, x: Array, *, regime_hint: int | None = None) -> Array:
        if regime_hint is None:
            raise ValueError("OracleRegimeClassifier requires a regime hint.")
        return self._model(regime_hint).predict_proba(x)

    def update(self, x: Array, y: Array, *, regime: int | None = None) -> None:
        if regime is None:
            raise ValueError("OracleRegimeClassifier requires a regime label on update.")
        self._model(regime).update(x, y)

    def diagnostics(self) -> Mapping[str, Any]:
        return {}


def default_baselines(window: int = 128) -> dict[str, OnlineClassifier]:
    """Model-agnostic stand-ins; pass TabPFN estimator factories for official runs."""

    return {
        "cumulative_vanilla": RefitContextClassifier(),
        "sliding_window": RefitContextClassifier(window=window),
        "exponential_context": RefitContextClassifier(decay=0.98),
        "retrieval_context": RetrievalContextClassifier(context_size=window),
        "oracle": OracleRegimeClassifier(),
    }


def tabpfn_context_baselines(
    vanilla_factory: Callable[[], ClassifierMixin],
    *,
    safe_adapter_factory: Callable[[], ClassifierMixin] | None = None,
    window: int = 128,
) -> dict[str, OnlineClassifier]:
    """Required comparison set using caller-supplied TabPFN-compatible estimators."""

    baselines: dict[str, OnlineClassifier] = {
        "cumulative_vanilla": RefitContextClassifier(vanilla_factory),
        "sliding_window": RefitContextClassifier(vanilla_factory, window=window),
        "exponential_context": RefitContextClassifier(vanilla_factory, decay=0.98),
        "retrieval_context": RetrievalContextClassifier(vanilla_factory, context_size=window),
        "oracle": OracleRegimeClassifier(vanilla_factory),
    }
    if safe_adapter_factory is not None:
        baselines["safe_single_residual_adapter"] = RefitContextClassifier(safe_adapter_factory)
    return baselines


def assert_exact_causality(
    predictor_factory: Callable[[], OnlineClassifier], stream: RegimeStream, *, batch_size: int = 16
) -> None:
    """Counterfactual test that current-batch labels cannot change its prediction."""

    for start in range(0, len(stream.y), batch_size):
        left = predictor_factory()
        right = predictor_factory()
        if start:
            # Both counterfactual worlds have exactly the same revealed past.
            left.update(stream.x[:start], stream.y[:start])
            right.update(stream.x[:start], stream.y[:start])
        stop = min(start + batch_size, len(stream.y))
        predicted_left = _positive_probability(left.predict_proba(stream.x[start:stop]))
        predicted_right = _positive_probability(right.predict_proba(stream.x[start:stop]))
        np.testing.assert_array_equal(predicted_left, predicted_right)
        labels = stream.y[start:stop]
        left.update(stream.x[start:stop], labels)
        # Counterfactual labels are revealed only after the compared prediction.
        right.update(stream.x[start:stop], 1 - labels)


def paired_bootstrap_improvement(
    particle_losses: Sequence[float], baseline_losses: Sequence[float], *, seed: int = 0, samples: int = 10_000
) -> dict[str, float | bool]:
    """Paired one-sided promotion check; negative differences favour particles."""

    particle = np.asarray(particle_losses, dtype=float)
    baseline = np.asarray(baseline_losses, dtype=float)
    if particle.shape != baseline.shape or particle.size < 3:
        raise ValueError("Paired promotion needs equally shaped losses from at least three seeds.")
    differences = particle - baseline
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, differences.size), replace=True).mean(axis=1)
    upper = float(np.quantile(draws, 0.95))
    return {
        "mean_log_loss_improvement": float(-differences.mean()),
        "upper_95_difference": upper,
        "significant": bool(upper < 0),
    }


def real_data_promotion(
    improvements: Mapping[str, Sequence[float]], *, catastrophic_regression: float = 0.02
) -> dict[str, Any]:
    """Apply the separate temporal/grouped real-environment promotion gates."""

    required = {"temporal", "grouped"}
    if set(improvements) != required:
        raise ValueError("Real-data promotion requires separate temporal and grouped slices.")
    values = {name: np.asarray(rows, dtype=float) for name, rows in improvements.items()}
    passed = all(rows.size and rows.mean() > 0 and rows.min() >= -catastrophic_regression for rows in values.values())
    return {
        "passed": bool(passed),
        "classification": "practical_improvement" if passed else "mechanistic_result_only",
        "temporal_mean_improvement": float(values["temporal"].mean()),
        "grouped_mean_improvement": float(values["grouped"].mean()),
        "worst_dataset_improvement": float(min(rows.min() for rows in values.values())),
    }


def synthetic_acceptance(
    *,
    particle_seed_losses: Sequence[float],
    best_baseline_seed_losses: Sequence[float],
    particle_post_switch_loss: float,
    cumulative_post_switch_loss: float,
    particle_recovery_delay: float,
    sliding_recovery_delay: float,
    first_a_recovery_delay: float,
    second_a_recovery_delay: float,
    stable_particle_auc: float,
    stable_vanilla_auc: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate the predeclared synthetic promotion gates without moving them."""

    promotion = paired_bootstrap_improvement(
        particle_seed_losses,
        best_baseline_seed_losses,
        seed=seed,
    )
    gates = {
        "paired_log_loss": bool(promotion["significant"]),
        "beats_cumulative_after_switch": particle_post_switch_loss < cumulative_post_switch_loss,
        "recovers_faster_than_sliding": particle_recovery_delay < sliding_recovery_delay,
        "recurrence_is_faster": second_a_recovery_delay < first_a_recovery_delay,
        "stable_auc_no_harm": stable_particle_auc >= stable_vanilla_auc - 0.002,
    }
    return {"passed": all(gates.values()), "gates": gates, "paired_test": promotion}


def clone_predictors(predictors: Mapping[str, OnlineClassifier]) -> dict[str, OnlineClassifier]:
    """Convenience for multi-seed runs without carrying context between streams."""

    return {name: copy.deepcopy(predictor) for name, predictor in predictors.items()}
