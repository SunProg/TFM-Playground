"""Derived temporal and grouped environment-adaptation protocols.

These are intentionally described as derived BeyondArena-compatible analyses,
not as the standard BeyondArena leaderboard protocol.  Data loading remains in
the caller so official train/test boundaries and dataset licensing stay intact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from tfmplayground.experiments.particle_benchmark import (
    Array,
    OnlineClassifier,
    RegimeStream,
    RegimeStreamSpec,
    evaluate_delayed_stream,
    expected_calibration_error,
)


@dataclass(frozen=True)
class TemporalClassificationData:
    name: str
    train_x: Array
    train_y: Array
    future_x: Array
    future_y: Array

    def __post_init__(self) -> None:
        if self.train_x.ndim != 2 or self.future_x.ndim != 2:
            raise ValueError("Features must be two-dimensional.")
        if self.train_x.shape[1] != self.future_x.shape[1]:
            raise ValueError("Training and future feature widths must match.")
        if self.train_y.shape != (len(self.train_x),) or self.future_y.shape != (len(self.future_x),):
            raise ValueError("Labels must align with feature rows.")


@dataclass(frozen=True)
class GroupedClassificationData:
    name: str
    x: Array
    y: Array
    groups: Array
    held_out_groups: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.y.shape != (len(self.x),) or self.groups.shape != (len(self.x),):
            raise ValueError("x, y, and groups must have aligned rows.")
        if not self.held_out_groups:
            raise ValueError("At least one held-out group is required.")


BEYOND_ARENA_TEMPORAL = ("kick", "hotel_booking_demand")
BEYOND_ARENA_GROUPED = (
    "parkinsons_biomedical_voice_measurements",
    "musk",
    "sepsis_prediction_1m",
)


@dataclass(frozen=True)
class BeyondArenaSlice:
    """Canonical repeat-0/fold-0 derived protocol plus immutable identity."""

    data: TemporalClassificationData | GroupedClassificationData
    dataset_uuid: str
    checksum: str
    repeat: int = 0
    fold: int = 0
    protocol: str = "derived_particle_adaptation_not_standard_leaderboard"


def _binary_labels(train, test) -> tuple[Array, Array]:
    values = list(dict.fromkeys(train.tolist()))
    if len(values) != 2:
        raise ValueError("The particle protocol supports binary datasets only.")
    mapping = {value: index for index, value in enumerate(values)}
    if any(value not in mapping for value in test.tolist()):
        raise ValueError("The held-out split contains a target unseen in training.")
    return np.asarray([mapping[value] for value in train], dtype=np.int64), np.asarray(
        [mapping[value] for value in test], dtype=np.int64
    )


def _train_fitted_features(train, test, *, max_features: int = 100) -> tuple[Array, Array]:
    """One-hot/median preprocessing and variance selection fitted on train only."""

    import pandas as pd

    train = train.copy()
    test = test.copy()
    columns = []
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            median = train[name].median()
            columns.append(
                (
                    np.nan_to_num(train[name].fillna(median).to_numpy(dtype=np.float32)),
                    np.nan_to_num(test[name].fillna(median).to_numpy(dtype=np.float32)),
                )
            )
        else:
            levels = sorted(map(str, train[name].dropna().unique().tolist()))
            train_values = train[name].fillna("__missing__").astype(str)
            test_values = test[name].fillna("__missing__").astype(str)
            columns.extend(
                (
                    (train_values == level).to_numpy(dtype=np.float32),
                    (test_values == level).to_numpy(dtype=np.float32),
                )
                for level in levels
            )
    if not columns:
        raise ValueError("No usable predictor columns remain.")
    train_x = np.column_stack([column[0] for column in columns]).astype(np.float32)
    test_x = np.column_stack([column[1] for column in columns]).astype(np.float32)
    if train_x.shape[1] > max_features:
        # Stable mergesort makes ties deterministic; test rows never influence selection.
        selected = np.argsort(-np.var(train_x, axis=0), kind="stable")[:max_features]
        train_x, test_x = train_x[:, selected], test_x[:, selected]
    return train_x, test_x


def beyondarena_from_container(container, name: str, *, repeat: int = 0, fold: int = 0) -> BeyondArenaSlice:
    """Build one bounded slice from an official Data Foundry container."""

    if name not in {*BEYOND_ARENA_TEMPORAL, *BEYOND_ARENA_GROUPED}:
        raise ValueError(f"Dataset {name!r} is not predeclared for this comparison.")
    frame = container.dataset
    target = container.task_metadata.target_column_name
    train_idx, test_idx = container.experiment_metadata.splits[repeat][fold]
    train_frame, test_frame = frame.iloc[train_idx], frame.iloc[test_idx]
    train_y, test_y = _binary_labels(train_frame[target], test_frame[target])
    group_on = getattr(container.task_metadata, "group_on", None)
    excluded = [target] + ([group_on] if group_on and group_on in frame.columns else [])
    train_x, test_x = _train_fitted_features(train_frame.drop(columns=excluded), test_frame.drop(columns=excluded))
    if name in BEYOND_ARENA_TEMPORAL:
        count = min(2048, len(test_y))
        data: TemporalClassificationData | GroupedClassificationData = TemporalClassificationData(
            name, train_x, train_y, test_x[:count], test_y[:count]
        )
    else:
        if not group_on:
            raise ValueError("Grouped dataset metadata must define group_on.")
        test_groups = test_frame[group_on].to_numpy()
        unique = list(dict.fromkeys(test_groups.tolist()))
        rng = np.random.default_rng(0)
        chosen = tuple(np.asarray(unique, dtype=object)[rng.permutation(len(unique))[:16]].tolist())
        keep = np.isin(test_groups, chosen)
        combined_x = np.concatenate((train_x, test_x[keep]))
        combined_y = np.concatenate((train_y, test_y[keep]))
        train_groups = np.full(len(train_y), "__official_training__", dtype=object)
        combined_groups = np.concatenate((train_groups, test_groups[keep].astype(object)))
        data = GroupedClassificationData(name, combined_x, combined_y, combined_groups, chosen)
    return BeyondArenaSlice(
        data=data,
        dataset_uuid=str(container.uuid),
        checksum=str(container.checksum),
        repeat=repeat,
        fold=fold,
    )


def load_beyondarena_slice(name: str, *, repeat: int = 0, fold: int = 0) -> BeyondArenaSlice:
    """Resolve only a named immutable container through the official collection API."""

    try:
        from data_foundry.collections import BEYOND_ARENA
    except ImportError as error:
        raise ImportError("Install the 'beyondarena' extra to download official slices.") from error
    return beyondarena_from_container(BEYOND_ARENA.get_dataset(name), name, repeat=repeat, fold=fold)


def _metrics(y: Array, probability: Array) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float).clip(1e-7, 1 - 1e-7)
    auc = roc_auc_score(y, probability) if np.unique(y).size == 2 else np.nan
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "auc": float(auc),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(np.mean((probability - y) ** 2)),
        "calibration_error": expected_calibration_error(y, probability),
    }


def _positive(raw: Array) -> Array:
    raw = np.asarray(raw)
    return (raw[:, 1] if raw.ndim == 2 else raw).astype(float)


def tune_on_chronological_training_windows(
    data: TemporalClassificationData,
    candidates: Mapping[str, Callable[[], OnlineClassifier]],
    *,
    windows: int = 3,
) -> tuple[str, dict[str, float]]:
    """Rolling-origin tuning entirely inside the official training period."""

    if windows < 1 or len(data.train_x) < windows + 1:
        raise ValueError("Not enough training rows for the requested chronological windows.")
    boundaries = np.linspace(0, len(data.train_x), windows + 2, dtype=int)
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    for split in range(1, windows + 1):
        context_stop, validation_stop = boundaries[split], boundaries[split + 1]
        if context_stop == 0 or validation_stop <= context_stop:
            continue
        for name, factory in candidates.items():
            predictor = factory()
            predictor.update(data.train_x[:context_stop], data.train_y[:context_stop])
            probability = _positive(predictor.predict_proba(data.train_x[context_stop:validation_stop]))
            scores[name].append(float(log_loss(data.train_y[context_stop:validation_stop], probability, labels=[0, 1])))
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    return min(means, key=means.get), means


def evaluate_temporal_delayed(
    data: TemporalClassificationData,
    predictors: Mapping[str, OnlineClassifier],
    *,
    batch_size: int = 32,
) -> Any:
    """Fit official training rows, then predict/reveal ordered future batches."""

    for predictor in predictors.values():
        predictor.update(data.train_x, data.train_y)
    stream = RegimeStream(
        x=data.future_x,
        y=data.future_y,
        regime=np.zeros(len(data.future_y), dtype=np.int64),
        candidate_y=data.future_y[None],
        segment_starts=(0,),
        spec=RegimeStreamSpec(
            pattern=(0,),
            dwell_lengths=(len(data.future_y),),
            n_features=min(100, max(1, data.future_x.shape[1])),
        ),
    )
    return evaluate_delayed_stream(stream, predictors, batch_size=batch_size, oracle_name="__no_oracle__")


def evaluate_temporal_few_shot(
    data: TemporalClassificationData,
    predictor_factory: Callable[[], OnlineClassifier],
    *,
    shots: Sequence[int] = (0, 8, 32, 128),
) -> list[dict[str, float | int | str]]:
    """Reveal the first m future rows, exclude them, and freeze thereafter."""

    results = []
    for count in shots:
        if count < 0 or count >= len(data.future_y):
            continue
        predictor = predictor_factory()
        predictor.update(data.train_x, data.train_y)
        if count:
            predictor.update(data.future_x[:count], data.future_y[:count])
        probability = _positive(predictor.predict_proba(data.future_x[count:]))
        results.append(
            {
                "dataset": data.name,
                "slice": "temporal",
                "shots": count,
                **_metrics(data.future_y[count:], probability),
            }
        )
    return results


def evaluate_grouped_few_shot(
    data: GroupedClassificationData,
    predictor_factory: Callable[[], OnlineClassifier],
    *,
    shots: Sequence[int] = (0, 8, 32, 128),
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Evaluate held-out groups in a deterministic seeded within-group order."""

    held_out = np.isin(data.groups, data.held_out_groups)
    train_x, train_y = data.x[~held_out], data.y[~held_out]
    rows: list[dict[str, float | int | str]] = []
    for group_index, group in enumerate(data.held_out_groups):
        indices = np.flatnonzero(data.groups == group)
        rng = np.random.default_rng(seed + group_index)
        indices = indices[rng.permutation(len(indices))]
        for count in shots:
            if count < 0 or len(indices) - count < 32:
                continue
            predictor = predictor_factory()
            predictor.update(train_x, train_y)
            if count:
                predictor.update(data.x[indices[:count]], data.y[indices[:count]])
            evaluation = indices[count:]
            probability = _positive(predictor.predict_proba(data.x[evaluation]))
            rows.append(
                {
                    "dataset": data.name,
                    "slice": "grouped",
                    "group": str(group),
                    "shots": count,
                    **_metrics(data.y[evaluation], probability),
                }
            )
    return rows
