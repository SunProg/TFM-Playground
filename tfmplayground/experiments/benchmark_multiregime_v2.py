"""Locked-episode benchmark and diagnostics for generalized multi-regime v2."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

from tfmplayground.experiments.multiregime_v2 import (
    RegimeEpisode,
    RegimeGeneratorConfig,
    episode_metadata_json,
    sample_regime_episode,
    tensor_hash,
)
from tfmplayground.models.slot_regime import SlotLogitsAdapter, SlotRegimePrediction

HIGHER_IS_BETTER = {"auroc", "auprc", "balanced_accuracy"}
LOWER_IS_BETTER = {"log_loss", "brier"}
METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: str
    generator: RegimeGeneratorConfig = RegimeGeneratorConfig()
    evaluation_seeds: tuple[int, ...] = (3402, 3403, 3404)
    episodes_per_seed: int = 32
    query_size: int = 256
    bootstrap_samples: int = 2_000
    include_main_grid: bool = True
    include_mechanisms: bool = True
    include_scm_grid: bool = True
    gate_only: bool = False
    gate_cell_index: int | None = None
    cell_index: int | None = None
    device: str = "cpu"
    cpu_workers: int = 4
    inference_batch_size: int = 4
    generation_workers: int = 1
    persist_episodes: bool = True
    compress_predictions: bool = False
    reuse_controls_from: str | None = None
    model_name: str = "tabpfn"

    def __post_init__(self) -> None:
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds cannot be empty.")
        if self.episodes_per_seed < 1 or self.query_size < 1 or self.bootstrap_samples < 1:
            raise ValueError("episode, query, and bootstrap counts must be positive.")
        if self.cpu_workers < 1 or self.inference_batch_size < 1 or self.generation_workers < 1:
            raise ValueError("cpu_workers, inference_batch_size, and generation_workers must be positive.")
        if not self.model_name or self.model_name in _CONTROL_MODELS:
            raise ValueError("model_name must be a non-empty name distinct from the benchmark control models.")
        if self.gate_cell_index is not None and self.gate_cell_index < 0:
            raise ValueError("gate_cell_index must be non-negative.")
        if self.cell_index is not None and self.cell_index < 0:
            raise ValueError("cell_index must be non-negative.")
        if self.gate_only and self.cell_index is not None:
            raise ValueError("cell_index cannot be combined with gate_only; use gate_cell_index.")
        if not (self.gate_only or self.include_main_grid or self.include_mechanisms or self.include_scm_grid):
            raise ValueError("At least one benchmark grid must be enabled.")


def analytic_grid() -> list[dict[str, Any]]:
    """The 305 non-duplicate primary coefficient cells."""
    cells = [
        {"backend": "analytic", "k": k, "alpha": alpha, "imbalance_ratio": ratio, "support_size": support,
         "difference_components": ("coefficients",)}
        for k in (2, 3, 4)
        for alpha in (0.0, 0.25, 0.5, 1.0, 2.0)
        for ratio in (1.0, 0.3, 0.1, 0.05)
        for support in (32, 64, 128, 256, 512)
    ]
    cells.extend(
        {"backend": "analytic", "k": 1, "alpha": 0.0, "imbalance_ratio": 1.0, "support_size": support,
         "difference_components": ("coefficients",)}
        for support in (32, 64, 128, 256, 512)
    )
    assert len(cells) == 305
    return cells


def mechanism_grid() -> list[dict[str, Any]]:
    return [
        {"backend": "analytic", "k": 3, "alpha": 1.0, "imbalance_ratio": 0.3, "support_size": 128,
         "difference_components": components}
        for components in (
            ("nonlinear",),
            ("feature_subset",),
            ("decision_boundary",),
            ("coefficients", "nonlinear", "feature_subset", "decision_boundary"),
        )
    ]


def gate_grid() -> list[dict[str, Any]]:
    """Small representative grid used to release downstream training early."""
    return [
        {
            "backend": "analytic",
            "k": 1,
            "alpha": 0.0,
            "imbalance_ratio": 1.0,
            "support_size": 128,
            "difference_components": ("coefficients",),
        },
        {
            "backend": "analytic",
            "k": 2,
            "alpha": 0.0,
            "imbalance_ratio": 1.0,
            "support_size": 128,
            "difference_components": ("coefficients",),
        },
        {
            "backend": "analytic",
            "k": 2,
            "alpha": 1.0,
            "imbalance_ratio": 0.3,
            "support_size": 128,
            "difference_components": ("coefficients",),
        },
    ]


def scm_grid() -> list[dict[str, Any]]:
    return [
        {"backend": "tabicl_scm", "k": k, "alpha": alpha, "imbalance_ratio": ratio,
         "support_size": support, "difference_components": ("coefficients",)}
        for k in (1, 2, 4)
        for alpha in (0.0, 0.5, 1.0, 2.0)
        for ratio in (1.0, 0.1)
        for support in (64, 128, 256)
    ]


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """All requested binary metrics, with explicit undefined reasons."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    result: dict[str, Any] = {"sample_count": int(len(y)), "class_balance": None}
    if len(y) == 0:
        for name in METRICS:
            result[name] = None
            result[f"{name}_reason"] = "no_rows"
        result["class_balance_reason"] = "no_rows"
        return result
    result["class_balance"] = float(y.mean())
    invalid = ~np.isfinite(p) | (p < 0) | (p > 1)
    if invalid.any():
        reason = "nonfinite_predictions" if (~np.isfinite(p)).any() else "probabilities_out_of_range"
        result["prediction_invalid_count"] = int(invalid.sum())
        result["prediction_invalid_reason"] = reason
        for name in METRICS:
            result[name] = None
            result[f"{name}_reason"] = reason
        return result
    result["prediction_invalid_count"] = 0
    result["prediction_invalid_reason"] = None
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    result["log_loss"] = float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean())
    result["brier"] = float(np.mean((p - y) ** 2))
    predicted = p >= 0.5
    result["balanced_accuracy"] = float(
        np.mean([np.mean(predicted[y == observed] == observed) for observed in np.unique(y)])
    )
    if np.unique(y).size < 2:
        result["auroc"] = None
        result["auroc_reason"] = "single_observed_class"
    else:
        result["auroc"] = float(roc_auc_score(y, p))
    if y.sum() == 0:
        result["auprc"] = None
        result["auprc_reason"] = "no_positive_labels"
    else:
        result["auprc"] = float(average_precision_score(y, p))
    return result


def recovery_score(metric: str, model: float | None, pooled: float | None, oracle: float | None) -> dict[str, Any]:
    """Unclipped recovery against oracle-HGB, or a reasoned null."""
    if model is None or pooled is None or oracle is None:
        return {"value": None, "reason": "undefined_metric"}
    if metric in HIGHER_IS_BETTER:
        denominator = oracle - pooled
        numerator = model - pooled
    elif metric in LOWER_IS_BETTER:
        denominator = pooled - oracle
        numerator = pooled - model
    else:
        raise ValueError(f"Unknown metric {metric!r}.")
    if denominator <= 1e-6:
        return {"value": None, "reason": "oracle_not_better_than_pooled"}
    return {"value": float(numerator / denominator), "reason": None}


def _constant_probability(labels: np.ndarray) -> float:
    return float((labels.sum() + 1) / (len(labels) + 2))


def pooled_hgb_predictions(episode: RegimeEpisode) -> np.ndarray:
    x = episode.support_x[0].cpu().numpy()
    y = episode.support_y[0].cpu().numpy().astype(int)
    query = episode.query_x[0].cpu().numpy()
    if np.unique(y).size == 1:
        return np.full(len(query), _constant_probability(y))
    classifier = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(x, y)
    return classifier.predict_proba(query)[:, 1]


def oracle_hgb_predictions(
    episode: RegimeEpisode, pooled: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    pooled = pooled_hgb_predictions(episode) if pooled is None else pooled
    support_x = episode.support_x[0].cpu().numpy()
    support_y = episode.support_y[0].cpu().numpy().astype(int)
    query_x = episode.query_x[0].cpu().numpy()
    support_z = episode.support_z[0].cpu().numpy()
    query_z = episode.query_z[0].cpu().numpy()
    prediction = np.empty(len(query_x), dtype=np.float64)
    missing: list[int] = []
    constant: list[int] = []
    for regime in range(episode.num_regimes):
        support_mask = support_z == regime
        query_mask = query_z == regime
        if not query_mask.any():
            continue
        if not support_mask.any():
            prediction[query_mask] = pooled[query_mask]
            missing.append(regime)
            continue
        labels = support_y[support_mask]
        if np.unique(labels).size == 1:
            prediction[query_mask] = _constant_probability(labels)
            constant.append(regime)
            continue
        classifier = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(
            support_x[support_mask], labels
        )
        prediction[query_mask] = classifier.predict_proba(query_x[query_mask])[:, 1]
    return prediction, {
        "missing_support_regimes": missing,
        "constant_support_regimes": constant,
        "fallback_count": int(sum((query_z == regime).sum() for regime in missing)),
    }


def generator_oracle_predictions(episode: RegimeEpisode) -> np.ndarray:
    probabilities = episode.counterfactual_query_probabilities[0].cpu().numpy()
    z = episode.query_z[0].cpu().numpy()
    return probabilities[z, np.arange(len(z))]


@torch.no_grad()
def _tabpfn_predictions_batch(
    model: torch.nn.Module,
    episodes: list[RegimeEpisode],
    *,
    oracle_one_hot: bool = False,
) -> list[tuple[np.ndarray, dict[str, Any] | None]]:
    """Run one model forward over a small same-shaped episode batch."""
    if not episodes:
        return []
    inputs = [episode.oracle_inputs() if oracle_one_hot else episode.latent_inputs() for episode in episodes]
    support_x = torch.cat([value[0] for value in inputs], dim=0)
    support_y = torch.cat([value[1] for value in inputs], dim=0)
    query_x = torch.cat([value[2] for value in inputs], dim=0)
    raw_model = model.model if isinstance(model, SlotLogitsAdapter) else model
    device = next(raw_model.parameters()).device
    output = raw_model(support_x.to(device), support_y.to(device), query_x.to(device))
    if isinstance(output, SlotRegimePrediction):
        probability = output.marginal_probabilities()[..., 1]
        diagnostics = []
        for index, episode in enumerate(episodes):
            single = SlotRegimePrediction(
                slot_logits=output.slot_logits[index : index + 1],
                log_gate=output.log_gate[index : index + 1],
                support_attention=output.support_attention[index : index + 1],
            )
            diagnostics.append(slot_diagnostics(single, episode))
    else:
        probability = output[..., :2].softmax(-1)[..., 1]
        diagnostics = [None] * len(episodes)
    return [
        (probability[index].detach().cpu().numpy(), diagnostics[index])
        for index in range(len(episodes))
    ]


def _tabpfn_prediction(
    model: torch.nn.Module, episode: RegimeEpisode, *, oracle_one_hot: bool = False
) -> tuple[np.ndarray, dict[str, Any] | None]:
    return _tabpfn_predictions_batch(model, [episode], oracle_one_hot=oracle_one_hot)[0]


def _cpu_control_predictions(
    episode: RegimeEpisode,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate the two CPU controls; safe to run in parallel across episodes."""
    pooled = pooled_hgb_predictions(episode)
    oracle, oracle_info = oracle_hgb_predictions(episode, pooled)
    return pooled, oracle, oracle_info


def tabpfn_predictions(model: torch.nn.Module, episode: RegimeEpisode, *, oracle_one_hot: bool = False) -> np.ndarray:
    return _tabpfn_prediction(model, episode, oracle_one_hot=oracle_one_hot)[0]


def _assignment_metrics(probability: np.ndarray, z: np.ndarray, num_regimes: int) -> dict[str, Any]:
    slots = probability.shape[1]
    predicted = probability.argmax(axis=1)
    cost = np.zeros((num_regimes, slots), dtype=np.float64)
    for regime in range(num_regimes):
        mask = z == regime
        cost[regime] = -np.log(np.clip(probability[mask], 1e-12, 1)).mean(axis=0) if mask.any() else 0.0
    rows, columns = linear_sum_assignment(cost)
    mapping = {slot: regime for regime, slot in zip(rows, columns, strict=True)}
    matched = np.asarray([mapping.get(slot, -1) for slot in predicted])
    utilization = probability.mean(axis=0)
    entropy = float(-(utilization * np.log(np.clip(utilization, 1e-12, 1))).sum())
    active = int((utilization >= 0.01).sum())
    return {
        "ari": float(adjusted_rand_score(z, predicted)),
        "nmi": float(normalized_mutual_info_score(z, predicted)),
        "hungarian_accuracy": float(np.mean(matched == z)),
        "soft_utilization": utilization.tolist(),
        "effective_slot_count": float(math.exp(entropy)),
        "mean_row_entropy": float(
            np.mean(-(probability * np.log(np.clip(probability, 1e-12, 1))).sum(axis=1))
        ),
        "unused_slots": np.flatnonzero(utilization < 0.01).tolist(),
        "collapsed": bool(utilization.max() >= 0.95 or active < min(num_regimes, slots)),
    }


def slot_diagnostics(prediction: SlotRegimePrediction, episode: RegimeEpisode) -> dict[str, Any]:
    """Support-attention and query-gate diagnostics are intentionally separate."""
    return {
        "support": _assignment_metrics(
            prediction.support_attention[0].detach().cpu().numpy(),
            episode.support_z[0].cpu().numpy(),
            episode.num_regimes,
        ),
        "query": _assignment_metrics(
            prediction.gate()[0].detach().cpu().numpy(),
            episode.query_z[0].cpu().numpy(),
            episode.num_regimes,
        ),
    }


def _cell_id(cell: dict[str, Any]) -> str:
    value = json.dumps(cell, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _locked_episode_worker(
    request: tuple[RegimeGeneratorConfig, int, int, str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Generate one deterministic locked episode in a worker process."""
    generator, num_regimes, episode_seed, episode_id, cell = request
    episode = sample_regime_episode(generator, num_regimes=num_regimes, seed=episode_seed, episode_id=episode_id)
    episode.metadata.update(
        {"cell_id": cell["cell_id"], "evaluation_seed": cell["evaluation_seed"], "episode_index": cell["episode_index"]}
    )
    tensor_values = {
        name: getattr(episode, name).numpy()
        for name in (
            "support_x",
            "support_y",
            "query_x",
            "query_y",
            "support_z",
            "query_z",
            "active_regime_mask",
            "support_gate_probabilities",
            "query_gate_probabilities",
            "counterfactual_support_probabilities",
            "counterfactual_query_probabilities",
        )
    }
    return tensor_values, dict(episode.metadata), {
        "episode_id": episode_id,
        "cell_id": cell["cell_id"],
        "evaluation_seed": cell["evaluation_seed"],
        "episode_index": cell["episode_index"],
        "tensor_hash": tensor_hash(episode),
        **cell["spec"],
    }


def build_locked_episodes(config: BenchmarkConfig) -> tuple[list[RegimeEpisode], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    if config.gate_only:
        gate_cells = gate_grid()
        if config.gate_cell_index is not None:
            if config.gate_cell_index >= len(gate_cells):
                raise ValueError(f"gate_cell_index {config.gate_cell_index} is out of range.")
            gate_cells = gate_cells[config.gate_cell_index : config.gate_cell_index + 1]
        cells.extend(gate_cells)
    elif config.include_main_grid:
        cells.extend(analytic_grid())
    if not config.gate_only and config.include_mechanisms:
        cells.extend(mechanism_grid())
    if not config.gate_only and config.include_scm_grid:
        cells.extend(scm_grid())
    # Keep the original global cell index when selecting a shard.  That index
    # participates in the episode seed stream, so a sharded run generates the
    # exact same tensors as the corresponding cell in an unsplit benchmark.
    indexed_cells = list(enumerate(cells))
    if config.cell_index is not None:
        if config.cell_index >= len(indexed_cells):
            raise ValueError(f"cell_index {config.cell_index} is out of range for {len(indexed_cells)} cells.")
        indexed_cells = [indexed_cells[config.cell_index]]

    requests: list[tuple[RegimeGeneratorConfig, int, int, str, dict[str, Any]]] = []
    for cell_index, cell in indexed_cells:
        cell_id = _cell_id(cell)
        for evaluation_seed in config.evaluation_seeds:
            for episode_index in range(config.episodes_per_seed):
                # Cell position is included, so streams cannot collide even if two
                # cells happen to serialize similarly in a future schema.
                seed_sequence = np.random.SeedSequence([evaluation_seed, cell_index, episode_index])
                episode_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
                generator = replace(
                    config.generator,
                    backend=cell["backend"],
                    regime_separation=cell["alpha"],
                    imbalance_ratio=cell["imbalance_ratio"],
                    support_size=cell["support_size"],
                    query_size=config.query_size,
                    difference_components=cell["difference_components"],
                    seed=episode_seed,
                    single_regime_source="matched",
                )
                episode_id = f"{cell_id}-{evaluation_seed}-{episode_index:03d}"
                requests.append(
                    (
                        generator,
                        cell["k"],
                        episode_seed,
                        episode_id,
                        {
                            "cell_id": cell_id,
                            "evaluation_seed": evaluation_seed,
                            "episode_index": episode_index,
                            "spec": cell,
                        },
                    )
                )
    if config.generation_workers == 1:
        results = map(_locked_episode_worker, requests)
    else:
        # Spawn clean workers because the parent may already have initialized a
        # CUDA checkpoint before entering ``run_benchmark``.  Forking a process
        # after CUDA initialization is unsupported and can deadlock or crash.
        with ProcessPoolExecutor(
            max_workers=config.generation_workers,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            results = pool.map(_locked_episode_worker, requests, chunksize=4)
    episodes = []
    manifest = []
    for tensor_values, metadata, manifest_row in results:
        episodes.append(
            RegimeEpisode(
                **{name: torch.from_numpy(value) for name, value in tensor_values.items()},
                metadata=metadata,
            )
        )
        manifest.append(manifest_row)
    return episodes, manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


_CONTROL_MODELS = frozenset(("pooled_hgb", "oracle_hgb", "generator_oracle"))
_METRIC_INT_FIELDS = frozenset(("evaluation_seed", "episode_index", "k", "support_size", "sample_count",
                                "prediction_invalid_count", "oracle_fallback_count", "regime"))
_METRIC_FLOAT_FIELDS = frozenset(("alpha", "imbalance_ratio", "class_balance", *METRICS))


def _coerce_metric_row(row: dict[str, str]) -> dict[str, Any]:
    """Restore numeric values from a previously written episode_metrics.csv."""
    converted: dict[str, Any] = dict(row)
    for field in _METRIC_INT_FIELDS:
        value = row.get(field, "")
        converted[field] = None if value in ("", "None", "null") else int(value)
    for field in _METRIC_FLOAT_FIELDS:
        value = row.get(field, "")
        converted[field] = None if value in ("", "None", "null") else float(value)
    return converted


def load_reused_control_metrics(
    source_dir: str | Path, manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Load pooled/oracle controls from a completed locked benchmark.

    The source benchmark and the current run must have the same deterministic
    episode IDs and tensor hashes.  Only control metric rows are reused; this
    avoids refitting HGB for every neural model while preserving paired
    recovery against the original controls.
    """
    source = Path(source_dir)
    manifest_path = source / "episode_manifest.json"
    metrics_path = source / "episode_metrics.csv"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Reusable benchmark controls require {manifest_path} and {metrics_path}.")
    source_manifest = json.loads(manifest_path.read_text())
    expected = {row["episode_id"]: row["tensor_hash"] for row in manifest}
    observed = {row["episode_id"]: row["tensor_hash"] for row in source_manifest}
    if expected != observed:
        missing = sorted(set(expected) - set(observed))[:3]
        extra = sorted(set(observed) - set(expected))[:3]
        mismatched = sorted(key for key in expected.keys() & observed.keys() if expected[key] != observed[key])[:3]
        raise ValueError(
            "Reusable controls do not match the locked episodes "
            f"(missing={missing}, extra={extra}, tensor_hash_mismatches={mismatched})."
        )
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    with metrics_path.open(newline="") as source_file:
        for raw in csv.DictReader(source_file):
            if raw.get("model") not in _CONTROL_MODELS:
                continue
            row = _coerce_metric_row(raw)
            if row.get("episode_id") not in expected:
                raise ValueError(f"Reusable controls contain an unknown episode_id={row.get('episode_id')!r}.")
            rows.append(row)
            key = (str(row["episode_id"]), str(row["model"]))
            counts[key] = counts.get(key, 0) + 1
    required = {(episode_id, model) for episode_id in expected for model in _CONTROL_MODELS}
    missing = sorted(required - counts.keys())[:3]
    if missing:
        raise ValueError(f"Reusable controls are incomplete; missing rows such as {missing}.")
    return rows


def _checkpoint_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_episode_metrics(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    keys = ("model", "scope", "regime", "backend", "k", "alpha", "imbalance_ratio", "support_size",
            "difference_components")
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        base = dict(zip(keys, group_key, strict=True))
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group if row.get(metric) not in (None, "")], dtype=float)
            if not len(values):
                summary.append({**base, "metric": metric, "mean": None, "ci_low": None, "ci_high": None,
                                "n_episodes": 0, "reason": "all_values_undefined"})
                continue
            low, high = _bootstrap_interval(values, rng, bootstrap_samples)
            summary.append({**base, "metric": metric, "mean": float(values.mean()), "ci_low": low,
                            "ci_high": high, "n_episodes": int(len(values)), "reason": None})

    # Recovery and improvement are computed from paired episode-level rows,
    # never from separately averaged populations.
    lookup = {(row["episode_id"], row["model"], row["scope"], row["regime"]): row for row in rows}
    models = sorted({row["model"] for row in rows} - {"pooled_hgb", "oracle_hgb", "generator_oracle"})
    paired_groups: dict[tuple, list[float]] = {}
    null_reasons: dict[tuple, list[str]] = {}
    for model in models:
        candidates = [row for row in rows if row["model"] == model]
        for row in candidates:
            pooled = lookup.get((row["episode_id"], "pooled_hgb", row["scope"], row["regime"]))
            oracle = lookup.get((row["episode_id"], "oracle_hgb", row["scope"], row["regime"]))
            if pooled is None or oracle is None:
                continue
            for metric in METRICS:
                recovery = recovery_score(metric, row.get(metric), pooled.get(metric), oracle.get(metric))
                group = tuple(row.get(key) for key in keys) + (f"{metric}_recovery",)
                if recovery["value"] is None:
                    null_reasons.setdefault(group, []).append(recovery["reason"])
                else:
                    paired_groups.setdefault(group, []).append(recovery["value"])
            if row.get("log_loss") is not None and pooled.get("log_loss") is not None:
                group = tuple(row.get(key) for key in keys) + ("paired_log_loss_improvement",)
                paired_groups.setdefault(group, []).append(pooled["log_loss"] - row["log_loss"])
    for group in sorted(set(paired_groups) | set(null_reasons), key=str):
        *base_values, metric = group
        base = dict(zip(keys, base_values, strict=True))
        values = np.asarray(paired_groups.get(group, []), dtype=float)
        if len(values):
            low, high = _bootstrap_interval(values, rng, bootstrap_samples)
            summary.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_episodes": int(len(values)),
                    "reason": None,
                }
            )
        else:
            reasons = null_reasons[group]
            summary.append(
                {
                    **base,
                    "metric": metric,
                    "mean": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n_episodes": 0,
                    "reason": max(set(reasons), key=reasons.count),
                }
            )
    return summary


def _regime_recovery_score(episode: RegimeEpisode) -> tuple[float | None, float]:
    """Held-out query accuracy of an HGB z probe and its majority baseline."""
    support_z = episode.support_z[0].cpu().numpy()
    query_z = episode.query_z[0].cpu().numpy()
    majority = float(np.max(np.bincount(query_z, minlength=episode.num_regimes)) / len(query_z))
    if np.unique(support_z).size < 2:
        return None, majority
    probe = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(
        episode.support_x[0].cpu().numpy(), support_z
    )
    return float(np.mean(probe.predict(episode.query_x[0].cpu().numpy()) == query_z)), majority


def execution_gate_report(
    episodes: list[RegimeEpisode], metric_rows: list[dict[str, Any]], *, models: set[str]
) -> dict[str, Any]:
    """Pre-GPU gate: generator recoverability, controls, and saved baselines."""
    recoverability = [_regime_recovery_score(episode) for episode in episodes if episode.num_regimes > 1]
    valid_recovery = [(score, majority) for score, majority in recoverability if score is not None]
    recovery_margin = (
        float(np.mean([score - majority for score, majority in valid_recovery])) if valid_recovery else None
    )
    overall = [row for row in metric_rows if row["scope"] == "overall"]
    by_episode_model = {(row["episode_id"], row["model"]): row for row in overall}
    alpha_zero_gaps = []
    separated_oracle_gains = []
    for row in overall:
        if row["model"] != "pooled_hgb":
            continue
        oracle = by_episode_model.get((row["episode_id"], "oracle_hgb"))
        if oracle is None:
            continue
        if row["alpha"] == 0 and row["log_loss"] is not None and oracle["log_loss"] is not None:
            alpha_zero_gaps.append(abs(row["log_loss"] - oracle["log_loss"]))
        if row["alpha"] >= 1 and row["log_loss"] is not None and oracle["log_loss"] is not None:
            separated_oracle_gains.append(row["log_loss"] - oracle["log_loss"])
    checks = {
        "support_to_query_z_recoverable": recovery_margin is not None and recovery_margin > 0.0,
        "alpha_zero_is_pooled_task": bool([episode for episode in episodes if episode.metadata["requested_alpha"] == 0])
        and all(
            torch.equal(
                episode.counterfactual_query_probabilities[:, : episode.num_regimes],
                episode.counterfactual_query_probabilities[:, :1].expand(
                    -1, episode.num_regimes, -1
                ),
            )
            for episode in episodes
            if episode.metadata["requested_alpha"] == 0
        ),
        "oracle_hgb_materially_exceeds_pooled": (
            bool(separated_oracle_gains) and float(np.mean(separated_oracle_gains)) > 0.01
        ),
        "standard_tabpfn_artifacts_saved": "tabpfn" in models,
        "oracle_one_hot_artifacts_saved": "oracle_one_hot_tabpfn" in models,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_z_recovery_margin": recovery_margin,
        "mean_alpha_zero_oracle_gap": float(np.mean(alpha_zero_gaps)) if alpha_zero_gaps else None,
        "mean_separated_oracle_log_loss_gain": (
            float(np.mean(separated_oracle_gains)) if separated_oracle_gains else None
        ),
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    tabpfn_model: torch.nn.Module | None = None,
    oracle_one_hot_model: torch.nn.Module | None = None,
    checkpoint_paths: dict[str, str | Path] | None = None,
) -> Path:
    """Generate once, then evaluate every model on those exact episodes."""
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark directory {output}.")
    output.mkdir(parents=True)
    progress_path = output / "progress.jsonl"
    progress_started = time.monotonic()

    def log_progress(event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            "elapsed_seconds": time.monotonic() - progress_started,
            **fields,
        }
        with progress_path.open("a") as progress_file:
            progress_file.write(json.dumps(record, sort_keys=True) + "\n")
        print("[benchmark] " + json.dumps(record, sort_keys=True), flush=True)

    checkpoints = {name: _checkpoint_hash(path) for name, path in (checkpoint_paths or {}).items()}
    (output / "config.json").write_text(
        json.dumps({**asdict(config), "checkpoint_hashes": checkpoints}, indent=2, sort_keys=True) + "\n"
    )
    episodes, manifest = build_locked_episodes(config)
    reused_control_rows = (
        load_reused_control_metrics(config.reuse_controls_from, manifest)
        if config.reuse_controls_from is not None
        else []
    )
    cell_order: list[str] = []
    for row in manifest:
        if row["cell_id"] not in cell_order:
            cell_order.append(row["cell_id"])
    log_progress(
        "generation_complete",
        total_episodes=len(episodes),
        total_cells=len(cell_order),
        models=sorted(
            ["pooled_hgb", "oracle_hgb", "generator_oracle"]
            + ([config.model_name] if tabpfn_model is not None else [])
            + (["oracle_one_hot_tabpfn"] if oracle_one_hot_model is not None else [])
        ),
        controls_reused=bool(config.reuse_controls_from),
    )
    (output / "episode_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output / "episode_metadata.jsonl").open("w") as destination:
        for episode in episodes:
            destination.write(episode_metadata_json(episode) + "\n")
    if config.persist_episodes:
        torch.save(episodes, output / "episodes.pt")
    else:
        log_progress("episode_tensor_persistence_skipped", reason="large_episodes_pt_disabled")

    # Reused controls are already paired to these exact episode IDs.  Neural
    # predictions are appended below; control prediction rows are intentionally
    # not duplicated into the new predictions file.
    metric_rows: list[dict[str, Any]] = list(reused_control_rows)
    evaluation_started = time.monotonic()
    cell_ranges: list[tuple[int, int]] = []
    range_start = 0
    while range_start < len(manifest):
        cell_id = manifest[range_start]["cell_id"]
        range_end = range_start + 1
        while range_end < len(manifest) and manifest[range_end]["cell_id"] == cell_id:
            range_end += 1
        cell_ranges.append((range_start, range_end))
        range_start = range_end

    prediction_fields = (
        "episode_id",
        "cell_id",
        "evaluation_seed",
        "backend",
        "k",
        "alpha",
        "imbalance_ratio",
        "support_size",
        "difference_components",
        "model",
        "query_index",
        "regime",
        "target",
        "probability",
    )
    prediction_path = output / ("predictions.csv.gz" if config.compress_predictions else "predictions.csv")
    prediction_destination = (
        gzip.open(prediction_path, "wt", newline="")
        if config.compress_predictions
        else prediction_path.open("w", newline="")
    )
    with prediction_destination, ThreadPoolExecutor(max_workers=config.cpu_workers) as cpu_pool:
        prediction_writer = csv.DictWriter(prediction_destination, fieldnames=prediction_fields)
        prediction_writer.writeheader()
        for cell_number, (range_start, range_end) in enumerate(cell_ranges, start=1):
            cell_id = manifest[range_start]["cell_id"]
            cell_episodes = episodes[range_start:range_end]
            first_manifest_row = manifest[range_start]
            log_progress(
                "cell_start",
                cell_index=cell_number,
                total_cells=len(cell_order),
                cell_id=cell_id,
                backend=first_manifest_row["backend"],
                k=first_manifest_row["k"],
                alpha=first_manifest_row["alpha"],
                imbalance_ratio=first_manifest_row["imbalance_ratio"],
                support_size=first_manifest_row["support_size"],
                difference_components=first_manifest_row["difference_components"],
                batch_size=min(config.inference_batch_size, len(cell_episodes)),
                cpu_workers=config.cpu_workers,
            )
            if config.reuse_controls_from is None:
                cpu_results = list(cpu_pool.map(_cpu_control_predictions, cell_episodes))
                log_progress(
                    "cpu_controls_complete",
                    cell_index=cell_number,
                    total_cells=len(cell_order),
                    cell_id=cell_id,
                    completed_episodes=range_start + len(cell_episodes),
                    total_episodes=len(episodes),
                )
            else:
                cpu_results = [None] * len(cell_episodes)
                if cell_number == 1 or cell_number % 16 == 0 or cell_number == len(cell_ranges):
                    log_progress(
                        "cpu_controls_reused",
                        cell_index=cell_number,
                        total_cells=len(cell_order),
                        cell_id=cell_id,
                        completed_episodes=range_start + len(cell_episodes),
                        total_episodes=len(episodes),
                    )
            model_results: dict[str, list[tuple[np.ndarray, dict[str, Any] | None]]] = {}
            for model_name, model, oracle_one_hot in (
                (config.model_name, tabpfn_model, False),
                ("oracle_one_hot_tabpfn", oracle_one_hot_model, True),
            ):
                if model is None:
                    continue
                batched_results: list[tuple[np.ndarray, dict[str, Any] | None]] = []
                batches = list(range(0, len(cell_episodes), config.inference_batch_size))
                for batch_number, batch_start in enumerate(batches, start=1):
                    batch = cell_episodes[batch_start : batch_start + config.inference_batch_size]
                    batched_results.extend(
                        _tabpfn_predictions_batch(model, batch, oracle_one_hot=oracle_one_hot)
                    )
                    if batch_number == 1 or batch_number == len(batches) or batch_number % 8 == 0:
                        log_progress(
                            "model_batch_progress",
                            cell_index=cell_number,
                            total_cells=len(cell_order),
                            cell_id=cell_id,
                            model=model_name,
                            completed_batches=batch_number,
                            total_batches=len(batches),
                            completed_episodes=range_start + min(
                                batch_start + len(batch), len(cell_episodes)
                            ),
                            total_episodes=len(episodes),
                        )
                model_results[model_name] = batched_results

            for offset, (episode, manifest_row, cpu_result) in enumerate(
                zip(cell_episodes, manifest[range_start:range_end], cpu_results, strict=True)
            ):
                episode_index = range_start + offset + 1
                if cpu_result is None:
                    # The paired control metrics were loaded before the loop;
                    # only the target model needs new prediction rows.
                    predictions: dict[str, np.ndarray] = {}
                    oracle_info = {"fallback_count": 0, "missing_support_regimes": [],
                                   "constant_support_regimes": []}
                else:
                    pooled, oracle, oracle_info = cpu_result
                    predictions = {
                        "pooled_hgb": pooled,
                        "oracle_hgb": oracle,
                        "generator_oracle": generator_oracle_predictions(episode),
                    }
                diagnostics_by_model: dict[str, dict[str, Any]] = {}
                for model_name, results in model_results.items():
                    predictions[model_name], diagnostics = results[offset]
                    if diagnostics is not None:
                        diagnostics_by_model[model_name] = diagnostics
                prediction_quality: dict[str, dict[str, Any]] = {}
                for model_name, probability in predictions.items():
                    probability = np.asarray(probability, dtype=np.float64)
                    invalid = ~np.isfinite(probability) | (probability < 0) | (probability > 1)
                    reason = "nonfinite_predictions" if (~np.isfinite(probability)).any() else (
                        "probabilities_out_of_range" if invalid.any() else None
                    )
                    prediction_quality[model_name] = {
                        "invalid_count": int(invalid.sum()),
                        "invalid_reason": reason,
                    }
                    predictions[model_name] = probability
                    if invalid.any():
                        log_progress(
                            "invalid_prediction",
                            episode_index=episode_index,
                            total_episodes=len(episodes),
                            cell_id=cell_id,
                            model=model_name,
                            invalid_count=int(invalid.sum()),
                            total_rows=int(invalid.size),
                            reason=reason,
                        )
                labels = episode.query_y[0].cpu().numpy()
                z = episode.query_z[0].cpu().numpy()
                common = {
                    "episode_id": manifest_row["episode_id"],
                    "cell_id": manifest_row["cell_id"],
                    "evaluation_seed": manifest_row["evaluation_seed"],
                    "backend": manifest_row["backend"],
                    "k": manifest_row["k"],
                    "alpha": manifest_row["alpha"],
                    "imbalance_ratio": manifest_row["imbalance_ratio"],
                    "support_size": manifest_row["support_size"],
                    "difference_components": "+".join(manifest_row["difference_components"]),
                }
                for model_name, probability in predictions.items():
                    for index, (target, regime, value) in enumerate(zip(labels, z, probability, strict=True)):
                        prediction_writer.writerow(
                            {**common, "model": model_name, "query_index": index, "regime": int(regime),
                             "target": int(target), "probability": float(value)}
                        )
                    for scope, regime, mask in [("overall", None, np.ones(len(labels), dtype=bool))] + [
                        ("regime", regime, z == regime) for regime in range(episode.num_regimes)
                    ]:
                        metrics = binary_metrics(labels[mask], probability[mask])
                        flattened_diagnostics: dict[str, Any] = {}
                        if scope == "overall" and model_name in diagnostics_by_model:
                            for side, values in diagnostics_by_model[model_name].items():
                                for key, value in values.items():
                                    flattened_diagnostics[f"slot_{side}_{key}"] = (
                                        json.dumps(value) if isinstance(value, list) else value
                                    )
                        metric_rows.append(
                            {
                                **common,
                                "model": model_name,
                                "scope": scope,
                                "regime": regime,
                                **metrics,
                                "prediction_invalid_count": prediction_quality[model_name]["invalid_count"],
                                "prediction_invalid_reason": prediction_quality[model_name]["invalid_reason"],
                                **flattened_diagnostics,
                                "oracle_fallback_count": oracle_info["fallback_count"] if model_name == "oracle_hgb" else 0,
                                "oracle_missing_support_regimes": (
                                    json.dumps(oracle_info["missing_support_regimes"]) if model_name == "oracle_hgb" else "[]"
                                ),
                                "oracle_constant_support_regimes": (
                                    json.dumps(oracle_info["constant_support_regimes"]) if model_name == "oracle_hgb" else "[]"
                                ),
                            }
                        )
                if episode_index == 1 or episode_index % 100 == 0 or offset == 0:
                    elapsed = time.monotonic() - evaluation_started
                    rate = episode_index / max(elapsed, 1e-9)
                    remaining = len(episodes) - episode_index
                    log_progress(
                        "evaluation_progress",
                        completed_episodes=episode_index,
                        total_episodes=len(episodes),
                        completed_cells=cell_number - 1,
                        current_cell_index=cell_number,
                        total_cells=len(cell_order),
                        current_cell_completed=offset + 1,
                        cell_id=cell_id,
                        evaluation_elapsed_seconds=elapsed,
                        episodes_per_second=rate,
                        estimated_seconds_remaining=remaining / max(rate, 1e-9),
                    )
            prediction_destination.flush()
    _write_csv(output / "episode_metrics.csv", metric_rows)
    summary = summarize_episode_metrics(
        metric_rows, bootstrap_samples=config.bootstrap_samples, seed=config.evaluation_seeds[0]
    )
    _write_csv(output / "summary.csv", summary)
    if config.reuse_controls_from is None:
        gate = execution_gate_report(episodes, metric_rows, models={row["model"] for row in metric_rows})
    else:
        # The completed source benchmark already ran the generator-recovery
        # probe and control checks.  Do not fit another HGB probe merely to
        # write an informational gate for a neural-only evaluation.
        source_gate_path = Path(config.reuse_controls_from) / "execution_gate.json"
        source_gate = json.loads(source_gate_path.read_text()) if source_gate_path.exists() else None
        gate = {
            "passed": bool(source_gate and source_gate.get("passed")),
            "checks": {"reused_completed_benchmark_gate": source_gate is not None},
            "controls_reused_from": str(config.reuse_controls_from),
            "source_gate": source_gate,
        }
    (output / "execution_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    log_progress(
        "completed",
        completed_episodes=len(episodes),
        total_episodes=len(episodes),
        total_cells=len(cell_order),
        output_files=sorted(path.name for path in output.iterdir() if path.is_file()),
    )
    (output / "episode_manifest.json").chmod(0o444)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tabpfn-checkpoint")
    parser.add_argument("--oracle-one-hot-checkpoint")
    parser.add_argument("--episodes-per-seed", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--generation-workers", type=int, default=1)
    parser.add_argument(
        "--reuse-controls-from",
        help="Completed benchmark seed directory whose pooled/oracle HGB metrics are reused without refitting.",
    )
    parser.add_argument("--model-name", default="tabpfn", help="Label for the supplied neural checkpoint in outputs.")
    parser.add_argument("--no-persist-episodes", dest="persist_episodes", action="store_false")
    parser.add_argument("--compress-predictions", action="store_true")
    parser.set_defaults(persist_episodes=True)
    parser.add_argument("--no-main-grid", dest="include_main_grid", action="store_false")
    parser.add_argument("--no-mechanisms", dest="include_mechanisms", action="store_false")
    parser.add_argument("--no-scm-grid", dest="include_scm_grid", action="store_false")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-cell-index", type=int)
    parser.add_argument("--cell-index", type=int)
    return parser


def main(argv: list[str] | None = None) -> Path:
    arguments = vars(build_parser().parse_args(argv))
    tabpfn_checkpoint = arguments.pop("tabpfn_checkpoint")
    oracle_checkpoint = arguments.pop("oracle_one_hot_checkpoint")
    config = BenchmarkConfig(**arguments)
    from tfmplayground.models.slot_regime import load_checkpoint_for_inference

    tabpfn = load_checkpoint_for_inference(tabpfn_checkpoint, config.device) if tabpfn_checkpoint else None
    oracle = load_checkpoint_for_inference(oracle_checkpoint, config.device) if oracle_checkpoint else None
    paths = {
        name: path
        for name, path in (("tabpfn", tabpfn_checkpoint), ("oracle_one_hot", oracle_checkpoint))
        if path
    }
    return run_benchmark(config, tabpfn_model=tabpfn, oracle_one_hot_model=oracle, checkpoint_paths=paths)


if __name__ == "__main__":
    print(main())


__all__ = [
    "BenchmarkConfig",
    "analytic_grid",
    "binary_metrics",
    "build_locked_episodes",
    "execution_gate_report",
    "gate_grid",
    "generator_oracle_predictions",
    "mechanism_grid",
    "oracle_hgb_predictions",
    "pooled_hgb_predictions",
    "recovery_score",
    "run_benchmark",
    "scm_grid",
    "slot_diagnostics",
    "summarize_episode_metrics",
    "tabpfn_predictions",
]
