"""Measure predictive multimodality on real TabArena classification tasks.

This is a diagnostic rather than a leaderboard benchmark.  A hypothesis is a
different, context-trained predictor.  A context is called ambiguous only when
at least two hypotheses retain meaningful posterior weight *and* disagree on
the held-out query region.  Query labels are used only for the final metrics.

The protocol uses the canonical TabArena split 0/repeat 0.  Rows from the
official training partition are divided into a labelled context and a stream;
the official test partition is used only for final queries.  Hypothesis weights
are initialized from context out-of-bag loss and updated with stream labels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
import torch
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tfmplayground.evaluation import TABARENA_TASKS
from tfmplayground.interface import get_feature_preprocessor, init_model_from_state_dict_file


@dataclass(frozen=True)
class MultimodalityConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    cache_directory: str | None = None
    max_n_features: int = 500
    max_n_samples: int = 10_000
    context_sizes: tuple[int, ...] = (32, 64, 128, 256)
    stream_size: int = 32
    query_size: int = 128
    episodes: int = 8
    plausible_weight: float = 0.10
    disagreement_threshold: float = 0.20
    temperature: float = 1.0
    bootstrap_samples: int = 2000
    include_vanilla: bool = True
    task_ids: tuple[int, ...] = tuple(TABARENA_TASKS)


HYPOTHESIS_NAMES = ("logreg", "random_forest", "extra_trees", "hist_gradient_boosting")


def _validate_config(config: MultimodalityConfig) -> None:
    if not config.context_sizes or min(config.context_sizes) < 4:
        raise ValueError("context_sizes must contain values >= 4")
    if config.stream_size < 1 or config.query_size < 1 or config.episodes < 1:
        raise ValueError("stream_size, query_size, and episodes must be positive")
    if not 0 < config.plausible_weight < 1:
        raise ValueError("plausible_weight must be in (0, 1)")
    if not 0 <= config.disagreement_threshold <= 1:
        raise ValueError("disagreement_threshold must be in [0, 1]")
    if config.temperature <= 0 or config.bootstrap_samples < 1:
        raise ValueError("temperature and bootstrap_samples must be positive")


def eligible_tasks(config: MultimodalityConfig) -> list[dict[str, Any]]:
    """Return binary classification tasks inside the configured size limits."""

    selected = []
    for task_id in config.task_ids:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue
            dataset = task.get_dataset(download_data=False)
            qualities = dataset.qualities
            if int(qualities.get("NumberOfClasses", 0)) != 2:
                continue
            n_features = int(qualities["NumberOfFeatures"])
            n_samples = int(qualities["NumberOfInstances"])
            if n_features > config.max_n_features or n_samples > config.max_n_samples:
                continue
            selected.append(
                {
                    "task_id": int(task_id),
                    "dataset": str(dataset.name),
                    "features": n_features,
                    "instances": n_samples,
                }
            )
        except Exception:
            # The caller records failures while loading data; discovery should
            # not make one unavailable OpenML task abort the whole audit.
            continue
    return selected


def _make_hypothesis(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=96, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=96, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=seed)
    raise ValueError(f"unknown hypothesis: {name}")


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("hypothesis returned non-finite probabilities")
    return probability.clip(1e-7, 1 - 1e-7)


def _fit_hypotheses(x: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Fit one bootstrapped predictor per family and score it out of bag."""

    models = []
    losses = []
    oob_counts = []
    rng = np.random.default_rng(seed)
    for offset, name in enumerate(HYPOTHESIS_NAMES):
        model = _make_hypothesis(name, seed + offset)
        oob = np.empty(0, dtype=int)
        # Retry so the bootstrap's out-of-bag rows normally contain both labels.
        for _ in range(32):
            bootstrap = rng.integers(0, len(y), size=len(y))
            in_bag = np.zeros(len(y), dtype=bool)
            in_bag[bootstrap] = True
            candidate_oob = np.flatnonzero(~in_bag)
            if len(np.unique(y[bootstrap])) == 2 and len(np.unique(y[candidate_oob])) == 2:
                oob = candidate_oob
                break
        if len(oob) == 0:
            bootstrap = np.arange(len(y))
            oob = np.arange(len(y))
        model.fit(x[bootstrap], y[bootstrap])
        probabilities = _positive_probability(model, x[oob])
        losses.append(float(log_loss(y[oob], probabilities, labels=[0, 1])))
        oob_counts.append(len(oob))
        models.append(model)
    return models, np.asarray(losses), np.asarray(oob_counts)


def _softmax_log_weights(losses: np.ndarray, temperature: float) -> np.ndarray:
    values = -np.asarray(losses, dtype=float) / temperature
    values -= values.max()
    weights = np.exp(values)
    return weights / weights.sum()


def _update_weights(weights: np.ndarray, probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    log_likelihood = np.log(np.where(y[:, None] == 1, probabilities, 1 - probabilities)).sum(axis=0)
    values = np.log(np.clip(weights, 1e-300, None)) + log_likelihood
    values -= values.max()
    updated = np.exp(values)
    return updated / updated.sum()


def _effective_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(np.exp(-(weights * np.log(np.clip(weights, 1e-300, None))).sum()))


def _pairwise_js(probabilities: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean Bernoulli JS divergence between hypothesis predictions."""

    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = 0.0
    normalizer = 0.0
    for left in range(len(weights)):
        for right in range(left + 1, len(weights)):
            pair_weight = weights[left] * weights[right]
            if pair_weight == 0:
                continue
            p = np.clip(probabilities[:, left], 1e-7, 1 - 1e-7)
            q = np.clip(probabilities[:, right], 1e-7, 1 - 1e-7)
            midpoint = (p + q) / 2
            js = 0.5 * (p * np.log(p / midpoint) + (1 - p) * np.log((1 - p) / (1 - midpoint)))
            js += 0.5 * (q * np.log(q / midpoint) + (1 - q) * np.log((1 - q) / (1 - midpoint)))
            total += float(pair_weight * js.mean())
            normalizer += float(pair_weight)
    return total / normalizer if normalizer else 0.0


def _max_pairwise_disagreement(probabilities: np.ndarray) -> float:
    if probabilities.shape[1] < 2:
        return 0.0
    return float(
        max(
            np.mean(np.abs(probabilities[:, i] - probabilities[:, j]))
            for i in range(probabilities.shape[1])
            for j in range(i + 1, probabilities.shape[1])
        )
    )


def _hypothesis_metrics(
    probabilities: np.ndarray, weights: np.ndarray, config: MultimodalityConfig
) -> dict[str, float | int]:
    plausible = weights >= config.plausible_weight
    plausible_probabilities = probabilities[:, plausible]
    plausible_weights = weights[plausible]
    if len(plausible_weights):
        plausible_weights = plausible_weights / plausible_weights.sum()
    plausible_disagreement = (
        _max_pairwise_disagreement(plausible_probabilities) if plausible_probabilities.shape[1] >= 2 else 0.0
    )
    return {
        "effective_hypothesis_count": _effective_count(weights),
        "plausible_hypothesis_count": int(plausible.sum()),
        "pairwise_js": _pairwise_js(probabilities, weights),
        "plausible_pairwise_js": (
            _pairwise_js(plausible_probabilities, plausible_weights) if plausible_probabilities.shape[1] >= 2 else 0.0
        ),
        "max_pairwise_disagreement": _max_pairwise_disagreement(probabilities),
        "plausible_max_disagreement": plausible_disagreement,
        "ambiguous": int(len(plausible_weights) >= 2 and plausible_disagreement >= config.disagreement_threshold),
    }


def _vanilla_probability(model, context_x, context_y, stream_x, stream_y, query_x, device: str) -> np.ndarray:
    support_x = torch.as_tensor(np.concatenate((context_x, stream_x)), dtype=torch.float32, device=device).unsqueeze(0)
    support_y = torch.as_tensor(np.concatenate((context_y, stream_y)), dtype=torch.float32, device=device).unsqueeze(0)
    query = torch.as_tensor(query_x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(
            (torch.cat((support_x, query), dim=1), support_y),
            train_test_split_index=support_x.shape[1],
            num_mem_chunks=1,
        )[..., :2]
        return logits.softmax(-1)[0, :, 1].cpu().numpy().clip(1e-7, 1 - 1e-7)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        value = float(values.mean()) if len(values) else math.nan
        return value, value
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _sample_episode(
    x: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    context_size: int,
    config: MultimodalityConfig,
    rng: np.random.Generator,
    vanilla_model: Any | None,
    task_id: int,
    dataset_name: str,
    episode: int,
) -> dict[str, Any]:
    required = context_size + config.stream_size
    if len(train_indices) < required:
        raise ValueError(f"training partition has {len(train_indices)} rows; need {required}")
    if len(test_indices) < config.query_size:
        raise ValueError(f"test partition has {len(test_indices)} rows; need {config.query_size}")
    train_y = y[train_indices]
    context_indices, remaining_indices = train_test_split(
        np.arange(len(train_indices)),
        train_size=context_size,
        stratify=train_y,
        random_state=int(rng.integers(2**31 - 1)),
    )
    remaining_y = train_y[remaining_indices]
    stream_indices, _ = train_test_split(
        remaining_indices,
        train_size=config.stream_size,
        stratify=remaining_y,
        random_state=int(rng.integers(2**31 - 1)),
    )
    query_indices = rng.choice(test_indices, size=config.query_size, replace=False)
    context_frame = x.iloc[train_indices[context_indices]]
    stream_frame = x.iloc[train_indices[stream_indices]]
    query_frame = x.iloc[query_indices]
    preprocessor = get_feature_preprocessor(context_frame)
    context_x = np.asarray(preprocessor.fit_transform(context_frame), dtype=np.float32)
    stream_x = np.asarray(preprocessor.transform(stream_frame), dtype=np.float32)
    query_x = np.asarray(preprocessor.transform(query_frame), dtype=np.float32)
    context_y = y[train_indices[context_indices]]
    stream_y = y[train_indices[stream_indices]]
    query_y = y[query_indices]

    models, context_losses, oob_counts = _fit_hypotheses(context_x, context_y, seed=config.seed + task_id + episode)
    context_weights = _softmax_log_weights(context_losses, config.temperature)
    stream_probabilities = np.column_stack([_positive_probability(model, stream_x) for model in models])
    query_probabilities = np.column_stack([_positive_probability(model, query_x) for model in models])
    updated_weights = _update_weights(context_weights, stream_probabilities, stream_y)
    updated_query_probabilities = query_probabilities @ updated_weights
    context_query_probabilities = query_probabilities @ context_weights
    metrics = {
        "task_id": task_id,
        "dataset": dataset_name,
        "episode": episode,
        "context_size": context_size,
        "stream_size": config.stream_size,
        "query_size": config.query_size,
        "context_class_balance": float(context_y.mean()),
        "stream_class_balance": float(stream_y.mean()),
        "query_class_balance": float(query_y.mean()),
        "mean_context_oob_nll": float(context_losses.mean()),
        "min_context_oob_nll": float(context_losses.min()),
        "mean_oob_rows": float(oob_counts.mean()),
    }
    for prefix, probabilities, weights in (
        ("before", query_probabilities, context_weights),
        ("after", query_probabilities, updated_weights),
    ):
        metrics.update(
            {f"{prefix}_{key}": value for key, value in _hypothesis_metrics(probabilities, weights, config).items()}
        )
    metrics.update(
        {
            "mixture_query_nll": float(log_loss(query_y, updated_query_probabilities, labels=[0, 1])),
            "context_weighted_query_nll": float(log_loss(query_y, context_query_probabilities, labels=[0, 1])),
            "best_context_hypothesis_query_nll": float(
                log_loss(query_y, query_probabilities[:, np.argmax(context_weights)], labels=[0, 1])
            ),
            "oracle_best_hypothesis_query_nll": float(
                min(
                    log_loss(query_y, query_probabilities[:, i], labels=[0, 1])
                    for i in range(query_probabilities.shape[1])
                )
            ),
        }
    )
    metrics["mixture_gain_vs_context_weighted"] = metrics["context_weighted_query_nll"] - metrics["mixture_query_nll"]
    metrics["mixture_gain_vs_context_best"] = (
        metrics["best_context_hypothesis_query_nll"] - metrics["mixture_query_nll"]
    )
    if vanilla_model is not None:
        vanilla_probability = _vanilla_probability(
            model=vanilla_model,
            context_x=context_x,
            context_y=context_y,
            stream_x=stream_x,
            stream_y=stream_y,
            query_x=query_x,
            device=config.device,
        )
        metrics["vanilla_query_nll"] = float(log_loss(query_y, vanilla_probability, labels=[0, 1]))
        metrics["mixture_gain_vs_vanilla"] = metrics["vanilla_query_nll"] - metrics["mixture_query_nll"]
    else:
        metrics["vanilla_query_nll"] = math.nan
        metrics["mixture_gain_vs_vanilla"] = math.nan
    metrics["after_disagreement_error_correlation_value"] = float(
        np.mean(np.abs(updated_query_probabilities - query_y))
    )
    metrics["context_weights"] = json.dumps(context_weights.tolist())
    metrics["updated_weights"] = json.dumps(updated_weights.tolist())
    return metrics


def _summarize(metrics: pd.DataFrame, config: MultimodalityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["task_id", "dataset", "context_size"]
    numeric = [
        column
        for column in metrics.columns
        if column not in group_columns + ["episode", "context_weights", "updated_weights"]
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    per_task = metrics.groupby(group_columns, as_index=False)[numeric].mean()
    per_task["predictive_confirmation"] = (
        (per_task["after_ambiguous"] > 0) & (per_task["mixture_gain_vs_context_weighted"] > 0)
    ).astype(int)
    rows = []
    rng = np.random.default_rng(config.seed + 991)
    for context_size, group in per_task.groupby("context_size", sort=True):
        row: dict[str, Any] = {
            "context_size": int(context_size),
            "task_count": int(group.task_id.nunique()),
            "episode_count": int(group.shape[0]),
            "predictively_confirmed_task_count": int(group["predictive_confirmation"].sum()),
            "predictively_confirmed_task_rate": float(group["predictive_confirmation"].mean()),
        }
        for column in (
            "before_ambiguous",
            "after_ambiguous",
            "after_pairwise_js",
            "after_plausible_max_disagreement",
            "mixture_query_nll",
            "mixture_gain_vs_context_weighted",
            "mixture_gain_vs_vanilla",
        ):
            if column in group:
                row[f"mean_{column}"] = float(group[column].mean())
                low, high = _bootstrap_ci(group[column].to_numpy(), rng, config.bootstrap_samples)
                row[f"{column}_ci_low"] = low
                row[f"{column}_ci_high"] = high
        if "after_disagreement_error_correlation_value" in group and len(group) >= 2:
            row["disagreement_error_correlation"] = float(
                np.corrcoef(group.after_plausible_max_disagreement, group.after_disagreement_error_correlation_value)[
                    0, 1
                ]
            )
        else:
            row["disagreement_error_correlation"] = math.nan
        rows.append(row)
    return per_task, pd.DataFrame(rows)


def _write_plots(metrics: pd.DataFrame, overall: pd.DataFrame, output_dir: Path) -> None:
    """Write optional diagnostic plots without making matplotlib a runtime requirement."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped = metrics.groupby("context_size", as_index=False).agg(
        ambiguity=("after_ambiguous", "mean"),
        disagreement=("after_plausible_max_disagreement", "mean"),
        mixture_gain=("mixture_gain_vs_context_weighted", "mean"),
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(grouped.context_size, grouped.ambiguity, marker="o", label="ambiguous episode rate")
    axis.set_xlabel("Labelled context rows")
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ambiguity_vs_context.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(grouped.disagreement, grouped.mixture_gain, s=45)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Plausible hypothesis disagreement")
    axis.set_ylabel("Mixture NLL gain vs context-weighted model")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "disagreement_vs_mixture_gain.png", dpi=160)
    plt.close(figure)

    if not overall.empty:
        overall.to_csv(output_dir / "overall_summary.csv", index=False)


def run(config: MultimodalityConfig) -> Path:
    _validate_config(config)
    if config.cache_directory is not None:
        set_root_cache_directory(config.cache_directory)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "tabarena_multimodality" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    vanilla_model = None
    if config.include_vanilla:
        vanilla_model = init_model_from_state_dict_file(config.official_checkpoint).to(config.device).eval()
    task_rows = []
    metric_rows = []
    tasks = eligible_tasks(config)
    eligible_ids = {entry["task_id"] for entry in tasks}
    for task_id in config.task_ids:
        task_record: dict[str, Any] = {"task_id": int(task_id), "status": "skipped", "reason": "not_eligible"}
        if task_id not in eligible_ids:
            task_rows.append(task_record)
            continue
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset(download_data=False)
            frame, target, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)
            # Fit the label mapping on training labels only.  The query labels
            # must not influence any preprocessing or model construction.
            encoder = LabelEncoder().fit(target.iloc[train_indices])
            y = encoder.transform(target)
            task_record.update(
                {
                    "dataset": str(dataset.name),
                    "features": int(dataset.qualities["NumberOfFeatures"]),
                    "instances": int(dataset.qualities["NumberOfInstances"]),
                }
            )
            for context_size in config.context_sizes:
                for episode in range(config.episodes):
                    try:
                        metric_rows.append(
                            _sample_episode(
                                frame,
                                y,
                                np.asarray(train_indices),
                                np.asarray(test_indices),
                                context_size,
                                config,
                                np.random.default_rng(config.seed + task_id * 10000 + context_size * 100 + episode),
                                vanilla_model,
                                int(task_id),
                                str(dataset.name),
                                episode,
                            )
                        )
                    except ValueError as error:
                        task_record.setdefault("episode_failures", []).append(
                            {"context_size": context_size, "episode": episode, "reason": str(error)}
                        )
            task_record["status"] = "evaluated"
            task_record["reason"] = None
        except Exception as error:
            task_record["reason"] = f"{type(error).__name__}: {error}"
        task_rows.append(task_record)
        pd.DataFrame(task_rows).to_csv(output_dir / "task_status.csv", index=False)
        if metric_rows:
            pd.DataFrame(metric_rows).to_csv(output_dir / "episode_metrics.csv", index=False)
    task_status = pd.DataFrame(task_rows)
    task_status.to_csv(output_dir / "task_status.csv", index=False)
    if not metric_rows:
        raise RuntimeError("No valid episodes were evaluated; inspect task_status.csv")
    metrics = pd.DataFrame(metric_rows)
    per_task, overall = _summarize(metrics, config)
    metrics.to_csv(output_dir / "episode_metrics.csv", index=False)
    per_task.to_csv(output_dir / "per_task_summary.csv", index=False)
    overall.to_csv(output_dir / "overall_summary.csv", index=False)
    _write_plots(metrics, overall, output_dir)
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--max-n-features", type=int, default=500)
    parser.add_argument("--max-n-samples", type=int, default=10_000)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--stream-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--plausible-weight", type=float, default=0.10)
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--include-vanilla", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TABARENA_TASKS))
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    config = MultimodalityConfig(**vars(build_parser().parse_args(argv)))
    output = run(config)
    print(f"Wrote TabArena multimodality artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
