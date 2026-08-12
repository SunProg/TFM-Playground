"""Legacy custom 15-dataset binary diagnostic (not an official TabArena run).

Use ``evaluate_task_posterior_tabarena`` for canonical splits, task metrics,
coverage rules, and Elo aggregation.  This file remains reproducible for the
historical sequential-filter results only.
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tfmplayground.evaluation import TABARENA_TASKS
from tfmplayground.interface import get_feature_preprocessor, init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import load_adaptive_checkpoint
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class TabArenaIntegratedConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    controlled_checkpoint: str = "runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth"
    tabicl_checkpoint: str = "runs/integrated_latent_filter/20260807-full-128x128/tabicl_checkpoint.pth"
    adaptive_checkpoint: str | None = "runs/adaptive_particle_filter/20260808-k4-exact-fallback/adaptive_checkpoint.pth"
    output_dir: str | None = None
    device: str = "cpu"
    cache_directory: str | None = None
    prior_count: int = 128
    update_count: int = 128
    use_all_samples: bool = False
    query_chunk_size: int = 128
    max_n_features: int = 500
    max_n_samples: int = 10_000
    num_mem_chunks: int = 8
    fold: int = 0
    repeat: int = 0
    task_ids: tuple[int, ...] = tuple(TABARENA_TASKS)


def validate_config(config: TabArenaIntegratedConfig) -> None:
    if config.update_count > config.prior_count:
        raise ValueError("Updates cannot exceed prior rows.")
    if config.use_all_samples and config.prior_count == 0:
        raise ValueError("prior_count must be positive to derive an all-samples split ratio.")
    if config.query_chunk_size < 1 or config.num_mem_chunks < 1:
        raise ValueError("Chunk counts must be positive.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def split_prior_update_counts(total: int, prior_count: int, update_count: int) -> tuple[int, int]:
    """Split `total` rows into prior/update counts preserving the configured ratio."""
    if total < 2:
        raise ValueError(f"Need at least 2 training rows, found {total}.")
    task_update_count = (total * update_count) // (prior_count + update_count)
    task_update_count = min(max(task_update_count, 1), total - 1)
    task_prior_count = total - task_update_count
    return task_prior_count, task_update_count


def release_device_memory(device: str) -> None:
    """Release cached accelerator memory so successive model calls don't stack reservations."""
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_train_indices(y: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    """Select a deterministic stratified training context and randomize its chronology."""
    if len(y) < count:
        raise ValueError(f"Need at least {count} training rows, found {len(y)}.")
    if count >= len(y):
        return np.random.default_rng(seed).permutation(len(y))
    selected, _ = train_test_split(
        np.arange(len(y)),
        train_size=count,
        stratify=y,
        random_state=seed,
    )
    return np.random.default_rng(seed + 1).permutation(selected)


@torch.no_grad()
def predict_vanilla(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
) -> np.ndarray:
    model.to(device).eval()
    support_x = torch.as_tensor(train_x, dtype=torch.float32, device=device).unsqueeze(0)
    support_y = torch.as_tensor(train_y, dtype=torch.float32, device=device).unsqueeze(0)
    probabilities = []
    for start in range(0, len(test_x), query_chunk_size):
        query = torch.as_tensor(test_x[start : start + query_chunk_size], dtype=torch.float32, device=device).unsqueeze(
            0
        )
        logits = model(
            (torch.cat((support_x, query), dim=1), support_y),
            train_test_split_index=len(train_x),
            num_mem_chunks=num_mem_chunks,
        )[..., :2]
        probabilities.append(logits.softmax(-1)[0, :, 1].cpu().numpy())
    return np.concatenate(probabilities)


@torch.no_grad()
def predict_integrated(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    prior_count: int,
    update_count: int,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
) -> np.ndarray:
    if update_count > prior_count:
        raise ValueError("Updates cannot exceed prior rows.")
    model.to(device).eval()
    support_x = torch.as_tensor(train_x[:prior_count], dtype=torch.float32, device=device).unsqueeze(0)
    support_y = torch.as_tensor(train_y[:prior_count], dtype=torch.float32, device=device).unsqueeze(0)
    stream_x = torch.as_tensor(
        train_x[prior_count : prior_count + update_count], dtype=torch.float32, device=device
    ).unsqueeze(0)
    stream_y = torch.as_tensor(
        train_y[prior_count : prior_count + update_count], dtype=torch.long, device=device
    ).unsqueeze(0)
    probabilities = []
    for start in range(0, len(test_x), query_chunk_size):
        query = torch.as_tensor(test_x[start : start + query_chunk_size], dtype=torch.float32, device=device).unsqueeze(
            0
        )
        prediction = model(
            support_x,
            support_y,
            stream_x,
            stream_y,
            query,
            num_mem_chunks=num_mem_chunks,
        )
        probabilities.append(prediction.marginal_probabilities()[0, -1, :, 1].cpu().numpy())
    return np.concatenate(probabilities)


@torch.no_grad()
def predict_adaptive(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    vanilla_probability: np.ndarray,
    *,
    prior_count: int,
    update_count: int,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
    ambiguity_override: float | None = None,
) -> tuple[np.ndarray, float]:
    model.to(device).eval()
    support_x = torch.as_tensor(train_x[:prior_count], dtype=torch.float32, device=device).unsqueeze(0)
    support_y = torch.as_tensor(train_y[:prior_count], dtype=torch.float32, device=device).unsqueeze(0)
    stream_x = torch.as_tensor(
        train_x[prior_count : prior_count + update_count], dtype=torch.float32, device=device
    ).unsqueeze(0)
    stream_y = torch.as_tensor(
        train_y[prior_count : prior_count + update_count], dtype=torch.long, device=device
    ).unsqueeze(0)
    probabilities = []
    alphas = []
    for start in range(0, len(test_x), query_chunk_size):
        query = torch.as_tensor(test_x[start : start + query_chunk_size], dtype=torch.float32, device=device).unsqueeze(
            0
        )
        prediction = model(
            support_x,
            support_y,
            stream_x,
            stream_y,
            query,
            num_mem_chunks=num_mem_chunks,
            ambiguity_override=ambiguity_override,
        )
        particle = prediction.particle_marginal_probabilities()[0, -1, :, 1].cpu().numpy()
        alpha = float(prediction.ambiguity_probability[0].cpu())
        base = vanilla_probability[start : start + len(particle)]
        probabilities.append((1 - alpha) * base + alpha * particle)
        alphas.append(alpha)
    return np.concatenate(probabilities), float(np.mean(alphas))


def _metric_rows(dataset: str, task_id: int, y_true: np.ndarray, predictions: dict[str, np.ndarray]):
    rows = []
    for model_name, probability in predictions.items():
        rows.append(
            {
                "dataset": dataset,
                "task_id": task_id,
                "model": model_name,
                "roc_auc": roc_auc_score(y_true, probability),
                "balanced_accuracy": balanced_accuracy_score(y_true, probability >= 0.5),
                "test_rows": len(y_true),
            }
        )
    return rows


def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"datasets_evaluated": int(metrics.dataset.nunique()), "models": {}}
    for model_name, group in metrics.groupby("model", sort=False):
        summary["models"][model_name] = {
            "mean_roc_auc": float(group.roc_auc.mean()),
            "sem_roc_auc": float(group.roc_auc.sem()) if len(group) > 1 else 0.0,
            "mean_balanced_accuracy": float(group.balanced_accuracy.mean()),
            "sem_balanced_accuracy": float(group.balanced_accuracy.sem()) if len(group) > 1 else 0.0,
        }
    pivot = metrics.pivot(index="dataset", columns="model", values="roc_auc")
    if "vanilla" in pivot:
        for model_name in (name for name in pivot if name != "vanilla"):
            difference = pivot[model_name] - pivot["vanilla"]
            summary["models"][model_name]["mean_roc_auc_delta_vs_vanilla"] = float(difference.mean())
            summary["models"][model_name]["wins_vs_vanilla"] = int((difference > 0).sum())
    return summary


def run(config: TabArenaIntegratedConfig) -> Path:
    validate_config(config)
    set_randomness_seed(config.seed)
    if config.cache_directory is not None:
        set_root_cache_directory(config.cache_directory)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "integrated_tabarena" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    vanilla = init_model_from_state_dict_file(config.official_checkpoint)
    controlled, controlled_metadata = load_integrated_checkpoint(config.controlled_checkpoint)
    tabicl, tabicl_metadata = load_integrated_checkpoint(config.tabicl_checkpoint)
    models = {"controlled": controlled, "tabicl": tabicl}
    adaptive = None
    adaptive_metadata = None
    adaptive_name = None
    if config.adaptive_checkpoint is not None:
        adaptive, adaptive_metadata = load_adaptive_checkpoint(config.adaptive_checkpoint)
        adaptive_name = f"adaptive_k{adaptive_metadata['architecture']['particle_count']}"
    metric_rows = []
    task_rows = []
    for position, task_id in enumerate(config.task_ids):
        task_record: dict[str, Any] = {"task_id": task_id, "status": "skipped", "reason": None}
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                task_record["reason"] = "not_classification"
                task_rows.append(task_record)
                continue
            dataset = task.get_dataset(download_data=False)
            task_record["dataset"] = str(dataset.name)
            n_features = int(dataset.qualities["NumberOfFeatures"])
            n_samples = int(dataset.qualities["NumberOfInstances"])
            task_record.update({"n_features": n_features, "n_samples": n_samples})
            if n_features > config.max_n_features or n_samples > config.max_n_samples:
                task_record["reason"] = "size_limit"
                task_rows.append(task_record)
                continue
            x, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            train_indices, test_indices = task.get_train_test_split_indices(fold=config.fold, repeat=config.repeat)
            label_encoder = LabelEncoder().fit(y.iloc[train_indices])
            if len(label_encoder.classes_) != 2:
                task_record["reason"] = "not_binary"
                task_rows.append(task_record)
                continue
            y_train = label_encoder.transform(y.iloc[train_indices])
            y_test = label_encoder.transform(y.iloc[test_indices])
            if config.use_all_samples:
                task_prior_count, task_update_count = split_prior_update_counts(
                    len(train_indices), config.prior_count, config.update_count
                )
            else:
                task_prior_count, task_update_count = config.prior_count, config.update_count
            context_count = task_prior_count + task_update_count
            selected = select_train_indices(y_train, context_count, seed=config.seed + task_id + position)
            train_frame = x.iloc[train_indices].iloc[selected]
            test_frame = x.iloc[test_indices]
            preprocessor = get_feature_preprocessor(train_frame)
            train_x = preprocessor.fit_transform(train_frame)
            test_x = preprocessor.transform(test_frame)
            selected_y = y_train[selected]
            predictions = {
                "vanilla": predict_vanilla(
                    vanilla,
                    train_x,
                    selected_y,
                    test_x,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
            }
            release_device_memory(config.device)
            for model_name, model in models.items():
                predictions[model_name] = predict_integrated(
                    model,
                    train_x,
                    selected_y,
                    test_x,
                    prior_count=task_prior_count,
                    update_count=task_update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                release_device_memory(config.device)
            if adaptive is not None and adaptive_name is not None:
                predictions[adaptive_name], adaptive_alpha = predict_adaptive(
                    adaptive,
                    train_x,
                    selected_y,
                    test_x,
                    predictions["vanilla"],
                    prior_count=task_prior_count,
                    update_count=task_update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                task_record["adaptive_alpha"] = adaptive_alpha
                release_device_memory(config.device)
            metric_rows.extend(_metric_rows(str(dataset.name), task_id, y_test, predictions))
            task_record.update(
                {
                    "status": "evaluated",
                    "reason": None,
                    "train_rows_used": context_count,
                    "prior_rows": task_prior_count,
                    "update_rows": task_update_count,
                    "test_rows": len(y_test),
                    "processed_features": train_x.shape[1],
                }
            )
        except Exception as error:  # preserve a complete benchmark audit trail
            task_record["reason"] = f"{type(error).__name__}: {error}"
        finally:
            release_device_memory(config.device)
        task_rows.append(task_record)
        pd.DataFrame(task_rows).to_csv(output_dir / "task_status.csv", index=False)
        if metric_rows:
            pd.DataFrame(metric_rows).to_csv(output_dir / "per_dataset_metrics.csv", index=False)

    pd.DataFrame(task_rows).to_csv(output_dir / "task_status.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(output_dir / "per_dataset_metrics.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No eligible binary diagnostic tasks were evaluated; see task_status.csv.")
    summary = summarize(metrics)
    evaluated = [row for row in task_rows if row["status"] == "evaluated"]
    train_rows_used = [row["train_rows_used"] for row in evaluated]
    summary["protocol"] = {
        "suite": "custom-binary-tabarena-like-diagnostic",
        "official_tabarena": False,
        "subset": "historical custom binary task list within configured size limits",
        "use_all_samples": config.use_all_samples,
        "training_rows_min": min(train_rows_used),
        "training_rows_max": max(train_rows_used),
        "training_rows_mean": float(np.mean(train_rows_used)),
        "configured_prior_count": config.prior_count,
        "configured_update_count": config.update_count,
        "fold": config.fold,
        "repeat": config.repeat,
        "aggregation": "unweighted mean AUC over datasets (non-TabArena aggregation)",
    }
    summary["checkpoint_metadata"] = {
        "controlled_stage": controlled_metadata["stage"],
        "controlled_source_sha256": controlled_metadata["source_checkpoint_sha256"],
        "tabicl_stage": tabicl_metadata["stage"],
        "tabicl_source_sha256": tabicl_metadata["source_checkpoint_sha256"],
    }
    if adaptive_metadata is not None:
        summary["checkpoint_metadata"].update(
            {
                "adaptive_particles": adaptive_metadata["architecture"]["particle_count"],
                "adaptive_source_sha256": adaptive_metadata["source_checkpoint_sha256"],
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--controlled-checkpoint",
        default="runs/integrated_latent_filter/20260807-full-128x128/selected_checkpoint.pth",
    )
    parser.add_argument(
        "--tabicl-checkpoint",
        default="runs/integrated_latent_filter/20260807-full-128x128/tabicl_checkpoint.pth",
    )
    parser.add_argument(
        "--adaptive-checkpoint",
        default="runs/adaptive_particle_filter/20260808-k4-exact-fallback/adaptive_checkpoint.pth",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--prior-count", type=int, default=128)
    parser.add_argument("--update-count", type=int, default=128)
    parser.add_argument(
        "--use-all-samples",
        action="store_true",
        help=(
            "Use every available training row per task instead of a fixed "
            "prior_count+update_count context, splitting rows into prior/update "
            "in the configured ratio."
        ),
    )
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--max-n-features", type=int, default=500)
    parser.add_argument("--max-n-samples", type=int, default=10_000)
    parser.add_argument("--num-mem-chunks", type=int, default=8)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    output_dir = run(TabArenaIntegratedConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote TabArena evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
