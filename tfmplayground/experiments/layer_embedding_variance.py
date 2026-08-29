"""Per-layer nanoTabPFN embedding variance under bootstrap resampling of the support set.

``tfmplayground.experiments.support_resampling.build_ensemble`` already draws
bootstrap resamples of the support set, runs one batched forward pass, and
exposes per-layer query-embedding dispersion across members
(``ResampleEnsemble.representation_dispersion`` /
``scale_free_representation_dispersion`` / ``effective_rank``) -- that machinery
was built for the gate evaluations in ``evaluate_resampling_synthetic.py`` and
``evaluate_resampling_tabarena.py`` (see ``SUPPORT_RESAMPLING_VARIANCE_TRIAL.md``
for why it exists and what it found). This module is a standalone report: it
reuses that machinery unchanged across a set of datasets/episodes and writes a
flat per-layer variance table, rather than scoring arms against a ground-truth
gate.

Two dataset sources, both scheme="bootstrap":

- Synthetic held-out-family episodes (``ambiguous``/``identifiable``/``noisy``),
  the same conditions used by the stage-1 gate evaluation.
- The 20 official TabArena binary tasks, using the identical deterministic
  context-selection and two-stage preprocessing protocol as
  ``SupportResamplingClassifier`` (AutoGluon's feature generator, then the
  nanoTabPFN feature preprocessor), so results are directly comparable to
  every other TabArena report in this repository.

Query labels are never read: dispersion comes from query *embeddings*, exactly
as in ``support_resampling.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.continuous_episodes import HELDOUT_REGIME, sample_episode
from tfmplayground.experiments.support_resampling import ResampleEnsemble, build_ensemble
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.support_resampling_interface import SupportResamplingClassifier

#: The 20 official TabArena binary tasks, in the order also used by
#: ``evaluate_resampling_tabarena.py``'s ``TABARENA_BINARY_TASK_NAMES``.
TABARENA_TASK_IDS = (
    363619,
    363621,
    363623,
    363624,
    363626,
    363629,
    363632,
    363671,
    363674,
    363676,
    363681,
    363682,
    363684,
    363689,
    363691,
    363694,
    363696,
    363700,
    363706,
    363712,
)


@dataclass(frozen=True)
class LayerVarianceConfig:
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    device: str = "cpu"
    seed: int = 2402
    members: int = 32
    episodes_per_condition: int = 20
    support_size: int = 128
    query_count: int = 6
    context_size: int = 1024
    num_mem_chunks: int = 8
    tabarena_query_cap: int = 256


def _layer_rows(ensemble: ResampleEnsemble, *, source: str, dataset: str) -> list[dict[str, Any]]:
    """One row per transformer block: query-averaged dispersion statistics.

    ``support_raw_variance`` / ``support_scale_free_variance`` are the same
    statistic computed over the *support/context* rows' own embeddings instead
    of the query rows' -- i.e. how much a support row's representation moves
    when the support set itself is bootstrap-resampled, grouped by original
    row identity (see ``ResampleEnsemble.support_representation_dispersion``).
    Present only when the ensemble was built with ``capture_support=True``.
    """
    raw = ensemble.representation_dispersion().mean(dim=-1)
    scale_free = ensemble.scale_free_representation_dispersion().mean(dim=-1)
    rank = ensemble.effective_rank().mean(dim=-1)
    num_layers = ensemble.num_layers
    has_support = ensemble.layer_support_embeddings is not None
    support_raw = ensemble.support_representation_dispersion() if has_support else None
    support_scale_free = ensemble.support_scale_free_representation_dispersion() if has_support else None
    return [
        {
            "source": source,
            "dataset": dataset,
            "layer_index": layer,
            "layer_fraction": layer / max(num_layers - 1, 1),
            "raw_variance": float(raw[layer]),
            "scale_free_variance": float(scale_free[layer]),
            "effective_rank": float(rank[layer]),
            "support_raw_variance": float(support_raw[layer]) if has_support else None,
            "support_scale_free_variance": float(support_scale_free[layer]) if has_support else None,
        }
        for layer in range(num_layers)
    ]


def synthetic_layer_variance(model: NanoTabPFNModel, config: LayerVarianceConfig) -> list[dict[str, Any]]:
    """Bootstrap layer-variance rows over held-out-family synthetic episodes."""
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []
    for condition in ("ambiguous", "identifiable", "noisy"):
        for episode_index in range(config.episodes_per_condition):
            episode = sample_episode(
                rng,
                regime=HELDOUT_REGIME,
                condition=condition,
                batch_size=1,
                support_size=config.support_size,
                query_count=config.query_count,
                device=config.device,
            )
            support_x, support_y, query_x = episode.support_x[0], episode.support_y[0], episode.query_x[0]
            ensemble = build_ensemble(
                model,
                support_x,
                support_y,
                query_x,
                scheme="bootstrap",
                members=config.members,
                seed=config.seed + episode_index,
                num_mem_chunks=config.num_mem_chunks,
                compute_gradient=False,
                capture_support=True,
            )
            rows.extend(_layer_rows(ensemble, source="synthetic", dataset=f"{condition}[{episode_index}]"))
    return rows


def tabarena_layer_variance(config: LayerVarianceConfig, task_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    """Bootstrap layer-variance rows over the official TabArena binary tasks.

    Reuses ``SupportResamplingClassifier`` purely for its context selection and
    preprocessing (``_fit_common`` / ``feature_preprocessor_`` /
    ``_context_tensors``) -- the identical deterministic protocol every other
    TabArena report in this repository uses -- then calls ``build_ensemble``
    directly to read off the per-layer embeddings, since the classifier's own
    ``predict_proba`` only surfaces probability dispersion, not representation
    dispersion.
    """
    from autogluon.features import AutoMLPipelineFeatureGenerator
    from tabarena.benchmark.task.openml.spec import OpenMLTaskSpec

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = OpenMLTaskSpec(task_id).load()
        dataset_name = task.dataset_name or str(task_id)
        try:
            train_indices, test_indices = task.get_split_indices(fold=0, repeat=0, sample=0)
            X_train, X_test = task.X.iloc[train_indices], task.X.iloc[test_indices]
            y_train = task.y.iloc[train_indices]
            preprocessor = AutoMLPipelineFeatureGenerator(verbosity=0)
            X_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)

            arm = SupportResamplingClassifier(
                config.checkpoint,
                context_size=config.context_size,
                random_state=0,
                device=config.device,
                num_mem_chunks=config.num_mem_chunks,
                scheme="bootstrap",
                members=config.members,
                ensemble_seed=config.seed,
            )
            arm.fit(X_train, y_train)
            support_x, support_y = arm._context_tensors()
            support_x, support_y = support_x[0], support_y[0]
            query = np.asarray(arm.feature_preprocessor_.transform(X_test), dtype=np.float32)
            if hasattr(query, "toarray"):
                query = query.toarray()
            query = torch.as_tensor(query[: config.tabarena_query_cap], device=arm.device_)

            ensemble = build_ensemble(
                arm.model_,
                support_x,
                support_y,
                query,
                scheme="bootstrap",
                members=config.members,
                seed=config.seed,
                num_mem_chunks=config.num_mem_chunks,
                compute_gradient=False,
                capture_support=True,
            )
            rows.extend(_layer_rows(ensemble, source="tabarena", dataset=dataset_name))
        except torch.cuda.OutOfMemoryError as error:
            print(f"  {dataset_name} ({task_id}): skipped, out of memory ({error})", flush=True)
        finally:
            del task
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def _write_report(
    rows: list[dict[str, Any]], config: LayerVarianceConfig, task_ids: tuple[int, ...], output_dir: str
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "layer_variance.csv", index=False)
    summary = (
        frame.groupby(["source", "layer_index"], as_index=False)
        .agg(
            layer_fraction=("layer_fraction", "first"),
            raw_variance_mean=("raw_variance", "mean"),
            scale_free_variance_mean=("scale_free_variance", "mean"),
            effective_rank_mean=("effective_rank", "mean"),
            support_raw_variance_mean=("support_raw_variance", "mean"),
            support_scale_free_variance_mean=("support_scale_free_variance", "mean"),
        )
        .to_dict(orient="records")
        if not frame.empty
        else []
    )
    result = {
        "config": asdict(config),
        "task_ids": list(task_ids),
        "row_count": len(rows),
        "summary": summary,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return output.resolve()


def run_layer_variance_report(
    config: LayerVarianceConfig,
    output_dir: str,
    *,
    task_ids: tuple[int, ...] = TABARENA_TASK_IDS,
    skip_tabarena: bool = False,
) -> Path:
    model = init_model_from_state_dict_file(str(Path(config.checkpoint).expanduser().resolve()))
    model = model.to(config.device).requires_grad_(False).eval()

    print("evaluating synthetic episodes", flush=True)
    rows = synthetic_layer_variance(model, config)

    used_task_ids: tuple[int, ...] = ()
    if not skip_tabarena:
        used_task_ids = task_ids
        for index, task_id in enumerate(task_ids, start=1):
            print(f"[{index}/{len(task_ids)}] evaluating tabarena task {task_id}", flush=True)
            rows.extend(tabarena_layer_variance(config, (task_id,)))

    return _write_report(rows, config, used_task_ids, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = LayerVarianceConfig()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--device", default=defaults.device)
    for name in (
        "seed",
        "members",
        "episodes_per_condition",
        "support_size",
        "query_count",
        "context_size",
        "num_mem_chunks",
        "tabarena_query_cap",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--task-ids", default=",".join(str(task_id) for task_id in TABARENA_TASK_IDS))
    parser.add_argument("--skip-tabarena", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    task_ids = tuple(int(task_id) for task_id in arguments.task_ids.split(",") if task_id.strip())
    config_fields = {field for field in LayerVarianceConfig.__dataclass_fields__}
    config = LayerVarianceConfig(**{key: value for key, value in vars(arguments).items() if key in config_fields})
    destination = run_layer_variance_report(
        config, arguments.output_dir, task_ids=task_ids, skip_tabarena=arguments.skip_tabarena
    )
    print(f"Wrote layer embedding variance report to {destination}")
