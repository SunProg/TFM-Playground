"""TabArena evaluation under the small-data protocol.

Protocol (mirrors the reference experimental setup):
  * binary-classification TabArena datasets with **no missing values** and **at most 10
    predictors**;
  * datapoints **subsampled to 200** (stratified);
  * **stratified 5-fold cross-validation with 20 repetitions** (100 fits per dataset);
  * fit time and predict time recorded separately. The reference setup measures accumulated
    *training* time, but in-context models (TabPFN, nanoTabPFN) do nearly all their work at
    predict time, so reporting fit time alone would make them look free.

This is a materially different regime from `evaluate_integrated_tabarena.py`, which allowed up
to 500 features and used the full training split (contexts of several thousand rows) on a
single fold. With 200 points and 5-fold CV the context is ~160 rows, so the historical
128:128 prior/update split does not fit; the split is derived from the available rows instead.

Real TabPFN is available via `--include-tabpfn` (needs the optional extra:
`uv sync --extra tabpfn`, which pins `tabpfn==2.2.1` and `scikit-learn` to 1.6.x, matching the
reference setup). It receives the **raw** frame rather than the nanoTabPFN preprocessor's
output, because TabPFN preprocesses internally and is designed to consume raw columns; the
`inputs` column records which each model got.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tfmplayground.evaluation import TABARENA_TASKS
from tfmplayground.experiments.evaluate_integrated_tabarena import (
    predict_adaptive,
    predict_integrated,
    predict_vanilla,
    release_device_memory,
    split_prior_update_counts,
)
from tfmplayground.interface import get_feature_preprocessor, init_model_from_state_dict_file
from tfmplayground.models.adaptive_particle_filter import load_adaptive_checkpoint
from tfmplayground.models.integrated_latent_filter import load_integrated_checkpoint
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class SmallTabArenaConfig:
    seed: int = 2402
    official_checkpoint: str = "checkpoints/nanotabpfn.pth"
    backbone_checkpoints: str = ""
    standalone_checkpoints: str = ""
    include_vanilla: bool = True
    integrated_checkpoints: str = ""
    adaptive_checkpoints: str = ""
    output_dir: str | None = None
    device: str = "cpu"
    cache_directory: str | None = None
    max_predictors: int = 10
    subsample: int = 200
    folds: int = 5
    repeats: int = 20
    query_chunk_size: int = 128
    num_mem_chunks: int = 1
    include_sklearn: bool = True
    include_tabpfn: bool = False
    label_source: str = "real"
    synthetic_hidden: int = 16
    synthetic_depth: int = 2
    synthetic_noise: float = 0.0
    contamination: float = 0.0
    synthetic_seed: int = 2402
    task_ids: tuple[int, ...] = tuple(TABARENA_TASKS)


def eligible_tasks(config: SmallTabArenaConfig) -> list[dict[str, Any]]:
    """Binary classification, no missing values, at most `max_predictors` predictors."""
    selected = []
    for task_id in config.task_ids:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue
            dataset = task.get_dataset(download_data=False)
            qualities = dataset.qualities
            classes = int(qualities["NumberOfClasses"])
            missing = int(qualities.get("NumberOfMissingValues", 0) or 0)
            # OpenML counts the target in NumberOfFeatures.
            predictors = int(qualities["NumberOfFeatures"]) - 1
            if classes != 2 or missing != 0 or predictors > config.max_predictors:
                continue
            selected.append(
                {
                    "task_id": task_id,
                    "dataset": str(dataset.name),
                    "predictors": predictors,
                    "instances": int(qualities["NumberOfInstances"]),
                }
            )
        except Exception:  # keep the audit trail complete rather than aborting discovery
            continue
    return selected


def _build_tabpfn(config: SmallTabArenaConfig):
    """Construct a real TabPFN classifier, imported lazily.

    `tabpfn` is an optional extra (`uv sync --extra tabpfn`), so it must not be imported at
    module scope or this whole module becomes unimportable without it. Package defaults are
    kept for `n_estimators` and friends because the reference setup does not specify them.
    """
    # tabpfn-common-utils[telemetry-interactive] ships PostHog analytics that phone home on
    # import/use, including a server-side config fetch. Opt out before importing so evaluation
    # runs do not emit telemetry.
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "--include-tabpfn requires the optional 'tabpfn' extra: uv sync --extra tabpfn"
        ) from error
    return TabPFNClassifier(device=config.device)


def load_backbone_checkpoint(path: str, *, reference_checkpoint: str):
    """Load a full-backbone fine-tune written by ``finetune_multiregime_backbone``.

    The fine-tune stores a state dict under ``"model"`` but deliberately does
    not duplicate the architecture metadata from the official checkpoint.
    Reconstruct the official architecture first, then overlay the fine-tuned
    parameters.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(
            f"Expected a full-backbone fine-tune checkpoint with a 'model' state dict: {path}"
        )
    model = init_model_from_state_dict_file(reference_checkpoint)
    model.load_state_dict(raw["model"])
    return model


def _artificial_labels(
    frame: pd.DataFrame, *, seed: int, hidden: int, depth: int, noise: float = 0.0
) -> np.ndarray:
    """Relabel real features with a randomly-initialised MLP ("new regime" arm).

    Keeps the real TabArena feature matrix and replaces only the labelling function, so the
    labelling process is the single thing that differs from the real-label arm. The random-MLP
    form is deliberately SCM-like: every filter/particle checkpoint in this repo was trained on
    SCM-style synthetic priors, so this tests whether they do relatively better once the
    "real labels are unlike our training prior" excuse is removed.

    The preprocessor is fit on the whole frame here. That is not leakage: this function *defines*
    the target rather than predicting a held-out one.

    The logit is thresholded at its median so classes come out ~balanced, which keeps ROC AUC
    well-defined and comparable against the real-label arm.
    """
    numeric = np.asarray(
        get_feature_preprocessor(frame).fit_transform(frame), dtype=np.float64
    )
    # standardise so the random projections are not dominated by raw column scale
    centre = numeric.mean(axis=0, keepdims=True)
    spread = numeric.std(axis=0, keepdims=True)
    activations = (numeric - centre) / np.where(spread > 0, spread, 1.0)

    generator = np.random.default_rng(seed)
    for _ in range(depth):
        weight = generator.normal(0.0, 1.0, size=(activations.shape[1], hidden))
        activations = np.tanh(activations @ weight / np.sqrt(activations.shape[1]))
    final = generator.normal(0.0, 1.0, size=(activations.shape[1], 1))
    logit = (activations @ final).ravel()
    labels = (logit > np.median(logit)).astype(np.int64)
    if noise > 0:
        # A noiseless random-MLP target is too easy (RandomForest reached 0.89-0.99 AUC in
        # calibration), which compresses the differences between models and makes the arm
        # non-discriminative. Flipping a calibrated fraction of labels caps the achievable AUC.
        flip = generator.random(labels.shape[0]) < noise
        labels = np.where(flip, 1 - labels, labels)
    return labels


def _subsample(x: pd.DataFrame, y: np.ndarray, size: int, seed: int):
    if len(y) <= size:
        return x, y
    index, _ = train_test_split(
        np.arange(len(y)), train_size=size, stratify=y, random_state=seed
    )
    return x.iloc[index], y[index]


def run(config: SmallTabArenaConfig) -> Path:
    set_randomness_seed(config.seed)
    if config.cache_directory is not None:
        set_root_cache_directory(config.cache_directory)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "tabarena_small" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    vanilla = init_model_from_state_dict_file(config.official_checkpoint) if config.include_vanilla else None
    backbone_models = {}
    for entry in filter(None, config.backbone_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        if name == "vanilla":
            raise ValueError("'vanilla' is reserved for the official checkpoint.")
        backbone_models[name] = load_backbone_checkpoint(path, reference_checkpoint=config.official_checkpoint)
    for entry in filter(None, config.standalone_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        if name == "vanilla" or name in backbone_models:
            raise ValueError(f"Duplicate or reserved standalone checkpoint name: {name!r}.")
        backbone_models[name] = init_model_from_state_dict_file(path)
    integrated = {}
    for entry in filter(None, config.integrated_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        integrated[name] = load_integrated_checkpoint(path)[0]
    adaptive = {}
    for entry in filter(None, config.adaptive_checkpoints.split(",")):
        name, path = entry.split("=", 1)
        adaptive[name] = load_adaptive_checkpoint(path)[0]

    tasks = eligible_tasks(config)
    pd.DataFrame(tasks).to_csv(output_dir / "eligible_tasks.csv", index=False)
    print(f"eligible datasets: {len(tasks)}", flush=True)

    # `rows` is the primary/total evaluation: held-out queries use the real labels, preserving
    # compatibility with the original protocol and the completed c=0 arm.  Contaminated arms
    # also populate `regime_rows`, which scores the same query features under each labelling
    # regime so a model's regime-specific behaviour is visible rather than hidden in one total.
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for task_info in tasks:
        task = openml.tasks.get_task(task_info["task_id"], download_splits=False)
        dataset = task.get_dataset(download_data=False)
        x, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        y = LabelEncoder().fit_transform(y)
        if config.label_source == "synthetic_mlp":
            # Relabel before subsampling so the stratified subsample balances the new labels.
            y = _artificial_labels(
                x,
                seed=config.synthetic_seed + task_info["task_id"],
                hidden=config.synthetic_hidden,
                depth=config.synthetic_depth,
                noise=config.synthetic_noise,
            )
        x, y = _subsample(x, y, config.subsample, config.seed)
        # Multi-regime arm: a second labelling regime, aligned row-for-row with the subsampled
        # features so it can be mixed into a context without disturbing X.
        regime_labels = (
            _artificial_labels(
                x,
                seed=config.synthetic_seed + task_info["task_id"],
                hidden=config.synthetic_hidden,
                depth=config.synthetic_depth,
                noise=config.synthetic_noise,
            )
            if config.contamination > 0
            else None
        )
        splitter = RepeatedStratifiedKFold(
            n_splits=config.folds, n_repeats=config.repeats, random_state=config.seed
        )
        for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y)):
            train_frame, test_frame = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            context_contaminated_rows = 0
            query_contaminated_rows = 0
            query_regimes = np.full(len(y_test), "real", dtype=object)
            query_labels = y_test.copy()
            if regime_labels is not None:
                # Build a genuinely multi-regime episode: the same contamination fraction is
                # applied independently to support/context rows and query rows. The model sees
                # only the mixed context labels; query labels remain hidden until scoring.
                context_contaminated_rows = int(round(len(y_train) * config.contamination))
                query_contaminated_rows = int(round(len(y_test) * config.contamination))
                if context_contaminated_rows or query_contaminated_rows:
                    chooser = np.random.default_rng(
                        config.seed + task_info["task_id"] * 1000 + fold_index
                    )
                    if context_contaminated_rows:
                        context_positions = chooser.choice(
                            len(y_train), size=context_contaminated_rows, replace=False
                        )
                        y_train = y_train.copy()
                        y_train[context_positions] = regime_labels[train_index][context_positions]
                    if query_contaminated_rows:
                        query_positions = chooser.choice(
                            len(y_test), size=query_contaminated_rows, replace=False
                        )
                        query_labels = y_test.copy()
                        query_labels[query_positions] = regime_labels[test_index][query_positions]
                        query_regimes[query_positions] = "synthetic"
                    if len(np.unique(y_train)) < 2 or len(np.unique(query_labels)) < 2:
                        continue  # contaminated context collapsed to one class
            preprocessor = get_feature_preprocessor(train_frame)
            train_x = preprocessor.fit_transform(train_frame)
            test_x = preprocessor.transform(test_frame)
            prior_count, update_count = split_prior_update_counts(len(y_train), 1, 1)

            def record(
                model_name: str,
                probability,
                fit_seconds: float,
                *,
                predict_seconds: float = float("nan"),
                inputs: str = "preprocessed",
                task_info=task_info,
                fold_index=fold_index,
                query_labels=query_labels,
                context_contaminated_rows=context_contaminated_rows,
                query_contaminated_rows=query_contaminated_rows,
                y_train=y_train,
                regime_labels=regime_labels,
                query_regimes=query_regimes,
            ):
                rows.append(
                    {
                        "dataset": task_info["dataset"],
                        "task_id": task_info["task_id"],
                        "predictors": task_info["predictors"],
                        "fold": fold_index,
                        "model": model_name,
                        "roc_auc": roc_auc_score(query_labels, probability),
                        "accuracy": accuracy_score(query_labels, probability >= 0.5),
                        "fit_seconds": fit_seconds,
                        # In-context models do essentially all their work at predict time, so a
                        # training-time-only comparison would make them look free. Recorded
                        # separately rather than folded into fit_seconds.
                        "predict_seconds": predict_seconds,
                        # TabPFN does its own preprocessing and receives the raw frame; the
                        # nano-family models receive the nanoTabPFN preprocessor's output. Kept
                        # explicit so the comparison is not silently ambiguous.
                        "inputs": inputs,
                        "labels": config.label_source,
                        "contamination": config.contamination,
                        "context_contaminated_rows": context_contaminated_rows,
                        "query_contaminated_rows": query_contaminated_rows,
                        "train_rows": len(y_train),
                        "test_rows": len(query_labels),
                    }
                )
                if regime_labels is not None:
                    # Score only the query rows belonging to each regime. This is a genuine
                    # per-regime query evaluation, rather than rescoring the entire query set
                    # against an alternate label vector.
                    for query_regime in ("real", "synthetic"):
                        mask = query_regimes == query_regime
                        regime_query_labels = query_labels[mask]
                        regime_probability = np.asarray(probability)[mask]
                        regime_rows.append(
                            {
                                "dataset": task_info["dataset"],
                                "task_id": task_info["task_id"],
                                "predictors": task_info["predictors"],
                                "fold": fold_index,
                                "model": model_name,
                                "query_regime": query_regime,
                                "roc_auc": (
                                    roc_auc_score(regime_query_labels, regime_probability)
                                    if len(np.unique(regime_query_labels)) == 2
                                    else float("nan")
                                ),
                                "accuracy": accuracy_score(
                                    regime_query_labels, regime_probability >= 0.5
                                ),
                                "inputs": inputs,
                                "labels": config.label_source,
                                "contamination": config.contamination,
                                "context_contaminated_rows": context_contaminated_rows,
                                "query_contaminated_rows": query_contaminated_rows,
                                "train_rows": len(y_train),
                                "test_rows": int(mask.sum()),
                            }
                        )

            vanilla_probability = None
            if vanilla is not None:
                start = time.perf_counter()
                vanilla_probability = predict_vanilla(
                    vanilla, train_x, y_train, test_x,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record("vanilla", vanilla_probability, time.perf_counter() - start)
                release_device_memory(config.device)

            for name, model in backbone_models.items():
                start = time.perf_counter()
                probability = predict_vanilla(
                    model, train_x, y_train, test_x,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record(name, probability, time.perf_counter() - start)
                release_device_memory(config.device)

            for name, model in integrated.items():
                start = time.perf_counter()
                probability = predict_integrated(
                    model, train_x, y_train, test_x,
                    prior_count=prior_count, update_count=update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record(name, probability, time.perf_counter() - start)
                release_device_memory(config.device)

            for name, model in adaptive.items():
                if vanilla_probability is None:
                    raise ValueError("Adaptive checkpoints require --include-vanilla for their vanilla reference.")
                start = time.perf_counter()
                probability, _ = predict_adaptive(
                    model, train_x, y_train, test_x, vanilla_probability,
                    prior_count=prior_count, update_count=update_count,
                    device=config.device,
                    query_chunk_size=config.query_chunk_size,
                    num_mem_chunks=config.num_mem_chunks,
                )
                record(name, probability, time.perf_counter() - start)
                release_device_memory(config.device)

            if config.include_sklearn:
                for name, estimator in (
                    # Unscaled features make lbfgs hit its iteration cap and emit convergence warnings,
                    # which would understate the baseline; scale inside the fold to avoid leakage.
                    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                    ("random_forest", RandomForestClassifier(random_state=config.seed)),
                ):
                    start = time.perf_counter()
                    estimator.fit(train_x, y_train)
                    fit_elapsed = time.perf_counter() - start
                    start = time.perf_counter()
                    probability = estimator.predict_proba(test_x)[:, 1]
                    record(
                        name,
                        probability,
                        fit_elapsed,
                        predict_seconds=time.perf_counter() - start,
                    )

            if config.include_tabpfn:
                classifier = _build_tabpfn(config)
                # Raw frame, not the nanoTabPFN preprocessor output: TabPFN performs its own
                # preprocessing and is designed to consume raw columns.
                start = time.perf_counter()
                classifier.fit(train_frame, y_train)
                fit_elapsed = time.perf_counter() - start
                start = time.perf_counter()
                probability = classifier.predict_proba(test_frame)[:, 1]
                record(
                    "tabpfn",
                    probability,
                    fit_elapsed,
                    predict_seconds=time.perf_counter() - start,
                    inputs="raw",
                )

        pd.DataFrame(rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        done = len({row["dataset"] for row in rows})
        print(f"[{done}/{len(tasks)}] {task_info['dataset']} complete", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    per_dataset = (
        metrics.groupby(["dataset", "model", "labels"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy=("accuracy", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            # min_count=1 so an all-NaN group stays NaN rather than collapsing to 0.0:
            # the nano-family in-context models have no separable fit/predict split, and
            # reporting 0.0 there would read as "prediction is free".
            predict_seconds_total=pd.NamedAgg(
                column="predict_seconds", aggfunc=lambda s: s.sum(min_count=1)
            ),
            folds=("roc_auc", "size"),
        )
    )
    per_dataset.to_csv(output_dir / "per_dataset.csv", index=False)
    overall = (
        per_dataset.groupby(["model", "labels"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            fit_seconds_total=("fit_seconds_total", "sum"),
            predict_seconds_total=pd.NamedAgg(
                column="predict_seconds_total", aggfunc=lambda s: s.sum(min_count=1)
            ),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    overall.to_csv(output_dir / "overall.csv", index=False)
    print(overall.round(4).to_string(index=False), flush=True)
    if regime_rows:
        regime_metrics = pd.DataFrame(regime_rows)
        regime_metrics.to_csv(output_dir / "regime_fold_metrics.csv", index=False)
        regime_per_dataset = (
            regime_metrics.groupby(
                ["dataset", "model", "labels", "query_regime"], as_index=False
            )
            .agg(
                roc_auc=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
                accuracy=("accuracy", "mean"),
                folds=("roc_auc", "size"),
            )
        )
        regime_per_dataset.to_csv(output_dir / "regime_per_dataset.csv", index=False)
        regime_overall = (
            regime_per_dataset.groupby(
                ["model", "labels", "query_regime"], as_index=False
            )
            .agg(
                mean_roc_auc=("roc_auc", "mean"),
                mean_accuracy=("accuracy", "mean"),
                folds=("folds", "sum"),
            )
            .sort_values(["query_regime", "mean_roc_auc"], ascending=[True, False])
        )
        regime_overall.to_csv(output_dir / "regime_overall.csv", index=False)
        print("Per-query-regime results:", flush=True)
        print(regime_overall.round(4).to_string(index=False), flush=True)
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument(
        "--backbone-checkpoints",
        default="",
        help="Comma-separated name=path list of full-backbone fine-tune checkpoints.",
    )
    parser.add_argument(
        "--standalone-checkpoints",
        default="",
        help="Comma-separated name=path list of inference-compatible standalone nanoTabPFN checkpoints.",
    )
    parser.add_argument(
        "--include-vanilla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the official checkpoint alongside supplied models.",
    )
    parser.add_argument(
        "--integrated-checkpoints",
        default="",
        help="Comma-separated name=path list of integrated (no-gate) latent filter checkpoints.",
    )
    parser.add_argument(
        "--adaptive-checkpoints",
        default="",
        help="Comma-separated name=path list of adaptive (gated) particle filter checkpoints.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--max-predictors", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--num-mem-chunks", type=int, default=1)
    parser.add_argument("--include-sklearn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-tabpfn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate real TabPFN (requires: uv sync --extra tabpfn). Off by default so "
            "existing runs stay reproducible."
        ),
    )
    parser.add_argument(
        "--label-source",
        choices=("real", "synthetic_mlp"),
        default="real",
        help="real TabArena labels, or relabel real features with a random MLP (new regime).",
    )
    parser.add_argument("--synthetic-hidden", type=int, default=16)
    parser.add_argument("--synthetic-depth", type=int, default=2)
    parser.add_argument("--synthetic-noise", type=float, default=0.0)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.0,
        help=(
            "Fraction of CONTEXT (training) rows relabelled by the synthetic regime, "
            "and the same fraction of query rows relabelled, building a multi-regime episode. "
            "overall.csv scores the mixed query set; contaminated arms also write per-query-"
            "regime metrics for real and synthetic query rows."
        ),
    )
    parser.add_argument("--synthetic-seed", type=int, default=2402)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> int:
    output = run(SmallTabArenaConfig(**vars(build_parser().parse_args(argv))))
    print(f"Wrote small-protocol TabArena artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
