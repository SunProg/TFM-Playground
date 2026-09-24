#!/usr/bin/env python3
"""sklearn and TabPFN (v2.2/v2.6/v3, with and without finetuning) baselines
on the v3 pilot's own synthetic evaluation bank.

Reconstructs the identical held-out episode bank every v3 pilot cell is
scored against (PilotConfig defaults fix the seed, so the episodes match
exactly -- see pretrain_multiregime_v3.evaluation_bank), and evaluates each
baseline per-episode: sklearn models fit on the episode's support set (its
"training set"), TabPFN either in-context (no finetuning) or fine-tuned
fresh per episode on that same support set, all scored on the episode's
query set. Metrics match pretrain_multiregime_v3.metrics (log_loss, brier,
accuracy, ece_10) plus roc_auc, averaged per family, so results are
directly comparable to the pilot's own cell result.json files.

Mirrors finetune_scm_tabpfn_official.py's per-episode finetuning pattern
(a fresh finetuned copy of the checkpoint per episode) rather than adding a
new one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfmplayground.experiments.multiregime_v3 import FAMILIES, OriginalPrior
from tfmplayground.experiments.pretrain_multiregime_v3 import PilotConfig, evaluation_bank, source_hash
from tfmplayground.experiments.pretrain_multiregime_v3 import metrics as pilot_metrics

# v2.6/v3 need an explicit local checkpoint to bypass the gated hosted-weights
# download, which fails non-interactively; on CREATE, v2.2's default resolves
# through a cache that is already warm on that host. Overridable per-host via
# --checkpoint-v22/--checkpoint-v26/--checkpoint-v3 (see build_tabpfn).
DEFAULT_CHECKPOINTS = {
    "v2.2": None,
    "v2.6": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "v3": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
}


def episode_arrays(episode):
    support_x = episode.support_x[0].numpy().astype(np.float32)
    support_y = episode.support_y[0].numpy().reshape(-1).astype(np.int64)
    query_x = episode.query_x[0].numpy().astype(np.float32)
    query_y = episode.query_y[0].numpy().reshape(-1).astype(np.int64)
    return support_x, support_y, query_x, query_y


def positive_probability(model, query_x: np.ndarray) -> np.ndarray:
    """Column for class 1, or all-zero if a degenerate support set never saw it."""
    proba = np.asarray(model.predict_proba(query_x))
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(query_x))


def full_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    result = pilot_metrics(y, p)
    result["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return result


def package_versions() -> dict[str, str | None]:
    """Return versions relevant to the baseline and episode generator."""
    distributions = (
        "numpy",
        "scipy",
        "torch",
        "scikit-learn",
        "tabicl",
        "tabpfn",
        "tabpfn-contrib",
    )
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_pilot_config(path: Path, *, device: str, validation_episodes: int | None) -> tuple[PilotConfig, dict]:
    metadata = json.loads(path.read_text())
    config_data = metadata.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"Pilot config {path} has no object-valued 'config' field.")
    config_data = dict(config_data)
    config_data["device"] = device
    if validation_episodes is not None:
        config_data["validation_episodes"] = validation_episodes
    allowed = set(PilotConfig.__dataclass_fields__)
    unknown = sorted(set(config_data) - allowed)
    if unknown:
        raise ValueError(f"Pilot config {path} has unknown fields: {unknown}")
    return PilotConfig(**config_data), metadata


def load_expected_hashes(path: Path, families: tuple[str, ...]) -> dict[str, list[str]]:
    metadata = json.loads(path.read_text())
    hashes = metadata.get("evaluation_bank_hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"Pilot result {path} has no evaluation_bank_hashes record.")
    expected = {}
    for family in families:
        family_hashes = hashes.get(family)
        if not isinstance(family_hashes, list) or not all(isinstance(item, str) for item in family_hashes):
            raise ValueError(f"Pilot result {path} has no valid hashes for family {family!r}.")
        expected[family] = family_hashes
    return expected


def write_environment_manifest(path: Path, *, args, config: PilotConfig, pilot_metadata: dict) -> dict:
    manifest = {
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": package_versions(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "device": args.device,
        "architecture": config.architecture(),
        "seed": config.seed,
        "source_hash": source_hash(),
        "pilot_source_hash": pilot_metadata.get("source_hash"),
        "pilot_config": jsonable(args.pilot_config),
        "pilot_result": jsonable(args.pilot_result),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def pooled_auc(predictions: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    if not predictions:
        return None
    labels = np.concatenate([labels for labels, _ in predictions])
    probabilities = np.concatenate([probabilities for _, probabilities in predictions])
    if len(np.unique(labels)) != 2:
        return None
    value = float(roc_auc_score(labels, probabilities))
    return value if np.isfinite(value) else None


def build_sklearn_models(seed: int) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "random_forest": RandomForestClassifier(random_state=seed),
    }


def build_tabpfn(version: str, *, device: str, finetune: bool, args, validation_split_ratio: float | None = None):
    from tabpfn.constants import ModelVersion

    model_version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[version]
    checkpoint = args.checkpoints[version]
    if not finetune:
        from tabpfn import TabPFNClassifier

        kwargs = dict(
            device=device,
            random_state=0,
            n_estimators=8,
            softmax_temperature=0.9,
            balance_probabilities=False,
            average_before_softmax=False,
            fit_mode="fit_preprocessors",
            show_progress_bar=False,
        )
        if checkpoint:
            return TabPFNClassifier(model_path=checkpoint, **kwargs)
        return TabPFNClassifier.create_default_for_version(model_version, **kwargs)

    from tabpfn.finetuning import FinetunedTabPFNClassifier

    extra = {"model_path": checkpoint} if checkpoint else {}
    return FinetunedTabPFNClassifier(
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        validation_split_ratio=(
            args.validation_split_ratio if validation_split_ratio is None else validation_split_ratio
        ),
        n_finetune_ctx_plus_query_samples=128,
        finetune_ctx_query_split_ratio=0.2,
        random_state=0,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        validation_frequency=1,
        min_delta=1e-4,
        grad_clip_value=1.0,
        use_lr_scheduler=True,
        lr_warmup_only=False,
        n_estimators_finetune=args.n_estimators_finetune,
        n_estimators_validation=args.n_estimators_validation,
        n_estimators_final_inference=args.n_estimators_final_inference,
        use_activation_checkpointing=True,
        save_checkpoint_interval=None,
        use_fixed_preprocessing_seed=True,
        extra_classifier_kwargs={
            **extra,
            "n_estimators": args.n_estimators_final_inference,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "show_progress_bar": False,
        },
        eval_metric="log_loss",
        model_version=model_version,
    )


def run(args) -> dict:
    config, pilot_metadata = load_pilot_config(
        args.pilot_config,
        device=args.device,
        validation_episodes=args.validation_episodes,
    )
    original = OriginalPrior(config.generator())
    bank = evaluation_bank(config, original)
    families = FAMILIES if not args.families else tuple(args.families)

    expected_hashes = load_expected_hashes(args.pilot_result, families)
    episode_hashes = {family: [episode.tensor_hash() for episode in bank[family]] for family in families}
    mismatches = {
        family: {
            "expected": expected_hashes[family],
            "generated": episode_hashes[family],
        }
        for family in families
        if episode_hashes[family] != expected_hashes[family]
    }
    if mismatches:
        raise RuntimeError(
            "Evaluation-bank hashes do not match the CREATE pilot; refusing to produce cross-group metrics: "
            + json.dumps(mismatches)
        )

    environment_manifest = write_environment_manifest(
        args.environment_manifest,
        args=args,
        config=config,
        pilot_metadata=pilot_metadata,
    )

    sklearn_models = build_sklearn_models(config.seed)
    tabpfn_versions = tuple(args.versions) if args.versions else ("v2.2", "v2.6", "v3")

    predictions = defaultdict(list)

    def evaluate_one(
        name,
        fit_predict,
        family,
        episode_index,
        episode_hash,
        support_x,
        support_y,
        query_x,
        query_y,
    ):
        t0 = time.monotonic()
        row = {
            "model": name,
            "family": family,
            "episode": episode_index,
            "episode_hash": episode_hash,
            "query_rows": len(query_y),
        }
        try:
            probability = np.asarray(fit_predict(support_x, support_y, query_x), dtype=np.float64).reshape(-1)
            if len(probability) != len(query_y):
                raise ValueError(f"predicted {len(probability)} probabilities for {len(query_y)} query rows")
            if not np.isfinite(probability).all():
                raise ValueError("predicted probabilities contain non-finite values")
            row.update(full_metrics(query_y, probability))
            predictions[(name, family)].append((query_y.copy(), probability.copy()))
        except Exception as error:  # one bad episode (e.g. a too-imbalanced support
            # set collapsing a finetuning validation split) must not lose every
            # other already-computed result -- record it and keep going.
            row["error"] = f"{type(error).__name__}: {error}"
        row["fit_seconds"] = time.monotonic() - t0
        print(json.dumps(row), flush=True)
        return row

    rows = []
    started = time.monotonic()
    for family in families:
        for episode_index, episode in enumerate(bank[family]):
            support_x, support_y, query_x, query_y = episode_arrays(episode)

            for name, model in sklearn_models.items():
                def fit_predict(sx, sy, qx, model=model):
                    model.fit(sx, sy)
                    return positive_probability(model, qx)

                rows.append(
                    evaluate_one(
                        name,
                        fit_predict,
                        family,
                        episode_index,
                        episode.tensor_hash(),
                        support_x,
                        support_y,
                        query_x,
                        query_y,
                    )
                )

            for version in tabpfn_versions:
                for finetune in (False, True):
                    name = f"tabpfn-{version}" + ("-finetuned" if finetune else "")

                    def fit_predict(sx, sy, qx, version=version, finetune=finetune):
                        validation_split_ratio = None
                        if finetune:
                            _, class_counts = np.unique(sy, return_counts=True)
                            # A stratified validation split is undefined when a
                            # support class has only one row. Train on all
                            # support rows for this episode and disable only
                            # the internal validation/early-stopping path.
                            if len(class_counts) < 2 or int(class_counts.min()) < 2:
                                validation_split_ratio = 0.0
                        classifier = build_tabpfn(
                            version,
                            device=args.device,
                            finetune=finetune,
                            args=args,
                            validation_split_ratio=validation_split_ratio,
                        )
                        classifier.fit(sx, sy)
                        probability = positive_probability(classifier, qx)
                        del classifier
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return probability

                    rows.append(
                        evaluate_one(
                            name,
                            fit_predict,
                            family,
                            episode_index,
                            episode.tensor_hash(),
                            support_x,
                            support_y,
                            query_x,
                            query_y,
                        )
                    )

            # Incremental checkpoint: a crash or kill partway through must not
            # lose every episode already computed, given how long this run is.
            args.output.write_text(
                json.dumps(
                    {
                        "rows": rows,
                        "elapsed_seconds": time.monotonic() - started,
                        "complete": False,
                        "episode_hashes": episode_hashes,
                        "pilot_source_hash": pilot_metadata.get("source_hash"),
                        "source_hash": source_hash(),
                    },
                    indent=2,
                )
                + "\n"
            )

    summary = {}
    for model_name in sorted({row["model"] for row in rows}):
        summary[model_name] = {}
        for family in families:
            family_rows = [
                row for row in rows if row["model"] == model_name and row["family"] == family and "error" not in row
            ]
            failures = sum(
                1 for row in rows if row["model"] == model_name and row["family"] == family and "error" in row
            )
            if not family_rows:
                continue
            family_predictions = predictions.get((model_name, family), [])
            per_episode_auc = [row["auc"] for row in family_rows if row.get("auc") is not None]
            pooled = pooled_auc(family_predictions)
            summary[model_name][family] = {
                metric: float(np.mean([row[metric] for row in family_rows]))
                for metric in ("log_loss", "brier", "accuracy", "ece_10")
            }
            summary[model_name][family]["pooled_auc"] = pooled
            summary[model_name][family]["mean_episode_auc"] = (
                float(np.mean(per_episode_auc)) if per_episode_auc else None
            )
            summary[model_name][family]["episodes"] = len(family_rows)
            summary[model_name][family]["query_rows"] = sum(row["query_rows"] for row in family_rows)
            if failures:
                summary[model_name][family]["failed_episodes"] = failures

    return {
        "config": jsonable({key: value for key, value in vars(args).items() if key != "output"}),
        "pilot_config": {
            "support_size": config.support_size,
            "query_size": config.query_size,
            "max_features": config.max_features,
            "num_groups": config.num_groups,
            "validation_episodes": config.validation_episodes,
            "seed": config.seed,
        },
        "architecture": config.architecture(),
        "seed": config.seed,
        "source_hash": source_hash(),
        "pilot_source_hash": pilot_metadata.get("source_hash"),
        "pilot_config_path": str(args.pilot_config),
        "pilot_result_path": str(args.pilot_result),
        "episode_hashes": episode_hashes,
        "episode_hashes_match": True,
        "package_versions": environment_manifest["package_versions"],
        "environment_manifest": str(args.environment_manifest),
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", nargs="*", choices=FAMILIES, default=None)
    parser.add_argument("--versions", nargs="*", choices=("v2.2", "v2.6", "v3"), default=None)
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=None,
        help="Overrides the pilot's own eval-bank size; only for smoke-testing, not the real comparison run.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-split-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--n-estimators-finetune", type=int, default=2)
    parser.add_argument("--n-estimators-validation", type=int, default=2)
    parser.add_argument("--n-estimators-final-inference", type=int, default=8)
    parser.add_argument("--checkpoint-v22", default=DEFAULT_CHECKPOINTS["v2.2"])
    parser.add_argument("--checkpoint-v26", default=DEFAULT_CHECKPOINTS["v2.6"])
    parser.add_argument("--checkpoint-v3", default=DEFAULT_CHECKPOINTS["v3"])
    args = parser.parse_args()
    args.checkpoints = {"v2.2": args.checkpoint_v22, "v2.6": args.checkpoint_v26, "v3": args.checkpoint_v3}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.environment_manifest is None:
        args.environment_manifest = args.output.with_suffix(".environment.json")
    args.environment_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
