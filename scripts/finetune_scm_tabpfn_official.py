"""Fine-tune official TabPFN checkpoints independently on SCM episodes.

Each episode is treated as one task: the official fine-tuning wrapper adapts a
fresh copy of the selected checkpoint on that episode's support set, then the
adapted model predicts the episode's query rows. This keeps the support/query
boundary intact and avoids pooling incompatible SCM mechanisms across tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig
from tfmplayground.experiments.scm_table_slot_head_sweep import PRIOR_PROFILES, episode_batch


def _episode_metrics(probability: np.ndarray, target: np.ndarray, regime: np.ndarray) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    predicted = probability.argmax(axis=1)

    def metrics(mask: np.ndarray) -> dict[str, float]:
        p = probability[mask]
        y = target[mask]
        return {
            "accuracy": float(np.mean(p.argmax(axis=1) == y)),
            "cross_entropy": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0)))),
            "auc": float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else float("nan"),
        }

    return {
        "overall": metrics(np.ones(len(target), dtype=bool)),
        "majority_z1": metrics(regime == 1),
        "minority_z0": metrics(regime == 0),
    }


def _build_finetuner(args):
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[args.version]
    extra = {"model_path": str(args.checkpoint)} if args.checkpoint else {}
    return FinetunedTabPFNClassifier(
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        validation_split_ratio=args.validation_split_ratio,
        n_finetune_ctx_plus_query_samples=128,
        finetune_ctx_query_split_ratio=0.2,
        random_state=args.random_state,
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
        model_version=version,
    )


def run(args) -> dict:
    prior = PRIOR_PROFILES[args.prior_profile]
    started = time.monotonic()
    rows = []
    for episode_index in range(args.episode_start, args.episode_start + args.episodes):
        episode = episode_batch(prior, args.namespace + episode_index, 1, query_size=prior.query_size)
        support_x = episode.support_x[0].numpy()
        support_y = episode.support_y[0].numpy().astype(np.int64)
        query_x = episode.query_x[0].numpy()
        query_y = episode.query_y[0].numpy().astype(np.int64)
        query_z = episode.query_z[0].numpy().astype(np.int64)

        finetuner = _build_finetuner(args)
        finetuner.fit(support_x, support_y)
        probability = finetuner.predict_proba(query_x)
        metrics = _episode_metrics(probability, query_y, query_z)
        row = {"episode": episode_index, "metrics": metrics}
        rows.append(row)
        print(json.dumps({"version": args.version, "episode": episode_index, "metrics": metrics}), flush=True)

    def mean(path: tuple[str, str]) -> float:
        return float(np.mean([row["metrics"][path[0]][path[1]] for row in rows]))

    return {
        "version": args.version,
        "prior_profile": args.prior_profile,
        "prior": asdict(prior),
        # ``checkpoint`` is a pathlib.Path for the local v2.6/v3 files;
        # normalize path-like CLI values before writing the JSON artifact.
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "output"
        },
        "episodes": rows,
        "summary": {
            regime: {metric: mean((regime, metric)) for metric in ("accuracy", "cross_entropy", "auc")}
            for regime in ("overall", "majority_z1", "minority_z0")
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v2.2", "v2.6", "v3"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-profile", choices=tuple(PRIOR_PROFILES), default="realistic")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--namespace", type=int, default=4_000_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-split-ratio", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--n-estimators-finetune", type=int, default=2)
    parser.add_argument("--n-estimators-validation", type=int, default=2)
    parser.add_argument("--n-estimators-final-inference", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    if args.checkpoint and not args.checkpoint.exists():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
