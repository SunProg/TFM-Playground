"""Evaluate the official TabPFN v2 classifier on the locked SCM test stream."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig
from tfmplayground.experiments.scm_table_slot_head_sweep import PRIOR_PROFILES, episode_batch


def _make_classifier(
    version: str,
    *,
    device: str,
    random_state: int,
    checkpoint: str | None = None,
):
    """Build a pinned official TabPFN checkpoint with matched inference settings."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    versions = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}
    try:
        model_version = versions[version]
    except KeyError as error:
        raise ValueError(f"Unsupported TabPFN version {version!r}") from error
    kwargs = {
        "device": device,
        "random_state": random_state,
        "n_estimators": 8,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": False,
        "fit_mode": "fit_preprocessors",
        "show_progress_bar": False,
    }
    if checkpoint:
        # Explicit local checkpoints bypass the Prior Labs gated download.
        return TabPFNClassifier(model_path=checkpoint, **kwargs)
    return TabPFNClassifier.create_default_for_version(model_version, **kwargs)


def _summary(values: list[float]) -> dict[str, float | int | list[float]]:
    values_np = np.asarray(values, dtype=float)
    mean = float(values_np.mean())
    half = 1.96 * float(values_np.std(ddof=1)) / np.sqrt(len(values_np)) if len(values_np) > 1 else 0.0
    return {"mean": mean, "ci95": [mean - half, mean + half], "n_episodes": len(values_np)}


def _episode_metrics(probability: np.ndarray, target: np.ndarray, regime: np.ndarray) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    predicted = probability.argmax(axis=1)
    selected = np.clip(probability[np.arange(len(target)), target], 1e-8, 1.0)
    result = {
        "accuracy": float(np.mean(predicted == target)),
        "cross_entropy": float(-np.mean(np.log(selected))),
    }
    for name, value in (("overall", np.ones(len(target), dtype=bool)), ("majority_z1", regime == 1), ("minority_z0", regime == 0)):
        if value.sum() == 0:
            result[f"{name}_accuracy"] = float("nan")
            result[f"{name}_cross_entropy"] = float("nan")
            continue
        p = probability[value]
        y = target[value]
        result[f"{name}_accuracy"] = float(np.mean(p.argmax(axis=1) == y))
        result[f"{name}_cross_entropy"] = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.0))))
    return result


def evaluate(
    prior: SCMRegimeConfig,
    *,
    version: str,
    episodes: int,
    namespace: int,
    device: str,
    random_state: int,
    checkpoint: str | None,
    shuffle_support_labels: bool,
) -> dict:
    episode_accuracy: list[float] = []
    episode_ce: list[float] = []
    episode_majority_accuracy: list[float] = []
    episode_minority_accuracy: list[float] = []
    episode_majority_ce: list[float] = []
    episode_minority_ce: list[float] = []
    pooled_probabilities: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    pooled_regimes: list[np.ndarray] = []
    started = time.monotonic()

    # Match the existing evaluators exactly: each block uses episode_batch with
    # namespace + block start, which fixes the episode seeds independent of model.
    for start in range(0, episodes, 8):
        count = min(8, episodes - start)
        batch = episode_batch(prior, namespace + start, count, query_size=prior.query_size)
        shuffled_support = batch.support_y.detach().cpu().numpy().astype(np.int64).copy()
        if shuffle_support_labels:
            # Match `_prediction_log_probabilities` in the existing evaluator:
            # one generator per block, then shuffle each support-label row.
            generator = np.random.default_rng(namespace + start)
            for row in shuffled_support:
                generator.shuffle(row)
        for index in range(count):
            support_x = batch.support_x[index].detach().cpu().numpy()
            support_y = batch.support_y[index].detach().cpu().numpy().astype(np.int64)
            query_x = batch.query_x[index].detach().cpu().numpy()
            query_y = batch.query_y[index].detach().cpu().numpy().astype(np.int64)
            query_z = batch.query_z[index].detach().cpu().numpy().astype(np.int64)
            if shuffle_support_labels:
                support_y = shuffled_support[index]

            classifier = _make_classifier(
                version,
                device=device,
                random_state=random_state,
                checkpoint=checkpoint,
            )
            classifier.fit(support_x, support_y)
            probability = classifier.predict_proba(query_x)
            metrics = _episode_metrics(probability, query_y, query_z)
            episode_accuracy.append(metrics["accuracy"])
            episode_ce.append(metrics["cross_entropy"])
            episode_majority_accuracy.append(metrics["majority_z1_accuracy"])
            episode_minority_accuracy.append(metrics["minority_z0_accuracy"])
            episode_majority_ce.append(metrics["majority_z1_cross_entropy"])
            episode_minority_ce.append(metrics["minority_z0_cross_entropy"])
            pooled_probabilities.append(probability[:, 1])
            pooled_targets.append(query_y)
            pooled_regimes.append(query_z)
        print(json.dumps({"episodes_done": min(start + count, episodes), "episodes": episodes, "seconds": time.monotonic() - started}), flush=True)

    probabilities = np.concatenate(pooled_probabilities)
    targets = np.concatenate(pooled_targets)
    regimes = np.concatenate(pooled_regimes)

    def auc(mask: np.ndarray) -> float:
        return float(roc_auc_score(targets[mask], probabilities[mask]))

    return {
        "shuffle_support_labels": shuffle_support_labels,
        "episodes": episodes,
        "episode_accuracy": _summary(episode_accuracy),
        "episode_cross_entropy": _summary(episode_ce),
        "majority_z1_accuracy": _summary(episode_majority_accuracy),
        "minority_z0_accuracy": _summary(episode_minority_accuracy),
        "majority_z1_cross_entropy": _summary(episode_majority_ce),
        "minority_z0_cross_entropy": _summary(episode_minority_ce),
        "pooled_auc": auc(np.ones(len(targets), dtype=bool)),
        "majority_z1_auc": auc(regimes == 1),
        "minority_z0_auc": auc(regimes == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-profile", choices=tuple(PRIOR_PROFILES), default="realistic")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--namespace", type=int, default=3_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--versions", nargs="+", choices=("v2.2", "v2.6", "v3"), default=("v2.6", "v3"))
    parser.add_argument("--v26-checkpoint", type=Path)
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument(
        "--skip-shuffled-support",
        action="store_true",
        help="Evaluate only normally labelled supports; useful for a paired fixed-weight reference.",
    )
    args = parser.parse_args()

    prior = PRIOR_PROFILES[args.prior_profile]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import importlib.metadata

    result = {
        "model": "TabPFNClassifier",
        "tabpfn_package_version": importlib.metadata.version("tabpfn"),
        "config": {
            "prior_profile": args.prior_profile,
            "prior": asdict(prior),
            "episodes": args.episodes,
            "namespace": args.namespace,
            "device": args.device,
            "random_state": args.random_state,
            "versions": list(args.versions),
            "checkpoints": {
                "v2.6": str(args.v26_checkpoint) if args.v26_checkpoint else None,
                "v3": str(args.v3_checkpoint) if args.v3_checkpoint else None,
            },
            "n_estimators": 8,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "fit_mode": "fit_preprocessors",
            "skip_shuffled_support": args.skip_shuffled_support,
        },
        "models": {},
    }
    for version in args.versions:
        checkpoint_path = (
            str(args.v26_checkpoint)
            if version == "v2.6" and args.v26_checkpoint
            else str(args.v3_checkpoint)
            if version == "v3" and args.v3_checkpoint
            else None
        )
        model_result = {
            "normal": evaluate(
                prior,
                version=version,
                episodes=args.episodes,
                namespace=args.namespace,
                device=args.device,
                random_state=args.random_state,
                checkpoint=checkpoint_path,
                shuffle_support_labels=False,
            )
        }
        if not args.skip_shuffled_support:
            model_result["shuffled_support"] = evaluate(
                prior,
                version=version,
                episodes=args.episodes,
                namespace=args.namespace,
                device=args.device,
                random_state=args.random_state,
                checkpoint=checkpoint_path,
                shuffle_support_labels=True,
            )
        result["models"][version] = model_result
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
