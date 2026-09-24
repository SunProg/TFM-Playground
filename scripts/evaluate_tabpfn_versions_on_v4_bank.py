"""Real TabPFN (v2.2 / v2.6 / v3) and TabICL (v1 / v1.1 / v2) in-context baselines on a multiregime-v4 bank.

Each bank episode is scored exactly like the pretrained NanoTabPFN runs are:
fit on the support set, predict the query set, per-episode cross entropy /
accuracy / OvR-AUC, then the same per-cell and per-factor summaries as
``evaluate_multiregime_v4_bank``. Output JSON has the identical layout so it
can be dropped into the existing extraction/plotting pipeline as a reference.

Model construction mirrors ``evaluate_multiregime_v3_baselines.build_tabpfn``
(same n_estimators / temperature / checkpoint handling). TabPFN's fit is only
preprocessing here, so per-episode fitting is cheap; use ``--limit`` for a
timing dry run and ``--start/--stop`` to shard the bank across jobs.

    python scripts/evaluate_tabpfn_versions_on_v4_bank.py \\
        --bank data/.../validation.h5 --versions v2.2 v2.6 v3 \\
        --output-dir runs_eval/tabpfn_baselines --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v4_evaluation import (
    FACTORS,
    MultiregimeV4EvaluationBank,
    _episode_auc,
    _nanmean,
    _summary,
)

DEFAULT_CHECKPOINTS = {
    "v2.2": None,
    "v2.6": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "v3": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
}
# TabICL checkpoints are addressed by name and auto-downloaded/cached by the package.
TABICL_CHECKPOINTS = {
    "tabicl-v1": "tabicl-classifier-v1-20250208.ckpt",
    "tabicl-v1.1": "tabicl-classifier-v1.1-20250506.ckpt",
    "tabicl-v2": "tabicl-classifier-v2-20260212.ckpt",
}
VERSIONS = ("v2.2", "v2.6", "v3", *TABICL_CHECKPOINTS)
EPS = 1e-6


def build_tabicl(version: str, *, device: str, n_estimators: int):
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        checkpoint_version=TABICL_CHECKPOINTS[version],
        n_estimators=n_estimators,
        softmax_temperature=0.9,
        device=device,
        random_state=0,
        verbose=False,
    )


def build_tabpfn(version: str, checkpoint: str | None, *, device: str, n_estimators: int):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model_version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[version]
    kwargs = dict(
        device=device,
        random_state=0,
        n_estimators=n_estimators,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        fit_mode="fit_preprocessors",
        show_progress_bar=False,
    )
    if checkpoint:
        return TabPFNClassifier(model_path=checkpoint, **kwargs)
    return TabPFNClassifier.create_default_for_version(model_version, **kwargs)


def prior_probabilities(query_count: int, num_classes: int, support_y: np.ndarray) -> np.ndarray:
    """Support class frequencies broadcast to every query row (the class-prior baseline)."""
    counts = np.bincount(support_y, minlength=num_classes).astype(np.float64)
    prior = np.clip(counts / counts.sum(), EPS, 1.0)
    return np.tile(prior / prior.sum(), (query_count, 1))


def full_probabilities(
    classifier, support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray, num_classes: int
) -> tuple[np.ndarray, bool]:
    """(query, num_classes) probabilities plus whether TabPFN was bypassed.

    Classes absent from the support get EPS mass. TabPFN cannot fit a
    single-class support or an all-constant feature table
    (``TabPFNValidationError``); such episodes fall back to the class-prior
    prediction, which is what the NanoTabPFN runs' fair metrics are measured
    against, so the fallback scores exactly 0 excess CE rather than crashing
    the whole bank.
    """
    if len(np.unique(support_y)) < 2 or bool(np.all(support_x == support_x[:1], axis=0).all()):
        # single-class support, or every feature constant (TabPFN raises, TabICL's
        # feature shuffle crashes with IndexError on the empty column set)
        return prior_probabilities(len(query_x), num_classes, support_y), True
    try:
        classifier.fit(support_x, support_y)
        pred = classifier.predict_proba(query_x)
    except Exception as error:  # noqa: BLE001 - TabPFNValidationError and friends
        if "ValidationError" not in type(error).__name__ and not isinstance(error, (ValueError, IndexError)):
            raise
        return prior_probabilities(len(query_x), num_classes, support_y), True
    probs = np.full((len(query_x), num_classes), EPS, dtype=np.float64)
    for col, cls in enumerate(classifier.classes_):
        probs[:, int(cls)] = pred[:, col]
    return probs / probs.sum(axis=1, keepdims=True), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--versions", nargs="+", choices=VERSIONS, default=list(VERSIONS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--checkpoint-v22", default=DEFAULT_CHECKPOINTS["v2.2"])
    parser.add_argument("--checkpoint-v26", default=DEFAULT_CHECKPOINTS["v2.6"])
    parser.add_argument("--checkpoint-v3", default=DEFAULT_CHECKPOINTS["v3"])
    parser.add_argument("--start", type=int, default=0, help="first episode index (for sharding)")
    parser.add_argument("--stop", type=int, default=None, help="one past the last episode index")
    parser.add_argument("--limit", type=int, default=None, help="score only this many episodes (timing dry run)")
    args = parser.parse_args()
    checkpoints = {"v2.2": args.checkpoint_v22, "v2.6": args.checkpoint_v26, "v3": args.checkpoint_v3}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with MultiregimeV4EvaluationBank(args.bank, device="cpu") as bank:
        stop = len(bank) if args.stop is None else min(args.stop, len(bank))
        indices = list(range(args.start, stop))
        if args.limit is not None:
            indices = indices[: args.limit]
        shard = f"__{args.start:06d}-{stop:06d}" if (args.start or args.stop is not None) else ""

        for version in args.versions:
            if version in TABICL_CHECKPOINTS:
                classifier = build_tabicl(version, device=args.device, n_estimators=args.n_estimators)
            else:
                classifier = build_tabpfn(version, checkpoints[version], device=args.device, n_estimators=args.n_estimators)
            per_episode: list[dict] = []
            t0 = time.perf_counter()
            for n, index in enumerate(indices, start=1):
                episode = bank.episode(index)
                meta = dict(episode["metadata"])
                sx = episode["support_x"][0].numpy().astype(np.float32)
                sy = episode["support_y"][0].numpy().reshape(-1).astype(np.int64)
                qx = episode["query_x"][0].numpy().astype(np.float32)
                qy = episode["query_y"][0].numpy().reshape(-1).astype(np.int64)
                probs, fallback = full_probabilities(classifier, sx, sy, qx, int(meta["num_classes"]))
                nll = -np.log(np.clip(probs[np.arange(len(qy)), qy], EPS, 1.0))
                meta["query_cross_entropy"] = float(nll.mean())
                meta["query_accuracy"] = float((probs.argmax(axis=1) == qy).mean())
                meta["query_auc"] = _episode_auc(torch.from_numpy(probs), torch.from_numpy(qy), int(meta["num_classes"]))
                meta["prior_fallback"] = fallback
                per_episode.append(meta)
                if n % 500 == 0 or n == len(indices):
                    rate = (time.perf_counter() - t0) / n
                    print(f"[{version}] {n}/{len(indices)} episodes, {rate*1000:.0f} ms/episode", flush=True)

            report = {
                "split": bank.split,
                "model": version if version in TABICL_CHECKPOINTS else f"tabpfn-{version}",
                "checkpoint": TABICL_CHECKPOINTS.get(version, checkpoints.get(version)),
                "n_estimators": args.n_estimators,
                "episodes": len(per_episode),
                "prior_fallback_episodes": int(sum(r["prior_fallback"] for r in per_episode)),
                "episode_range": [indices[0], indices[-1] + 1] if indices else [args.start, args.start],
                "overall": {
                    "query_cross_entropy": float(np.mean([r["query_cross_entropy"] for r in per_episode])),
                    "query_accuracy": float(np.mean([r["query_accuracy"] for r in per_episode])),
                    "query_auc": _nanmean(r["query_auc"] for r in per_episode),
                },
                "per_episode": per_episode,
                "by_cell": _summary(per_episode, FACTORS),
                "by_num_regimes": _summary(per_episode, ("num_regimes",)),
                "by_num_classes": _summary(per_episode, ("num_classes",)),
                "by_support_size": _summary(per_episode, ("support_size",)),
                "by_rule_mode": _summary(per_episode, ("rule_mode",)),
                "by_mechanism_mode": _summary(per_episode, ("mechanism_mode",)),
                "by_task_family": _summary(per_episode, ("task_family",)),
            }
            stem = version.replace("-", "_") if version in TABICL_CHECKPOINTS else f"tabpfn_{version}"
            out = out_dir / f"{stem}__{bank.split}{shard}.json"
            out.write_text(json.dumps(report, indent=1) + "\n")
            o = report["overall"]
            print(f"[{version}] {bank.split}: loss={o['query_cross_entropy']:.4f} acc={o['query_accuracy']:.4f} "
                  f"auc={o['query_auc']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
