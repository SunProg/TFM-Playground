"""Measure classical and neural learning on the constrained TabICL SCM regime prior."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tfmplayground.experiments.clustered_prior_learning import neural_experiment, summary
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode


class SCMBatchSampler:
    def __init__(self, config):
        self.config = config

    def episode(self, seed):
        child_seed = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
        return sample_scm_regime_episode(self.config, seed=child_seed)

    def __call__(self, seed, batch_size=1, support_size=128, query_size=256):
        config = replace(self.config, support_size=support_size, query_size=query_size)
        seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, batch_size)
        episodes = [sample_scm_regime_episode(config, seed=int(s)) for s in seeds]
        x = np.concatenate([torch.cat([e.support_x, e.query_x], 1).numpy() for e in episodes])
        y = np.concatenate([torch.cat([e.support_y.long(), e.query_y], 1).numpy() for e in episodes])
        z = np.concatenate([torch.cat([e.support_z, e.query_z], 1).numpy() for e in episodes])
        return x, y, z


def fit_predict(x, y, q):
    if len(np.unique(y)) < 2:
        return np.full(len(q), int(y.mean() >= 0.5) if len(y) else 0)
    # Fixed hyperparameters. Scaling is fit solely on the selected support rows.
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(x, y).predict(q)


def routed_predict(sx, sy, qx, sz, qz):
    predicted = np.zeros(len(qx), dtype=np.int64)
    for k in (0, 1):
        selected = qz == k
        if selected.any():
            predicted[selected] = fit_predict(sx[sz == k, 1:], sy[sz == k], qx[selected, 1:])
    return predicted


def classical_experiment(sampler, count):
    records = []
    for i in range(count):
        e = sampler.episode(3_000_000 + i)
        sx, sy, qx = [t[0].numpy() for t in e.latent_inputs()]
        qy, sz, qz = [t[0].numpy() for t in (e.query_y, e.support_z, e.query_z)]
        clustered = KMeans(2, n_init=5, random_state=0).fit(sx)
        sr, qr = clustered.labels_, clustered.predict(qx)
        shuffled = np.random.default_rng(4_000_000 + i).permutation(sy)
        predictions = {
            "clustered_rbf": routed_predict(sx, sy, qx, sr, qr),
            "oracle_rbf": routed_predict(sx, sy, qx, sz, qz),
            "pooled_rbf": fit_predict(sx, sy, qx),
            "shuffled_clustered_rbf": routed_predict(sx, shuffled, qx, sr, qr),
        }
        # Bayes comparator knows the actual generating mechanisms, but not sampled query z.
        cf = e.counterfactual_query_probabilities[0].numpy()
        gate = e.query_gate_probabilities[0].numpy()
        predictions["known_mechanisms_bayes"] = (np.sum(gate * cf.T, axis=1) >= 0.5).astype(int)
        agreement = float((qr == qz).mean())
        records.append(
            {
                "seed": 3_000_000 + i,
                **{key: float((p == qy).mean()) for key, p in predictions.items()},
                "routing_accuracy": max(agreement, 1 - agreement),
                "calibration_disagreement": e.metadata["calibration_disagreement"],
                "query_rule_disagreement": e.metadata["query_rule_disagreement"],
                "pair_attempts": e.metadata["pair_attempts"],
            }
        )
    metrics = {key: summary([r[key] for r in records]) for key in records[0] if key != "seed"}
    metrics["clustered_minus_oracle"] = summary([r["clustered_rbf"] - r["oracle_rbf"] for r in records])
    return {"metrics": metrics, "episodes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--cue-separation", type=float, default=3.0)
    parser.add_argument("--task-features", type=int, default=2)
    parser.add_argument("--scm-layers", type=int, default=2)
    parser.add_argument("--scm-width", type=int, default=8)
    parser.add_argument("--classical-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.test_episodes < 1 or args.validation_episodes < 1:
        parser.error("Step and episode counts must be positive.")
    if any(seed not in (11, 12, 13) for seed in args.seeds):
        parser.error("This experiment reserves training seeds 11, 12 and 13 to keep namespaces disjoint.")
    if args.steps >= 100_000:
        parser.error("Use fewer than 100,000 steps to preserve disjoint training seed namespaces.")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("Output already contains results; choose a fresh directory.")
    config = SCMRegimeConfig(
        cue_separation=args.cue_separation,
        task_features=args.task_features,
        num_layers=args.scm_layers,
        hidden_dim=args.scm_width,
    )
    sampler = SCMBatchSampler(config)
    results = {
        "config": {
            **vars(args),
            "output": str(args.output),
            "prior": asdict(config),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "classical": classical_experiment(sampler, args.test_episodes),
        "neural": [],
    }
    path = args.output / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"classical": results["classical"]["metrics"]}), flush=True)
    if not args.classical_only:
        for seed in args.seeds:
            for oracle in (False, True):
                run = neural_experiment(
                    seed, oracle, args.steps, args.validation_episodes, args.test_episodes, args.output, sampler=sampler
                )
                results["neural"].append(run)
                path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
