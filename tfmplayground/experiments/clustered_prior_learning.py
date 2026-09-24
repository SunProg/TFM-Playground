"""Local proof-of-learning experiment for an intentionally easy two-regime prior.

Run: python -m tfmplayground.experiments.clustered_prior_learning --output results/clustered_prior_learning
Regime IDs and query labels are never passed to latent learners. All model
selection uses validation episodes; final tests use a separate seed namespace.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def sample_batch(seed, batch_size=1, support_size=128, query_size=256, separation=3.0):
    """Independent balanced-in-expectation episodes; fresh orthogonal rules per table."""
    rng = np.random.default_rng(seed)
    rows = support_size + query_size
    z = rng.integers(0, 2, (batch_size, rows))
    x = rng.normal(size=(batch_size, rows, 3)).astype(np.float32)
    x[:, :, 0] += separation * (2 * z - 1)
    angle = rng.uniform(0, 2 * np.pi, batch_size)
    handedness = rng.choice([-1, 1], batch_size)
    w0 = np.stack([np.cos(angle), np.sin(angle)], -1)
    w1 = handedness[:, None] * np.stack([-np.sin(angle), np.cos(angle)], -1)
    weights = np.stack([w0, w1], 1)
    scores = np.einsum("bnd,bkd->bnk", x[:, :, 1:], weights)
    y = (np.take_along_axis(scores, z[:, :, None], -1)[:, :, 0] > 0).astype(np.int64)
    return x, y, z


def summary(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": mean, "ci95": [mean - half, mean + half], "n_episodes": len(a)}


def fit_probability(x, y, q):
    if len(np.unique(y)) < 2:
        return np.full(len(q), (y.sum() + 1) / (len(y) + 2))
    model = LogisticRegression(C=10, fit_intercept=False, max_iter=200, solver="liblinear", random_state=0)
    return model.fit(x, y).predict_proba(q)[:, 1]


def routed_probability(x, y, q, support_route, query_route):
    result = np.zeros(len(q))
    for k in (0, 1):
        selected = support_route == k
        selected_q = query_route == k
        if selected_q.any():
            result[selected_q] = fit_probability(x[selected, 1:], y[selected], q[selected_q, 1:])
    return result


def classical_experiment(episodes):
    results = {}
    for n in (8, 16, 32, 64, 128):
        scores = {key: [] for key in ("pooled", "clustered", "oracle", "clustered_shuffled_labels", "routing_accuracy")}
        for index in range(episodes):
            # Generate one common 128-row support table, then take nested prefixes.
            x, y, z = [a[0] for a in sample_batch(3_000_000 + index)]
            sx, sy, qx, qy = x[:n], y[:n], x[128:], y[128:]
            cluster = KMeans(2, n_init=5, random_state=0).fit(sx)
            route = cluster.predict(qx)
            agreement = float((route == z[128:]).mean())
            scores["routing_accuracy"].append(max(agreement, 1 - agreement))
            shuffled = np.random.default_rng(4_000_000 + index).permutation(sy)
            predictions = {
                "pooled": fit_probability(sx, sy, qx),
                "clustered": routed_probability(sx, sy, qx, cluster.labels_, route),
                "oracle": routed_probability(sx, sy, qx, z[:n], z[128:]),
                "clustered_shuffled_labels": routed_probability(sx, shuffled, qx, cluster.labels_, route),
            }
            for key, p in predictions.items():
                scores[key].append(float(((p >= 0.5) == qy).mean()))
        results[str(n)] = {key: summary(value) for key, value in scores.items()}
        results[str(n)]["clustered_minus_oracle"] = summary(np.array(scores["clustered"]) - scores["oracle"])
        print(json.dumps({"classical_support": n, "metrics": results[str(n)]}), flush=True)
    return results


def model_inputs(x, y, z, support, oracle, shuffle=False):
    if oracle:
        x = np.concatenate([x, np.eye(2, dtype=np.float32)[z]], -1)
    sy = y[:, :support].copy()
    if shuffle:
        rng = np.random.default_rng(98765)
        for row in sy:
            rng.shuffle(row)
    return torch.from_numpy(x[:, :support]), torch.from_numpy(sy).float(), torch.from_numpy(x[:, support:])


@torch.no_grad()
def evaluate(model, oracle, episodes, namespace, shuffle=False, *, sampler=sample_batch):
    model.eval()
    accuracies, losses = [], []
    for start in range(0, episodes, 8):
        batch = min(8, episodes - start)
        tables = [sampler(namespace + i) for i in range(start, start + batch)]
        x, y, z = [np.concatenate([table[j] for table in tables]) for j in range(3)]
        logits = model(*model_inputs(x, y, z, 128, oracle, shuffle))
        target = torch.from_numpy(y[:, 128:])
        accuracies.extend((logits.argmax(-1) == target).float().mean(-1).tolist())
        losses.extend(F.cross_entropy(logits.transpose(1, 2), target, reduction="none").mean(-1).tolist())
    return {"accuracy": summary(accuracies), "cross_entropy": summary(losses)}


def neural_experiment(seed, oracle, steps, validation_episodes, test_episodes, output, *, sampler=sample_batch):
    torch.manual_seed(seed)
    model = NanoTabPFNModel(32, 4, 64, 2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    name = f"{'oracle' if oracle else 'latent'}-seed-{seed}"
    history = []
    best = float("inf")
    start = time.monotonic()
    checkpoint = output / f"{name}.pth"
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            metrics = evaluate(model, oracle, validation_episodes, 2_000_000, sampler=sampler)
            row = {"step": step, "seconds": time.monotonic() - start, **metrics}
            history.append(row)
            if metrics["cross_entropy"]["mean"] < best:
                best = metrics["cross_entropy"]["mean"]
                torch.save({"state_dict": model.state_dict(), "step": step}, checkpoint)
            print(json.dumps({"run": name, **row}), flush=True)
        if step == steps:
            break
        model.train()
        # Same episode stream for latent/oracle arms within each training seed.
        x, y, z = sampler(seed * 100_000 + step, batch_size=8, query_size=32)
        logits = model(*model_inputs(x, y, z, 128, oracle))
        loss = F.cross_entropy(logits.transpose(1, 2), torch.from_numpy(y[:, 128:]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    selected = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(selected["state_dict"])
    result = {
        "seed": seed,
        "oracle": oracle,
        "selected_step": selected["step"],
        "history": history,
        "test": evaluate(model, oracle, test_episodes, 3_000_000, sampler=sampler),
        "test_shuffled_support_labels": evaluate(
            model, oracle, test_episodes, 3_000_000, shuffle=True, sampler=sampler
        ),
    }
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"completed": name, "test": result["test"], "shuffled": result["test_shuffled_support_labels"]}),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--validation-episodes", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "output": str(args.output),
        "torch": torch.__version__,
        "device": "cpu",
        "architecture": {"embedding_size": 32, "heads": 4, "hidden_size": 64, "layers": 2},
        "support_size": 128,
        "train_query_size": 32,
        "test_query_size": 256,
        "batch_size": 8,
        "separation": 3.0,
        "label_noise": 0.0,
        "learning_rate": 0.001,
    }
    results = {"config": config, "classical": classical_experiment(args.test_episodes), "neural": []}
    path = args.output / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    for seed in args.seeds:
        for oracle in (False, True):
            results["neural"].append(
                neural_experiment(seed, oracle, args.steps, args.validation_episodes, args.test_episodes, args.output)
            )
            path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
