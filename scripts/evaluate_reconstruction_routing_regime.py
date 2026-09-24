"""Evaluate reconstruction-routing checkpoints by hidden SCM regime."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.reconstruction_routing_pilot import PlainControl, PreviousControl
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.reconstruction_routing import ReconstructionRouter


def roc_auc(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def build_model(cell, architecture):
    if cell["arm"] == "vanilla":
        return PlainControl(**{k: architecture[k] for k in ("width", "hidden", "layers", "heads")})
    if cell["arm"] == "values":
        return ReconstructionRouter(scope=cell["scope"], variant="values", **architecture)
    return PreviousControl(cell["scope"], cell["arm"], **architecture)


def checkpoint_rows(slot_root=None, vanilla_root=None, run_root=None, task_ids=None):
    if run_root is not None:
        roots = (("all", Path(run_root)),)
    else:
        roots = (("slot", Path(slot_root)), ("vanilla", Path(vanilla_root)))
    for kind, root in roots:
        for task_dir in sorted(root.glob("task-*"), key=lambda p: int(p.name.split("-")[-1])):
            task_id = int(task_dir.name.split("-")[-1])
            if task_ids is not None and task_id not in task_ids:
                continue
            config_path = task_dir / "config.json"
            checkpoint = task_dir / "best.pth"
            if not checkpoint.exists():
                checkpoint = task_dir / "latest.pth"
            if not config_path.exists() or not checkpoint.exists():
                continue
            cell = json.loads(config_path.read_text())["cell"]
            row_kind = "vanilla" if cell["arm"] == "vanilla" else "slot"
            yield {
                "kind": row_kind if kind == "all" else kind,
                "task": task_id,
                "directory": task_dir,
                "cell": cell,
                "checkpoint": checkpoint,
                "complete": (task_dir / "complete.json").exists(),
            }


def evaluate_model(model, episodes, device):
    values = {
        regime: {"nll": [], "accuracy": [], "y": [], "score": []}
        for regime in ("minor", "major")
    }
    model.eval()
    with torch.no_grad():
        for episode in episodes:
            prediction, _ = model(*episode.latent_inputs())
            logp = prediction.marginal_log_probabilities()[0]
            y = episode.query_y[0].long()
            z = episode.query_z[0].long()
            score = logp[:, 1].exp()
            nll = -logp.gather(1, y[:, None]).squeeze(1)
            correct = (logp.argmax(-1) == y).float()
            for regime_name, regime_id in (("minor", 0), ("major", 1)):
                mask = z == regime_id
                item = values[regime_name]
                item["nll"].extend(nll[mask].cpu().tolist())
                item["accuracy"].extend(correct[mask].cpu().tolist())
                item["y"].extend(y[mask].cpu().tolist())
                item["score"].extend(score[mask].cpu().tolist())
    result = {}
    for regime, item in values.items():
        result[regime] = {
            "n": len(item["y"]),
            "nll": float(np.mean(item["nll"])),
            "accuracy": float(np.mean(item["accuracy"])),
            "auc": roc_auc(item["y"], item["score"]),
        }
    return result


def main(args):
    device = torch.device(args.device)
    if args.prior_profile == "realistic":
        prior = SCMRegimeConfig(
            support_size=128,
            query_size=256,
            task_features=4,
            cue_separation=1.5,
            regime_probability=0.65,
            num_layers=2,
            hidden_dim=12,
            init_std=0.5,
            label_noise=0.05,
            calibration_size=1024,
            min_disagreement=0.20,
            max_disagreement=0.80,
            max_attempts=128,
        )
    else:
        prior = SCMRegimeConfig(
            support_size=128,
            query_size=32,
            task_features=2,
            cue_separation=1.5,
            regime_probability=0.65,
            label_noise=0.05,
            calibration_size=128,
        )
    test_prior = replace(prior, query_size=args.query_size)
    episodes = [
        sample_scm_regime_episode(test_prior, seed=args.episode_start + i).to(device)
        for i in range(args.episodes)
    ]
    architecture = dict(width=192, hidden=768, layers=6, heads=6, num_slots=4)
    results = {
        "test_set": {
            "episodes": args.episodes,
            "episode_seeds": [args.episode_start + i for i in range(args.episodes)],
            "episode_start": args.episode_start,
            "support_size": test_prior.support_size,
            "query_size": test_prior.query_size,
            "major_regime": "query_z == 1 (sampling probability 0.65)",
            "minor_regime": "query_z == 0 (sampling probability 0.35)",
            "regime_labels_are_diagnostics": True,
            "auc": "ROC-AUC for predicting query_y=1 within each hidden regime",
        },
        "models": [],
    }
    for row in checkpoint_rows(args.slot_root, args.vanilla_root, args.run_root, args.tasks):
        model = build_model(row["cell"], architecture).to(device)
        state = torch.load(row["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        metrics = evaluate_model(model, episodes, device)
        results["models"].append(
            {
                "kind": row["kind"],
                "task": row["task"],
                "cell": row["cell"],
                "checkpoint_step": state["step"],
                "checkpoint_file": row["checkpoint"].name,
                "complete": row["complete"],
                "metrics": metrics,
            }
        )
        del model, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--run-root")
    roots.add_argument("--slot-root")
    parser.add_argument("--vanilla-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prior-profile", choices=("scale", "realistic"), default="scale")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--episode-start", type=int, default=900000)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--tasks", type=int, nargs="+", default=None)
    main(parser.parse_args())
