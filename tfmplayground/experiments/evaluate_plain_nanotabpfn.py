"""Locked ordinary and multiregime evaluation for plain nanoTabPFN checkpoints.

Episodes are generated once, before any model is loaded.  Consequently every
named checkpoint receives exactly the same feature rows, labels, regime tags,
and contamination levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.continuous_episodes import (
    TRAIN_REGIME,
    sample_scm_multiregime_episode,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import PlainPretrainingConfig, make_prior
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import SlotLogitsAdapter, load_checkpoint_for_inference
from tfmplayground.utils import set_randomness_seed

SCM_SOURCES: tuple[str | tuple[str, str], ...] = (
    "mlp_scm",
    "tree_scm",
    ("mlp_scm", "tree_scm"),
)


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 44_040
    device: str = "cuda"
    support_size: int = 128
    query_size: int = 32
    min_features: int = 2
    max_features: int = 12
    ordinary_episodes: int = 100
    multiregime_episodes: int = 100
    contaminations: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)


@dataclass(frozen=True)
class EvaluationEpisode:
    episode_id: str
    source: str
    contamination: float | None
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    query_regime_source: torch.Tensor | None

    def manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for value in (self.support_x, self.support_y, self.query_x, self.query_y, self.query_regime_source):
            if value is not None:
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return {
            "episode_id": self.episode_id,
            "source": self.source,
            "contamination": self.contamination,
            "support_shape": list(self.support_x.shape),
            "query_shape": list(self.query_x.shape),
            "sha256": digest.hexdigest(),
        }


def _ordinary_episodes(config: EvaluationConfig) -> list[EvaluationEpisode]:
    prior_config = PlainPretrainingConfig(
        device="cpu",
        micro_batch_size=1,
        support_size=config.support_size,
        query_size=config.query_size,
        min_features=config.min_features,
        max_features=config.max_features,
        max_classes=2,
    )
    prior = make_prior(prior_config, batches=config.ordinary_episodes, device="cpu")
    episodes = []
    for index, batch in enumerate(prior):
        split = int(batch["train_test_split_index"])
        # Evaluation batches contain one independent table to make episode
        # identity and per-episode confidence intervals unambiguous.
        if batch["x"].shape[0] != 1:
            raise RuntimeError("Ordinary evaluation requires a unit batch size.")
        episodes.append(
            EvaluationEpisode(
                f"ordinary-{index:04d}",
                "ordinary_mix_scm",
                None,
                batch["x"][0, :split].cpu(),
                batch["y"][0, :split].cpu(),
                batch["x"][0, split:].cpu(),
                batch["y"][0, split:].long().cpu(),
                None,
            )
        )
    return episodes


def generate_evaluation_episodes(config: EvaluationConfig) -> list[EvaluationEpisode]:
    """Generate the complete locked evaluation set independently of model names."""
    set_randomness_seed(config.seed)
    episodes = _ordinary_episodes(config)
    rng = np.random.default_rng(config.seed + 1)
    for source in SCM_SOURCES:
        source_name = source if isinstance(source, str) else "+".join(source)
        for contamination in config.contaminations:
            for index in range(config.multiregime_episodes):
                episode = sample_scm_multiregime_episode(
                    rng,
                    regime=TRAIN_REGIME,
                    family=source,
                    batch_size=1,
                    support_size=config.support_size,
                    query_count=config.query_size,
                    noise=0.0,
                    contamination=contamination,
                    device="cpu",
                )
                episodes.append(
                    EvaluationEpisode(
                        f"{source_name}-c{contamination:.1f}-{index:04d}",
                        source_name,
                        contamination,
                        episode.support_x[0].cpu(),
                        episode.support_y[0].cpu(),
                        episode.query_x[0].cpu(),
                        episode.query_y[0].long().cpu(),
                        episode.query_regime_source[0].long().cpu(),
                    )
                )
    return episodes


def load_model(checkpoint: str | Path, device: str) -> NanoTabPFNModel | SlotLogitsAdapter:
    """Load a plain or a slot checkpoint behind one calling convention.

    A slot model is returned wrapped in :class:`SlotLogitsAdapter`, which emits
    log mixture probabilities.  Those are valid logits, so ``predict`` below and
    every other scorer in this module work on both without a second code path.
    """
    return load_checkpoint_for_inference(checkpoint, device)


@torch.no_grad()
def predict(model: NanoTabPFNModel, episode: EvaluationEpisode, device: str) -> np.ndarray:
    support_x = episode.support_x.unsqueeze(0).to(device)
    support_y = episode.support_y.unsqueeze(0).to(device)
    query_x = episode.query_x.unsqueeze(0).to(device)
    logits = model(support_x, support_y, query_x)[0, :, :2]
    return logits.softmax(dim=-1)[:, 1].cpu().numpy()


def _metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    if len(target) == 0:
        return {"query_cross_entropy": None, "accuracy": None, "auroc": None, "brier": None, "n_rows": 0}
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    target = target.astype(np.int64)
    auroc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else None
    return {
        "query_cross_entropy": float(-np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability))),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
        "auroc": auroc,
        "brier": float(np.mean((probability - target) ** 2)),
        "n_rows": int(len(target)),
    }


def _groups(episode: EvaluationEpisode) -> dict[str, np.ndarray]:
    size = len(episode.query_y)
    if episode.query_regime_source is None:
        return {"overall": np.ones(size, dtype=bool), "base": np.ones(size, dtype=bool)}
    source = episode.query_regime_source.numpy()
    groups = {"overall": np.ones(size, dtype=bool), "base": source == 0}
    if (source == 1).any():
        groups["other"] = source == 1
    return groups


def evaluate_models(
    checkpoints: dict[str, str | Path],
    config: EvaluationConfig,
    output_dir: str | Path,
) -> Path:
    """Evaluate all named checkpoints on one immutable, persisted episode set."""
    if not checkpoints:
        raise ValueError("At least one NAME=CHECKPOINT must be supplied.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    episodes = generate_evaluation_episodes(config)
    with (output / "episode_manifest.jsonl").open("w") as manifest:
        for episode in episodes:
            manifest.write(json.dumps(episode.manifest(), sort_keys=True) + "\n")

    per_episode: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for model_name, checkpoint in checkpoints.items():
        model = load_model(checkpoint, config.device)
        for episode in episodes:
            probability = predict(model, episode, config.device)
            target = episode.query_y.numpy()
            regime = None if episode.query_regime_source is None else episode.query_regime_source.numpy()
            for row_index, (p, y) in enumerate(zip(probability, target, strict=True)):
                predictions.append(
                    {
                        "model": model_name,
                        "episode_id": episode.episode_id,
                        "source": episode.source,
                        "contamination": episode.contamination,
                        "query_index": row_index,
                        "probability": float(p),
                        "target": int(y),
                        "regime_source": None if regime is None else int(regime[row_index]),
                    }
                )
            for group, mask in _groups(episode).items():
                per_episode.append(
                    {
                        "model": model_name,
                        "episode_id": episode.episode_id,
                        "source": episode.source,
                        "contamination": episode.contamination,
                        "group": group,
                        **_metrics(probability[mask], target[mask]),
                    }
                )
        del model

    for name, rows in (("per_episode_metrics.jsonl", per_episode), ("predictions.jsonl", predictions)):
        with (output / name).open("w") as destination:
            for row in rows:
                destination.write(json.dumps(row, sort_keys=True) + "\n")

    summary: list[dict[str, Any]] = []
    for model_name in checkpoints:
        groups = {
            (row["source"], row["contamination"], row["group"])
            for row in per_episode
            if row["model"] == model_name
        }
        for source, contamination, group in sorted(groups, key=lambda item: (item[0], str(item[1]), item[2])):
            selected = [
                row
                for row in predictions
                if row["model"] == model_name
                and row["source"] == source
                and row["contamination"] == contamination
                and (
                    group == "overall"
                    or (group == "base" and row["regime_source"] in (None, 0))
                    or (group == "other" and row["regime_source"] == 1)
                )
            ]
            summary.append(
                {
                    "model": model_name,
                    "source": source,
                    "contamination": contamination,
                    "group": group,
                    "episodes": sum(
                        1
                        for row in per_episode
                        if row["model"] == model_name
                        and row["source"] == source
                        and row["contamination"] == contamination
                        and row["group"] == group
                    ),
                    **_metrics(
                        np.asarray([row["probability"] for row in selected]),
                        np.asarray([row["target"] for row in selected]),
                    ),
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = EvaluationConfig()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=defaults.device)
    for name in (
        "seed",
        "support_size",
        "query_size",
        "min_features",
        "max_features",
        "ordinary_episodes",
        "multiregime_episodes",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    return parser


if __name__ == "__main__":
    args = vars(build_parser().parse_args())
    destination = args.pop("output_dir")
    entries = args.pop("checkpoint")
    checkpoints = dict(item.split("=", 1) for item in entries)
    print(evaluate_models(checkpoints, EvaluationConfig(**args), destination))
