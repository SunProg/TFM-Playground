"""Probe which encoder design can represent epistemic uncertainty at all.

Two pilots showed the learned dispersion gate going flat across conditions whose
true epistemic content differs by 0.16 nats, at three very different capacity
budgets.  That points at the *inputs* of the uncertainty head rather than at its
capacity or its loss weighting.

This experiment isolates the representation question.  Small supervised heads are
trained on frozen uncertainty representations to regress the projected-teacher
mutual information directly: no posterior sampling, no bounded dispersion gate,
and no energy distance in the way.  If a design cannot fit the teacher here, it
cannot learn it inside the full objective either.

Candidate contexts:

``deepsets``
    The current global mean/variance pool of support target embeddings.
``cross_attention``
    Each query attends over the support target embeddings, giving a per-query
    context.  Support embeddings are already label-conditioned, so this can
    express "the labelled evidence relevant to *this* query is decisive".
``cross_attention_local``
    Adds explicit locality features (kNN label agreement, neighbour distances,
    and the vanilla mean).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfmplayground.experiments.continuous_episodes import (
    HELDOUT_REGIME,
    TRAIN_REGIME,
    ContinuousEpisode,
    EpisodeRegime,
    sample_episode,
)
from tfmplayground.experiments.train_continuous_bayesian import teacher_targets
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.continuous_posterior import (
    DeepSetsSupportEncoder,
    NanoTabPFNContinuousPosteriorModel,
)
from tfmplayground.utils import set_randomness_seed

CONTEXT_MODES = ("deepsets", "cross_attention", "cross_attention_local")
#: Neighbour counts used by the locality features.
LOCALITY_NEIGHBOURS = (4, 16, 64)
#: A design must clear these to justify changing the model.
MIN_R2 = 0.30
MIN_CONDITION_AUC = 0.85


@dataclass(frozen=True)
class EncoderProbeConfig:
    seed: int = 5505
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    device: str = "cpu"
    train_episodes: int = 240
    validation_episodes: int = 60
    evaluation_episodes: int = 60
    support_size: int = 128
    query_count: int = 6
    steps: int = 1500
    validation_interval: int = 50
    patience: int = 6
    learning_rate: float = 1e-3
    hidden_size: int = 128
    context_size: int = 96
    num_attention_heads: int = 4


@dataclass
class ProbeBatch:
    """Frozen representations and teacher targets, stacked over episodes.

    Every episode in a probe run uses the same support size and query count, so
    the whole set fits in one batched tensor and each training step is a single
    forward pass rather than a Python loop over episodes.
    """

    support_embeddings: torch.Tensor
    query_embeddings: torch.Tensor
    locality: torch.Tensor
    mutual_information: torch.Tensor
    epistemic_variance: torch.Tensor
    conditions: list[str]

    @property
    def episode_count(self) -> int:
        return self.support_embeddings.shape[0]

    def condition_mask(self, *names: str) -> torch.Tensor:
        return torch.tensor([condition in names for condition in self.conditions])


def locality_features(episode: ContinuousEpisode, base_positive: torch.Tensor) -> torch.Tensor:
    """Per-query kNN label agreement, neighbour distances, and the vanilla mean.

    The support labels are the only labels used, so this stays inside the
    protocol: query labels never enter any feature.
    """
    support_x, query_x = episode.support_x[0], episode.query_x[0]
    support_y = episode.support_y[0]
    distances = torch.cdist(query_x, support_x)
    columns = [base_positive[0][:, None]]
    for neighbours in LOCALITY_NEIGHBOURS:
        count = min(neighbours, support_x.shape[0])
        nearest = distances.topk(count, largest=False)
        labels = support_y[nearest.indices]
        # Agreement is one when every neighbour shares a label and zero when the
        # neighbourhood is evenly split.
        columns.append(((labels.mean(dim=-1) - 0.5).abs() * 2.0)[:, None])
        columns.append(labels.var(dim=-1, unbiased=False)[:, None])
        columns.append(nearest.values.mean(dim=-1)[:, None])
    return torch.cat(columns, dim=-1)


@torch.no_grad()
def collect_episodes(
    model: NanoTabPFNContinuousPosteriorModel,
    config: EncoderProbeConfig,
    *,
    regime: EpisodeRegime,
    count: int,
    seed: int,
) -> ProbeBatch:
    """Draw episodes and cache their frozen representations and teacher targets."""
    rng = np.random.default_rng(seed)
    conditions = ("ambiguous", "identifiable", "noisy")
    supports, queries, localities, informations, variances, labels = [], [], [], [], [], []
    for index in range(count):
        condition = conditions[index % len(conditions)]
        episode = sample_episode(
            rng,
            regime=regime,
            condition=condition,
            batch_size=1,
            support_size=config.support_size,
            query_count=config.query_count,
            device=config.device,
        )
        base = model._frozen_mean(
            episode.support_x, episode.support_y, episode.query_x, num_mem_chunks=1
        )[..., 1]
        support, query = model._uncertainty_representations(
            episode.support_x, episode.support_y, episode.query_x, num_mem_chunks=1
        )
        targets = teacher_targets(episode, base)
        supports.append(support[0])
        queries.append(query[0])
        localities.append(locality_features(episode, base))
        informations.append(targets.mutual_information[0])
        variances.append(targets.epistemic_variance[0])
        labels.append(condition)
    return ProbeBatch(
        torch.stack(supports),
        torch.stack(queries),
        torch.stack(localities),
        torch.stack(informations),
        torch.stack(variances),
        labels,
    )


class ProbeHead(nn.Module):
    """One candidate context design plus a small regression head."""

    def __init__(self, embedding_size: int, locality_size: int, config: EncoderProbeConfig, mode: str):
        super().__init__()
        if mode not in CONTEXT_MODES:
            raise ValueError(f"mode must be one of {CONTEXT_MODES}.")
        self.mode = mode
        self.context_size = config.context_size
        if mode == "deepsets":
            self.context = DeepSetsSupportEncoder(embedding_size, config.hidden_size, config.context_size)
            feature_size = embedding_size + config.context_size
        else:
            self.query_projection = nn.Linear(embedding_size, embedding_size)
            self.attention = nn.MultiheadAttention(
                embedding_size, config.num_attention_heads, batch_first=True
            )
            self.context_projection = nn.Linear(embedding_size, config.context_size)
            self.norm = nn.LayerNorm(config.context_size)
            feature_size = embedding_size + config.context_size
        if mode == "cross_attention_local":
            feature_size += locality_size
        self.head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
        )

    def features(self, batch: ProbeBatch) -> torch.Tensor:
        support = batch.support_embeddings
        query = batch.query_embeddings
        if self.mode == "deepsets":
            # DeepSets pooling returns one global context per episode.
            context = self.context(support)[:, None].expand(-1, query.shape[1], -1)
        else:
            attended = self.attention(
                self.query_projection(query), support, support, need_weights=False
            )[0]
            context = self.norm(self.context_projection(attended))
        parts = [query, context]
        if self.mode == "cross_attention_local":
            parts.append(batch.locality)
        return torch.cat(parts, dim=-1)

    def forward(self, batch: ProbeBatch) -> torch.Tensor:
        """Return ``(episode, query, 2)`` mutual information and epistemic variance."""
        # Both targets are non-negative; softplus keeps the head honest without
        # bounding it away from the teacher's scale.
        return nn.functional.softplus(self.head(self.features(batch)))


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    variance = float(((target - target.mean()) ** 2).sum())
    if variance <= 0:
        return float("nan")
    return float(1.0 - ((prediction - target) ** 2).sum() / variance)


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.allclose(prediction, prediction[0]) or np.allclose(target, target[0]):
        return float("nan")
    return float(spearmanr(prediction, target).statistic)


def _condition_auc(prediction: np.ndarray, labels: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, prediction))


def _objective(prediction: torch.Tensor, batch: ProbeBatch) -> torch.Tensor:
    return nn.functional.mse_loss(prediction[..., 0], batch.mutual_information) + nn.functional.mse_loss(
        prediction[..., 1], batch.epistemic_variance
    )


@torch.no_grad()
def score(head: ProbeHead, batch: ProbeBatch) -> dict[str, Any]:
    """R², Spearman, and ambiguous-versus-identifiable AUC on one split."""
    head.eval()
    prediction = head(batch)
    information = prediction[..., 0].reshape(-1).numpy()
    mask = batch.condition_mask("ambiguous", "identifiable")
    is_ambiguous = np.asarray([condition == "ambiguous" for condition in batch.conditions], dtype=int)[
        mask.numpy()
    ]
    condition_labels = np.repeat(is_ambiguous, batch.query_embeddings.shape[1])
    return {
        "loss": float(_objective(prediction, batch)),
        "mutual_information_r2": _r2(information, batch.mutual_information.reshape(-1).numpy()),
        "epistemic_variance_r2": _r2(
            prediction[..., 1].reshape(-1).numpy(), batch.epistemic_variance.reshape(-1).numpy()
        ),
        "mutual_information_spearman": _spearman(
            information, batch.mutual_information.reshape(-1).numpy()
        ),
        "condition_auc": _condition_auc(prediction[..., 0][mask].reshape(-1).numpy(), condition_labels),
    }


def train_probe(
    mode: str,
    train_batch: ProbeBatch,
    validation_batch: ProbeBatch,
    evaluation_batch: ProbeBatch,
    config: EncoderProbeConfig,
) -> dict[str, Any]:
    """Fit one candidate design, early stopping on *in-family* validation.

    Reporting both the in-family and the held-out-family split separates two
    very different failures: a design that fits its own families but not new
    ones is a data-shift problem, while a design that fails in-family cannot
    represent the target at all.
    """
    import copy

    torch.manual_seed(config.seed)
    head = ProbeHead(
        train_batch.query_embeddings.shape[-1], train_batch.locality.shape[-1], config, mode
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    best_state = copy.deepcopy(head.state_dict())
    best_validation = math.inf
    best_step = 0
    stale = 0
    train_loss = math.nan
    for step in range(1, config.steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(head(train_batch), train_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        train_loss = float(loss.detach())
        if step % config.validation_interval == 0:
            validation = score(head, validation_batch)["loss"]
            if validation < best_validation:
                best_validation, best_step, stale = validation, step, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
    head.load_state_dict(best_state)
    return {
        "mode": mode,
        "parameters": int(sum(p.numel() for p in head.parameters())),
        "train_loss": train_loss,
        "best_step": best_step,
        "in_family": score(head, validation_batch),
        "held_out_family": score(head, evaluation_batch),
    }


def run_probe(config: EncoderProbeConfig, output_dir: str) -> Path:
    set_randomness_seed(config.seed)
    backbone = init_model_from_state_dict_file(str(Path(config.checkpoint).expanduser().resolve()))
    model = NanoTabPFNContinuousPosteriorModel(backbone, uncertainty_mode="frozen").to(config.device).eval()
    print("collecting training episodes (training families)", flush=True)
    train_batch = collect_episodes(
        model, config, regime=TRAIN_REGIME, count=config.train_episodes, seed=config.seed
    )
    print("collecting in-family validation episodes (training families, fresh seeds)", flush=True)
    validation_batch = collect_episodes(
        model, config, regime=TRAIN_REGIME, count=config.validation_episodes, seed=config.seed + 313
    )
    print("collecting held-out-family episodes", flush=True)
    evaluation_batch = collect_episodes(
        model, config, regime=HELDOUT_REGIME, count=config.evaluation_episodes, seed=config.seed + 991
    )
    results = [
        train_probe(mode, train_batch, validation_batch, evaluation_batch, config) for mode in CONTEXT_MODES
    ]
    for row in results:
        print(json.dumps(row, sort_keys=True), flush=True)

    baseline = next(row for row in results if row["mode"] == "deepsets")
    ranked = sorted(results, key=lambda row: -_finite(row["held_out_family"]["mutual_information_r2"]))
    best = ranked[0]
    best_r2 = _finite(best["held_out_family"]["mutual_information_r2"])
    best_auc = best["held_out_family"]["condition_auc"] or 0.0
    decision = {
        "best_mode": best["mode"],
        "best_in_family_r2": _finite(best["in_family"]["mutual_information_r2"]),
        "best_held_out_family_r2": best_r2,
        "beats_deepsets": bool(
            best_r2 > _finite(baseline["held_out_family"]["mutual_information_r2"])
        ),
        "clears_r2_floor": bool(best_r2 >= MIN_R2),
        "clears_condition_auc_floor": bool(best_auc >= MIN_CONDITION_AUC),
        # An in-family fit that does not transfer is a data-shift result, not an
        # encoder result, and must not be read as either.
        "generalizes_in_family": bool(_finite(best["in_family"]["mutual_information_r2"]) >= MIN_R2),
    }
    decision["proceed"] = bool(
        decision["beats_deepsets"] and decision["clears_r2_floor"] and decision["clears_condition_auc_floor"]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "encoder_probe.json").write_text(
        json.dumps({"config": asdict(config), "results": results, "decision": decision}, indent=2) + "\n"
    )
    (output / "encoder_probe.csv").write_text(_csv(results))
    print(json.dumps(decision, indent=2), flush=True)
    return output.resolve()


def _finite(value: float | None) -> float:
    return -math.inf if value is None or not math.isfinite(value) else float(value)


def _csv(results: list[dict[str, Any]]) -> str:
    flat = [
        {
            "mode": row["mode"],
            "parameters": row["parameters"],
            "best_step": row["best_step"],
            "train_loss": row["train_loss"],
            **{f"in_family.{key}": value for key, value in row["in_family"].items()},
            **{f"held_out_family.{key}": value for key, value in row["held_out_family"].items()},
        }
        for row in results
    ]
    keys = list(flat[0])
    body = "\n".join(",".join(str(row[key]) for key in keys) for row in flat)
    return ",".join(keys) + "\n" + body + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = EncoderProbeConfig()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--device", default=defaults.device)
    for name in (
        "seed",
        "train_episodes",
        "validation_episodes",
        "evaluation_episodes",
        "support_size",
        "query_count",
        "steps",
        "validation_interval",
        "patience",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    return parser


if __name__ == "__main__":
    arguments = vars(build_parser().parse_args())
    destination = arguments.pop("output_dir")
    print(f"Wrote encoder probe to {run_probe(EncoderProbeConfig(**arguments), destination)}")
