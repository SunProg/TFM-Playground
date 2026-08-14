"""Fine-tune the nanoTabPFN backbone itself on a multiregime curriculum.

Every prior attempt at this class of problem trained something *on top of* a
frozen backbone: a learned slot posterior (`MEAN_PRESERVING_BAYESIAN_TRIAL.md`),
a continuous dispersion gate (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`), and
a training-free resampling estimator (`SUPPORT_RESAMPLING_VARIANCE_TRIAL.md`,
`paper/tabpfn_multi_regime_results.md`), plus this branch's own first
multiregime pilot (`train_continuous_bayesian.py`'s `"multiregime"` condition,
which -- like every other condition there -- trains only the uncertainty
adapters while the mean path stays frozen, per that pipeline's core
invariant). All of them found the same thing: nothing bolted onto a
mean-only-trained backbone can extract a usable multi-regime signal.

This script trains the backbone's own predictive mean, not a head next to it.
`NanoTabPFNModel` is loaded from the pretrained checkpoint and *every*
parameter -- feature encoder, target encoder, all six transformer blocks, and
the decoder -- is fine-tuned with a plain query cross-entropy loss, on a
curriculum that mixes ordinary single-regime episodes with
`sample_multiregime_episode` draws. If the backbone's own predictions and
predictive entropy still show no signal for which queries were answered under
a contaminated context after this, the limitation is not "needs a fancier
head" -- every architecture in this repo's history has now been tried, from
persistent slots to particle filters to resampling to backbone fine-tuning.
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
import torch.nn.functional as F

from tfmplayground.experiments.continuous_episodes import (
    CONDITIONS,
    HELDOUT_REGIME,
    TRAIN_REGIME,
    sample_episode,
    sample_multiregime_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.utils import set_randomness_seed

#: Conditions drawn for the ordinary (non-multiregime) share of the curriculum.
#: "paired" is excluded here -- it needs its own two-forward-pass loss and adds
#: nothing to the question this script asks.
ORDINARY_CONDITIONS = tuple(condition for condition in CONDITIONS if condition not in ("paired", "multiregime"))


@dataclass(frozen=True)
class BackboneFinetuneConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    output_dir: str | None = None
    device: str = "cpu"
    require_cuda: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 1500
    validation_interval: int = 100
    patience: int = 8
    min_delta: float = 1e-4
    batch_size: int = 2
    accumulate_gradients: int = 1
    gradient_clip: float = 1.0
    support_size: int = 128
    query_count: int = 6
    validation_episodes: int = 16
    #: Share of steps drawing a multiregime episode; the rest split evenly
    #: across the ordinary conditions.
    multiregime_probability: float = 0.30
    multiregime_contamination: float | None = None


def _curriculum_weights(config: BackboneFinetuneConfig) -> dict[str, float]:
    remaining = 1.0 - config.multiregime_probability
    per_ordinary = remaining / len(ORDINARY_CONDITIONS)
    weights = {condition: per_ordinary for condition in ORDINARY_CONDITIONS}
    weights["multiregime"] = config.multiregime_probability
    return weights


def _draw_condition(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    probabilities = np.asarray([weights[name] for name in names])
    return names[int(rng.choice(len(names), p=probabilities))]


def sample_condition_episode(rng: np.random.Generator, regime, condition: str, config: BackboneFinetuneConfig):
    if condition == "multiregime":
        return sample_multiregime_episode(
            rng,
            regime=regime,
            batch_size=config.batch_size,
            support_size=config.support_size,
            query_count=config.query_count,
            contamination=config.multiregime_contamination,
            device=config.device,
        )
    return sample_episode(
        rng,
        regime=regime,
        condition=condition,
        batch_size=config.batch_size,
        support_size=config.support_size,
        query_count=config.query_count,
        device=config.device,
    )


def query_cross_entropy(model, episode) -> torch.Tensor:
    """Plain supervised loss on the predictive mean -- no teacher, no candidates."""
    logits = model(episode.support_x, episode.support_y, episode.query_x)[..., :2]
    return F.cross_entropy(logits.reshape(-1, 2), episode.query_y.reshape(-1).long())


@torch.no_grad()
def multiregime_diagnostics(model, episode) -> dict[str, float]:
    """Base/other error and error-detection AUROC from the *trained* mean itself.

    Unlike the frozen-mean pilot, `probability` here comes from the same
    parameters the loss trains, so both the error gap and the error-detection
    AUROC can genuinely move with training.
    """
    if episode.query_regime_source is None:
        return {}
    logits = model(episode.support_x, episode.support_y, episode.query_x)[..., :2]
    probability = logits.softmax(dim=-1)[..., 1]
    label = episode.query_y.to(probability.dtype)
    error = (probability - label).abs()
    source = episode.query_regime_source
    base_mask, other_mask = source == 0, source == 1
    result: dict[str, float] = {}
    if base_mask.any():
        result["multiregime_base_error"] = float(error[base_mask].mean())
    if other_mask.any():
        result["multiregime_other_error"] = float(error[other_mask].mean())
    if base_mask.any() and other_mask.any():
        result["multiregime_error_gap"] = result["multiregime_other_error"] - result["multiregime_base_error"]

    predicted = (probability >= 0.5).long()
    correct = (predicted == episode.query_y.long()).float().reshape(-1)
    positive = probability.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(positive * positive.log() + (1.0 - positive) * (1.0 - positive).log()).reshape(-1)
    mistakes = (1.0 - correct).cpu().numpy().astype(int)
    if len(set(mistakes.tolist())) == 2:
        from sklearn.metrics import roc_auc_score

        result["multiregime_error_detection_auroc"] = float(roc_auc_score(mistakes, entropy.cpu().numpy()))
    return result


@torch.no_grad()
def validate(model, config: BackboneFinetuneConfig) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(config.seed + 10_001)
    totals: dict[str, list[float]] = {}
    for index in range(config.validation_episodes):
        condition = ORDINARY_CONDITIONS[index % len(ORDINARY_CONDITIONS)] if index % 3 else "multiregime"
        episode = sample_condition_episode(rng, HELDOUT_REGIME, condition, config)
        loss = float(query_cross_entropy(model, episode))
        totals.setdefault("cross_entropy", []).append(loss)
        for key, value in multiregime_diagnostics(model, episode).items():
            totals.setdefault(key, []).append(value)
    model.train()
    return {key: float(np.mean(values)) for key, values in totals.items()}


def finetune(model, config: BackboneFinetuneConfig) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed + 1)
    weights = _curriculum_weights(config)
    model.to(config.device)
    model.train()

    import copy

    best_state = copy.deepcopy(model.state_dict())
    best_value = math.inf
    best_step = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for _micro in range(config.accumulate_gradients):
            condition = _draw_condition(rng, weights)
            episode = sample_condition_episode(rng, TRAIN_REGIME, condition, config)
            loss = query_cross_entropy(model, episode)
            (loss / config.accumulate_gradients).backward()
            totals["cross_entropy"] = (
                totals.get("cross_entropy", 0.0) + float(loss.detach()) / config.accumulate_gradients
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        row: dict[str, Any] = {"step": step, "gradient_norm": float(grad_norm), **totals}
        if step % config.validation_interval == 0 or step == config.max_steps:
            validation = validate(model, config)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation["cross_entropy"] < best_value - config.min_delta:
                best_value, best_step, stale = validation["cross_entropy"], step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    selection = {"best_step": best_step, "best_validation_cross_entropy": best_value}
    return model, history, selection


def run_finetune(config: BackboneFinetuneConfig, output_dir: str) -> Path:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available.")
    set_randomness_seed(config.seed)
    model = init_model_from_state_dict_file(str(Path(config.checkpoint).expanduser().resolve()))
    model, history, selection = finetune(model, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "history.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in history) + "\n")
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": asdict(config)}, output / "backbone.pth")
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BackboneFinetuneConfig()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=defaults.checkpoint)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--require-cuda", action="store_true")
    for name in (
        "seed",
        "max_steps",
        "validation_interval",
        "patience",
        "batch_size",
        "accumulate_gradients",
        "support_size",
        "query_count",
        "validation_episodes",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "min_delta", "gradient_clip", "multiregime_probability"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--multiregime-contamination", type=float, default=None)
    return parser


if __name__ == "__main__":
    arguments = vars(build_parser().parse_args())
    destination = arguments.pop("output_dir")
    print(f"Wrote backbone fine-tune artifacts to {run_finetune(BackboneFinetuneConfig(**arguments), destination)}")
