"""Fine-tune nanoTabPFN for coherent binary task hypotheses.

The default ``all`` stage first consistency-fine-tunes the unchanged model and
then trains a two-slot coherent hypothesis head on the resulting backbone.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.experiments.coherent_hypotheses import (
    enumerate_chain_joint_torch,
    exact_binary_joint,
    jensen_shannon_divergence,
    outcome_indices,
)
from tfmplayground.experiments.hypothesis_collapse import (
    NanoTabPFNBinaryPredictor,
    compute_trial_metrics,
    enumerate_chain_joint,
    exact_joint_distribution,
    generate_global_ambiguity_tasks,
    summarize_metrics,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.hypothesis import NanoTabPFNHypothesisModel, save_hypothesis_checkpoint
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed


@dataclass(frozen=True)
class CoherentTrainingConfig:
    seed: int = 2402
    stage: str = "all"
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    consistency_checkpoint: str | None = None
    output_dir: str | None = None
    device: str = "cpu"
    num_hypotheses: int = 2
    query_count: int = 4
    batch_size: int = 2
    accumulate_gradients: int = 8
    consistency_steps: int = 300
    slot_frozen_steps: int = 100
    slot_unfrozen_steps: int = 300
    validation_interval: int = 50
    patience: int = 4
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    controlled_only: bool = False
    evaluation_trials: int = 32
    ordinary_evaluation_batches: int = 8


@dataclass
class TrainingBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    posterior_b: torch.Tensor | None = None

    @property
    def controlled(self) -> bool:
        return self.posterior_b is not None


def validate_training_config(config: CoherentTrainingConfig) -> None:
    if config.stage not in {"all", "consistency", "slots"}:
        raise ValueError("stage must be all, consistency, or slots.")
    if config.num_hypotheses != 2:
        raise ValueError("The binary research model requires exactly two hypotheses.")
    if not 1 <= config.query_count <= 4:
        raise ValueError("query_count must be between one and four.")
    for name in (
        "batch_size",
        "accumulate_gradients",
        "validation_interval",
        "patience",
        "ordinary_evaluation_batches",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be at least one.")
    if min(config.consistency_steps, config.slot_frozen_steps, config.slot_unfrozen_steps) < 0:
        raise ValueError("training step counts cannot be negative.")
    if config.output_dir is not None and Path(config.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def controlled_batch(
    config: CoherentTrainingConfig,
    rng: np.random.Generator,
    *,
    support_sizes: Sequence[int] = (16, 32, 64),
    evidence_counts: Sequence[int] = (0, 2, 8, 16, 32, 64),
) -> TrainingBatch:
    support_size = int(rng.choice(support_sizes))
    evidence_count = int(rng.choice(evidence_counts))
    batch = generate_global_ambiguity_tasks(
        trials=config.batch_size,
        query_count=config.query_count,
        evidence_count=evidence_count,
        common_support_size=support_size,
        support_noise=0.1,
        rng=rng,
    )
    device = torch.device(config.device)
    return TrainingBatch(
        support_x=torch.from_numpy(batch.support_x).to(device),
        support_y=torch.from_numpy(batch.support_y).to(device),
        query_x=torch.from_numpy(batch.query_x).to(device),
        query_y=torch.from_numpy(np.repeat(batch.latent_task[:, None], config.query_count, axis=1).astype(np.int64)).to(
            device
        ),
        posterior_b=torch.from_numpy(batch.posterior_b.astype(np.float32)).to(device),
    )


def make_ordinary_iterator(config: CoherentTrainingConfig, num_steps: int) -> Iterator[dict] | None:
    if config.controlled_only:
        return None
    from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=num_steps,
        batch_size=config.batch_size,
        num_datapoints_min=32,
        num_datapoints_max=128,
        min_features=1,
        max_features=3,
        max_num_classes=2,
        prior_type="mix_scm",
        device=torch.device(config.device),
    )
    return iter(loader)


def next_ordinary_batch(iterator: Iterator[dict], query_count: int) -> TrainingBatch:
    for _ in range(20):
        raw = next(iterator)
        split = int(raw["train_test_split_index"])
        available_queries = raw["x"].shape[1] - split
        if available_queries >= query_count:
            return TrainingBatch(
                support_x=raw["x"][:, :split].float(),
                support_y=raw["y"][:, :split].float(),
                query_x=raw["x"][:, split : split + query_count].float(),
                query_y=raw["target_y"][:, split : split + query_count].long(),
            )
    raise RuntimeError("TabICL did not provide enough held-out rows after 20 attempts.")


def _initial_logits(model: NanoTabPFNModel, batch: TrainingBatch) -> torch.Tensor:
    full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
    return model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])[..., :2]


def consistency_loss(
    model: NanoTabPFNModel,
    batch: TrainingBatch,
    second_order: Sequence[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    query_count = batch.query_x.shape[1]
    canonical_order = tuple(range(query_count))
    canonical = enumerate_chain_joint_torch(model, batch.support_x, batch.support_y, batch.query_x, canonical_order)
    reordered = enumerate_chain_joint_torch(model, batch.support_x, batch.support_y, batch.query_x, second_order)
    logits = _initial_logits(model, batch)

    if batch.controlled:
        target_joint = exact_binary_joint(batch.posterior_b, query_count)
        joint_loss = (
            -0.5
            * (
                (target_joint * canonical.clamp_min(1e-12).log()).sum(-1)
                + (target_joint * reordered.clamp_min(1e-12).log()).sum(-1)
            ).mean()
        )
        target_marginals = batch.posterior_b[:, None].expand(-1, query_count)
        target_binary = torch.stack((1.0 - target_marginals, target_marginals), dim=-1)
        marginal_loss = -(target_binary * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    else:
        indices = outcome_indices(batch.query_y)
        joint_loss = (
            -0.5
            * (
                canonical.gather(1, indices[:, None]).clamp_min(1e-12).log()
                + reordered.gather(1, indices[:, None]).clamp_min(1e-12).log()
            ).mean()
        )
        marginal_loss = F.cross_entropy(logits.reshape(-1, 2), batch.query_y.reshape(-1))
    order_loss = jensen_shannon_divergence(canonical, reordered).mean()
    total = joint_loss + marginal_loss + 0.1 * order_loss
    return total, {
        "loss": float(total.detach()),
        "joint_loss": float(joint_loss.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "order_loss": float(order_loss.detach()),
    }


def hypothesis_loss(
    model: NanoTabPFNHypothesisModel,
    batch: TrainingBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
    prediction = model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])
    joint = prediction.joint_probabilities()
    marginals = prediction.marginal_probabilities()
    query_count = batch.query_x.shape[1]
    weight_loss = torch.zeros((), device=full_x.device)
    slot_loss = torch.zeros((), device=full_x.device)

    if batch.controlled:
        target_joint = exact_binary_joint(batch.posterior_b, query_count)
        joint_loss = -(target_joint * joint.clamp_min(1e-12).log()).sum(-1).mean()
        target_marginals = batch.posterior_b[:, None].expand(-1, query_count)
        target_binary = torch.stack((1.0 - target_marginals, target_marginals), dim=-1)
        marginal_loss = -(target_binary * marginals.clamp_min(1e-12).log()).sum(-1).mean()
        target_weights = torch.stack((1.0 - batch.posterior_b, batch.posterior_b), dim=-1)
        weight_loss = -(target_weights * prediction.slot_log_weights).sum(-1).mean()
        zeros = torch.zeros(batch.query_y.shape, device=full_x.device, dtype=torch.long)
        ones = torch.ones(batch.query_y.shape, device=full_x.device, dtype=torch.long)
        slot_loss = 0.5 * (
            F.cross_entropy(prediction.slot_logits[:, :, 0, :].reshape(-1, 2), zeros.reshape(-1))
            + F.cross_entropy(prediction.slot_logits[:, :, 1, :].reshape(-1, 2), ones.reshape(-1))
        )
    else:
        indices = outcome_indices(batch.query_y)
        joint_loss = -joint.gather(1, indices[:, None]).clamp_min(1e-12).log().mean()
        marginal_loss = F.nll_loss(marginals.clamp_min(1e-12).log().reshape(-1, 2), batch.query_y.reshape(-1))
    total = joint_loss + marginal_loss + 0.25 * weight_loss + 0.25 * slot_loss
    return total, {
        "loss": float(total.detach()),
        "joint_loss": float(joint_loss.detach()),
        "marginal_loss": float(marginal_loss.detach()),
        "weight_loss": float(weight_loss.detach()),
        "slot_loss": float(slot_loss.detach()),
    }


def _validation_loss(model, config: CoherentTrainingConfig, stage: str) -> float:
    rng = np.random.default_rng(config.seed + 10_000)
    model.eval()
    values = []
    with torch.no_grad():
        for support_size, evidence_count in ((16, 0), (16, 16), (64, 64)):
            batch = controlled_batch(
                config,
                rng,
                support_sizes=(support_size,),
                evidence_counts=(evidence_count,),
            )
            if stage == "consistency":
                value, _ = consistency_loss(model, batch, tuple(reversed(range(config.query_count))))
            else:
                value, _ = hypothesis_loss(model, batch)
            values.append(float(value))
    model.train()
    return float(np.mean(values))


def train_consistency_model(
    model: NanoTabPFNModel,
    config: CoherentTrainingConfig,
    rng: np.random.Generator,
) -> tuple[NanoTabPFNModel, list[dict]]:
    model.to(config.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.backbone_lr)
    ordinary = make_ordinary_iterator(config, config.consistency_steps * config.accumulate_gradients)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale_validations = 0
    microstep = 0

    for step in range(1, config.consistency_steps + 1):
        totals: dict[str, float] = {}
        optimizer.zero_grad()
        for _ in range(config.accumulate_gradients):
            use_controlled = ordinary is None or microstep % 2 == 0
            batch = (
                controlled_batch(config, rng) if use_controlled else next_ordinary_batch(ordinary, config.query_count)
            )
            order = tuple(int(value) for value in rng.permutation(config.query_count))
            loss, metrics = consistency_loss(model, batch, order)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
            microstep += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"stage": "consistency", "step": step, **totals}
        if step % config.validation_interval == 0 or step == config.consistency_steps:
            row["validation_loss"] = _validation_loss(model, config, "consistency")
            if row["validation_loss"] < best_validation:
                best_validation = row["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale_validations = 0
            else:
                stale_validations += 1
        history.append(row)
        if stale_validations >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history


def train_hypothesis_model(
    model: NanoTabPFNHypothesisModel,
    config: CoherentTrainingConfig,
    rng: np.random.Generator,
) -> tuple[NanoTabPFNHypothesisModel, list[dict]]:
    model.to(config.device)
    model.freeze_backbone()
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": head_parameters, "lr": config.head_lr},
        ]
    )
    total_steps = config.slot_frozen_steps + config.slot_unfrozen_steps
    ordinary = make_ordinary_iterator(config, total_steps * config.accumulate_gradients)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale_validations = 0
    microstep = 0

    for step in range(1, total_steps + 1):
        if step == config.slot_frozen_steps + 1:
            model.unfreeze_final_backbone_blocks(2)
        model.train()
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for _ in range(config.accumulate_gradients):
            use_controlled = ordinary is None or microstep % 2 == 0
            batch = (
                controlled_batch(config, rng) if use_controlled else next_ordinary_batch(ordinary, config.query_count)
            )
            loss, metrics = hypothesis_loss(model, batch)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
            microstep += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"stage": "slots", "step": step, **totals}
        if step % config.validation_interval == 0 or step == total_steps:
            row["validation_loss"] = _validation_loss(model, config, "slots")
            if row["validation_loss"] < best_validation:
                best_validation = row["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                stale_validations = 0
            else:
                stale_validations += 1
        history.append(row)
        if stale_validations >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history


class _SlotPredictor:
    def __init__(self, model: NanoTabPFNHypothesisModel, device: str):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def distributions(self, support_x, support_y, query_x) -> tuple[np.ndarray, np.ndarray]:
        support_x = torch.as_tensor(support_x, device=self.device)
        support_y = torch.as_tensor(support_y, device=self.device)
        query_x = torch.as_tensor(query_x, device=self.device)
        prediction = self.model(
            (torch.cat((support_x, query_x), dim=1), support_y),
            train_test_split_index=support_x.shape[1],
        )
        return prediction.joint_probabilities().cpu().numpy(), prediction.marginal_probabilities()[..., 1].cpu().numpy()


def evaluate_collapse_models(
    models: dict[str, NanoTabPFNModel | NanoTabPFNHypothesisModel],
    config: CoherentTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    seeds = (config.seed + 101, config.seed + 202, config.seed + 303)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for support_size in (16, 64, 128):
            for evidence_count in (0, support_size):
                batch = generate_global_ambiguity_tasks(
                    trials=config.evaluation_trials,
                    query_count=config.query_count,
                    evidence_count=evidence_count,
                    common_support_size=support_size,
                    support_noise=0.1,
                    rng=rng,
                )
                bayes_joint = exact_joint_distribution(batch.posterior_b, config.query_count)
                bayes_marginals = np.repeat(batch.posterior_b[:, None], config.query_count, axis=1)
                order = tuple(range(config.query_count))
                reverse_order = tuple(reversed(order))
                for model_name, model in models.items():
                    if isinstance(model, NanoTabPFNHypothesisModel):
                        joint, marginals = _SlotPredictor(model, config.device).distributions(
                            batch.support_x, batch.support_y, batch.query_x
                        )
                        reverse = joint
                    else:
                        predictor = NanoTabPFNBinaryPredictor(model, config.device)
                        joint = enumerate_chain_joint(predictor, batch.support_x, batch.support_y, batch.query_x, order)
                        reverse = enumerate_chain_joint(
                            predictor, batch.support_x, batch.support_y, batch.query_x, reverse_order
                        )
                        marginals = predictor.predict_binary_proba(batch.support_x, batch.support_y, batch.query_x)[
                            ..., 1
                        ]
                    metrics = compute_trial_metrics(
                        bayes_joint=bayes_joint,
                        predicted_joint=joint,
                        reverse_joint=reverse,
                        bayes_marginals=bayes_marginals,
                        predicted_marginals=marginals,
                    )
                    for trial in range(config.evaluation_trials):
                        row = {
                            "seed": seed,
                            "trial": trial,
                            "query_count": config.query_count,
                            "evidence_count": evidence_count,
                            "common_support_size": support_size,
                            "model": model_name,
                        }
                        row.update({name: values[trial] for name, values in metrics.items()})
                        rows.append(row)
    trials = pd.DataFrame(rows)
    summary_input = trials.drop(columns="seed")
    summary = summarize_metrics(summary_input)
    target = summary[
        (summary["model"] == "slots")
        & (summary["query_count"] == 4)
        & (summary["common_support_size"] == 128)
        & (summary["evidence_count"] == 128)
    ].set_index("metric")
    acceptance = {
        "incoherent_mass_at_m4_n128_r128": float(target.loc["incoherent_mass", "mean"]) if not target.empty else None,
        "marginal_js_at_m4_n128_r128": float(target.loc["marginal_js", "mean"]) if not target.empty else None,
    }
    acceptance["passes_incoherent_mass"] = (
        acceptance["incoherent_mass_at_m4_n128_r128"] is not None
        and acceptance["incoherent_mass_at_m4_n128_r128"] <= 0.171
    )
    acceptance["passes_marginal_js"] = (
        acceptance["marginal_js_at_m4_n128_r128"] is not None and acceptance["marginal_js_at_m4_n128_r128"] <= 0.0414
    )
    return trials, summary, acceptance


@torch.no_grad()
def evaluate_ordinary_accuracy(
    models: dict[str, NanoTabPFNModel | NanoTabPFNHypothesisModel],
    config: CoherentTrainingConfig,
) -> dict[str, float | None]:
    """Evaluate balanced accuracy on held-out on-the-fly binary TabICL tasks."""
    iterator = make_ordinary_iterator(config, config.ordinary_evaluation_batches)
    if iterator is None:
        return {name: None for name in models}
    class_correct = {name: torch.zeros(2, dtype=torch.long) for name in models}
    class_total = torch.zeros(2, dtype=torch.long)
    for _ in range(config.ordinary_evaluation_batches):
        batch = next_ordinary_batch(iterator, config.query_count)
        labels = batch.query_y.cpu()
        for class_index in range(2):
            class_total[class_index] += (labels == class_index).sum()
        full_x = torch.cat((batch.support_x, batch.query_x), dim=1)
        for model_name, model in models.items():
            model.to(config.device).eval()
            if isinstance(model, NanoTabPFNHypothesisModel):
                probabilities = model(
                    (full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1]
                ).marginal_probabilities()
                predicted = probabilities.argmax(-1).cpu()
            else:
                predicted = (
                    model((full_x, batch.support_y), train_test_split_index=batch.support_x.shape[1])[..., :2]
                    .argmax(-1)
                    .cpu()
                )
            for class_index in range(2):
                class_correct[model_name][class_index] += ((predicted == class_index) & (labels == class_index)).sum()
    scores = {}
    present = class_total > 0
    for model_name in models:
        recalls = class_correct[model_name][present].float() / class_total[present].float()
        scores[model_name] = float(recalls.mean()) if recalls.numel() else None
    return scores


def _architecture(model: NanoTabPFNModel) -> dict:
    return {
        "num_layers": model.num_layers,
        "embedding_size": model.embedding_size,
        "num_attention_heads": model.num_attention_heads,
        "mlp_hidden_size": model.mlp_hidden_size,
        "num_outputs": model.num_outputs,
    }


def save_consistency_checkpoint(
    path: Path,
    model: NanoTabPFNModel,
    config: CoherentTrainingConfig,
    source_hash: str,
) -> None:
    torch.save(
        {
            "model_type": "nanotabpfn_consistency",
            "architecture": _architecture(model),
            "model": model.state_dict(),
            "source_checkpoint_sha256": source_hash,
            "training_config": asdict(config),
            "stage": "consistency",
        },
        path,
    )


def load_consistency_checkpoint(path: str | Path) -> NanoTabPFNModel:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("model_type") != "nanotabpfn_consistency":
        raise ValueError("Checkpoint is not a consistency-fine-tuned nanoTabPFN model.")
    architecture = checkpoint["architecture"]
    model = NanoTabPFNModel(**architecture)
    model.load_state_dict(checkpoint["model"])
    return model


def run_training(config: CoherentTrainingConfig) -> Path:
    validate_training_config(config)
    set_randomness_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if config.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / "coherent_hypotheses" / timestamp
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    baseline = init_model_from_state_dict_file(str(checkpoint_path))
    consistency = (
        load_consistency_checkpoint(config.consistency_checkpoint)
        if config.stage == "slots" and config.consistency_checkpoint
        else init_model_from_state_dict_file(str(checkpoint_path))
    )
    histories = []
    evaluated_models: dict[str, NanoTabPFNModel | NanoTabPFNHypothesisModel] = {"baseline": baseline}

    if config.stage in {"all", "consistency"}:
        consistency, history = train_consistency_model(consistency, config, rng)
        histories.extend(history)
        save_consistency_checkpoint(output_dir / "consistency_checkpoint.pth", consistency, config, source_hash)
        evaluated_models["consistency"] = consistency

    if config.stage in {"all", "slots"}:
        slot_model = NanoTabPFNHypothesisModel(consistency, config.num_hypotheses)
        slot_model, history = train_hypothesis_model(slot_model, config, rng)
        histories.extend(history)
        save_hypothesis_checkpoint(
            output_dir / "hypothesis_checkpoint.pth",
            slot_model,
            training_config=asdict(config),
            source_checkpoint_sha256=source_hash,
            stage="slots",
        )
        evaluated_models["slots"] = slot_model

    pd.DataFrame(histories).to_csv(output_dir / "learning_curves.csv", index=False)
    trials, summary, acceptance = evaluate_collapse_models(evaluated_models, config)
    ordinary_accuracy = evaluate_ordinary_accuracy(evaluated_models, config)
    baseline_accuracy = ordinary_accuracy.get("baseline")
    slot_accuracy = ordinary_accuracy.get("slots")
    acceptance["ordinary_balanced_accuracy"] = ordinary_accuracy
    acceptance["passes_ordinary_accuracy"] = (
        baseline_accuracy is not None and slot_accuracy is not None and slot_accuracy >= baseline_accuracy - 0.01
    )
    trials.to_csv(output_dir / "evaluation_trial_metrics.csv", index=False)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    (output_dir / "ordinary_accuracy.json").write_text(json.dumps(ordinary_accuracy, indent=2) + "\n")
    (output_dir / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "consistency", "slots"), default="all")
    parser.add_argument("--checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--consistency-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=str(get_default_device()))
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--num-hypotheses", type=int, default=2)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-gradients", type=int, default=8)
    parser.add_argument("--consistency-steps", type=int, default=300)
    parser.add_argument("--slot-frozen-steps", type=int, default=100)
    parser.add_argument("--slot-unfrozen-steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--controlled-only", action="store_true")
    parser.add_argument("--evaluation-trials", type=int, default=32)
    parser.add_argument("--ordinary-evaluation-batches", type=int, default=8)
    return parser


def config_from_args(args: argparse.Namespace) -> CoherentTrainingConfig:
    return CoherentTrainingConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    output_dir = run_training(config)
    print(f"Wrote coherent-hypothesis training artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
