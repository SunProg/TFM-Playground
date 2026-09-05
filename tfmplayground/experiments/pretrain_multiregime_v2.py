"""V2 curriculum pretraining for standard, slot, and oracle nanoTabPFN."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from tfmplayground.experiments.multiregime_prior_dump import PriorDumpReader
from tfmplayground.experiments.multiregime_v2 import (
    RegimeEpisode,
    RegimeGeneratorConfig,
    legacy_single_regime_batch,
    sample_regime_episode,
    stack_regime_episodes,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    PlainPretrainingConfig,
    _restore_rng_state,
    _scheduler_lambda,
    _serializable_rng_state,
    make_prior,
    query_loss,
)
from tfmplayground.models.mufasa_slot_tabpfn import MufasaSlotTabPFNModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_backbone import (
    SlotBackboneMixtureModel,
    collect_support_attention_for_loss,
    install_slot_layers,
)
from tfmplayground.models.slot_regime import SlotRegimePrediction, slot_regime_loss
from tfmplayground.models.supervised_tabpfn import SupervisedNanoTabPFNModel, supervised_regime_loss
from tfmplayground.models.table_slot import SLOT_SCOPES, TableSlotModel
from tfmplayground.utils import set_randomness_seed

ModelType = Literal[
    "tabpfn",
    "supervised_tabpfn",
    "slot_head",
    "slot_backbone",
    "slot_tabpfn",
    "mufasa_slot_tabpfn",
    "table_slot_head",
    "table_slot_backbone",
    "table_slot_mufasa",
]
InputMode = Literal["latent", "oracle_one_hot"]
CurriculumMode = Literal["plain", "multiregime", "mixed", "curriculum"]
AUXILIARY_WEIGHTS = (0.01, 0.1, 1.0)
TRAINING_SEEDS = (2402, 2403, 2404)


@dataclass(frozen=True)
class CurriculumCell:
    num_regimes: int
    alpha: float
    imbalance_ratio: float


@dataclass(frozen=True)
class V2TrainingConfig:
    seed: int = 2402
    device: str = "cuda"
    require_cuda: bool = False
    max_steps: int = 10_000
    micro_batch_size: int = 8
    accumulate_gradients: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 2_000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    checkpoint_interval: int = 10_000
    validation_interval: int = 5_000
    validation_episodes: int = 16
    progress_interval: int = 0
    model_type: ModelType = "tabpfn"
    input_mode: InputMode = "latent"
    curriculum_mode: CurriculumMode = "curriculum"
    num_slots: int = 4
    aux_regime_weight: float = 0.0
    slot_layer_index: int = 0
    slot_layer_indices: tuple[int, ...] = (3, 4, 5)
    #: Which competitions a ``table_slot_*`` model runs: over the cells of each
    #: row, over the feature-pooled rows, or both.
    table_slot_scope: str = "cell_and_data"
    prior_dump: str | None = None
    num_slot_iterations: int = 3
    support_size: int = 128
    query_size: int = 32
    min_features: int = 2
    max_features: int = 12
    backend: Literal["analytic", "tabicl_scm"] = "analytic"
    difference_components: tuple[str, ...] = ("coefficients",)
    label_noise: float = 0.0
    gate_strength: float = 1.0
    single_regime_source: Literal["legacy", "matched"] = "matched"
    benchmark_gate_path: str | None = None
    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 6
    num_outputs: int = 2

    def architecture(self) -> dict[str, int]:
        return {
            "embedding_size": self.embedding_size,
            "num_attention_heads": self.num_attention_heads,
            "mlp_hidden_size": self.mlp_hidden_size,
            "num_layers": self.num_layers,
            "num_outputs": self.num_outputs,
        }

    def generator(self) -> RegimeGeneratorConfig:
        return RegimeGeneratorConfig(
            backend=self.backend,
            max_regimes=4,
            difference_components=self.difference_components,
            support_size=self.support_size,
            query_size=self.query_size,
            min_features=self.min_features,
            max_features=self.max_features,
            label_noise=self.label_noise,
            gate_strength=self.gate_strength,
            seed=self.seed,
            single_regime_source=self.single_regime_source,
        )


def validate_config(config: V2TrainingConfig) -> None:
    if config.seed not in TRAINING_SEEDS:
        # Tiny tests and debugging may intentionally use another seed, so this
        # is metadata guidance rather than a restriction.
        pass
    if config.model_type not in (
        "tabpfn",
        "supervised_tabpfn",
        "slot_head",
        "slot_backbone",
        "slot_tabpfn",
        "mufasa_slot_tabpfn",
        "table_slot_head",
        "table_slot_backbone",
        "table_slot_mufasa",
    ):
        raise ValueError(
            "model_type must be 'tabpfn', 'supervised_tabpfn', 'slot_head', 'slot_backbone', "
            "'slot_tabpfn', or 'mufasa_slot_tabpfn'."
        )
    if config.input_mode not in ("latent", "oracle_one_hot"):
        raise ValueError("input_mode must be 'latent' or 'oracle_one_hot'.")
    if config.curriculum_mode not in ("plain", "multiregime", "mixed", "curriculum"):
        raise ValueError("curriculum_mode must be plain, multiregime, mixed, or curriculum.")
    if config.max_steps < 1 or config.micro_batch_size < 1 or config.accumulate_gradients < 1:
        raise ValueError("step and batch counts must be positive.")
    if config.validation_interval < 1 or config.validation_episodes < 1:
        raise ValueError("validation_interval and validation_episodes must be positive.")
    if config.progress_interval < 0:
        raise ValueError("progress_interval must be non-negative.")
    if config.model_type in ("tabpfn", "slot_backbone") and config.aux_regime_weight != 0:
        raise ValueError("aux_regime_weight requires a model with a regime supervision path.")
    if config.aux_regime_weight < 0:
        raise ValueError("aux_regime_weight must be non-negative.")
    if config.aux_regime_weight not in (0.0, *AUXILIARY_WEIGHTS):
        raise ValueError(f"aux_regime_weight must be zero or one of {AUXILIARY_WEIGHTS}.")
    if (
        config.aux_regime_weight > 0
        and config.model_type
        in (
            "slot_head",
            "slot_tabpfn",
            "mufasa_slot_tabpfn",
        )
        and config.num_slots < 4
    ):
        raise ValueError("Auxiliary supervision requires num_slots >= every curriculum K (four).")
    if config.num_slots < 1:
        raise ValueError("num_slots must be positive.")
    if not 0 <= config.slot_layer_index < config.num_layers:
        raise ValueError("slot_layer_index must identify a transformer block.")
    if config.table_slot_scope not in SLOT_SCOPES:
        raise ValueError(f"table_slot_scope must be one of {SLOT_SCOPES}, got {config.table_slot_scope!r}.")
    if config.table_slot_scope != "cell_and_data" and not config.model_type.startswith("table_slot_"):
        raise ValueError(f"table_slot_scope is a table-slot setting, not model_type={config.model_type!r}.")
    if config.model_type in ("mufasa_slot_tabpfn", "table_slot_backbone", "table_slot_mufasa"):
        if not config.slot_layer_indices:
            raise ValueError("MUFASA requires at least one tapped layer.")
        if len(set(config.slot_layer_indices)) != len(config.slot_layer_indices):
            raise ValueError("slot_layer_indices must be unique.")
        if any(index < 0 or index >= config.num_layers for index in config.slot_layer_indices):
            raise ValueError("slot_layer_indices must identify transformer blocks.")
    if config.single_regime_source == "legacy" and config.input_mode != "latent":
        raise ValueError("Legacy K=1 batches have no oracle z columns; use latent input or matched K=1.")
    if config.single_regime_source == "legacy" and config.model_type == "supervised_tabpfn":
        raise ValueError("supervised_tabpfn requires matched K=1 episodes so the auxiliary z loss is defined.")
    if config.prior_dump is not None and config.single_regime_source == "legacy":
        raise ValueError("prior_dump requires single_regime_source='matched'.")
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is unavailable.")
    if config.model_type in (
        "supervised_tabpfn",
        "slot_head",
        "slot_backbone",
        "slot_tabpfn",
        "mufasa_slot_tabpfn",
        "table_slot_head",
        "table_slot_backbone",
        "table_slot_mufasa",
    ) and config.device.startswith("cuda"):
        if config.benchmark_gate_path is None:
            raise ValueError("GPU v2 models require a passing benchmark execution_gate.json.")
        gate = json.loads(Path(config.benchmark_gate_path).read_text())
        if not gate.get("passed"):
            failed = sorted(name for name, passed in gate.get("checks", {}).items() if not passed)
            raise ValueError(f"GPU Slot-TabPFN benchmark gate has not passed: {failed}.")
    config.generator()


def curriculum_cell(
    step: int,
    max_steps: int,
    rng: np.random.Generator,
    *,
    mode: CurriculumMode = "mixed",
) -> CurriculumCell:
    """Draw one SCM regime cell under an explicit prior-composition mode."""
    if not 1 <= step <= max_steps:
        raise ValueError("step must lie in [1, max_steps].")
    if mode == "plain":
        return CurriculumCell(1, 0.0, 1.0)
    if mode == "multiregime":
        return CurriculumCell(
            int(rng.choice((2, 3, 4), p=(0.4, 0.35, 0.25))),
            float(rng.choice((0.25, 0.5, 1.0, 2.0))),
            float(rng.choice((1.0, 0.3, 0.1, 0.05))),
        )
    if mode == "mixed":
        if rng.random() >= 0.30:
            return CurriculumCell(1, 0.0, 1.0)
        return curriculum_cell(step, max_steps, rng, mode="multiregime")
    if mode != "curriculum":
        raise ValueError("mode must be plain, multiregime, mixed, or curriculum.")
    progress = (step - 1) / max_steps
    if progress < 0.20:
        return CurriculumCell(1, 0.0, 1.0)
    if progress < 0.40:
        return CurriculumCell(int(rng.choice((1, 2), p=(0.5, 0.5))), 0.25, 1.0)
    if progress < 0.70:
        return CurriculumCell(
            int(rng.choice((1, 2), p=(0.2, 0.8))),
            float(rng.choice((0.25, 0.5, 1.0))),
            1.0,
        )
    return CurriculumCell(
        int(rng.choice((1, 2, 3, 4), p=(0.1, 0.4, 0.3, 0.2))),
        float(rng.choice((0.0, 0.25, 0.5, 1.0, 2.0))),
        float(rng.choice((1.0, 0.3, 0.1, 0.05))),
    )


def build_model(config: V2TrainingConfig) -> torch.nn.Module:
    backbone = NanoTabPFNModel(**config.architecture())
    if config.model_type == "tabpfn":
        return backbone.to(config.device)
    if config.model_type == "supervised_tabpfn":
        return SupervisedNanoTabPFNModel(backbone, max_regimes=4).to(config.device)
    if config.model_type == "slot_head":
        from tfmplayground.models.slot_regime import NanoTabPFNSlotRegimeModel

        return NanoTabPFNSlotRegimeModel(
            backbone,
            num_slots=config.num_slots,
            max_classes=config.num_outputs,
            num_slot_iterations=config.num_slot_iterations,
            competitive_slots=True,
        ).to(config.device)
    if config.model_type == "mufasa_slot_tabpfn":
        return MufasaSlotTabPFNModel(
            backbone,
            num_slots=config.num_slots,
            layer_indices=config.slot_layer_indices,
            num_slot_iterations=config.num_slot_iterations,
            max_classes=config.num_outputs,
        ).to(config.device)
    if config.model_type.startswith("table_slot_"):
        return TableSlotModel(
            backbone,
            mode=config.model_type.removeprefix("table_slot_"),
            num_slots=config.num_slots,
            layer_indices=config.slot_layer_indices,
            num_slot_iterations=config.num_slot_iterations,
            max_classes=config.num_outputs,
            scope=config.table_slot_scope,
        ).to(config.device)
    install_slot_layers(
        backbone,
        num_slots=config.num_slots,
        num_slot_iterations=config.num_slot_iterations,
        slot_position="after_datapoint",
        max_classes=config.num_outputs,
        layer_indices=(config.slot_layer_index,),
    )
    if config.model_type == "slot_backbone":
        return backbone.to(config.device)
    return SlotBackboneMixtureModel(backbone, max_classes=config.num_outputs).to(config.device)


def _regime_slot_cost(probability: torch.Tensor, z: torch.Tensor, num_regimes: int) -> torch.Tensor:
    """Detached-compatible K-by-S assignment NLL for one episode."""
    rows = []
    for regime in range(num_regimes):
        mask = z == regime
        if not bool(mask.any()):
            rows.append(torch.zeros(probability.shape[-1], device=probability.device, dtype=probability.dtype))
        else:
            rows.append(-probability[mask].clamp_min(1e-12).log().mean(dim=0))
    return torch.stack(rows)


def hungarian_auxiliary_loss(
    prediction: SlotRegimePrediction,
    episode: RegimeEpisode,
    *,
    support_attention: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[np.ndarray, np.ndarray]]]:
    """Permutation-invariant L_z with equal support and query contributions."""
    support_probability = prediction.support_attention if support_attention is None else support_attention
    query_probability = prediction.gate()
    active_counts = episode.active_regime_mask.sum(dim=-1)
    if support_probability.shape[-1] < int(active_counts.max().item()):
        raise ValueError("Auxiliary supervision requires num_slots >= K.")
    losses = []
    assignments: list[tuple[np.ndarray, np.ndarray]] = []
    for batch in range(support_probability.shape[0]):
        k = int(episode.active_regime_mask[batch].sum().item())
        support_cost = _regime_slot_cost(support_probability[batch], episode.support_z[batch], k)
        query_cost = _regime_slot_cost(query_probability[batch], episode.query_z[batch], k)
        cost = 0.5 * (support_cost + query_cost)
        regimes, slots = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append((regimes, slots))
        regime_index = torch.as_tensor(regimes, device=cost.device)
        slot_index = torch.as_tensor(slots, device=cost.device)
        matched_support = support_cost[regime_index, slot_index].mean()
        matched_query = query_cost[regime_index, slot_index].mean()
        losses.append(0.5 * (matched_support + matched_query))
    return torch.stack(losses).mean(), assignments


def select_auxiliary_weight(validation_log_losses: dict[float, float]) -> float:
    """Held-out predictive selection, breaking exact ties toward smaller lambda."""
    candidates = {float(weight): float(loss) for weight, loss in validation_log_losses.items()}
    if set(candidates) != set(AUXILIARY_WEIGHTS):
        raise ValueError(f"Validation losses must contain exactly {AUXILIARY_WEIGHTS}.")
    return min(candidates, key=lambda weight: (candidates[weight], weight))


def _model_inputs(episode: RegimeEpisode, mode: InputMode) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return episode.oracle_inputs() if mode == "oracle_one_hot" else episode.latent_inputs()


def episode_loss(
    model: torch.nn.Module, episode: RegimeEpisode, config: V2TrainingConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    support_x, support_y, query_x = _model_inputs(episode, config.input_mode)
    prediction = model(support_x, support_y, query_x)
    if isinstance(prediction, SlotRegimePrediction):
        target_loss = slot_regime_loss(prediction, episode.query_y)
        auxiliary = target_loss.new_zeros(())
        if config.aux_regime_weight:
            if isinstance(model, SlotBackboneMixtureModel):
                support = collect_support_attention_for_loss(model.backbone)
            else:
                support = getattr(model, "last_support_attention_for_loss", None)
            if support is None:
                raise RuntimeError("Configured slot layer did not expose support attention for L_z.")
            auxiliary, _ = hungarian_auxiliary_loss(prediction, episode, support_attention=support)
        total = target_loss + config.aux_regime_weight * auxiliary
        return total, {"target_loss": float(target_loss.detach()), "auxiliary_loss": float(auxiliary.detach())}
    classes = prediction.shape[-1]
    target_loss = F.cross_entropy(prediction.reshape(-1, classes), episode.query_y.reshape(-1).long())
    auxiliary = target_loss.new_zeros(())
    if isinstance(model, SupervisedNanoTabPFNModel) and config.aux_regime_weight:
        z = torch.cat((episode.support_z, episode.query_z), dim=1)
        auxiliary = supervised_regime_loss(model, z)
    total = target_loss + config.aux_regime_weight * auxiliary
    return total, {"target_loss": float(target_loss.detach()), "auxiliary_loss": float(auxiliary.detach())}


def _legacy_config(config: V2TrainingConfig) -> PlainPretrainingConfig:
    return PlainPretrainingConfig(
        seed=config.seed,
        device=config.device,
        max_steps=config.max_steps,
        micro_batch_size=config.micro_batch_size,
        accumulate_gradients=config.accumulate_gradients,
        support_size=config.support_size,
        query_size=config.query_size,
        min_features=config.min_features,
        max_features=config.max_features,
        embedding_size=config.embedding_size,
        num_attention_heads=config.num_attention_heads,
        mlp_hidden_size=config.mlp_hidden_size,
        num_layers=config.num_layers,
        tensorboard=False,
    )


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: V2TrainingConfig,
    step: int,
    episode_rng: np.random.Generator,
    validation: dict[str, float] | None,
    prior_dump_index: int | None = None,
) -> dict[str, Any]:
    architecture: dict[str, Any] = config.architecture()
    if config.model_type == "supervised_tabpfn":
        architecture.update(
            {
                "model_kind": "supervised_tabpfn",
                "max_regimes": 4,
                "auxiliary_weight": config.aux_regime_weight,
            }
        )
    elif config.model_type == "slot_head":
        architecture.update(
            {
                "model_kind": "slot_head",
                "backbone_num_outputs": config.num_outputs,
                "num_slots": config.num_slots,
                "num_slot_iterations": config.num_slot_iterations,
                "competitive_slots": True,
                "max_classes": config.num_outputs,
            }
        )
    elif config.model_type in (
        "slot_backbone",
        "slot_tabpfn",
        "mufasa_slot_tabpfn",
        "table_slot_head",
        "table_slot_backbone",
        "table_slot_mufasa",
    ):
        architecture.update(
            {
                "model_kind": (
                    config.model_type
                    if config.model_type.startswith("table_slot_")
                    else "slot_backbone"
                    if config.model_type == "slot_backbone"
                    else "slot_backbone_mixture"
                    if config.model_type == "slot_tabpfn"
                    else "mufasa_slot_tabpfn"
                ),
                "num_slots": config.num_slots,
                "num_slot_iterations": config.num_slot_iterations,
                "competitive_slots": True,
                "slot_compatibility": "dot",
                "slot_position": "after_datapoint",
                "slot_layer_index": config.slot_layer_index,
                "slot_layer_indices": list(config.slot_layer_indices),
                "table_slot_scope": config.table_slot_scope,
                "max_classes": config.num_outputs,
                "target_inclusive_routing": config.model_type.startswith("table_slot_"),
            }
        )
    return {
        "model_type": f"multiregime_v2_{config.model_type}",
        "architecture": architecture,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_config": asdict(config),
        "generator_schema": asdict(config.generator()),
        "input_mode": config.input_mode,
        "aux_regime_weight": config.aux_regime_weight,
        "seed": config.seed,
        "step": step,
        "validation": validation,
        "rng_state": _serializable_rng_state(),
        "episode_rng_state": episode_rng.bit_generator.state,
        "prior_dump_index": prior_dump_index,
    }


@torch.no_grad()
def validate(model: torch.nn.Module, config: V2TrainingConfig) -> dict[str, float]:
    model.eval()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    losses = []
    try:
        for index in range(config.validation_episodes):
            generator = replace(
                config.generator(),
                regime_separation=(0.25, 0.5, 1.0, 2.0)[index % 4],
                imbalance_ratio=(1.0, 0.3, 0.1, 0.05)[(index // 4) % 4],
                single_regime_source="matched",
            )
            episode = sample_regime_episode(
                generator,
                num_regimes=(1, 2, 3, 4)[index % 4],
                seed=config.seed + 100_000 + index,
            ).to(config.device)
            loss, _ = episode_loss(model, episode, replace(config, aux_regime_weight=0.0))
            losses.append(float(loss))
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train()
    return {"prediction_log_loss": float(np.mean(losses)), "episodes": len(losses)}


def run_pretraining(
    config: V2TrainingConfig,
    output: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    validate_config(config)
    if resume_checkpoint is None:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite v2 run directory {output}.")
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    elif not output.is_dir():
        raise ValueError("A resumed run must use its existing output directory.")

    set_randomness_seed(config.seed)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule_config = _legacy_config(config)
    schedule_config = replace(
        schedule_config,
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scheduler_lambda(schedule_config))
    episode_rng = np.random.default_rng(config.seed + 1)
    start_step = 0
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng_state"])
        episode_rng.bit_generator.state = state["episode_rng_state"]
        start_step = int(state["step"])

    legacy_prior = None
    if config.single_regime_source == "legacy":
        legacy_prior = make_prior(_legacy_config(config), batches=1, device=config.device)
    prior_reader = PriorDumpReader(config.prior_dump) if config.prior_dump is not None else None
    if prior_reader is not None:
        expected = start_step * config.micro_batch_size * config.accumulate_gradients
        prior_reader.seek(int(state.get("prior_dump_index", expected)) if resume_checkpoint else expected)
    prior_dump_index = prior_reader.index if prior_reader is not None else None
    if prior_reader is not None:
        dump_mode = prior_reader.manifest.get("curriculum_mode", "mixed")
        if dump_mode != config.curriculum_mode:
            raise ValueError(
                f"prior dump curriculum_mode={dump_mode!r} does not match training curriculum_mode="
                f"{config.curriculum_mode!r}."
            )
    history_path = output / "history.jsonl"
    metadata_path = output / "episode_metadata.jsonl"
    mode = "a" if resume_checkpoint else "w"
    started = time.perf_counter()
    with history_path.open(mode) as history, metadata_path.open(mode) as metadata:
        for step in range(start_step + 1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_total = target_total = auxiliary_total = 0.0
            cells: list[CurriculumCell] = []
            normalizer = config.accumulate_gradients * config.micro_batch_size
            for _ in range(config.accumulate_gradients):
                if config.single_regime_source == "legacy":
                    # Preserve the byte-identical legacy K=1 path explicitly.
                    for _ in range(config.micro_batch_size):
                        cell = curriculum_cell(step, config.max_steps, episode_rng)
                        cells.append(cell)
                        assert legacy_prior is not None
                        batch = legacy_single_regime_batch(legacy_prior)
                        if config.model_type == "tabpfn":
                            loss = query_loss(model, batch)
                        else:
                            split = int(batch["train_test_split_index"])
                            output_prediction = model(
                                batch["x"][:, :split], batch["y"][:, :split], batch["x"][:, split:]
                            )
                            loss = slot_regime_loss(output_prediction, batch["y"][:, split:])
                        (loss / normalizer).backward()
                        loss_total += float(loss.detach()) / normalizer
                        target_total += float(loss.detach()) / normalizer
                else:
                    episodes: list[RegimeEpisode] = []
                    for _ in range(config.micro_batch_size):
                        if prior_reader is not None:
                            episode, episode_metadata = prior_reader.next_episode()
                            cell = CurriculumCell(
                                int(episode_metadata["num_regimes"]),
                                float(episode_metadata["requested_alpha"]),
                                float(episode_metadata["imbalance_ratio"]),
                            )
                        else:
                            cell = curriculum_cell(step, config.max_steps, episode_rng, mode=config.curriculum_mode)
                            episode_seed = int(episode_rng.integers(0, 2**32, dtype=np.uint32))
                            generator = replace(
                                config.generator(),
                                regime_separation=cell.alpha,
                                imbalance_ratio=cell.imbalance_ratio,
                                seed=episode_seed,
                                single_regime_source="matched",
                            )
                            episode = sample_regime_episode(
                                generator,
                                num_regimes=cell.num_regimes,
                                seed=episode_seed,
                                episode_id=f"train-{config.seed}-{step:06d}-{len(cells) + 1:03d}",
                            )
                            episode_metadata = episode.metadata
                        cells.append(cell)
                        metadata.write(json.dumps(episode_metadata, sort_keys=True, allow_nan=False) + "\n")
                        episodes.append(episode)
                        if prior_reader is not None:
                            prior_dump_index = prior_reader.index
                    batch_episode = stack_regime_episodes(episodes).to(config.device)
                    loss, parts = episode_loss(model, batch_episode, config)
                    (loss / config.accumulate_gradients).backward()
                    loss_total += float(loss.detach()) / config.accumulate_gradients
                    target_total += parts["target_loss"] / config.accumulate_gradients
                    auxiliary_total += parts["auxiliary_loss"] / config.accumulate_gradients
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
            if not math.isfinite(gradient_norm):
                # Some TabICL-SCM draws are numerically degenerate.  Do not let
                # one such streamed episode poison an otherwise reproducible
                # long run: parameters have not been stepped yet, so clearing
                # gradients and advancing the episode stream is equivalent to
                # rejecting this optimizer batch.
                optimizer.zero_grad(set_to_none=True)
                history.write(
                    json.dumps(
                        {
                            "step": step,
                            "skipped_non_finite_batch": True,
                            "learning_rate": float(scheduler.get_last_lr()[0]),
                            "curriculum_k": [cell.num_regimes for cell in cells],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                history.flush()
                metadata.flush()
                continue
            optimizer.step()
            scheduler.step()
            validation = (
                validate(model, config) if step % config.validation_interval == 0 or step == config.max_steps else None
            )
            row = {
                "step": step,
                "loss": loss_total,
                "target_loss": target_total,
                "auxiliary_loss": auxiliary_total,
                "gradient_norm": gradient_norm,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "curriculum_k": [cell.num_regimes for cell in cells],
                "curriculum_alpha": [cell.alpha for cell in cells],
                "curriculum_imbalance_ratio": [cell.imbalance_ratio for cell in cells],
                "elapsed_seconds": time.perf_counter() - started,
                **({f"validation_{key}": value for key, value in validation.items()} if validation else {}),
            }
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            metadata.flush()
            if validation is not None or (config.progress_interval and step % config.progress_interval == 0):
                # Match the older pretraining harness: validation checkpoints
                # are visible immediately in Slurm stdout as well as durable
                # JSONL, so `tail -f slurm-*.out` is useful during a long run.
                print(json.dumps(row, sort_keys=True), flush=True)
            if config.checkpoint_interval and step % config.checkpoint_interval == 0:
                torch.save(
                    _checkpoint(model, optimizer, scheduler, config, step, episode_rng, validation, prior_dump_index),
                    output / f"step-{step:06d}-checkpoint.pth",
                )

    final_validation = validate(model, config)
    final = output / "final_checkpoint.pth"
    torch.save(
        _checkpoint(
            model, optimizer, scheduler, config, config.max_steps, episode_rng, final_validation, prior_dump_index
        ),
        final,
    )
    (output / "selection.json").write_text(json.dumps(final_validation, indent=2, sort_keys=True) + "\n")
    return final


def build_parser() -> argparse.ArgumentParser:
    defaults = V2TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--model-type",
        choices=(
            "tabpfn",
            "supervised_tabpfn",
            "slot_head",
            "slot_backbone",
            "slot_tabpfn",
            "mufasa_slot_tabpfn",
            "table_slot_head",
            "table_slot_backbone",
            "table_slot_mufasa",
        ),
        default=defaults.model_type,
    )
    parser.add_argument("--input-mode", choices=("latent", "oracle_one_hot"), default=defaults.input_mode)
    parser.add_argument(
        "--curriculum-mode",
        choices=("plain", "multiregime", "mixed", "curriculum"),
        default=defaults.curriculum_mode,
    )
    parser.add_argument("--backend", choices=("analytic", "tabicl_scm"), default=defaults.backend)
    parser.add_argument("--single-regime-source", choices=("legacy", "matched"), default=defaults.single_regime_source)
    parser.add_argument("--benchmark-gate-path", default=defaults.benchmark_gate_path)
    parser.add_argument("--prior-dump", default=defaults.prior_dump)
    parser.add_argument("--slot-layer-indices", nargs="+", type=int, default=defaults.slot_layer_indices)
    parser.add_argument("--table-slot-scope", choices=SLOT_SCOPES, default=defaults.table_slot_scope)
    parser.add_argument("--difference-components", nargs="+", default=defaults.difference_components)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--require-cuda", action="store_true")
    for name in (
        "seed",
        "max_steps",
        "micro_batch_size",
        "accumulate_gradients",
        "warmup_steps",
        "checkpoint_interval",
        "validation_interval",
        "validation_episodes",
        "progress_interval",
        "num_slots",
        "slot_layer_index",
        "num_slot_iterations",
        "support_size",
        "query_size",
        "min_features",
        "max_features",
        "embedding_size",
        "num_attention_heads",
        "mlp_hidden_size",
        "num_layers",
        "num_outputs",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in (
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "gradient_clip",
        "aux_regime_weight",
        "label_noise",
        "gate_strength",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    return parser


def main(argv: list[str] | None = None) -> Path:
    values = vars(build_parser().parse_args(argv))
    output = Path(values.pop("output_dir"))
    resume = values.pop("resume_from")
    values["difference_components"] = tuple(values["difference_components"])
    return run_pretraining(V2TrainingConfig(**values), output, Path(resume) if resume else None)


if __name__ == "__main__":
    print(main())


__all__ = [
    "AUXILIARY_WEIGHTS",
    "TRAINING_SEEDS",
    "CurriculumCell",
    "CurriculumMode",
    "V2TrainingConfig",
    "build_model",
    "curriculum_cell",
    "episode_loss",
    "hungarian_auxiliary_loss",
    "run_pretraining",
    "select_auxiliary_weight",
    "validate_config",
]
