"""Train a two-slot latent filter on paired TabICL SCM hypotheses."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    _rng_state,
    _set_rng_state,
    generate_h5_prior_bimodal_episodes,
    generate_prior_bimodal_episodes,
)
from tfmplayground.experiments.train_integrated_latent_filter import integrated_loss
from tfmplayground.experiments.train_sequential_latent_filter import (
    SequentialEpisodeBatch,
    _outcome_indices,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.integrated_latent_filter import (
    NanoTabPFNIntegratedLatentFilter,
    load_integrated_checkpoint,
    save_integrated_checkpoint,
)
from tfmplayground.utils import set_randomness_seed

# Checkpoint-selection criteria, both "lower is better". `validation_loss` is the training
# objective itself (what every run before this change used); `ensemble_query_nll` is a
# task-level metric the objective does not contain, and is the default because selecting on
# the composite loss optimises the diversity/coherence/coverage regularisers alongside fit.
SELECTION_METRICS = ("ensemble_query_nll", "validation_loss")


@dataclass(frozen=True)
class PriorBimodalTrainingConfig:
    seed: int = 2402
    checkpoint: str = "checkpoints/nanotabpfn.pth"
    controlled_checkpoint: str | None = None
    prior_dump: str | None = None
    output_dir: str | None = None
    device: str = "cpu"
    initial_support_count: int = 32
    stream_count: int = 32
    query_count: int = 4
    min_features: int = 1
    max_features: int = 16
    batch_size: int = 2
    accumulate_gradients: int = 4
    frozen_steps: int = 200
    partial_extra_steps: int = 100
    full_extra_steps: int = 100
    validation_interval: int = 25
    validation_episodes: int = 64
    selection_metric: str = "ensemble_query_nll"
    patience: int = 4
    head_learning_rate: float = 1e-4
    partial_backbone_learning_rate: float = 1e-5
    full_backbone_learning_rate: float = 5e-6
    diversity_weight: float = 0.05
    diversity_target: float = 0.20
    coherence_weight: float = 0.10
    residual_weight: float = 0.01
    best_particle_weight: float = 0.25
    posterior_weight: float = 0.0
    coverage_weight: float = 0.0
    assignment_weight: float = 0.0
    use_diversity: bool = True
    support_disagreement_max: float = 0.20
    stream_disagreement_min: float = 0.25
    query_disagreement_min: float = 0.25
    max_pair_attempts: int = 64
    evaluation_trials: int = 64
    evaluation_report_interval: int = 1000
    prior_type: str = "mix_scm"


def _episode_config(config: PriorBimodalTrainingConfig) -> PriorBimodalConfig:
    return PriorBimodalConfig(
        initial_support_count=config.initial_support_count,
        stream_count=config.stream_count,
        query_count=config.query_count,
        min_features=config.min_features,
        max_features=config.max_features,
        support_disagreement_max=config.support_disagreement_max,
        stream_disagreement_min=config.stream_disagreement_min,
        query_disagreement_min=config.query_disagreement_min,
        max_pair_attempts=config.max_pair_attempts,
        device=config.device,
        prior_type=config.prior_type,
    )


def _candidate_slot_log_likelihood(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """Log-likelihood each particle assigns to each candidate task's query labels.

    Returns shape (batch, num_particles, num_candidates).
    """
    slot_joint = prediction.slot_joint_log_probabilities()  # (b, K, 2**query_count)
    candidates = batch.candidate_query_y  # (b, num_candidates, query_count)
    num_candidates = candidates.shape[1]
    per_candidate = []
    for candidate in range(num_candidates):
        index = _outcome_indices(candidates[:, candidate])  # (b,)
        per_candidate.append(slot_joint.gather(-1, index[:, None, None].expand(-1, slot_joint.shape[1], 1)).squeeze(-1))
    return torch.stack(per_candidate, dim=-1)


def coverage_loss(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """Every candidate hypothesis must be explained by *some* particle.

    This is the dual of `best_particle`, which only ever references the TRUE candidate and is
    therefore fully satisfied by one competent particle plus K-1 junk ones. Summing over all
    candidates is what actually forces distinct particles to exist: particle collapse makes it
    impossible to explain two mutually-disagreeing candidates at once.
    """
    likelihood = _candidate_slot_log_likelihood(prediction, batch)  # (b, K, C)
    per_candidate = -torch.logsumexp(likelihood, dim=1)  # (b, C)
    return per_candidate.mean()


def assignment_loss(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """Permutation-invariant assignment of candidates to *distinct* particles.

    Enumerates injective candidate->particle matchings and charges the best one (Hungarian by
    enumeration; with C=2 candidates and K particles there are only K*(K-1) options). Unlike
    `coverage_loss`, this forbids one particle from claiming both hypotheses, which is the
    remaining way to satisfy coverage without differentiating.
    """
    likelihood = _candidate_slot_log_likelihood(prediction, batch)  # (b, K, C)
    num_particles, num_candidates = likelihood.shape[1], likelihood.shape[2]
    if num_particles < num_candidates:
        raise ValueError(
            f"need at least as many particles as candidates, got K={num_particles} and C={num_candidates}."
        )
    best = None
    for assignment in itertools.permutations(range(num_particles), num_candidates):
        score = sum(likelihood[:, assignment[c], c] for c in range(num_candidates))
        best = score if best is None else torch.maximum(best, score)
    return -(best / num_candidates).mean()


def posterior_supervision_loss(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """Supervise the particle posterior (`log_weights`) toward the true hypothesis.

    Nothing else in this family trains `log_weights` -- it appears only in evaluation
    metrics -- which is why `effective_particle_count` sat at the uniform value and the
    filter never learned to *update* its belief as stream evidence arrived.

    Particle identity is permutation-invariant, so there is no canonical index for a
    candidate. The target particle is therefore derived as a detached E-step: the particle
    with the largest *discriminative margin* for the true candidate over the alternative.
    Using the margin rather than plain argmax-likelihood targets the particle that best
    distinguishes the hypotheses, not merely one with high likelihood overall.

    Only the final posterior is supervised. The episodes are constructed so the support is
    deliberately uninformative about which candidate is true, so the correct belief early in
    the stream *is* near-uniform; supervising every step would force premature confidence.
    """
    likelihood = _candidate_slot_log_likelihood(prediction, batch)  # (b, K, C)
    true_candidate = batch.candidate_task.long()  # (b,)
    num_candidates = likelihood.shape[-1]
    true_index = true_candidate[:, None, None].expand(-1, likelihood.shape[1], 1)
    true_likelihood = likelihood.gather(-1, true_index).squeeze(-1)  # (b, K)
    # best alternative among the other candidates
    mask = torch.zeros_like(likelihood, dtype=torch.bool)
    mask.scatter_(-1, true_index, True)
    other_likelihood = likelihood.masked_fill(mask, float("-inf")).amax(-1)  # (b, K)
    margin = true_likelihood - other_likelihood
    if num_candidates < 2:
        margin = true_likelihood
    target = margin.detach().argmax(-1)  # (b,)
    final_log_weights = prediction.log_weights[:, -1]  # (b, K), already normalised
    return F.nll_loss(final_log_weights, target)


def bimodal_loss(
    model: NanoTabPFNIntegratedLatentFilter,
    batch: SequentialEpisodeBatch,
    config: PriorBimodalTrainingConfig,
    *,
    include_diversity: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config if include_diversity else replace(config, diversity_weight=0.0)
    base, metrics = integrated_loss(model, batch, loss_config, controlled=True)
    prediction = model(
        batch.initial_support_x,
        batch.initial_support_y,
        batch.stream_x,
        batch.stream_y,
        batch.query_x,
    )
    outcomes = _outcome_indices(batch.query_y)
    # slot_joint_log_probabilities() already returns LOG probabilities, so the gathered
    # values feed straight into logsumexp. A previous version applied
    # `.clamp_min(1e-12).log()` to them, which mapped every (negative) log-prob to the same
    # 1e-12 and then to a constant -27.631 -- making this whole term a constant 26.938 with
    # zero gradient regardless of the model's predictions. That silently disabled the only
    # pressure for particles to explain the query differently, which is why `slot_joint_js`
    # sat at ~0 in every run of this family.
    slot_joint = prediction.slot_joint_log_probabilities()
    slot_log_prob = slot_joint.gather(-1, outcomes[:, None, None].expand(-1, slot_joint.shape[1], 1)).squeeze(-1)
    # Permutation-invariant: at least one slot must explain the query. The
    # existing diversity term prevents both slots from solving it identically.
    best_particle = -torch.logsumexp(slot_log_prob, dim=1).mean()
    total = base + config.best_particle_weight * best_particle
    metrics = dict(metrics)
    # Optional (default off, so existing runs are unaffected): supervise the particle
    # posterior toward the true hypothesis. See posterior_supervision_loss.
    has_candidates = batch.candidate_query_y is not None
    if config.coverage_weight > 0 and has_candidates:
        coverage = coverage_loss(prediction, batch)
        total = total + config.coverage_weight * coverage
        metrics["coverage_loss"] = float(coverage.detach())
    if config.assignment_weight > 0 and has_candidates:
        assignment = assignment_loss(prediction, batch)
        total = total + config.assignment_weight * assignment
        metrics["assignment_loss"] = float(assignment.detach())
    if config.posterior_weight > 0 and batch.candidate_task is not None:
        posterior = posterior_supervision_loss(prediction, batch)
        total = total + config.posterior_weight * posterior
        metrics["posterior_loss"] = float(posterior.detach())
    metrics["best_particle_loss"] = float(best_particle.detach())
    if batch.pair_attempts is not None:
        metrics["pair_attempts"] = float(batch.pair_attempts.float().mean())
    metrics["loss"] = float(total.detach())
    return total, metrics


def _make_batch(config: PriorBimodalTrainingConfig, seed: int) -> SequentialEpisodeBatch:
    episode_config = _episode_config(config)
    rng = np.random.default_rng(seed)
    if config.prior_dump:
        return generate_h5_prior_bimodal_episodes(config.prior_dump, episode_config, rng, batch_size=config.batch_size)
    return generate_prior_bimodal_episodes(episode_config, rng, batch_size=config.batch_size)


def ensemble_query_nll(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """NLL the final ensemble posterior assigns to the true query labels, per episode."""
    indices = _outcome_indices(batch.query_y)
    joint = prediction.joint_probabilities()[:, -1]
    return -joint.gather(-1, indices[:, None]).squeeze(-1).clamp_min(1e-12).log()


def true_task_recovered(prediction, batch: SequentialEpisodeBatch) -> torch.Tensor:
    """Whether *some* particle scores the true candidate's query labels best, per episode.

    Order-invariant: candidates are ranked by likelihood, never by index.
    """
    slot_joint = prediction.slot_joint_log_probabilities().exp()
    candidates = batch.candidate_query_y
    num_candidates = candidates.shape[1]
    candidate_nll = torch.stack(
        [
            -slot_joint.gather(
                -1,
                _outcome_indices(candidates[:, candidate])[:, None, None].expand(-1, slot_joint.shape[1], 1),
            )
            .squeeze(-1)
            .clamp_min(1e-12)
            .log()
            for candidate in range(num_candidates)
        ],
        dim=-1,
    )
    closest = candidate_nll.argmin(-1)
    return (closest == batch.candidate_task[:, None]).any(-1).float()


@torch.no_grad()
def _validation_metrics(model, config: PriorBimodalTrainingConfig) -> dict[str, float]:
    """Held-out metrics on a FIXED validation set.

    The seed deliberately does not depend on the training step. An earlier version used
    `config.seed + 100_000 + step`, which redrew the episodes at every validation call, so
    comparing `validation_loss` across steps compared different data and best-state
    selection partly picked a lucky draw rather than a better model.

    A fixed per-batch seed is necessary but not sufficient: the SCM episode path draws task
    networks through the *global* torch/numpy RNG (`_candidate_dataset`, `prior.hp_sampling`),
    so identical `rng` seeds still yield different episodes depending on how much training has
    run. The global state is therefore pinned for the pass and restored afterwards, which also
    stops validation from consuming draws that training would otherwise have made -- training
    is now reproducible independently of `validation_interval`.

    `ensemble_query_nll` and `true_task_recovered` are the two task-level metrics the final
    evaluation reports, recorded here so that checkpoint selection can be judged against --
    or driven by -- something the objective itself does not contain.
    """
    model.eval()
    batches = max(1, -(-config.validation_episodes // config.batch_size))
    losses, nlls, recoveries = [], [], []
    outer_state = _rng_state()
    try:
        set_randomness_seed(config.seed + 100_000)
        for index in range(batches):
            batch = _make_batch(config, config.seed + 100_000 + index)
            value, _ = bimodal_loss(model, batch, config, include_diversity=config.use_diversity)
            losses.append(float(value))
            prediction = model(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.stream_y,
                batch.query_x,
            )
            nlls.append(float(ensemble_query_nll(prediction, batch).mean()))
            if batch.candidate_task is not None and batch.candidate_query_y is not None:
                recoveries.append(float(true_task_recovered(prediction, batch).mean()))
    finally:
        _set_rng_state(outer_state)
    model.train()
    metrics = {
        "validation_loss": float(np.mean(losses)),
        "validation_ensemble_query_nll": float(np.mean(nlls)),
        "validation_episodes": float(batches * config.batch_size),
    }
    if recoveries:
        metrics["validation_true_task_recovered"] = float(np.mean(recoveries))
    return metrics


def _selection_value(metrics: dict[str, float], config: PriorBimodalTrainingConfig) -> float:
    """Lower is better for both options."""
    if config.selection_metric == "ensemble_query_nll":
        return metrics["validation_ensemble_query_nll"]
    return metrics["validation_loss"]


def _train_stage(
    model: NanoTabPFNIntegratedLatentFilter,
    config: PriorBimodalTrainingConfig,
    *,
    stage: str,
    start_step: int,
    end_step: int,
) -> tuple[list[dict], torch.optim.Optimizer]:
    model.set_trainability(stage)
    model.to(config.device).train()
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.") and p.requires_grad]
    lr = (
        config.head_learning_rate
        if stage == "frozen"
        else (config.partial_backbone_learning_rate if stage == "partial" else config.full_backbone_learning_rate)
    )
    groups = [{"params": head, "lr": config.head_learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    trainable = list(model.trainable_parameters())
    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale = 0
    history = []
    for step in range(start_step + 1, end_step + 1):
        optimizer.zero_grad()
        totals: dict[str, float] = {}
        for accumulation in range(config.accumulate_gradients):
            batch = _make_batch(config, config.seed + step * config.accumulate_gradients + accumulation)
            loss, metrics = bimodal_loss(model, batch, config, include_diversity=config.use_diversity)
            (loss / config.accumulate_gradients).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / config.accumulate_gradients
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {"stage": stage, "step": step, **totals}
        if step % config.validation_interval == 0 or step == end_step:
            validation = _validation_metrics(model, config)
            row.update(validation)
            selection = _selection_value(validation, config)
            row["selection_value"] = selection
            if selection < best_validation:
                best_validation = selection
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return history, optimizer


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
        + (right * (right.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
    )


@torch.no_grad()
def evaluate(
    model,
    config: PriorBimodalTrainingConfig,
    vanilla_backbone=None,
    controlled_model=None,
) -> pd.DataFrame:
    model.to(config.device).eval()
    if vanilla_backbone is not None:
        vanilla_backbone.to(config.device).eval()
    if controlled_model is not None:
        controlled_model.to(config.device).eval()
    rows = []
    for trial in range(config.evaluation_trials):
        batch = _make_batch(config, config.seed + 300_000 + trial)
        prediction = model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        query_labels = batch.query_y
        indices = _outcome_indices(query_labels)
        slot_query_nll = (
            -slot_joint.gather(-1, indices[:, None, None].expand(-1, slot_joint.shape[1], 1))
            .squeeze(-1)
            .clamp_min(1e-12)
            .log()
        )
        true_task = batch.candidate_task
        recovered = true_task_recovered(prediction, batch)
        vanilla_nll = torch.full_like(slot_query_nll, float("nan"))
        controlled_nll = torch.full_like(slot_query_nll, float("nan"))
        if controlled_model is not None:
            controlled_prediction = controlled_model(
                batch.initial_support_x,
                batch.initial_support_y,
                batch.stream_x,
                batch.stream_y,
                batch.query_x,
            )
            controlled_joint = controlled_prediction.joint_probabilities()[:, -1]
            controlled_nll = -controlled_joint.gather(-1, indices[:, None]).squeeze(-1).clamp_min(1e-12).log()
        if vanilla_backbone is not None:
            support_x = torch.cat((batch.initial_support_x, batch.stream_x), dim=1)
            support_y = torch.cat((batch.initial_support_y, batch.stream_y), dim=1)
            vanilla_logits = vanilla_backbone(
                (torch.cat((support_x, batch.query_x), dim=1), support_y),
                train_test_split_index=support_x.shape[1],
            )[..., :2][:, -config.query_count :]
            vanilla_prob = vanilla_logits.softmax(-1)
            vanilla_joint = vanilla_prob[:, :, 1].prod(-1)
            # The binary joint probability is gathered explicitly below to
            # avoid assuming all query labels are one.
            vanilla_joint = torch.ones(vanilla_prob.shape[0], device=vanilla_prob.device)
            for query_index in range(config.query_count):
                vanilla_joint = vanilla_joint * vanilla_prob[:, query_index].gather(
                    -1, batch.query_y[:, query_index, None].long()
                ).squeeze(-1)
            vanilla_nll = -vanilla_joint.clamp_min(1e-12).log()
        slot_prob = slot_joint
        diversity = _js_divergence(slot_prob[:, 0], slot_prob[:, 1])
        rows.append(
            {
                "trial": trial,
                "feature_count": batch.initial_support_x.shape[-1],
                "true_task": int(true_task[0]),
                "support_disagreement": float(batch.support_disagreement[0]),
                "stream_disagreement": float(batch.stream_disagreement[0]),
                "query_disagreement": float(batch.query_disagreement[0]),
                "query_nll": float(slot_query_nll.mean()),
                "ensemble_query_nll": float(ensemble_query_nll(prediction, batch).mean()),
                "vanilla_query_nll": float(vanilla_nll.mean()),
                "controlled_query_nll": float(controlled_nll.mean()),
                "prequential_log_likelihood": float(prediction.prequential_log_likelihood.mean()),
                "slot_joint_js": float(diversity.mean()),
                "effective_particle_count": float((1.0 / prediction.log_weights.exp().square().sum(-1)[:, -1]).mean()),
                "true_task_recovered": float(recovered.mean()),
                "particle_query_disagreement": float(
                    F.softmax(prediction.query_logits, dim=-1)[:, :, 0]
                    .sub(F.softmax(prediction.query_logits, dim=-1)[:, :, 1])
                    .abs()
                    .mean()
                ),
                "pair_attempts": float(
                    batch.pair_attempts.float().mean() if batch.pair_attempts is not None else float("nan")
                ),
            }
        )
        if (trial + 1) % config.evaluation_report_interval == 0:
            block = pd.DataFrame(rows[-config.evaluation_report_interval :])
            print(
                f"evaluation episodes {trial + 1}/{config.evaluation_trials} | "
                f"ensemble_nll={block['ensemble_query_nll'].mean():.4f} | "
                f"vanilla_nll={block['vanilla_query_nll'].mean():.4f} | "
                f"recovery={block['true_task_recovered'].mean():.3f} | "
                f"slot_js={block['slot_joint_js'].mean():.6f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def run(config: PriorBimodalTrainingConfig) -> Path:
    if config.initial_support_count < 4 or config.stream_count < 1 or config.query_count < 1:
        raise ValueError("Invalid episode sizes.")
    if config.min_features < 1 or config.max_features < config.min_features:
        raise ValueError("Invalid feature range.")
    if config.evaluation_report_interval < 1:
        raise ValueError("evaluation_report_interval must be positive.")
    if config.validation_episodes < 1:
        raise ValueError("validation_episodes must be positive.")
    if config.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}.")
    set_randomness_seed(config.seed)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path("runs") / "prior_bimodal_filter" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    baseline = init_model_from_state_dict_file(config.checkpoint)
    model = NanoTabPFNIntegratedLatentFilter(copy.deepcopy(baseline), num_hypotheses=2).to(config.device)
    histories = []
    stage_ranges = (
        ("frozen", 0, config.frozen_steps),
        ("partial", config.frozen_steps, config.frozen_steps + config.partial_extra_steps),
        (
            "full",
            config.frozen_steps + config.partial_extra_steps,
            config.frozen_steps + config.partial_extra_steps + config.full_extra_steps,
        ),
    )
    optimizer = None
    for stage, start, end in stage_ranges:
        if end <= start:
            continue
        history, optimizer = _train_stage(model, config, stage=stage, start_step=start, end_step=end)
        histories.extend(history)
    pd.DataFrame(histories).to_csv(output_dir / "learning_curves.csv", index=False)
    controlled_model = None
    if config.controlled_checkpoint:
        controlled_model, _ = load_integrated_checkpoint(config.controlled_checkpoint, map_location=config.device)
    metrics = evaluate(
        model,
        config,
        vanilla_backbone=baseline,
        controlled_model=controlled_model,
    )
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    summary = metrics.groupby("feature_count", as_index=False).mean(numeric_only=True)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    save_integrated_checkpoint(
        output_dir / "selected_checkpoint.pth",
        model,
        training_config=asdict(config),
        source_checkpoint_sha256="prior-generated-from-" + config.checkpoint,
        stage="prior_bimodal_selected",
        lineage={"parent": config.checkpoint, "particle_count": 2},
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
    )
    return output_dir.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "checkpoint",
        "controlled_checkpoint",
        "prior_dump",
        "output_dir",
        "device",
        "prior_type",
    ):
        parser.add_argument(f"--{field.replace('_', '-')}", default=getattr(PriorBimodalTrainingConfig, field))
    for field in (
        "seed",
        "initial_support_count",
        "stream_count",
        "query_count",
        "min_features",
        "max_features",
        "batch_size",
        "accumulate_gradients",
        "frozen_steps",
        "partial_extra_steps",
        "full_extra_steps",
        "validation_interval",
        "validation_episodes",
        "patience",
        "evaluation_trials",
        "max_pair_attempts",
    ):
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=int, default=getattr(PriorBimodalTrainingConfig, field)
        )
    parser.add_argument(
        "--evaluation-report-interval",
        type=int,
        default=PriorBimodalTrainingConfig.evaluation_report_interval,
    )
    for field in (
        "head_learning_rate",
        "partial_backbone_learning_rate",
        "full_backbone_learning_rate",
        "diversity_weight",
        "diversity_target",
        "best_particle_weight",
        "support_disagreement_max",
        "stream_disagreement_min",
        "query_disagreement_min",
        "coherence_weight",
        "residual_weight",
        "posterior_weight",
        "coverage_weight",
        "assignment_weight",
    ):
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=float, default=getattr(PriorBimodalTrainingConfig, field)
        )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default=PriorBimodalTrainingConfig.selection_metric,
        help=(
            "Which held-out metric picks the best checkpoint (lower is better). "
            "'validation_loss' reproduces the pre-fix behaviour of selecting on the training "
            "objective, including its diversity/coherence/coverage regularisers."
        ),
    )
    parser.add_argument(
        "--use-diversity",
        action=argparse.BooleanOptionalAction,
        default=PriorBimodalTrainingConfig.use_diversity,
        help="Include the particle diversity regularizer (disable for the ablation).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    config = PriorBimodalTrainingConfig(**vars(build_parser().parse_args(argv)))
    print(f"Wrote prior-generated bimodal artifacts to {run(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
