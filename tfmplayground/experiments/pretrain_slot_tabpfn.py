"""From-scratch slot TabPFN pretraining under four prior compositions.

The arms answer one question: which mixture of single-regime and multiregime
prior actually teaches a slot model to represent both regimes at once?

    plain        single-regime only            multiregime share 0.0
    multiregime  multiregime only              multiregime share 1.0
    mixed        70% single + 30% multiregime  multiregime share `multiregime_share`
    curriculum   single first, then ramp up    see `multiregime_probability`

Only ``mixed`` is new.  The other three delegate to
``pretrain_plain_nanotabpfn.multiregime_probability`` rather than restating its
arithmetic, so this script and the vanilla baseline runs cannot drift apart.
Everything else -- the TabICL prior construction, the multiregime episode
sampler, the warmup-cosine schedule, the RNG snapshotting and the non-finite
batch retry -- is imported from that module for the same reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.dump_multiregime_episodes import MultiregimeDumpLoader
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    _MAX_NON_FINITE_BATCH_RETRIES,
    _make_tensorboard_writer,
    _NullWriter,
    _preserved_rng_state,
    _restore_rng_state,
    _scheduler_lambda,
    _serializable_rng_state,
    make_prior,
    multiregime_batch,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    multiregime_probability as plain_multiregime_probability,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_attention import slot_assignment_entropy
from tfmplayground.models.slot_backbone import (
    COMPATIBILITY_MODES,
    SlotBackboneMixtureModel,
    collect_support_attention,
    install_slot_layers,
)
from tfmplayground.models.slot_regime import (
    NanoTabPFNSlotRegimeModel,
    SlotRegimePrediction,
    slot_regime_checkpoint,
    slot_regime_loss,
)
from tfmplayground.utils import set_randomness_seed

PRIOR_MODES = ("plain", "multiregime", "mixed", "curriculum")
MODEL_KINDS = ("slot", "vanilla", "slot_backbone", "slot_backbone_mixture")
#: The kinds whose slots live inside the transformer layers, so the competition
#: runs before full row attention has mixed the regimes together.  They differ
#: only in the readout: ``slot_backbone`` decodes the finished representation
#: and trains on one cross entropy, so its objective never mentions slots;
#: ``slot_backbone_mixture`` decodes one prediction per slot and trains on the
#: mixture NLL, so it does.
_SLOT_LAYER_KINDS = ("slot_backbone", "slot_backbone_mixture")


@dataclass(frozen=True)
class SlotPretrainingConfig:
    """One reproducible slot-TabPFN pretraining seed.

    Field names mirror ``PlainPretrainingConfig`` wherever the imported helpers
    read them, so the same ``make_prior``/``multiregime_batch``/scheduler code
    serves both scripts unchanged.
    """

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
    validation_interval: int = 5_000
    validation_batches: int = 16
    validation_episodes: int = 8
    support_size: int = 128
    query_size: int = 32
    min_features: int = 2
    max_features: int = 12
    max_classes: int = 2
    prior_type: str = "mix_scm"
    prior_mode: Literal["plain", "multiregime", "mixed", "curriculum"] = "plain"
    #: Constant multiregime share for ``prior_mode="mixed"``.  0.30 matches the
    #: share `finetune_multiregime_backbone.py` already uses.
    multiregime_share: float = 0.30
    multiregime_contamination: float = 0.3
    #: How feature-coherent the contaminated group is; 0.0 reproduces the
    #: original design exactly.  Above zero the relabelled rows concentrate on
    #: one side of a per-episode hyperplane, so regime membership becomes a
    #: latent *but structured* grouping -- the case slot attention's competition
    #: is built for, and the one the tag-independent design never presented.
    #: See ``continuous_episodes._contaminated_positions``.
    regime_coherence: float = 0.0
    #: Optional HDF5 dump (file or shard directory) to stream multiregime
    #: episodes from instead of generating them.  Generation is the CPU
    #: bottleneck of the multiregime arm; streaming removes it.
    multiregime_dump: str | None = None
    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 6
    #: "slot" is the mixture-over-slots model; "vanilla" is a plain
    #: NanoTabPFNModel trained in this identical harness -- same dump, same
    #: validation cadence, same TabArena settings -- so the architecture is
    #: the only variable when the two are compared.
    #: "slot_backbone" puts the competition inside every transformer layer
    #: instead of on top of the finished representation.
    model_kind: Literal["slot", "vanilla", "slot_backbone", "slot_backbone_mixture"] = "slot"
    num_slots: int = 2
    num_slot_iterations: int = 3
    competitive_slots: bool = True
    #: What a slot's claim on a support row is scored by, for ``slot_backbone``.
    #: "dot" is Locatello's own compatibility; "likelihood" scores
    #: ``log p(y | x, slot)``; "additive" keeps both.  See
    #: ``slot_backbone.COMPATIBILITY_MODES``.
    slot_compatibility: Literal["dot", "likelihood", "additive"] = "dot"
    #: One "epoch" for progress reporting: TabArena runs on each boundary.
    epoch_steps: int = 500
    #: Retain a durable checkpoint this often.  Every epoch also overwrites a
    #: single rolling `epoch-latest-checkpoint.pth` for TabArena to read, so a
    #: 500-step TabArena cadence does not mean 100 retained 55MB files per arm.
    checkpoint_interval: int = 10_000
    tabarena_every_epoch: bool = False
    tabarena_folds: int = 5
    tabarena_repeats: int = 10
    tabarena_subsample: int = 2_048
    tabarena_cache_directory: str | None = None
    tensorboard: bool = True

    @property
    def rows(self) -> int:
        return self.support_size + self.query_size

    def architecture(self) -> dict[str, int]:
        return {
            "embedding_size": self.embedding_size,
            "num_attention_heads": self.num_attention_heads,
            "mlp_hidden_size": self.mlp_hidden_size,
            "num_layers": self.num_layers,
            "num_outputs": self.max_classes,
        }


def validate_config(config: SlotPretrainingConfig) -> None:
    if config.prior_mode not in PRIOR_MODES:
        raise ValueError(f"prior_mode must be one of {PRIOR_MODES}, got {config.prior_mode!r}.")
    if not 0.0 <= config.multiregime_share <= 1.0:
        raise ValueError("multiregime_share must lie in [0, 1].")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if config.model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {config.model_kind!r}.")
    if config.slot_compatibility not in COMPATIBILITY_MODES:
        raise ValueError(
            f"slot_compatibility must be one of {COMPATIBILITY_MODES}, got {config.slot_compatibility!r}."
        )
    if config.slot_compatibility != "dot" and config.model_kind not in _SLOT_LAYER_KINDS:
        # The head variant builds its slots from the finished representation and
        # has no per-layer classifier to score a likelihood with, so silently
        # accepting the flag there would report a run nobody configured.
        raise ValueError(
            f"slot_compatibility={config.slot_compatibility!r} needs in-backbone slot layers "
            f"({' or '.join(_SLOT_LAYER_KINDS)}), not model_kind={config.model_kind!r}."
        )
    if config.num_slots < 1:
        raise ValueError("num_slots must be positive.")
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but no CUDA device is available.")


def multiregime_probability(config: SlotPretrainingConfig, step: int) -> float:
    """Multiregime share for this optimizer step.

    ``mixed`` is a constant share.  Every other mode delegates verbatim to the
    plain pretraining script so the ``plain``/``multiregime``/``curriculum``
    definitions stay identical to the vanilla runs this sweep compares against.
    """
    if config.prior_mode == "mixed":
        return config.multiregime_share
    return plain_multiregime_probability(config, step)


def training_batch(
    config: SlotPretrainingConfig,
    prior,
    episode_rng: np.random.Generator,
    step: int,
    multiregime_source=None,
):
    """Draw one batch under the configured prior composition.

    ``multiregime_source`` is an optional dump loader; when absent the
    episodes are generated on the fly exactly as the vanilla script does.
    The draw against ``probability`` happens either way, so the two paths
    consume the same RNG stream and stay comparable.
    """
    probability = multiregime_probability(config, step)
    if probability == 0.0 or episode_rng.random() >= probability:
        if prior is None:
            raise RuntimeError("The ordinary TabICL prior is required for this batch.")
        return next(iter(prior))
    if multiregime_source is not None:
        return multiregime_source.sample()
    return multiregime_batch(config, episode_rng)


def slot_batch_loss(model, batch) -> torch.Tensor:
    """Query-label loss, for either batch shape and either model.

    Dict batches come from the TabICL prior and carry their split index;
    ``ContinuousEpisode`` batches come from the multiregime sampler.  For the
    slot model this is the mixture NLL; for the vanilla control it is the plain
    cross entropy `pretrain_plain_nanotabpfn.query_loss` uses, so the two differ
    only in the model, not in how they are scored.
    """
    if isinstance(batch, dict):
        split = int(batch["train_test_split_index"])
        x, y = batch["x"], batch["y"]
        support_x, support_y, query_x, target = x[:, :split], y[:, :split], x[:, split:], y[:, split:]
    else:
        support_x, support_y, query_x, target = batch.support_x, batch.support_y, batch.query_x, batch.query_y
    output = model(support_x, support_y, query_x)
    if isinstance(output, SlotRegimePrediction):
        return slot_regime_loss(output, target)
    classes = output.shape[-1]
    return F.cross_entropy(output.reshape(-1, classes), target.reshape(-1).long())


def gate_regime_auc(prediction, episode) -> float | None:
    """Does the per-query slot gate separate base rows from contaminated rows?

    ``query_regime_source`` is diagnostic only and never reaches the model
    (``continuous_episodes.py:105-110``); it is read here purely to score.

    Slots are anonymous, so which slot corresponds to the base regime is
    arbitrary and the raw AUC is only meaningful up to a flip.  Taking
    ``max(auc, 1 - auc)`` is the slot-to-regime matching done *in the metric*,
    which is the one place matching belongs -- never in the loss.
    """
    if getattr(episode, "query_regime_source", None) is None:
        return None
    labels = episode.query_regime_source.reshape(-1).cpu().numpy()
    if labels.min() == labels.max():
        return None
    scores = prediction.gate()[..., 0].reshape(-1).detach().cpu().numpy()
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def identifiable_support_rows(candidate_support_positive) -> torch.Tensor | None:
    """Rows where the two label functions actually disagree, ``(batch, support)``.

    Both functions are evaluated on the same features, so a row relabelled to
    the value the base function would have produced anyway is tagged
    contaminated while being observationally identical to a clean row.  Half
    of all contaminated rows are in that position (measured: 0.500 over 19,200
    rows), which caps a perfect detector at AUC 0.75 rather than 1.0.  Scoring
    only the identifiable rows restores a ceiling of 1.0 and stops the metric
    reporting an unreachable target.
    """
    if candidate_support_positive is None:
        return None
    candidates = candidate_support_positive
    if candidates.shape[1] != 2:
        return None
    # `_candidate_probabilities` encodes a probability of class 1, so it means
    # nothing above two classes -- there it returns the raw label index, which
    # leaves the [0, 1] range.  Scoring unrestricted is correct then; silently
    # thresholding a label index at 0.5 would not be.
    if float(candidates.min()) < 0.0 or float(candidates.max()) > 1.0:
        return None
    return (candidates[:, 0] > 0.5) != (candidates[:, 1] > 0.5)


def support_binding_scores(
    support_attention: torch.Tensor, support_regime_source, identifiable=None
) -> dict[str, float]:
    """Did the slot competition actually partition the *context* by regime?

    This is the mechanism question, distinct from predictive accuracy.  Slot
    attention earns its keep only if a minority-regime support row is claimed by
    a slot of its own rather than averaged into the majority, so it is scored
    directly against the held-out per-row tag.

    ``support_regime_source`` is diagnostic only and never reaches the model
    (``continuous_episodes.py``); it is read here purely to score.

    Returns three numbers, each answering a different question:

    ``support_binding_auc``
        Threshold-free separation.  Each slot's attention column is used as a
        score for "this row is contaminated" and the best-separating slot wins,
        up to the arbitrary slot-to-regime flip.  0.5 is chance.
    ``support_binding_purity``
        Hard reading: assign every row to its ``argmax`` slot, then ask what
        fraction of rows sit in a slot whose majority regime is their own.
        Compare against ``support_regime_base_rate`` -- a single slot claiming
        everything already scores the majority fraction, so only the excess is
        evidence of binding.
    ``support_attention_entropy``
        Mean normalized entropy of the per-row assignment.  Near one means the
        attention is uniform and no row was claimed at all, which makes the
        other two numbers meaningless.
    """
    if support_regime_source is None:
        return {}
    labels = support_regime_source.reshape(-1).detach().cpu().numpy().astype(np.int64)
    if labels.min() == labels.max():
        return {}
    attention = support_attention.detach().reshape(-1, support_attention.shape[-1]).cpu().numpy()
    # Drop contaminated rows the two label functions agreed on: they carry the
    # tag but nothing observable distinguishes them, so counting them as misses
    # understates binding against a ceiling no detector can reach.
    keep = np.ones(labels.shape, dtype=bool)
    if identifiable is not None:
        distinct = identifiable.reshape(-1).detach().cpu().numpy().astype(bool)
        keep = distinct | (labels == 0)
        if keep.sum() < 2 or len(set(labels[keep])) < 2:
            return {}
    scored_labels, scored_attention = labels[keep], attention[keep]

    # Best-separating slot, up to the flip: matching in the metric, never the loss.
    aucs = [float(roc_auc_score(scored_labels, scored_attention[:, slot])) for slot in range(attention.shape[1])]
    auc = max(max(value, 1.0 - value) for value in aucs)

    assignment = scored_attention.argmax(axis=1)
    correct = 0
    for slot in range(scored_attention.shape[1]):
        members = scored_labels[assignment == slot]
        if members.size:
            correct += int(np.bincount(members, minlength=2).max())
    base_rate = float(np.bincount(scored_labels, minlength=2).max() / scored_labels.size)

    entropy = float(slot_assignment_entropy(support_attention.detach()).mean())
    return {
        "support_binding_auc": auc,
        "support_binding_purity": float(correct / scored_labels.size),
        "support_identifiable_fraction": float(keep.mean()),
        "support_regime_base_rate": base_rate,
        "support_attention_entropy": entropy,
    }


@torch.no_grad()
def validate(model: NanoTabPFNSlotRegimeModel, config: SlotPretrainingConfig) -> dict[str, float]:
    """Score every arm on the same two held-out streams, plus slot diagnostics.

    Both the ordinary and the multiregime stream are measured for every arm,
    regardless of what the arm trained on, so the four prior compositions are
    ranked on identical distributions.
    """
    with _preserved_rng_state():
        set_randomness_seed(config.seed + 100_000)
        model.eval()
        ordinary_losses: list[float] = []
        batches = iter(make_prior(config, batches=config.validation_batches * _MAX_NON_FINITE_BATCH_RETRIES))
        for _ in range(config.validation_batches):
            for _attempt in range(_MAX_NON_FINITE_BATCH_RETRIES):
                loss = slot_batch_loss(model, next(batches))
                if torch.isfinite(loss):
                    ordinary_losses.append(float(loss))
                    break
            else:
                raise RuntimeError("Could not draw a finite ordinary validation batch.")

        episode_rng = np.random.default_rng(config.seed + 200_000)
        multiregime_losses: list[float] = []
        gate_aucs: list[float] = []
        gate_entropies: list[float] = []
        binding: dict[str, list[float]] = {}
        for _ in range(config.validation_episodes):
            episode = multiregime_batch(config, episode_rng)
            loss = slot_batch_loss(model, episode)
            if not torch.isfinite(loss):
                continue
            multiregime_losses.append(float(loss))
            if config.model_kind == "vanilla":
                # No slots, so the binding diagnostics do not apply; only the
                # losses are comparable against the other arms.
                continue
            if config.model_kind == "slot_backbone":
                # The competition happens inside the layers, so the attention is
                # read back off the deepest layer that produced any.  This
                # variant has no query gate.
                attention = collect_support_attention(model)
                if attention is not None:
                    for key, value in support_binding_scores(
                        attention,
                        episode.support_regime_source,
                        identifiable_support_rows(episode.candidate_support_positive),
                    ).items():
                        binding.setdefault(key, []).append(value)
                continue
            prediction = model(episode.support_x, episode.support_y, episode.query_x)
            gate_entropies.append(float(prediction.gate_entropy().mean()))
            auc = gate_regime_auc(prediction, episode)
            if auc is not None:
                gate_aucs.append(auc)
            for key, value in support_binding_scores(
                prediction.support_attention,
                episode.support_regime_source,
                identifiable_support_rows(episode.candidate_support_positive),
            ).items():
                binding.setdefault(key, []).append(value)
    model.train()
    metrics: dict[str, float] = {"validation_batches": len(ordinary_losses)}
    # Every metric carries its own dispersion: these are means over a handful of
    # episodes, and without an interval there is no way to tell a real movement
    # from sampling noise.
    samples = {
        "query_cross_entropy": ordinary_losses,
        "multiregime_cross_entropy": multiregime_losses,
        "gate_entropy": gate_entropies,
        # The binding question: did a slot actually take the contaminated rows?
        # `gate_*` scores the query side, `support_*` the labelled context,
        # which is where the slots actually compete.
        "gate_regime_auc": gate_aucs,
        **{
            key: binding.get(key, [])
            for key in (
                "support_binding_auc",
                "support_binding_purity",
                "support_regime_base_rate",
                "support_attention_entropy",
                # How many rows survived the identifiability filter, so the
                # exclusion is visible rather than silent.
                "support_identifiable_fraction",
            )
        },
    }
    for name, values in samples.items():
        summary = summarize_samples(name, values)
        metrics.update(summary or {name: float("nan")})
    return metrics


def evaluate_tabarena_epoch(
    checkpoint: Path, output: Path, config: SlotPretrainingConfig, epoch: int
) -> dict[str, float]:
    """Progress-only TabArena-small evaluation, without disturbing training state.

    Mirrors ``pretrain_plain_nanotabpfn.evaluate_tabarena_epoch`` so the curve is
    directly comparable with the vanilla runs.  The checkpoint is read back from
    disk by ``evaluate_tabarena_small``, which routes slot checkpoints through
    ``SlotLogitsAdapter``, so the slot mixture is scored on real tables exactly
    as a plain backbone would be.
    """
    from tfmplayground.experiments.evaluate_tabarena_small import SmallTabArenaConfig, run

    destination = output / "tabarena" / f"epoch-{epoch:03d}"
    with _preserved_rng_state():
        run(
            SmallTabArenaConfig(
                standalone_checkpoints=f"current={checkpoint}",
                include_vanilla=False,
                output_dir=str(destination),
                device=config.device,
                cache_directory=config.tabarena_cache_directory,
                subsample=config.tabarena_subsample,
                folds=config.tabarena_folds,
                repeats=config.tabarena_repeats,
                include_sklearn=False,
                include_tabpfn=False,
                label_source="real",
                contamination=0.0,
                seed=config.seed,
            )
        )
    with (destination / "overall.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    current = next(row for row in rows if row["model"] == "current")
    metrics = {
        "tabarena_mean_roc_auc": float(current["mean_roc_auc"]),
        "tabarena_mean_accuracy": float(current["mean_accuracy"]),
    }
    # `overall.csv` holds only the means.  The per-fold rows are the sample the
    # interval needs, and at 5 folds x 10 repeats across 5 datasets there are
    # 250 of them, so this dispersion is far better resolved than the
    # per-episode validation ones.
    fold_path = destination / "fold_metrics.csv"
    if fold_path.is_file():
        with fold_path.open(newline="") as source:
            folds = [row for row in csv.DictReader(source) if row["model"] == "current"]
        for column, name in (("roc_auc", "tabarena_roc_auc"), ("accuracy", "tabarena_accuracy")):
            values = [float(row[column]) for row in folds if row.get(column) not in (None, "")]
            metrics.update(summarize_samples(name, values))
    return metrics


def summarize_samples(name: str, values: list[float]) -> dict[str, float]:
    """Mean plus dispersion for one metric's per-episode (or per-fold) samples.

    Every validation number here is an average over a handful of episodes, so a
    bare mean hides whether a movement is real.  This reports the sample
    standard deviation, the standard error and a 95% normal-approximation
    interval, the convention `hypothesis_collapse_results_interpretation.md`
    already documents for this repository's summaries.

    That document's caveat applies unchanged: the interval is an unbounded
    normal approximation over a small sample, so an endpoint can fall outside a
    metric's natural range -- below zero, or above one for an AUC.  That is a
    presentation artifact, not a claim the metric can take such a value.
    """
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    count = int(array.size)
    mean = float(array.mean())
    summary = {name: mean, f"{name}_n": count}
    if count < 2:
        return summary
    deviation = float(array.std(ddof=1))
    error = deviation / math.sqrt(count)
    summary[f"{name}_std"] = deviation
    summary[f"{name}_stderr"] = error
    summary[f"{name}_ci_low"] = mean - 1.96 * error
    summary[f"{name}_ci_high"] = mean + 1.96 * error
    return summary


def build_model(config: SlotPretrainingConfig):
    """The slot mixture model, or the plain backbone it is measured against."""
    backbone = NanoTabPFNModel(**config.architecture())
    if config.model_kind == "vanilla":
        return backbone.to(config.device)
    if config.model_kind in _SLOT_LAYER_KINDS:
        install_slot_layers(
            backbone,
            num_slots=config.num_slots,
            num_slot_iterations=config.num_slot_iterations,
            competitive_slots=config.competitive_slots,
            compatibility=config.slot_compatibility,
            max_classes=config.max_classes,
        )
        if config.model_kind == "slot_backbone":
            return backbone.to(config.device)
        return SlotBackboneMixtureModel(backbone, max_classes=config.max_classes).to(config.device)
    return NanoTabPFNSlotRegimeModel(
        backbone,
        num_slots=config.num_slots,
        max_classes=config.max_classes,
        num_slot_iterations=config.num_slot_iterations,
        competitive_slots=config.competitive_slots,
    ).to(config.device)


def _checkpoint(
    model: NanoTabPFNSlotRegimeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: SlotPretrainingConfig,
    step: int,
    validation: dict[str, float] | None,
    episode_rng: np.random.Generator,
) -> dict[str, Any]:
    if config.model_kind in _SLOT_LAYER_KINDS:
        # Slot settings must travel with the checkpoint: the layers cannot be
        # rebuilt from the five backbone keys alone.  The mixture variant adds a
        # decoder outside the backbone, so its state dictionary is prefixed and
        # the two kinds must not be loaded into one another -- `model_kind`
        # records which is which.
        checkpoint = {
            "architecture": {
                **config.architecture(),
                "model_kind": config.model_kind,
                "num_slots": config.num_slots,
                "num_slot_iterations": config.num_slot_iterations,
                "competitive_slots": config.competitive_slots,
                "slot_compatibility": config.slot_compatibility,
                "max_classes": config.max_classes,
            },
            "model": model.state_dict(),
            "training_config": asdict(config),
        }
    elif config.model_kind == "vanilla":
        # The plain inference format, so init_model_from_state_dict_file and
        # every existing evaluator read this control exactly as they read the
        # published vanilla runs.
        checkpoint = {
            "architecture": config.architecture(),
            "model": model.state_dict(),
            "training_config": asdict(config),
        }
    else:
        checkpoint = slot_regime_checkpoint(model, training_config=asdict(config))
    # Keep the prior composition visible in run metadata, as the vanilla script
    # does; `is_slot_regime_checkpoint` matches on architecture, not this string.
    checkpoint["model_type"] = f"{config.model_kind}_tabpfn_{config.prior_mode}_scm_pretraining"
    checkpoint.update(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "seed": config.seed,
            "step": step,
            "validation": validation,
            "rng_state": _serializable_rng_state(),
            "episode_rng_state": episode_rng.bit_generator.state,
        }
    )
    return checkpoint


def run_pretraining(
    config: SlotPretrainingConfig,
    output: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    validate_config(config)
    if resume_checkpoint is None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    elif not output.is_dir():
        raise ValueError("A resumed run must use the existing output directory.")

    set_randomness_seed(config.seed)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _scheduler_lambda(config))
    start_step = 0
    state = None
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        rng_state = state.get("rng_state")
        if rng_state is None:
            raise ValueError("Resume checkpoint is missing RNG state and cannot resume reproducibly.")
        _restore_rng_state(rng_state)

    prior = make_prior(config, batches=1) if config.prior_mode != "multiregime" else None
    multiregime_source = None
    # "none" is a sentinel rather than an empty string so that a sweep cell can
    # override the dump the batch script passes unconditionally: the flags are
    # word-split, and an empty argument would vanish and let `--multiregime-dump`
    # swallow the next flag instead.
    if config.multiregime_dump and config.multiregime_dump.lower() != "none":
        multiregime_source = MultiregimeDumpLoader(
            config.multiregime_dump,
            batch_size=config.micro_batch_size,
            device=config.device,
            seed=config.seed,
        )
        # A dump fixes the episodes, and with them the regime assignment. Reading
        # a coherence-0 dump while configured for a coherent prior would train on
        # the old task and report the new one, which is exactly the silent
        # mismatch that made twelve earlier runs measure the wrong model.
        mismatches = {
            name: (recorded, configured)
            for name, recorded, configured in (
                ("regime_coherence", multiregime_source.regime_coherence, config.regime_coherence),
                ("contamination", multiregime_source.contamination, config.multiregime_contamination),
            )
            if recorded != configured
        }
        if mismatches:
            raise ValueError(
                f"Dump at {config.multiregime_dump} does not match this run: "
                + "; ".join(f"{name} is {was} in the dump, {now} in the config" for name, (was, now) in mismatches.items())
                + ". Regenerate the dump or pass --multiregime-dump none to generate episodes on the fly."
            )
    episode_rng = np.random.default_rng(config.seed + 1)
    if state is not None:
        episode_rng.bit_generator.state = state["episode_rng_state"]

    history_path = output / "history.jsonl"
    writer = _make_tensorboard_writer(output) if config.tensorboard else _NullWriter()
    with history_path.open("a" if resume_checkpoint is not None else "w") as history:
        started_at = time.perf_counter()
        for step in range(start_step + 1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for _ in range(config.accumulate_gradients):
                for _attempt in range(_MAX_NON_FINITE_BATCH_RETRIES):
                    batch = training_batch(config, prior, episode_rng, step, multiregime_source)
                    loss = slot_batch_loss(model, batch)
                    if torch.isfinite(loss):
                        break
                else:
                    raise RuntimeError(f"Could not draw a finite training batch at step {step}.")
                (loss / config.accumulate_gradients).backward()
                loss_total += float(loss.detach()) / config.accumulate_gradients
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step}.")
            optimizer.step()
            scheduler.step()

            validation = None
            if step % config.validation_interval == 0 or step == config.max_steps:
                validation = validate(model, config)

            tabarena = None
            if config.epoch_steps and step % config.epoch_steps == 0:
                epoch = step // config.epoch_steps
                snapshot = _checkpoint(model, optimizer, scheduler, config, step, validation, episode_rng)
                # A single rolling file every epoch, so a 500-step TabArena
                # cadence costs one checkpoint of disk rather than one per epoch.
                rolling_path = output / "epoch-latest-checkpoint.pth"
                torch.save(snapshot, rolling_path)
                if config.checkpoint_interval and step % config.checkpoint_interval == 0:
                    torch.save(snapshot, output / f"step-{step:06d}-checkpoint.pth")
                if config.tabarena_every_epoch:
                    tabarena = evaluate_tabarena_epoch(rolling_path, output, config, epoch)

            row: dict[str, Any] = {
                "step": step,
                "loss": loss_total,
                "gradient_norm": gradient_norm,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "multiregime_probability": multiregime_probability(config, step),
                "elapsed_seconds": time.perf_counter() - started_at,
                **(tabarena or {}),
            }
            if validation is not None:
                row.update({f"validation_{key}": value for key, value in validation.items()})
                for key, value in validation.items():
                    writer.add_scalar(f"validation/{key}", value, step)
            if tabarena is not None:
                writer.add_scalar("tabarena/mean_roc_auc", tabarena["tabarena_mean_roc_auc"], step)
                writer.add_scalar("tabarena/mean_accuracy", tabarena["tabarena_mean_accuracy"], step)
            writer.add_scalar("train/loss", loss_total, step)
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    final_validation = validate(model, config)
    final_path = output / "final_checkpoint.pth"
    torch.save(
        _checkpoint(model, optimizer, scheduler, config, config.max_steps, final_validation, episode_rng),
        final_path,
    )
    (output / "selection.json").write_text(json.dumps(final_validation, indent=2, sort_keys=True) + "\n")
    writer.flush()
    writer.close()
    return final_path


def build_parser() -> argparse.ArgumentParser:
    defaults = SlotPretrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default=defaults.prior_mode)
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default=defaults.model_kind)
    parser.add_argument(
        "--slot-compatibility", choices=COMPATIBILITY_MODES, default=defaults.slot_compatibility
    )
    parser.add_argument("--prior-type", default=defaults.prior_type)
    parser.add_argument("--multiregime-dump", default=defaults.multiregime_dump)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--tabarena-every-epoch", dest="tabarena_every_epoch", action="store_true")
    # TabArena is binary; a multiclass run would score `logits[..., :2]` of a
    # wider head, which is meaningless rather than merely uninformative.
    parser.add_argument("--no-tabarena-every-epoch", dest="tabarena_every_epoch", action="store_false")
    parser.add_argument("--tabarena-cache-directory", default=defaults.tabarena_cache_directory)
    parser.add_argument("--no-competitive-slots", dest="competitive_slots", action="store_false")
    parser.set_defaults(
        tensorboard=defaults.tensorboard,
        competitive_slots=defaults.competitive_slots,
        tabarena_every_epoch=defaults.tabarena_every_epoch,
    )
    integer_fields = (
        "seed",
        "max_steps",
        "micro_batch_size",
        "accumulate_gradients",
        "warmup_steps",
        "validation_interval",
        "validation_batches",
        "validation_episodes",
        "support_size",
        "query_size",
        "min_features",
        "max_features",
        "max_classes",
        "embedding_size",
        "num_attention_heads",
        "mlp_hidden_size",
        "num_layers",
        "num_slots",
        "num_slot_iterations",
        "epoch_steps",
        "checkpoint_interval",
        "tabarena_folds",
        "tabarena_repeats",
        "tabarena_subsample",
    )
    float_fields = (
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "gradient_clip",
        "multiregime_share",
        "multiregime_contamination",
        "regime_coherence",
    )
    for name in integer_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in float_fields:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    return parser


def main(argv: list[str] | None = None) -> Path:
    arguments = vars(build_parser().parse_args(argv))
    output = Path(arguments.pop("output_dir"))
    resume = arguments.pop("resume_from")
    config = SlotPretrainingConfig(**arguments)
    return run_pretraining(config, output, Path(resume) if resume else None)


if __name__ == "__main__":
    print(main())
