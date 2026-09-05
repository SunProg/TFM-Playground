"""Array mapping and ranking for the slot-TabPFN prior-composition sweep.

Same structure as `multiregime_sweep.py`: the array index to configuration map
lives here, not in the Slurm script, so it is testable and reproducible.

Four arms, one per prior composition.  Every other training flag is held
identical across them -- the values are copied from
`scripts/slurm/pretrain_plain_nanotabpfn_a30.sbatch`, whose comment states the
rule this sweep depends on: "the training prior must be the only difference."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: One arm per prior composition, in a fixed documented order.
PRIOR_MODES = ("plain", "multiregime", "mixed", "curriculum")

#: Slot counts screened.  K=2 matches the prior's two label functions exactly;
#: K=3 and K=4 over-provision, which is how slot attention is normally used --
#: spare slots may sit empty rather than forcing a homogeneous context to split.
#: Ordered outermost so indices 0-3 remain the original K=2 arms and an
#: in-flight array keeps its index-to-arm mapping.
SLOT_COUNTS = (2, 3, 4)

#: Further slot counts for the in-backbone variant only.  Kept as a separate
#: constant rather than extending SLOT_COUNTS, because that constant also
#: sizes the head block: widening it would renumber every index after the
#: head cells, and an array's pending tasks resolve their flags at launch,
#: so a live job would silently start running a different arm.
EXTENDED_SLOT_COUNTS = (5, 6, 7, 8)

#: Constant multiregime share for the ``mixed`` arm: 70% single + 30% multi.
MULTIREGIME_SHARE = 0.30

#: Feature coherence of the contaminated group for the coherent-regime block.
#: At 0 (every earlier cell) the regime tag is independent of the features, so
#: nothing in a row predicts its group and slot competition has nothing to
#: compete over -- which is the leading explanation for 28 cells of flat
#: binding.  At 2.0 a row one standard deviation along the episode's hyperplane
#: is e^4 ~ 55 times likelier to be relabelled than one a standard deviation the
#: other way: strongly grouped, yet still stochastic, so a single row's regime
#: is never determined by its own features.  That last part matters -- at
#: infinite coherence the latent variable disappears and the episode collapses
#: to one piecewise label function that needs no mixture to fit.
REGIME_COHERENCE = 2.0

#: Single-competition table-slot scopes screened against the both-paths cells.
#: "cell_and_data" is not listed: it is what indices 114-125 already run.
TABLE_SLOT_SCOPES_SCREENED = ("cell", "data")

#: The design the measured detection ceiling actually leaves headroom on.
#: ``detection_ceiling`` job 36881374 puts the 2-class/128-support design at
#: 0.505 -- chance -- so every binding number from indices 0-149 was scored
#: against a target nothing could reach.  Three classes lifts identifiability
#: (two rules collide with probability ~1/C) and 512 support rows keep the
#: majority rule inferable, which is what the class count fights: measured
#: 0.6887 against 0.5049.  Four and five classes were measured too and do not
#: improve on three, because ``clean_fit_accuracy`` falls faster than
#: identifiability rises.  Feature count is worth +/-0.04 with no consistent
#: sign, so it stays at the twelve the ceiling cell used.
READABLE_DESIGN = {
    "max_classes": 3,
    "min_features": 12,
    "max_features": 12,
    "support_size": 512,
    "tabarena_max_predictors": 30,
    # 544 rows per episode against 160 at support 128, and the in-backbone
    # slot write-back attends over all of them: micro-batch 8 exhausted a
    # 24 GiB A30 at step 854.  The effective batch is held at 32 by raising
    # accumulation in step, so this changes peak memory and nothing else.
    #
    # Measured at micro-batch 2: 8.1 GiB on the in-backbone placement and
    # 3.0-3.9 GiB on the others, against a 24 GiB card -- activation memory
    # scales with the micro-batch, so 8 lands near 32 GiB (the observed OOM)
    # and 4 near 16 GiB.  Four is the largest that fits, and sixteen passes
    # per step left the GPU doing many small inefficient ones.
    "micro_batch_size": 4,
    "accumulate_gradients": 8,
}

#: Per-placement micro-batch for the readable design, holding the effective
#: batch at 32.  Measured at micro-batch 2: the head adapter used 3.0-3.9 GiB
#: against the in-backbone placement's 8.1 GiB, so the head can afford twice
#: the passes' worth of activations on a 24 GiB card and the others cannot.
#:
#: This is safe here and would not be elsewhere: the sampler draws feature
#: count, support size, query count, noise and contamination *once per call*
#: and shares them across the micro-batch, so micro-batch normally changes the
#: training distribution.  This block pins every one of those with flags, which
#: leaves grouping and RNG consumption as the only difference.  Comparisons
#: within a placement -- the scope axis this block exists for -- stay paired on
#: identical episodes; comparisons across placements become unpaired.
READABLE_MICRO_BATCH = {"table_slot_head": 8, "vanilla": 8}

#: Scopes screened on the readable design -- all three, including the
#: both-paths default, because no cell of that design has been run before.
TABLE_SLOT_SCOPES_READABLE = ("cell_and_data", "cell", "data")

#: Priors screened on the readable design.  "multiregime" is excluded: all six
#: runs of it on indices 114-149 sat at ln 2, i.e. chance, on every placement
#: and scope, so it measures a collapsed model rather than a design.
PRIOR_MODES_READABLE = ("plain", "mixed", "curriculum")

#: Compatibility functions screened in the compatibility block.  "dot" is not
#: among them: indices 16-27 already ran it at these settings, so re-running it
#: would buy nothing the existing cells do not already report.
COMPATIBILITY_MODES_SCREENED = ("likelihood", "additive")

#: The positive control's gate strength.  At 2.0 the regime stays genuinely
#: latent; at 8.0 the contaminated rows are all but a deterministic half-space,
#: so group membership becomes a property of the row itself -- which is the
#: assumption slot attention is built on and the one this task violates.
#: Measured supervised `x -> tag` AUC at this setting: 0.980.
#:
#: The task is *worse* as a benchmark here -- a single piecewise label function
#: fits it and no mixture is needed -- and that is deliberate.  This is not a
#: proposed design.  It asks only whether the competition can group rows when
#: the grouping is element-wise, so that a null result elsewhere cannot be
#: attributed to a bug in the implementation.
CONTROL_COHERENCE = 8.0

#: The one design measured to be both learnable *and* still to require
#: discovering the latent structure: `detection_ceiling` puts its achievable
#: AUC at 0.742, against 0.505 for the design every earlier arm trained on.
#: Three classes, four features, 15% contamination -- one label per row, regime
#: still latent, so a model scoring well has actually inferred something.
#: TabArena stays on for these cells.  A wider head is not a problem there:
#: at `max_classes=3` the TabICL prior yields a mix of 2- and 3-class episodes,
#: so the model is trained on binary tables too and learns to use its first two
#: outputs for them -- which is exactly what `predict_vanilla`'s `logits[..., :2]`
#: reads, and how TabPFN handles a variable class count generally.
LEARNABLE_DESIGN = {
    "max_classes": 3,
    "min_features": 4,
    "max_features": 4,
    "multiregime_contamination": 0.15,
}

#: The two closure objectives screened against the matched baseline at indices
#: 114/116/117.  Every earlier cell trains ``table_slot_head`` on the query
#: mixture NLL alone, which reads only the slot logits and the query gate: the
#: support competition carries no gradient, so one slot taking every row costs
#: nothing and `purity - base` has been exactly zero throughout.  These add the
#: tabular reading of Slot Attention's mask-weighted reconstruction -- the
#: assignment must explain the labels it claims -- alone and with a
#: balanced-sharpness term.  ``(reconstruction weight, MI weight)``.
#:
#: The reconstruction runs a second, label-blind backbone pass, so these cells
#: cost about twice a baseline step -- measured 0.99 s against 0.479 s per
#: micro-batch on CPU at these settings.  At 5,000 steps that is still far
#: inside the array's 24 h limit, but it is why the pilot screens six cells
#: rather than the full prior set.
CLOSURE_WEIGHTS = ((1.0, 0.0), (1.0, 0.05))

#: The two query-routing fixes screened against the current gate, per the
#: module comment where they are used below.
QUERY_ROUTING_MODES_SCREENED = ("blind_decoder", "blind_similarity")

#: The trainer's own default budget.  20 epochs of 500 steps.
SCREENING_STEPS = 10_000
#: Short budget for the coherent-regime block: 10 epochs, enough to see whether
#: binding moves off the null at all before spending the full budget on it.
COHERENT_STEPS = 5_000
SCREENING_SEED = 2402
#: Seeds for a three-seed rerun of whichever arm wins the screening pass.
FINAL_SEEDS = (2402, 2403, 2404)

#: Held identical across all four arms.
SHARED_FLAGS: tuple[str, ...] = (
    "--micro-batch-size 8",
    "--accumulate-gradients 4",
    "--learning-rate 0.0001",
    "--min-learning-rate 0.000001",
    "--warmup-steps 2000",
    "--weight-decay 0.01",
    "--gradient-clip 1.0",
    # Validate on every epoch boundary, alongside TabArena, so the synthetic
    # and real-table curves are sampled at the same 20 points.
    "--validation-interval 500",
    "--validation-batches 16",
    "--validation-episodes 8",
    "--support-size 128",
    "--query-size 32",
    "--min-features 2",
    "--max-features 12",
    "--max-classes 2",
    "--prior-type mix_scm",
    "--num-slot-iterations 3",
    # TabArena progress curve every epoch (500 steps), as the plain script
    # supports.  Full 5x10 fidelity costs ~68 s per evaluation, ~1.9 h over
    # the run; affordable because the episode dump removed ~11.5 h from the
    # multiregime arm, which was the binding wall-clock constraint.
    "--epoch-steps 500",
    "--tabarena-every-epoch",
    "--tabarena-folds 5",
    "--tabarena-repeats 10",
    # Only 5 durable checkpoints per arm; every epoch overwrites one rolling
    # file instead, which is what filled the volume on job 36848044.
    "--checkpoint-interval 10000",
)


def screening_configurations() -> list[dict[str, Any]]:
    """Every screened configuration, in a fixed order.

    Slot count outermost, then the vanilla controls appended last, so that
    extending the grid never re-maps an index an in-flight array is using.
    """
    grid = [
        {"prior_mode": prior_mode, "num_slots": num_slots, "model_kind": "slot"}
        for num_slots in SLOT_COUNTS
        for prior_mode in PRIOR_MODES
    ]
    # A plain nanoTabPFN per prior mode, trained in the identical harness.
    # Without it there is no way to tell whether the slot machinery changes
    # anything at all -- which is the question the K sweep left open.
    grid += [{"prior_mode": prior_mode, "num_slots": 2, "model_kind": "vanilla"} for prior_mode in PRIOR_MODES]
    # Slots inside every transformer layer rather than on top of the finished
    # representation.  Slot count outermost again, so the K=2 block keeps the
    # indices an in-flight array is already using.
    grid += [
        {"prior_mode": prior_mode, "num_slots": num_slots, "model_kind": "slot_backbone"}
        for num_slots in SLOT_COUNTS
        for prior_mode in PRIOR_MODES
    ]
    # Over-provisioned slot counts, appended after everything already submitted.
    grid += [
        {"prior_mode": prior_mode, "num_slots": num_slots, "model_kind": "slot_backbone"}
        for num_slots in EXTENDED_SLOT_COUNTS
        for prior_mode in PRIOR_MODES
    ]
    # The coherent-regime block.  Everything above ran at regime_coherence 0,
    # where the contaminated rows are a uniform random subset and no function of
    # a row's features can predict its group -- so slot competition had nothing
    # to bind to, and binding sat at its permutation null in all 28 cells.  This
    # block changes the task rather than the model: same code, same flags, same
    # seed, contaminated rows now drawn as a feature-coherent group.  A vanilla
    # control rides alongside each slot count, because a coherent prior is an
    # easier task for *any* model and the slot numbers mean nothing without it.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": num_slots,
            "model_kind": model_kind,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
        }
        for model_kind, num_slots in (("vanilla", 2), ("slot_backbone", 2), ("slot_backbone", 3))
        for prior_mode in PRIOR_MODES
    ]
    # The compatibility block.  Everything above scores a slot's claim on a row
    # by `<k(h_n), q(slot_k)>`, which asks whether the row *resembles* the slot.
    # These score `log p(y_n | x_n, slot_k)` instead -- whether the slot's
    # hypothesis *explains* the row's label -- either alone or added to the dot
    # product.
    #
    # Run at the same coherence as the block above, so the task is one a group
    # is findable in at all and the only thing left varying is what the
    # competition scores.  Cells 48-55 are then the matched dot-product control
    # at identical settings, and the comparison across the two blocks is exactly
    # the compatibility function.  A vanilla control rides in 44-47.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": num_slots,
            "model_kind": "slot_backbone",
            "slot_compatibility": compatibility,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
        }
        for compatibility in COMPATIBILITY_MODES_SCREENED
        for num_slots in (2, 3)
        for prior_mode in PRIOR_MODES
    ]
    # The mixture-readout block.  Every cell above trains `slot_backbone` on one
    # cross entropy over the finished representation, so the objective never
    # mentions slots: nothing rewards using two rather than one, and
    # one-slot-takes-all costs nothing.  That held across both compatibility
    # functions and both coherences -- `purity - base` was exactly zero in every
    # cell -- which is why changing what the competition *scores* could not have
    # helped on its own.
    #
    # `slot_backbone_mixture` decodes one prediction per slot and trains on the
    # mixture NLL, so the loss decomposes over slots while the competition still
    # runs inside the layers.  Both compatibilities are carried across, because
    # the loss and the score are separate claims and this is the first time
    # either has been tested with an objective that can see the split.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": num_slots,
            "model_kind": "slot_backbone_mixture",
            "slot_compatibility": compatibility,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
        }
        for compatibility in ("dot", "likelihood")
        for num_slots in (2, 3)
        for prior_mode in PRIOR_MODES
    ]
    # The learnable design.  Every block above trains on a task whose achievable
    # detection AUC is about 0.505 -- chance -- so a model result on it could
    # never be read: "the model failed" and "nothing can succeed" produce the
    # same number.  These run the one measured design where that is not true.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 2,
            "model_kind": model_kind,
            "max_steps": COHERENT_STEPS,
            **LEARNABLE_DESIGN,
        }
        for model_kind in ("vanilla", "slot", "slot_backbone")
        for prior_mode in PRIOR_MODES
    ]
    # The mixture backbone on the learnable design, appended rather than folded
    # into the block above so the cells already running keep their indices.
    # It is the only variant with both properties the argument needs -- the
    # competition runs before full row attention mixes the regimes together,
    # *and* the loss decomposes over slots -- and it was left out of the first
    # pass by carrying the earlier blocks' three model kinds over unexamined.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 2,
            "model_kind": "slot_backbone_mixture",
            "max_steps": COHERENT_STEPS,
            **LEARNABLE_DESIGN,
        }
        for prior_mode in PRIOR_MODES
    ]
    # The positive control.  Every null result in this project is consistent
    # with two stories: the competition is correct and the task violates its
    # inductive bias, or the implementation is quietly broken and would fail on
    # object discovery too.  Nothing measured so far separates them.  Here the
    # regime is nearly a deterministic function of the features, so a working
    # competition must group the rows; if binding stays flat even here, the
    # mechanism is broken rather than mismatched.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 2,
            "model_kind": model_kind,
            "regime_coherence": CONTROL_COHERENCE,
            "max_steps": COHERENT_STEPS,
            **LEARNABLE_DESIGN,
        }
        for model_kind in ("slot", "slot_backbone", "slot_backbone_mixture")
        for prior_mode in ("multiregime", "mixed")
    ]
    # Move the competition before datapoint attention.  These are matched to
    # positive controls 106-109, changing only the slot hook selected in each
    # transformer layer.  Appending preserves every historical array mapping.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 2,
            "model_kind": model_kind,
            "slot_position": "after_feature",
            "regime_coherence": CONTROL_COHERENCE,
            "max_steps": COHERENT_STEPS,
            **LEARNABLE_DESIGN,
        }
        for model_kind in ("slot_backbone", "slot_backbone_mixture")
        for prior_mode in ("multiregime", "mixed")
    ]
    # Table-slot variants on the existing coherent TabICL-SCM task.  This is a
    # model-only extension: the prior composition, coherent row assignment,
    # checkpoint cadence, validation, and in-process TabArena evaluation remain
    # the established slot pretraining path.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 4,
            "model_kind": model_kind,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
        }
        for model_kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
        for prior_mode in PRIOR_MODES
    ]
    # The scope ablation for the block above, which runs both competitions.  A
    # table-slot arm competes over the cells of each row and then over the
    # feature-pooled rows; these cells run one of the two alone, so a scope
    # comparison holds the placement, prior and task fixed and varies only
    # which competition exists.  Appended, so 0-125 keep their mapping.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 4,
            "model_kind": model_kind,
            "slot_scope": slot_scope,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
        }
        for slot_scope in TABLE_SLOT_SCOPES_SCREENED
        for model_kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
        for prior_mode in PRIOR_MODES
    ]
    # The readable-design block.  Everything above runs on a task whose
    # achievable detection AUC is 0.505, so its binding nulls cannot be
    # distinguished from "nothing was detectable"; these cells run the same
    # three scopes and three placements on the design that measures 0.6887,
    # and evaluate TabArena over fifteen binary datasets instead of five.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 4,
            "model_kind": model_kind,
            "slot_scope": slot_scope,
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
            **READABLE_DESIGN,
            **_readable_batch(model_kind),
        }
        for slot_scope in TABLE_SLOT_SCOPES_READABLE
        for model_kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
        for prior_mode in PRIOR_MODES_READABLE
    ]
    # The vanilla control for the block above, matched cell for cell on the
    # task.  Without it a number on the readable design has nothing to be
    # better or worse than.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 2,
            "model_kind": "vanilla",
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
            **READABLE_DESIGN,
            **_readable_batch("vanilla"),
        }
        for prior_mode in PRIOR_MODES_READABLE
    ]
    # The closure block.  Model-only, matched cell for cell against the
    # both-paths head arms at 114/116/117: same task, same slot count, same
    # budget, same seed, so the objective is the only thing that varies.
    # `multiregime` is left out for the reason the readable block leaves it
    # out -- every run of it sat at chance, which measures a collapsed model
    # rather than an objective.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 4,
            "model_kind": "table_slot_head",
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
            "support_reconstruction_weight": reconstruction,
            "slot_mi_weight": mi,
        }
        for reconstruction, mi in CLOSURE_WEIGHTS
        for prior_mode in PRIOR_MODES_READABLE
    ]
    # The query-routing block.  Every cell above -- closure or not -- routes
    # a query row through `_SlotDecoder`'s learned gate on the *labelled*-pass
    # embedding, which has already attended over real support labels through
    # ordinary full self-attention before the slot competition or its gate
    # ever runs.  A query can therefore get a good prediction without the
    # slot mechanism contributing anything, which an empirical check found is
    # consistent with what happens: query CE/AUC barely move between the
    # baseline and the closure arms even though the closure loss sharpens the
    # support side and per-slot query predictions genuinely diverge.
    #
    # `blind_decoder` reads the query embedding from the label-blind pass
    # instead (no label can reach it by attending over labelled support
    # states), keeping the learned gate.  `blind_similarity` additionally
    # replaces that gate with cosine similarity to each slot's blind-support
    # centroid, so the routing key is guaranteed recoverable from `x` alone on
    # both sides of the comparison -- never a label, for either the query or
    # the centroids it is compared against.
    #
    # Closure weights are fixed at the passing configuration
    # (`CLOSURE_WEIGHTS[1]`, reconstruction=1.0/mi=0.05) rather than re-swept:
    # this block varies the routing mechanism, not the support-side objective,
    # which 183-185 already screened.  Crossed with every scope, since a
    # routing fix's value could depend on which competition produced the
    # slots it routes against.  `multiregime` excluded for the same reason as
    # every other block above.
    grid += [
        {
            "prior_mode": prior_mode,
            "num_slots": 4,
            "model_kind": "table_slot_head",
            "regime_coherence": REGIME_COHERENCE,
            "max_steps": COHERENT_STEPS,
            "support_reconstruction_weight": CLOSURE_WEIGHTS[1][0],
            "slot_mi_weight": CLOSURE_WEIGHTS[1][1],
            "slot_scope": scope,
            "query_routing_mode": routing_mode,
            # Unlike the closure block, these score TabArena in-training over
            # the fifteen-task/thirty-predictor slice rather than the trainer's
            # five-task default, since a routing fix's payoff is exactly the
            # real-table query prediction number and there is no cost to
            # widening it before these cells start rather than after.
            "tabarena_max_predictors": 30,
        }
        for routing_mode in QUERY_ROUTING_MODES_SCREENED
        for scope in TABLE_SLOT_SCOPES_READABLE
        for prior_mode in PRIOR_MODES_READABLE
    ]
    return grid


def _readable_batch(model_kind: str) -> dict[str, int]:
    """Micro-batch and accumulation for one readable-design placement."""
    micro_batch = READABLE_MICRO_BATCH.get(model_kind)
    if micro_batch is None:
        return {}
    effective = READABLE_DESIGN["micro_batch_size"] * READABLE_DESIGN["accumulate_gradients"]
    if effective % micro_batch:
        raise ValueError(f"micro-batch {micro_batch} does not divide the effective batch {effective}.")
    return {"micro_batch_size": micro_batch, "accumulate_gradients": effective // micro_batch}


def configuration_label(configuration: dict[str, Any]) -> str:
    """Run directory name.

    K=2 keeps the bare prior-mode label it already had, so the arms already
    running under that mapping are unaffected by the grid being extended.
    """
    prior_mode, num_slots = configuration["prior_mode"], configuration["num_slots"]
    kind = configuration.get("model_kind", "slot")
    # A coherent-regime cell is a different task, not a different arm of the
    # same one, so it must never share a run directory with its coherence-0
    # counterpart -- resume would silently continue the wrong training.
    coherence = configuration.get("regime_coherence", 0.0)
    suffix = "" if coherence == 0.0 else f"-coh{coherence:g}"
    # Likewise a different compatibility is a different model, sharing every
    # other setting with the dot-product cells it must not overwrite.
    compatibility = configuration.get("slot_compatibility", "dot")
    if compatibility != "dot":
        suffix += f"-{compatibility}"
    if configuration.get("support_size") is not None:
        # A different support size is a different task, and the readable-design
        # cells share prior, scope and placement with cells that already exist.
        suffix += f"-s{configuration['support_size']}"
    if int(configuration.get("max_classes", 2)) != 2:
        # A different task, not a different arm: it must not share a run
        # directory with a cell trained on the unlearnable design.  This has to
        # precede every early return below, or the vanilla cells collide with
        # the vanilla arms of the original block.
        suffix += "-learnable"
    # A scope is a different model sharing every other setting with the
    # both-paths cells, so it needs its own directory for the same reason a
    # compatibility does.  The default keeps the running arms' labels.
    slot_scope = configuration.get("slot_scope", "cell_and_data")
    if slot_scope != "cell_and_data":
        suffix += f"-{slot_scope}"
    # A different objective is a different model sharing every other setting
    # with the baseline cells, so it needs its own directory for the same
    # reason a scope or a compatibility does.
    for key, marker in (("support_reconstruction_weight", "rec"), ("slot_mi_weight", "mi")):
        if configuration.get(key):
            suffix += f"-{marker}{configuration[key]:g}"
    # A different routing mode is a different model sharing every other
    # setting with the closure cells, for the same reason a scope is.
    routing_mode = configuration.get("query_routing_mode", "decoder")
    if routing_mode != "decoder":
        suffix += f"-{routing_mode}"
    position = configuration.get("slot_position", "after_datapoint")
    position_suffix = {
        "after_datapoint": "",
        "before_feature": "-before-feature",
        "after_feature": "-after-feature",
        "before_and_after_feature": "-before-and-after-feature",
    }
    suffix += position_suffix[position]
    if kind.startswith("table_slot_"):
        return f"{prior_mode}-{kind}-s{num_slots}{suffix}"
    if kind == "vanilla":
        return f"{prior_mode}-{kind}{suffix}"
    if kind == "slot_backbone_mixture":
        # A new block, so nothing constrains its names: K is always written out.
        # The older blocks leave K=2 unmarked, which reads badly next to the
        # `-coh2` coherence suffix -- four directories showing no slot count and
        # one showing `-k3` invites exactly the misreading it looks like.
        return f"{prior_mode}-slot_mixture-k{num_slots}{suffix}"
    if kind == "slot_backbone":
        # K=2 keeps the bare label the already-running block was submitted with.
        base = f"{prior_mode}-{kind}" if num_slots == 2 else f"{prior_mode}-{kind}-k{num_slots}"
        return f"{base}{suffix}"
    return f"{prior_mode}{suffix}" if num_slots == 2 else f"{prior_mode}-k{num_slots}{suffix}"


def configuration_flags(index: int, *, final: bool = False, seed: int | None = None) -> str:
    """Command-line flags for one array task."""
    configurations = screening_configurations()
    if not 0 <= index < len(configurations):
        raise IndexError(f"Array index {index} is outside 0..{len(configurations) - 1}.")
    configuration = configurations[index]
    coherence = float(configuration.get("regime_coherence", 0.0))
    flags = [
        f"--prior-mode {configuration['prior_mode']}",
        f"--num-slots {configuration['num_slots']}",
        f"--model-kind {configuration.get('model_kind', 'slot')}",
        f"--slot-compatibility {configuration.get('slot_compatibility', 'dot')}",
        f"--multiregime-share {MULTIREGIME_SHARE:g}",
        f"--regime-coherence {coherence:g}",
        f"--max-steps {configuration.get('max_steps', SCREENING_STEPS)}",
        f"--seed {seed if seed is not None else SCREENING_SEED}",
        *SHARED_FLAGS,
    ]
    # Keep indices 0-109 byte-identical: the trainer's default is the historical
    # placement, so only the appended experimental cells need an explicit flag.
    position = configuration.get("slot_position", "after_datapoint")
    if position != "after_datapoint":
        flags.append(f"--slot-position {position}")
    # Same reasoning for the scope: both paths is the trainer's default, so
    # only the appended single-scope cells name it.
    slot_scope = configuration.get("slot_scope", "cell_and_data")
    if slot_scope != "cell_and_data":
        flags.append(f"--table-slot-scope {slot_scope}")
    # Same reasoning again: the learned gate on the labelled embedding is the
    # trainer's default, so only the two screened alternatives name it.
    routing_mode = configuration.get("query_routing_mode", "decoder")
    if routing_mode != "decoder":
        flags.append(f"--query-routing-mode {routing_mode}")
    # The dump fixes the episodes and everything about them: coherence 0, two
    # classes, twelve features, 30% contamination.  Any cell that departs from
    # that has to generate its own, or it trains on the dump's task while
    # reporting its own.  Keyed on the overrides themselves rather than on
    # coherence alone, which is what let the learnable cells through.
    overrides_the_dump = coherence != 0.0 or any(
        key in configuration
        for key in (
            "max_classes",
            "min_features",
            "max_features",
            "multiregime_contamination",
            "support_size",
        )
    )
    if overrides_the_dump:
        # This comes after SHARED_FLAGS and after the batch script's own
        # `--multiregime-dump`, and argparse keeps the last occurrence, so it
        # wins.  "none" rather than "" because the flags are word-split.
        flags.append("--multiregime-dump none")
    # Per-cell overrides come after SHARED_FLAGS so argparse's last-wins
    # behaviour applies them; SHARED_FLAGS fixes features and class count for
    # every earlier block and must keep doing so.
    for name in (
        "max_classes",
        "min_features",
        "max_features",
        "multiregime_contamination",
        "support_size",
        "tabarena_max_predictors",
        "micro_batch_size",
        "accumulate_gradients",
        "support_reconstruction_weight",
        "slot_mi_weight",
    ):
        if name in configuration:
            flags.append(f"--{name.replace('_', '-')} {configuration[name]:g}")
    if final:
        flags.append("--no-tensorboard")
    return " ".join(flags)


def _read_run(directory: Path) -> dict[str, Any] | None:
    selection_path = directory / "selection.json"
    config_path = directory / "config.json"
    if not selection_path.is_file() or not config_path.is_file():
        return None
    selection = json.loads(selection_path.read_text())
    config = json.loads(config_path.read_text())
    return {
        "run": directory.name,
        "prior_mode": config.get("prior_mode"),
        "num_slots": config.get("num_slots"),
        "model_kind": config.get("model_kind", "slot"),
        "slot_compatibility": config.get("slot_compatibility", "dot"),
        "slot_position": config.get("slot_position", "after_datapoint"),
        "regime_coherence": config.get("regime_coherence", 0.0),
        "seed": config.get("seed"),
        "multiregime_cross_entropy": selection.get("multiregime_cross_entropy"),
        "query_cross_entropy": selection.get("query_cross_entropy"),
        "gate_regime_auc": selection.get("gate_regime_auc"),
        "gate_entropy": selection.get("gate_entropy"),
    }


def summarize_sweep(root: Path) -> list[dict[str, Any]]:
    """Rank finished runs by held-out multiregime cross entropy, lower first.

    ``gate_regime_auc`` rides alongside rather than driving the ranking: it says
    whether a slot actually bound to the contaminated rows, which is the
    mechanism question, while the cross entropy is the performance question.
    """
    runs = [run for directory in sorted(root.iterdir()) if directory.is_dir() and (run := _read_run(directory))]
    return sorted(
        runs,
        key=lambda run: (
            run["multiregime_cross_entropy"] is None,
            run["multiregime_cross_entropy"] if run["multiregime_cross_entropy"] is not None else 0.0,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Print the flags for one array task.")
    group.add_argument("--label", type=int, help="Print the run label for one array task.")
    group.add_argument("--summarize", type=Path, help="Rank finished runs under this directory.")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> str:
    arguments = build_parser().parse_args(argv)
    if arguments.summarize is not None:
        return json.dumps(summarize_sweep(arguments.summarize), indent=2)
    if arguments.label is not None:
        configurations = screening_configurations()
        if not 0 <= arguments.label < len(configurations):
            raise IndexError(f"Array index {arguments.label} is outside 0..{len(configurations) - 1}.")
        return configuration_label(configurations[arguments.label])
    return configuration_flags(arguments.index, final=arguments.final, seed=arguments.seed)


if __name__ == "__main__":
    print(main())
