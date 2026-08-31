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

#: Compatibility functions screened in the compatibility block.  "dot" is not
#: among them: indices 16-27 already ran it at these settings, so re-running it
#: would buy nothing the existing cells do not already report.
COMPATIBILITY_MODES_SCREENED = ("likelihood", "additive")

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
    return grid


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
    if kind == "vanilla":
        return f"{prior_mode}-{kind}{suffix}"
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
    if coherence != 0.0:
        # The dump was generated at coherence 0, so it holds the old task's
        # episodes; generate on the fly instead.  This comes after SHARED_FLAGS
        # and after the batch script's own `--multiregime-dump`, and argparse
        # keeps the last occurrence, so it wins.  "none" rather than "" because
        # the flags are word-split by the shell.
        flags.append("--multiregime-dump none")
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
