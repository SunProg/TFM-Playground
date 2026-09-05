import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import sample_scm_multiregime_episode
from tfmplayground.experiments.dump_multiregime_episodes import (
    DumpConfig,
    MultiregimeDumpLoader,
    _shard_episodes,
    dump_multiregime_episodes,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    PlainPretrainingConfig,
)
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    multiregime_probability as plain_multiregime_probability,
)
from tfmplayground.experiments.pretrain_slot_tabpfn import (
    PRIOR_MODES,
    SlotPretrainingConfig,
    _checkpoint,
    build_model,
    identifiable_support_rows,
    multiregime_probability,
    summarize_samples,
    support_binding_scores,
    validate_config,
)
from tfmplayground.experiments.slot_tabpfn_sweep import (
    CLOSURE_WEIGHTS,
    COHERENT_STEPS,
    COMPATIBILITY_MODES_SCREENED,
    CONTROL_COHERENCE,
    EXTENDED_SLOT_COUNTS,
    LEARNABLE_DESIGN,
    MULTIREGIME_SHARE,
    PRIOR_MODES_READABLE,
    COMPOSITING_ROUTING_MODES,
    QUERY_ROUTING_MODES_SCREENED,
    READABLE_DESIGN,
    READABLE_MICRO_BATCH,
    REGIME_COHERENCE,
    SCREENING_STEPS,
    SLOT_COUNTS,
    TABLE_SLOT_SCOPES_READABLE,
    TABLE_SLOT_SCOPES_SCREENED,
    configuration_flags,
    configuration_label,
    screening_configurations,
    summarize_sweep,
)

MAX_STEPS = 1000


def config(prior_mode: str, **kwargs) -> SlotPretrainingConfig:
    return SlotPretrainingConfig(prior_mode=prior_mode, max_steps=MAX_STEPS, device="cpu", **kwargs)


class CurriculumTests(unittest.TestCase):
    """The four arms, and the guarantee that three of them match the vanilla script."""

    def test_constant_modes(self):
        for step in (0, 1, MAX_STEPS // 2, MAX_STEPS):
            self.assertEqual(multiregime_probability(config("plain"), step), 0.0)
            self.assertEqual(multiregime_probability(config("multiregime"), step), 1.0)
            self.assertEqual(multiregime_probability(config("mixed"), step), 0.30)

    def test_mixed_share_is_configurable(self):
        self.assertAlmostEqual(multiregime_probability(config("mixed", multiregime_share=0.5), 7), 0.5)

    def test_curriculum_shape(self):
        curriculum = config("curriculum")
        self.assertEqual(multiregime_probability(curriculum, 0), 0.0)
        # Flat at zero through the first 10% of the budget.
        self.assertEqual(multiregime_probability(curriculum, int(0.10 * MAX_STEPS)), 0.0)
        # Linear ramp to 0.5 at 50%, then flat.
        self.assertAlmostEqual(multiregime_probability(curriculum, int(0.30 * MAX_STEPS)), 0.25, places=6)
        self.assertAlmostEqual(multiregime_probability(curriculum, int(0.50 * MAX_STEPS)), 0.5, places=6)
        self.assertAlmostEqual(multiregime_probability(curriculum, MAX_STEPS), 0.5, places=6)

    def test_shared_modes_match_the_plain_implementation_exactly(self):
        """This is what stops the two scripts drifting apart.

        `plain`, `multiregime` and `curriculum` must stay byte-identical to the
        vanilla pretraining script, because the sweep is compared against runs
        produced by it.  `mixed` is the only arm this module defines itself.
        """
        for prior_mode in ("plain", "multiregime", "curriculum"):
            plain = replace(PlainPretrainingConfig(), prior_mode=prior_mode, max_steps=MAX_STEPS)
            slot = config(prior_mode)
            for step in range(0, MAX_STEPS + 1, 37):
                with self.subTest(prior_mode=prior_mode, step=step):
                    self.assertAlmostEqual(
                        multiregime_probability(slot, step),
                        plain_multiregime_probability(plain, step),
                        places=12,
                    )

    def test_config_validation(self):
        validate_config(config("mixed"))
        with self.assertRaises(ValueError):
            validate_config(config("nonsense"))
        with self.assertRaises(ValueError):
            validate_config(config("mixed", multiregime_share=1.5))
        with self.assertRaises(ValueError):
            validate_config(replace(config("plain"), max_steps=0))
        for position in ("before_feature", "after_feature", "before_and_after_feature"):
            for model_kind in ("vanilla", "slot"):
                with self.assertRaises(ValueError):
                    validate_config(config("plain", model_kind=model_kind, slot_position=position))
            validate_config(config("plain", model_kind="slot_backbone", slot_position=position))

    def test_checkpoint_architecture_records_slot_position(self):
        configured = config("plain", model_kind="slot_backbone", slot_position="before_and_after_feature")
        model = build_model(configured)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        checkpoint = _checkpoint(model, optimizer, scheduler, configured, 0, None, np.random.default_rng(1))
        self.assertEqual(checkpoint["architecture"]["slot_position"], "before_and_after_feature")


class SweepTests(unittest.TestCase):
    def test_grid_layout_is_append_only_and_labels_are_unique(self):
        """Extending the grid must never re-map an index a live array is using.

        Each block is appended after the last, so the indices submitted for an
        earlier block keep meaning the same thing.  These are the exact
        index/label pairs jobs 36853727, 36854390, 36858884 and 36861990 were
        submitted against.
        """
        configurations = screening_configurations()
        labels = [configuration_label(c) for c in configurations]
        self.assertEqual(len(set(labels)), len(labels))

        # Blocks appended after the table-slot ones are located from the end,
        # so every later append shifts them.  Rebasing onto the slice that
        # precedes the newest block keeps that arithmetic correct without
        # renumbering four offsets each time.
        # The compositing block and its matched baseline, appended newest.
        compositing_cells = 2 * len(COMPOSITING_ROUTING_MODES) * len(PRIOR_MODES_READABLE)
        before_compositing = configurations[:-compositing_cells]

        routing_cells = len(QUERY_ROUTING_MODES_SCREENED) * len(TABLE_SLOT_SCOPES_READABLE) * len(PRIOR_MODES_READABLE)
        before_routing = before_compositing[:-routing_cells]

        closure_cells = len(CLOSURE_WEIGHTS) * len(PRIOR_MODES_READABLE)
        before_closure = before_routing[:-closure_cells]

        slot_cells = len(PRIOR_MODES) * len(SLOT_COUNTS)
        self.assertTrue(all(c["model_kind"] == "slot" for c in configurations[:slot_cells]))
        # The K=2 slot arms keep the bare prior-mode labels they were run under.
        self.assertEqual(labels[: len(PRIOR_MODES)], list(PRIOR_MODES))

        vanilla = configurations[slot_cells : slot_cells + len(PRIOR_MODES)]
        self.assertTrue(all(c["model_kind"] == "vanilla" for c in vanilla))
        self.assertEqual(
            labels[slot_cells : slot_cells + len(PRIOR_MODES)],
            [f"{prior_mode}-vanilla" for prior_mode in PRIOR_MODES],
        )

        backbone_start = slot_cells + len(PRIOR_MODES)
        all_counts = SLOT_COUNTS + EXTENDED_SLOT_COUNTS
        coherent_start = backbone_start + len(PRIOR_MODES) * len(all_counts)
        backbone = configurations[backbone_start:coherent_start]
        self.assertEqual(len(backbone), len(PRIOR_MODES) * len(all_counts))
        # Its slot counts appear in order, each as a contiguous block of priors.
        self.assertEqual(
            [c["num_slots"] for c in backbone],
            [k for k in all_counts for _ in PRIOR_MODES],
        )
        self.assertTrue(all(c["model_kind"] == "slot_backbone" for c in backbone))
        # Its K=2 block likewise keeps the labels it was submitted under.
        self.assertEqual(
            labels[backbone_start : backbone_start + len(PRIOR_MODES)],
            [f"{prior_mode}-slot_backbone" for prior_mode in PRIOR_MODES],
        )
        self.assertTrue(all(c["num_slots"] == 2 for c in backbone[: len(PRIOR_MODES)]))

        # Everything above runs at coherence 0, and its labels carry no suffix.
        self.assertTrue(all(c.get("regime_coherence", 0.0) == 0.0 for c in configurations[:coherent_start]))
        self.assertTrue(all("coh" not in label for label in labels[:coherent_start]))

        # The coherent-regime block, appended last.  It changes the *task*, not
        # the model, so it repeats three model cells already in the grid; the
        # label suffix is what keeps those repeats from colliding with the
        # coherence-0 runs' directories, where a resume would silently continue
        # training on the other task.
        compatibility_start = coherent_start + 3 * len(PRIOR_MODES)
        coherent = configurations[coherent_start:compatibility_start]
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in coherent))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in coherent))
        self.assertEqual(
            [(c["model_kind"], c["num_slots"]) for c in coherent],
            [pair for pair in (("vanilla", 2), ("slot_backbone", 2), ("slot_backbone", 3)) for _ in PRIOR_MODES],
        )
        self.assertTrue(
            all(label.endswith(f"-coh{REGIME_COHERENCE:g}") for label in labels[coherent_start:compatibility_start])
        )

        # The compatibility block, appended last.  It changes what a slot's
        # claim on a row is *scored by*, holding the coherent task fixed, and
        # shares every other setting with the dot-product cells at 48-55 -- so
        # the label suffix is again what stops it overwriting them.
        mixture_start = compatibility_start + len(COMPATIBILITY_MODES_SCREENED) * 2 * len(PRIOR_MODES)
        compatibility = configurations[compatibility_start:mixture_start]
        self.assertEqual(
            [(c["slot_compatibility"], c["num_slots"]) for c in compatibility],
            [(mode, k) for mode in COMPATIBILITY_MODES_SCREENED for k in (2, 3) for _ in PRIOR_MODES],
        )
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in compatibility))
        self.assertTrue(all(c["model_kind"] == "slot_backbone" for c in compatibility))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in compatibility))
        for label, c in zip(labels[compatibility_start:mixture_start], compatibility, strict=True):
            self.assertTrue(label.endswith(f"-coh{REGIME_COHERENCE:g}-{c['slot_compatibility']}"))
        # Everything before it scores by dot product.
        self.assertTrue(all(c.get("slot_compatibility", "dot") == "dot" for c in configurations[:compatibility_start]))

        # The mixture-readout block, appended last.  Everything before it trains
        # on one cross entropy over the finished representation, so the loss
        # never mentions slots; these decode per slot and train on the mixture
        # NLL.  Their labels always write K out, unlike the older blocks.
        learnable_start = mixture_start + 2 * 2 * len(PRIOR_MODES)
        mixture = configurations[mixture_start:learnable_start]
        self.assertEqual(
            [(c["slot_compatibility"], c["num_slots"]) for c in mixture],
            [(mode, k) for mode in ("dot", "likelihood") for k in (2, 3) for _ in PRIOR_MODES],
        )
        self.assertTrue(all(c["model_kind"] == "slot_backbone_mixture" for c in mixture))
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in mixture))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in mixture))
        for label, c in zip(labels[mixture_start:learnable_start], mixture, strict=True):
            self.assertIn(f"-slot_mixture-k{c['num_slots']}-", label + "-")
        self.assertTrue(all(c["model_kind"] != "slot_backbone_mixture" for c in configurations[:mixture_start]))

        # The learnable-design block, appended last.  Every block before it
        # trains on a task whose achievable detection AUC is 0.505 -- chance --
        # so a model number on it cannot be read at all.
        learnable = [
            c
            for c in configurations
            if c.get("max_classes") == LEARNABLE_DESIGN["max_classes"]
            and c.get("regime_coherence", 0.0) != CONTROL_COHERENCE
            # The readable-design block shares the class count but not the
            # support size, features or contamination; it is its own block.
            and "support_size" not in c
            # The compositing block reuses this design deliberately -- it is
            # the only one whose numbers can be read -- but it is a separate
            # block, asserted on its own below.
            and not c.get("support_reconstruction_weight")
        ]
        self.assertEqual(len(learnable), 4 * len(PRIOR_MODES))
        self.assertEqual(
            [c["model_kind"] for c in learnable],
            [kind for kind in ("vanilla", "slot", "slot_backbone", "slot_backbone_mixture") for _ in PRIOR_MODES],
        )
        # Every model kind the trainer supports is exercised on the one design
        # whose results can be read; leaving one out is how the mixture backbone
        # was missed on the first pass.
        self.assertEqual(
            {c["model_kind"] for c in learnable},
            {"vanilla", "slot", "slot_backbone", "slot_backbone_mixture"},
        )
        self.assertTrue(all(all(c[k] == v for k, v in LEARNABLE_DESIGN.items()) for c in learnable))
        before_table = before_closure[:-66]
        self.assertEqual(before_table[len(before_table) - len(learnable) - 10 : -10], learnable)

        # The positive control, appended last.  It asks only whether the
        # competition can group rows when membership is element-wise, so a null
        # elsewhere cannot be blamed on a broken implementation.  Vanilla is
        # absent by design: it has no slots, so no binding to score.
        control = [
            c
            for c in configurations
            if c.get("regime_coherence", 0.0) == CONTROL_COHERENCE
            and c.get("slot_position", "after_datapoint") == "after_datapoint"
        ]
        self.assertEqual(len(control), 6)
        self.assertEqual(before_table[-10:-4], control)
        self.assertNotIn("vanilla", {c["model_kind"] for c in control})
        self.assertTrue(all(all(c[k] == v for k, v in LEARNABLE_DESIGN.items()) for c in control))
        self.assertTrue(all(f"-coh{CONTROL_COHERENCE:g}-" in configuration_label(c) for c in control))

        after_feature = before_table[-4:]
        self.assertEqual(
            [(c["model_kind"], c["prior_mode"]) for c in after_feature],
            [
                ("slot_backbone", "multiregime"),
                ("slot_backbone", "mixed"),
                ("slot_backbone_mixture", "multiregime"),
                ("slot_backbone_mixture", "mixed"),
            ],
        )
        self.assertTrue(all(c["slot_position"] == "after_feature" for c in after_feature))
        self.assertTrue(all(c["num_slots"] == 2 and c["max_steps"] == COHERENT_STEPS for c in after_feature))
        self.assertTrue(all(all(c[k] == v for k, v in LEARNABLE_DESIGN.items()) for c in after_feature))
        self.assertTrue(all(configuration_label(c).endswith("-after-feature") for c in after_feature))

        table = before_closure[-66:-54]
        self.assertEqual(
            [(c["model_kind"], c["prior_mode"]) for c in table],
            [
                (kind, mode)
                for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
                for mode in PRIOR_MODES
            ],
        )
        self.assertTrue(all(c["num_slots"] == 4 for c in table))
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in table))
        # The both-paths cells predate the scope axis and must keep running
        # under labels that carry no scope suffix, or a resubmission of
        # 114-125 would land in a fresh directory.
        self.assertTrue(all("slot_scope" not in c for c in table))

        # The scope ablation, appended last: one cell per scope, placement and
        # prior, holding the task and slot count fixed.
        scoped = before_closure[-54:-30]
        self.assertEqual(
            [(c["slot_scope"], c["model_kind"], c["prior_mode"]) for c in scoped],
            [
                (scope, kind, mode)
                for scope in TABLE_SLOT_SCOPES_SCREENED
                for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
                for mode in PRIOR_MODES
            ],
        )
        self.assertNotIn("cell_and_data", TABLE_SLOT_SCOPES_SCREENED)
        self.assertTrue(all(c["num_slots"] == 4 for c in scoped))
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in scoped))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in scoped))
        # Each scoped cell shares every other setting with a both-paths cell,
        # so only the suffix keeps it out of that cell's run directory.
        for scoped_cell in scoped:
            twin = {k: v for k, v in scoped_cell.items() if k != "slot_scope"}
            self.assertIn(twin, table)
            self.assertEqual(
                configuration_label(scoped_cell),
                f"{configuration_label(twin)}-{scoped_cell['slot_scope']}",
            )

        # The readable-design block, appended last: the same three scopes and
        # three placements on the task the measured detection ceiling leaves
        # headroom on, plus its own vanilla control.
        readable = before_closure[-30:]
        self.assertEqual(len(readable), 3 * 3 * 3 + 3)
        slot_cells, vanilla_cells = readable[:-3], readable[-3:]
        self.assertEqual(
            [(c["slot_scope"], c["model_kind"], c["prior_mode"]) for c in slot_cells],
            [
                (scope, kind, mode)
                for scope in TABLE_SLOT_SCOPES_READABLE
                for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa")
                for mode in PRIOR_MODES_READABLE
            ],
        )
        self.assertTrue(all(c["model_kind"] == "vanilla" for c in vanilla_cells))
        # The collapsed prior is excluded by design, and every cell carries the
        # whole design rather than part of it.
        self.assertNotIn("multiregime", {c["prior_mode"] for c in readable})
        # Every design key holds except the micro-batch pair, which the
        # placements set individually to fit a 24 GiB card.  What must hold
        # there is the *effective* batch: the optimization has to be identical
        # across placements even when the memory footprint cannot be.
        batch_keys = {"micro_batch_size", "accumulate_gradients"}
        self.assertTrue(
            all(all(c[k] == v for k, v in READABLE_DESIGN.items() if k not in batch_keys) for c in readable)
        )
        effective = READABLE_DESIGN["micro_batch_size"] * READABLE_DESIGN["accumulate_gradients"]
        for cell in readable:
            self.assertEqual(cell["micro_batch_size"] * cell["accumulate_gradients"], effective)
            self.assertEqual(
                cell["micro_batch_size"],
                READABLE_MICRO_BATCH.get(cell["model_kind"], READABLE_DESIGN["micro_batch_size"]),
            )
        # A larger support set is a different task and must not share a run
        # directory with the 128-row cells it otherwise matches.
        for cell in readable:
            self.assertIn("-s512", configuration_label(cell))

        # The closure block, appended last.  It changes the *objective*: every
        # cell before it trains the head on the query mixture NLL alone, which
        # never mentions the support competition, so one slot taking every row
        # costs nothing.  Matched cell for cell against the both-paths head
        # arms so the objective is the only thing that varies.
        closure = before_routing[-closure_cells:]
        self.assertEqual(
            [(c["support_reconstruction_weight"], c["slot_mi_weight"], c["prior_mode"]) for c in closure],
            [(rec, mi, mode) for rec, mi in CLOSURE_WEIGHTS for mode in PRIOR_MODES_READABLE],
        )
        self.assertTrue(all(c["model_kind"] == "table_slot_head" for c in closure))
        self.assertTrue(all(c["num_slots"] == 4 for c in closure))
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in closure))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in closure))
        # The collapsed prior is excluded for the reason the readable block
        # excludes it: every run of it sat at chance.
        self.assertNotIn("multiregime", {c["prior_mode"] for c in closure})
        # Every earlier cell trains the unmodified objective, and none of them
        # may gain a suffix -- 114-125 and 150-179 were live when this landed.
        self.assertTrue(
            all(
                not c.get("support_reconstruction_weight") and not c.get("slot_mi_weight") for c in before_closure
            )
        )
        # Each closure cell shares every other setting with a both-paths head
        # cell, so only the suffix keeps it out of that cell's run directory.
        for cell in closure:
            twin = {k: v for k, v in cell.items() if k not in ("support_reconstruction_weight", "slot_mi_weight")}
            self.assertIn(twin, table)
            suffix = f"-rec{cell['support_reconstruction_weight']:g}"
            if cell["slot_mi_weight"]:
                suffix += f"-mi{cell['slot_mi_weight']:g}"
            self.assertEqual(configuration_label(cell), f"{configuration_label(twin)}{suffix}")

        # The query-routing block, appended last.  Every cell before it --
        # closure or not -- routes a query through the learned gate on the
        # *labelled*-pass embedding; these fix the closure weights at the
        # passing configuration and vary only the routing mechanism, crossed
        # with every scope since a fix's value could depend on which
        # competition produced the slots it routes against.
        routing = before_compositing[-routing_cells:]
        self.assertEqual(
            [(c["query_routing_mode"], c["slot_scope"], c["prior_mode"]) for c in routing],
            [
                (mode, scope, prior)
                for mode in QUERY_ROUTING_MODES_SCREENED
                for scope in TABLE_SLOT_SCOPES_READABLE
                for prior in PRIOR_MODES_READABLE
            ],
        )
        self.assertTrue(all(c["model_kind"] == "table_slot_head" for c in routing))
        self.assertTrue(all(c["num_slots"] == 4 for c in routing))
        self.assertTrue(all(c["regime_coherence"] == REGIME_COHERENCE for c in routing))
        self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in routing))
        # Fixed at the passing closure configuration (183-185), not re-swept.
        self.assertTrue(
            all(
                c["support_reconstruction_weight"] == CLOSURE_WEIGHTS[1][0]
                and c["slot_mi_weight"] == CLOSURE_WEIGHTS[1][1]
                for c in routing
            )
        )
        self.assertNotIn("multiregime", {c["prior_mode"] for c in routing})
        # No earlier cell -- including the closure block itself -- may carry
        # this axis, or a resubmission of 180-185 would land somewhere new.
        self.assertTrue(all(c.get("query_routing_mode", "decoder") == "decoder" for c in before_routing))
        # In-training TabArena widens to the fifteen-task slice here, unlike
        # the closure block it is otherwise matched to -- a routing fix's
        # payoff is exactly the real-table query number, and there is no cost
        # to widening it before these cells start.
        self.assertTrue(all(c.get("tabarena_max_predictors") == 30 for c in routing))
        # The closure block specifically -- not every earlier cell, since the
        # readable-design block also carries this key at 30 for its own
        # unrelated reason -- must not have gained it retroactively.
        self.assertTrue(all(c.get("tabarena_max_predictors") is None for c in closure))
        # Each routing cell shares every other setting -- including scope --
        # with some non-routing cell, so only the routing-mode suffix and the
        # TabArena breadth above keep it out of that cell's run directory. At
        # the default scope that twin is a closure cell directly; at cell/data
        # scope it is not (the closure block never ran those), so only the
        # suffix's well-formedness is checked there.
        for cell in routing:
            twin = {k: v for k, v in cell.items() if k not in ("query_routing_mode", "tabarena_max_predictors")}
            if cell["slot_scope"] == "cell_and_data":
                self.assertIn({k: v for k, v in twin.items() if k != "slot_scope"}, closure)
            self.assertEqual(
                configuration_label(cell), f"{configuration_label(twin)}-{cell['query_routing_mode']}"
            )

        # The compositing block and its matched baseline, appended newest.
        # Every cell before these weights the support reconstruction by
        # `a[i,k]` and discards the decoder's alpha -- the same alpha the query
        # side gates on -- so the model runs two routings and trains one.
        # `reconstruction_mixture="alpha"` composites the way Locatello does,
        # which makes `L_rec` and the query mixture one expression.
        compositing = configurations[-compositing_cells:]
        alpha, baseline = compositing[: compositing_cells // 2], compositing[compositing_cells // 2 :]
        self.assertTrue(all(c["reconstruction_mixture"] == "alpha" for c in alpha))
        self.assertTrue(all("reconstruction_mixture" not in c for c in baseline))
        # No earlier cell may carry the axis, or a resubmission of the closure
        # or routing blocks would land somewhere new.
        self.assertTrue(all("reconstruction_mixture" not in c for c in before_compositing))
        for half in (alpha, baseline):
            self.assertEqual(
                [(c["query_routing_mode"], c["prior_mode"]) for c in half],
                [(mode, prior) for mode in COMPOSITING_ROUTING_MODES for prior in PRIOR_MODES_READABLE],
            )
            self.assertTrue(all(c["model_kind"] == "table_slot_head" for c in half))
            self.assertTrue(all(c["num_slots"] == 4 for c in half))
            self.assertTrue(all(c["max_steps"] == COHERENT_STEPS for c in half))
            # Fixed at the passing closure configuration, not re-swept: this
            # block varies the compositing rule, not the support-side weights.
            self.assertTrue(
                all(
                    c["support_reconstruction_weight"] == CLOSURE_WEIGHTS[1][0]
                    and c["slot_mi_weight"] == CLOSURE_WEIGHTS[1][1]
                    for c in half
                )
            )
            # On the learnable design, unlike every block above.
            # `detection_ceiling` puts the coherence-2.0 design at ~0.505
            # achievable, so a flat number there cannot be read at all.
            self.assertTrue(all(all(c[k] == v for k, v in LEARNABLE_DESIGN.items()) for c in half))
            self.assertNotIn("multiregime", {c["prior_mode"] for c in half})
        # `blind_similarity` is deliberately absent: it replaces the learned
        # alpha with a cosine similarity, which is the one property the shared
        # decoder exists to preserve, so it cannot be the arm a compositing fix
        # is screened with.
        self.assertNotIn("blind_similarity", {c["query_routing_mode"] for c in compositing})
        # Each alpha cell differs from its baseline twin in exactly the
        # compositing rule, so only the suffix keeps it out of that directory.
        for cell, twin in zip(alpha, baseline, strict=True):
            self.assertEqual({k: v for k, v in cell.items() if k != "reconstruction_mixture"}, twin)
            self.assertEqual(configuration_label(cell), f"{configuration_label(twin)}-alpha")

    def test_flags_carry_the_arm_and_hold_everything_else_fixed(self):
        configurations = screening_configurations()
        flag_sets = [configuration_flags(index) for index in range(len(configurations))]
        for configuration, flags in zip(configurations, flag_sets, strict=True):
            coherence = configuration.get("regime_coherence", 0.0)
            self.assertIn(f"--prior-mode {configuration['prior_mode']}", flags)
            self.assertIn(f"--num-slots {configuration['num_slots']}", flags)
            self.assertIn(f"--model-kind {configuration['model_kind']}", flags)
            self.assertIn(f"--max-steps {configuration.get('max_steps', SCREENING_STEPS)}", flags)
            self.assertIn(f"--regime-coherence {coherence:g}", flags)
            self.assertIn(f"--slot-compatibility {configuration.get('slot_compatibility', 'dot')}", flags)
            self.assertIn(f"--multiregime-share {MULTIREGIME_SHARE:g}", flags)
            self.assertEqual(
                "--slot-position after_feature" in flags,
                configuration.get("slot_position", "after_datapoint") == "after_feature",
            )
            # Naming the default would rewrite the flags of every cell that
            # predates the scope axis, including the twelve now running.
            scope = configuration.get("slot_scope", "cell_and_data")
            self.assertEqual("--table-slot-scope" in flags, scope != "cell_and_data")
            # The readable design's overrides must all reach the trainer, or a
            # cell trains on the old task while reporting the new one.
            for name, flag in (
                ("support_size", "--support-size"),
                ("tabarena_max_predictors", "--tabarena-max-predictors"),
            ):
                if name not in configuration:
                    continue
                words = flags.split()
                last = len(words) - 1 - words[::-1].index(flag)
                self.assertEqual(words[last + 1], f"{configuration[name]:g}")
            if scope != "cell_and_data":
                self.assertIn(f"--table-slot-scope {scope}", flags)
            # A coherent cell must not stream the dump, which holds coherence-0
            # episodes; a coherence-0 cell must leave the batch script's dump
            # flag alone.  Getting this backwards trains on one task and reports
            # the other, silently.
            # The dump fixes coherence, class count, features and
            # contamination together.  Any cell that departs from any of them
            # must generate its own episodes, or it trains on the dump's task
            # while reporting its own -- which coherence-keyed logic missed for
            # the learnable cells, since those run at coherence 0.
            departs = coherence != 0.0 or any(
                key in configuration
                for key in ("max_classes", "min_features", "max_features", "multiregime_contamination")
            )
            self.assertEqual("--multiregime-dump none" in flags, departs)
            # Per-cell overrides must come *after* SHARED_FLAGS, since argparse
            # keeps the last occurrence and SHARED_FLAGS pins the defaults.
            if "max_classes" in configuration:
                shared = flags.index("--max-classes")
                self.assertGreater(flags.rindex(f"--max-classes {configuration['max_classes']}"), shared)
                # TabArena stays on: the prior mixes 2- and 3-class episodes
                # at max_classes=3, so the model learns to use its first two
                # outputs for binary tables, which is what TabArena reads.
                self.assertNotIn("--no-tabarena-every-epoch", flags)
                self.assertIn("--tabarena-every-epoch", flags)

        # Prior mode, slot count and model kind are the only axes *within* one
        # block: strip all three and the remaining flags must be byte identical
        # across that block, so the grid isolates them and nothing else drifts.
        # Coherence and compatibility are the axes *between* blocks, and each
        # legitimately carries its own step budget and dump override.
        def block(configuration):
            # Class count joins the key: the learnable block overrides features,
            # contamination and TabArena along with it, and those overrides are
            # the point rather than drift.
            return (
                configuration.get("regime_coherence", 0.0),
                configuration.get("slot_compatibility", "dot"),
                configuration.get("max_classes", 2),
                configuration.get("slot_position", "after_datapoint"),
                # Scope is a between-block axis for the same reason
                # compatibility is: it names a different model, not a different
                # arm of one.
                configuration.get("slot_scope", "cell_and_data"),
                # The objective is a between-block axis too: a closure cell
                # matches the baseline in every flag but these two.
                configuration.get("support_reconstruction_weight", 0.0),
                configuration.get("slot_mi_weight", 0.0),
                # And the routing mode is a between-block axis for the same
                # reason: a routing cell matches its closure twin in every
                # flag but this one.
                configuration.get("query_routing_mode", "decoder"),
                # And the compositing rule, for the same reason again: a
                # compositing cell matches its baseline twin in every flag but
                # this one.
                configuration.get("reconstruction_mixture", "attention"),
            )

        def without_axes(flags: str, configuration: dict) -> str:
            """Flags with the per-arm axes removed, for byte comparison.

            The micro-batch pair is an axis in the same sense the model kind
            is: the readable-design placements set it to whatever fits a 24 GiB
            card, and the *effective* batch they multiply out to is asserted
            constant separately.  Both the shared default and any per-cell
            override are dropped, or a block would read as drifting when only
            the memory footprint differs.
            """
            text = (
                flags.replace(f"--prior-mode {configuration['prior_mode']} ", "")
                .replace(f"--num-slots {configuration['num_slots']} ", "")
                .replace(f"--model-kind {configuration['model_kind']} ", "")
            )
            for flag in ("--micro-batch-size", "--accumulate-gradients"):
                text = re.sub(rf"{flag} \d+", "", text)
            return " ".join(text.split())

        for key in {block(c) for c in configurations}:
            stripped = {
                without_axes(flags, c) for c, flags in zip(configurations, flag_sets, strict=True) if block(c) == key
            }
            self.assertEqual(len(stripped), 1, msg=f"flags drift inside block {key}")

        # Appending the new option must not rewrite command lines already used
        # by the active 0-109 controls.
        self.assertTrue(all("--slot-position" not in flags for flags in flag_sets[:110]))
        # Likewise for the twelve both-paths table-slot cells, which were live
        # when the scope axis was appended.
        self.assertTrue(all("--table-slot-scope" not in flags for flags in flag_sets[:126]))
        # Likewise for every cell that predates the closure objective: 114-125
        # and 150-179 were live when it was appended.
        self.assertTrue(
            all(
                "--support-reconstruction-weight" not in flags and "--slot-mi-weight" not in flags
                for flags in flag_sets[:180]
            )
        )
        # Likewise for every cell that predates the query-routing block:
        # 0-185 (114-125 and 150-185 in particular) were live when it landed.
        self.assertTrue(all("--query-routing-mode" not in flags for flags in flag_sets[:186]))

    def test_seed_override_and_index_bounds(self):
        self.assertIn("--seed 99", configuration_flags(0, seed=99))
        for index in (-1, len(screening_configurations())):
            with self.assertRaises(IndexError):
                configuration_flags(index)

    def test_summarize_ranks_by_multiregime_cross_entropy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, prior_mode, value in (
                ("run-a", "plain", 0.90),
                ("run-b", "curriculum", 0.40),
                ("run-c", "mixed", 0.60),
            ):
                run = root / name
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps({"prior_mode": prior_mode, "num_slots": 2, "model_kind": "slot", "seed": 2402})
                )
                (run / "selection.json").write_text(
                    json.dumps(
                        {
                            "multiregime_cross_entropy": value,
                            "query_cross_entropy": 0.5,
                            "gate_regime_auc": 0.7,
                            "gate_entropy": 0.3,
                        }
                    )
                )
            (root / "unfinished").mkdir()
            ranked = summarize_sweep(root)
        self.assertEqual([run["prior_mode"] for run in ranked], ["curriculum", "mixed", "plain"])
        self.assertEqual(ranked[0]["gate_regime_auc"], 0.7)


class SupportBindingTests(unittest.TestCase):
    """The mechanism metric: did slot competition partition the context by regime?"""

    def test_perfect_binding_scores_one(self):
        attention = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        source = torch.tensor([[0, 0, 1, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_auc"], 1.0, places=6)
        self.assertAlmostEqual(scores["support_binding_purity"], 1.0, places=6)
        self.assertAlmostEqual(scores["support_attention_entropy"], 0.0, places=6)
        self.assertAlmostEqual(scores["support_regime_base_rate"], 0.5, places=6)

    def test_uniform_attention_is_chance_with_maximal_entropy(self):
        attention = torch.full((1, 6, 2), 0.5)
        source = torch.tensor([[0, 0, 0, 1, 1, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_auc"], 0.5, places=6)
        self.assertAlmostEqual(scores["support_attention_entropy"], 1.0, places=6)

    def test_scores_are_invariant_to_slot_order(self):
        """Slots are anonymous, so relabelling them must not move the metric."""
        attention = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]])
        source = torch.tensor([[0, 0, 1, 1]])
        original = support_binding_scores(attention, source)
        flipped = support_binding_scores(attention.flip(-1), source)
        self.assertEqual(original, flipped)

    def test_purity_is_read_against_the_base_rate(self):
        """One slot claiming everything already scores the majority fraction."""
        attention = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        source = torch.tensor([[0, 0, 0, 1]])
        scores = support_binding_scores(attention, source)
        self.assertAlmostEqual(scores["support_binding_purity"], 0.75, places=6)
        self.assertAlmostEqual(scores["support_regime_base_rate"], 0.75, places=6)

    def test_missing_or_single_valued_tags_yield_no_scores(self):
        attention = torch.full((1, 4, 2), 0.5)
        self.assertEqual(support_binding_scores(attention, None), {})
        self.assertEqual(support_binding_scores(attention, torch.zeros(1, 4, dtype=torch.long)), {})

    def test_sampler_actually_populates_the_support_tag(self):
        """The tag used to be computed and discarded; it must now survive."""
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(3),
            family="mlp_scm",
            batch_size=1,
            support_size=32,
            query_count=8,
            contamination=0.4,
            device="cpu",
        )
        self.assertIsNotNone(episode.support_regime_source)
        self.assertEqual(episode.support_regime_source.shape, episode.support_y.shape)
        self.assertEqual(set(episode.support_regime_source.reshape(-1).tolist()), {0, 1})
        # It must survive a device move alongside the query-side tag.
        moved = episode.to("cpu")
        self.assertIsNotNone(moved.support_regime_source)


class MultiregimeDumpTests(unittest.TestCase):
    """The dump must reproduce what on-the-fly generation would have given."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        dump_multiregime_episodes(
            DumpConfig(
                output=str(cls.root / "shard-000.h5"),
                episodes=12,
                batch_size=4,
                support_size=16,
                query_count=4,
                contamination=0.25,
                seed=7,
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _loader(self, **kwargs):
        return MultiregimeDumpLoader(self.root, batch_size=4, **kwargs)

    def test_episode_shapes_and_split(self):
        episode = self._loader().sample()
        self.assertEqual(episode.support_x.shape[:2], (4, 16))
        self.assertEqual(episode.query_x.shape[:2], (4, 4))
        self.assertEqual(episode.support_y.shape, (4, 16))
        self.assertEqual(episode.query_y.shape, (4, 4))
        self.assertEqual(episode.support_x.shape[2], episode.query_x.shape[2])

    def test_dump_records_its_regime_coherence(self):
        """A dump fixes the episodes and with them the regime assignment, so a
        run configured for a different coherence must be able to tell.  Old
        dumps predate the field and were all written at coherence 0."""
        loader = self._loader()
        self.assertEqual(loader.regime_coherence, 0.0)
        self.assertEqual(loader.sample().metadata["regime_coherence"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump_multiregime_episodes(
                DumpConfig(
                    output=str(root / "shard-000.h5"),
                    episodes=4,
                    batch_size=4,
                    support_size=16,
                    query_count=4,
                    contamination=0.25,
                    regime_coherence=3.0,
                    seed=7,
                )
            )
            self.assertEqual(MultiregimeDumpLoader(root, batch_size=4).regime_coherence, 3.0)

    def test_dump_records_its_contamination(self):
        """Coherence was not the only thing a dump fixes; a run configured for a
        different contamination would otherwise stream the dump's silently."""
        self.assertEqual(self._loader().contamination, 0.25)

    def test_regime_tags_survive_the_round_trip(self):
        """Without these the slot-binding diagnostic cannot be computed at all."""
        episode = self._loader().sample()
        self.assertEqual(episode.support_regime_source.shape, episode.support_y.shape)
        self.assertEqual(episode.query_regime_source.shape, episode.query_y.shape)
        tags = torch.cat((episode.support_regime_source, episode.query_regime_source), dim=1)
        self.assertEqual(sorted(set(tags.reshape(-1).tolist())), [0, 1])
        # Contamination is the fraction of rows relabelled under the second regime.
        self.assertAlmostEqual(float(tags.float().mean()), 0.25, places=6)

    def test_candidates_and_posterior_are_preserved(self):
        episode = self._loader().sample()
        self.assertEqual(episode.candidate_support_positive.shape, (4, 2, 16))
        self.assertEqual(episode.candidate_query_positive.shape, (4, 2, 4))
        torch.testing.assert_close(episode.posterior.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_dump_is_usable_as_a_training_batch(self):
        episode = self._loader().sample()
        self.assertEqual(episode.condition, "multiregime")
        self.assertTrue(torch.isfinite(episode.support_x).all())
        self.assertEqual(set(episode.query_y.reshape(-1).tolist()) - {0, 1}, set())

    def test_exhausting_the_shard_cycles_rather_than_stopping(self):
        loader = self._loader()
        for _ in range(6):  # 12 episodes at batch 4 -> wraps partway through
            self.assertEqual(loader.sample().support_x.shape[0], 4)

    def test_shards_split_the_work_without_overlap(self):
        counts = [_shard_episodes(DumpConfig(output="", episodes=10, shard_index=i, num_shards=3)) for i in range(3)]
        self.assertEqual(sum(counts), 10)
        self.assertEqual(counts, [4, 3, 3])

    def test_invalid_shard_configuration_raises(self):
        with self.assertRaises(ValueError):
            dump_multiregime_episodes(DumpConfig(output="", episodes=1, shard_index=3, num_shards=3))


class SummarySampleTests(unittest.TestCase):
    """Every logged metric carries dispersion, not just a mean."""

    def test_reports_mean_spread_and_interval(self):
        summary = summarize_samples("metric", [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(summary["metric"], 2.5)
        self.assertEqual(summary["metric_n"], 4)
        self.assertAlmostEqual(summary["metric_std"], 1.2909944, places=6)
        self.assertAlmostEqual(summary["metric_stderr"], 0.6454972, places=6)
        # 95% normal approximation, the convention this repo's summaries use.
        self.assertAlmostEqual(summary["metric_ci_low"], 2.5 - 1.96 * 0.6454972, places=6)
        self.assertAlmostEqual(summary["metric_ci_high"], 2.5 + 1.96 * 0.6454972, places=6)

    def test_single_sample_has_no_interval(self):
        summary = summarize_samples("metric", [0.7])
        self.assertEqual(summary, {"metric": 0.7, "metric_n": 1})

    def test_zero_variance_collapses_the_interval(self):
        """Purity pinned to the base rate is a real signal, not missing data."""
        summary = summarize_samples("metric", [0.6875] * 5)
        self.assertAlmostEqual(summary["metric_ci_low"], 0.6875)
        self.assertAlmostEqual(summary["metric_ci_high"], 0.6875)

    def test_empty_samples_yield_nothing(self):
        self.assertEqual(summarize_samples("metric", []), {})


class IdentifiabilityTests(unittest.TestCase):
    """Contaminated rows the two label functions agreed on cannot be detected."""

    def test_mask_marks_only_disagreeing_rows(self):
        candidates = torch.tensor([[[0.9, 0.8, 0.2, 0.1], [0.9, 0.2, 0.8, 0.1]]])
        mask = identifiable_support_rows(candidates)
        self.assertEqual(mask.reshape(-1).tolist(), [False, True, True, False])

    def test_missing_or_wrong_shaped_candidates_disable_the_mask(self):
        self.assertIsNone(identifiable_support_rows(None))
        self.assertIsNone(identifiable_support_rows(torch.zeros(1, 3, 4)))

    def test_unidentifiable_rows_are_excluded_from_scoring(self):
        # Rows 0,1 clean; rows 2,3 contaminated but only row 2 is identifiable.
        attention = torch.tensor([[[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]])
        source = torch.tensor([[0, 0, 1, 1]])
        candidates = torch.tensor([[[0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.1, 0.9]]])
        scored = support_binding_scores(attention, source, identifiable_support_rows(candidates))
        # Row 3 dropped, so the remaining three separate perfectly.
        self.assertAlmostEqual(scored["support_binding_auc"], 1.0, places=6)
        self.assertAlmostEqual(scored["support_binding_purity"], 1.0, places=6)
        self.assertAlmostEqual(scored["support_identifiable_fraction"], 0.75, places=6)

    def test_without_the_mask_the_same_case_scores_lower(self):
        """The unmasked metric charges for a row nothing could have detected."""
        attention = torch.tensor([[[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]])
        source = torch.tensor([[0, 0, 1, 1]])
        unmasked = support_binding_scores(attention, source)
        self.assertLess(unmasked["support_binding_auc"], 1.0)
        self.assertAlmostEqual(unmasked["support_identifiable_fraction"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
