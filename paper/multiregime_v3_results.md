# Multi-regime v3 pilot: consolidated results

Date: 2026-09-11. Consolidates everything gathered across two SLURM studies
on the KCL CREATE cluster; supersedes nothing, ties together
`multiregime_v3_tabarena15.md` and `multiregime_v3_zero_shot_generalization.md`.

## Design, in brief

Episode generator: TabICL `mix_scm` (family `original`) plus four v3-generated
families (`shared_rule`, `soft_gate`, `independent`, `persistent`), described
in `multiregime_v3.py`/`paper/multiregime_v3.md`. Three training modes:
`original` (0% mixture, always), `fixed` (50% constant), `curriculum` (ramps
0%->50% over the first 40% of steps). Architecture: embedding=192, heads=6,
mlp=768, layers=6 (matches `pretrain_slot_tabpfn.py`'s `SlotPretrainingConfig`
defaults). 10,000 steps, effective batch 32 (micro-batch 8 x accumulate 4),
lr 1e-4 with 1e-6 cosine floor, 2,000-step warmup, AdamW.

Two studies exist:

- **`runs/37103332`** (completed): 6 cells, `plain`/`slot` x 3 modes. `slot`
  used `NanoTabPFNSlotRegimeModel` (slot_regime.py) and `num_groups=8` fixed
  (all episodes, including `original`, always padded with the same 8-wide
  nuisance one-hot block). This is the run TabArena-evaluated in
  `multiregime_v3_tabarena15.md`.
- **`runs/37113090`** (in progress, 9/21 cells done as of this note): 21
  cells, `plain` + six `table_slot` variants (`TableSlotModel`, replacing
  `NanoTabPFNSlotRegimeModel` -- see "table_slot vs slot_regime" below) x 3
  modes. `num_groups=5` fixed width, but the *realized* group count is
  sampled uniformly 1-5 per episode (`active_groups`, decoupled from the
  fixed one-hot width so batches stay a consistent feature size -- see "group
  count" below). `original`-mode training episodes carry **no** nuisance
  code block at all (see "original-mode padding fix" below); the shared
  evaluation bank (used by every cell, every study) still pads `original`
  episodes, for comparability across cells.

These two studies are not directly comparable cell-for-cell: different slot
model, different `num_groups` scheme, different `original`-mode padding.
Numbers from each are reported separately below.

## Design changes made along the way, and why

1. **Scaled to match `pretrain_slot_tabpfn.py`**: the pilot originally ran at
   smoke-test scale (1,000 steps, batch 4, embedding 96/hidden 384/layers
   3/heads 4). Rescaled to the harness's real settings (above) so the pilot
   trains at the same scale as the rest of the project's slot work.
2. **Sampled `num_groups`**: an oracle-only sweep (no model, just
   `oracle_probabilities`) measuring how much group evidence helps an oracle
   with perfect knowledge of the generator showed identifiability peaking at
   2-4 groups and declining by 8, while the one-hot nuisance block grows from
   14% to 40%+ of the input width over that range. Decoupled the fixed
   one-hot width (`num_groups`, now 5) from the realized per-episode group
   count (`active_groups`, sampled uniformly 1-5), so training now sees a
   spread of difficulty instead of a fixed, likely-suboptimal one.
3. **`table_slot` instead of `slot_regime`**: switched `build_model`'s "slot"
   kind from `NanoTabPFNSlotRegimeModel` to `TableSlotModel` (mode="head",
   `query_routing_mode="decoder"`, `reconstruction_mixture="attention"` --
   the `decoder_baseline` condition `scm_table_slot_head_sweep.py` already
   treats as the reference design elsewhere in the repo). Both return the
   same `SlotRegimePrediction` dataclass, so `log_predictions`/`evaluate`/
   `recovery_metrics` needed no changes. Extended from one slot condition to
   all six `table_slot` placements (21 cells total): the four `mode="head"`
   routing/reconstruction conditions, plus `mode="backbone"` and
   `mode="mufasa"`.
4. **Original-mode padding fix**: TabArena AUC for `runs/37103332`'s
   `plain-original` cell (0.6229) was far below a matched external baseline
   -- same TabICL `mix_scm` prior, same architecture, same steps, same batch
   size -- that scored 0.7298 (`paper/slot_sweep_vanilla_baseline_metrics.md`,
   the `plain` arm of `pretrain_slot_tabpfn.py`'s vanilla sweep). Traced to
   `OriginalPrior.sample` unconditionally padding every episode, including
   `original`-mode ones, with a nuisance code block that exists specifically
   to stop a model from telling "original" apart from the other four
   families by checking whether that block is empty. In pure `original`-mode
   training no other family is ever interleaved, so nothing can leak against
   -- the padding there was pure downside. Fixed: `OriginalPrior.sample`
   gained `pad_groups=True` (default), and `training_episode` now passes
   `pad_groups=(mode != "original")`. `fixed`/`curriculum` cells still pad
   `original`-family draws (real leak risk there). The shared evaluation
   bank is untouched, so cross-cell comparisons stay matched; this trades a
   small train/eval mismatch inside `original`-mode cells' own validation
   metrics for removing a much larger one against real TabArena data.
   Applied to `runs/37113090`; `runs/37103332` predates it.
5. **Bug fixes surfaced along the way**: `tfmplayground/interface.py`'s
   `init_model_from_state_dict_file` loaded checkpoints with the (newer
   torch) default `weights_only=True`, which fails on any checkpoint
   carrying more than a bare model (optimizer state, `torch.__version__`'s
   `TorchVersion` object) -- fixed to match its sibling loader's
   `weights_only=False`. `TableSlotModel`'s `backbone`/`mufasa` modes default
   `layer_indices=(3,4,5)`, out of range for backbones shallower than 6
   layers -- added `table_slot_layer_indices(config)` to scale with
   `config.layers`. Two sbatch scripts failed instantly on submission from a
   bash quirk: an apostrophe inside `${VAR:?message}` breaks bash's parser
   even within double quotes.

## `runs/37103332` (completed, 6 cells) -- synthetic validation

| cell | kind | mode | original | shared_rule | soft_gate | independent | persistent | group_info_gain |
|---|---|---|---|---|---|---|---|---|
| 0 | plain | original | 0.5160 | 0.6424 | 0.6442 | 0.6306 | 0.6715 | 0.0047 |
| 1 | slot | original | 0.5136 | 0.6435 | 0.6438 | 0.6269 | 0.6703 | 0.0038 |
| 2 | plain | fixed | 0.5107 | 0.5488 | 0.6018 | 0.5691 | 0.5682 | 0.0681 |
| 3 | slot | fixed | 0.5107 | 0.5550 | 0.6083 | 0.5598 | 0.5730 | 0.0853 |
| 4 | plain | curriculum | 0.5107 | 0.5546 | 0.6094 | 0.5665 | 0.5659 | 0.1287 |
| 5 | slot | curriculum | 0.5102 | 0.5614 | 0.6038 | 0.5586 | 0.5630 | 0.0955 |

(log_loss only; full brier/accuracy/ece_10 table was reported in-conversation
and is reproducible from `runs/37103332/cell-*/result.json`.)

## `runs/37103332` -- TabArena (tabarena15 protocol: max_predictors=30,
subsample=2048, 5-fold x 10-repeat CV, real labels, 15 eligible datasets)

| model | mean_roc_auc | mean_accuracy | mean_cross_entropy | mean_brier |
|---|---:|---:|---:|---:|
| random_forest (sklearn baseline) | 0.7761 | 0.8208 | 0.4578 | 0.1260 |
| logreg (sklearn baseline) | 0.7506 | 0.8060 | 0.4140 | 0.1323 |
| **plain-fixed** | **0.7305** | 0.7929 | 0.4396 | 0.1406 |
| slot-fixed | 0.7293 | 0.7940 | 0.4357 | 0.1395 |
| plain-curriculum | 0.7208 | 0.7912 | 0.4402 | 0.1411 |
| slot-curriculum | 0.7149 | 0.7908 | 0.4419 | 0.1420 |
| plain-original | 0.6229 | 0.7676 | 0.4885 | 0.1598 |
| slot-original | 0.6223 | 0.7678 | 0.4880 | 0.1595 |

All six lose to both sklearn baselines. `fixed`/`curriculum` cells generalize
far better to real tables than `original`-only cells (this `original` result
predates the padding fix above -- not directly comparable to a re-run under
the fix). Full per-dataset AUC breakdown (15 datasets) is in
`multiregime_v3_tabarena15.md`. TabPFN v2.2/v2.6/v3 baselines are still
outstanding -- that job depends on a `--tabpfn-model-path` CLI flag that
exists only in another session's uncommitted local edit to
`evaluate_tabarena_small.py`, not yet committed.

## `runs/37113090` (in progress, 9/21 cells complete as of this note)

Completed: `plain` and `table_slot_head_decoder_baseline`/
`table_slot_head_decoder_alpha`, each x 3 modes. Full brier/accuracy/ece_10
also recorded in each cell's `result.json`; log_loss shown here.

| cell | kind | mode | original | shared_rule | soft_gate | independent | persistent | group_info_gain |
|---|---|---|---|---|---|---|---|---|
| 0 | plain | original | 0.5149 | 0.6458 | 0.6456 | 0.6379 | 0.6690 | 0.0037 |
| 1 | plain | fixed | 0.5107 | 0.5488 | 0.6018 | 0.5691 | 0.5682 | 0.0681 |
| 2 | plain | curriculum | 0.5107 | 0.5546 | 0.6094 | 0.5665 | 0.5659 | 0.1287 |
| 3 | table_slot_head_decoder_baseline | original | 0.5084 | 0.5705 | 0.6087 | 0.5548 | 0.5888 | 0.0687 |
| 4 | table_slot_head_decoder_baseline | fixed | 0.5071 | 0.5570 | 0.5924 | 0.5566 | 0.5797 | 0.0625 |
| 5 | table_slot_head_decoder_baseline | curriculum | 0.5112 | 0.5575 | 0.5980 | 0.5569 | 0.5731 | 0.0823 |
| 6 | table_slot_head_decoder_alpha | original | 0.5084 | 0.5705 | 0.6087 | 0.5548 | 0.5888 | 0.0687 |
| 7 | table_slot_head_decoder_alpha | fixed | 0.5071 | 0.5570 | 0.5924 | 0.5566 | 0.5797 | 0.0625 |
| 8 | table_slot_head_decoder_alpha | curriculum | 0.5112 | 0.5575 | 0.5980 | 0.5569 | 0.5731 | 0.0823 |

Cells 6-8 are bit-identical to cells 3-5 (see "decoder_baseline/decoder_alpha
are twins" below) -- not independent data points.

Still running (as of this note): cells 9-12 (`table_slot_head_blind_decoder`,
all 3 modes, plus the start of `blind_similarity`). Queued: the remainder up
to cell 20 (`table_slot_mufasa`/curriculum).

### Finding: table_slot generalizes zero-shot to unseen families; plain doesn't

Under `original`-mode training, *neither* `plain` nor `table_slot` ever sees
`shared_rule`/`soft_gate`/`independent`/`persistent` during training. On
those held-out families:

| family | plain/original | table_slot/original | delta (log_loss) |
|---|---|---|---|
| shared_rule | 0.6458 | 0.5705 | table_slot −0.075 |
| soft_gate | 0.6456 | 0.6087 | table_slot −0.037 |
| independent | 0.6379 | 0.5548 | table_slot −0.083 |
| persistent | 0.6690 | 0.5888 | table_slot −0.080 |

table_slot is consistently better across all four, despite identical (zero)
training exposure. Under `fixed`-mode, where both models *do* train on all
five families, the gap closes to near-nothing and flips sign by family (plain
better on `shared_rule`/`persistent`, table_slot better on
`soft_gate`/`independent`, deltas all under 0.013) -- no consistent winner
once both get direct exposure. Reading: plain nanoTabPFN needs to see the
mixture during training to handle it; table_slot generalizes to it
zero-shot, closing most of the gap to what plain only reaches after direct
exposure. Single seed; effectively one independent table_slot data point so
far (see next finding). Not yet known whether this holds for the other four
`table_slot` variants.

### Finding: `decoder_baseline` and `decoder_alpha` are twins under this loss

`reconstruction_mixture` (the only setting distinguishing the two) only
affects the `reconstruction` field of the model's output (an auxiliary
support-reconstruction target); the pilot's training loss is a single
`F.nll_loss` call on query predictions, computed twice in
`pretrain_multiregime_v3.py` (lines ~417, ~537), and never touches that
field. Same seed, same data stream, same optimizer -> identical gradients
throughout training -> bit-identical checkpoints. Confirmed to 4 decimal
places across all 5 families, both cells. Will hold for the rest of the run;
not a bug.

### Finding: `slot_gate_entropy` is near-maximal everywhere so far

Every completed slot cell, every family: 0.9998-0.9999 (normalized entropy,
1.0 = fully uniform gate across all 4 slots, chance-level; 0.0 = confident
single-slot routing). The query gate is not confidently routing to any
specific slot in any completed cell, despite the zero-shot generalization
advantage above. Unreconciled: whatever table_slot is doing better than
plain here is not visible as slot specialization by this metric -- may come
from architecture capacity/pooling rather than genuine regime
discrimination, or the specialization may exist but not show up in gate
entropy specifically. Worth checking against `support_ari`/`query_ari`
(chance-adjusted partition recovery, also recorded per cell) once more
variants finish.

## Outstanding

- 12 of 21 `runs/37113090` cells not yet complete (`blind_decoder` finishing,
  `blind_similarity`/`backbone`/`mufasa` not started or mid-run).
- TabArena evaluation of `runs/37113090`'s cells: job `37118637`
  (auto-discovers finished `cell-*/checkpoint.pth`, no wrap step needed since
  these checkpoints carry `architecture` natively) submitted, still `PENDING`
  in the SLURM queue as of this note.
- TabPFN v2.2/v2.6/v3 baselines: blocked on an uncommitted `--tabpfn-model-path`
  flag in `evaluate_tabarena_small.py` (belongs to another session's
  in-progress local edit on this shared worktree). Not attempted via a
  self-contained workaround yet.
- Whether the zero-shot generalization finding and the near-uniform
  `slot_gate_entropy` finding hold for `blind_decoder`, `blind_similarity`,
  `backbone`, and `mufasa`.
