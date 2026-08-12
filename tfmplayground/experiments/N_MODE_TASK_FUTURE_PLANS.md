# N-mode synthetic task: future direction (Option B, not implemented)

## Context

`train_four_mode_particle_filter.py` was generalized from a fixed 4-mode task
to an arbitrary `num_modes` (power of 2), plus a `mode_curriculum` that
interleaves multiple `num_modes` values during a single training run
(implemented; see `FourModeConfig.mode_curriculum`, `train_four_mode`,
`evaluate_four_mode`).

For every `num_modes >= 2`, the task is built from `num_regions =
log2(num_modes)` evidence regions, each a fresh, never-before-seen cluster
(`_query_points`: region `r` centered at `2.0 + r`). `num_modes == 1` is
handled as a fully separate code path (`_evaluate_trivial_mode` /
the `num_modes == 1` branch in `generate_four_mode_episodes`): the query and
stream reuse the *same* sign-threshold rule as the prior/support, so there is
zero hidden state — the answer is already fully determined by context the
model has already seen. This is "Option A": a clean floor case, isolated from
the region-based family, that doesn't change what `num_modes=2/4/8` mean.

## Option B (rejected for now, worth revisiting)

Instead of a separate `num_modes=1` path, **alias region 0 to the prior's own
rule for every `num_modes`**, not just `num_modes=1`. Concretely:

- Region 0 stops being a fresh cluster at `x≈2.0`; it becomes "the same
  sign-threshold rule already shown in the prior support."
- Regions `1..R-1` remain fresh clusters as today (`x≈3.0, 4.0, ...`).
- `num_modes=1` becomes the true base case of the same family (0 *new*
  regions), and `num_modes=2` is exactly "`num_modes=1` plus one genuinely new
  region," `num_modes=4` is "plus two new regions," etc. — a clean, nested
  difficulty curve where each doubling of `num_modes` adds exactly one bit of
  *new* information, and the rest of the model's job (recognize the reused
  region) is factored out as a constant.

### Why this is more elegant

- Lets you meaningfully plot "performance vs. number of *new* regions" as a
  single monotonic axis, instead of comparing structurally different task
  families (M=1 vs M>=2).
- `num_modes=1` stops needing its own bespoke generator/loss/eval branch
  (`_evaluate_trivial_mode`, the `if num_modes == 1` branches in
  `generate_four_mode_episodes` and `four_mode_loss`) — one code path covers
  every `num_modes >= 1`.

### Why it wasn't done now

Redefining region 0 changes the task definition for **every** `num_modes`,
not just 1. That breaks direct comparability with results already collected
under the current definition:

- The existing `k8_baseline` / `k16_specialized` checkpoints
  (`runs/four_mode_particle_filter/20260808-k8-four-modes-baseline`,
  `.../20260808-k16-four-modes-specialized`) were trained with region 0 as a
  fresh cluster, and were evaluated against real TabArena datasets in this
  session (see `/tmp/particle_count_sweep.csv` at the time of writing, or the
  conversation history for the per-dataset table).
- `consistent_0`'s semantics change: today it means "both regions predict
  label 0"; under Option B it would mean "region 0 follows the known prior
  rule, region 1 predicts label 0" — a different scenario, not a relabeling.

If Option B is picked up later, plan to:
1. Re-run the full `k8_baseline`/`k16_specialized`-equivalent training and the
   real-dataset TabArena sweep under the new region-0 definition, so results
   are internally consistent again (don't try to compare old and new
   checkpoints directly).
2. Decide whether `consistent_0` should keep its name with new semantics, or
   be renamed (e.g. `consistent_known_0`) to avoid confusing old and new
   result tables.
3. Collapse `_evaluate_trivial_mode`/the `num_modes==1` special cases in
   `generate_four_mode_episodes` and `four_mode_loss` into the general
   region-based path once region 0 is aliased — they become dead code.
