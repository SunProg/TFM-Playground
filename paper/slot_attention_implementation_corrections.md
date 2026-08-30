# Slot attention for nanoTabPFN: implementation corrections

## Status

- Date: 2026-08-30
- Branch: `exp/slot-attention`, commits `b0ef0b2` … `bafc2d6`
- Companion document: [`slot_attention_prior_composition_results.md`](slot_attention_prior_composition_results.md)
  records what the runs measured; this one records what was wrong with the
  implementation and how it was found

This is a record of defects, not of results. Several of them silently
invalidated experiments that had already run, so it is worth having the failure
modes written down separately from the numbers.

## Part 1 — pre-existing problems in the repository

### The `slot_attention` attribute was not slot attention

Four heads carried an attribute named `slot_attention` (or `evidence_attention`)
implementing the same copy-pasted block — `hypothesis.py:200-207`,
`coherent_correction.py:87-96`, `sequential_latent_filter.py:93-99`,
`task_posterior_adapter.py:122-129`:

```python
seeds = self.hypothesis_queries[None].expand(batch_size, -1, -1)
attended = self.slot_attention(seeds, support, support, need_weights=False)[0]
slots = self.slot_norm(seeds + attended)
```

That is ordinary cross-attention. It softmaxes over **evidence keys**, runs
**once**, has **no GRU**, and gives each slot its **own persistent learned
vector**. Slot Attention softmaxes over **slots** so they compete for inputs,
iterates that competition with a GRUCell, and draws all slots from a **shared**
`(mu, log_sigma)` so they are anonymous and exchangeable. Every one of those
four differences is load-bearing; none was present.

Replaced by a faithful implementation in `models/slot_attention.py`, with
`competitive=False` retained as a controlled ablation that changes only the
normalization axis.

### Slot supervision that the mechanism is meant to replace

`train_coherent_hypotheses.hypothesis_loss` pinned slot **index** 0 to "all
zeros" and index 1 to "all ones"; `train_bayesian_nanotabpfn.static_bayesian_loss`
scored Hungarian-matched slots against named candidate functions. Slot attention
is never told which slot owns which object — specialization is supposed to come
from competition plus a reconstruction bottleneck.

Both removed. Matching survives only in diagnostics, never in a gradient path.
The rule adopted: **matching in the metric, never in the loss.**

Measured with no slot supervision at all, on the coherent-hypotheses task:

| | loss ratio | slot disagreement | support-row split |
|---|---|---|---|
| competitive | 0.407 | 0.008 → **1.000** | 0.035 → **0.574** |
| non-competitive | 0.538 | 0.008 → 0.009 | 0.053 → 0.165 |

Competition alone produced full specialization; the non-competitive twin
collapsed. This also retires the loss-side anti-collapse patches that
`MEAN_PRESERVING_BAYESIAN_TRIAL.md` and `H5_ALLTERMS_MODEL.md` had already
recorded failing.

### `slot_prior_logits`

A learned prior indexed by slot position, in two heads. With slots drawn from a
shared distribution the index carries no stable meaning, so the parameter
described nothing. Removed.

### The support-side regime tag was discarded

`_build_multiregime_item` and `_build_scm_multiregime_item` both computed
`support_source` — which label function produced each support row's label — and
dropped it on the floor. Only the query-side tag reached `ContinuousEpisode`.
Slot binding happens on the *support* rows, so the central diagnostic could not
be computed at all. Now carried through as `support_regime_source`, diagnostic
only, never an input.

### Multiregime episode generation dominated the wall clock

The multiregime arm ran at 39 steps/min against the single-regime arm's 85.
Training compute is identical between them, so the gap is generation: ~0.83 s per
step building 8 SCM episodes on the CPU, about 11.5 h of a 21.5 h run against a
24 h limit. Pre-generating into an HDF5 dump costs 38 ms per episode once and
0.09 ms to read one back.

### Checkpoint growth exhausted the disk quota

`epoch_steps` defaulted to 500 and every epoch checkpoint was retained: 100 files
of ~55 MB per arm, ~22 GB for four arms, on a volume with a few GB free. The
first sweep died at ~2 h with `OSError: [Errno 122]`. Now one rolling file for
the in-training TabArena evaluation to read, plus durable snapshots every 10,000
steps.

## Part 2 — defects introduced while building this, then found

### The in-backbone slot layers were a no-op

The most consequential. Slot layers wrote back as
`target + tanh(row_gate) * reconstruction` with `row_gate` initialized to zero,
copying the convention from `BottleneckAdapter` and `IntegratedLatentAdapter`.
That convention exists so an adapter attached to *pretrained* weights begins as
an exact identity. It does not transfer to training from scratch: there is no
behaviour to preserve, and a zero-initialized additive residual is a side-path
the optimizer can ignore at no cost.

It did. From the completed arm of job `36861990`, after 10,000 steps:

```
layer  row_gate     tanh(row_gate)
  0    -0.000086    -0.000086
  1    +0.000137    +0.000137
  2    -0.000061    -0.000061
  3    +0.000445    +0.000445
  4    +0.000029    +0.000029
  5    +0.000121    +0.000121
mean |tanh(gate)| = 0.000146
```

The arm consequently reproduced plain nanoTabPFN to four decimals — `mr_ce`
0.6831 against 0.6830, `ord_ce` 0.7326 against 0.7310, TabArena 0.717 against
0.718. **Twelve runs measured vanilla with an inert attachment.** Read without
checking the gate, they would have supported a false conclusion that in-backbone
slots do not help.

Fixed by replacing the gated residual with a convex blend:

```python
mix = torch.sigmoid(self.slot_mix)          # starts at 0.5
updated = (1 - mix) * target + mix * reconstruction
```

Slots now carry half of every row state from the first step, so silencing them is
work the optimizer must choose to do. A short check moves the mix 0.0018 in 40
steps against 0.0001 in 10,000 before. A test asserts the output differs from
the plain backbone at initialization — the exact opposite of the test the old
design carried — and another asserts that driving the mix to zero still recovers
the plain backbone, so the escape hatch exists but is not free.

**Generalizable lesson:** a zero-initialized residual guarantees "starts as the
base model", which is a *safety* property for fine-tuning and an *inertness* risk
for from-scratch training. Whenever such a gate exists, its learned value is a
required diagnostic, not an implementation detail.

### The query gate was decoupled from the decoder

Query rows carry no label and cannot compete for slots, so they were routed by a
bilinear similarity `<W_q query, W_s slot>` with its own projections and no
parameters shared with the decoder. That was invention, not the reference
architecture: the vision decoder emits RGB **and** alpha from one pathway, tying
what a slot predicts to where it applies. Decoupled, routing can collapse
independently of what the decoder does. Now emitted as an extra decoder output
channel, softmaxed over slots.

### The binding metric was measured against an unreachable target

Both label functions are evaluated on the same feature matrix, so a row
relabelled to the value the base function would have produced anyway is tagged
contaminated while being observationally identical to a clean row. Measured over
19,200 rows:

| quantity | value |
|---|---|
| rows where the two label functions agree | 0.507 |
| rows tagged contaminated | 0.300 |
| contaminated rows that are indistinguishable | **0.500** |

With 30% positives of which half are undetectable, a perfect detector scores
`0.5 * 1.0 + 0.5 * 0.5 = 0.75`, and purity tops out near 0.85 against a 0.703
base rate. The reported binding numbers had been compared against 1.0. Scoring is
now restricted to rows where the two candidates actually disagree, restoring a
ceiling of 1.0, and `support_identifiable_fraction` records how much was kept
(0.834) so the exclusion is visible rather than silent.

### A metric was computed but never reported

`support_identifiable_fraction` was calculated inside `support_binding_scores`
but never reached the history row: the edit adding it to the summarisation list
silently failed to match after the file had been reformatted. Found only by
inspecting a run's output and noticing the key was absent.

## Part 3 — process failures

Recorded because they cost real time and GPU allocation.

- **Pushed twice with failing tests**, by running the suite and `git commit` in
  one command and not reading the result before it landed. Both were stale grid
  tests after adding an axis. Running the suite as its own step fixed the habit.
- **Broke a source file** with a string replacement anchored on text that
  appeared twice; it patched the wrong occurrence and produced a syntax error.
  Restored from git and redone with unique anchors.
- **Presented a comparison table sourced from my own earlier prose** rather than
  from the logs, after the run directories had been deleted. It later verified
  correct against the surviving logs, but the provenance should have been stated
  at the time.
- **Checked `df` rather than the user quota.** Filesystem free space was 11 P
  while the binding constraint was a per-user quota around 187 G, so the second
  sweep died the same way as the first.
- **Missed an existing job.** Job `36860987` had already re-run the vanilla arms
  to completion; a resubmission was nearly issued for work that was done.

## Recovery note

Every history row is printed to stdout as JSON, so
`slurm-slot-tabpfn-sweep-<job>_<task>.out` is a complete backup of
`history.jsonl` even after a run directory is deleted. Both the vanilla control
results and the verification of the slot K=2 numbers were recovered this way
after the run directories were cleaned up.
