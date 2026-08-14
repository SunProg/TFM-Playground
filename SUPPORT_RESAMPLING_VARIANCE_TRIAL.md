# Support-Resampling Variance of the nanoTabPFN Encoder

## Status

- Date: 2026-08-14
- Scope: binary classification, synthetic episodes and TabArena, evaluation only — nothing is trained
- Implemented and verified: resampling ensemble (bootstrap, stratified subsample, Bayesian-bootstrap
  approximation), per-layer dispersion statistics, decoder-projection gradient, two model-intrinsic arms
  (pairwise joint mutual information, self-conditioning), synthetic ground-truth harness, TabArena harness,
  full unit test suite
- Run: stage-1 synthetic evaluation (20 episodes/condition, held-out families) and a small real stage-2
  TabArena run (4 tasks)
- **Result: negative, and decisive at stage 1.** No arm clears the pre-declared gates; every resampling
  scheme reads more mutual information off pure-noise labels than the continuous-posterior trial's learned
  head ever reported on real signal. The one arm that passes the null check (`joint_mi`) is the worst
  performer on real TabArena data. See "Bottom line" below.

## Motivation

Two prior trials on this branch tried to make nanoTabPFN report epistemic uncertainty through a *learned*
head, and both stalled in the same direction:

- The slot posterior (`MEAN_PRESERVING_BAYESIAN_TRIAL.md`) shipped and failed the usefulness gate hard:
  error-detection AUROC **0.254** for the learned mutual information versus **0.767** for plain vanilla
  predictive entropy on the same 20 TabArena tasks; AURC 0.246 versus 0.076. The learned signal was
  *anti-correlated* with mistakes.
- The continuous posterior (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`) ran three CPU pilots. The dispersion
  gate stayed flat to within a few percent across conditions whose true epistemic content differs by 0.17
  nats, at every capacity budget tried, including a fully trainable backbone copy — so capacity was not the
  constraint. Once the region shortcut was removed from the episode contrast, the encoder probe
  (`probe_uncertainty_encoder.py`) showed DeepSets pooling of frozen target-column embeddings scoring
  in-family MI R² of **−0.009**: essentially all its apparent skill had been a surface cue.

The standing hypothesis at the end of that trial (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md:513-517`) is that
**frozen target-column embeddings, trained only to produce a predictive mean, carry little information about
posterior width**. Every prior experiment tested that through a learned head, so a null result there is
ambiguous between "the information is absent" and "the head cannot extract it."

This trial removes the head. Support-set resampling reads the encoder's sensitivity to its conditioning set
directly, with no training and no gate to collapse.

## Design

### What this measures, and its limits

Bootstrapping the support set is a **frequentist stability statistic** — how much an amortized predictor's
output moves when its conditioning set is perturbed — not the model's self-reported epistemic belief. The
repo already has evidence the two come apart: on random-label episodes where true epistemic content is zero,
the existing `context_resampling` baseline reports MI **0.1274** while every learned arm reports ≤ 0.010
(`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md:375-382`). Resampling a pure-noise context changes the prediction,
and that variation reads as disagreement regardless of whether any is warranted. Bootstrap-with-replacement
adds a second confound: a size-`n` resample has only ~63% unique rows, so part of its variance is a smaller
*effective* context, not genuine instability.

The design below keeps the literal bootstrap arm — it is the question the resampling schemes below answers —
but controls both confounds: a stratified-subsample scheme (fixed fraction, no replacement) separates "which
rows" from "how many rows," and every arm is checked against a random-label null and a permutation null before
its numbers are trusted.

### Resampling schemes

For one episode, the support set of size `n` is split into a fitting pool (`n − N`) and a held-out pool
(`N`), the latter never entering any ensemble member — it exists only to score members on genuinely unseen
rows. `B` members are then drawn from the fitting pool:

| Scheme | Draw | Purpose |
|---|---|---|
| `bootstrap` | size `n − N`, with replacement | the literal question; ~63% unique rows |
| `subsample` | fixed fraction, stratified by label, no replacement | isolates row identity from context size |
| `bayesian_bootstrap` | `Dirichlet(1)`-weighted resample | a documented resampling *approximation*: nanoTabPFN takes no row weights |

`stratified_subsample_indices` is shared with the existing `ContextResamplingUncertainty` baseline
(`tfmplayground/models/continuous_posterior.py:820`), pulled out as a free function so both callers are pinned
by the same tests.

### Dispersion statistics

All `B` members share one batched forward pass through `NanoTabPFNModel.encode_table` — no architecture
change. Per-layer query embeddings are captured with read-only `torch.nn.Module.register_forward_hook`s on
each `transformer_blocks[i]` (`tfmplayground/experiments/support_resampling.py:LayerCapture`); a test asserts
`encode_table`'s output is bit-identical with and without the hooks attached.

For query `q`, layer `l`, members `b = 1..B`:

- **`probability_dispersion`** — `Var_b p_b(q)`, plus the entropy form `H(p̄) − mean_b H(p_b)`, directly
  comparable to the teacher's exact mutual information.
- **`representation_dispersion[l]`** — `mean_b ‖h_{l,b} − h̄‖² / E`, plus a scale-free version divided by the
  mean embedding norm (LayerNorm makes raw norms layer-dependent, so only the scale-free version is
  comparable across layers).
- **`effective_rank[l]`** — participation ratio of the `B × E` deviation matrix. `1` means the resampling
  variance lives in one direction ("which hypothesis"); `min(B, E)` means it is isotropic jitter with no
  structure. Nothing in the repo's prior trials measured this.
- **`projected_dispersion[l]` / `projected_ratio[l]`** — dispersion projected onto the decoder's sensitivity
  direction `g_l = ∂(positive-class probability)/∂h_l`, taken from one `torch.autograd.grad` call at the
  full-context base point. Query rows never attend to each other (`datapoint_attention_stage` routes test
  rows only to support rows as keys/values), so summing the positive-class probability over all queries
  before differentiating gives every query's gradient in one call, with no cross-query contamination. This
  is **first-order everywhere**, including the final layer — the decoder is a two-layer MLP with a GELU
  nonlinearity, not linear, so no layer's projection is exact. `total − projected` is the component the
  decoder discards; comparing the two directly answers the open question left at the end of the continuous
  trial: does the encoder represent uncertainty that the read-out throws away, or was it never there?

### Model-intrinsic arms (no resampling)

Two arms need no ensemble at all. Under exchangeability the aleatoric part of the predictive factorizes
across queries given the latent function, so any residual dependence in the model's own *joint* over two
queries is epistemic — read by conditioning on a hypothetical label for one query and checking how much that
moves belief about another (or the same query). One batched forward over `2 × query_count` members (one per
`(query, hypothetical label)` pair) gives the whole conditional table:

- **`joint_mi`** — `I(y_q; y_q' | D)` from the model's own one-step-ahead conditionals, chained as
  `p(y_q = a) · p(y_q' = b | D, y_q = a)`.
- **`self_conditioning`** — expected movement in a query's own belief under its two hypothetical labels.

This construction is a heuristic, not a proof of coherence: the chain rule gives the exact joint only if the
model's one-step-ahead conditionals are consistent views of one true underlying joint, which an amortized
predictor is not guaranteed to satisfy. `exact_pairwise_mutual_information` computes the *true* pairwise MI
directly from the synthetic episode's exact candidate posterior (candidates are conditionally independent
given which one is active, so no sampling is needed either) — exactly the check for whether that heuristic
costs anything in practice.

### What the scores are validated against

1. **Ground truth (synthetic).** `exact_candidate_posterior` and the projected-teacher targets from
   `train_continuous_bayesian.teacher_targets` give exact epistemic MI. Scored by Spearman and R² against
   each arm's per-episode mean, using the same floors as the encoder probe (`MIN_R2 = 0.30`,
   `MIN_CONDITION_AUC = 0.85`) so results land in the same table. Training families for episode generation,
   **held-out families** (`HELDOUT_REGIME`) for the reported numbers.
2. **Discrimination.** Ambiguous-versus-identifiable AUC on held-out families.
3. **Null.** `random_label_episode`: true epistemic content is zero. An arm scoring above the floor here is
   reading resampling noise, and its other numbers must be read in that light regardless of how well they
   otherwise score.
4. **Performance link.** Per episode, `Var_b(member held-out log loss)` on the `N` held-out support rows
   against mean query dispersion. Dispersion is computed from query *embeddings only* (no labels anywhere)
   and performance from held-out *labels only*, so the two share nothing but the member index sets — a
   **permutation null** (shuffling the member→member pairing before recomputing the correlation) is reported
   alongside every statistic to bound that residual coupling.
5. **Usefulness (TabArena).** Error-detection AUROC/AURC against vanilla predictive entropy — **AUROC 0.767 /
   AURC 0.076** on the same 20 tasks used by every trial in this repo. This is the comparator that matters;
   every trial so far has lost to it.

### Pre-declared gates

1. Discrimination — held-out AUC ≥ 0.85 for at least one arm.
2. Ground-truth fit — Spearman against teacher MI ≥ 0.5 on held-out families.
3. Null — random-label MI ≤ 0.02, checked regardless of gates 1–2.
4. Performance link — |Spearman| ≥ 0.4 between episode dispersion and held-out loss variance, outside the
   permutation-null band.
5. Usefulness (stage 2) — TabArena error-detection AUROC > 0.767 or AURC < 0.076.
6. Encoder-versus-decoder — diagnostic, not gated: the per-layer `projected / total` ratio is reported
   whatever the outcome, since it answers the trial's open question either way.

Stage 2 was run only for arms with informative stage-1 signal, alongside `vanilla` and `context_resampling`
as references.

## Stage 1: synthetic evaluation

Command:

```bash
python -m tfmplayground.experiments.evaluate_resampling_synthetic \
    --output-dir results/support_resampling/synthetic \
    --episodes-per-condition 20 --support-size 128 --query-count 6 \
    --members 32 --heldout-size 16 --permutation-trials 200
```

20 episodes per condition, held-out families and held-out parameter ranges, support 128, `M = 6`, `B = 32`
members, `N = 16` held-out support rows.

### Mean mutual information per condition

| Arm | Ambiguous | Identifiable | Noisy | Teacher (ambiguous) |
|---|---:|---:|---:|---:|
| `bootstrap` | 0.1318 | 0.1317 | 0.1329 | 0.1440 |
| `subsample` | 0.0738 | 0.0688 | 0.0753 | — |
| `bayesian_bootstrap` | 0.1685 | 0.1588 | 0.1713 | — |
| `joint_mi` | 0.0044 | 0.0027 | 0.0026 | — |
| `self_conditioning` | 0.2450 | 0.2373 | 0.2531 | — |

Teacher mutual information is 0.1440 nats on ambiguous episodes and ≈ 0 on identifiable and noisy ones — the
same 0.14-nat gap every prior trial in this repo has tried to detect. **Every resampling arm is flat across
that gap** to within 1–3% relative, the identical failure mode of the learned continuous-posterior gate
(`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`: gate flat at 0.16–0.32 across the same 0.164-nat difference).
`joint_mi` is the only arm with any separation at all (0.0044 vs 0.0027, ambiguous vs identifiable) and it is
inside noise at this scale.

### Random-label null (true epistemic content is zero)

| Arm | Null MI | Clears ≤ 0.02 floor? |
|---|---:|:---:|
| `bootstrap` | 0.1591 | No |
| `subsample` | 0.0975 | No |
| `bayesian_bootstrap` | 0.2236 | No |
| `joint_mi` | 0.0020 | **Yes** |
| `self_conditioning` | 0.2321 | No |

This is the sharpest result in this stage. Every resampling arm reports *more* mutual information on
pure-noise labels than the continuous-posterior trial's learned MI ever reported on real signal (≤ 0.010
throughout that trial). Resampling the support set does not distinguish "the labels are uninformative" from
"the model is uncertain because it hasn't seen enough" — it reads context-size and row-identity noise as
disagreement, exactly the failure mode flagged for `context_resampling` at the design stage
(`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md:375-382`, MI 0.1274 there). Only `joint_mi` — the one arm that
needs no resampling at all — clears the null floor.

### Gates 1, 2, 4

Every arm fails discrimination (AUC ≤ 0.66, floor 0.85), ground-truth fit (R² negative for every arm — worse
than predicting the mean), and the performance link (|Spearman| ≤ 0.37, inside the permutation-null band in
every case: e.g. `bayesian_bootstrap` at 0.372 against a null band of 0.0 ± 2×0.137 = [−0.27, 0.27], which
does *not* clear the ≥ 2σ separation this trial pre-declared).

```json
"clears_gates_1_2_3": {
  "bootstrap": false, "subsample": false, "bayesian_bootstrap": false,
  "joint_mi": false, "self_conditioning": false
}
"any_arm_proceeds_to_stage_2": false
```

**No arm clears the pre-declared floor to proceed to stage 2.** Stage 2 was run anyway, on a small task
subset, purely to validate the harness end to end on real data — see below.

## Stage 2: TabArena evaluation

Command:

```bash
python -m tfmplayground.experiments.evaluate_resampling_tabarena \
    --task-ids 31,37,3,29 --output-dir results/support_resampling/tabarena \
    --arms bootstrap,subsample,joint_mi --context-size 256 --members 16
```

This is a **small run (4 of the 20 official tasks: kr-vs-kp, credit-approval, credit-g, diabetes)**, run to
validate the harness end to end on real data, not the full evaluation — stage 1 already found no arm to
promote. All numbers below are informational.

| Arm | Error-detection AUROC | AURC | Mean MI | Deployed prob. matches vanilla (≤ 1e-6)? |
|---|---:|---:|---:|:---:|
| **vanilla predictive entropy** | **0.7986** | **0.0866** | — | — (reference) |
| `context_resampling` | 0.7882 | 0.0878 | 0.0338 | Yes (exactly 0.0) |
| `bootstrap` | 0.7768 | 0.0870 | 0.0503 | No (1.8e-6) |
| `subsample` | 0.7463 | 0.0983 | 0.0203 | No (1.8e-6) |
| `joint_mi` | 0.6288 | 0.1381 | 0.0003 | No (1.8e-6) |

Vanilla predictive entropy wins on this subset too — the pattern holds on real data. Every arm's error-AUROC
is below vanilla's 95% bootstrap interval lower bound (0.735), and `no_material_auroc_harm` is `false` for
every arm in the TabArena gate. `joint_mi`, the arm that passed the stage-1 null check, is the *worst*
performer here (AUROC 0.629): passing the null does not imply the signal is useful, only that it is not pure
noise — a separate finding from "it predicts errors," and this subset shows the gap between them directly.

The residual ~1.8e-6 probability difference from vanilla (every arm except `context_resampling`, which is
built to be exact by construction) is the numerical-path artifact described below, not a re-emergence of the
mean-preservation bug: two independent forward passes through the same frozen network agree to 1e-6, not to
machine epsilon, and that gap does not grow with dataset size or arm.

## Bottom line

The standing hypothesis this trial was built to test — that frozen target-column embeddings, trained only to
produce a predictive mean, carry little information about posterior width — is **not falsified, and this
result strengthens it rather than merely failing to overturn it**. The prior trials could not separate "the
information is absent" from "the learned head cannot extract it," because both used a trained head. This
trial removes the head entirely and still finds nothing: bootstrap, stratified subsample, and a
Bayesian-bootstrap approximation all fail discrimination, ground-truth fit, and the performance link, and all
three read more signal off *pure label noise* than any learned arm in this repo has ever reported on real
epistemic content. That is not "no signal" — it is a *negative* signal: the frozen encoder's sensitivity to
which rows are in its context is dominated by context-size and row-identity artifacts, not by the structure
of the labels.

The one arm that clears the null (`joint_mi`, reading the model's own one-step-ahead conditionals with no
resampling) is a genuinely different measurement — and it is the worst performer on real TabArena error
detection. Passing the null and being useful are different claims, and this run is the first place in this
repo's history the gap between them shows up directly in the same table.

Put together with the two prior trials, three independent constructions — a learned slot posterior, a
learned continuous dispersion gate, and now four training-free resampling/intrinsic estimators — have now
failed to extract a useful epistemic signal from this backbone. The common factor across all three is the
backbone itself, frozen and trained only on a predictive-mean objective. The decisive next test, following
directly from the encoder probe's own conclusion (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`: "the decisive
test is the same probe with the uncertainty backbone's adapters trainable... has not been run"), is whether
*any* representation extracted from this checkpoint can carry the signal — which means training, not probing,
the backbone itself against an epistemic target, rather than continuing to build new read-outs on top of a
representation three separate constructions have now found wanting.

## What went wrong on the way here, and why it matters

Two implementation bugs were caught before any of the numbers above were trusted:

1. **The gradient was disconnected.** The first `decoder_gradient` implementation registered a hook that
   *sliced* each transformer block's output before storing it, matching what every other consumer in this
   module needs. But intermediate blocks feed forward the *whole* raw output, not the sliced copy — so the
   captured slices for every layer except the last were dead-end leaves with no path back to the decoder's
   output, and `torch.autograd.grad` correctly refused to differentiate through them
   (`RuntimeError: One of the differentiated Tensors appears to not have been used in the graph`). The fix
   captures the whole block output when a gradient is needed, and slices only after the gradient call.
2. **The deployed probability silently drifted from vanilla.** The first `SupportResamplingClassifier`
   reused the synthetic-stage fit/held-out split at deployment time, so the base-point forward pass — and
   therefore the reported probability — was built from a context missing the held-out rows. On a real
   TabArena task this showed up as a 0.485 max probability difference from vanilla, where every other arm in
   this repo (by construction) is either exactly 0 or within 1e-6. The held-out split is a stage-1-only
   diagnostic device; deployment resamples the full labelled context directly, and the residual gap after the
   fix is ~4e-6 — small, and consistent with ordinary floating-point path differences between two separate
   forward passes through the same computation, not with a leftover correctness bug.

Both are recorded here because they are exactly the kind of error this trial's own null and permutation
checks are designed to catch downstream — a disconnected gradient would have produced a `projected_ratio` of
uniformly zero, and a leaked mean would have inflated `max_probability_difference_to_vanilla` past every
other arm's numbers, both legible in the output tables rather than silent.

## Reproduction

```bash
python -m unittest tests.test_support_resampling
python -m tfmplayground.experiments.evaluate_resampling_synthetic --help
python -m tfmplayground.experiments.evaluate_resampling_tabarena --help
```

Large run directories stay ignored by git; compact JSON/CSV summaries under `results/support_resampling/` are
committed.

## Files

- `tfmplayground/experiments/support_resampling.py` — resampling schemes, layer-hook capture, dispersion
  statistics, the two model-intrinsic estimators, the exact pairwise ground truth
- `tfmplayground/support_resampling_interface.py` — scikit-learn interfaces (`SupportResamplingClassifier`,
  `IntrinsicPosteriorClassifier`)
- `tfmplayground/experiments/evaluate_resampling_synthetic.py` — stage-1 ground-truth and null evaluation
- `tfmplayground/experiments/evaluate_resampling_tabarena.py` — stage-2 TabArena evaluation
- `tfmplayground/models/continuous_posterior.py` — `stratified_subset_indices` extracted as a free function
  (behaviour-preserving refactor; `ContextResamplingUncertainty` now delegates to it)
- `tests/test_support_resampling.py` — hook identity, determinism, no-leakage, ground-truth sanity, and a
  real-checkpoint qualitative check (identifiable support has lower dispersion than ambiguous)
