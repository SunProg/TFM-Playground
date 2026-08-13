# The `h5_allterms` model (`runs/h5_prior_bimodal_filter/k2-all-terms-v1`)

## What it is in one line

A K=2 particle latent Bayes filter head bolted onto a frozen-then-partially-unfrozen
nanoTabPFN backbone, trained on paired-hypothesis episodes drawn from the HDF5 prior dump
`300k_150x5_2.h5`, with **all three of the late-added loss terms enabled at once** —
`coverage_weight`, `assignment_weight`, and `posterior_weight`, all 1.0. "All terms" is
exactly what the name means: it is the loss-ablation arm of the
`h5_prior_bimodal_filter` family in which nothing is switched off.

| | |
|---|---|
| Run directory | `runs/h5_prior_bimodal_filter/k2-all-terms-v1/` |
| Trainer | `tfmplayground/experiments/train_h5_prior_bimodal_filter.py` → `train_prior_bimodal_filter.py` |
| Model class | `NanoTabPFNIntegratedLatentFilter(backbone, num_hypotheses=2)` |
| Episode generator | `generate_h5_prior_bimodal_episodes` (`prior_bimodal_episodes.py:203`) |
| Data | `300k_150x5_2.h5` (300k records, 150 rows × 5 features) |
| Parent checkpoint | `checkpoints/nanotabpfn.pth` |
| Checkpoint written | `selected_checkpoint.pth`, `stage="prior_bimodal_selected"` |
| Device / seed | mps / 2402 |

## Why it exists

Three separate bugs and gaps were found in this family, each of which silently removed the
pressure the architecture was supposed to be under. Each fix added a loss term, defaulting
to **off** so that earlier runs stayed reproducible. `k2-all-terms-v1` is the run that turns
all of them on together:

1. **`coverage_loss`** (`train_prior_bimodal_filter.py:110`). `best_particle` only ever
   references the *true* candidate, so it is fully satisfied by one competent particle plus
   K−1 junk ones. Coverage sums over **every** candidate, so particle collapse becomes
   impossible — one particle cannot explain two mutually-disagreeing hypotheses at once.
2. **`assignment_loss`** (`:123`). Coverage can still be satisfied by a single particle
   claiming both hypotheses. Assignment enumerates injective candidate→particle matchings
   (Hungarian by enumeration; with C=2 and K particles there are only K·(K−1) options) and
   charges the best one, forbidding that.
3. **`posterior_supervision_loss`** (`:144`). Nothing else in the family trains
   `log_weights` at all — it appeared only in evaluation metrics, which is why
   `effective_particle_count` sat pinned at the uniform value and the filter never learned
   to *update* belief as stream evidence arrived. Particle identity is permutation-invariant,
   so the target particle is a detached E-step: the particle with the largest
   **discriminative margin** for the true candidate over the best alternative (margin, not
   plain argmax-likelihood, so it targets the particle that best *distinguishes* the
   hypotheses). Only the **final** posterior is supervised — the support is deliberately
   uninformative, so the correct early belief really is near-uniform and supervising every
   step would force premature confidence.

A fourth fix is baked into the shared path rather than gated by a weight, and matters for
reading any older number in this family: `best_particle` used to apply
`.clamp_min(1e-12).log()` to values that were **already log-probabilities**, mapping every
negative log-prob to the same constant −27.631. The term was a constant 26.938 with zero
gradient regardless of predictions. That silently disabled the only pressure for particles to
explain the query differently — which is why `slot_joint_js` sat at ~0 in every run of this
family before `k2-fixedloss-v1`.

## The task

Deliberately constructed two-hypothesis ambiguity, from real prior records rather than a
hand-built synthetic family. For each episode:

- Take one record's feature matrix `X`, and pair it with the labels of a **different** record
  having the same active feature count and eval position. The two label vectors are the two
  candidate hypotheses over identical covariates.
- Accept the pair only if it is genuinely ambiguous up front and resolvable later:
  `support_disagreement ≤ 0.20`, `stream_disagreement ≥ 0.25`, `query_disagreement ≥ 0.25`.
- Layout is fixed at **32 support : 32 stream : 4 query** (the adapter hard-requires it,
  `prior_bimodal_episodes.py:219`).
- Pick a true candidate uniformly; its labels are what the model sees.

The model receives only support/stream/query tensors. All candidate metadata
(`candidate_task`, `candidate_*_y`, the three disagreement rates) is **evaluator-only** — but
note that the three new loss terms above read `candidate_query_y` and `candidate_task`, so
under `all-terms` the training signal *is* privileged relative to the older arms. That is
intended (it is supervision, not leakage into the forward pass), but it makes `all-terms` not
strictly comparable to arms that never touch candidate metadata.

Pair rejection is expensive: mean **76 attempts per accepted example** at evaluation
(`pair_attempts`), against a budget of `max_pair_attempts=5000`.

## Loss

Total = `integrated_loss` base + `best_particle` + the three gated terms.

Base (`train_integrated_latent_filter.py:149`), all unweighted except as noted:
prequential NLL + joint-trajectory NLL + marginal NLL + `diversity_weight`·hinge(0.20 −
min-pairwise slot JS) + `coherence_weight`·incoherent mass + `residual_weight`·|stream
residual|.

Added here:

| term | weight | what it forces |
|---|---|---|
| `best_particle` | 0.25 | at least one particle explains the true query, permutation-invariantly (`logsumexp` over particles) |
| `coverage` | 1.0 | *every* candidate is explained by *some* particle |
| `assignment` | 1.0 | candidates map to **distinct** particles |
| `posterior` | 1.0 | final `log_weights` point at the max-margin particle for the true candidate |

## Training schedule

Three-stage backbone unfreezing via `set_trainability`, with per-stage early stopping
(`patience=4`, validation every 25 steps, best-state restore):

| stage | steps | backbone LR | head LR |
|---|---|---|---|
| `frozen` | 1–200 | — | 1e-4 |
| `partial` (last 2 blocks) | 201–300 | 1e-5 | 1e-4 |
| `full` | 301–400 | 5e-6 | 1e-4 |

`batch_size=2`, `accumulate_gradients=4`, AdamW `weight_decay=1e-2`, grad-norm clip 1.0.
Best validation loss **5.6548 at step 325** (inside the `full` stage).

### Checkpoint selection in this run was unreliable — since fixed

`k2-all-terms-v1` selected on `validation_loss`, i.e. **the training objective itself**
(diversity hinge, coherence, coverage, assignment and posterior terms included), measured on
**6 episodes** whose seed was `config.seed + 100_000 + step` — so the validation set was
*redrawn at every call*. Across the 16 validation points the sd is **1.257**, while the
selected step (325, 5.655) beat the runner-up (200, 5.768) by **0.11**. Selection here was
mostly picking a favourable draw. No task-level metric was computed during training at all;
`true_task_recovered` and `ensemble_query_nll` existed only in the post-hoc `evaluate()`.

The trainer has since been fixed (see "Validation and selection" below). **This checkpoint
predates the fix** — its selected step should be treated as arbitrary within ~1 sd, and any
rerun will not reproduce it.

## Validation and selection (current trainer)

Three changes in `train_prior_bimodal_filter.py`:

1. **Fixed validation set.** The seed is `config.seed + 100_000`, with no `step` term, so the
   same episodes are scored at every validation point and values are comparable across steps.
   A per-batch seed alone was not enough: the SCM episode path draws task networks through the
   *global* torch/numpy RNG, so `_validation_metrics` now pins the global state for the pass
   and restores it afterwards. That also stops validation from consuming draws training would
   have made, making training reproducible independently of `validation_interval`.
2. **`validation_episodes: int = 64`** (was a hardcoded 3 batches = 6 episodes).
3. **`selection_metric`, default `ensemble_query_nll`** — a task-level metric the objective
   does not contain. `--selection-metric validation_loss` reproduces the old criterion.
   `validation_loss`, `validation_ensemble_query_nll`, `validation_true_task_recovered`,
   `validation_episodes` and `selection_value` are all written to `learning_curves.csv`
   regardless of which one drives selection.

## Results

Evaluation: 256 episodes, `config.seed + 300_000 + trial`.

| metric | value |
|---|---|
| `ensemble_query_nll` | **1.939** |
| `query_nll` (per-particle mean) | 1.984 |
| `vanilla_query_nll` (frozen backbone, stream in context) | 3.163 |
| `slot_joint_js` | 0.0206 |
| `effective_particle_count` (of 2) | 1.604 |
| `true_task_recovered` | **0.490** |
| `prequential_log_likelihood` | −0.462 |

By feature count (`evaluation_summary.csv`) — the dump's widths that survive pairing:

| feature_count | trials | ensemble_nll | vanilla_nll | slot_JS | eff. K | recovered |
|---|---|---|---|---|---|---|
| 4 | 4 | 3.028 | 4.141 | 0.0195 | 1.242 | 0.500 |
| 5 | 252 | 1.935 | 3.160 | 0.0206 | 1.605 | 0.490 |

Loss-term trajectory (mean of first/last 5 steps of each stage):

| term | frozen start → end | partial end | full end |
|---|---|---|---|
| `loss` | 10.02 → 6.95 | 7.55 | 7.37 |
| `coverage_loss` | 2.000 → 1.386 | 1.369 | 1.336 |
| `assignment_loss` | 2.686 → 1.908 | 1.974 | 1.855 |
| `posterior_loss` | 0.681 → 0.533 | 0.803 | 0.943 |
| `slot_joint_js` | 0.0001 → 0.031 | 0.010 | 0.041 |
| `best_particle_loss` | 2.006 → 1.181 | 1.313 | 1.245 |

## Where this run stands against its siblings

All arms, 256-trial evaluations unless noted (`post`/`cov`/`asg` = the three gated weights;
blank means the field did not exist in that run's code):

| run | trials | post | cov | asg | ens_nll | van_nll | slot_JS | eff_K | recovered |
|---|---|---|---|---|---|---|---|---|---|
| `k2-proper-v1` | 256 | – | – | – | 2.825 | 3.163 | 0.0000 | 1.989 | 0.498 |
| `k2-eval1000-smoke-v2` | 1000 | – | – | – | 2.833 | 3.356 | 0.0001 | 1.959 | 0.508 |
| `k2-fixedloss-v1` | 256 | – | – | – | 1.929 | 3.163 | 0.0483 | 1.490 | 0.516 |
| `k2-posterior-v1` | 256 | 1.0 | – | – | 1.957 | 3.163 | 0.0037 | 1.858 | 0.496 |
| **`k2-all-terms-v1`** | 256 | 1.0 | 1.0 | 1.0 | **1.939** | 3.163 | 0.0206 | 1.604 | **0.490** |

Read honestly:

- **The `best_particle` log-bug fix is the only change that moved the NLL.** 2.825 → 1.929
  between `k2-proper-v1` and `k2-fixedloss-v1`. Everything after that is flat: all-terms
  1.939 vs fixedloss 1.929 — all-terms is marginally *worse*.
- **The three added terms did not buy diversity.** `slot_joint_js` is 0.0206 under all-terms
  versus **0.0483** under fixedloss, i.e. adding coverage + assignment + posterior more than
  halved particle diversity relative to just fixing the bug. Both are far below the 0.20
  diversity target the base loss is nominally hinged on.
- **`posterior_loss` rises across training** (0.533 → 0.803 → 0.943) while total loss also
  drifts up after the frozen stage. The posterior term is losing to the rest of the
  objective, not being satisfied by it.
- **`effective_particle_count` moved off uniform** (1.604 vs 1.989 for the no-posterior arms),
  so posterior supervision does do the mechanical thing it was written for — the filter now
  concentrates belief. It just does not convert into task identification.
- **`true_task_recovered = 0.490` is chance.** And chance is the wrong bar anyway.

## The bar this model fails

`probe_task_identifiability.py` (documented in `ADAPTIVE_PARTICLE_FILTER_RESEARCH.md` §12)
asks whether **plain frozen vanilla nanoTabPFN**, handed support + the first 16 stream rows
as ordinary in-context data, can pick the true candidate by scoring both candidates' labels
on the held-out stream suffix. No filter, no particles, no gate, no training.

| | true-task identification |
|---|---|
| chance | 0.500 |
| plain frozen vanilla, stream as context | **0.621** (159/256, p=0.00013, 95% CI [0.559, 0.681]) |
| `k2-all-terms-v1` (`true_task_recovered`) | 0.490 |

So the filter is not merely undertrained — it is **below the do-nothing-clever baseline** on
the same episodes. The correct reference for any `true_task_recovered` number in this
codebase is 0.621, not 0.500. The one encouraging number is `ensemble_query_nll` 1.939 vs
vanilla 3.163: on *query likelihood* the two-particle ensemble genuinely beats the frozen
backbone. It just cannot say which hypothesis it is in.

The research log also notes the headroom above 0.62 looks representational rather than
evidence-limited: the high-candidate-disagreement half of episodes scores 0.631 vs 0.621
overall, so more distinguishing evidence barely helps.

## Known hazards

- **`300k_150x5_2.h5` has 5 corrupted records** of 300,000 (ids 91466, 176585, 263212,
  278868, 291026), each ~300 of 750 cells non-finite. Drawing one NaNs the run permanently;
  under a fixed seed this is deterministic (two earlier runs died at exactly step 158).
  `generate_h5_prior_bimodal_episodes` now scans and excludes them once per path and caches
  the index (`prior_bimodal_episodes.py:237-268`), emitting a `RuntimeWarning`.
- **Feature width is one value per batch**, sampled from the dump's natural distribution
  (`:281`), so tensors stay dense. In practice almost everything is width 5 — 252 of 256
  evaluation episodes. The `min_features=1, max_features=16` config range is therefore
  mostly inert for this dump.
- **Pair acceptance is the bottleneck**, ~76 attempts per example. Tightening the
  disagreement thresholds further will make runs much slower before it makes them better.
- **`all-terms` consumes candidate metadata in the loss.** Comparisons against
  `k2-proper-v1` / `k2-fixedloss-v1`, which do not, are not apples-to-apples.
- **Single seed (2402), 256 evaluation episodes.** The `slot_joint_js` gap versus
  `k2-fixedloss-v1` and the ~0.02 recovery differences across arms are within the noise this
  family has already produced spurious findings at.

## Reproduce

```bash
PYTHONPATH=. python -m tfmplayground.experiments.train_h5_prior_bimodal_filter \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir runs/h5_prior_bimodal_filter/k2-all-terms-v2 \
    --device mps --seed 2402 \
    --batch-size 2 --accumulate-gradients 4 \
    --frozen-steps 200 --partial-extra-steps 100 --full-extra-steps 100 \
    --validation-interval 25 --patience 4 \
    --posterior-weight 1.0 --coverage-weight 1.0 --assignment-weight 1.0 \
    --best-particle-weight 0.25 \
    --validation-episodes 64 --selection-metric ensemble_query_nll \
    --max-pair-attempts 5000 --evaluation-trials 256 --evaluation-report-interval 1000
```

Add `--selection-metric validation_loss --validation-episodes 6` to reproduce the original
run's (unreliable) selection behaviour instead.

The `--prior-dump 300k_150x5_2.h5` flag is injected by the `train_h5_prior_bimodal_filter`
wrapper if absent. Drop the three `--*-weight 1.0` flags to get the `k2-fixedloss-v1` arm;
add `--no-use-diversity` for the diversity ablation.

Artifacts written: `config.json`, `learning_curves.csv`, `evaluation_metrics.csv`,
`evaluation_summary.csv` (grouped by `feature_count`), `selected_checkpoint.pth`.
