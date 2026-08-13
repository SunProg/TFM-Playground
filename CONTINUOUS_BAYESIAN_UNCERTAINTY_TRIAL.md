# Slot-Free Continuous Bayesian Uncertainty for nanoTabPFN

Follow-up to [MEAN_PRESERVING_BAYESIAN_TRIAL.md](MEAN_PRESERVING_BAYESIAN_TRIAL.md), which is retained as the
documented negative benchmark. That trial's slot model, checkpoints, and checkpoint loader are unchanged.

## Status

- Date: 2026-08-13
- Scope: binary classification only
- Mean path: frozen vanilla nanoTabPFN from `checkpoints/nanotabpfn.pth` (6 layers, embedding 192, 6 heads,
  feed-forward 768, 3,717,514 parameters)
- Training data: synthetic structured episodes only
- TabArena: evaluation only; no TabArena labels or tasks are used for training or for selecting `lambda`
- Implemented and verified: model, episode generator, objective, sweep grid, synthetic evaluation, TabArena
  evaluation harness, CPU pilot training of all four learned arms (400 steps each) and a held-out-family
  synthetic evaluation of all five arms
- **Not yet run: the CREATE Slurm screening sweep, the three-seed final training, and the TabArena
  evaluation.** Those sections below state the exact submission commands and are explicitly marked as
  pending rather than filled with numbers.

## Architecture

Two independent paths are used, and they never share gradients.

1. **Frozen mean path.** An untouched vanilla `NanoTabPFNModel` computes `mu_q`. Its parameters are
   `requires_grad_(False)` for the lifetime of the model, and the forward pass runs under `torch.no_grad()`.
   The deployed prediction is always `mu_q`.
2. **Trainable uncertainty path.** A second `NanoTabPFNModel`, deep-copied from the same checkpoint,
   produces representations used only for uncertainty. Three modes are compared:
   - `frozen`: the uncertainty backbone is completely frozen and its embeddings are detached;
   - `adapters`: the backbone is frozen and residual bottleneck adapters are trained (**primary model**);
   - `full`: every parameter of the uncertainty copy is fine-tuned (maximum-capacity diagnostic).

### Adapters

`TransformerEncoderLayer.forward` was refactored into three stages with identity hooks
(`adapt_after_feature_attention`, `adapt_after_datapoint_attention`, `adapt_after_mlp`). The hooks return
their input unchanged in the ordinary model, so `NanoTabPFNModel` is numerically identical to before; the
full existing test suite passes unchanged, and a dedicated test asserts hook identity.

`AdaptedTransformerEncoderLayer` overrides the three hooks with a `BottleneckAdapter`:

```text
adapter(h) = h + W_up(GELU(W_down(LayerNorm(h))))
```

`W_up` is zero-initialized, so a freshly built adapter encoder reproduces the pretrained representation
exactly (tested with `atol=0`). Six layers times three stages gives **18 adapters**; the default bottleneck
is 32. Pretrained parameters keep their original state-dictionary names, so the vanilla checkpoint loads
into an adapted layer without renaming.

### Uncertainty head

- **Support pooling.** Target-column embeddings of the labelled rows go through a per-row MLP; the mean and
  variance across support rows are concatenated and mapped to the episode context `c_D`. This is a DeepSets
  encoder, so it is permutation invariant. There are no learned hypothesis tokens anywhere in the model.
- **Latent draws.** `z_s = latent_generator(c_D, epsilon_s)`, a conditional residual MLP whose output is
  added to `epsilon_s`. The noise `epsilon_s` comes from a scrambled Sobol sequence mapped to standard
  normal values by the Gaussian inverse CDF, in antithetic pairs (`z` and `-z`), so the noise set has an
  exactly symmetric empirical mean. The same `z_s` is used for every query in the episode, which makes one
  draw one coherent possible prediction function over the whole query set.
- **Query decoder.** `r_qs = query_decoder(query_representation_q, c_D, z_s)`.
- **Dispersion gate.** `g_q = sigmoid(gate(query_representation_q, c_D))`.

There are no persistent slots, no candidate identities, no posterior slot weights, no Hungarian matching,
and no hypothesis count `K`. Sample index `s` carries no meaning between episodes; `S` is a Monte-Carlo
count only. Effective sample size is deliberately **not** reported for equal-weight anonymous samples.

### Exact mean preservation

```text
d_qs      = r_qs - average_over_samples(r_qs)
b_q       = largest scale with mu_q + b_q * d_qs inside (0, 1) for every s
a_q       = g_q * b_q
p_qs      = mu_q + a_q * d_qs
```

Because `d_qs` sums to zero over samples and `a_q` does not depend on `s`,
`average_over_samples(p_qs) = mu_q` up to floating point. Observed maximum error in every test and pilot
run is `<= 6e-8`, well inside the `1e-6` requirement, and the deployed probability difference against
vanilla is exactly `0.0`.

### Beta ablation

The Beta arm reuses the same adapter-equipped encoder and predicts one concentration `kappa_q`:

```text
alpha_q = mu_q * kappa_q
beta_q  = (1 - mu_q) * kappa_q
```

The Beta **distribution** mean is exactly `mu_q` and the deployed probability is exactly `mu_q`, but its
*sample* mean is only correct in expectation because Beta draws are not a centred deviation construction.
Its `sample_mean_preservation_error` is therefore a Monte-Carlo quantity and is reported separately from
the deployed probability difference; the two are never conflated. Beta samples use reparameterized
`rsample` inside a forked, seeded RNG, so they are reproducible for a given inference seed (recorded in the
checkpoint as `sample_generation.scheme = "beta_rsample"` rather than the Sobol scheme).

## Symbols

| Symbol | Meaning |
|---|---|
| `D`, `N`, `x_i`, `y_i` | labelled support set, its size, support row `i`, and its binary label |
| `x_q`, `M` | query row `q` and the number of query rows |
| `mu_q` | vanilla nanoTabPFN probability of class 1 for query `q` |
| `c_D` | pooled permutation-invariant support representation |
| `epsilon_s`, `z_s`, `S` | Sobol noise draw, anonymous latent draw, and the number of posterior samples |
| `r_qs`, `d_qs` | raw and sample-centred deviation for query `q` under sample `s` |
| `b_q`, `g_q`, `a_q` | largest safe scale, learned dispersion gate in `(0, 1)`, and `a_q = g_q * b_q` |
| `p_qs` | class-1 probability for query `q` under posterior sample `s` |
| `C`, `h`, `rho_h` | number of true candidate functions, candidate index, exact candidate posterior weight |
| `theta_qh` | candidate `h`'s class-1 probability for query `q` |
| `eta` | controlled label-noise probability |
| `H(p)` | binary entropy `-p log p - (1-p) log(1-p)` in nats |
| `lambda` | weight of epistemic uncertainty in the combined risk score |

## Uncertainty outputs

`ContinuousPosteriorPrediction` returns vanilla binary probabilities, posterior sample probabilities,
predictive entropy `H(mu_q)`, expected conditional entropy `average_s H(p_qs)`, mutual information
`H(mu_q) - average_s H(p_qs)`, epistemic variance `average_s (p_qs - mu_q)^2`, the query-to-query epistemic
covariance matrix, and the maximum mean-preservation error. Natural logarithms are used throughout.

## Training data

| Property | Value |
|---|---|
| Candidate counts `C` | 2, 4, 8, 16 (an episode property; the model has no `K`) |
| Support sizes | 32, 64, 128, 256, 512 |
| Query counts | 4 to 8, so all `2^M` joint label vectors can be enumerated |
| Label noise `eta` | 0.00, 0.05, 0.10, 0.20, 0.35 (a `1e-3` floor keeps the posterior finite) |
| Curriculum | 40% ambiguous, 25% identifiable, 20% structured noisy-label, 15% paired support-prefix |
| Training families | `linear`, `threshold`, `tree`, `sparse_interaction`, `smooth`, `mlp_scm` |
| Held-out families | `dense_interaction`, `tree_scm` |
| Training parameter range | 2–12 features, threshold quantile 0.35–0.65, column scales `10^[-1, 1]` |
| Held-out parameter range | 13–16 features, threshold quantile 0.20–0.35, column scales `10^[-2, -1]` |

Every candidate function in an episode is evaluated on the *same* feature rows, so candidates are different
latent functions on one table rather than different datasets. Class imbalance comes from the threshold
quantile, feature-scale variation from the per-column scales, and irrelevant features arise wherever a
candidate ignores a column. Candidate probabilities are `1-eta` for the class selected by the deterministic
function and `eta` for the opposite class, and `rho_h` is the exact Bayes posterior from support labels
under that noise model with a uniform candidate prior.

For ambiguous episodes the support rows are made exactly uninformative (all candidates are forced to agree
there) while structured disagreement is kept on the queries. With `C = 16`, naturally agreeing rows are far
too rare to select, and a partially identifying support would make the condition a misnomer; the exact
posterior is uniform by construction, which is what the model is asked to reproduce. Paired episodes share
queries and candidates, and the extension adds support rows on which candidates genuinely disagree.

Random-label episodes are diagnostic only and are never training batches. Model selection uses held-out
*families*, not only held-out seeds.

## Objective

Distributions are compared directly; no sample is ever matched to a candidate identity.

```text
total_loss = 1.00 * energy_distance
           + 0.50 * mutual_information_loss
           + 0.50 * variance_loss
           + 0.25 * covariance_loss
           + 1.00 * joint_query_loss
           + 0.25 * evidence_monotonicity_loss
```

- **Energy distance** between the equal-weight model sample vectors and the `rho`-weighted projected
  candidate vectors, with Euclidean query-vector distances normalized by `sqrt(M)`.
- **Mutual-information loss**: MSE against the projected teacher, normalized by `log(2)`.
- **Variance loss**: MSE against the projected teacher, normalized by `0.25`.
- **Covariance loss**: MSE between query-to-query covariance matrices, normalized by `0.25`.
- **Joint query-vector loss**: cross-entropy over all `2^M` binary label vectors between the coherent model
  joint and the coherent projected-candidate joint.
- **Evidence monotonicity**: `relu(MI_long - MI_short)` on paired support-prefix episodes.

Ordinary marginal cross-entropy is deliberately absent: the mean is fixed, so it carries no information
about posterior spread. Every component is logged separately. The early-stopping/selection statistic
subtracts the teacher's own joint entropy, which depends on the episode's candidate count rather than on
model quality.

### Mean-preserving teacher

```text
teacher_mean_q        = sum_h rho_h * theta_qh
candidate_deviation   = theta_qh - teacher_mean_q
projected_theta_qh    = mu_q + safe_scale_q * candidate_deviation_qh
```

`safe_scale_q` is the largest value at most one keeping every projected probability inside `(0, 1)`. The
projection preserves the exact weights `rho_h` and the direction and relative structure of candidate
disagreement, contracts it only as far as the fixed vanilla mean requires, and has weighted mean exactly
`mu_q` (tested to `1e-6`). Calibration is reported against both the exact posterior
(`exact_*` metrics) and the feasible projected teacher, and the two are always labelled distinctly.

## Compared arms

| Arm | Learned parameters | Notes |
|---|---|---|
| `vanilla` | none | mean and predictive entropy reference |
| `context_resampling` | none | 16 deterministic label-stratified subsets at fractions 0.50/0.75/0.90, re-centred on the full-context vanilla mean |
| `beta_adapter` | adapters + concentration head | deliberately low-capacity output distribution |
| `frozen_continuous` | head only | tests whether frozen representations suffice |
| `adapter_continuous` | adapters + head | **primary model** |
| `full_continuous` | whole uncertainty copy + head | maximum-capacity diagnostic |
| `slot` | previous trial's checkpoint | documented negative benchmark |

## Checkpoints

New explicit formats: `nanotabpfn_continuous_posterior` and `nanotabpfn_beta_concentration`, both
`format_version = 1`. Each checkpoint stores the frozen source-checkpoint path and SHA-256, the uncertainty
mode, adapter configuration, latent dimension, sample-generation configuration, model and optimizer state,
the training step, validation metrics, random seeds, and the selected hyperparameters. Reload reproduces
deterministic Sobol-sample predictions exactly for the same inference seed (tested with `atol=0`). The slot
loader `load_bayesian_checkpoint` is untouched and still loads old checkpoints; the new loader rejects them
with a message pointing at the old one.

## Hyperparameter sweep (pending on CREATE)

Partition `biomed_a30_gpu`, one A30 per task, `--array=0-33%8` (eight concurrent tasks; the limit is never
below eight). The trainer asserts `torch.cuda.is_available()` under `--require-cuda` and logs GPU name,
CUDA version, hostname, Slurm job id, and array index to `environment.json`.

| Architecture | Grid | Configurations |
|---|---|---:|
| adapter continuous | bottleneck {16, 32, 64} x latent {16, 32, 64} x lr {1e-4, 3e-4} | 18 |
| frozen continuous | latent {16, 32, 64} x lr {1e-4, 3e-4} | 6 |
| full uncertainty copy | latent {32, 64} x lr {1e-5, 3e-5} | 4 |
| Beta adapter | bottleneck {16, 32, 64} x lr {1e-4, 3e-4} | 6 |

Training posterior samples are 32 everywhere. Screening runs at most 1,500 steps with validation every 100
steps, patience 8, `min_delta = 1e-4`. The two best configurations per architecture, ranked by held-out-
family validation, are retrained with seeds 2402/2403/2404 for at most 5,000 steps with patience 10.
AdamW, gradient clipping at norm 1.0, and best-checkpoint (not final-checkpoint) selection are used
throughout.

```bash
sbatch scripts/slurm/sweep_continuous_bayesian_a30.sbatch
python -m tfmplayground.experiments.continuous_sweep \
    --summarize runs/continuous_bayesian/sweep-<ARRAY_JOB_ID> \
    --output-dir results/continuous_bayesian/sweep
SCREENING_INDEX=<index> sbatch scripts/slurm/train_continuous_bayesian_finalist_a30.sbatch
```

Slurm job IDs: **pending — the sweep has not been submitted.**

## CPU pilot

A CPU pilot was run locally to verify the whole pipeline end to end and to give a first, honest comparison.
It is **not** the CREATE sweep: one configuration per architecture, 400 steps, batch size 1 with two
gradient accumulation micro-steps, validation every 50 steps on 8 held-out-family episodes. Treat it as a
smoke-scale signal about plumbing and gross behaviour, not as evidence about final model quality.

Configuration for all arms: adapter bottleneck 32, latent dimension 32, `S = 32`, `max_support_size = 128`
(a compute cap; the CREATE sweep uses the full 32–512 range), seed 2402. Learning rate 3e-4 for the head
and adapter arms and 3e-5 for the fully trainable copy. Artifacts:
`results/continuous_bayesian/cpu_pilot_training.csv` and
`results/continuous_bayesian/cpu-pilot-synthetic/`.

### Training

| Arm | Best step | Validation selection loss (lower) | Validation energy distance (lower) | Mean-preservation error |
|---|---:|---:|---:|---:|
| adapter continuous | 300 | 0.065080 | 0.046126 | 3.8e-08 |
| frozen continuous | 250 | 0.065455 | 0.046957 | 5.2e-08 |
| full copy continuous | 350 | 0.062941 | 0.044783 | 4.5e-08 |
| Beta adapter | 250 | 0.070318 | 0.051159 | 1.7e-02 (Monte-Carlo; see the Beta note) |

The frozen mean backbone was bit-identical before and after training in every run.

### Held-out-family synthetic metrics

Averaged over the ambiguous, identifiable, and noisy conditions on held-out families and held-out parameter
ranges, 6 episodes per condition, support 128, `M = 6`.

| Metric | Direction | Context resampling | Beta | Frozen continuous | Adapter continuous | Full copy |
|---|:---:|---:|---:|---:|---:|---:|
| Energy distance | lower | 0.086013 | 0.065454 | 0.062642 | **0.060794** | 0.062409 |
| Mutual-information MAE (projected) | lower | 0.088549 | 0.058270 | 0.057820 | **0.057735** | 0.059085 |
| Mutual-information MAE (exact) | lower | 0.154162 | 0.123258 | 0.122808 | **0.122723** | 0.124073 |
| Epistemic-variance MAE (projected) | lower | 0.031030 | 0.019529 | 0.019035 | **0.018964** | 0.019601 |
| Covariance error | lower | 0.013256 | **0.007913** | 0.009080 | 0.008860 | 0.009830 |
| Joint query-vector NLL | lower | 6.331922 | **5.988700** | 6.075784 | 6.070929 | 6.061705 |
| Posterior sample coverage | higher | **0.694444** | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| Candidate-mass total variation | lower | 0.336806 | **0.307292** | 0.409722 | 0.409722 | 0.407986 |
| Deployed probability difference | lower | **0.000000** | **0.000000** | **0.000000** | **0.000000** | **0.000000** |
| Sample mean-preservation error | lower | 4.3e-08 | 1.9e-02 | 4.5e-08 | 4.7e-08 | 4.2e-08 |

Teacher mutual information on the same episodes is 0.054703 and the teacher safe scale is 0.778606.

### Uncertainty as support evidence increases (mean MI, ambiguous episodes, `eta = 0.05`)

| Support size | Context resampling | Beta | Frozen | Adapter | Full copy |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.0509 | 0.0123 | 0.0257 | 0.0114 | 0.0188 |
| 64 | 0.0508 | 0.0103 | 0.0188 | 0.0062 | 0.0153 |
| 128 | 0.0701 | 0.0175 | 0.0042 | 0.0034 | 0.0073 |
| 256 | 0.0502 | 0.0114 | 0.0055 | 0.0044 | 0.0088 |
| 512 | 0.0236 | 0.0088 | 0.0031 | 0.0027 | 0.0065 |

Every learned arm reduces mutual information as support evidence grows. The non-learned context-resampling
baseline does not.

### Label noise (candidates agree; all epistemic uncertainty should be zero)

| `eta` | Adapter: expected conditional entropy | Adapter: MI | Beta: entropy | Beta: MI | Full copy: entropy | Full copy: MI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.3549 | 0.0049 | 0.3465 | 0.0137 | 0.3536 | 0.0062 |
| 0.05 | 0.3784 | 0.0052 | 0.3729 | 0.0123 | 0.3768 | 0.0068 |
| 0.10 | 0.4543 | 0.0045 | 0.4459 | 0.0150 | 0.4518 | 0.0070 |
| 0.20 | 0.5902 | 0.0081 | 0.5852 | 0.0137 | 0.5861 | 0.0122 |
| 0.35 | 0.5687 | 0.0076 | 0.5636 | 0.0139 | 0.5650 | 0.0113 |

Expected conditional entropy rises with structured noise while epistemic mutual information stays flat, for
every learned arm.

### Random-label diagnostic (never a training batch)

| Arm | Expected conditional entropy (should be high) | Mutual information (should be low) |
|---|---:|---:|
| Context resampling | 0.4878 | 0.1274 |
| Beta | 0.6041 | 0.0112 |
| Frozen continuous | 0.6081 | 0.0071 |
| Adapter continuous | 0.6062 | 0.0090 |
| Full copy | 0.6001 | 0.0151 |

The learned arms separate aleatoric from epistemic uncertainty here; context resampling does not, because
subsampling a pure-noise context changes the prediction and that variation is scored as disagreement.

### Pilot gate values

| Gate | Value | Pilot verdict |
|---|---|---|
| 2 representation benefit | MI gain 0.15%, variance gain 0.37%, joint NLL change +0.08% | **Fail** (needs `>= 10%`) |
| 3 continuous-posterior benefit over Beta | improves energy distance (+7.1%), MI error (+0.9%), variance error (+2.9%); worse covariance (-12.0%) and joint NLL (-1.4%) | **Pass** (3 of 5) |
| 4 evidence behaviour | paired MI drop: frozen +0.0045, adapter -0.0016, full -0.0041, Beta -0.0026 | **Fail** for the adapter arm on paired episodes, though MI does fall monotonically with support size |
| 5 aleatoric/epistemic separation | see the label-noise table | **Pass** |

`select_risk_lambda` chose `lambda = 0` for the adapter arm, 0.5 for frozen, and 4 for Beta and the full
copy at this scale, which is itself a sign that the pilot MI signal is too weak to be worth much weight.

## TabArena evaluation (pending)

The harness evaluates every arm on the same 20 official binary tasks used by the slot trial, with identical
deterministic context rows (`random_state=0`, context size 1,024 where available), uncertainty built only
from labelled context rows, and scoring only on the untouched official test partition. Query labels never
enter model construction: the classifiers accept no query-label argument at all. Uncertainty sampling is
repeated with inference seeds 0, 1, and 2, and per-task metrics, macro averages, three-seed variability,
and task-level bootstrap confidence intervals are written out.

```bash
python -m tfmplayground.experiments.evaluate_continuous_tabarena \
    --task-ids <comma separated ids or a JSON file> \
    --output-dir results/continuous_bayesian/tabarena \
    --checkpoint adapter_continuous=<path> \
    --checkpoint frozen_continuous=<path> \
    --checkpoint full_continuous=<path> \
    --checkpoint beta_adapter=<path> \
    --slot-checkpoint <previous trial checkpoint> \
    --risk-lambda <selected on held-out synthetic episodes>
```

`lambda` is selected only from `{0, 0.25, 0.5, 1, 2, 4}` on held-out synthetic ordinary episodes by
`select_risk_lambda`, never using TabArena labels.

Results: **pending — the TabArena evaluation has not been run.**

## Acceptance gates

| # | Gate | Status |
|---:|---|---|
| 1 | Exact mean preservation, max difference `<= 1e-6` | **Pass** — deployed difference is exactly `0.0`; sample-mean error `<= 6e-8` for the continuous arms |
| 2 | Adapters beat the frozen encoder by `>= 10%` relative on MI or variance error, joint NLL not worse by more than 2% | **Fail in the CPU pilot** (0.15% / 0.37%); decide from the CREATE sweep |
| 3 | Adapter continuous improves at least two of five metrics over the Beta ablation | **Pass in the CPU pilot** (3 of 5); confirm on the CREATE sweep |
| 4 | Mutual information decreases as identifying support evidence is added | **Partial**: monotone in support size for every learned arm; the paired-prefix test is at the noise floor |
| 5 | Structured label noise raises expected conditional entropy without materially raising epistemic MI | **Pass in the CPU pilot** for every learned arm |
| 6 | TabArena usefulness: no AUROC drop `> 0.01`, no AURC rise `> 0.005`, and at least one improvement at those margins | Pending — the evaluation has not been run |
| 7 | Reliability: support permutation invariance, query permutation equivariance, finite gradients, deterministic checkpoint reload, CPU smoke tests, CUDA training test, no query-label leakage | **Pass on CPU**; the CUDA training test is present and skips without a GPU |

Gates 2 and 3 are computed automatically by `representation_benefit` and `continuous_posterior_benefit`;
gates 4 and 5 by `qualitative_gates`; gate 6 by `tabarena_gate`.

## Comparison with the slot trial

| Property | Slot trial | This trial |
|---|---|---|
| Posterior representation | `K` persistent learned slots with posterior weights | anonymous continuous latent draws, no `K` |
| Sample-to-target association | Hungarian matching in the loss | none; distributions compared directly |
| Uncertainty encoder | frozen backbone only | frozen / adapters / full copy, compared |
| Mean preservation | `2.98e-7` max difference | `0.0` deployed difference |
| Candidate counts in training | fixed `K` per run | `C` in {2, 4, 8, 16} per episode |
| Reported ESS | yes | deliberately not reported (equal-weight anonymous samples) |
| TabArena uncertainty result | failed the gate (MI AUROC 0.254 versus vanilla entropy 0.767) | pending |

## Interpretation

What the pilot already settles:

- **The construction is sound.** Mean preservation is exact by construction rather than by tuning: the
  deployed probability difference is `0.0` and the sample-mean error is `<= 6e-8` for every continuous arm,
  across every condition, support size, and noise level tested. Nothing in the uncertainty path can move
  the prediction, so this trial cannot repeat the slot trial's ambiguity about whether uncertainty helped
  or hurt accuracy: predictive metrics are identical to vanilla by construction.
- **Aleatoric and epistemic uncertainty separate correctly.** Structured label noise raises expected
  conditional entropy by roughly 0.22 nats from `eta = 0` to `eta = 0.20` while leaving mutual information
  flat, and random labels give high conditional entropy with near-zero mutual information.
- **The non-learned baseline is a genuinely different failure mode.** Context resampling produces the
  largest mutual information of any arm, but it is not epistemic: it is highest on pure-noise labels
  (0.1274) and it does not decrease as support evidence grows. Subsampling variance is being read as
  hypothesis disagreement. This is a useful control precisely because it fails in a legible way.

What the pilot does not settle, and why:

- **Gate 2 fails at pilot scale, but the pilot cannot distinguish "adapters do not help" from "adapters
  have not moved yet".** Adapters are zero-initialized, so the adapter arm *starts* exactly equal to the
  frozen arm; 400 steps at 3e-4 with a capped support range moves them by a fraction of a percent on every
  metric. The full trainable copy is likewise within a percent of frozen. The screening sweep (up to 1,500
  steps) and the three-seed final training (up to 5,000 steps) exist to answer this question, and gate 2
  should be read only from those runs.
- **All learned arms currently *under*-predict epistemic uncertainty.** Mean predicted MI is 0.006–0.012
  against a projected-teacher value of 0.055, and the ambiguous-episode MI never clears the 0.01 threshold
  in `qualitative_gates`. The direction of this failure — MI collapsing toward zero — is the same direction
  the slot trial failed in, and it is the single most important thing to watch in the full sweep. If MI
  stays collapsed at 5,000 steps across bottlenecks, latent dimensions, and learning rates, the limitation
  is the objective or the training distribution, not the encoder capacity.
- **Gate 4 is ambiguous at pilot scale.** Mutual information falls monotonically with support size for
  every learned arm, which is the behaviour the gate is meant to capture, but the paired support-prefix
  test — the sharper version, holding queries and candidates fixed — is at the noise floor
  (|drop| <= 0.005) for three of four arms. With MI itself around 0.006, a drop cannot be resolved. This
  gate becomes meaningful only once the arms predict non-trivial MI.
- **TabArena is entirely open.** No TabArena number in this report exists yet, and gate 6 is the gate the
  previous trial failed. The harness, the identical-context protocol, the seed repetition, and the
  bootstrap intervals are implemented and unit-tested, but nothing should be concluded about transfer
  until the evaluation is actually run against trained checkpoints.

Provisional reading of "freezing versus posterior capacity versus data": at pilot scale, capacity is
clearly *not* the binding constraint — frozen, adapter, and fully trainable arms are within a percent of
each other while all three sit far below the teacher's mutual information. That pattern points at the
training distribution or the uncertainty objective rather than at frozen pretrained representations. It is
stated here as a hypothesis to test with the full sweep, not as a conclusion: 400 CPU steps is too little
evidence to rule out that adapters simply had not yet departed from their zero initialization.

## Reproduction

```bash
python -m unittest tests.test_continuous_bayesian_uncertainty
python -m tfmplayground.experiments.train_continuous_bayesian --help
python -m tfmplayground.experiments.continuous_sweep --count
python -m tfmplayground.experiments.evaluate_continuous_synthetic --help
```

Large checkpoints and run directories stay ignored by git; compact Markdown and CSV summaries under
`results/continuous_bayesian/` are committed.

## Files

- `tfmplayground/models/continuous_posterior.py` — adapters, continuous posterior, Beta ablation,
  context-resampling baseline, checkpoint format
- `tfmplayground/experiments/continuous_episodes.py` — episode generator and function families
- `tfmplayground/experiments/train_continuous_bayesian.py` — teacher projection, objective, training loop
- `tfmplayground/experiments/continuous_sweep.py` — screening grid and selection
- `tfmplayground/experiments/evaluate_continuous_synthetic.py` — held-out synthetic evaluation and gates
- `tfmplayground/experiments/evaluate_continuous_tabarena.py` — TabArena evaluation harness
- `tfmplayground/continuous_interface.py` — scikit-learn interfaces
- `scripts/slurm/sweep_continuous_bayesian_a30.sbatch`,
  `scripts/slurm/train_continuous_bayesian_finalist_a30.sbatch`
- `tests/test_continuous_bayesian_uncertainty.py`
