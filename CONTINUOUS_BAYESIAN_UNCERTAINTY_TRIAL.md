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
  evaluation harness, two rounds of CPU pilot training and held-out-family synthetic evaluation
- Fixed after the first pilot: a dispersion-throttling defect in the safe-scale computation (see
  "Dispersion throttling" below). The first pilot's near-zero mutual information was misattributed to
  under-training in the first version of this report.
- **Not yet run: the CREATE Slurm screening sweep, the three-seed final training, and the TabArena
  evaluation.** The sweep is deliberately *not* submitted: the second pilot shows the learned dispersion
  gate is near-constant across conditions at three very different capacity budgets, which the sweep's
  bottleneck / latent-dimension / learning-rate axes cannot address. Those sections state the exact
  submission commands and are marked pending rather than filled with numbers.

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
  Note that the DeepSets pooling belongs to this head only: nanoTabPFN is attention-only and does no set
  pooling, and its datapoint attention has already conditioned every query embedding on the support rows
  before the head sees them. `c_D` is therefore an additional global summary, not the head's only route to
  support information.
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
c_qs      = r_qs - average_over_samples(r_qs)
u_qs      = c_qs / median_over_samples(|c_qs|)
v_qs      = deviation_clip * tanh(u_qs / deviation_clip)
d_qs      = v_qs - average_over_samples(v_qs)
b_q       = largest scale with mu_q + b_q * d_qs inside (0, 1) for every s
a_q       = g_q * b_q
p_qs      = mu_q + a_q * d_qs
```

The standardize-and-soft-clip step exists because `b_q` is set by the *largest* deviation: without it a
single outlier sample consumes the whole probability headroom and throttles the other samples. See
"Dispersion throttling" below. The scale is the median absolute deviation rather than the RMS on purpose,
since an outlier inflates the RMS and would leave the bulk in the linear part of `tanh`. `tanh` is
monotone, so the ordering — and therefore the direction and relative structure of the disagreement — is
preserved, and re-centring afterwards keeps the sample mean exactly zero. `deviation_clip = None` selects
the original un-clipped behaviour and is retained so that checkpoints written before the fix reload
exactly.

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

| Architecture | Grid | Base | With moment weight |
|---|---|---:|---:|
| adapter continuous | bottleneck {16, 32, 64} x latent {16, 32, 64} x lr {1e-4, 3e-4} | 18 | 36 |
| frozen continuous | latent {16, 32, 64} x lr {1e-4, 3e-4} | 6 | 12 |
| full uncertainty copy | latent {32, 64} x lr {1e-5, 3e-5} | 4 | 8 |
| Beta adapter | bottleneck {16, 32, 64} x lr {1e-4, 3e-4} | 6 | 12 |
| **Total** | | **34** | **68** |

Every base configuration is screened at moment weight 1 and 4. The moment weight multiplies the
mutual-information and variance loss weights only; weight 1 reproduces the declared weights exactly. It is
swept rather than asserted because those terms are numerically swamped by the joint query loss (~2.1
against ~0.03 and ~0.005 at the declared weights), and it is applied to all four architectures so that the
adapter-versus-frozen and adapter-versus-Beta gates stay like for like. At roughly 1.5 h per screening run
with eight concurrent tasks, the grid is about 13 h of wall clock.

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

The numbers below are from the **second** pilot, run after the dispersion-throttling fix described in the
next section. A `moment_weight = 4` variant of the adapter arm is included because the moment terms are
numerically swamped by the joint query loss.

> **Superseded.** This pilot used the region-based ambiguous/identifiable contrast, which the encoder
> probe later showed to be a shortcut. The episode generator has since changed, so these numbers are not
> comparable with the third pilot below. They are kept because the dispersion-throttling narrative depends
> on them.

Configuration for all arms: adapter bottleneck 32, latent dimension 32, `S = 32`, `max_support_size = 128`
(a compute cap; the CREATE sweep uses the full 32–512 range), seed 2402. Learning rate 3e-4 for the head
and adapter arms and 3e-5 for the fully trainable copy. Artifacts:
`results/continuous_bayesian/cpu_pilot_training.csv` and
`results/continuous_bayesian/cpu-pilot-synthetic/`.

### Training

| Arm | Best step | Validation selection loss (lower) | Predicted MI | Teacher MI | Shape ratio | Gate | Mean-preservation error |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter continuous | 400 | 0.066290 | 0.0015 | 0.0297 | 0.619 | 0.176 | 4.2e-08 |
| adapter continuous, moment weight 4 | 100 | 0.069290 | 0.0043 | 0.0297 | 0.625 | 0.299 | 4.5e-08 |
| frozen continuous | 300 | 0.066670 | 0.0019 | 0.0297 | 0.683 | 0.161 | 4.3e-08 |
| full copy continuous | 300 | 0.064760 | 0.0038 | 0.0297 | 0.601 | 0.320 | 5.3e-08 |
| Beta adapter | 250 | 0.070320 | 0.0119 | 0.0297 | 0.379 | n/a | 1.7e-02 (Monte-Carlo) |

The frozen mean backbone was bit-identical before and after training in every run. The Beta arm has no
dispersion gate, so its gate column is not applicable.

### Held-out-family synthetic metrics

Averaged over the ambiguous, identifiable, and noisy conditions on held-out families and held-out parameter
ranges, 6 episodes per condition, support 128, `M = 6`.

| Metric | Direction | Context resampling | Beta | Frozen | Adapter | Adapter, moment 4 | Full copy |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Energy distance | lower | 0.086013 | 0.065454 | **0.063045** | 0.063213 | 0.068440 | 0.064186 |
| Mutual-information MAE (projected) | lower | 0.088549 | 0.058270 | **0.055566** | 0.055732 | 0.058132 | 0.057025 |
| Mutual-information MAE (exact) | lower | 0.154162 | 0.123258 | **0.120554** | 0.120719 | 0.123120 | 0.122012 |
| Epistemic-variance MAE (projected) | lower | 0.031030 | 0.019529 | **0.018048** | 0.018114 | 0.019140 | 0.018709 |
| Covariance error | lower | 0.013256 | 0.007913 | **0.007771** | 0.007847 | 0.009179 | 0.008643 |
| Joint query-vector NLL | lower | 6.331922 | **5.988700** | 6.075528 | 6.074077 | 6.071064 | 6.068452 |
| Posterior sample coverage | higher | **0.694444** | 0.666667 | 0.666667 | 0.666667 | 0.666667 | 0.666667 |
| Candidate-mass total variation | lower | 0.336806 | **0.307292** | 0.402778 | 0.406250 | 0.411458 | 0.401042 |
| Deviation shape ratio | higher | 0.448050 | 0.419023 | **0.682041** | 0.619329 | 0.624695 | 0.597662 |
| Dispersion gate | n/a | 1.000000 | n/a | 0.166178 | 0.183961 | 0.319035 | 0.295989 |
| Deployed probability difference | lower | **0.000000** | **0.000000** | **0.000000** | **0.000000** | **0.000000** | **0.000000** |
| Sample mean-preservation error | lower | 0.000000 | 0.018942 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Teacher mutual information on the same episodes is 0.054703 and the teacher safe scale is 0.778606.

### The gate does not vary with the condition

This is the central result of the pilot. Per condition, for each learned arm:

| Arm | Gate: ambiguous / identifiable / noisy | MI: ambiguous / identifiable / noisy | Teacher MI: ambiguous / identifiable / noisy |
|---|---|---|---|
| adapter continuous | 0.186 / 0.188 / 0.179 | 0.0024 / 0.0034 / 0.0021 | 0.1641 / 0.0000 / 0.0000 |
| frozen continuous | 0.174 / 0.157 / 0.168 | 0.0024 / 0.0027 / 0.0023 | 0.1641 / 0.0000 / 0.0000 |
| full copy | 0.309 / 0.288 / 0.291 | 0.0050 / 0.0067 / 0.0052 | 0.1641 / 0.0000 / 0.0000 |
| adapter, moment 4 | 0.304 / 0.333 / 0.321 | 0.0061 / 0.0092 / 0.0072 | 0.1641 / 0.0000 / 0.0000 |
| context resampling | 1.000 (fixed) | 0.0730 / 0.0873 / 0.0785 | 0.1641 / 0.0000 / 0.0000 |

The gate is flat to within a few percent across conditions whose true epistemic content differs by 0.164
nats, and predicted MI is consistently *higher* on identifiable episodes (target 0) than on ambiguous ones
(target 0.164). Every arm is inverted, including the non-learned baseline.

### Uncertainty as support evidence increases (mean MI, ambiguous episodes, `eta = 0.05`)

| Support size | Context resampling | Beta | Frozen | Adapter | Adapter, moment 4 | Full copy |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.0509 | 0.0123 | 0.0089 | 0.0035 | 0.0086 | 0.0063 |
| 64 | 0.0508 | 0.0103 | 0.0121 | 0.0044 | 0.0091 | 0.0068 |
| 128 | 0.0701 | 0.0175 | 0.0010 | 0.0015 | 0.0047 | 0.0040 |
| 256 | 0.0502 | 0.0114 | 0.0017 | 0.0020 | 0.0064 | 0.0049 |
| 512 | 0.0236 | 0.0088 | 0.0009 | 0.0014 | 0.0040 | 0.0039 |

Every learned arm still reduces mutual information as support evidence grows; the non-learned baseline does
not. With MI at this magnitude, though, the trend is close to the noise floor.

### Label noise (candidates agree; all epistemic uncertainty should be zero)

| `eta` | Adapter: conditional entropy | Adapter: MI | Full copy: entropy | Full copy: MI | Beta: entropy | Beta: MI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.3576 | 0.0022 | 0.3559 | 0.0039 | 0.3465 | 0.0137 |
| 0.05 | 0.3815 | 0.0022 | 0.3790 | 0.0046 | 0.3729 | 0.0123 |
| 0.10 | 0.4572 | 0.0016 | 0.4542 | 0.0047 | 0.4459 | 0.0150 |
| 0.20 | 0.5955 | 0.0027 | 0.5914 | 0.0068 | 0.5852 | 0.0137 |
| 0.35 | 0.5737 | 0.0026 | 0.5696 | 0.0067 | 0.5636 | 0.0139 |

Expected conditional entropy rises by about 0.24 nats from `eta = 0` to `eta = 0.20` while epistemic mutual
information stays flat, for every learned arm.

### Random-label diagnostic (never a training batch)

| Arm | Expected conditional entropy (should be high) | Mutual information (should be low) |
|---|---:|---:|
| Context resampling | 0.4878 | 0.1274 |
| Beta | 0.6041 | 0.0112 |
| Frozen continuous | 0.6129 | 0.0023 |
| Adapter continuous | 0.6120 | 0.0033 |
| Adapter, moment 4 | 0.6052 | 0.0101 |
| Full copy | 0.6084 | 0.0068 |

The learned arms separate aleatoric from epistemic uncertainty here; context resampling does not, because
subsampling a pure-noise context changes the prediction and that variation is scored as disagreement.

### Pilot gate values

| Gate | Value | Pilot verdict |
|---|---|---|
| 2 representation benefit | adapter versus frozen: MI -0.30%, variance -0.37%, joint NLL +0.02% | **Fail** (needs `>= 10%`) |
| 3 continuous-posterior benefit over Beta | improves energy distance (+3.4%), MI error (+4.4%), variance error (+7.2%), covariance (+0.8%); worse joint NLL (-1.4%) | **Pass** (4 of 5) |
| 4 evidence behaviour | paired MI drop: frozen +0.0024, adapter +0.0002, full -0.0026, moment 4 -0.0026 | **Inconclusive**: at the noise floor because MI itself is ~0.003 |
| 5 aleatoric/epistemic separation | see the label-noise table | **Pass** |

The go/no-go declared before this pilot was that the sweep should be submitted only if ambiguous-episode MI
reached ~0.05 *and* exceeded MI on identifiable and noisy episodes. **Neither condition holds**, so the
CREATE sweep has not been submitted.

## Dispersion throttling: a construction defect found after the first pilot

The first pilot (400 steps, committed in `d0b1f88`) reported mutual information near zero for every learned
arm, and the first version of this report attributed that to under-training. **That reading was wrong.** A
per-condition diagnostic over the pilot checkpoints showed:

| Arm | Condition | Gate `g_q` | Max deviation | MI | Teacher MI |
|---|---|---:|---:|---:|---:|
| adapter | ambiguous | 0.438 | 0.209 | 0.0049 | 0.1775 |
| adapter | identifiable | 0.454 | 0.206 | 0.0062 | 0.0000 |
| adapter | noisy | 0.488 | 0.187 | 0.0060 | 0.0000 |
| frozen | ambiguous | 0.379 | 0.173 | 0.0063 | 0.1775 |
| full copy | ambiguous | 0.544 | 0.229 | 0.0091 | 0.1775 |

The gate was healthy (~0.44, near its 0.5 initialization) and the maximum deviation was large (~0.21), yet
MI of 0.005 implies a *typical* deviation of only ~0.05. The cause was `safe_dispersion_bound`: it derives
`b_q` from the maximum extent across samples, so one outlier sample consumed the whole headroom and
throttled the other 31. Dispersion was structurally under-delivered no matter how well the gate trained.

This mattered beyond the pilot. Gates 2 and 4 were untestable, because every arm carried the same throttle:
the sweep would have spent GPU time comparing hobbled models, and a null result would have been
uninterpretable.

The fix is the standardize-and-soft-clip step documented above. Measured on synthetic deviation shapes at
`mu = 0.5` with an open gate:

| Deviation shape | RMS/max, legacy | RMS/max, clipped | MI, legacy | MI, clipped |
|---|---:|---:|---:|---:|
| Heavy-tailed (one outlier) | 0.183 | 0.357 | 0.0228 | 0.0730 |
| Gaussian | 0.404 | 0.629 | 0.0929 | 0.2337 |
| Well-separated bimodal | 0.914 | 0.935 | 0.5197 | 0.5539 |

The clip roughly doubles the usable headroom for spiky shapes and leaves genuine multimodality essentially
untouched, which is the intended behaviour: only outliers are compressed. Mean preservation stays exact in
every case.

`deviation_shape_ratio` (deviation RMS divided by its maximum) is now reported per condition by both the
trainer and the synthetic evaluation, so a throttled shape is distinguishable from a collapsed gate without
writing an ad-hoc diagnostic again.

The second observation from the same diagnostic is *not* fixed by this change and remains the open
question: MI was higher on identifiable and noisy episodes (target 0) than on ambiguous ones (target
0.178), which is the slot trial's failure signature. That is a condition-discrimination problem, and it is
what the sweep exists to answer.

## Encoder probe and the region shortcut

The second pilot left one hypothesis standing: the uncertainty head cannot tell ambiguous episodes from
identifiable ones, so a near-constant small gate is its loss-minimizing output. That hypothesis is about
the *inputs* of the head, so it was tested directly rather than through another training run.

`tfmplayground/experiments/probe_uncertainty_encoder.py` trains a small supervised head on **frozen**
uncertainty representations to regress the projected-teacher mutual information and epistemic variance. No
posterior sampling, no bounded gate, and no energy distance are involved, so a design that cannot fit the
teacher here cannot learn it inside the full objective either. Three contexts are compared: the current
global `deepsets` pool, per-query `cross_attention` over the support rows, and `cross_attention_local`
which adds kNN label-agreement and neighbour-distance features. Training uses the training families,
early stopping uses a fresh *in-family* validation split, and scoring uses held-out families — the
in-family/held-out split separates "cannot represent the target" from "cannot transfer it".

A first version of this probe trained on a fixed 96-episode set and produced a misleading result:
cross-attention reached a training loss of 6.5e-05 against DeepSets' 9.5e-03 while scoring a held-out
condition AUC of 0.32, below chance. That was memorization of a small fixed set, an artifact of the probe
rather than a fact about the encoder, and it is why the reported protocol uses three splits with early
stopping.

| Context | Episode contrast | In-family MI R² | Held-out MI R² | In-family AUC | Held-out AUC |
|---|---|---:|---:|---:|---:|
| deepsets | region (old) | 0.263 | 0.012 | 0.662 | 0.423 |
| cross_attention | region (old) | 0.337 | −0.031 | 0.621 | 0.352 |
| cross_attention_local | region (old) | 0.324 | −0.017 | 0.594 | 0.367 |
| deepsets | evidence (new) | **−0.009** | −0.031 | 0.572 | 0.428 |
| cross_attention | evidence (new) | 0.180 | **0.090** | 0.547 | 0.464 |
| cross_attention_local | evidence (new) | 0.090 | −0.001 | 0.590 | 0.433 |

Under the original region-based contrast, no design transferred: held-out R² was at or below zero and the
held-out condition AUC was *below chance* for all three, meaning the learned mapping inverted on new
function families. That inversion is the same one the full pilots show, reproduced in a setting where
sampling, the gate, and the objective are all removed.

### The region shortcut

The generator used to draw ambiguous support rows from the low-disagreement region and identifiable
support rows from the high-disagreement region, so the conditions differed in support *difficulty* as well
as in evidence content. Difficulty is a family-specific surface cue, and a head that keys on it will
invert whenever a new family's geometry reverses the association.

`_select_rows` now takes a single `identifying_fraction`. Queries are always the highest-disagreement
rows, the support always comes from the same base region with the same size, and only the share of
genuinely disagreeing support rows varies: 0.0 for ambiguous, 0.25–1.0 for identifiable. The
non-identifying part of the support is forced to exact agreement, so the identifying rows are the only
evidence about which candidate is active. `sample_paired_episode` is now the same helper called at
fractions 0 and `f`, so its two arms share support size, queries, and candidates — a sharper test than the
old prefix-versus-extension pair, whose arms also differed in how much data they had.

Removing the shortcut is what the second half of the table measures, and it is decisive in an unwelcome
direction: **DeepSets' in-family R² collapses from 0.263 to −0.009.** Essentially all of its apparent
skill was the shortcut. Cross-attention retains real signal (0.180 in-family) and produces the only
positive held-out R² anywhere in either experiment (0.090), with its inversion largely gone (held-out AUC
0.352 → 0.464).

So the per-query context is a genuine improvement over the global pool, and it is now available as
`context_mode="cross_attention"` with `deepsets` kept as the default so existing checkpoints reload
unchanged. But it clears neither promotion floor (R² 0.30, AUC 0.85), so the encoder change alone does not
justify the CREATE sweep.

Two cautions on how much this result supports. First, the gain is measured against a baseline that the
same table shows to be worthless once the shortcut is gone (in-family R² −0.009), so "better than
DeepSets" is a low bar. Second, because nanoTabPFN's datapoint attention already conditions each query on
the support rows, this context is a seventh such attention layer rather than a newly added capability;
that is consistent with a small gain and it means the result should not be read as having located the
missing mechanism.

The hypothesis these numbers actually leave standing is that frozen target-column embeddings, trained to
predict the mean, carry little information about posterior width. The decisive test is the same probe with
the uncertainty backbone's **adapters trainable**, which asks whether adapting the representation makes
the target learnable across families. That is exactly the "is freezing the limiting factor" question this
trial is meant to answer, and it has not been run.

## Third pilot: the evidence contrast, and cross-attention in the full model

Same budget as before (400 steps, `max_support_size = 128`, seed 2402, bottleneck 32, latent 32, `S = 32`,
`moment_weight = 4`), now on the evidence-contrast generator. Artifacts:
`results/continuous_bayesian/cpu-pilot3-synthetic/`.

| Arm | Condition | Gate | MI | Teacher MI |
|---|---|---:|---:|---:|
| deepsets, adapters | ambiguous | 0.260 | **0.0053** | 0.1699 |
| deepsets, adapters | identifiable | 0.252 | 0.0043 | 0.0000 |
| deepsets, adapters | noisy | 0.256 | 0.0053 | 0.0000 |
| cross-attention, adapters | ambiguous | 0.066 | 0.0003 | 0.1699 |
| cross-attention, adapters | identifiable | 0.062 | 0.0003 | 0.0000 |
| cross-attention, frozen | ambiguous | 0.157 | 0.0021 | 0.1699 |
| cross-attention, frozen | identifiable | 0.159 | 0.0019 | 0.0000 |
| context resampling | ambiguous | 1.000 | 0.0629 | 0.1699 |
| context resampling | identifiable | 1.000 | 0.0781 | 0.0000 |

Three things to record, one good and two not.

**The inversion is gone.** For the first time a learned arm orders the conditions correctly: the DeepSets
adapter arm predicts more mutual information on ambiguous episodes (0.0053) than on identifiable ones
(0.0043), where every earlier pilot had the ordering backwards. Removing the region shortcut is what did
this, and it is the one clean win of this iteration. The non-learned context-resampling baseline stays
inverted (0.0629 versus 0.0781), as expected, since nothing about it changed.

**The magnitude did not move.** The gate is still flat — 0.260 / 0.252 / 0.256 across conditions whose
teacher mutual information differs by 0.17 nats. A correct *ordering* with a 2% gate difference is not a
usable epistemic signal.

**Cross-attention regressed the full model.** Its gate collapsed to 0.066 and its mutual information to
0.0003, an order of magnitude below the DeepSets arm it was meant to improve on. This is the opposite of
the probe result and is the clearest evidence for the correction noted above: because nanoTabPFN's
datapoint attention already conditions each query on the support rows, the extra attention layer adds
parameters without adding information, and the extra parameters make collapse easier. `deepsets` therefore
stays the default, and `cross_attention` is retained as a documented option that lost in the only test that
matters.

Every arm early-stopped at step 50, meaning validation *worsened* from there on. Combined with the flat
gate, that says the objective is actively driving these models toward zero dispersion rather than failing
to reach it.

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
| 2 | Adapters beat the frozen encoder by `>= 10%` relative on MI or variance error, joint NLL not worse by more than 2% | **Fail in the CPU pilot** (-0.30% / -0.37%), and the fully trainable copy behaves the same, so this is not a capacity limit |
| 3 | Adapter continuous improves at least two of five metrics over the Beta ablation | **Pass in the CPU pilot** (4 of 5) |
| 4 | Mutual information decreases as identifying support evidence is added | **Inconclusive**: monotone in support size for every learned arm, but the paired-prefix test is at the noise floor while MI is ~0.003 |
| 5 | Structured label noise raises expected conditional entropy without materially raising epistemic MI | **Pass in the CPU pilot** for every learned arm, but trivially so while the epistemic output is near-constant |
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
  conditional entropy by roughly 0.24 nats from `eta = 0` to `eta = 0.20` while leaving mutual information
  flat, and random labels give high conditional entropy with near-zero mutual information. This gate is
  passed for a weak reason, though: a model whose epistemic output is near-constant passes it trivially.
- **The non-learned baseline is a genuinely different failure mode.** Context resampling produces the
  largest mutual information of any arm, but it is not epistemic: it is highest on pure-noise labels
  (0.1274) and it does not decrease as support evidence grows. Subsampling variance is being read as
  hypothesis disagreement. This is a useful control precisely because it fails in a legible way.

What the second pilot newly settles, and what it rules out:

- **The dispersion parameterization is no longer the limitation.** The shape ratio went from ~0.18 to
  0.60–0.68, so the typical posterior sample now gets most of the available probability headroom rather
  than being throttled by one outlier. This removes what was previously a confound on every other
  conclusion.
- **With the throttle removed, training drove the gate down instead.** The gate fell from ~0.44 in the
  first pilot to 0.16–0.32, and mean predicted MI is 0.002–0.007 against a projected-teacher value of
  0.055. Fixing the parameterization did not raise MI; it relocated the collapse from the sample shape to
  the learned gate.
- **The gate is essentially constant across conditions.** This is the diagnostic that matters. For the
  adapter arm the gate is 0.186 / 0.188 / 0.179 on ambiguous / identifiable / noisy episodes, whose true
  epistemic content is 0.164 / 0.000 / 0.000 nats. The model is not modulating dispersion by ambiguity at
  all; it emits one near-constant spread everywhere and picks a small value.
- **A near-constant small gate is the loss-minimizing response to not being able to discriminate.** With
  45% of training episodes (25% identifiable plus 20% noisy) having a teacher MI of exactly zero, and the
  remainder having modest projected targets, a model that cannot tell the conditions apart minimizes both
  the moment MSEs and the energy distance by predicting close to the average target, which is near zero.
  Emitting spread in the *wrong* direction is scored worse than emitting none.
- **Capacity is not the binding constraint.** The fully trainable uncertainty copy behaves like the frozen
  encoder (gate 0.309 / 0.288 / 0.291, equally flat), and adapters versus frozen is now a -0.30% difference
  on MI error. Three very different capacity budgets produce the same flat, condition-independent gate.
- **Loss weighting is not the binding constraint either, though it is not neutral.** `moment_weight = 4`
  raises the gate (0.30 versus 0.19) and MI (0.006 versus 0.002), which is the predicted direction, but the
  gate stays just as flat across conditions (0.304 / 0.333 / 0.321). More weight scales the constant; it
  does not make it responsive.
- **Every arm, including the non-learned baseline, is inverted.** Predicted MI is consistently *higher* on
  identifiable episodes (target 0) than on ambiguous ones (target 0.164). This is the slot trial's failure
  signature reproduced in a slot-free model, which is evidence that the failure was never about slots.

Reading of "freezing versus posterior capacity versus data": the evidence points away from both freezing
and posterior capacity, and at the **uncertainty signal carried by the frozen representation**.

One clarification matters for interpreting this, because an earlier version of this section got the
mechanism wrong. nanoTabPFN itself contains no set pooling: it is attention-only, and its datapoint
attention already has every query row attend to the labelled support rows, once per layer for six layers.
The query target embedding handed to the uncertainty head is therefore *already* a support-conditioned
representation. The `DeepSetsSupportEncoder` is part of this trial's head, not part of the backbone, and
its role is to summarize the support set *again* into a single global vector.

That reframes the problem. The head does not lack access to query-relevant support evidence — the
backbone supplies it. What the head adds is a lossy global bottleneck on top, and what the frozen
representation lacks is not locality but *posterior width*: the target-column embeddings are optimized to
produce the predictive mean, and nothing in the pretraining objective requires them to encode how much
plausible functions consistent with the support disagree at a query.

This also right-sizes the per-query `cross_attention` context introduced below. It is a seventh
query-to-support attention layer stacked on six existing ones, not a missing mechanism being restored, and
its held-out gain is correspondingly modest.

What is still open:

- **The sweep has not been submitted**, per the pre-declared go/no-go. Running 68 GPU configurations to
  vary bottleneck, latent dimension, and learning rate would not address a failure that is invariant to
  capacity across three very different budgets.
- **Gate 4 remains inconclusive.** MI falls with support size for every learned arm, but the sharper
  paired-prefix test is at the noise floor while MI itself is ~0.003.
- **TabArena is entirely open.** No TabArena number in this report exists yet, and gate 6 is the gate the
  previous trial failed. The harness is implemented and unit-tested, but nothing should be concluded about
  transfer until it is run against checkpoints that predict non-trivial MI — which none of these do.

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
