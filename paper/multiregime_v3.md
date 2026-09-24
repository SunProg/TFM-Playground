# Multi-regime prior v3: related mechanisms and observable evidence

Date: 2026-09-10 (design discussed on 2026-09-09)

## Decision

Build a hierarchical episode prior with shared predictive structure, sparse regime
perturbations, probabilistic outcomes, and multiple assignment families. Preserve
the original TabICL `mix_scm` prior explicitly. Evaluate plain nanoTabPFN and an
unsupervised slot model on the same episodes and initialization of their common
backbone. Treat prediction, routing, and mechanism recovery as separate outcomes.

The purpose of the first run is a controlled synthetic pilot. It cannot establish
real-world transfer; that requires a separate, held-out real-data evaluation.

## Generative model

For each task, sample one covariate process and one shared outcome mechanism.
Regimes use the same feature schema, with related outcome mechanisms:

$$f_k(x)=f_0(x)+\alpha\Delta_k(x),\qquad
p_k(x)=\sigma(f_k(x)),\qquad Y_i\mid X_i,Z_i=k\sim\mathrm{Bernoulli}(p_k(X_i)).$$

Sample mechanisms and calibration statistics before the support/query rows. Use
an independent calibration pool for scale normalization. Query batch composition
must not change the target function. Perturb a small subset of coefficients,
interactions, or nonlinear terms; sample separation independently from gate
strength and regime prevalence. Include near-zero and zero separation.

The first implementation uses a compact, explicitly sampled nonlinear SCM for
v3 covariates and a mixture of linear, additive, interaction, and threshold terms
for related outcome mechanisms. The original TabICL `mix_scm` stream is replayed
without replacing its targets. The v3 generator is not claimed to reproduce the
full TabPFN/TabICL prior.

## Episode families

| Family | Construction | Primary question |
|---|---|---|
| Original prior | Unmodified TabICL `mix_scm` episode | Is ordinary ICL retained? |
| Shared rule | Feature-dependent groups, identical outcome mechanism | Does the model avoid unnecessary specialization? |
| Soft gate | Sample Z from a nonlinear softmax gate on X | Does routing account for uncertainty? |
| Independent rows | Each row draws Z independently with fixed weights | Does the model learn the observable mixture? |
| Persistent groups | Multiple observations share a group-level Z | Can support evidence identify a mechanism relevant to a query? |

Persistent groups represent batches, sites, sessions, or entities. An episode-local
group code is visible to both models; it contains group membership, not regime
identity. Several groups can share one regime. Codes and group-to-regime mappings
are randomized per episode. V3 appends matched nuisance group codes in all other
families so the presence of code columns does not reveal the family. Regime tags,
component probabilities, and generator parameters remain outside model inputs.

## Identifiability and interpretation

For independent rows with one binary response per observation:

$$P(Y=1\mid x)=\sum_k\pi_k(x)p_k(x).$$

Unrestricted component functions are not generally identified by this marginal.
In particular, equal weights and opposite logits yield
`0.5 * sigmoid(a) + 0.5 * sigmoid(-a) = 0.5`, regardless of separation.
Known mechanisms can help assign labeled support rows, but a fresh independent
query regime remains a random draw. This family tests marginalization and
calibration; perfect query regime recovery is not a valid target.

A deterministic gate also defines an ordinary piecewise function. Predictive
improvement by itself does not establish recovery of the intended latent
partition. Persistent groups supply joint evidence, but do not by themselves
guarantee identifiability of arbitrary mechanisms.

With known generator parameters, support responsibilities are proportional to
`pi_k(x) * p_k(y | x)`. For persistent groups, multiply likelihoods of all visible
support observations in the group, in log space, and normalize. Query oracle
predictions use that posterior without reading query labels. An oracle with
the true query regime is a separate, privileged reference, not the attainable
target for a latent-input model.

Weak learned detectors provide baselines, not theoretical ceilings. ARI, purity,
and Hungarian matching are secondary diagnostics and need suitable nulls.
Ambiguous assignments must not be interpreted as architecture failure.

## Pilot composition and curriculum

Use 50% original-prior replay and 50% v3 episodes as the initial fixed-mixture
hypothesis. Within the v3 share use 20% shared-rule, 30% soft-gate, 20% independent
rows, and 30% persistent groups. Non-shared families draw K in {2,3,4}, biased
toward 2; shared-rule episodes have no predictive regime differences even if
their diagnostic grouping has K > 1. These weights are experimental choices,
not estimates of real-world regime frequencies.

Compare a fixed mixture with a curriculum that uses only original episodes for
the first 10% of steps, ramps v3 probability to 50% by 40%, then holds it at 50%.
Do not simultaneously change the ambiguity schedule in this first pilot: doing
so would confound the curriculum comparison. Randomize separation, imbalance,
and soft-gate strength throughout the v3 part of either arm.

Retaining analytic matched K=1 tasks is not a substitute for original-prior
replay: the existing analytic K=1 family is much narrower than `mix_scm`.

## Matched experiment

Run six cells per seed: {plain nanoTabPFN, slot head} crossed with
{original-only, fixed v3 mixture, v3 curriculum}. Shared backbones start from the
same initial state. Use the same task seeds and samples for architecture pairs,
separate task RNG from model RNG, and keep support/query counts, optimizer
updates, batch size, learning rate, and evaluation episodes matched. Record the
extra parameters and runtime of the slot model rather than claiming equal FLOPs.

Initial pilot: one seed (2402), 1,000 updates, batch size 4, support 128, query 32,
3 layers, width 96, 4 heads, hidden width 384, 4 slots. AdamW at 1e-4 with 100-step
warmup and cosine decay; evaluate every 250 updates. This is a bounded pilot,
not the final multi-seed training budget. Use the existing unsupervised slot head
without changing the ongoing reconstruction-routing implementation.

Generate a fixed, independent evaluation bank for every family. Report log loss,
Brier score, accuracy, calibration error, parameter-aware marginal oracle loss,
privileged true-regime oracle loss, and per-regime performance. Keep family
results separate. For grouped tasks additionally evaluate a control with query
group codes scrambled: improvement from correct grouping should be measurable.
Store per-episode records so paired uncertainty can be computed across episodes.
Do not bootstrap individual rows as if grouped observations were independent.

## Execution checks

Before GPU training, verify reproducibility; finite probabilities; no query-label
access in model inputs or oracle responsibilities; invariance of support and
mechanisms to query-count changes; persistent group assignments; the opposing
logit cancellation example; and evidence accumulation for a known-mechanism
group oracle. Run finite forward/backward checks for both models. Existing v2
benchmark gates remain intact; v3 records its own implementation checks and
oracle headroom rather than misusing a v2 gate for a different generator.

Training jobs must save full configs, source hashes, actual episode composition,
initial/final validation, checkpoints, and machine-readable results. Use an
isolated cluster source snapshot so unrelated active experiments are unaffected.

## After the pilot

If the implementation checks pass, inspect whether each model improves on each
family, whether grouping information is used, and whether original-prior
performance is retained. Scaling requires replicated seeds and a real-data
development collection. Reserve final datasets for final testing.

Real-world extensions include covariate shifts with an invariant target rule,
correlated mixed-type features, class imbalance, informative missingness,
measurement changes, and support/query population shifts. Shared feature schema
does not imply identical feature distributions across regimes. TableShift is a
candidate domain-shift benchmark; its domain labels are not proof of distinct
outcome mechanisms. No synthetic pilot result establishes real-world suitability.

## References

- [Mixtures of Experts Models](https://arxiv.org/abs/1806.08200)
- [TabPFN](https://www.nature.com/articles/s41586-024-08328-6)
- [MITRA: Mixed Synthetic Priors](https://proceedings.neurips.cc/paper_files/paper/2025/file/177d68f4adef163b7b123b5c5adb3c60-Paper-Conference.pdf)
- [TableShift](https://arxiv.org/abs/2312.07577)

## Run record

Implemented:

- `tfmplayground/experiments/multiregime_v3.py`: v3 families, original-prior replay,
  randomized group codes, and parameter-aware oracle responsibilities.
- `tfmplayground/experiments/pretrain_multiregime_v3.py`: six-cell matched pilot,
  source-bound preflight, provenance, checkpoints, grouping controls, and
  support/query ARI with permutation nulls and oracle references.
- `scripts/report_multiregime_v3.py`: paired comparisons and episode-bootstrap
  intervals, with stream/backbone/evaluation matching checks.
- `scripts/slurm/pretrain_multiregime_v3_pilot_a30.sbatch`: bounded CREATE launch.

Local verification on 2026-09-10: 9 targeted tests pass, Ruff passes, launcher
passes `bash -n`, and the full pilot dimensions pass finite forward/backward
checks for both models. Plain has 486,242 parameters; slot has 756,869. Common
backbone hashes match. The gate is recorded at
`results/multiregime_v3/preflight-20260910/execution_gate.json`.

On the 24 fixed evaluation episodes per family, the persistent-group oracle's
mean log loss is 0.4652 using support-group evidence, versus 0.5575 without it;
the privileged true-regime reference is 0.4508. These are known-mechanism
references, not trained-model results or evidence of real-world transfer.

Reproduce local verification:

```bash
.venv/bin/python -m unittest tests.test_multiregime_v3
.venv/bin/python -m tfmplayground.experiments.pretrain_multiregime_v3 \
  --preflight --device cpu --output results/multiregime_v3/new-preflight
```

The launcher runs its own preflight on each allocated GPU, then trains the cell.
No training job runs if its exact source/configuration fails the v3 gate. Cluster
submission IDs and snapshot checksum are recorded separately in
`paper/multiregime_v3_submission.json` after submission.

Submitted to CREATE on 2026-09-10: training array **37097926** (six cells,
at most two concurrent GPUs), followed by report job **37097927**. Both the
uploaded archive checksum and extracted Python-source hash match the local
verified source. The snapshot is `submissions/multiregime-v3-20260910` under the
existing cluster project; results are written to `runs/37097926` inside it.
Each GPU cell must pass its own preflight before training starts. These jobs
all completed with exit code 0. All six GPU preflights passed. Final metrics,
provenance, and the paired report have been retrieved locally.

## Completed pilot results

All cells completed 1,000 updates (4,000 training episodes) with seed 2402.
Architecture pairs have identical initial backbones, training-stream hashes,
and evaluation banks. All 13 report matching checks pass.

Query log loss (lower is better):

| Model | Training prior | Original | Shared rule | Soft gate | Independent | Persistent |
|---|---|---:|---:|---:|---:|---:|
| Plain | Original only | 0.4748 | 0.6667 | 0.6705 | 0.6888 | 0.6747 |
| Slot | Original only | 0.4761 | 0.6655 | 0.6704 | 0.6889 | 0.6737 |
| Plain | Fixed v3 mixture | 0.4764 | 0.6652 | 0.6698 | 0.6875 | 0.6735 |
| Slot | Fixed v3 mixture | 0.4783 | 0.6644 | 0.6697 | 0.6873 | 0.6730 |
| Plain | v3 curriculum | 0.4768 | 0.6646 | 0.6697 | 0.6873 | 0.6733 |
| Slot | v3 curriculum | 0.4781 | 0.6641 | 0.6697 | 0.6875 | 0.6727 |

The fixed mixture actually contained 2,005 original and 1,995 v3 episodes per
model; the curriculum contained 2,483 original and 1,517 v3 episodes. This
comparison matches optimizer updates, not total v3 exposure, and cannot isolate
ordering from exposure volume. Family-level original-vs-v3 gains are mostly
around 0.001-0.002 NLL, with a similarly small cost on the original prior. Most
paired prior-change intervals overlap zero. These are episode-bootstrap
intervals for one trained seed, not uncertainty over training seeds; the many
comparisons have no multiple-testing adjustment.

There is little evidence of useful regime discovery at this budget. For the
slot/curriculum model on persistent groups:

- NLL: 0.6727; known-mechanism group oracle: 0.4652.
- Scrambling query group codes worsens NLL by only 0.000137.
- Normalized slot gate entropy: 0.9991 (one is uniform).
- Support ARI: 0.0260, permutation null -0.0021, oracle reference 0.7401.
- Query ARI: 0.1134, permutation null 0.1218, oracle reference 0.7484.

The nonzero query null illustrates why raw ARI means also need care: episodes
with a single occupied true regime and a single occupied predicted slot can
score one even after tag permutations. The group oracle establishes available
signal when mechanisms are known; it does not prove a model can infer those
mechanisms from this small support set or with this short pretraining budget.

**Decision after this pilot:** keep v3 as a verified experimental prior, but do
not claim real-world suitability or a successful slot-discovery result. Before a
large scale-up, use a simpler positive control and longer matched runs to
separate finite-support difficulty, training-budget limitations, and architectural
limitations; replicate seeds before treating small differences as reliable.

Full results: [paired report](../results/multiregime_v3/pilot-37097926/report.md),
[machine-readable summary](../results/multiregime_v3/pilot-37097926/summary.json),
and `cell-*/evaluation.jsonl` in the same directory. Checkpoints remain in the
cluster snapshot at `runs/37097926/cell-*/checkpoint.pth`.
