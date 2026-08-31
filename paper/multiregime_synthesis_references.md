# Synthesising multiregime data: what the literature says, and what it says about ours

## Why this document exists

The multiregime prior in this repository has a measured detection ceiling of
about **0.54 AUC** -- essentially chance (see
[`slot_coherence_and_compatibility_results.md`](slot_coherence_and_compatibility_results.md)).
Every binding result was therefore scored against a target nothing could reach,
and "the model failed to separate regimes" was never distinguishable from
"nothing can". Before redesigning the prior by trial and error, this collects
what is already known about when a latent-regime construction is recoverable at
all.

The short version: our construction is the case the literature singles out as
**not identifiable**, and the standard fixes are known.

## What our design is, in the literature's terms

Huber ε-contamination with an unobserved, feature-independent component label.
A fraction of rows have their label drawn from a second rule; which rows is
chosen by `rng.choice` over indices, so the component label is independent of
`x`.

The instance-dependent label-noise literature addresses exactly this and is
blunt about it: `P(Ỹ | Y, X)` is **not identifiable in general**. Recovery needs
additional assumptions -- *anchor points* (instances known to belong to a class
with probability one), or structural constraints on the noise form. Neither is
present in our generator. The 0.54 ceiling is that result appearing as a number.

- [Identifiability of Label Noise Transition Matrix](https://arxiv.org/abs/2202.02016) --
  when the problem is and is not identifiable; anchor-point assumption.
- [Part-dependent Label Noise: Towards Instance-dependent Label Noise](https://arxiv.org/pdf/2006.07836) --
  decomposes instance-dependent noise into parts to make it tractable; notes the
  anchor-point requirement is strong.
- [Instance-dependent Label-noise Learning under a Structural Causal Model](https://arxiv.org/pdf/2109.02986) --
  causal structure as the extra assumption that buys identifiability.

## Why `regime_coherence` did not rescue it

Finite mixtures of regressions and mixtures of experts are identifiable up to
label permutation under **two** conditions: a covariate-dependent gate, *and*
**well-separated components** over non-degenerate covariates. Generic
non-identifiability appears in configurations lacking the second.

`regime_coherence` supplied the first. Gumbel-top-k against `exp(c · w·x)` is a
softmax gate `π(x)`, which is the mixture-of-experts construction. What was
never controlled is component separation: `f_A` and `f_B` are two *independent*
SCM draws, so how far apart they are is whatever chance delivers -- and at two
classes they collide on about half the rows, which is precisely the fraction of
contaminated rows that carry the tag while being observationally identical to a
clean row.

**We added the gate and left separation to luck.** That is why the measured
ceiling barely moved (0.540 at coherence 0, 0.507 at coherence 2.0).

It also explains why raising the class count worked: `1 - 1/C` is a crude proxy
for separation, and the measured optimum was C=3 (0.742 against 0.505 at C=2).
A continuous target is the clean version of the same lever -- components drawn
from a continuum cannot collide at all.

- [Mixtures of Experts Models](https://arxiv.org/pdf/1806.08200) -- survey;
  identifiability conditions and estimation.
- [On the identifiability of mixtures-of-experts](https://www.semanticscholar.org/paper/On-the-identifiability-of-mixtures-of-experts-Jiang-Tanner/453ba396d8bf4efae3b1c7ae282620992b46ae8d)
  (Jiang & Tanner) -- the identifiability-up-to-permutation result.
- [Mixture of Experts Provably Detect and Learn the Latent Cluster Structure](https://openreview.net/forum?id=2xmOEbpYv1) --
  gradient-based learning recovers latent clusters when each expert weakly
  recovers its own component; the mechanism this project was hoping for.
- [Finite Mixture of Regressions Model](https://www.emergentmind.com/topics/finite-mixture-of-regressions-model) --
  identifiability under separated clusters and non-degenerate covariates.

## Three principled constructions

### 1. Mixture of experts with explicit component separation

The standard generative model, and the only option whose identifiability
conditions are known in advance -- so the ceiling can be *reasoned about* rather
than measured after a sweep has already run.

Regime drawn from `π(x) = softmax(g(x))`; components parameterised by an
explicit **separation** knob instead of being two independent draws. The regime
stays genuinely latent for any single row while remaining recoverable in
aggregate, which is the property this project needs and the current design
lacks.

Nearest to what exists: `regime_coherence` already supplies `π(x)`. The missing
half is the separation parameter.

### 2. Repeated labels (Dawid-Skene)

Give each row labels from more than one regime. Given the true label the
annotators' responses are conditionally independent, and that assumption makes
both the confusion matrices and the label prior identifiable; pairwise
co-occurrence methods achieve it from second-order statistics alone, avoiding
the sample complexity of tensor methods.

**This is the strongest option**: it converts an unidentifiable per-instance
problem into an identifiable one by construction, rather than by making the
instance easier. The cost is a change to the episode format -- more than one
label column -- which touches every consumer of `ContinuousEpisode`.

- [Crowdsourcing via Pairwise Co-occurrences: Identifiability and Algorithms](https://proceedings.neurips.cc/paper/2019/file/c0e19ce0dbabbc0d17a4f8d4324cc8e3-Paper.pdf)
- [CROWDLAB: inferring consensus labels and quality scores from multiple annotators](https://arxiv.org/pdf/2210.06812)
- [Learning From Crowdsourced Noisy Labels](https://arxiv.org/pdf/2407.06902)

### 3. Structural assumptions

Anchor points, or a part-dependent decomposition of the noise. Cheapest to bolt
onto the current generator, weakest guarantees, and the literature itself
describes the assumptions as strong.

## The real-data question

A high ceiling makes a benchmark *measurable*, not *realistic*. Every lever that
raised ours moves away from real tables -- four features instead of two to
twelve, 15% contamination instead of 30%, three classes where real multiregime
problems are usually binary. None of it licenses a claim about real data.

The repository's own real-data path has the same defect: `evaluate_tabarena_small`
can build multiregime episodes on real OpenML tables, but assigns the regime with
`chooser.choice` over indices -- identical construction, so identical
unidentifiability, and its ceiling is likewise unmeasured. Its second regime is
also a synthetic MLP rather than a second real process.

**[TableShift](https://arxiv.org/html/2312.07577v2)** is the substrate that
answers this: a tabular distribution-shift benchmark of heterogeneous real
datasets across finance, policy, civic participation and medical diagnosis, with
genuine domain splits. The regimes there are real, which no amount of tuning our
prior can produce.

- [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/html/2312.07577v2)
- [On the Need for a Language Describing Distribution Shifts](https://papers.neurips.cc/paper_files/paper/2023/file/a134eaebd55b7406ff29cd75d5f1a622-Paper-Datasets_and_Benchmarks.pdf)

## Decision

Take **(1)** for the prior: the gate exists, so this is a separation parameter
plus the continuous-target option, and it makes the ceiling predictable instead
of discovered. Take **TableShift** as the real evaluation. Keep **(2)** in
reserve as the guaranteed-identifiable fallback if separation alone does not
clear the bar.

Measure the ceiling of any candidate design *before* training on it. That is the
methodological lesson of this whole line: a few CPU-minutes of
`detection_ceiling.py` would have preempted days of GPU spent on a task whose
signal was never reachable.
