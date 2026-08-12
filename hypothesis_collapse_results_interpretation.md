# nanoTabPFN Hypothesis-Coherence Experiment: Interpretation Guide

## Purpose

This experiment tests whether nanoTabPFN's predictions for several related queries behave like one coherent posterior over a shared latent task.

The central distinction is:

- **Coherent Bayesian averaging:** uncertainty is over complete task hypotheses. If task A is true, all query labels follow task A; if task B is true, all follow task B.
- **Independent predictive averaging:** every query has the correct marginal probability, but the queries are treated as if their labels were independent. This assigns probability to combinations that no valid task can produce.

The interactive result viewer is [hypothesis_collapse_results.html](hypothesis_collapse_results.html). The raw run is stored under [`runs/hypothesis_collapse/20260807-150738/`](runs/hypothesis_collapse/20260807-150738/).

## Synthetic experiment

### Latent tasks

Each dataset has an unobserved task identity

$$
z \in \{A,B\}, \qquad P(z=A)=P(z=B)=0.5.
$$

The two tasks agree on the common support region, so those examples cannot reveal which task generated the dataset. They disagree in the positive query region:

- Task A gives label `0` to every query.
- Task B gives label `1` to every query.

For four queries, the only valid label vectors are therefore `0000` and `1111`. Mixed vectors such as `0011` or `1010` are incoherent because they combine parts of two mutually exclusive tasks.

### Common support rows

Every trial contains 16 common support rows, balanced between the two classes:

- Eight feature values are sampled from $[-2.0,-1.25]$ and labelled `1`.
- Eight feature values are sampled from $[-0.75,-0.25]$ and labelled `0`.

Both classes are deliberately present. This avoids confusing hypothesis coherence with the separate problem of predicting a class that was never observed in the support set.

### Disambiguating evidence

The variable $r$ is the number of additional support rows in the positive region, where tasks A and B disagree. The experiment uses

$$
r \in \{0,1,2,4,8\}.
$$

Each disambiguating label is flipped with probability 0.1. This noise applies only to support evidence; the query-label vectors remain deterministically all-zero or all-one.

Support-only noise creates a gradual posterior update:

- At $r=0$, the exact posterior is 50% task A and 50% task B.
- With a few observations, one task becomes more probable while both retain posterior mass.
- With consistent accumulated evidence, the posterior concentrates strongly on one task.

Without this noise, a single disambiguating row would identify the task perfectly, leaving no gradual transition to study.

### Query points

The number of linked queries is

$$
m \in \{2,3,4\}.
$$

Their feature values are evenly spaced from 0.25 to 1.0. For a given $m$, there are $2^m$ possible binary label vectors but only two coherent vectors: all zeros and all ones.

## How the implied joint distribution is recovered

nanoTabPFN directly produces a separate probability distribution for each test row, not a joint distribution over all test labels. The experiment constructs an implied joint using the probability chain rule.

For two queries:

$$
\hat p(y_1,y_2\mid D)
=
\hat p(y_1\mid D)
\hat p(y_2\mid D,y_1).
$$

The second factor is obtained by temporarily adding each hypothetical value of $y_1$ to the support set and asking nanoTabPFN to predict $y_2$. For $m$ queries, the process branches over every possible prefix:

$$
\hat p(\mathbf y\mid D)
=
\prod_{j=1}^{m}
\hat p(y_j\mid D,y_1,\ldots,y_{j-1}).
$$

This is exact chain-rule enumeration of nanoTabPFN's conditionals. It does not modify or retrain the model.

The calculation is performed twice:

- **Canonical order:** first query to last query.
- **Reverse order:** last query to first query.

If all conditionals belong to one coherent joint distribution, both factorizations should give approximately the same answer.

## Models and baselines

| Name | Meaning | Expected behaviour |
|---|---|---|
| Exact Bayes | Uses the known task prior and support-noise likelihood to calculate the true posterior over A and B. | Places probability only on the two coherent label vectors and is order invariant. |
| Independent oracle | Uses the exact Bayes marginal probability for every query but multiplies those marginals independently. | Has perfect marginals but spreads probability over impossible mixed vectors while ambiguity remains. |
| nanoTabPFN | Official pretrained nanoTabPFN classifier, evaluated through conditional chain-rule enumeration. | Reveals whether the learned predictor preserves and updates shared task hypotheses coherently. |

The independent oracle is a diagnostic baseline, not a realistic competitor. It proves that perfect per-query calibration does not guarantee a correct joint prediction.

## Glossary of terms

| Term | Explanation |
|---|---|
| Support set, $D$ | The labelled examples supplied to the model as in-context training data. |
| Query | An unlabelled feature vector whose label the model must predict. |
| Latent task, $z$ | The unobserved data-generating rule shared by the support rows and all queries in one trial. |
| Hypothesis or mode | One possible latent task, such as A or B. |
| Ambiguity | A situation where the observed support set is compatible with more than one latent task. |
| Posterior | The probability assigned to each latent task after observing the support set. |
| Posterior concentration | The desirable movement of probability toward one task when evidence supports it. |
| Hypothesis collapse | Premature loss of distinct task hypotheses while the data remain ambiguous. |
| Marginal prediction | The probability distribution for one query considered alone, such as $p(y_j=1\mid D)$. |
| Joint prediction | The probability distribution over a complete vector of query labels, such as $p(y_1,y_2,y_3,y_4\mid D)$. |
| Conditional prediction | A prediction made after assuming that one or more earlier query labels have particular values. |
| Chain-rule joint | A joint distribution assembled by multiplying sequential conditional probabilities. |
| Coherent label vector | A complete query-label vector generated by at least one valid latent task. Here these are only the all-zero and all-one vectors. |
| Incoherent label vector | A mixed vector that cannot be generated by either task, such as `0101`. |
| Canonical order | Conditioning on queries from first to last. |
| Reverse order | Conditioning on queries from last to first. |
| Trial | One independently generated support dataset and latent task realization. |
| Evidence count, $r$ | The number of support rows sampled from the region where tasks A and B disagree. |
| Query count, $m$ | The number of linked queries in the joint prediction. |

## Metrics

All reported divergence metrics use natural logarithms and are lower-is-better.

### Marginal cross-entropy

Marginal cross-entropy measures how well the predicted probability for each individual query matches its exact Bayesian marginal:

$$
H(p,q)=-\sum_y p(y)\log q(y).
$$

It is averaged over queries. Cross-entropy is not zero even for the exact predictor when the true distribution has uncertainty; its optimum equals the entropy of the exact distribution. It is therefore most useful for comparing models under the same condition.

### Marginal Jensen-Shannon divergence

Marginal JS divergence measures the discrepancy between the exact and predicted distribution for each query, averaged over queries:

$$
\operatorname{JS}(p,q)
=
\tfrac12\operatorname{KL}(p\|m)
+
\tfrac12\operatorname{KL}(q\|m),
\qquad
m=\tfrac12(p+q).
$$

It is zero when the distributions match and is bounded above by $\ln 2\approx0.693$. A low marginal JS value means individual query probabilities are good; it says nothing by itself about their dependence.

### Joint cross-entropy

Joint cross-entropy applies the same cross-entropy calculation to all $2^m$ complete label vectors. It penalizes assigning insufficient probability to the coherent vectors in proportion to their exact Bayesian posterior weights.

### Joint Jensen-Shannon divergence

Joint JS divergence compares the complete exact and predicted joint distributions. It detects errors invisible to marginal metrics, including probability assigned to impossible cross-mode combinations.

The signature of concern is:

$$
\text{low marginal divergence}
\quad+\quad
\text{high joint divergence}.
$$

### Incoherent mass

Incoherent mass is the total predicted probability assigned to every invalid mixed vector:

$$
C_{\mathrm{incoherent}}
=
1-\hat p(00\ldots0)-\hat p(11\ldots1).
$$

Interpretation:

- `0` means all probability is on coherent task-level outcomes.
- `0.30` means 30% of the model's probability is assigned to label combinations that neither task can generate.

At $r=0$, the independent oracle has incoherent mass

$$
1-2^{1-m},
$$

which equals 0.50, 0.75, and 0.875 for two, three, and four queries respectively.

### Order inconsistency

Order inconsistency is the JS divergence between the canonical and reverse chain-rule joints:

$$
C_{\mathrm{order}}
=
\operatorname{JS}
\left(
\hat p_{1\rightarrow m},
\hat p_{m\rightarrow1}
\right).
$$

A value near zero means the two factorizations agree. A large value means nanoTabPFN's conditionals do not behave like conditionals from one order-invariant joint distribution.

Order inconsistency is related to, but distinct from, hypothesis collapse. A model may preserve coherent modes in one ordering while producing a different joint in another ordering.

### Confidence intervals

`summary.csv` reports the mean, sample standard deviation, standard error, and a 95% normal-approximation interval across the 32 trials:

$$
\bar x \pm 1.96\frac{s}{\sqrt{n}}.
$$

Because these are unbounded normal approximations, a lower endpoint can be slightly negative even though the underlying metric cannot be negative. This is a presentation artifact, not a negative divergence or probability.

## Run settings

| Setting | Value |
|---|---:|
| Random seed | 2402 |
| Trials per query/evidence configuration | 32 |
| Query counts | 2, 3, 4 |
| Evidence counts | 0, 1, 2, 4, 8 |
| Common support rows | 16 |
| Evidence-label flip probability | 0.10 |
| Models | Exact Bayes, independent oracle, nanoTabPFN |
| Device | CPU |
| Memory chunks | 8 |
| nanoTabPFN layers | 6 |
| Embedding size | 192 |
| Attention heads | 6 |
| MLP hidden size | 768 |
| Checkpoint outputs | 10, with the first two retained for this binary experiment |
| Checkpoint SHA-256 | `0458de6d75f3e02c72735e6e61d4cd9a0141041a4d09ce2f7a695b2d8d6eb0fc` |

Each query-count/evidence-count configuration receives a separately generated batch of 32 trials from one deterministic seeded random-number stream. Results across configurations are reproducible but are not paired trial by trial.

## Result interpretation

### 1. nanoTabPFN preserves the two modes when no disambiguating evidence is present

At $r=0$, nanoTabPFN's canonical joint is close to the exact two-mode posterior:

| Queries | Marginal JS | Joint JS | Incoherent mass |
|---:|---:|---:|---:|
| 2 | 0.0092 | 0.0131 | 0.0183 |
| 3 | 0.0103 | 0.0139 | 0.0156 |
| 4 | 0.0061 | 0.0080 | 0.0144 |

For four queries, the independent oracle assigns 87.5% of its mass to impossible mixed vectors, whereas nanoTabPFN assigns only about 1.4%. The representative joint prediction places almost all probability on `0000` and `1111`.

This is evidence against the simplest hypothesis-collapse story. The pretrained model can represent unresolved task-level alternatives even though its hidden state is deterministic.

### 2. The strongest failure appears during evidence-conditioned updating

When disambiguating support rows are added, the exact posterior should move toward one mode. The independent oracle also becomes increasingly coherent because its exact marginals approach zero or one.

nanoTabPFN does not show the same reliable improvement. At $r=8$:

| Queries | Marginal JS | Joint JS | Incoherent mass |
|---:|---:|---:|---:|
| 2 | 0.0612 | 0.1265 | 0.2212 |
| 3 | 0.0552 | 0.1595 | 0.3055 |
| 4 | 0.0417 | 0.1606 | 0.3018 |

The four-query result is especially diagnostic: individual probabilities are not extremely far from the exact marginals, but 30.2% of the implied joint probability is assigned to impossible mixed vectors.

The evidence sweep is non-monotonic. nanoTabPFN improves at some intermediate values and worsens again at $r=8$. This likely reflects both the randomized noisy evidence and a learned update rule that is not matched to the known synthetic likelihood. It should not be interpreted as a smooth learning curve.

### 3. Query ordering exposes a separate consistency problem

At $r=0$, order inconsistency is:

- 0.310 for two queries.
- 0.116 for three queries.
- 0.029 for four queries.

Thus, the canonical joint can look close to Bayes while the reverse-order factorization gives a materially different result, particularly for two queries. With strong evidence, order inconsistency generally becomes small, although four-query runs remain noticeably order-sensitive around $r=1$ and $r=2$.

This indicates that nanoTabPFN's repeated conditional predictions do not always correspond to one exchangeable joint posterior.

## Overall conclusion

The experiment does **not** show immediate collapse of unresolved hypotheses. In the fully ambiguous condition, nanoTabPFN preserves the two coherent modes remarkably well.

The more compelling weakness is **incoherent posterior updating**: after task-disambiguating evidence is supplied, individual probabilities can remain plausible while their implied joint assigns substantial mass to impossible cross-task combinations. Some conditions are also sensitive to the order in which the chain-rule joint is constructed.

A precise summary is therefore:

> nanoTabPFN appears capable of encoding multiple unresolved task hypotheses, but its conditionals do not consistently update or factorize those hypotheses as one coherent posterior.

This provides motivation for explicit latent-task, multi-trajectory, or posterior-consistency mechanisms, but it is not by itself proof that nanoTabPFN's hidden representation has collapsed.

## Limitations

- This is an output-level diagnostic. It does not directly inspect hidden representations or prove a particular internal mechanism.
- The joint distribution is induced through repeated conditional calls; nanoTabPFN does not natively emit a joint distribution.
- The piecewise one-dimensional task may differ from the model's pretraining distribution. Some errors may therefore be out-of-distribution generalization failures rather than a universal architectural limitation.
- The model is not explicitly told the 10% evidence-noise rate; the exact Bayes baseline is.
- Only one pretrained checkpoint, one task family, and one seed were evaluated.
- The 32 trials characterize this controlled experiment but are not enough to claim broad statistical generality.
- Canonical and reverse orders start from different query locations. Order sensitivity can reflect both probabilistic inconsistency and location-dependent extrapolation.

## Files

- [Interactive standalone viewer](hypothesis_collapse_results.html)
- [Complete configuration](runs/hypothesis_collapse/20260807-150738/config.json)
- [Per-trial metrics](runs/hypothesis_collapse/20260807-150738/trial_metrics.csv)
- [Complete joint probabilities](runs/hypothesis_collapse/20260807-150738/joint_probabilities.csv)
- [Aggregated summary](runs/hypothesis_collapse/20260807-150738/summary.csv)
- [Experiment implementation](tfmplayground/experiments/hypothesis_collapse.py)
- [Original diagnostic plan](nanoTabPFN_hypothesis_collapse_diagnostic_plan.md)
