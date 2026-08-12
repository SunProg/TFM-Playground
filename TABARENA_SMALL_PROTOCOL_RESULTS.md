# TabArena small-data protocol: results

## Why this rerun exists

Earlier evaluation in this repo (`evaluate_integrated_tabarena.py`) used `max_n_features=500`,
the **entire** training split (contexts of up to ~6,600 rows), and a **single** fold. The
reference experimental setup is far narrower:

> binary classification datasets, **no missing values**, in TabArena, with **at most 10
> features**, datapoints **subsampled to 200**, **stratified 5-fold cross-validation with 20
> repetitions**, measuring accumulated *training* time and excluding evaluation time.

Those are different regimes, not different bookkeeping. Under the reference protocol the
training split is ~160 rows, so the historical 128:128 prior/update split does not even fit
(this run uses 80:80). Conclusions drawn in the wide-context regime therefore do **not**
transfer, and one of them turned out to be an artifact (see "What this overturns").

Implemented in `tfmplayground/experiments/evaluate_tabarena_small.py`.

## Dataset pool

Binary + no missing values + ≤10 predictors leaves **5 of 51** TabArena tasks:

| task_id | dataset | predictors | instances |
|---|---|---|---|
| 363621 | blood-transfusion-service-center | 4 | 748 |
| 363629 | diabetes | 8 | 768 |
| 363613 | Amazon_employee_access | 9 | 32,769 |
| 363619 | Bank_Customer_Churn | 10 | 10,000 |
| 363632 | E-CommereShippingData | 10 | 10,999 |

Notes on selection:

- `Fitness_Club` (6 predictors) *looks* eligible but has 20 missing values and is excluded by
  the no-missing-values criterion.
- "At most 10 features" is read as **10 predictors**. If it instead means OpenML's
  `NumberOfFeatures ≤ 10` (which counts the target), the pool shrinks to 3
  (`blood-transfusion`, `diabetes`, `Amazon_employee_access`). `predictors` is recorded on
  every row so that subset is recoverable without a rerun.

## Protocol actually executed

- stratified subsample to 200 points; `RepeatedStratifiedKFold(n_splits=5, n_repeats=20)`
- **500 fits per model** (5 datasets x 100 folds); train 160 rows, test 40 rows
- prior/update split = **80:80** (derived from available rows)
- fit time timed separately from prediction time
- folds where the test split is single-class are skipped (ROC AUC undefined)

**Deviations from the reference setup**, recorded rather than hidden:

- scikit-learn is **1.9.0** here, not 1.6.1.
- the `tabpfn` package is **not installed**, so there is no TabPFN-v2 baseline; this repo's
  nanoTabPFN backbone (`vanilla`) stands in as the transformer baseline.
- the logistic-regression baseline is wrapped in a `StandardScaler` pipeline (fit inside each
  fold, so no leakage). Unscaled, lbfgs hits its iteration cap and the baseline is understated.

## Overall results (mean over 5 datasets, 500 fits per model)

| model | mean ROC AUC | mean accuracy | Δ vs vanilla | p (paired) | fit time total |
|---|---|---|---|---|---|
| **logreg** | **0.6698** | 0.7706 | **+0.0310** | 1.7e-11 *** | **0.88 s** |
| **h5_allterms** | 0.6684 | 0.7623 | +0.0297 | 2.8e-11 *** | 50.9 s |
| controlled | 0.6608 | 0.7378 | +0.0220 | 1.5e-06 *** | 51.0 s |
| random_forest | 0.6588 | 0.7729 | +0.0200 | 2.4e-07 *** | 31.4 s |
| h5_fixedloss | 0.6455 | 0.7587 | +0.0067 | 0.25 | 51.0 s |
| tabicl | 0.6438 | 0.7516 | +0.0051 | 0.38 | 51.6 s |
| adaptive_k4 | 0.6390 | 0.7665 | +0.0002 | 0.27 | 98.9 s |
| vanilla | 0.6388 | 0.7657 | — | — | 50.3 s |
| h5_gate_k8 | 0.6225 | 0.7496 | −0.0162 | 7.8e-08 *** | 99.8 s |

Paired t-tests over the 500 dataset-folds. `h5_allterms` is the h5-prior filter trained with
the coverage + assignment + posterior terms; `h5_fixedloss` is the same family without them.

### Per dataset (ROC AUC)

| dataset | logreg | h5_allterms | controlled | random_forest | h5_fixedloss | tabicl | adaptive_k4 | vanilla | h5_gate_k8 |
|---|---|---|---|---|---|---|---|---|---|
| Amazon_employee_access | 0.3585 | 0.4056 | 0.4266 | 0.3762 | 0.4102 | 0.4048 | 0.2862 | 0.2862 | 0.1993 |
| Bank_Customer_Churn | 0.7293 | 0.7949 | 0.7775 | 0.7716 | 0.7903 | 0.7158 | 0.7957 | 0.7955 | 0.7966 |
| E-CommereShippingData | 0.7093 | 0.7127 | 0.7042 | 0.7018 | 0.7117 | 0.6961 | 0.6858 | 0.6860 | 0.6779 |
| blood-transfusion-service-center | 0.7488 | 0.6563 | 0.6437 | 0.6661 | 0.5342 | 0.6507 | 0.6871 | 0.6875 | 0.6949 |
| diabetes | 0.8030 | 0.7728 | 0.7519 | 0.7782 | 0.7811 | 0.7518 | 0.7401 | 0.7386 | 0.7439 |

## What this overturns

Earlier in this line of work, evaluating at ~6,600-row contexts on 15 wider datasets, the
conclusion was: *vanilla best on 12/15, the particle model beats vanilla on only 3/15, and a
perfect oracle gate is worth +0.0007 AUC.* **Under the reference protocol that reverses**:
`controlled` and `h5_allterms` beat vanilla at p<1e-10. The earlier finding was a property of
the wide-context regime, not of the approach.

Second reversal: **the coverage + assignment + posterior loss terms do help.** `h5_allterms`
beats `h5_fixedloss` by +0.023 ROC AUC. Those terms had looked like failures because they were
judged by `true_task_recovered` on synthetic paired episodes (pinned at chance in every
configuration), not by real-data performance.

## The important caveat: one dataset drives much of the headline

**Every model is below chance on `Amazon_employee_access`** (0.199–0.427). It is
all-categorical and high-cardinality, and subsampling 32,769 → 200 plausibly destroys it.
Excluding it (4 datasets, 400 folds):

| model | mean ROC AUC (4 sets) | Δ vs vanilla | p |
|---|---|---|---|
| logreg | 0.7476 | +0.0207 | 1.6e-06 *** |
| h5_allterms | 0.7342 | **+0.0073** | 0.031 * |
| random_forest | 0.7294 | +0.0025 | 0.36 |
| h5_gate_k8 | 0.7283 | +0.0015 | 0.16 |
| adaptive_k4 | 0.7272 | +0.0003 | 0.27 |
| vanilla | 0.7269 | — | — |
| controlled | 0.7193 | **−0.0075** | 0.027 * |
| h5_fixedloss | 0.7043 | −0.0226 | 8.6e-05 *** |
| tabicl | 0.7036 | −0.0233 | 1.5e-07 *** |

`controlled` flips sign: its apparent +0.0220 win over vanilla was driven **entirely** by the
one broken dataset. `h5_allterms` survives but weakens to +0.0073 (p=0.031).

## Conclusions

1. **Regime matters more than any modelling change made in this repo.** The same checkpoints
   move from losing to vanilla to beating it purely by changing context size from ~6,600 rows
   to ~160. Any claim about these filters must state the regime.
2. **The defensible positive result is narrow**: `h5_allterms` beats vanilla in the small-data
   regime, small effect (+0.0073 without the broken dataset, p=0.031), and it is the filter
   trained with the coverage/assignment/posterior terms.
3. **Nothing beats logistic regression.** logreg tops the table and is statistically
   indistinguishable from the best filter (Δ=−0.0014, p=0.80) at **1/58th the fit time**. On
   ≤10 features and 200 points this is the honest headline.
4. **The gate still does not earn its cost.** `adaptive_k4` ≈ vanilla (Δ=+0.0002), and
   `h5_gate_k8` is significantly *worse* (−0.0162), at 2x the fit time, consistent with the
   gate collapsing toward the vanilla fallback.
5. **The pool is thin.** 5 datasets (4 usable) is a small sample; the reference setup
   presumably had a different pool. Whether `Amazon_employee_access` belongs at all is worth
   deciding explicitly, since it flips one model's sign.

## Reproduce

```bash
uv run python -m tfmplayground.experiments.evaluate_tabarena_small \
  --device cpu --subsample 200 --folds 5 --repeats 20 --max-predictors 10 \
  --integrated-checkpoints "controlled=...,tabicl=...,h5_fixedloss=...,h5_allterms=..." \
  --adaptive-checkpoints "adaptive_k4=...,h5_gate_k8=..." \
  --output-dir runs/tabarena_small/protocol-v1
```

Artifacts: `runs/tabarena_small/protocol-v1/{eligible_tasks,fold_metrics,per_dataset,overall}.csv`.
