# Vanilla nanoTabPFN baseline for the slot sweep: full metrics

## Status

- Date: 2026-08-31
- Slurm array `36860987`, indices 12-15, all four arms COMPLETED at 10,000 steps
- Branch `exp/slot-attention`; trained by `pretrain_slot_tabpfn.py` with
  `--model-kind vanilla`, i.e. a plain `NanoTabPFNModel` in the identical
  harness the slot arms use -- same episode dump, validation cadence, TabArena
  settings and metric summarisation -- so architecture is the only variable
- Seed 2402 throughout, single seed
- Companion documents:
  [`slot_attention_prior_composition_results.md`](slot_attention_prior_composition_results.md)
  (the slot arms) and
  [`slot_attention_implementation_corrections.md`](slot_attention_implementation_corrections.md)
  (defects found along the way)

## Why this exists

The slot-attention screening grid measured twelve configurations against each
other but nothing external, so it could not say whether the slot machinery
changed anything at all. These are the matched controls. They also turn out to
be the more informative half: several conclusions that looked like statements
about slots are really statements about the evaluation.

## Configuration

```
10,000 steps (20 epochs of 500), seed 2402, warmup 2,000, cosine to 1e-6
micro-batch 8 x accumulate 4 = 32 episodes per optimizer step
support 128, query 32, features 2-12, contamination 0.3, label noise 0.0
loss: plain query cross entropy
validation every 500 steps; TabArena-small every 500 steps at 5 folds x 10 repeats
multiregime episodes streamed from a 375,000-episode dump
```

Four prior compositions: `plain` (single-regime only), `multiregime`
(multiregime only), `mixed` (70/30 constant), `curriculum` (single first, ramping
to a 0.5 share by half way).

## Synthetic validation

| prior | mr_ce | ord_ce |
|---|---|---|
| plain | 0.6847 [0.6763, 0.6930] | 0.4438 [0.4018, 0.4859] |
| multiregime | 0.6830 [0.6748, 0.6911] | 0.7310 [0.6664, 0.7956] |
| mixed | 0.6838 [0.6745, 0.6932] | 0.4475 [0.4064, 0.4886] |
| curriculum | 0.6834 [0.6752, 0.6916] | 0.4465 [0.4053, 0.4877] |

Chance is ln 2 = 0.6931. **No arm learned the multiregime task**: every `mr_ce`
interval touches or contains chance. Ordinary data is learned normally (0.444 to
0.448) by the three arms that see it. The `multiregime` arm's `ord_ce` of 0.7310
is worse than chance on clean data -- but see the calibration section: that is a
threshold statement, not a competence one.

## TabArena aggregate, 250 fits per arm

Mean plus or minus half a 95% interval across fits.

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.7298 ± 0.0168 | 0.7179 ± 0.0163 | 0.7188 ± 0.0189 | 0.7295 ± 0.0165 |
| auprc | 0.7373 ± 0.0194 | 0.7025 ± 0.0224 | 0.7358 ± 0.0192 | 0.7369 ± 0.0195 |
| accuracy | 0.7867 ± 0.0121 | 0.6055 ± 0.0345 | 0.7895 ± 0.0112 | 0.7938 ± 0.0110 |
| precision | 0.6614 ± 0.0428 | 0.3278 ± 0.0519 | 0.6948 ± 0.0446 | 0.6908 ± 0.0437 |
| recall | 0.4774 ± 0.0470 | 0.2231 ± 0.0357 | 0.4281 ± 0.0456 | 0.4395 ± 0.0434 |
| f1 | 0.5028 ± 0.0429 | 0.2516 ± 0.0385 | 0.4729 ± 0.0449 | 0.4988 ± 0.0419 |
| specificity | 0.6716 ± 0.0494 | 0.9577 ± 0.0108 | 0.7429 ± 0.0471 | 0.7523 ± 0.0473 |
| cross_entropy | 0.4309 ± 0.0141 | 0.6887 ± 0.0383 | 0.4365 ± 0.0142 | 0.4297 ± 0.0141 |
| brier | 0.1385 ± 0.0058 | 0.2472 ± 0.0174 | 0.1399 ± 0.0058 | 0.1379 ± 0.0058 |

Class balance is identical across arms (46.6% positive) because all four score
the same locked episodes. `fit_seconds` is 0.222 to 0.225 for all four.

### TabArena ROC AUC over training

| step | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| 500 | 0.5271 | 0.5498 | 0.5456 | 0.5271 |
| 1000 | 0.5574 | 0.4454 | 0.5441 | 0.5574 |
| 1500 | 0.6968 | 0.5386 | 0.5589 | 0.6631 |
| 2000 | 0.7245 | 0.5337 | 0.7081 | 0.7164 |
| 2500 | 0.7356 | 0.5542 | 0.7344 | 0.7333 |
| 3000 | 0.7299 | 0.5474 | 0.7177 | 0.7205 |
| 4000 | 0.7376 | 0.6915 | 0.7118 | 0.7207 |
| 5000 | 0.7254 | 0.6939 | 0.7198 | 0.7161 |
| 6000 | 0.7344 | 0.6941 | 0.7185 | 0.7308 |
| 7000 | 0.7311 | 0.7090 | 0.7239 | 0.7302 |
| 8000 | 0.7321 | 0.7200 | 0.7222 | 0.7281 |
| 9000 | 0.7282 | 0.7192 | 0.7188 | 0.7303 |
| 10000 | 0.7298 | 0.7179 | 0.7188 | 0.7295 |

Three arms plateau by step 2000-2500 and stay flat for the remaining 8,000
steps; `multiregime` is stuck near chance until step 4000, then converges to the
same place. **On real tables everything has converged well before the budget
ends**, which weakens the "these models are undertrained at 10k steps" caveat as
an explanation for the synthetic-side failures.

## Per dataset

Positive rate is the AUPRC floor for that dataset.

### Amazon_employee_access (positive rate 0.942)

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.4811 | 0.4718 | 0.4346 | 0.4853 |
| auprc | 0.9391 | 0.9357 | 0.9347 | 0.9434 |
| accuracy | 0.9419 | 0.0581 | 0.9419 | 0.9419 |
| precision | 0.9419 | 0.0000 | 0.9419 | 0.9419 |
| recall | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| f1 | 0.9701 | 0.0000 | 0.9701 | 0.9701 |
| specificity | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| cross_entropy | 0.2232 | 1.2983 | 0.2250 | 0.2242 |
| brier | 0.0549 | 0.5241 | 0.0552 | 0.0551 |

### Bank_Customer_Churn (positive rate 0.204)

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.8356 | 0.8014 | 0.8313 | 0.8350 |
| auprc | 0.6472 | 0.5068 | 0.6355 | 0.6407 |
| accuracy | 0.8218 | 0.7964 | 0.8095 | 0.8238 |
| precision | 0.9013 | 0.0000 | 0.9623 | 0.8947 |
| recall | 0.1413 | 0.0000 | 0.0683 | 0.1571 |
| f1 | 0.2432 | 0.0000 | 0.1262 | 0.2647 |
| specificity | 0.9958 | 1.0000 | 0.9990 | 0.9943 |
| cross_entropy | 0.3980 | 0.4782 | 0.4145 | 0.3942 |
| brier | 0.1230 | 0.1525 | 0.1295 | 0.1220 |

### E-CommereShippingData (positive rate 0.597)

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.7455 | 0.7453 | 0.7436 | 0.7456 |
| auprc | 0.8561 | 0.8560 | 0.8555 | 0.8566 |
| accuracy | 0.6445 | 0.6568 | 0.6683 | 0.6790 |
| precision | 0.6745 | 1.0000 | 0.8162 | 0.8516 |
| recall | 0.7850 | 0.4249 | 0.5756 | 0.5612 |
| f1 | 0.7242 | 0.5960 | 0.6734 | 0.6755 |
| specificity | 0.4368 | 1.0000 | 0.8053 | 0.8533 |
| cross_entropy | 0.5323 | 0.6187 | 0.5410 | 0.5319 |
| brier | 0.1874 | 0.2148 | 0.1875 | 0.1859 |

### blood-transfusion-service-center (positive rate 0.238)

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.7509 | 0.7505 | 0.7444 | 0.7487 |
| auprc | 0.5174 | 0.5093 | 0.5222 | 0.5193 |
| accuracy | 0.7611 | 0.7620 | 0.7619 | 0.7620 |
| precision | 0.0100 | 0.0000 | 0.0000 | 0.0100 |
| recall | 0.0011 | 0.0000 | 0.0000 | 0.0006 |
| f1 | 0.0020 | 0.0000 | 0.0000 | 0.0011 |
| specificity | 0.9984 | 1.0000 | 0.9998 | 0.9998 |
| cross_entropy | 0.4974 | 0.5013 | 0.5039 | 0.4992 |
| brier | 0.1616 | 0.1633 | 0.1644 | 0.1628 |

### diabetes (positive rate 0.349)

| metric | plain | multiregime | mixed | curriculum |
|---|---|---|---|---|
| roc_auc | 0.8360 | 0.8205 | 0.8403 | 0.8331 |
| auprc | 0.7269 | 0.7047 | 0.7314 | 0.7246 |
| accuracy | 0.7641 | 0.7543 | 0.7659 | 0.7622 |
| precision | 0.7794 | 0.6392 | 0.7535 | 0.7558 |
| recall | 0.4598 | 0.6908 | 0.4967 | 0.4788 |
| f1 | 0.5746 | 0.6619 | 0.5949 | 0.5826 |
| specificity | 0.9272 | 0.7884 | 0.9102 | 0.9142 |
| cross_entropy | 0.5034 | 0.5469 | 0.4979 | 0.4992 |
| brier | 0.1654 | 0.1811 | 0.1627 | 0.1639 |

### Dataset characteristics and cost

Identical across arms, since all four score the same locked episodes.
`contamination` is 0 because the in-training TabArena evaluation uses
`label_source="real"` with `contamination=0.0` -- these are clean tables, not
multiregime episodes, which is what makes them an out-of-distribution check on
models trained on contaminated priors.

| dataset | predictors | train rows | test rows | positive % | fit seconds |
|---|---|---|---|---|---|
| Amazon_employee_access | 9 | 1638 | 409.6 | 94.19 | 0.3291 |
| Bank_Customer_Churn | 10 | 1638 | 409.6 | 20.36 | 0.3645 |
| E-CommereShippingData | 10 | 1638 | 409.6 | 59.67 | 0.3649 |
| blood-transfusion-service-center | 4 | 598.4 | 149.6 | 23.80 | 0.0243 |
| diabetes | 8 | 614.4 | 153.6 | 34.90 | 0.0376 |

Row counts are fold means, hence fractional. `predict_seconds` is not recorded
by the harness. `support_positive_pct` equals `query_positive_pct` on every
dataset, so only one column is shown.

Note the train-row counts: 598 to 1638, not the 128 the *synthetic* validation
uses. The threshold pathology described below therefore is not simply a
small-sample artifact of 128 rows.

## Analysis

### Amazon_employee_access is anti-predictive and distorts the aggregate

Every arm scores ROC AUC **below chance** on it (0.435 to 0.485), and its AUPRC
of 0.935 to 0.943 sits at or under its own 0.942 floor. At 94.2% positive, three
arms predict all-positive (recall 1.000, specificity 0.000) and `multiregime`
predicts all-negative. No arm learns anything here.

It nonetheless contributes the largest AUPRC values in the aggregate, inflating
it, while contributing negative ranking signal. Reporting it separately, or
excluding it, would make every comparison in this sweep cleaner. That is a change
to the shared evaluation protocol and has not been made.

### The `multiregime` arm's aggregate collapse is one dataset

Aggregate accuracy falls 0.7867 to 0.6055, a gap of 0.181. Amazon alone accounts
for (0.9419 - 0.0581)/5 = **0.177** of that. The arm did not degrade broadly; it
flipped which majority class it defaults to on a single table.

Per-dataset ROC AUC confirms it: `multiregime` is within about 0.03 of the other
arms on all five datasets. Its *ranking* is comparable throughout.

### Discrimination is fine; thresholding is broken, for every arm

On `blood-transfusion` every arm has recall about 0.001 and precision about 0.01
-- pure all-negative prediction, accuracy 0.761 exactly matching the negative
rate -- while ROC AUC is 0.75 and AUPRC 0.52 against a 0.238 floor. That is a
well-ranked model with a useless decision threshold. `Bank_Customer_Churn` has
the same shape (recall 0.14, precision 0.90).

It is worth being clear that this is **not** a small-sample artifact: these
TabArena tables give 598 to 1638 training rows, not the 128 the synthetic
validation uses. The models are miscalibrated on real tables at realistic sizes.
The likely cause is the prior itself -- nanoTabPFN is trained on synthetic SCM
episodes whose class balance and difficulty need not match these datasets, and
nothing in the objective ties its output scale to a real table's base rate.

It means **accuracy, precision, recall and F1 are not usable as headline metrics
in this sweep**: they mostly report which majority class a model defaults to.
ROC AUC and AUPRC-above-floor are the honest summaries.

It also reframes the synthetic numbers. `mr_ce` and `ord_ce` are cross entropies,
so they are partly measuring calibration. The `multiregime` arm's `ord_ce` of
0.7310, "worse than chance", is a miscalibration statement -- its aggregate
cross entropy is 0.6887 and Brier 0.2472 against about 0.139 elsewhere -- and not
evidence that it cannot discriminate, which its ROC AUC of 0.718 shows it can.
Training only on 30%-contaminated labels shifts the learned probability scale;
random contamination does not systematically reorder examples, so ranking
survives and thresholds do not.

### Where the `multiregime` arm is better

On `diabetes` it has the highest recall (0.691 against 0.460 for `plain`) and the
highest F1 (0.662 against 0.575). Its threshold pathology happens to help where
positives are otherwise under-predicted.

## Limitations

- Single seed (2402).
- 10,000 steps against the 50,000 used by the published baselines; though the
  TabArena curves show convergence by step 2500 for three of four arms.
- This protocol uses a 2,048-row subsample at 5 folds x 10 repeats. It is **not**
  comparable to `paper/finetuned_nanotabpfn_tabarena_results.md`, which uses a
  200-row subsample at 5 x 20.
- Five datasets, of which one (Amazon) is anti-predictive and one
  (blood-transfusion) is near the AUPRC floor for every arm, so the effective
  comparison rests on three tables.
- AUPRC, precision, recall, F1, specificity and Brier are **not** written to
  `history.jsonl`; `evaluate_tabarena_epoch` lifts only `mean_roc_auc` and
  `mean_accuracy`. Everything above beyond those two was read from the per-epoch
  `tabarena/epoch-*/fold_metrics.csv` files. Logging them would make this
  reproducible from the history alone.
