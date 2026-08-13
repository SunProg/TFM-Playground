# Mean-Preserving Bayesian nanoTabPFN Trial

## Status

- Date: 2026-08-13
- Compute: CREATE Slurm, `biomed_a30_gpu` partition, NVIDIA A30 GPUs
- Scope: binary classification only
- Backbone: frozen vanilla nanoTabPFN from `checkpoints/nanotabpfn.pth`
- Training data: synthetic structured TabICL/SCM episodes only
- Real-data use: TabArena evaluation only; no TabArena labels or tasks were used for training
- Outcome: predictive-mean preservation passed, but the learned slot uncertainty failed the TabArena uncertainty gate

## Model and training

The trial added a frozen-backbone uncertainty head with `K` learned hypothesis slots. For each query, vanilla nanoTabPFN supplied the predictive mean `mu`. Cross-fitted support likelihoods produced normalized posterior slot weights, and slot deviations were centered and bounded so that

```text
sum_k w_k p_qk = mu_q
```

up to floating-point precision. Consequently, the Bayesian mixture could not improve or degrade vanilla NLL, Brier score, ROC AUC, or accuracy. It only learned a decomposition of the fixed mean into candidate predictions and uncertainty statistics.

Training used a 50/30/20 curriculum of ambiguous paired SCM, identifiable/single-task SCM, and structured noisy-label episodes. Independent random labels were diagnostic-only. The backbone and vanilla decoder remained frozen; no full-fine-tuning stage was run. The uncertainty-head objective combined joint query-vector NLL, posterior-weight cross-entropy, slot-specific query loss, mutual-information calibration, and epistemic-variance calibration. Early stopping used validation checks every 100 steps, patience 10, and `min_delta=1e-4`, with a 5,000-step ceiling.

## Long-run sweep

Arrows show the desired direction. Validation losses are comparable within the long-run trials, but values across different `K` describe different candidate-posterior dimensions and should not be interpreted alone as model quality.

| K | learning rate | tau | best step | validation loss (lower) | ambiguous MI | identifiable MI | synthetic gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 3e-5 | 1e-2 | 1,600 | 0.710436 | 0.012893 | 0.018604 | Fail |
| 4 | 3e-5 | 1e-2 | 1,700 | 0.828853 | 0.010874 | 0.009540 | Pass |
| 4 | 1e-4 | 1e-2 | 700 | 0.830124 | 0.023574 | 0.017936 | Pass |
| 4 | 1e-4 | 3e-3 | 2,100 | 0.836244 | 0.017787 | 0.014377 | Pass |
| 4 | 3e-5 | 3e-3 | 4,600 | 0.841589 | 0.011751 | 0.010305 | Pass |
| 8 | 1e-4 | 1e-2 | 2,200 | 0.906840 | 0.009919 | 0.008624 | Pass |
| 8 | 3e-5 | 1e-2 | 700 | 0.907548 | 0.004370 | 0.003392 | Pass |
| 16 | 1e-4 | 1e-2 | 2,000 | 0.852373 | 0.002844 | 0.002362 | Pass |
| 16 | 3e-5 | 1e-2 | 2,100 | 0.855531 | 0.002033 | 0.001654 | Pass |
| 32 | 3e-5 | 1e-2 | 1,400 | 0.874999 | 0.000223 | 0.000167 | Pass |

Increasing `K` did not yield a useful scaling trend. In particular, ambiguous-episode MI generally shrank at larger `K`, reaching `0.000223` for `K=32`. The `K=2` run failed because its MI was higher on identifiable than ambiguous episodes. The `K=4`, learning-rate `3e-5`, `tau=1e-2` checkpoint at step 1,700 was used for the real-data test.

## Binary TabArena evaluation

The selected model and vanilla nanoTabPFN were evaluated with identical deterministic context rows on 20 official binary tasks. Each model used 1,024 labelled context rows where available, and the untouched official test partition was used only for scoring. Query labels were not used to construct predictions or the posterior. No multiclass task was coerced into binary.

### Predictive metrics averaged over 20 tasks

| Metric | Bayesian | Vanilla | Direction |
|---|---:|---:|:---:|
| NLL | 0.350754 | 0.350754 | lower |
| Brier score | 0.110422 | 0.110422 | lower |
| ROC AUC | 0.829560 | 0.829560 | higher |
| Accuracy | 0.844142 | 0.844142 | higher |
| ECE | 0.030112 | 0.030112 | lower |

The maximum Bayesian-versus-vanilla probability difference was `2.98e-7`, satisfying the `1e-6` mean-preservation requirement. Equal predictive metrics are therefore expected and are not evidence that the uncertainty head improved prediction.

### Uncertainty metrics

| Metric | Bayesian slot MI | Vanilla predictive entropy | Direction |
|---|---:|---:|:---:|
| Error-detection AUROC | 0.254 | 0.767 | higher |
| AURC | 0.246 | 0.076 | lower |

Additional Bayesian diagnostics were mean MI `0.00434`, mean epistemic variance `0.00119`, mean effective sample size `1.20` out of four slots, and posterior-collapse fraction `0.60`.

The uncertainty acceptance gate failed: the learned MI was strongly anti-correlated with mistakes and produced substantially worse selective risk than vanilla predictive entropy. The head preserved the mean correctly but did not learn transferable epistemic uncertainty on TabArena.

## Interpretation and next direction

Fixed learned slots are a poor fit for this form of meta-learning. Candidate SCM functions change between episodes, so a globally persistent slot has no stable hypothesis semantics. Episode-local Hungarian matching resolves ordering only for the loss; it does not create reusable slot identities. Duplicate/collapsed slots can also make posterior entropy and effective hypothesis count misleading. The absence of meaningful improvement as `K` increased supports this diagnosis.

Keep this implementation and its checkpoints as a negative benchmark. Do not promote it as the default Bayesian model. The next comparison should preserve the vanilla mean while removing persistent slots:

1. A non-learned context-resampling uncertainty baseline.
2. An anonymous continuous function-space posterior with shared latent draws across queries.
3. A mean-constrained Beta concentration head as a deliberately low-capacity ablation, not the primary model.

That follow-up experiment is implemented and documented in
[CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md](CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md). This trial and its
checkpoints remain the documented negative benchmark for it.

The continuous posterior should be judged against both the context-resampling and Beta controls. Synthetic evaluation must test exact posterior moments and uncertainty reduction with identifying evidence; TabArena evaluation must retain identical contexts and require useful error ranking or selective risk without changing the predictive mean.

## Evaluated binary tasks

`Bank_Customer_Churn`, `blood-transfusion-service-center`, `churn`, `coil2000_insurance_policies`, `credit-g`, `diabetes`, `E-CommereShippingData`, `Fitness_Club`, `hazelnut-spread-contaminant-detection`, `heloc`, `in_vehicle_coupon_recommendation`, `Is-this-a-good-customer`, `Marketing_Campaign`, `NATICUSdroid`, `online_shoppers_intention`, `polish_companies_bankruptcy`, `qsar-biodeg`, `seismic-bumps`, `taiwanese_bankruptcy_prediction`, and `jm1`.
