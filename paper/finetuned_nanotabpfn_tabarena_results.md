# Full-backbone fine-tuning on clean TabArena-small

## Status and provenance

- Date: 2026-08-27
- CREATE Slurm fine-tuning array: `36807068` (three seeds; all completed successfully)
- CREATE Slurm evaluation: `36807069` (completed successfully)
- Models: the untouched `nanotabpfn.pth` checkpoint (`vanilla`) and three
  full-backbone fine-tunes trained on the multiregime curriculum (`seed2402`,
  `seed2403`, and `seed2404`)
- Comparator: real `TabPFNClassifier` from the completed matching clean
  `runs/tabarena_small/protocol-v2/` evaluation
- Raw local artifacts:
  `runs/tabarena_small/finetuned-backbone-36807068-36807069/`

## Protocol

This is the existing **clean** TabArena-small protocol, not the mixed-regime
variant. All labels are the real TabArena targets, and contamination is zero.
Each model is evaluated on exactly the same data splits:

- five eligible binary TabArena datasets with no missing values and at most ten predictors;
- a stratified 200-row subsample per dataset;
- 5-fold cross-validation repeated 20 times (100 folds per dataset; 500 per model);
- support/context size of roughly 160 rows and a held-out test set of roughly 40 rows per fold.

The fine-tunes are loaded as ordinary nanoTabPFN backbones and receive the same
preprocessed fold data as vanilla nanoTabPFN. TabPFN receives the raw feature
frame and applies its own preprocessing, as intended by its interface. No
adapter, particle-filter, or synthetic-label arm is included in this
comparison.

## Overall results

| model | mean ROC AUC | delta vs. vanilla | mean accuracy |
|---|---:|---:|---:|
| TabPFN | **0.6908** | **+0.0521** | 0.7904 |
| vanilla | 0.6388 | — | 0.7658 |
| seed2402 | 0.6608 | +0.0220 | 0.7799 |
| seed2403 | 0.6645 | +0.0258 | 0.7879 |
| seed2404 | 0.6708 | +0.0321 | **0.7924** |

## Per-dataset ROC AUC

| dataset | TabPFN | vanilla | seed2402 | seed2403 | seed2404 |
|---|---:|---:|---:|---:|---:|
| Amazon_employee_access | 0.3504 | 0.2862 | 0.2556 | 0.2944 | 0.3229 |
| Bank_Customer_Churn | 0.8073 | 0.7955 | 0.8038 | 0.8021 | 0.8116 |
| E-CommereShippingData | 0.7497 | 0.6860 | 0.7268 | 0.7358 | 0.7245 |
| blood-transfusion-service-center | 0.7401 | 0.6875 | 0.7232 | 0.7091 | 0.7200 |
| diabetes | 0.8066 | 0.7386 | 0.7945 | 0.7813 | 0.7751 |

Each fine-tune improves four of five datasets. The only exception is
`Amazon_employee_access`: seed2402 is below vanilla, seed2403 is modestly
above it, and seed2404 improves substantially.

## Interpretation

Under this restricted clean TabArena protocol, the multiregime-curriculum
fine-tunes improve ordinary IID predictive performance over the original
nanoTabPFN checkpoint. The mean improvement ranges from +0.022 to +0.032 ROC
AUC across the three independently retrained seeds. Real TabPFN remains best
on average (+0.052 ROC AUC over vanilla), although seed2404 is narrowly higher
on `Bank_Customer_Churn` (0.8116 versus 0.8073).

This result does not contradict the mixed-regime result: the previous
synthetic study found that fine-tuning did **not** reliably improve prediction
on the minority/contaminating regime. The two findings together indicate that
this fine-tuning can improve conventional single-regime performance while not
being a general solution to per-row latent-regime mixing.

Paired t-tests over the 500 repeated dataset-folds give p-values of
`1.1e-09`, `3.1e-15`, and `2.1e-23` for seeds 2402, 2403, and 2404,
respectively. These folds are repeated splits of only five datasets, so the
p-values should be treated as descriptive rather than as independent-dataset
evidence.

## Fixed-budget multi-regime trajectory with continuation-only control

- Date: 2026-08-28
- CREATE training array: `36808725` (18 runs: 0%, 10%, ..., 50% multi-regime
  frequency by three seeds; all completed successfully)
- CREATE TabArena evaluation: `36808744` (180 pre-registered checkpoints; completed successfully)
- Every run used exactly 10,000 optimizer updates, with **no early stopping**.
  Synthetic validation and a checkpoint were recorded every 1,000 updates.
- The 0% arm is the matched continued-pretraining control: it uses only
  ordinary single-regime MLP-SCM/tree-SCM episodes. Nonzero arms balance
  within-MLP, within-tree, and cross-family MLP--tree row-level mixtures.
- TabArena was never used for gradient updates or checkpoint selection. Its
  repeated evaluations form a diagnostic learning curve, so the best observed
  frequency still needs an independent confirmation.
- CREATE artifact:
  `runs/tabarena_small/multiregime-trajectory-36808725-36808744/`

### Pretrained baseline and 10k endpoint

| model / curriculum | mean ROC AUC | delta vs. pretrained | delta vs. 0% at 10k |
|---|---:|---:|---:|
| **Pretrained nanoTabPFN (step 0)** | 0.6388 | — | — |
| 0% multi-regime (continued pretraining control) | 0.6631 | +0.0243 | — |
| 10% multi-regime | 0.6730 | +0.0343 | +0.0099 |
| 20% multi-regime | 0.6645 | +0.0258 | +0.0015 |
| 30% multi-regime | 0.6799 | +0.0411 | +0.0168 |
| **40% multi-regime** | **0.6800** | **+0.0412** | **+0.0169** |
| 50% multi-regime | 0.6642 | +0.0254 | +0.0011 |

### Per-dataset ROC AUC at the 10k endpoint

Values for p0--p50 are the mean over the three fixed-budget seeds. The
pretrained column is the untouched step-0 nanoTabPFN backbone.

| dataset | pretrained | p0 | p10 | p20 | p30 | p40 | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon_employee_access | 0.2862 | 0.2714 | 0.3148 | 0.2867 | 0.3348 | **0.3374** | 0.2606 |
| Bank_Customer_Churn | 0.7955 | 0.8030 | 0.8008 | 0.8012 | **0.8117** | 0.8009 | 0.8090 |
| E-CommereShippingData | 0.6860 | 0.7425 | 0.7427 | 0.7349 | **0.7486** | 0.7387 | 0.7342 |
| blood-transfusion-service-center | 0.6875 | 0.7146 | 0.7152 | 0.7071 | 0.7170 | **0.7296** | 0.7225 |
| diabetes | 0.7386 | 0.7840 | 0.7915 | 0.7928 | 0.7874 | 0.7935 | **0.7947** |

At 10,000 updates, the 40% arm exceeds its seed-matched 0% control for all
three seeds (AUC differences +0.0185, +0.0181, and +0.0141). Its advantage is
mainly from Amazon employee access (+0.0660), blood transfusion (+0.0149), and
diabetes (+0.0095); it is essentially tied/slightly lower on the two remaining
datasets. Continued pretraining is itself important: the 0% control reaches
0.6791 at 2,000 updates. Synthetic validation cross-entropy does not usefully
predict TabArena AUC across this trajectory (Spearman -0.107).
