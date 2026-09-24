# Multi-regime v3 pilot: TabArena results

Date: 2026-09-10

## What this is

TabArena small-protocol evaluation (`tabarena15` settings: `max_predictors=30`,
`subsample=2048`, 5-fold x 10-repeat CV, real labels, no contamination) of the
six checkpoints from the completed v3 pilot (`runs/37103332`, 10,000 steps,
effective batch 32, architecture embedding=192/heads=6/mlp=768/layers=6),
against logistic regression and random forest baselines on the identical test
set (same seed, so `eligible_tasks()` and the `RepeatedStratifiedKFold` splits
match exactly).

This is **not** TabPFN v2.2/v2.6/v3 comparison -- that job failed on an
argument the deployed `evaluate_tabarena_small.py` doesn't support yet
(depends on another session's uncommitted local edit) and has not been rerun.

Jobs: `37112362` (v3 checkpoints), `37110454` (sklearn baselines). 15 eligible
datasets (binary, no missing values, <=30 predictors).

## Overall (mean over 15 datasets, 50 folds each)

| model | mean_roc_auc | mean_accuracy | mean_cross_entropy | mean_brier |
|---|---:|---:|---:|---:|
| random_forest | 0.7761 | 0.8208 | 0.4578 | 0.1260 |
| logreg | 0.7506 | 0.8060 | 0.4140 | 0.1323 |
| **plain-fixed** | **0.7305** | 0.7929 | 0.4396 | 0.1406 |
| slot-fixed | 0.7293 | 0.7940 | 0.4357 | 0.1395 |
| plain-curriculum | 0.7208 | 0.7912 | 0.4402 | 0.1411 |
| slot-curriculum | 0.7149 | 0.7908 | 0.4419 | 0.1420 |
| plain-original | 0.6229 | 0.7676 | 0.4885 | 0.1598 |
| slot-original | 0.6223 | 0.7678 | 0.4880 | 0.1595 |

**All six v3 checkpoints lose to both sklearn baselines** on AUC and
accuracy. Best v3 cell (`plain-fixed`, AUC 0.7305) is ~0.02 below logistic
regression and ~0.046 below random forest -- consistent with the pattern
observed across every model tried on TabArena so far in this project.

`original`-mode training (no exposure to the four v3-generated episode
families during pretraining) generalizes far worse to real tables (AUC
~0.62) than `fixed`/`curriculum` (AUC ~0.72-0.73), mirroring what the
pilot's own synthetic validation already showed. `slot` vs. `plain`
differences are small and inconsistent across modes (slot slightly ahead on
`fixed`, slightly behind on `curriculum`/`original`).

## Per-dataset ROC AUC

| dataset | predictors | logreg | random_forest | plain-original | slot-original | plain-fixed | slot-fixed | plain-curriculum | slot-curriculum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon_employee_access | 9 | 0.545 | 0.634 | 0.464 | 0.478 | 0.456 | 0.500 | 0.480 | 0.458 |
| Bank_Customer_Churn | 10 | 0.744 | 0.839 | 0.605 | 0.592 | 0.808 | 0.807 | 0.794 | 0.790 |
| E-CommereShippingData | 10 | 0.731 | 0.739 | 0.632 | 0.631 | 0.732 | 0.732 | 0.745 | 0.737 |
| Is-this-a-good-customer | 13 | 0.712 | 0.727 | 0.560 | 0.567 | 0.634 | 0.692 | 0.700 | 0.661 |
| bank-marketing | 13 | 0.723 | 0.704 | 0.614 | 0.604 | 0.615 | 0.654 | 0.649 | 0.613 |
| blood-transfusion-service-center | 4 | 0.751 | 0.680 | 0.587 | 0.577 | 0.725 | 0.757 | 0.754 | 0.750 |
| churn | 19 | 0.815 | 0.903 | 0.644 | 0.644 | 0.823 | 0.831 | 0.723 | 0.777 |
| credit-g | 20 | 0.731 | 0.781 | 0.568 | 0.566 | 0.757 | 0.750 | 0.751 | 0.747 |
| credit_card_clients_default | 23 | 0.721 | 0.737 | 0.616 | 0.616 | 0.725 | 0.747 | 0.720 | 0.720 |
| diabetes | 8 | 0.832 | 0.820 | 0.775 | 0.771 | 0.828 | 0.819 | 0.825 | 0.825 |
| hazelnut-spread-contaminant-detection | 30 | 0.952 | 0.957 | 0.826 | 0.828 | 0.880 | 0.877 | 0.880 | 0.887 |
| heloc | 23 | 0.759 | 0.764 | 0.531 | 0.540 | 0.760 | 0.755 | 0.758 | 0.749 |
| in_vehicle_coupon_recommendation | 24 | 0.627 | 0.712 | 0.564 | 0.565 | 0.624 | 0.424 | 0.478 | 0.416 |
| online_shoppers_intention | 17 | 0.869 | 0.915 | 0.635 | 0.627 | 0.880 | 0.880 | 0.868 | 0.871 |
| seismic-bumps | 15 | 0.748 | 0.731 | 0.723 | 0.728 | 0.710 | 0.715 | 0.687 | 0.724 |

Notable: `in_vehicle_coupon_recommendation` is the one dataset where v3's
`fixed`/`curriculum` cells fall *below* their own `original` cell and well
below both baselines (slot-curriculum 0.416 vs. logreg 0.627) -- worth a
second look rather than averaging away.

## Caveats

- Single training seed per cell; no error bars on the v3 side (the v3 pilot's
  own synthetic eval reports `episode_log_loss_se`, but that doesn't carry
  over to real-data AUC). `roc_auc_std` in the raw per-dataset CSVs gives
  fold-level spread within one seed, not across seeds.
- v3 pretraining pads inputs to `max_features + num_groups` (12 + 5 = 17)
  columns for nuisance group codes; TabArena tables have none of those and
  arrive at their own native width (`predictors` column above). This is a
  real distribution shift in feature count, not a shape bug --
  `NanoTabPFNModel`'s per-feature encoder embeds each column independently
  and attends across whatever count is present, so it runs without error,
  but the shift itself may be part of why these underperform baselines that
  never saw the padded synthetic format at all.
- TabPFN v2.2/v2.6/v3 baselines are still outstanding (see above).
