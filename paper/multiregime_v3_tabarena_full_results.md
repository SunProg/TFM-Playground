# v3 21-cell table_slot sweep: full TabArena results

Date: 2026-09-12, updated 2026-09-14. Covers `runs/37113090` (21 cells:
`plain` + six `table_slot` variants x `original`/`fixed`/`curriculum`, all
COMPLETED) evaluated on TabArena (job `37118637`, `tabarena15` protocol:
`max_predictors=30`, `subsample=2048`, 5-fold x 10-repeat CV, real labels, 15
eligible datasets), plus sklearn baselines (logreg, random forest) from a
separate job (`37110454`) run with identical seed/subsample/fold/repeat
settings, so the same 15 datasets and folds apply to both. TabPFN v2.2/v2.6/v3
are not in this comparison -- that job never successfully ran against real
TabArena data (blocked on an uncommitted `--tabpfn-model-path` CLI flag);
TabPFN numbers exist only on the pilot's own synthetic eval bank, see
`multiregime_v3_synthetic_baseline_comparison.md`.

**2026-09-14 addition**: `attention_slot_router_frozen_tabpfn/mix_scm` --
`AttentionSlotRouterFrozenTabPFN` (the same two-hop attention-retrieval head
as the `table_slot_head_decoder*` family, but with a **frozen real TabPFN-v3**
backbone instead of a jointly-trained `NanoTabPFNModel`) pretrained for 10,000
steps on the `mix_scm` prior (job `37203197`, best validation NLL 0.347 at
step 8750), evaluated on the identical `tabarena15` protocol (job `37226114`:
same `max_predictors=30`, `subsample=2048`, 5-fold x 10-repeat CV, real
labels, same 15 eligible datasets). Folded into every table below as a 24th
model. A real-vanilla-TabPFN-v3 head-to-head comparison run (job `37228293`)
was also submitted but had not completed as of this update; see Caveats.

`table_slot_head_decoder_baseline` and `table_slot_head_decoder_alpha` are
bit-identical checkpoints under this pilot's training loss (confirmed
earlier, `reconstruction_mixture` never enters the loss) -- their TabArena
numbers below are correspondingly identical to the last digit, shown
separately for completeness, not as independent data points.

## Overall (mean over 15 datasets, 50 folds each), sorted by AUC

| model | mean_roc_auc | mean_accuracy | mean_cross_entropy |
|---|---:|---:|---:|
| random_forest | 0.7762 | 0.8208 | 0.4578 |
| attention_slot_router_frozen_tabpfn/mix_scm | 0.7670 | 0.7897 | 0.4396 |
| table_slot_head_decoder_alpha/original | 0.7631 | 0.7941 | 0.4362 |
| table_slot_head_decoder_baseline/original | 0.7631 | 0.7941 | 0.4362 |
| table_slot_head_decoder_alpha/fixed | 0.7618 | 0.7975 | 0.4301 |
| table_slot_head_decoder_baseline/fixed | 0.7618 | 0.7975 | 0.4301 |
| table_slot_head_blind_decoder/original | 0.7617 | 0.7952 | 0.4353 |
| table_slot_mufasa/curriculum | 0.7589 | 0.7984 | 0.4330 |
| table_slot_head_blind_similarity/original | 0.7569 | 0.7958 | 0.4343 |
| table_slot_backbone/curriculum | 0.7535 | 0.7958 | 0.4315 |
| table_slot_head_blind_similarity/curriculum | 0.7525 | 0.7980 | 0.4306 |
| table_slot_mufasa/fixed | 0.7522 | 0.7991 | 0.4280 |
| logreg | 0.7507 | 0.8060 | 0.4140 |
| table_slot_mufasa/original | 0.7493 | 0.7945 | 0.4384 |
| table_slot_backbone/original | 0.7492 | 0.7956 | 0.4374 |
| table_slot_head_decoder_baseline/curriculum | 0.7490 | 0.7983 | 0.4315 |
| table_slot_head_decoder_alpha/curriculum | 0.7490 | 0.7983 | 0.4315 |
| table_slot_head_blind_similarity/fixed | 0.7453 | 0.7967 | 0.4358 |
| table_slot_head_blind_decoder/fixed | 0.7447 | 0.7963 | 0.4339 |
| table_slot_head_blind_decoder/curriculum | 0.7444 | 0.7979 | 0.4307 |
| plain/fixed | 0.7305 | 0.7929 | 0.4396 |
| plain/curriculum | 0.7208 | 0.7912 | 0.4402 |
| table_slot_backbone/fixed | 0.7029 | 0.7943 | 0.4460 |
| plain/original | 0.6265 | 0.7672 | 0.4873 |

**Every table_slot cell beats every plain cell except `table_slot_backbone/fixed`**
(0.7029, below `plain/fixed`'s 0.7305 and `plain/curriculum`'s 0.7208) --
17 of 18 table_slot cells outperform plain's best. `random_forest` still
wins outright on mean AUC, but the frozen-real-TabPFN-backed slot router
(`attention_slot_router_frozen_tabpfn/mix_scm`) now sits second, ahead of
every NanoTabPFN-backed `table_slot` cell and both sklearn baselines --
despite never being trained or fine-tuned on real TabArena data itself, only
on the synthetic `mix_scm` prior. Its accuracy and cross-entropy are more
middling (see the rank-vs-AUC discussion below and the per-dataset notes),
so this AUC ranking alone somewhat overstates how usable its predictions are
out of the box.

## Mean rank across 15 datasets (1 = best AUC on that dataset that model)

Rank is more robust than mean AUC to a model winning big on a few datasets
and being middling elsewhere -- ties (the decoder_baseline/decoder_alpha
pairs) use fractional/average ranking, not an arbitrary tiebreak.

| overall_rank | model | mean_rank | mean_auc |
|---:|---|---:|---:|
| 1 | table_slot_head_blind_decoder/original | 8.43 | 0.7617 |
| 2 | random_forest | 8.47 | 0.7762 |
| 3 | table_slot_head_decoder_baseline/fixed | 8.93 | 0.7618 |
| 3 | table_slot_head_decoder_alpha/fixed | 8.93 | 0.7618 |
| 5 | table_slot_head_decoder_baseline/original | 9.27 | 0.7631 |
| 5 | table_slot_head_decoder_alpha/original | 9.27 | 0.7631 |
| 7 | table_slot_backbone/original | 9.93 | 0.7492 |
| 8 | table_slot_head_blind_similarity/original | 10.07 | 0.7569 |
| 9 | attention_slot_router_frozen_tabpfn/mix_scm | 10.40 | 0.7670 |
| 10 | table_slot_mufasa/original | 10.73 | 0.7493 |
| 11 | table_slot_head_blind_similarity/curriculum | 10.87 | 0.7525 |
| 12 | table_slot_head_blind_decoder/curriculum | 11.47 | 0.7444 |
| 13 | table_slot_mufasa/curriculum | 12.47 | 0.7589 |
| 14 | logreg | 12.83 | 0.7507 |
| 15 | table_slot_mufasa/fixed | 12.93 | 0.7522 |
| 16 | table_slot_head_decoder_baseline/curriculum | 13.13 | 0.7490 |
| 16 | table_slot_head_decoder_alpha/curriculum | 13.13 | 0.7490 |
| 18 | table_slot_head_blind_similarity/fixed | 14.20 | 0.7453 |
| 19 | table_slot_head_blind_decoder/fixed | 14.40 | 0.7447 |
| 19 | table_slot_backbone/curriculum | 14.40 | 0.7535 |
| 21 | table_slot_backbone/fixed | 15.53 | 0.7029 |
| 22 | plain/curriculum | 18.40 | 0.7208 |
| 23 | plain/fixed | 18.93 | 0.7305 |
| 24 | plain/original | 22.87 | 0.6265 |

**Adding `attention_slot_router_frozen_tabpfn/mix_scm` shuffles the top of
the table but doesn't change its shape.** `table_slot_head_blind_decoder/
original` (8.43) now edges out `random_forest` (8.47) for the top mean-rank
spot -- the same "consistently competitive beats occasionally-dominant"
pattern as before, just with a new #1. The new model itself lands at rank 9
(10.40) despite having the 2nd-best mean AUC (0.7670, behind only
`random_forest`'s 0.7762): it wins outright (rank 1 of 24) on
`Is-this-a-good-customer`, `bank-marketing`, and `credit_card_clients_default`,
but ranks 20-23 of 24 on `Bank_Customer_Churn`, `churn`, `diabetes`,
`hazelnut-spread-contaminant-detection`, and `online_shoppers_intention` --
exactly the datasets with the most skewed class balance (positive rate
6-22%). On those, precision/recall/F1 are all 0.0 at the default 0.5
threshold (specificity 1.0): the model's probability *ranking* stays
reasonable there (hence AUC only drops to ~0.73-0.85, not to chance), but its
thresholded predictions collapse to the majority class, consistent with
`mix_scm` training episodes being closer to class-balanced than these real
datasets are. High AUC, high per-dataset rank variance -- the same profile
`random_forest` has, but more extreme, and for a different underlying
reason (threshold miscalibration under real-data class imbalance rather
than a few standout wins).

## Per-dataset ROC AUC (all 24 models, sorted by mean rank)

| dataset | table_slot_head_blind_decoder/original | random_forest | table_slot_head_decoder_baseline/fixed | table_slot_head_decoder_alpha/fixed | table_slot_head_decoder_baseline/original | table_slot_head_decoder_alpha/original | table_slot_backbone/original | table_slot_head_blind_similarity/original | attention_slot_router_frozen_tabpfn/mix_scm | table_slot_mufasa/original | table_slot_head_blind_similarity/curriculum | table_slot_head_blind_decoder/curriculum | table_slot_mufasa/curriculum | logreg | table_slot_mufasa/fixed | table_slot_head_decoder_baseline/curriculum | table_slot_head_decoder_alpha/curriculum | table_slot_head_blind_similarity/fixed | table_slot_head_blind_decoder/fixed | table_slot_backbone/curriculum | table_slot_backbone/fixed | plain/curriculum | plain/fixed | plain/original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon_employee_access | 0.498 | 0.634 | 0.543 | 0.543 | 0.543 | 0.543 | 0.558 | 0.485 | 0.620 | 0.459 | 0.564 | 0.581 | 0.543 | 0.545 | 0.537 | 0.581 | 0.581 | 0.516 | 0.522 | 0.543 | 0.516 | 0.480 | 0.456 | 0.454 |
| Bank_Customer_Churn | 0.845 | 0.839 | 0.839 | 0.839 | 0.847 | 0.847 | 0.843 | 0.843 | 0.803 | 0.842 | 0.840 | 0.837 | 0.836 | 0.744 | 0.825 | 0.834 | 0.834 | 0.837 | 0.839 | 0.836 | 0.838 | 0.794 | 0.808 | 0.613 |
| E-CommereShippingData | 0.736 | 0.739 | 0.749 | 0.749 | 0.736 | 0.736 | 0.750 | 0.738 | 0.735 | 0.748 | 0.743 | 0.733 | 0.731 | 0.731 | 0.742 | 0.740 | 0.740 | 0.734 | 0.735 | 0.726 | 0.741 | 0.745 | 0.732 | 0.606 |
| Is-this-a-good-customer | 0.715 | 0.727 | 0.696 | 0.696 | 0.704 | 0.704 | 0.714 | 0.701 | 0.733 | 0.630 | 0.667 | 0.506 | 0.727 | 0.712 | 0.715 | 0.687 | 0.687 | 0.660 | 0.629 | 0.714 | 0.524 | 0.700 | 0.634 | 0.560 |
| bank-marketing | 0.710 | 0.704 | 0.737 | 0.737 | 0.715 | 0.715 | 0.721 | 0.711 | 0.741 | 0.699 | 0.668 | 0.674 | 0.716 | 0.723 | 0.618 | 0.679 | 0.679 | 0.681 | 0.705 | 0.691 | 0.499 | 0.649 | 0.615 | 0.611 |
| blood-transfusion-service-center | 0.746 | 0.680 | 0.728 | 0.728 | 0.751 | 0.751 | 0.746 | 0.746 | 0.747 | 0.726 | 0.754 | 0.754 | 0.744 | 0.751 | 0.739 | 0.744 | 0.744 | 0.729 | 0.731 | 0.723 | 0.386 | 0.754 | 0.725 | 0.601 |
| churn | 0.862 | 0.903 | 0.868 | 0.868 | 0.860 | 0.860 | 0.857 | 0.862 | 0.849 | 0.860 | 0.861 | 0.865 | 0.861 | 0.815 | 0.854 | 0.864 | 0.864 | 0.868 | 0.858 | 0.859 | 0.864 | 0.723 | 0.823 | 0.697 |
| credit-g | 0.773 | 0.781 | 0.769 | 0.769 | 0.770 | 0.770 | 0.776 | 0.773 | 0.770 | 0.770 | 0.761 | 0.763 | 0.769 | 0.731 | 0.767 | 0.763 | 0.763 | 0.770 | 0.766 | 0.766 | 0.767 | 0.751 | 0.757 | 0.580 |
| credit_card_clients_default | 0.745 | 0.737 | 0.745 | 0.745 | 0.738 | 0.738 | 0.742 | 0.739 | 0.749 | 0.734 | 0.743 | 0.744 | 0.741 | 0.721 | 0.741 | 0.743 | 0.743 | 0.732 | 0.719 | 0.726 | 0.692 | 0.720 | 0.725 | 0.607 |
| diabetes | 0.827 | 0.820 | 0.832 | 0.832 | 0.828 | 0.828 | 0.832 | 0.829 | 0.826 | 0.834 | 0.832 | 0.829 | 0.829 | 0.832 | 0.828 | 0.835 | 0.835 | 0.827 | 0.829 | 0.830 | 0.829 | 0.825 | 0.828 | 0.778 |
| hazelnut-spread-contaminant-detection | 0.913 | 0.957 | 0.900 | 0.900 | 0.914 | 0.914 | 0.895 | 0.913 | 0.893 | 0.906 | 0.907 | 0.911 | 0.895 | 0.952 | 0.899 | 0.904 | 0.904 | 0.905 | 0.902 | 0.901 | 0.895 | 0.880 | 0.880 | 0.825 |
| heloc | 0.767 | 0.764 | 0.747 | 0.747 | 0.766 | 0.766 | 0.763 | 0.761 | 0.759 | 0.767 | 0.747 | 0.758 | 0.752 | 0.759 | 0.767 | 0.751 | 0.751 | 0.757 | 0.756 | 0.765 | 0.769 | 0.758 | 0.760 | 0.533 |
| in_vehicle_coupon_recommendation | 0.646 | 0.712 | 0.612 | 0.612 | 0.638 | 0.638 | 0.412 | 0.615 | 0.657 | 0.609 | 0.547 | 0.558 | 0.612 | 0.627 | 0.616 | 0.507 | 0.507 | 0.535 | 0.532 | 0.620 | 0.610 | 0.478 | 0.624 | 0.564 |
| online_shoppers_intention | 0.891 | 0.915 | 0.896 | 0.896 | 0.891 | 0.891 | 0.882 | 0.892 | 0.859 | 0.895 | 0.893 | 0.894 | 0.894 | 0.869 | 0.885 | 0.887 | 0.887 | 0.894 | 0.896 | 0.883 | 0.875 | 0.868 | 0.880 | 0.641 |
| seismic-bumps | 0.751 | 0.731 | 0.766 | 0.766 | 0.746 | 0.746 | 0.747 | 0.746 | 0.763 | 0.760 | 0.760 | 0.759 | 0.733 | 0.748 | 0.750 | 0.716 | 0.716 | 0.734 | 0.752 | 0.719 | 0.738 | 0.687 | 0.710 | 0.728 |

Notable: `table_slot_backbone/fixed`'s poor overall average (0.7029, worst
table_slot cell) is not uniform mediocrity -- it's best-or-near-best on
`churn` (0.864) and `heloc` (0.769), but catastrophic on `bank-marketing`
(0.499, chance-level) and `blood-transfusion-service-center` (0.386, below
chance). High per-dataset variance, not consistent weakness.

`attention_slot_router_frozen_tabpfn/mix_scm` has an even more polarized
profile: best-of-all-24 on `Is-this-a-good-customer` (0.733),
`bank-marketing` (0.741), and `credit_card_clients_default` (0.749), but its
worst dataset (`Bank_Customer_Churn`, 0.803) is still well above chance --
unlike `table_slot_backbone/fixed`, it never drops to chance-level or below
on any dataset. The weakness is calibration, not ranking: see the
mean-rank section above for why those same datasets still produce
precision/recall of exactly 0.0.

## Caveats

- Single training seed per pilot cell; sklearn/random_forest fit fresh per
  fold (no seed variation issue there, but no cross-seed error bars on our
  side either).
- `decoder_baseline`/`decoder_alpha` are not independent data points (see
  above).
- TabPFN v2.2/v2.6/v3 are absent from this TabArena comparison -- the
  `--tabpfn-model-path` blocker is now fixed and synced (2026-09-14), and a
  head-to-head run against real vanilla TabPFN-v3 on this identical protocol
  was submitted (job `37228293`) but had not completed as of this update;
  update this file once it finishes rather than re-deriving the fix.
- `attention_slot_router_frozen_tabpfn/mix_scm` is a single training seed
  (same caveat as the `table_slot` cells) and, unlike them, only 10,000
  training steps rather than a matched full-length run -- its rank here is
  from one pretraining run, not an average over seeds or step budgets.
- Its TabArena inference runs the frozen TabPFN-v3 backbone on **CPU**
  regardless of `--device cuda`: `evaluate_tabarena_small.py`'s
  `load_checkpoint_for_inference` call never threads `--device` through (see
  `attention_slot_router_tabarena.py`'s
  `build_attention_slot_router_frozen_tabpfn_inference` docstring), and
  `FrozenTabPFNBackbone` isn't an `nn.Module`, so the usual per-call
  `.to(device)` re-homing that other model_kinds get is a no-op for it. Ran
  fine at `tabarena15`'s `subsample<=2048` (TabPFN's own CPU sample limit),
  just slower than a GPU-resident backbone would be.
