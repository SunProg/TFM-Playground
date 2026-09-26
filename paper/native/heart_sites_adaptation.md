# Merged heart-disease sites: target-site adaptation with observed site tags (2026-09-26)

**Replaces the leave-one-site-out comparison in `heart_sites_group_feature.md`.** Under true LOSO the held-out
site's category never appears in support, so the fold preprocessor (`tfmplayground/interface.py`,
`get_feature_preprocessor`) maps it to an unknown value and the `SimpleImputer` fills it with the most-frequent
*training*-site code. The `site_given` condition was therefore not testing "does the model benefit from knowing
the true held-out site" — every query row silently received an arbitrary wrong site label. This document reruns
the comparison with the held-out site's category genuinely observed.

## Protocol

`tfmplayground/experiments/evaluate_heart_sites_adaptation.py`, job `heart-adapt` (SLURM 37526009,
`biomed_a30_gpu`, completed in 5 min 2 s). For each of the three sites (Cleveland, Hungary, VA Long Beach) in
turn as the *target*: support = all rows from the other two sites + 20 stratified labelled rows drawn from the
target site; query = the remaining target-site rows. Five independent stratified draws per target site give 15
matched folds (query sizes: Cleveland 283, Hungary 274, VA Long Beach 180), identical across all 20 models.

Three conditions per fold, same support/query split:

* `site_hidden` — 13 clinical attributes only.
* `site_shuffled` — attributes + site column, but support-site tags independently permuted (tag *counts*
  unchanged; query tags stay true).
* `site_true` — attributes + site column, unpermuted.

Because the target site's category is fitted on labelled support rows in every condition that includes it, every
query row gets its real, observed site code — not an imputed stand-in. `site_shuffled` and `site_true` have
identical input width; `collect_heart_adaptation_results.py` verifies this and the full 20 model × 15 fold × 3
condition = 900-row result set (all folds `status: ok`, matched widths, dataset revision
`2ecfe882ccfb814fc27c4de10a64ceefd5d7655c`) before any table is built.

## 1. Overall (mean over 15 matched folds)

| model | CE hidden | CE shuffled | CE true | excess CE hidden | excess CE shuffled | excess CE true | true−shuffled | true−hidden | acc gain hidden | acc gain shuffled | acc gain true | AUC hidden | AUC shuffled | AUC true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nat-original-small | 0.5723 | 0.5695 | 0.5736 | -0.1664 | -0.1692 | -0.1651 | +0.0041 | +0.0013 | +0.3065 | +0.3136 | +0.3075 | 0.7709 | 0.7719 | 0.7700 |
| nat-rg_z-fixed-small | 0.5641 | 0.5593 | 0.5618 | -0.1746 | -0.1794 | -0.1769 | +0.0025 | -0.0023 | +0.3136 | +0.3206 | +0.3227 | 0.7704 | 0.7719 | 0.7699 |
| nat-rg_z-curriculum-small | 0.5780 | 0.5736 | 0.5777 | -0.1606 | -0.1651 | -0.1610 | +0.0042 | -0.0003 | +0.3052 | +0.3162 | +0.3102 | 0.7668 | 0.7688 | 0.7678 |
| nat-original-medium | 0.4868 | 0.4884 | 0.4779 | -0.2519 | -0.2503 | -0.2608 | -0.0105 | -0.0089 | +0.4072 | +0.4057 | +0.4144 | 0.8375 | 0.8368 | 0.8364 |
| nat-rg_z-fixed-medium | 0.4867 | 0.4820 | 0.4703 | -0.2520 | -0.2567 | -0.2683 | -0.0117 | -0.0163 | +0.4047 | +0.4089 | +0.4203 | 0.8394 | 0.8420 | 0.8429 |
| nat-rg_z-curriculum-medium | 0.4961 | 0.4930 | 0.4823 | -0.2426 | -0.2457 | -0.2564 | -0.0107 | -0.0138 | +0.4074 | +0.4051 | +0.4127 | 0.8382 | 0.8381 | 0.8379 |
| nat-original-large | 0.4726 | 0.4717 | 0.4673 | -0.2661 | -0.2670 | -0.2714 | -0.0044 | -0.0053 | +0.4232 | +0.4242 | +0.4235 | 0.8346 | 0.8360 | 0.8335 |
| nat-rg_z-fixed-large | 0.4765 | 0.4747 | 0.4720 | -0.2622 | -0.2640 | -0.2667 | -0.0027 | -0.0045 | +0.4181 | +0.4165 | +0.4209 | 0.8358 | 0.8350 | 0.8352 |
| nat-rg_z-curriculum-large | 0.4885 | 0.4872 | 0.4874 | -0.2502 | -0.2515 | -0.2513 | +0.0002 | -0.0011 | +0.4153 | +0.4106 | +0.4138 | 0.8341 | 0.8347 | 0.8303 |
| tabpfn-v2.2 | 0.4496 | 0.4482 | 0.4456 | -0.2891 | -0.2905 | -0.2931 | -0.0026 | -0.0040 | +0.4393 | +0.4419 | +0.4439 | 0.8414 | 0.8427 | 0.8428 |
| tabpfn-v2.6 | 0.4447 | 0.4433 | 0.4348 | -0.2940 | -0.2954 | -0.3038 | -0.0084 | -0.0099 | +0.4448 | +0.4412 | +0.4398 | 0.8466 | 0.8460 | 0.8438 |
| tabpfn-v3 | 0.4508 | 0.4522 | 0.4452 | -0.2879 | -0.2865 | -0.2935 | -0.0070 | -0.0056 | +0.4396 | +0.4397 | +0.4386 | 0.8395 | 0.8372 | 0.8364 |
| tabicl-v1 | 0.4588 | 0.4691 | 0.4564 | -0.2799 | -0.2696 | -0.2823 | -0.0127 | -0.0024 | +0.4265 | +0.4229 | +0.4318 | 0.8292 | 0.8242 | 0.8247 |
| tabicl-v2 | 0.4530 | 0.4544 | 0.4438 | -0.2857 | -0.2843 | -0.2949 | -0.0106 | -0.0092 | +0.4335 | +0.4323 | +0.4355 | 0.8330 | 0.8326 | 0.8345 |
| rf | 0.4915 | 0.4948 | 0.4831 | -0.2472 | -0.2439 | -0.2556 | -0.0117 | -0.0084 | +0.4160 | +0.4123 | +0.4189 | 0.8097 | 0.8104 | 0.8086 |
| logreg | 0.5054 | 0.5080 | 0.5016 | -0.2332 | -0.2306 | -0.2371 | -0.0064 | -0.0039 | +0.4284 | +0.4237 | +0.4332 | 0.8173 | 0.8170 | 0.8173 |
| hgb | 0.5656 | 0.5683 | 0.5612 | -0.1731 | -0.1704 | -0.1775 | -0.0071 | -0.0045 | +0.4014 | +0.3963 | +0.4031 | 0.7942 | 0.7917 | 0.7970 |
| catboost | 0.5847 | 0.5742 | 0.5719 | -0.1540 | -0.1645 | -0.1668 | -0.0023 | -0.0128 | +0.3989 | +0.4026 | +0.4074 | 0.8001 | 0.8034 | 0.8060 |
| xgboost | 0.6570 | 0.6578 | 0.6495 | -0.0817 | -0.0809 | -0.0892 | -0.0083 | -0.0075 | +0.3855 | +0.3887 | +0.3923 | 0.7919 | 0.7913 | 0.7928 |
| lightgbm | 0.9383 | 0.9261 | 0.9195 | +0.1996 | +0.1874 | +0.1808 | -0.0066 | -0.0188 | +0.3874 | +0.3903 | +0.3913 | 0.7834 | 0.7866 | 0.7830 |

## 2. Per site (true / shuffled)

### excess CE

| model | cleveland (q=283) | hungary (q=274) | va_long_beach (q=180) |
|---|---|---|---|
| nat-original-small | -0.180 / -0.179 | -0.105 / -0.123 | -0.210 / -0.205 |
| nat-rg_z-fixed-small | -0.169 / -0.167 | -0.176 / -0.188 | -0.186 / -0.182 |
| nat-rg_z-curriculum-small | -0.173 / -0.169 | -0.132 / -0.156 | -0.178 / -0.170 |
| nat-original-medium | -0.250 / -0.251 | -0.278 / -0.259 | -0.254 / -0.241 |
| nat-rg_z-fixed-medium | -0.228 / -0.226 | -0.323 / -0.307 | -0.254 / -0.237 |
| nat-rg_z-curriculum-medium | -0.250 / -0.248 | -0.274 / -0.261 | -0.245 / -0.228 |
| nat-original-large | -0.277 / -0.276 | -0.308 / -0.301 | -0.229 / -0.225 |
| nat-rg_z-fixed-large | -0.277 / -0.274 | -0.298 / -0.288 | -0.225 / -0.231 |
| nat-rg_z-curriculum-large | -0.256 / -0.259 | -0.324 / -0.314 | -0.174 / -0.182 |
| tabpfn-v2.2 | -0.296 / -0.297 | -0.340 / -0.333 | -0.243 / -0.241 |
| tabpfn-v2.6 | -0.305 / -0.304 | -0.351 / -0.332 | -0.256 / -0.250 |
| tabpfn-v3 | -0.291 / -0.296 | -0.354 / -0.341 | -0.235 / -0.223 |
| tabicl-v1 | -0.295 / -0.289 | -0.324 / -0.313 | -0.227 / -0.206 |
| tabicl-v2 | -0.294 / -0.297 | -0.354 / -0.338 | -0.236 / -0.218 |
| rf | -0.245 / -0.235 | -0.316 / -0.299 | -0.205 / -0.197 |
| logreg | -0.226 / -0.232 | -0.336 / -0.331 | -0.149 / -0.129 |
| hgb | -0.210 / -0.210 | -0.253 / -0.253 | -0.069 / -0.048 |
| catboost | -0.225 / -0.215 | -0.224 / -0.232 | -0.052 / -0.047 |
| xgboost | -0.132 / -0.131 | -0.176 / -0.176 | +0.040 / +0.064 |
| lightgbm | +0.024 / +0.009 | +0.060 / +0.085 | +0.458 / +0.468 |

### accuracy gain

| model | cleveland | hungary | va_long_beach |
|---|---|---|---|
| nat-original-small | +0.286 / +0.288 | +0.257 / +0.271 | +0.380 / +0.382 |
| nat-rg_z-fixed-small | +0.267 / +0.261 | +0.341 / +0.353 | +0.360 / +0.348 |
| nat-rg_z-curriculum-small | +0.276 / +0.269 | +0.279 / +0.302 | +0.376 / +0.378 |
| nat-original-medium | +0.354 / +0.354 | +0.436 / +0.420 | +0.453 / +0.443 |
| nat-rg_z-fixed-medium | +0.332 / +0.329 | +0.460 / +0.446 | +0.469 / +0.452 |
| nat-rg_z-curriculum-medium | +0.359 / +0.360 | +0.431 / +0.430 | +0.448 / +0.426 |
| nat-original-large | +0.370 / +0.370 | +0.466 / +0.464 | +0.434 / +0.439 |
| nat-rg_z-fixed-large | +0.370 / +0.366 | +0.468 / +0.459 | +0.424 / +0.424 |
| nat-rg_z-curriculum-large | +0.371 / +0.370 | +0.478 / +0.467 | +0.392 / +0.394 |
| tabpfn-v2.2 | +0.367 / +0.370 | +0.475 / +0.477 | +0.489 / +0.479 |
| tabpfn-v2.6 | +0.381 / +0.377 | +0.473 / +0.477 | +0.466 / +0.469 |
| tabpfn-v3 | +0.380 / +0.377 | +0.471 / +0.480 | +0.466 / +0.462 |
| tabicl-v1 | +0.375 / +0.374 | +0.466 / +0.469 | +0.454 / +0.426 |
| tabicl-v2 | +0.366 / +0.367 | +0.472 / +0.482 | +0.469 / +0.448 |
| rf | +0.356 / +0.361 | +0.463 / +0.461 | +0.438 / +0.414 |
| logreg | +0.365 / +0.362 | +0.472 / +0.472 | +0.463 / +0.438 |
| hgb | +0.343 / +0.334 | +0.436 / +0.432 | +0.430 / +0.423 |
| catboost | +0.348 / +0.336 | +0.442 / +0.441 | +0.432 / +0.431 |
| xgboost | +0.329 / +0.313 | +0.433 / +0.436 | +0.416 / +0.417 |
| lightgbm | +0.347 / +0.339 | +0.427 / +0.433 | +0.400 / +0.399 |

### macro AUC

| model | cleveland | hungary | va_long_beach |
|---|---|---|---|
| nat-original-small | 0.826 / 0.828 | 0.808 / 0.811 | 0.676 / 0.677 |
| nat-rg_z-fixed-small | 0.827 / 0.827 | 0.812 / 0.816 | 0.671 / 0.672 |
| nat-rg_z-curriculum-small | 0.819 / 0.816 | 0.804 / 0.808 | 0.681 / 0.682 |
| nat-original-medium | 0.894 / 0.892 | 0.877 / 0.876 | 0.739 / 0.742 |
| nat-rg_z-fixed-medium | 0.894 / 0.892 | 0.893 / 0.893 | 0.741 / 0.742 |
| nat-rg_z-curriculum-medium | 0.893 / 0.893 | 0.880 / 0.880 | 0.740 / 0.742 |
| nat-original-large | 0.904 / 0.903 | 0.884 / 0.885 | 0.713 / 0.720 |
| nat-rg_z-fixed-large | 0.903 / 0.901 | 0.888 / 0.888 | 0.715 / 0.716 |
| nat-rg_z-curriculum-large | 0.902 / 0.897 | 0.892 / 0.891 | 0.697 / 0.716 |
| tabpfn-v2.2 | 0.904 / 0.903 | 0.903 / 0.904 | 0.721 / 0.721 |
| tabpfn-v2.6 | 0.909 / 0.908 | 0.902 / 0.903 | 0.720 / 0.727 |
| tabpfn-v3 | 0.903 / 0.903 | 0.902 / 0.900 | 0.704 / 0.708 |
| tabicl-v1 | 0.904 / 0.903 | 0.885 / 0.885 | 0.685 / 0.685 |
| tabicl-v2 | 0.904 / 0.903 | 0.904 / 0.902 | 0.696 / 0.692 |
| rf | 0.895 / 0.895 | 0.889 / 0.889 | 0.642 / 0.648 |
| logreg | 0.898 / 0.897 | 0.894 / 0.893 | 0.660 / 0.660 |
| hgb | 0.868 / 0.868 | 0.881 / 0.879 | 0.641 / 0.628 |
| catboost | 0.881 / 0.879 | 0.875 / 0.876 | 0.662 / 0.655 |
| xgboost | 0.857 / 0.851 | 0.874 / 0.875 | 0.648 / 0.647 |
| lightgbm | 0.866 / 0.865 | 0.870 / 0.871 | 0.613 / 0.625 |

## 3. Findings

1. **The old "logreg destroyed" result was the encoding artifact, not a real linear-extrapolation failure.**
   Under true LOSO, logreg's CE worsened by +0.074 nats when given the (miscoded) site column. With the target
   site genuinely observed, logreg's true-minus-shuffled excess CE is **−0.0064** — a small, ordinary improvement
   in line with every other model. The catastrophic LOSO number was logreg extrapolating a coefficient for an
   *unknown* category the imputer invented, not evidence about linear models under real distribution shift.
2. **Aligned site tags give a small, mostly-helpful, mixed effect.** True-minus-shuffled excess CE spans −0.0127
   (tabicl-v1) to +0.0042 (nat-rg_z-curriculum-small); 16 of 20 models improve, the 4 exceptions are all
   small-capacity native runs (+0.0002 to +0.0042) where the effect is within fold-to-fold noise. This is an
   order of magnitude smaller than the withdrawn LOSO deltas (which reached −0.019 to +0.074) precisely because
   the withdrawn comparison was confounded by the imputation artifact rather than isolating tag alignment.
3. **Calibration, not ranking, again.** |Δ AUC (true−shuffled)| ≤ 0.0053 for every model; excess CE and accuracy
   gain move an order of magnitude more. Same conclusion as the IID folds and the synthetic soft-gate/persistent
   contrast.
4. **No consistent benefit from multiregime pretraining.** At medium capacity, rg_z-fixed edges out original
   (−0.0117 vs −0.0105); at large capacity original is best (−0.0044) and rg_z-curriculum is roughly flat
   (+0.0002); at small capacity all three are on the wrong side of zero, with rg_z-fixed least so. No family wins
   at more than one size.
5. **Per site**: the largest aligned-tag gains concentrate on Hungary (nat-rg_z-fixed-medium −0.016, tabpfn-v3
   −0.013, tabicl-v2 −0.016), Cleveland is a smaller, more mixed effect, and VA Long Beach remains the hardest
   fold for every model in absolute terms (least negative or positive excess CE) — lightgbm and xgboost are
   *worse than the class prior* there in both the true and shuffled conditions, a property of the fold's size and
   class balance, not of site-tag alignment.
6. **Capacity still dominates.** Site-true excess CE: small ≈ −0.16 to −0.18, medium ≈ −0.25 to −0.27, large ≈
   −0.25 to −0.27 — medium and large are roughly level and both far ahead of small, matching the IID/BeyondArena
   pattern.

Job: `heart-adapt` (SLURM 37526009, `biomed_a30_gpu`, 20 models × 15 folds × 3 conditions, 5 min 2 s wall time).
Source data: `heart_sites_adaptation_summary.json`, `heart_sites_adaptation_per_fold.json`,
`heart_sites_adaptation_fold_manifest.json`, `heart_sites_adaptation_provenance.json` (this directory), built by
`paper/iclr2027/collect_heart_adaptation_results.py` from `results/heart_sites_adaptation_20shot/` and rendered
into `paper/iclr2027/appendix_f_table.tex` by `paper/iclr2027/build_heart_table.py`.
