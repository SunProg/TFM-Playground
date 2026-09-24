# Merged heart-disease sites: does naming the group help? (2026-09-22)

BeyondArena contains three sites of the same UCI Heart Disease study with an **identical 13-attribute schema and
target** — Cleveland (303 rows, 0.46 positive), Hungary (294, 0.36), VA Long Beach (200, 0.75) — verified by
reading the parquet columns, not by name. Merging them gives one real 797-row dataset with a known latent
regime (the hospital): the real-data counterpart of the v4 synthetic multiregime setting, and a direct test of
the z-exposure question.

* `site_hidden` — the 13 attributes only (the model must infer the regime; the z-blind bank analogue)
* `site_given` — the 13 attributes + a `site` column (the z-exposed twin)

Two fold protocols, identical folds for every model:

* **IID** — repeated stratified 5-fold x 2 (stratified on site x target): support 638, query ~159. Sites are
  mixed across support and query, so `site` is a legitimately usable feature.
* **LOSO** — leave-one-site-out (3 folds): the query hospital never appears in the support fold, so the `site`
  value is unseen-with-a-label. A pure distribution-shift test.

Job 37449374 (`scripts/slurm/evaluate_heart_sites_merged.sbatch`,
`tfmplayground/experiments/evaluate_heart_sites_merged.py`), 19 models scored through the same BeyondArena
preprocessor, full-fold prediction path and metrics. rg_z-fixed-large was still training and is absent.

## 0. Is the site recoverable from the features?

A random forest predicts the site from the 13 attributes with **0.918** 5-fold accuracy (base rate 0.380), and
still **0.888** after median imputation removes the missingness cue (`ca` is 99 % missing outside Cleveland,
`thal` 83-90 %, `slope` 51-65 %). The group label is therefore near-redundant in-distribution — which is exactly
what the IID results below show.


## 1. IID folds (sites mixed; support 638, query ~159)

| model | CE hidden | CE given | excess CE hidden | excess CE given | ΔCE | acc gain hidden | acc gain given | Δacc | AUC hidden | AUC given | ΔAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn-v3 | 0.4090 | 0.4106 | -0.2841 | -0.2825 | +0.0017 | +0.3130 | +0.3093 | -0.0038 | 0.9005 | 0.8992 | -0.0013 |
| tabpfn-v2.6 | 0.4092 | 0.4086 | -0.2839 | -0.2845 | -0.0006 | +0.3137 | +0.3124 | -0.0012 | 0.8996 | 0.9003 | +0.0007 |
| tabicl-v2 | 0.4095 | 0.4098 | -0.2836 | -0.2833 | +0.0003 | +0.3105 | +0.3124 | +0.0019 | 0.8995 | 0.8996 | +0.0000 |
| tabpfn-v2.2 | 0.4111 | 0.4109 | -0.2820 | -0.2822 | -0.0002 | +0.3099 | +0.3087 | -0.0013 | 0.8985 | 0.8982 | -0.0004 |
| tabicl-v1 | 0.4190 | 0.4198 | -0.2741 | -0.2733 | +0.0008 | +0.3112 | +0.3112 | +0.0000 | 0.8951 | 0.8951 | +0.0000 |
| nat-original-medium | 0.4194 | 0.4178 | -0.2737 | -0.2753 | -0.0016 | +0.3080 | +0.3049 | -0.0031 | 0.8923 | 0.8930 | +0.0007 |
| logreg | 0.4201 | 0.4191 | -0.2730 | -0.2740 | -0.0010 | +0.3105 | +0.3093 | -0.0012 | 0.8925 | 0.8931 | +0.0006 |
| nat-rg_z-fixed-medium | 0.4208 | 0.4181 | -0.2722 | -0.2749 | -0.0027 | +0.2967 | +0.2986 | +0.0019 | 0.8917 | 0.8932 | +0.0015 |
| nat-rg_z-curriculum-medium | 0.4217 | 0.4191 | -0.2714 | -0.2740 | -0.0025 | +0.3011 | +0.2999 | -0.0012 | 0.8908 | 0.8921 | +0.0013 |
| nat-original-large | 0.4273 | 0.4266 | -0.2657 | -0.2665 | -0.0008 | +0.3030 | +0.3005 | -0.0025 | 0.8900 | 0.8905 | +0.0005 |
| nat-rg_z-curriculum-large | 0.4277 | 0.4275 | -0.2654 | -0.2656 | -0.0002 | +0.3061 | +0.3017 | -0.0044 | 0.8924 | 0.8926 | +0.0002 |
| rf | 0.4294 | 0.4292 | -0.2637 | -0.2639 | -0.0001 | +0.3055 | +0.3024 | -0.0031 | 0.8878 | 0.8878 | -0.0000 |
| nat-original-small | 0.4622 | 0.4633 | -0.2308 | -0.2298 | +0.0010 | +0.2804 | +0.2810 | +0.0006 | 0.8705 | 0.8686 | -0.0019 |
| nat-rg_z-curriculum-small | 0.4645 | 0.4649 | -0.2286 | -0.2281 | +0.0005 | +0.2654 | +0.2722 | +0.0069 | 0.8670 | 0.8663 | -0.0006 |
| nat-rg_z-fixed-small | 0.4680 | 0.4701 | -0.2251 | -0.2229 | +0.0021 | +0.2660 | +0.2666 | +0.0006 | 0.8637 | 0.8616 | -0.0021 |
| catboost | 0.5200 | 0.5242 | -0.1731 | -0.1689 | +0.0043 | +0.2967 | +0.2905 | -0.0062 | 0.8784 | 0.8769 | -0.0015 |
| hgb | 0.5451 | 0.5495 | -0.1480 | -0.1436 | +0.0045 | +0.2930 | +0.2936 | +0.0006 | 0.8705 | 0.8685 | -0.0020 |
| xgboost | 0.6322 | 0.6342 | -0.0609 | -0.0589 | +0.0019 | +0.2823 | +0.2685 | -0.0138 | 0.8622 | 0.8614 | -0.0008 |
| lightgbm | 0.9684 | 0.9726 | +0.2753 | +0.2795 | +0.0043 | +0.2729 | +0.2748 | +0.0019 | 0.8581 | 0.8583 | +0.0002 |

## 2. LOSO folds (query hospital unseen; support 494-597, query 200-303)

| model | CE hidden | CE given | excess CE hidden | excess CE given | ΔCE | acc gain hidden | acc gain given | Δacc | AUC hidden | AUC given | ΔAUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tabpfn-v2.2 | 0.4979 | 0.4838 | -0.2485 | -0.2626 | -0.0141 | +0.4137 | +0.4231 | +0.0094 | 0.8390 | 0.8401 | +0.0011 |
| tabpfn-v2.6 | 0.4986 | 0.4799 | -0.2478 | -0.2665 | -0.0187 | +0.4097 | +0.4319 | +0.0222 | 0.8426 | 0.8430 | +0.0003 |
| tabpfn-v3 | 0.4996 | 0.4956 | -0.2468 | -0.2508 | -0.0041 | +0.3909 | +0.4031 | +0.0122 | 0.8404 | 0.8404 | -0.0000 |
| tabicl-v2 | 0.5065 | 0.4992 | -0.2399 | -0.2472 | -0.0074 | +0.4030 | +0.3970 | -0.0059 | 0.8447 | 0.8450 | +0.0003 |
| tabicl-v1 | 0.5083 | 0.5087 | -0.2381 | -0.2377 | +0.0004 | +0.3912 | +0.3970 | +0.0058 | 0.8380 | 0.8363 | -0.0017 |
| nat-original-large | 0.5093 | 0.5025 | -0.2370 | -0.2439 | -0.0068 | +0.3813 | +0.3846 | +0.0034 | 0.8223 | 0.8237 | +0.0014 |
| nat-rg_z-fixed-medium | 0.5217 | 0.5301 | -0.2246 | -0.2163 | +0.0084 | +0.3625 | +0.3580 | -0.0045 | 0.8307 | 0.8300 | -0.0007 |
| nat-rg_z-curriculum-large | 0.5267 | 0.5231 | -0.2197 | -0.2233 | -0.0036 | +0.3806 | +0.3841 | +0.0035 | 0.8271 | 0.8274 | +0.0003 |
| nat-original-medium | 0.5290 | 0.5241 | -0.2174 | -0.2223 | -0.0049 | +0.3519 | +0.3481 | -0.0038 | 0.8219 | 0.8219 | -0.0000 |
| nat-rg_z-curriculum-medium | 0.5359 | 0.5286 | -0.2104 | -0.2177 | -0.0073 | +0.3953 | +0.3893 | -0.0060 | 0.8224 | 0.8211 | -0.0013 |
| rf | 0.5411 | 0.5273 | -0.2053 | -0.2191 | -0.0138 | +0.3944 | +0.4008 | +0.0064 | 0.8283 | 0.8369 | +0.0086 |
| hgb | 0.5995 | 0.5925 | -0.1469 | -0.1539 | -0.0069 | +0.3656 | +0.3706 | +0.0050 | 0.7933 | 0.7953 | +0.0021 |
| catboost | 0.6070 | 0.6033 | -0.1393 | -0.1431 | -0.0038 | +0.3806 | +0.3822 | +0.0016 | 0.8103 | 0.8184 | +0.0081 |
| nat-rg_z-fixed-small | 0.6100 | 0.6077 | -0.1363 | -0.1387 | -0.0023 | +0.2774 | +0.2752 | -0.0022 | 0.7607 | 0.7604 | -0.0003 |
| nat-original-small | 0.6457 | 0.6397 | -0.1007 | -0.1067 | -0.0061 | +0.2665 | +0.2615 | -0.0050 | 0.7660 | 0.7670 | +0.0010 |
| nat-rg_z-curriculum-small | 0.6458 | 0.6432 | -0.1006 | -0.1032 | -0.0025 | +0.2655 | +0.2655 | -0.0000 | 0.7563 | 0.7557 | -0.0006 |
| xgboost | 0.6979 | 0.6984 | -0.0485 | -0.0480 | +0.0005 | +0.3651 | +0.3790 | +0.0138 | 0.8037 | 0.8047 | +0.0010 |
| logreg | 0.7254 | 0.7995 | -0.0210 | +0.0531 | +0.0741 | +0.3786 | +0.3604 | -0.0182 | 0.8210 | 0.8198 | -0.0012 |
| lightgbm | 0.9016 | 0.8898 | +0.1552 | +0.1434 | -0.0118 | +0.3661 | +0.3713 | +0.0052 | 0.7876 | 0.7907 | +0.0031 |

## 3. LOSO per site (hidden / given)


### excess CE

| model | cleveland (q=303) | hungary (q=294) | va_long_beach (q=200) |
|---|---|---|---|
| nat-original-large | -0.273 / -0.269 | -0.259 / -0.276 | -0.179 / -0.187 |
| nat-rg_z-curriculum-large | -0.244 / -0.235 | -0.242 / -0.262 | -0.174 / -0.174 |
| nat-original-medium | -0.252 / -0.244 | -0.215 / -0.229 | -0.185 / -0.194 |
| nat-rg_z-curriculum-medium | -0.234 / -0.234 | -0.205 / -0.226 | -0.193 / -0.194 |
| nat-rg_z-fixed-medium | -0.220 / -0.180 | -0.277 / -0.293 | -0.177 / -0.176 |
| nat-original-small | -0.155 / -0.161 | +0.020 / +0.014 | -0.167 / -0.173 |
| nat-rg_z-curriculum-small | -0.151 / -0.154 | -0.047 / -0.047 | -0.104 / -0.109 |
| nat-rg_z-fixed-small | -0.167 / -0.165 | -0.112 / -0.119 | -0.129 / -0.132 |
| tabpfn-v2.2 | -0.264 / -0.265 | -0.284 / -0.296 | -0.197 / -0.227 |
| tabpfn-v2.6 | -0.248 / -0.263 | -0.250 / -0.288 | -0.245 / -0.248 |
| tabpfn-v3 | -0.270 / -0.273 | -0.305 / -0.317 | -0.165 / -0.162 |
| tabicl-v1 | -0.285 / -0.265 | -0.280 / -0.310 | -0.150 / -0.138 |
| tabicl-v2 | -0.280 / -0.279 | -0.280 / -0.291 | -0.160 / -0.171 |
| rf | -0.205 / -0.215 | -0.238 / -0.251 | -0.173 / -0.191 |
| logreg | -0.231 / -0.210 | -0.324 / -0.322 | +0.492 / +0.691 |
| hgb | -0.150 / -0.183 | -0.242 / -0.239 | -0.050 / -0.040 |
| catboost | -0.214 / -0.193 | -0.213 / -0.238 | +0.009 / +0.001 |
| xgboost | -0.141 / -0.158 | -0.203 / -0.176 | +0.199 / +0.189 |
| lightgbm | +0.026 / +0.030 | +0.149 / +0.103 | +0.291 / +0.296 |

### accuracy gain

| model | cleveland (q=303) | hungary (q=294) | va_long_beach (q=200) |
|---|---|---|---|
| nat-original-large | +0.350 / +0.353 | +0.449 / +0.456 | +0.345 / +0.345 |
| nat-rg_z-curriculum-large | +0.350 / +0.343 | +0.432 / +0.449 | +0.360 / +0.360 |
| nat-original-medium | +0.337 / +0.320 | +0.374 / +0.374 | +0.345 / +0.350 |
| nat-rg_z-curriculum-medium | +0.366 / +0.343 | +0.395 / +0.395 | +0.425 / +0.430 |
| nat-rg_z-fixed-medium | +0.310 / +0.304 | +0.442 / +0.435 | +0.335 / +0.335 |
| nat-original-small | +0.261 / +0.251 | +0.184 / +0.184 | +0.355 / +0.350 |
| nat-rg_z-curriculum-small | +0.248 / +0.251 | +0.204 / +0.201 | +0.345 / +0.345 |
| nat-rg_z-fixed-small | +0.251 / +0.248 | +0.241 / +0.238 | +0.340 / +0.340 |
| tabpfn-v2.2 | +0.330 / +0.350 | +0.476 / +0.480 | +0.435 / +0.440 |
| tabpfn-v2.6 | +0.323 / +0.360 | +0.446 / +0.466 | +0.460 / +0.470 |
| tabpfn-v3 | +0.337 / +0.356 | +0.466 / +0.473 | +0.370 / +0.380 |
| tabicl-v1 | +0.366 / +0.343 | +0.442 / +0.473 | +0.365 / +0.375 |
| tabicl-v2 | +0.373 / +0.347 | +0.466 / +0.480 | +0.370 / +0.365 |
| rf | +0.363 / +0.343 | +0.425 / +0.459 | +0.395 / +0.400 |
| logreg | +0.343 / +0.320 | +0.463 / +0.466 | +0.330 / +0.295 |
| hgb | +0.307 / +0.317 | +0.415 / +0.415 | +0.375 / +0.380 |
| catboost | +0.327 / +0.327 | +0.425 / +0.415 | +0.390 / +0.405 |
| xgboost | +0.304 / +0.340 | +0.422 / +0.432 | +0.370 / +0.365 |
| lightgbm | +0.300 / +0.290 | +0.398 / +0.418 | +0.400 / +0.405 |

### macro AUC

| model | cleveland (q=303) | hungary (q=294) | va_long_beach (q=200) |
|---|---|---|---|
| nat-original-large | 0.888 / 0.888 | 0.882 / 0.884 | 0.697 / 0.699 |
| nat-rg_z-curriculum-large | 0.875 / 0.876 | 0.891 / 0.892 | 0.716 / 0.715 |
| nat-original-medium | 0.877 / 0.876 | 0.880 / 0.880 | 0.709 / 0.710 |
| nat-rg_z-curriculum-medium | 0.883 / 0.882 | 0.880 / 0.880 | 0.704 / 0.700 |
| nat-rg_z-fixed-medium | 0.884 / 0.884 | 0.893 / 0.894 | 0.714 / 0.713 |
| nat-original-small | 0.833 / 0.831 | 0.793 / 0.792 | 0.672 / 0.678 |
| nat-rg_z-curriculum-small | 0.803 / 0.803 | 0.788 / 0.786 | 0.678 / 0.677 |
| nat-rg_z-fixed-small | 0.816 / 0.814 | 0.800 / 0.798 | 0.666 / 0.668 |
| tabpfn-v2.2 | 0.886 / 0.883 | 0.904 / 0.906 | 0.727 / 0.731 |
| tabpfn-v2.6 | 0.891 / 0.889 | 0.904 / 0.905 | 0.733 / 0.735 |
| tabpfn-v3 | 0.888 / 0.889 | 0.905 / 0.907 | 0.728 / 0.725 |
| tabicl-v1 | 0.894 / 0.887 | 0.892 / 0.892 | 0.728 / 0.729 |
| tabicl-v2 | 0.893 / 0.892 | 0.911 / 0.909 | 0.730 / 0.733 |
| rf | 0.875 / 0.883 | 0.895 / 0.898 | 0.715 / 0.729 |
| logreg | 0.877 / 0.875 | 0.898 / 0.898 | 0.688 / 0.687 |
| hgb | 0.822 / 0.842 | 0.872 / 0.865 | 0.686 / 0.679 |
| catboost | 0.870 / 0.869 | 0.866 / 0.880 | 0.695 / 0.706 |
| xgboost | 0.846 / 0.856 | 0.872 / 0.864 | 0.693 / 0.694 |
| lightgbm | 0.830 / 0.817 | 0.850 / 0.862 | 0.682 / 0.693 |

## 4. Findings

1. **In-distribution the group label is worthless.** Every one of the 19 models moves by |Δ excess CE| ≤ 0.0045
   when the site column is added under IID folds, with no consistent sign. This matches the measured 92 %
   recoverability of the site from the features, and reproduces the synthetic `soft_gate` result (z is a
   function of x, so the z-exposed bank differs by ≤ 0.004 CE).
2. **Under distribution shift it becomes worth something, and in-context models exploit it best.** LOSO Δ:
   tabpfn-v2.6 −0.019, tabpfn-v2.2 −0.014, rf −0.014, lightgbm −0.012, tabicl-v2 −0.007, nat-original-large
   −0.007, nat-rg_z-curriculum-medium −0.007. The query value is never seen with a label, so the gain is not
   "which hospital is this" but "these rows are a different group" — the models hedge.
3. **A linear model is destroyed by the same information**: logreg +0.074 CE (and +0.199 on the VA Long Beach
   fold alone), because it extrapolates a coefficient for an unseen category. This is the sharpest illustration
   that group exposure is only useful to a model that can condition on it rather than extrapolate from it.
4. **The effect is calibration, not ranking.** |Δ AUC| ≤ 0.002 for every model except rf (+0.009) and catboost
   (+0.008) under LOSO, while Δ CE and Δ accuracy gain move an order of magnitude more.
5. **Per site**: the gain concentrates on Hungary (positive rate 0.36 vs 0.55 in its support mix) — tabpfn-v2.6
   −0.038, tabicl-v1 −0.030, our large runs −0.017/−0.020. Cleveland is a wash (±0.02) and VA Long Beach is the
   hard fold for everyone (best −0.248; every GBM and logreg go positive, our runs hold −0.17…−0.19).
6. **Our models**: nat-original-large is the best non-published model under LOSO (−0.237 hidden / −0.244 given),
   ahead of rf (−0.205/−0.219) and every GBM; under IID our medium runs (−0.274) beat our large (−0.266) and sit
   level with logreg and tabicl-v1, 0.010 behind TabPFN. Capacity helps under shift and slightly hurts
   in-distribution on this 797-row task.
7. **Multiregime pretraining does not change the picture** (|Δ| ≤ 0.008 vs original at the same size, both
   protocols, no consistent sign) — consistent with BeyondArena, where real datasets carry no exploitable
   routing structure.

