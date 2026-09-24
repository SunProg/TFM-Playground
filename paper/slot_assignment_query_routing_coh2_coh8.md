# Query cross-entropy and ROC AUC: coh2/coh8 x six routing arms, 18 TabArena tasks

Mean over the 15 binary + 3 three-class TabArena tasks in `artifacts/slot-assignment-distributions/tabarena-18/tasks.csv`. ROC AUC is macro one-vs-rest on the three-class tasks, ordinary binary AUC on the rest -- see `prediction_metrics` in `tfmplayground/experiments/slot_assignment_distributions.py`.

## Training provenance and multiregime CE (pulled from cluster checkpoints)

The step counts and `multiregime_cross_entropy` below come directly from `torch.load(...)["step"]` / `["validation"]` on the actual `final_checkpoint.pth` files under `runs/slot_tabpfn/<STUDY_ID>/` on CREATE (not present in this git worktree) -- not from the sweep's *submitted* `--max-steps`, which is 5000 for every cell in this block.

**Important caveat:** every coh8 cell stopped at exactly 5000 steps as originally submitted. Every coh2 cell was resumed past that -- to 10,000 or 15,000 steps depending on the arm -- via `resume_slot_tabpfn_to_20k_a30.sbatch`. So coh2 and coh8 differ in training length as well as in `--regime-coherence`, which confounds the coh2-vs-coh8 comparison throughout this document.

| model | prior | coh | actual step | multiregime CE | synthetic query CE | gate_regime_auc |
|---|---|---|---:|---:|---:|---:|
| `decoder-attention` | plain | coh2 | 10000 | 0.4908 | 0.5290 | 0.5293 |
| `decoder-attention` | mixed | coh2 | 15000 | 0.4838 | 0.5260 | 0.5275 |
| `decoder-attention` | curriculum | coh2 | 15000 | 0.4835 | 0.5280 | 0.5367 |
| `decoder-attention` | plain | coh8 | 5000 | 0.4882 | 0.5318 | 0.5385 |
| `decoder-attention` | mixed | coh8 | 5000 | 0.4835 | 0.5343 | 0.5428 |
| `decoder-attention` | curriculum | coh8 | 5000 | 0.4832 | 0.5352 | 0.5294 |
| `decoder-alpha` | plain | coh2 | 10000 | 0.4877 | 0.5304 | 0.5400 |
| `decoder-alpha` | mixed | coh2 | 15000 | 0.4861 | 0.5315 | 0.5252 |
| `decoder-alpha` | curriculum | coh2 | 10000 | 0.4833 | 0.5360 | 0.5409 |
| `decoder-alpha` | plain | coh8 | 5000 | 0.4858 | 0.5354 | 0.5383 |
| `decoder-alpha` | mixed | coh8 | 5000 | 0.4830 | 0.5408 | 0.5519 |
| `decoder-alpha` | curriculum | coh8 | 5000 | 0.4825 | 0.5436 | 0.5888 |
| `blind_decoder-attention` | plain | coh2 | 10000 | 0.4882 | 0.5245 | 0.5081 |
| `blind_decoder-attention` | mixed | coh2 | 10000 | 0.4836 | 0.5273 | 0.5102 |
| `blind_decoder-attention` | curriculum | coh2 | 15000 | 0.4839 | 0.5269 | 0.5114 |
| `blind_decoder-attention` | plain | coh8 | 5000 | 0.4865 | 0.5297 | 0.5063 |
| `blind_decoder-attention` | mixed | coh8 | 5000 | 0.4845 | 0.5294 | 0.5142 |
| `blind_decoder-attention` | curriculum | coh8 | 5000 | 0.4820 | 0.5335 | 0.5041 |
| `blind_decoder-alpha` | plain | coh2 | 10000 | 0.4872 | 0.5297 | 0.5594 |
| `blind_decoder-alpha` | mixed | coh2 | 10000 | 0.4844 | 0.5330 | 0.5451 |
| `blind_decoder-alpha` | curriculum | coh2 | 10000 | 0.4846 | 0.5383 | 0.5398 |
| `blind_decoder-alpha` | plain | coh8 | 5000 | 0.4866 | 0.5379 | 0.5442 |
| `blind_decoder-alpha` | mixed | coh8 | 5000 | 0.4839 | 0.5416 | 0.5447 |
| `blind_decoder-alpha` | curriculum | coh8 | 5000 | 0.4831 | 0.5443 | 0.5482 |
| `blind_similarity` | plain | coh2 | 15000 | 0.4906 | 0.5268 | 0.5365 |
| `blind_similarity` | mixed | coh2 | 15000 | 0.4851 | 0.5260 | 0.5497 |
| `blind_similarity` | curriculum | coh2 | 15000 | 0.4853 | 0.5255 | 0.5347 |
| `blind_similarity` | plain | coh8 | 5000 | 0.4869 | 0.5301 | 0.5759 |
| `blind_similarity` | mixed | coh8 | 5000 | 0.4841 | 0.5323 | 0.5545 |
| `blind_similarity` | curriculum | coh8 | 5000 | 0.4833 | 0.5406 | 0.6060 |
| `vanilla` | plain | coh2 | 10000 | 0.4865 | 0.5263 | -- |
| `vanilla` | mixed | coh2 | 10000 | 0.4828 | 0.5293 | -- |
| `vanilla` | curriculum | coh2 | 10000 | 0.4845 | 0.5290 | -- |
| `vanilla` | plain | coh8 | 5000 | 0.4870 | 0.5317 | -- |
| `vanilla` | mixed | coh8 | 5000 | 0.4840 | 0.5341 | -- |
| `vanilla` | curriculum | coh8 | 5000 | 0.4832 | 0.5362 | -- |

Notes:

- `multiregime_cross_entropy` and the "synthetic query CE" columns are measured on held-out synthetic `mix_scm` validation episodes at training time, via the checkpoint's own `validation` dict -- they are a different metric from the `query_cross_entropy` / `query_roc_auc` columns elsewhere in this document, which come from real TabArena tasks.
- Chance on the (binary) regime tag is `ln(2) ≈ 0.693`; every cell sits at 0.48-0.49, well below chance, confirming the `LEARNABLE_DESIGN` task (3 classes, 4 features, 15% contamination) is genuinely learnable here, unlike the `regime_coherence=2.0` block on the original mix_scm design (`table_slot_head/backbone/mufasa-s4-coh2`, indices 114-117), which sat at chance (~0.66).
- `multiregime_cross_entropy` barely varies across routing arms (0.483-0.491) -- the choice of query-routing mechanism does not appear to move this metric; the routing arms differentiate mainly on the real-task `query_cross_entropy`/`query_roc_auc`/NMI numbers reported above, not here.
- `vanilla` has no slot gate, so `gate_regime_auc` is undefined (NaN in the checkpoint).

## Query cross-entropy (mean over tasks; lower is better)

| model | plain-coh2 | mixed-coh2 | curriculum-coh2 | plain-coh8 | mixed-coh8 | curriculum-coh8 |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla` | 0.4515 | 0.4414 | 0.4403 | 0.4729 | 0.4532 | 0.4521 |
| `decoder-alpha` | 0.4593 | 0.4360 | 0.4488 | 0.4913 | 0.4457 | 0.4487 |
| `decoder-attention` | 0.4455 | 0.4201 | 0.4215 | 0.4778 | 0.4355 | 0.4458 |
| `blind_decoder-alpha` | 0.4694 | 0.4369 | 0.4432 | 0.4989 | 0.4459 | 0.4556 |
| `blind_decoder-attention` | 0.4660 | 0.4266 | 0.4282 | 0.5533 | 0.4252 | 0.4494 |
| `blind_similarity` | 0.4149 | 0.4127 | 0.4136 | 0.4655 | 0.4403 | 0.4381 |

## Query ROC AUC (mean over tasks; higher is better)

| model | plain-coh2 | mixed-coh2 | curriculum-coh2 | plain-coh8 | mixed-coh8 | curriculum-coh8 |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla` | 0.7901 | 0.8002 | 0.7961 | 0.7726 | 0.7975 | 0.7919 |
| `decoder-alpha` | 0.7833 | 0.7985 | 0.7981 | 0.7821 | 0.7921 | 0.7911 |
| `decoder-attention` | 0.7906 | 0.8125 | 0.8064 | 0.7817 | 0.7977 | 0.7961 |
| `blind_decoder-alpha` | 0.7788 | 0.7980 | 0.8035 | 0.7774 | 0.7884 | 0.7905 |
| `blind_decoder-attention` | 0.8008 | 0.8095 | 0.8125 | 0.7769 | 0.8070 | 0.7948 |
| `blind_similarity` | 0.8039 | 0.8111 | 0.8113 | 0.7985 | 0.8033 | 0.7994 |

## Support hard-slot NMI with label (mean over tasks)

| model | plain-coh2 | mixed-coh2 | curriculum-coh2 | plain-coh8 | mixed-coh8 | curriculum-coh8 |
|---|---:|---:|---:|---:|---:|---:|
| `vanilla` | -- | -- | -- | -- | -- | -- |
| `decoder-alpha` | 0.2699 | 0.0426 | 0.2531 | 0.4135 | 0.0585 | 0.2867 |
| `decoder-attention` | 0.9182 | 0.9501 | 0.9728 | 0.9257 | 0.9617 | 0.9922 |
| `blind_decoder-alpha` | 0.0316 | 0.0301 | 0.0314 | 0.0333 | 0.0333 | 0.0339 |
| `blind_decoder-attention` | 0.0256 | 0.0352 | 0.0230 | 0.0106 | 0.0041 | 0.0060 |
| `blind_similarity` | 0.0271 | 0.0328 | 0.0322 | 0.0348 | 0.0108 | 0.0306 |

## Rank analysis

Model ranking is computed by ranking the 6 routing arms against each other within each comparison cell (1 = best), then averaging that rank. Two task sets are used: all 18 TabArena tasks, and the 15 binary-only subset (excludes the three 3-class tasks `maternal_health_risk`, `SDSS17`, `website_phishing`).

### Overall average rank (18 tasks x 6 prior conditions = 108 cells, 1 = best)

| model | mean CE rank | mean AUC rank |
|---|---:|---:|
| `blind_similarity` | 2.36 | 2.61 |
| `decoder-attention` | 2.73 | 3.44 |
| `blind_decoder-attention` | 3.75 | 2.70 |
| `vanilla` | 3.98 | 3.93 |
| `decoder-alpha` | 4.03 | 4.16 |
| `blind_decoder-alpha` | 4.15 | 4.16 |

### coh2 average rank (18 tasks x 3 prior-type cells = 54 cells, 1 = best)

| model | mean CE rank | mean AUC rank |
|---|---:|---:|
| `blind_similarity` | 2.28 | 2.65 |
| `decoder-attention` | 2.56 | 3.09 |
| `blind_decoder-attention` | 3.74 | 2.52 |
| `vanilla` | 4.09 | 3.96 |
| `blind_decoder-alpha` | 4.17 | 4.39 |
| `decoder-alpha` | 4.17 | 4.39 |

### coh8 average rank (18 tasks x 3 prior-type cells = 54 cells, 1 = best)

| model | mean CE rank | mean AUC rank |
|---|---:|---:|
| `blind_similarity` | 2.44 | 2.57 |
| `decoder-attention` | 2.91 | 3.80 |
| `blind_decoder-attention` | 3.76 | 2.89 |
| `vanilla` | 3.87 | 3.89 |
| `decoder-alpha` | 3.89 | 3.93 |
| `blind_decoder-alpha` | 4.13 | 3.93 |

### 15 binary tasks only (excludes `maternal_health_risk`, `SDSS17`, `website_phishing`)

#### Mean query CE, 15 binary tasks (lower is better)

| model | plain-coh2 | mixed-coh2 | curriculum-coh2 | plain-coh8 | mixed-coh8 | curriculum-coh8 |
|---|---:|---:|---:|---:|---:|---:|
| `blind_decoder-alpha` | 0.4456 | 0.4079 | 0.4052 | 0.4637 | 0.4046 | 0.4109 |
| `blind_decoder-attention` | 0.4608 | 0.4056 | 0.4025 | 0.5437 | 0.3961 | 0.4035 |
| `blind_similarity` | 0.3953 | 0.3923 | 0.3923 | 0.4317 | 0.3997 | 0.3938 |
| `decoder-alpha` | 0.4369 | 0.4121 | 0.4142 | 0.4586 | 0.4036 | 0.4039 |
| `decoder-attention` | 0.4215 | 0.3986 | 0.3928 | 0.4528 | 0.4002 | 0.4067 |
| `vanilla` | 0.4389 | 0.4157 | 0.4088 | 0.4508 | 0.4185 | 0.4146 |

#### Mean query ROC AUC, 15 binary tasks (higher is better)

| model | plain-coh2 | mixed-coh2 | curriculum-coh2 | plain-coh8 | mixed-coh8 | curriculum-coh8 |
|---|---:|---:|---:|---:|---:|---:|
| `blind_decoder-alpha` | 0.7614 | 0.7854 | 0.7973 | 0.7692 | 0.7852 | 0.7889 |
| `blind_decoder-attention` | 0.7821 | 0.7958 | 0.7971 | 0.7601 | 0.7944 | 0.7866 |
| `blind_similarity` | 0.7892 | 0.7961 | 0.7985 | 0.7883 | 0.7944 | 0.7964 |
| `decoder-alpha` | 0.7649 | 0.7836 | 0.7883 | 0.7718 | 0.7871 | 0.7860 |
| `decoder-attention` | 0.7751 | 0.7983 | 0.7938 | 0.7653 | 0.7879 | 0.7888 |
| `vanilla` | 0.7711 | 0.7874 | 0.7866 | 0.7559 | 0.7901 | 0.7862 |

#### Average rank, 15 binary tasks x 6 prior conditions = 90 cells (1 = best)

| model | mean CE rank | mean AUC rank |
|---|---:|---:|
| `blind_similarity` | 2.29 | 2.56 |
| `decoder-attention` | 2.72 | 3.59 |
| `decoder-alpha` | 3.90 | 4.13 |
| `blind_decoder-alpha` | 3.93 | 3.88 |
| `blind_decoder-attention` | 4.00 | 2.86 |
| `vanilla` | 4.16 | 3.99 |

### Model average rank split by prior type, 15 binary tasks (30 cells per column: 15 tasks x 2 coherence levels; 1 = best)

#### Mean CE rank

| model | plain | mixed | curriculum |
|---|---:|---:|---:|
| `blind_similarity` | 2.07 | 2.40 | 2.40 |
| `decoder-attention` | 2.90 | 2.63 | 2.63 |
| `decoder-alpha` | 3.60 | 4.13 | 3.97 |
| `vanilla` | 3.90 | 4.23 | 4.33 |
| `blind_decoder-alpha` | 3.97 | 3.90 | 3.93 |
| `blind_decoder-attention` | 4.57 | 3.70 | 3.73 |

#### Mean AUC rank

| model | plain | mixed | curriculum |
|---|---:|---:|---:|
| `blind_similarity` | 2.13 | 2.87 | 2.67 |
| `blind_decoder-attention` | 2.73 | 2.70 | 3.13 |
| `decoder-attention` | 3.73 | 3.30 | 3.73 |
| `blind_decoder-alpha` | 4.00 | 4.10 | 3.53 |
| `vanilla` | 4.17 | 3.87 | 3.93 |
| `decoder-alpha` | 4.23 | 4.17 | 4.00 |

### (model, prior) treated as 36 distinct configurations, ranked together within each of the 15 binary tasks (1 = best of 36)

| model | prior | mean CE rank | mean AUC rank |
|---|---|---:|---:|
| `decoder-attention` | curriculum-coh2 | 8.33 | 13.47 |
| `blind_similarity` | mixed-coh2 | 9.13 | 11.27 |
| `blind_similarity` | curriculum-coh8 | 9.53 | 14.20 |
| `blind_similarity` | curriculum-coh2 | 11.20 | 10.60 |
| `decoder-attention` | mixed-coh2 | 11.20 | 7.53 |
| `decoder-attention` | mixed-coh8 | 11.67 | 18.00 |
| `decoder-attention` | curriculum-coh8 | 11.73 | 19.87 |
| `blind_similarity` | plain-coh2 | 12.40 | 17.20 |
| `blind_similarity` | mixed-coh8 | 13.00 | 13.20 |
| `blind_decoder-attention` | mixed-coh8 | 13.93 | 13.33 |
| `decoder-alpha` | mixed-coh8 | 14.27 | 16.73 |
| `blind_decoder-attention` | curriculum-coh8 | 15.40 | 18.67 |
| `decoder-alpha` | curriculum-coh8 | 15.60 | 18.80 |
| `blind_decoder-attention` | curriculum-coh2 | 15.73 | 11.53 |
| `vanilla` | curriculum-coh2 | 16.73 | 17.80 |
| `blind_decoder-alpha` | curriculum-coh2 | 17.07 | 16.07 |
| `blind_decoder-alpha` | curriculum-coh8 | 17.93 | 16.27 |
| `blind_decoder-alpha` | mixed-coh8 | 17.93 | 19.80 |
| `decoder-attention` | plain-coh2 | 18.87 | 24.73 |
| `vanilla` | curriculum-coh8 | 18.87 | 19.27 |
| `blind_decoder-attention` | mixed-coh2 | 19.00 | 10.27 |
| `blind_decoder-alpha` | mixed-coh2 | 19.07 | 18.67 |
| `vanilla` | mixed-coh2 | 20.07 | 16.33 |
| `vanilla` | mixed-coh8 | 20.13 | 16.73 |
| `decoder-alpha` | curriculum-coh2 | 20.60 | 18.93 |
| `blind_similarity` | plain-coh8 | 21.87 | 19.07 |
| `decoder-alpha` | mixed-coh2 | 21.87 | 18.40 |
| `decoder-alpha` | plain-coh2 | 23.07 | 28.80 |
| `blind_decoder-attention` | plain-coh2 | 24.33 | 13.93 |
| `blind_decoder-alpha` | plain-coh8 | 26.33 | 25.27 |
| `blind_decoder-alpha` | plain-coh2 | 26.80 | 28.33 |
| `decoder-attention` | plain-coh8 | 27.20 | 26.33 |
| `vanilla` | plain-coh2 | 27.33 | 25.13 |
| `decoder-alpha` | plain-coh8 | 27.60 | 27.13 |
| `vanilla` | plain-coh8 | 28.93 | 31.13 |
| `blind_decoder-attention` | plain-coh8 | 31.27 | 23.20 |

## Per-task detail

| task | model | prior | query CE | query AUC |
|---|---|---|---:|---:|
| Bank_Customer_Churn | `vanilla` | plain-coh2 | 0.4322 | 0.7970 |
| Bank_Customer_Churn | `vanilla` | mixed-coh2 | 0.4211 | 0.7934 |
| Bank_Customer_Churn | `vanilla` | curriculum-coh2 | 0.4120 | 0.7997 |
| Bank_Customer_Churn | `vanilla` | plain-coh8 | 0.4504 | 0.7724 |
| Bank_Customer_Churn | `vanilla` | mixed-coh8 | 0.4164 | 0.8019 |
| Bank_Customer_Churn | `vanilla` | curriculum-coh8 | 0.4202 | 0.7978 |
| Bank_Customer_Churn | `decoder-alpha` | plain-coh2 | 0.4434 | 0.7581 |
| Bank_Customer_Churn | `decoder-alpha` | mixed-coh2 | 0.4187 | 0.7868 |
| Bank_Customer_Churn | `decoder-alpha` | curriculum-coh2 | 0.4287 | 0.7898 |
| Bank_Customer_Churn | `decoder-alpha` | plain-coh8 | 0.4487 | 0.7404 |
| Bank_Customer_Churn | `decoder-alpha` | mixed-coh8 | 0.4176 | 0.7859 |
| Bank_Customer_Churn | `decoder-alpha` | curriculum-coh8 | 0.4202 | 0.7829 |
| Bank_Customer_Churn | `decoder-attention` | plain-coh2 | 0.4212 | 0.7923 |
| Bank_Customer_Churn | `decoder-attention` | mixed-coh2 | 0.4109 | 0.7876 |
| Bank_Customer_Churn | `decoder-attention` | curriculum-coh2 | 0.4083 | 0.7920 |
| Bank_Customer_Churn | `decoder-attention` | plain-coh8 | 0.4355 | 0.7887 |
| Bank_Customer_Churn | `decoder-attention` | mixed-coh8 | 0.4172 | 0.7788 |
| Bank_Customer_Churn | `decoder-attention` | curriculum-coh8 | 0.4190 | 0.7829 |
| Bank_Customer_Churn | `blind_decoder-alpha` | plain-coh2 | 0.4439 | 0.7514 |
| Bank_Customer_Churn | `blind_decoder-alpha` | mixed-coh2 | 0.4154 | 0.8006 |
| Bank_Customer_Churn | `blind_decoder-alpha` | curriculum-coh2 | 0.4298 | 0.7788 |
| Bank_Customer_Churn | `blind_decoder-alpha` | plain-coh8 | 0.4555 | 0.7672 |
| Bank_Customer_Churn | `blind_decoder-alpha` | mixed-coh8 | 0.4238 | 0.7906 |
| Bank_Customer_Churn | `blind_decoder-alpha` | curriculum-coh8 | 0.4191 | 0.7810 |
| Bank_Customer_Churn | `blind_decoder-attention` | plain-coh2 | 0.4068 | 0.7937 |
| Bank_Customer_Churn | `blind_decoder-attention` | mixed-coh2 | 0.4252 | 0.7887 |
| Bank_Customer_Churn | `blind_decoder-attention` | curriculum-coh2 | 0.4090 | 0.8030 |
| Bank_Customer_Churn | `blind_decoder-attention` | plain-coh8 | 0.4188 | 0.7961 |
| Bank_Customer_Churn | `blind_decoder-attention` | mixed-coh8 | 0.4205 | 0.7713 |
| Bank_Customer_Churn | `blind_decoder-attention` | curriculum-coh8 | 0.4190 | 0.7870 |
| Bank_Customer_Churn | `blind_similarity` | plain-coh2 | 0.4222 | 0.7754 |
| Bank_Customer_Churn | `blind_similarity` | mixed-coh2 | 0.4118 | 0.7796 |
| Bank_Customer_Churn | `blind_similarity` | curriculum-coh2 | 0.4141 | 0.7857 |
| Bank_Customer_Churn | `blind_similarity` | plain-coh8 | 0.4277 | 0.7777 |
| Bank_Customer_Churn | `blind_similarity` | mixed-coh8 | 0.4181 | 0.7774 |
| Bank_Customer_Churn | `blind_similarity` | curriculum-coh8 | 0.4169 | 0.7854 |
| E-CommereShippingData | `vanilla` | plain-coh2 | 0.5817 | 0.6889 |
| E-CommereShippingData | `vanilla` | mixed-coh2 | 0.5503 | 0.7031 |
| E-CommereShippingData | `vanilla` | curriculum-coh2 | 0.5451 | 0.7031 |
| E-CommereShippingData | `vanilla` | plain-coh8 | 0.5827 | 0.6936 |
| E-CommereShippingData | `vanilla` | mixed-coh8 | 0.5384 | 0.7231 |
| E-CommereShippingData | `vanilla` | curriculum-coh8 | 0.5442 | 0.7110 |
| E-CommereShippingData | `decoder-alpha` | plain-coh2 | 0.6142 | 0.6765 |
| E-CommereShippingData | `decoder-alpha` | mixed-coh2 | 0.5804 | 0.6909 |
| E-CommereShippingData | `decoder-alpha` | curriculum-coh2 | 0.5922 | 0.6841 |
| E-CommereShippingData | `decoder-alpha` | plain-coh8 | 0.6082 | 0.6862 |
| E-CommereShippingData | `decoder-alpha` | mixed-coh8 | 0.5807 | 0.6923 |
| E-CommereShippingData | `decoder-alpha` | curriculum-coh8 | 0.5672 | 0.7004 |
| E-CommereShippingData | `decoder-attention` | plain-coh2 | 0.5686 | 0.6893 |
| E-CommereShippingData | `decoder-attention` | mixed-coh2 | 0.5313 | 0.7065 |
| E-CommereShippingData | `decoder-attention` | curriculum-coh2 | 0.5355 | 0.7053 |
| E-CommereShippingData | `decoder-attention` | plain-coh8 | 0.5910 | 0.6812 |
| E-CommereShippingData | `decoder-attention` | mixed-coh8 | 0.5675 | 0.6911 |
| E-CommereShippingData | `decoder-attention` | curriculum-coh8 | 0.5767 | 0.6859 |
| E-CommereShippingData | `blind_decoder-alpha` | plain-coh2 | 0.5957 | 0.6884 |
| E-CommereShippingData | `blind_decoder-alpha` | mixed-coh2 | 0.5629 | 0.7130 |
| E-CommereShippingData | `blind_decoder-alpha` | curriculum-coh2 | 0.5702 | 0.6995 |
| E-CommereShippingData | `blind_decoder-alpha` | plain-coh8 | 0.6004 | 0.7114 |
| E-CommereShippingData | `blind_decoder-alpha` | mixed-coh8 | 0.5777 | 0.7038 |
| E-CommereShippingData | `blind_decoder-alpha` | curriculum-coh8 | 0.5753 | 0.7013 |
| E-CommereShippingData | `blind_decoder-attention` | plain-coh2 | 0.5509 | 0.7125 |
| E-CommereShippingData | `blind_decoder-attention` | mixed-coh2 | 0.5387 | 0.6915 |
| E-CommereShippingData | `blind_decoder-attention` | curriculum-coh2 | 0.5296 | 0.7162 |
| E-CommereShippingData | `blind_decoder-attention` | plain-coh8 | 0.5836 | 0.7042 |
| E-CommereShippingData | `blind_decoder-attention` | mixed-coh8 | 0.5296 | 0.6947 |
| E-CommereShippingData | `blind_decoder-attention` | curriculum-coh8 | 0.5486 | 0.6866 |
| E-CommereShippingData | `blind_similarity` | plain-coh2 | 0.5254 | 0.6911 |
| E-CommereShippingData | `blind_similarity` | mixed-coh2 | 0.5319 | 0.6985 |
| E-CommereShippingData | `blind_similarity` | curriculum-coh2 | 0.5244 | 0.7085 |
| E-CommereShippingData | `blind_similarity` | plain-coh8 | 0.5389 | 0.6999 |
| E-CommereShippingData | `blind_similarity` | mixed-coh8 | 0.5185 | 0.6968 |
| E-CommereShippingData | `blind_similarity` | curriculum-coh8 | 0.5359 | 0.7073 |
| Fitness_Club | `vanilla` | plain-coh2 | 0.4425 | 0.8578 |
| Fitness_Club | `vanilla` | mixed-coh2 | 0.4283 | 0.8651 |
| Fitness_Club | `vanilla` | curriculum-coh2 | 0.4328 | 0.8628 |
| Fitness_Club | `vanilla` | plain-coh8 | 0.4614 | 0.8529 |
| Fitness_Club | `vanilla` | mixed-coh8 | 0.4372 | 0.8583 |
| Fitness_Club | `vanilla` | curriculum-coh8 | 0.4445 | 0.8576 |
| Fitness_Club | `decoder-alpha` | plain-coh2 | 0.4596 | 0.8538 |
| Fitness_Club | `decoder-alpha` | mixed-coh2 | 0.4486 | 0.8637 |
| Fitness_Club | `decoder-alpha` | curriculum-coh2 | 0.4567 | 0.8576 |
| Fitness_Club | `decoder-alpha` | plain-coh8 | 0.4512 | 0.8578 |
| Fitness_Club | `decoder-alpha` | mixed-coh8 | 0.4267 | 0.8628 |
| Fitness_Club | `decoder-alpha` | curriculum-coh8 | 0.4415 | 0.8623 |
| Fitness_Club | `decoder-attention` | plain-coh2 | 0.4601 | 0.8597 |
| Fitness_Club | `decoder-attention` | mixed-coh2 | 0.4421 | 0.8649 |
| Fitness_Club | `decoder-attention` | curriculum-coh2 | 0.4474 | 0.8578 |
| Fitness_Club | `decoder-attention` | plain-coh8 | 0.4649 | 0.8604 |
| Fitness_Club | `decoder-attention` | mixed-coh8 | 0.4318 | 0.8573 |
| Fitness_Club | `decoder-attention` | curriculum-coh8 | 0.4365 | 0.8609 |
| Fitness_Club | `blind_decoder-alpha` | plain-coh2 | 0.4424 | 0.8581 |
| Fitness_Club | `blind_decoder-alpha` | mixed-coh2 | 0.4480 | 0.8606 |
| Fitness_Club | `blind_decoder-alpha` | curriculum-coh2 | 0.4414 | 0.8597 |
| Fitness_Club | `blind_decoder-alpha` | plain-coh8 | 0.4488 | 0.8552 |
| Fitness_Club | `blind_decoder-alpha` | mixed-coh8 | 0.4317 | 0.8597 |
| Fitness_Club | `blind_decoder-alpha` | curriculum-coh8 | 0.4328 | 0.8618 |
| Fitness_Club | `blind_decoder-attention` | plain-coh2 | 0.4138 | 0.8649 |
| Fitness_Club | `blind_decoder-attention` | mixed-coh2 | 0.4474 | 0.8621 |
| Fitness_Club | `blind_decoder-attention` | curriculum-coh2 | 0.4417 | 0.8602 |
| Fitness_Club | `blind_decoder-attention` | plain-coh8 | 0.4228 | 0.8682 |
| Fitness_Club | `blind_decoder-attention` | mixed-coh8 | 0.4264 | 0.8621 |
| Fitness_Club | `blind_decoder-attention` | curriculum-coh8 | 0.4456 | 0.8630 |
| Fitness_Club | `blind_similarity` | plain-coh2 | 0.4243 | 0.8613 |
| Fitness_Club | `blind_similarity` | mixed-coh2 | 0.4338 | 0.8644 |
| Fitness_Club | `blind_similarity` | curriculum-coh2 | 0.4375 | 0.8630 |
| Fitness_Club | `blind_similarity` | plain-coh8 | 0.4458 | 0.8677 |
| Fitness_Club | `blind_similarity` | mixed-coh8 | 0.4261 | 0.8630 |
| Fitness_Club | `blind_similarity` | curriculum-coh8 | 0.4336 | 0.8616 |
| Is-this-a-good-customer | `vanilla` | plain-coh2 | 0.4059 | 0.6873 |
| Is-this-a-good-customer | `vanilla` | mixed-coh2 | 0.4032 | 0.7145 |
| Is-this-a-good-customer | `vanilla` | curriculum-coh2 | 0.3991 | 0.6939 |
| Is-this-a-good-customer | `vanilla` | plain-coh8 | 0.4074 | 0.6880 |
| Is-this-a-good-customer | `vanilla` | mixed-coh8 | 0.4032 | 0.7424 |
| Is-this-a-good-customer | `vanilla` | curriculum-coh8 | 0.4031 | 0.7197 |
| Is-this-a-good-customer | `decoder-alpha` | plain-coh2 | 0.3988 | 0.7018 |
| Is-this-a-good-customer | `decoder-alpha` | mixed-coh2 | 0.4042 | 0.7128 |
| Is-this-a-good-customer | `decoder-alpha` | curriculum-coh2 | 0.3987 | 0.7163 |
| Is-this-a-good-customer | `decoder-alpha` | plain-coh8 | 0.3961 | 0.6959 |
| Is-this-a-good-customer | `decoder-alpha` | mixed-coh8 | 0.3916 | 0.7163 |
| Is-this-a-good-customer | `decoder-alpha` | curriculum-coh8 | 0.3908 | 0.6994 |
| Is-this-a-good-customer | `decoder-attention` | plain-coh2 | 0.4054 | 0.6932 |
| Is-this-a-good-customer | `decoder-attention` | mixed-coh2 | 0.3975 | 0.7087 |
| Is-this-a-good-customer | `decoder-attention` | curriculum-coh2 | 0.3949 | 0.6935 |
| Is-this-a-good-customer | `decoder-attention` | plain-coh8 | 0.3927 | 0.7097 |
| Is-this-a-good-customer | `decoder-attention` | mixed-coh8 | 0.3934 | 0.7121 |
| Is-this-a-good-customer | `decoder-attention` | curriculum-coh8 | 0.3924 | 0.6997 |
| Is-this-a-good-customer | `blind_decoder-alpha` | plain-coh2 | 0.4054 | 0.6711 |
| Is-this-a-good-customer | `blind_decoder-alpha` | mixed-coh2 | 0.4032 | 0.6684 |
| Is-this-a-good-customer | `blind_decoder-alpha` | curriculum-coh2 | 0.3985 | 0.7304 |
| Is-this-a-good-customer | `blind_decoder-alpha` | plain-coh8 | 0.3967 | 0.6732 |
| Is-this-a-good-customer | `blind_decoder-alpha` | mixed-coh8 | 0.4043 | 0.6718 |
| Is-this-a-good-customer | `blind_decoder-alpha` | curriculum-coh8 | 0.3892 | 0.7011 |
| Is-this-a-good-customer | `blind_decoder-attention` | plain-coh2 | 0.4068 | 0.6901 |
| Is-this-a-good-customer | `blind_decoder-attention` | mixed-coh2 | 0.4034 | 0.6977 |
| Is-this-a-good-customer | `blind_decoder-attention` | curriculum-coh2 | 0.4009 | 0.6873 |
| Is-this-a-good-customer | `blind_decoder-attention` | plain-coh8 | 0.4148 | 0.6966 |
| Is-this-a-good-customer | `blind_decoder-attention` | mixed-coh8 | 0.4044 | 0.6880 |
| Is-this-a-good-customer | `blind_decoder-attention` | curriculum-coh8 | 0.3961 | 0.6925 |
| Is-this-a-good-customer | `blind_similarity` | plain-coh2 | 0.4134 | 0.6873 |
| Is-this-a-good-customer | `blind_similarity` | mixed-coh2 | 0.3988 | 0.6839 |
| Is-this-a-good-customer | `blind_similarity` | curriculum-coh2 | 0.3997 | 0.6942 |
| Is-this-a-good-customer | `blind_similarity` | plain-coh8 | 0.4056 | 0.7049 |
| Is-this-a-good-customer | `blind_similarity` | mixed-coh8 | 0.4023 | 0.6818 |
| Is-this-a-good-customer | `blind_similarity` | curriculum-coh8 | 0.3907 | 0.6908 |
| Marketing_Campaign | `vanilla` | plain-coh2 | 0.3640 | 0.8416 |
| Marketing_Campaign | `vanilla` | mixed-coh2 | 0.3535 | 0.8523 |
| Marketing_Campaign | `vanilla` | curriculum-coh2 | 0.3435 | 0.8440 |
| Marketing_Campaign | `vanilla` | plain-coh8 | 0.3615 | 0.8237 |
| Marketing_Campaign | `vanilla` | mixed-coh8 | 0.3517 | 0.8416 |
| Marketing_Campaign | `vanilla` | curriculum-coh8 | 0.3467 | 0.8413 |
| Marketing_Campaign | `decoder-alpha` | plain-coh2 | 0.3434 | 0.8154 |
| Marketing_Campaign | `decoder-alpha` | mixed-coh2 | 0.3542 | 0.8223 |
| Marketing_Campaign | `decoder-alpha` | curriculum-coh2 | 0.3477 | 0.8275 |
| Marketing_Campaign | `decoder-alpha` | plain-coh8 | 0.3342 | 0.8275 |
| Marketing_Campaign | `decoder-alpha` | mixed-coh8 | 0.3338 | 0.8419 |
| Marketing_Campaign | `decoder-alpha` | curriculum-coh8 | 0.3349 | 0.8233 |
| Marketing_Campaign | `decoder-attention` | plain-coh2 | 0.3311 | 0.8333 |
| Marketing_Campaign | `decoder-attention` | mixed-coh2 | 0.3300 | 0.8629 |
| Marketing_Campaign | `decoder-attention` | curriculum-coh2 | 0.3173 | 0.8636 |
| Marketing_Campaign | `decoder-attention` | plain-coh8 | 0.3566 | 0.8264 |
| Marketing_Campaign | `decoder-attention` | mixed-coh8 | 0.3289 | 0.8402 |
| Marketing_Campaign | `decoder-attention` | curriculum-coh8 | 0.3210 | 0.8619 |
| Marketing_Campaign | `blind_decoder-alpha` | plain-coh2 | 0.3425 | 0.8185 |
| Marketing_Campaign | `blind_decoder-alpha` | mixed-coh2 | 0.3490 | 0.8278 |
| Marketing_Campaign | `blind_decoder-alpha` | curriculum-coh2 | 0.3541 | 0.8433 |
| Marketing_Campaign | `blind_decoder-alpha` | plain-coh8 | 0.3412 | 0.8330 |
| Marketing_Campaign | `blind_decoder-alpha` | mixed-coh8 | 0.3255 | 0.8537 |
| Marketing_Campaign | `blind_decoder-alpha` | curriculum-coh8 | 0.3436 | 0.8213 |
| Marketing_Campaign | `blind_decoder-attention` | plain-coh2 | 0.3370 | 0.8557 |
| Marketing_Campaign | `blind_decoder-attention` | mixed-coh2 | 0.3335 | 0.8660 |
| Marketing_Campaign | `blind_decoder-attention` | curriculum-coh2 | 0.3224 | 0.8592 |
| Marketing_Campaign | `blind_decoder-attention` | plain-coh8 | 0.3827 | 0.8454 |
| Marketing_Campaign | `blind_decoder-attention` | mixed-coh8 | 0.3261 | 0.8609 |
| Marketing_Campaign | `blind_decoder-attention` | curriculum-coh8 | 0.3289 | 0.8595 |
| Marketing_Campaign | `blind_similarity` | plain-coh2 | 0.3382 | 0.8464 |
| Marketing_Campaign | `blind_similarity` | mixed-coh2 | 0.3306 | 0.8612 |
| Marketing_Campaign | `blind_similarity` | curriculum-coh2 | 0.3272 | 0.8561 |
| Marketing_Campaign | `blind_similarity` | plain-coh8 | 0.3581 | 0.8385 |
| Marketing_Campaign | `blind_similarity` | mixed-coh8 | 0.3296 | 0.8578 |
| Marketing_Campaign | `blind_similarity` | curriculum-coh8 | 0.3200 | 0.8674 |
| NATICUSdroid | `vanilla` | plain-coh2 | 0.4836 | 0.8899 |
| NATICUSdroid | `vanilla` | mixed-coh2 | 0.3462 | 0.9638 |
| NATICUSdroid | `vanilla` | curriculum-coh2 | 0.3367 | 0.9557 |
| NATICUSdroid | `vanilla` | plain-coh8 | 0.5265 | 0.8181 |
| NATICUSdroid | `vanilla` | mixed-coh8 | 0.4157 | 0.9359 |
| NATICUSdroid | `vanilla` | curriculum-coh8 | 0.3976 | 0.9265 |
| NATICUSdroid | `decoder-alpha` | plain-coh2 | 0.4819 | 0.8338 |
| NATICUSdroid | `decoder-alpha` | mixed-coh2 | 0.2570 | 0.9541 |
| NATICUSdroid | `decoder-alpha` | curriculum-coh2 | 0.3010 | 0.9648 |
| NATICUSdroid | `decoder-alpha` | plain-coh8 | 0.6635 | 0.8902 |
| NATICUSdroid | `decoder-alpha` | mixed-coh8 | 0.2992 | 0.9724 |
| NATICUSdroid | `decoder-alpha` | curriculum-coh8 | 0.2969 | 0.9714 |
| NATICUSdroid | `decoder-attention` | plain-coh2 | 0.4897 | 0.9226 |
| NATICUSdroid | `decoder-attention` | mixed-coh2 | 0.2648 | 0.9732 |
| NATICUSdroid | `decoder-attention` | curriculum-coh2 | 0.2406 | 0.9689 |
| NATICUSdroid | `decoder-attention` | plain-coh8 | 0.6614 | 0.8262 |
| NATICUSdroid | `decoder-attention` | mixed-coh8 | 0.2790 | 0.9761 |
| NATICUSdroid | `decoder-attention` | curriculum-coh8 | 0.3764 | 0.9627 |
| NATICUSdroid | `blind_decoder-alpha` | plain-coh2 | 0.5567 | 0.7672 |
| NATICUSdroid | `blind_decoder-alpha` | mixed-coh2 | 0.2534 | 0.9603 |
| NATICUSdroid | `blind_decoder-alpha` | curriculum-coh2 | 0.2337 | 0.9646 |
| NATICUSdroid | `blind_decoder-alpha` | plain-coh8 | 0.7582 | 0.8352 |
| NATICUSdroid | `blind_decoder-alpha` | mixed-coh8 | 0.2737 | 0.9707 |
| NATICUSdroid | `blind_decoder-alpha` | curriculum-coh8 | 0.3207 | 0.9724 |
| NATICUSdroid | `blind_decoder-attention` | plain-coh2 | 0.7862 | 0.8445 |
| NATICUSdroid | `blind_decoder-attention` | mixed-coh2 | 0.2366 | 0.9755 |
| NATICUSdroid | `blind_decoder-attention` | curriculum-coh2 | 0.2659 | 0.9578 |
| NATICUSdroid | `blind_decoder-attention` | plain-coh8 | 1.0405 | 0.7602 |
| NATICUSdroid | `blind_decoder-attention` | mixed-coh8 | 0.2218 | 0.9773 |
| NATICUSdroid | `blind_decoder-attention` | curriculum-coh8 | 0.2686 | 0.9730 |
| NATICUSdroid | `blind_similarity` | plain-coh2 | 0.2424 | 0.9656 |
| NATICUSdroid | `blind_similarity` | mixed-coh2 | 0.2048 | 0.9788 |
| NATICUSdroid | `blind_similarity` | curriculum-coh2 | 0.1934 | 0.9763 |
| NATICUSdroid | `blind_similarity` | plain-coh8 | 0.5435 | 0.9223 |
| NATICUSdroid | `blind_similarity` | mixed-coh8 | 0.2925 | 0.9802 |
| NATICUSdroid | `blind_similarity` | curriculum-coh8 | 0.2456 | 0.9718 |
| SDSS17 | `vanilla` | plain-coh2 | 0.3628 | 0.9620 |
| SDSS17 | `vanilla` | mixed-coh2 | 0.5235 | 0.9224 |
| SDSS17 | `vanilla` | curriculum-coh2 | 0.5818 | 0.9061 |
| SDSS17 | `vanilla` | plain-coh8 | 0.5546 | 0.9197 |
| SDSS17 | `vanilla` | mixed-coh8 | 0.6641 | 0.8851 |
| SDSS17 | `vanilla` | curriculum-coh8 | 0.6790 | 0.8732 |
| SDSS17 | `decoder-alpha` | plain-coh2 | 0.5301 | 0.9277 |
| SDSS17 | `decoder-alpha` | mixed-coh2 | 0.5131 | 0.9353 |
| SDSS17 | `decoder-alpha` | curriculum-coh2 | 0.6273 | 0.9234 |
| SDSS17 | `decoder-alpha` | plain-coh8 | 0.6806 | 0.9008 |
| SDSS17 | `decoder-alpha` | mixed-coh8 | 0.7218 | 0.8769 |
| SDSS17 | `decoder-alpha` | curriculum-coh8 | 0.7338 | 0.8697 |
| SDSS17 | `decoder-attention` | plain-coh2 | 0.4673 | 0.9448 |
| SDSS17 | `decoder-attention` | mixed-coh2 | 0.4454 | 0.9443 |
| SDSS17 | `decoder-attention` | curriculum-coh2 | 0.5281 | 0.9238 |
| SDSS17 | `decoder-attention` | plain-coh8 | 0.5892 | 0.9126 |
| SDSS17 | `decoder-attention` | mixed-coh8 | 0.6139 | 0.8969 |
| SDSS17 | `decoder-attention` | curriculum-coh8 | 0.6922 | 0.8713 |
| SDSS17 | `blind_decoder-alpha` | plain-coh2 | 0.5617 | 0.9197 |
| SDSS17 | `blind_decoder-alpha` | mixed-coh2 | 0.5637 | 0.9213 |
| SDSS17 | `blind_decoder-alpha` | curriculum-coh2 | 0.6659 | 0.8903 |
| SDSS17 | `blind_decoder-alpha` | plain-coh8 | 0.7210 | 0.8864 |
| SDSS17 | `blind_decoder-alpha` | mixed-coh8 | 0.7138 | 0.8749 |
| SDSS17 | `blind_decoder-alpha` | curriculum-coh8 | 0.7774 | 0.8554 |
| SDSS17 | `blind_decoder-attention` | plain-coh2 | 0.3829 | 0.9559 |
| SDSS17 | `blind_decoder-attention` | mixed-coh2 | 0.4461 | 0.9420 |
| SDSS17 | `blind_decoder-attention` | curriculum-coh2 | 0.4957 | 0.9422 |
| SDSS17 | `blind_decoder-attention` | plain-coh8 | 0.6045 | 0.9159 |
| SDSS17 | `blind_decoder-attention` | mixed-coh8 | 0.5547 | 0.9312 |
| SDSS17 | `blind_decoder-attention` | curriculum-coh8 | 0.7771 | 0.8908 |
| SDSS17 | `blind_similarity` | plain-coh2 | 0.3678 | 0.9562 |
| SDSS17 | `blind_similarity` | mixed-coh2 | 0.3922 | 0.9488 |
| SDSS17 | `blind_similarity` | curriculum-coh2 | 0.4245 | 0.9550 |
| SDSS17 | `blind_similarity` | plain-coh8 | 0.6401 | 0.9153 |
| SDSS17 | `blind_similarity` | mixed-coh8 | 0.7141 | 0.9068 |
| SDSS17 | `blind_similarity` | curriculum-coh8 | 0.7317 | 0.8760 |
| blood-transfusion-service-center | `vanilla` | plain-coh2 | 0.5034 | 0.7260 |
| blood-transfusion-service-center | `vanilla` | mixed-coh2 | 0.5018 | 0.7220 |
| blood-transfusion-service-center | `vanilla` | curriculum-coh2 | 0.4987 | 0.7312 |
| blood-transfusion-service-center | `vanilla` | plain-coh8 | 0.4925 | 0.7277 |
| blood-transfusion-service-center | `vanilla` | mixed-coh8 | 0.5025 | 0.7202 |
| blood-transfusion-service-center | `vanilla` | curriculum-coh8 | 0.4941 | 0.7292 |
| blood-transfusion-service-center | `decoder-alpha` | plain-coh2 | 0.4878 | 0.7319 |
| blood-transfusion-service-center | `decoder-alpha` | mixed-coh2 | 0.4947 | 0.7344 |
| blood-transfusion-service-center | `decoder-alpha` | curriculum-coh2 | 0.4926 | 0.7327 |
| blood-transfusion-service-center | `decoder-alpha` | plain-coh8 | 0.4943 | 0.7220 |
| blood-transfusion-service-center | `decoder-alpha` | mixed-coh8 | 0.4887 | 0.7352 |
| blood-transfusion-service-center | `decoder-alpha` | curriculum-coh8 | 0.4938 | 0.7299 |
| blood-transfusion-service-center | `decoder-attention` | plain-coh2 | 0.4933 | 0.7312 |
| blood-transfusion-service-center | `decoder-attention` | mixed-coh2 | 0.4940 | 0.7386 |
| blood-transfusion-service-center | `decoder-attention` | curriculum-coh2 | 0.4934 | 0.7289 |
| blood-transfusion-service-center | `decoder-attention` | plain-coh8 | 0.4912 | 0.7357 |
| blood-transfusion-service-center | `decoder-attention` | mixed-coh8 | 0.4918 | 0.7426 |
| blood-transfusion-service-center | `decoder-attention` | curriculum-coh8 | 0.4878 | 0.7272 |
| blood-transfusion-service-center | `blind_decoder-alpha` | plain-coh2 | 0.4897 | 0.7426 |
| blood-transfusion-service-center | `blind_decoder-alpha` | mixed-coh2 | 0.4936 | 0.7299 |
| blood-transfusion-service-center | `blind_decoder-alpha` | curriculum-coh2 | 0.4954 | 0.7227 |
| blood-transfusion-service-center | `blind_decoder-alpha` | plain-coh8 | 0.4886 | 0.7406 |
| blood-transfusion-service-center | `blind_decoder-alpha` | mixed-coh8 | 0.5003 | 0.7053 |
| blood-transfusion-service-center | `blind_decoder-alpha` | curriculum-coh8 | 0.4926 | 0.7347 |
| blood-transfusion-service-center | `blind_decoder-attention` | plain-coh2 | 0.4981 | 0.7371 |
| blood-transfusion-service-center | `blind_decoder-attention` | mixed-coh2 | 0.4991 | 0.7361 |
| blood-transfusion-service-center | `blind_decoder-attention` | curriculum-coh2 | 0.4955 | 0.7317 |
| blood-transfusion-service-center | `blind_decoder-attention` | plain-coh8 | 0.4983 | 0.7302 |
| blood-transfusion-service-center | `blind_decoder-attention` | mixed-coh8 | 0.4964 | 0.7322 |
| blood-transfusion-service-center | `blind_decoder-attention` | curriculum-coh8 | 0.4948 | 0.7292 |
| blood-transfusion-service-center | `blind_similarity` | plain-coh2 | 0.4932 | 0.7421 |
| blood-transfusion-service-center | `blind_similarity` | mixed-coh2 | 0.4924 | 0.7371 |
| blood-transfusion-service-center | `blind_similarity` | curriculum-coh2 | 0.4969 | 0.7275 |
| blood-transfusion-service-center | `blind_similarity` | plain-coh8 | 0.4909 | 0.7342 |
| blood-transfusion-service-center | `blind_similarity` | mixed-coh8 | 0.4945 | 0.7399 |
| blood-transfusion-service-center | `blind_similarity` | curriculum-coh8 | 0.4943 | 0.7275 |
| churn | `vanilla` | plain-coh2 | 0.3963 | 0.9338 |
| churn | `vanilla` | mixed-coh2 | 0.3800 | 0.9402 |
| churn | `vanilla` | curriculum-coh2 | 0.3681 | 0.9431 |
| churn | `vanilla` | plain-coh8 | 0.3945 | 0.9286 |
| churn | `vanilla` | mixed-coh8 | 0.3599 | 0.9487 |
| churn | `vanilla` | curriculum-coh8 | 0.3533 | 0.9425 |
| churn | `decoder-alpha` | plain-coh2 | 0.3813 | 0.9280 |
| churn | `decoder-alpha` | mixed-coh2 | 0.3794 | 0.9405 |
| churn | `decoder-alpha` | curriculum-coh2 | 0.3530 | 0.9493 |
| churn | `decoder-alpha` | plain-coh8 | 0.3337 | 0.9493 |
| churn | `decoder-alpha` | mixed-coh8 | 0.3257 | 0.9516 |
| churn | `decoder-alpha` | curriculum-coh8 | 0.3149 | 0.9630 |
| churn | `decoder-attention` | plain-coh2 | 0.3566 | 0.9329 |
| churn | `decoder-attention` | mixed-coh2 | 0.3291 | 0.9583 |
| churn | `decoder-attention` | curriculum-coh2 | 0.3185 | 0.9533 |
| churn | `decoder-attention` | plain-coh8 | 0.3890 | 0.9277 |
| churn | `decoder-attention` | mixed-coh8 | 0.3192 | 0.9516 |
| churn | `decoder-attention` | curriculum-coh8 | 0.3371 | 0.9434 |
| churn | `blind_decoder-alpha` | plain-coh2 | 0.3929 | 0.9408 |
| churn | `blind_decoder-alpha` | mixed-coh2 | 0.3833 | 0.9472 |
| churn | `blind_decoder-alpha` | curriculum-coh2 | 0.3609 | 0.9294 |
| churn | `blind_decoder-alpha` | plain-coh8 | 0.3671 | 0.9376 |
| churn | `blind_decoder-alpha` | mixed-coh8 | 0.3423 | 0.9545 |
| churn | `blind_decoder-alpha` | curriculum-coh8 | 0.3406 | 0.9694 |
| churn | `blind_decoder-attention` | plain-coh2 | 0.3408 | 0.9493 |
| churn | `blind_decoder-attention` | mixed-coh2 | 0.3509 | 0.9688 |
| churn | `blind_decoder-attention` | curriculum-coh2 | 0.3446 | 0.9466 |
| churn | `blind_decoder-attention` | plain-coh8 | 0.3995 | 0.9501 |
| churn | `blind_decoder-attention` | mixed-coh8 | 0.3376 | 0.9574 |
| churn | `blind_decoder-attention` | curriculum-coh8 | 0.3303 | 0.9463 |
| churn | `blind_similarity` | plain-coh2 | 0.2967 | 0.9536 |
| churn | `blind_similarity` | mixed-coh2 | 0.3187 | 0.9676 |
| churn | `blind_similarity` | curriculum-coh2 | 0.3181 | 0.9627 |
| churn | `blind_similarity` | plain-coh8 | 0.3606 | 0.9463 |
| churn | `blind_similarity` | mixed-coh8 | 0.3356 | 0.9565 |
| churn | `blind_similarity` | curriculum-coh8 | 0.3306 | 0.9618 |
| coil2000_insurance_policies | `vanilla` | plain-coh2 | 0.2673 | 0.3808 |
| coil2000_insurance_policies | `vanilla` | mixed-coh2 | 0.2681 | 0.3884 |
| coil2000_insurance_policies | `vanilla` | curriculum-coh2 | 0.2717 | 0.4463 |
| coil2000_insurance_policies | `vanilla` | plain-coh8 | 0.2735 | 0.3617 |
| coil2000_insurance_policies | `vanilla` | mixed-coh8 | 0.2737 | 0.4075 |
| coil2000_insurance_policies | `vanilla` | curriculum-coh8 | 0.2746 | 0.4558 |
| coil2000_insurance_policies | `decoder-alpha` | plain-coh2 | 0.2750 | 0.4202 |
| coil2000_insurance_policies | `decoder-alpha` | mixed-coh2 | 0.2787 | 0.4348 |
| coil2000_insurance_policies | `decoder-alpha` | curriculum-coh2 | 0.2800 | 0.4832 |
| coil2000_insurance_policies | `decoder-alpha` | plain-coh8 | 0.3062 | 0.4374 |
| coil2000_insurance_policies | `decoder-alpha` | mixed-coh8 | 0.2951 | 0.4170 |
| coil2000_insurance_policies | `decoder-alpha` | curriculum-coh8 | 0.3018 | 0.4425 |
| coil2000_insurance_policies | `decoder-attention` | plain-coh2 | 0.2773 | 0.3802 |
| coil2000_insurance_policies | `decoder-attention` | mixed-coh2 | 0.2900 | 0.4037 |
| coil2000_insurance_policies | `decoder-attention` | curriculum-coh2 | 0.2653 | 0.4081 |
| coil2000_insurance_policies | `decoder-attention` | plain-coh8 | 0.3006 | 0.3713 |
| coil2000_insurance_policies | `decoder-attention` | mixed-coh8 | 0.2861 | 0.4075 |
| coil2000_insurance_policies | `decoder-attention` | curriculum-coh8 | 0.2716 | 0.4361 |
| coil2000_insurance_policies | `blind_decoder-alpha` | plain-coh2 | 0.2912 | 0.4310 |
| coil2000_insurance_policies | `blind_decoder-alpha` | mixed-coh2 | 0.2898 | 0.4259 |
| coil2000_insurance_policies | `blind_decoder-alpha` | curriculum-coh2 | 0.2712 | 0.4768 |
| coil2000_insurance_policies | `blind_decoder-alpha` | plain-coh8 | 0.3136 | 0.4406 |
| coil2000_insurance_policies | `blind_decoder-alpha` | mixed-coh8 | 0.2999 | 0.4317 |
| coil2000_insurance_policies | `blind_decoder-alpha` | curriculum-coh8 | 0.3044 | 0.4431 |
| coil2000_insurance_policies | `blind_decoder-attention` | plain-coh2 | 0.3011 | 0.3821 |
| coil2000_insurance_policies | `blind_decoder-attention` | mixed-coh2 | 0.2792 | 0.4170 |
| coil2000_insurance_policies | `blind_decoder-attention` | curriculum-coh2 | 0.2916 | 0.4399 |
| coil2000_insurance_policies | `blind_decoder-attention` | plain-coh8 | 0.3137 | 0.3694 |
| coil2000_insurance_policies | `blind_decoder-attention` | mixed-coh8 | 0.2795 | 0.4450 |
| coil2000_insurance_policies | `blind_decoder-attention` | curriculum-coh8 | 0.3025 | 0.4062 |
| coil2000_insurance_policies | `blind_similarity` | plain-coh2 | 0.2939 | 0.4050 |
| coil2000_insurance_policies | `blind_similarity` | mixed-coh2 | 0.2891 | 0.4005 |
| coil2000_insurance_policies | `blind_similarity` | curriculum-coh2 | 0.2889 | 0.4336 |
| coil2000_insurance_policies | `blind_similarity` | plain-coh8 | 0.2739 | 0.5010 |
| coil2000_insurance_policies | `blind_similarity` | mixed-coh8 | 0.2808 | 0.4145 |
| coil2000_insurance_policies | `blind_similarity` | curriculum-coh8 | 0.2852 | 0.4565 |
| hazelnut-spread-contaminant-detection | `vanilla` | plain-coh2 | 0.4622 | 0.9362 |
| hazelnut-spread-contaminant-detection | `vanilla` | mixed-coh2 | 0.4110 | 0.9349 |
| hazelnut-spread-contaminant-detection | `vanilla` | curriculum-coh2 | 0.3824 | 0.9325 |
| hazelnut-spread-contaminant-detection | `vanilla` | plain-coh8 | 0.5065 | 0.9287 |
| hazelnut-spread-contaminant-detection | `vanilla` | mixed-coh8 | 0.4064 | 0.9359 |
| hazelnut-spread-contaminant-detection | `vanilla` | curriculum-coh8 | 0.3739 | 0.9308 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | plain-coh2 | 0.4062 | 0.9262 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | mixed-coh2 | 0.3813 | 0.9415 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | curriculum-coh2 | 0.4022 | 0.9371 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | plain-coh8 | 0.4375 | 0.9253 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | mixed-coh8 | 0.3475 | 0.9398 |
| hazelnut-spread-contaminant-detection | `decoder-alpha` | curriculum-coh8 | 0.3460 | 0.9381 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | plain-coh2 | 0.3863 | 0.9318 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | mixed-coh2 | 0.3830 | 0.9434 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | curriculum-coh2 | 0.3799 | 0.9423 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | plain-coh8 | 0.4204 | 0.9116 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | mixed-coh8 | 0.3799 | 0.9310 |
| hazelnut-spread-contaminant-detection | `decoder-attention` | curriculum-coh8 | 0.3594 | 0.9274 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | plain-coh2 | 0.4134 | 0.9289 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | mixed-coh2 | 0.3707 | 0.9405 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | curriculum-coh2 | 0.3976 | 0.9366 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | plain-coh8 | 0.4292 | 0.9296 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | mixed-coh8 | 0.3539 | 0.9335 |
| hazelnut-spread-contaminant-detection | `blind_decoder-alpha` | curriculum-coh8 | 0.3704 | 0.9384 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | plain-coh2 | 0.4764 | 0.9435 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | mixed-coh2 | 0.4334 | 0.9374 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | curriculum-coh2 | 0.3973 | 0.9420 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | plain-coh8 | 0.7179 | 0.9267 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | mixed-coh8 | 0.3951 | 0.9340 |
| hazelnut-spread-contaminant-detection | `blind_decoder-attention` | curriculum-coh8 | 0.3686 | 0.9333 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | plain-coh2 | 0.3521 | 0.9340 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | mixed-coh2 | 0.3847 | 0.9371 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | curriculum-coh2 | 0.3738 | 0.9384 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | plain-coh8 | 0.4019 | 0.9325 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | mixed-coh8 | 0.3880 | 0.9371 |
| hazelnut-spread-contaminant-detection | `blind_similarity` | curriculum-coh8 | 0.3587 | 0.9294 |
| heloc | `vanilla` | plain-coh2 | 0.6419 | 0.7177 |
| heloc | `vanilla` | mixed-coh2 | 0.6230 | 0.7419 |
| heloc | `vanilla` | curriculum-coh2 | 0.6183 | 0.7323 |
| heloc | `vanilla` | plain-coh8 | 0.6497 | 0.7067 |
| heloc | `vanilla` | mixed-coh8 | 0.6262 | 0.7286 |
| heloc | `vanilla` | curriculum-coh8 | 0.6220 | 0.7177 |
| heloc | `decoder-alpha` | plain-coh2 | 0.6624 | 0.6875 |
| heloc | `decoder-alpha` | mixed-coh2 | 0.6301 | 0.7313 |
| heloc | `decoder-alpha` | curriculum-coh2 | 0.6305 | 0.7092 |
| heloc | `decoder-alpha` | plain-coh8 | 0.6884 | 0.6814 |
| heloc | `decoder-alpha` | mixed-coh8 | 0.6373 | 0.7021 |
| heloc | `decoder-alpha` | curriculum-coh8 | 0.6299 | 0.7042 |
| heloc | `decoder-attention` | plain-coh2 | 0.6297 | 0.7128 |
| heloc | `decoder-attention` | mixed-coh2 | 0.6206 | 0.7374 |
| heloc | `decoder-attention` | curriculum-coh2 | 0.6267 | 0.7306 |
| heloc | `decoder-attention` | plain-coh8 | 0.6353 | 0.6924 |
| heloc | `decoder-attention` | mixed-coh8 | 0.6333 | 0.7006 |
| heloc | `decoder-attention` | curriculum-coh8 | 0.6174 | 0.7220 |
| heloc | `blind_decoder-alpha` | plain-coh2 | 0.6507 | 0.6827 |
| heloc | `blind_decoder-alpha` | mixed-coh2 | 0.6366 | 0.7028 |
| heloc | `blind_decoder-alpha` | curriculum-coh2 | 0.6238 | 0.7350 |
| heloc | `blind_decoder-alpha` | plain-coh8 | 0.6652 | 0.6750 |
| heloc | `blind_decoder-alpha` | mixed-coh8 | 0.6284 | 0.7069 |
| heloc | `blind_decoder-alpha` | curriculum-coh8 | 0.6272 | 0.7143 |
| heloc | `blind_decoder-attention` | plain-coh2 | 0.6963 | 0.7172 |
| heloc | `blind_decoder-attention` | mixed-coh2 | 0.6370 | 0.7387 |
| heloc | `blind_decoder-attention` | curriculum-coh2 | 0.6429 | 0.7281 |
| heloc | `blind_decoder-attention` | plain-coh8 | 0.8132 | 0.6956 |
| heloc | `blind_decoder-attention` | mixed-coh8 | 0.6370 | 0.7134 |
| heloc | `blind_decoder-attention` | curriculum-coh8 | 0.6207 | 0.7219 |
| heloc | `blind_similarity` | plain-coh2 | 0.6247 | 0.7296 |
| heloc | `blind_similarity` | mixed-coh2 | 0.6195 | 0.7365 |
| heloc | `blind_similarity` | curriculum-coh2 | 0.6225 | 0.7374 |
| heloc | `blind_similarity` | plain-coh8 | 0.6492 | 0.6885 |
| heloc | `blind_similarity` | mixed-coh8 | 0.6282 | 0.7295 |
| heloc | `blind_similarity` | curriculum-coh8 | 0.6125 | 0.7305 |
| in_vehicle_coupon_recommendation | `vanilla` | plain-coh2 | 0.6685 | 0.5410 |
| in_vehicle_coupon_recommendation | `vanilla` | mixed-coh2 | 0.6722 | 0.5773 |
| in_vehicle_coupon_recommendation | `vanilla` | curriculum-coh2 | 0.6725 | 0.5610 |
| in_vehicle_coupon_recommendation | `vanilla` | plain-coh8 | 0.6724 | 0.5100 |
| in_vehicle_coupon_recommendation | `vanilla` | mixed-coh8 | 0.6705 | 0.5951 |
| in_vehicle_coupon_recommendation | `vanilla` | curriculum-coh8 | 0.6703 | 0.5702 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | plain-coh2 | 0.6601 | 0.5660 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | mixed-coh2 | 0.6637 | 0.5394 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | curriculum-coh2 | 0.6587 | 0.5733 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | plain-coh8 | 0.6697 | 0.5984 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | mixed-coh8 | 0.6589 | 0.5846 |
| in_vehicle_coupon_recommendation | `decoder-alpha` | curriculum-coh8 | 0.6614 | 0.5842 |
| in_vehicle_coupon_recommendation | `decoder-attention` | plain-coh2 | 0.6606 | 0.5594 |
| in_vehicle_coupon_recommendation | `decoder-attention` | mixed-coh2 | 0.6523 | 0.6148 |
| in_vehicle_coupon_recommendation | `decoder-attention` | curriculum-coh2 | 0.6604 | 0.5969 |
| in_vehicle_coupon_recommendation | `decoder-attention` | plain-coh8 | 0.6932 | 0.6060 |
| in_vehicle_coupon_recommendation | `decoder-attention` | mixed-coh8 | 0.6563 | 0.6046 |
| in_vehicle_coupon_recommendation | `decoder-attention` | curriculum-coh8 | 0.6767 | 0.6020 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | plain-coh2 | 0.6639 | 0.5787 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | mixed-coh2 | 0.6617 | 0.5833 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | curriculum-coh2 | 0.6589 | 0.6540 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | plain-coh8 | 0.6540 | 0.5897 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | mixed-coh8 | 0.6713 | 0.5658 |
| in_vehicle_coupon_recommendation | `blind_decoder-alpha` | curriculum-coh8 | 0.6615 | 0.5993 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | plain-coh2 | 0.7121 | 0.5971 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | mixed-coh2 | 0.6662 | 0.6068 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | curriculum-coh2 | 0.6686 | 0.6192 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | plain-coh8 | 0.8876 | 0.5951 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | mixed-coh8 | 0.6723 | 0.6126 |
| in_vehicle_coupon_recommendation | `blind_decoder-attention` | curriculum-coh8 | 0.6603 | 0.5966 |
| in_vehicle_coupon_recommendation | `blind_similarity` | plain-coh2 | 0.6539 | 0.6297 |
| in_vehicle_coupon_recommendation | `blind_similarity` | mixed-coh2 | 0.6506 | 0.6359 |
| in_vehicle_coupon_recommendation | `blind_similarity` | curriculum-coh2 | 0.6620 | 0.6325 |
| in_vehicle_coupon_recommendation | `blind_similarity` | plain-coh8 | 0.6508 | 0.6219 |
| in_vehicle_coupon_recommendation | `blind_similarity` | mixed-coh8 | 0.6654 | 0.6327 |
| in_vehicle_coupon_recommendation | `blind_similarity` | curriculum-coh8 | 0.6597 | 0.6170 |
| maternal_health_risk | `vanilla` | plain-coh2 | 0.7581 | 0.8118 |
| maternal_health_risk | `vanilla` | mixed-coh2 | 0.7563 | 0.7930 |
| maternal_health_risk | `vanilla` | curriculum-coh2 | 0.7789 | 0.7748 |
| maternal_health_risk | `vanilla` | plain-coh8 | 0.7589 | 0.7927 |
| maternal_health_risk | `vanilla` | mixed-coh8 | 0.7844 | 0.7516 |
| maternal_health_risk | `vanilla` | curriculum-coh8 | 0.8049 | 0.7425 |
| maternal_health_risk | `decoder-alpha` | plain-coh2 | 0.7346 | 0.8018 |
| maternal_health_risk | `decoder-alpha` | mixed-coh2 | 0.7202 | 0.8155 |
| maternal_health_risk | `decoder-alpha` | curriculum-coh2 | 0.7899 | 0.7832 |
| maternal_health_risk | `decoder-alpha` | plain-coh8 | 0.7738 | 0.7896 |
| maternal_health_risk | `decoder-alpha` | mixed-coh8 | 0.7750 | 0.7562 |
| maternal_health_risk | `decoder-alpha` | curriculum-coh8 | 0.8063 | 0.7509 |
| maternal_health_risk | `decoder-attention` | plain-coh2 | 0.7385 | 0.7938 |
| maternal_health_risk | `decoder-attention` | mixed-coh2 | 0.7180 | 0.8290 |
| maternal_health_risk | `decoder-attention` | curriculum-coh2 | 0.7447 | 0.8022 |
| maternal_health_risk | `decoder-attention` | plain-coh8 | 0.7561 | 0.7997 |
| maternal_health_risk | `decoder-attention` | mixed-coh8 | 0.7851 | 0.7660 |
| maternal_health_risk | `decoder-attention` | curriculum-coh8 | 0.7761 | 0.7644 |
| maternal_health_risk | `blind_decoder-alpha` | plain-coh2 | 0.7477 | 0.7993 |
| maternal_health_risk | `blind_decoder-alpha` | mixed-coh2 | 0.7405 | 0.7922 |
| maternal_health_risk | `blind_decoder-alpha` | curriculum-coh2 | 0.7827 | 0.7624 |
| maternal_health_risk | `blind_decoder-alpha` | plain-coh8 | 0.7929 | 0.7752 |
| maternal_health_risk | `blind_decoder-alpha` | mixed-coh8 | 0.7978 | 0.7529 |
| maternal_health_risk | `blind_decoder-alpha` | curriculum-coh8 | 0.7924 | 0.7380 |
| maternal_health_risk | `blind_decoder-attention` | plain-coh2 | 0.6990 | 0.8225 |
| maternal_health_risk | `blind_decoder-attention` | mixed-coh2 | 0.7449 | 0.8039 |
| maternal_health_risk | `blind_decoder-attention` | curriculum-coh2 | 0.7432 | 0.8138 |
| maternal_health_risk | `blind_decoder-attention` | plain-coh8 | 0.7601 | 0.7838 |
| maternal_health_risk | `blind_decoder-attention` | mixed-coh8 | 0.7403 | 0.7830 |
| maternal_health_risk | `blind_decoder-attention` | curriculum-coh8 | 0.8055 | 0.7610 |
| maternal_health_risk | `blind_similarity` | plain-coh2 | 0.7511 | 0.7878 |
| maternal_health_risk | `blind_similarity` | mixed-coh2 | 0.7356 | 0.8195 |
| maternal_health_risk | `blind_similarity` | curriculum-coh2 | 0.7241 | 0.7996 |
| maternal_health_risk | `blind_similarity` | plain-coh8 | 0.7702 | 0.7885 |
| maternal_health_risk | `blind_similarity` | mixed-coh8 | 0.7708 | 0.7888 |
| maternal_health_risk | `blind_similarity` | curriculum-coh8 | 0.8007 | 0.7301 |
| online_shoppers_intention | `vanilla` | plain-coh2 | 0.3051 | 0.8882 |
| online_shoppers_intention | `vanilla` | mixed-coh2 | 0.2764 | 0.8948 |
| online_shoppers_intention | `vanilla` | curriculum-coh2 | 0.2660 | 0.8941 |
| online_shoppers_intention | `vanilla` | plain-coh8 | 0.3298 | 0.8676 |
| online_shoppers_intention | `vanilla` | mixed-coh8 | 0.2837 | 0.8955 |
| online_shoppers_intention | `vanilla` | curriculum-coh8 | 0.2822 | 0.8882 |
| online_shoppers_intention | `decoder-alpha` | plain-coh2 | 0.3248 | 0.8556 |
| online_shoppers_intention | `decoder-alpha` | mixed-coh2 | 0.3037 | 0.8945 |
| online_shoppers_intention | `decoder-alpha` | curriculum-coh2 | 0.2924 | 0.8888 |
| online_shoppers_intention | `decoder-alpha` | plain-coh8 | 0.3048 | 0.8662 |
| online_shoppers_intention | `decoder-alpha` | mixed-coh8 | 0.2889 | 0.8759 |
| online_shoppers_intention | `decoder-alpha` | curriculum-coh8 | 0.2952 | 0.8745 |
| online_shoppers_intention | `decoder-attention` | plain-coh2 | 0.2700 | 0.8898 |
| online_shoppers_intention | `decoder-attention` | mixed-coh2 | 0.2757 | 0.8978 |
| online_shoppers_intention | `decoder-attention` | curriculum-coh2 | 0.2766 | 0.8911 |
| online_shoppers_intention | `decoder-attention` | plain-coh8 | 0.2863 | 0.8808 |
| online_shoppers_intention | `decoder-attention` | mixed-coh8 | 0.2638 | 0.8938 |
| online_shoppers_intention | `decoder-attention` | curriculum-coh8 | 0.2668 | 0.8858 |
| online_shoppers_intention | `blind_decoder-alpha` | plain-coh2 | 0.3152 | 0.8699 |
| online_shoppers_intention | `blind_decoder-alpha` | mixed-coh2 | 0.2967 | 0.8828 |
| online_shoppers_intention | `blind_decoder-alpha` | curriculum-coh2 | 0.2859 | 0.8815 |
| online_shoppers_intention | `blind_decoder-alpha` | plain-coh8 | 0.2971 | 0.8808 |
| online_shoppers_intention | `blind_decoder-alpha` | mixed-coh8 | 0.2769 | 0.8802 |
| online_shoppers_intention | `blind_decoder-alpha` | curriculum-coh8 | 0.3048 | 0.8775 |
| online_shoppers_intention | `blind_decoder-attention` | plain-coh2 | 0.2822 | 0.8911 |
| online_shoppers_intention | `blind_decoder-attention` | mixed-coh2 | 0.2799 | 0.8961 |
| online_shoppers_intention | `blind_decoder-attention` | curriculum-coh2 | 0.2736 | 0.9031 |
| online_shoppers_intention | `blind_decoder-attention` | plain-coh8 | 0.3456 | 0.8493 |
| online_shoppers_intention | `blind_decoder-attention` | mixed-coh8 | 0.2672 | 0.8971 |
| online_shoppers_intention | `blind_decoder-attention` | curriculum-coh8 | 0.2711 | 0.8951 |
| online_shoppers_intention | `blind_similarity` | plain-coh2 | 0.2570 | 0.8948 |
| online_shoppers_intention | `blind_similarity` | mixed-coh2 | 0.2768 | 0.8955 |
| online_shoppers_intention | `blind_similarity` | curriculum-coh2 | 0.2808 | 0.8935 |
| online_shoppers_intention | `blind_similarity` | plain-coh8 | 0.2823 | 0.8878 |
| online_shoppers_intention | `blind_similarity` | mixed-coh8 | 0.2803 | 0.8911 |
| online_shoppers_intention | `blind_similarity` | curriculum-coh8 | 0.2843 | 0.8888 |
| polish_companies_bankruptcy | `vanilla` | plain-coh2 | 0.2225 | 0.7438 |
| polish_companies_bankruptcy | `vanilla` | mixed-coh2 | 0.2203 | 0.7764 |
| polish_companies_bankruptcy | `vanilla` | curriculum-coh2 | 0.2247 | 0.7660 |
| polish_companies_bankruptcy | `vanilla` | plain-coh8 | 0.2269 | 0.7396 |
| polish_companies_bankruptcy | `vanilla` | mixed-coh8 | 0.2217 | 0.7833 |
| polish_companies_bankruptcy | `vanilla` | curriculum-coh8 | 0.2265 | 0.7854 |
| polish_companies_bankruptcy | `decoder-alpha` | plain-coh2 | 0.2103 | 0.8083 |
| polish_companies_bankruptcy | `decoder-alpha` | mixed-coh2 | 0.2318 | 0.7694 |
| polish_companies_bankruptcy | `decoder-alpha` | curriculum-coh2 | 0.2172 | 0.7687 |
| polish_companies_bankruptcy | `decoder-alpha` | plain-coh8 | 0.2649 | 0.7812 |
| polish_companies_bankruptcy | `decoder-alpha` | mixed-coh8 | 0.2299 | 0.7896 |
| polish_companies_bankruptcy | `decoder-alpha` | curriculum-coh8 | 0.2223 | 0.7764 |
| polish_companies_bankruptcy | `decoder-attention` | plain-coh2 | 0.2187 | 0.7750 |
| polish_companies_bankruptcy | `decoder-attention` | mixed-coh2 | 0.2075 | 0.8229 |
| polish_companies_bankruptcy | `decoder-attention` | curriculum-coh2 | 0.1979 | 0.8271 |
| polish_companies_bankruptcy | `decoder-attention` | plain-coh8 | 0.2593 | 0.7660 |
| polish_companies_bankruptcy | `decoder-attention` | mixed-coh8 | 0.2331 | 0.7896 |
| polish_companies_bankruptcy | `decoder-attention` | curriculum-coh8 | 0.2163 | 0.8104 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | plain-coh2 | 0.2307 | 0.7736 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | mixed-coh2 | 0.2228 | 0.7986 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | curriculum-coh2 | 0.2097 | 0.7938 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | plain-coh8 | 0.2653 | 0.7771 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | mixed-coh8 | 0.2189 | 0.8104 |
| polish_companies_bankruptcy | `blind_decoder-alpha` | curriculum-coh8 | 0.2248 | 0.7799 |
| polish_companies_bankruptcy | `blind_decoder-attention` | plain-coh2 | 0.2335 | 0.7958 |
| polish_companies_bankruptcy | `blind_decoder-attention` | mixed-coh2 | 0.2005 | 0.8014 |
| polish_companies_bankruptcy | `blind_decoder-attention` | curriculum-coh2 | 0.2227 | 0.8007 |
| polish_companies_bankruptcy | `blind_decoder-attention` | plain-coh8 | 0.2477 | 0.6979 |
| polish_companies_bankruptcy | `blind_decoder-attention` | mixed-coh8 | 0.1981 | 0.8229 |
| polish_companies_bankruptcy | `blind_decoder-attention` | curriculum-coh8 | 0.2542 | 0.7778 |
| polish_companies_bankruptcy | `blind_similarity` | plain-coh2 | 0.2116 | 0.7847 |
| polish_companies_bankruptcy | `blind_similarity` | mixed-coh2 | 0.2003 | 0.8201 |
| polish_companies_bankruptcy | `blind_similarity` | curriculum-coh2 | 0.2074 | 0.8104 |
| polish_companies_bankruptcy | `blind_similarity` | plain-coh8 | 0.2433 | 0.7729 |
| polish_companies_bankruptcy | `blind_similarity` | mixed-coh8 | 0.2086 | 0.8118 |
| polish_companies_bankruptcy | `blind_similarity` | curriculum-coh8 | 0.2078 | 0.8125 |
| qsar-biodeg | `vanilla` | plain-coh2 | 0.4065 | 0.9367 |
| qsar-biodeg | `vanilla` | mixed-coh2 | 0.3802 | 0.9427 |
| qsar-biodeg | `vanilla` | curriculum-coh2 | 0.3603 | 0.9333 |
| qsar-biodeg | `vanilla` | plain-coh8 | 0.4258 | 0.9193 |
| qsar-biodeg | `vanilla` | mixed-coh8 | 0.3703 | 0.9340 |
| qsar-biodeg | `vanilla` | curriculum-coh8 | 0.3657 | 0.9195 |
| qsar-biodeg | `decoder-alpha` | plain-coh2 | 0.4045 | 0.9104 |
| qsar-biodeg | `decoder-alpha` | mixed-coh2 | 0.3548 | 0.9383 |
| qsar-biodeg | `decoder-alpha` | curriculum-coh2 | 0.3620 | 0.9417 |
| qsar-biodeg | `decoder-alpha` | plain-coh8 | 0.4781 | 0.9182 |
| qsar-biodeg | `decoder-alpha` | mixed-coh8 | 0.3319 | 0.9389 |
| qsar-biodeg | `decoder-alpha` | curriculum-coh8 | 0.3413 | 0.9376 |
| qsar-biodeg | `decoder-attention` | plain-coh2 | 0.3538 | 0.9227 |
| qsar-biodeg | `decoder-attention` | mixed-coh2 | 0.3499 | 0.9532 |
| qsar-biodeg | `decoder-attention` | curriculum-coh2 | 0.3294 | 0.9478 |
| qsar-biodeg | `decoder-attention` | plain-coh8 | 0.4148 | 0.8959 |
| qsar-biodeg | `decoder-attention` | mixed-coh8 | 0.3223 | 0.9416 |
| qsar-biodeg | `decoder-attention` | curriculum-coh8 | 0.3453 | 0.9240 |
| qsar-biodeg | `blind_decoder-alpha` | plain-coh2 | 0.4491 | 0.9176 |
| qsar-biodeg | `blind_decoder-alpha` | mixed-coh2 | 0.3310 | 0.9395 |
| qsar-biodeg | `blind_decoder-alpha` | curriculum-coh2 | 0.3465 | 0.9531 |
| qsar-biodeg | `blind_decoder-alpha` | plain-coh8 | 0.4739 | 0.8918 |
| qsar-biodeg | `blind_decoder-alpha` | mixed-coh8 | 0.3410 | 0.9397 |
| qsar-biodeg | `blind_decoder-alpha` | curriculum-coh8 | 0.3558 | 0.9385 |
| qsar-biodeg | `blind_decoder-attention` | plain-coh2 | 0.4693 | 0.9563 |
| qsar-biodeg | `blind_decoder-attention` | mixed-coh2 | 0.3524 | 0.9527 |
| qsar-biodeg | `blind_decoder-attention` | curriculum-coh2 | 0.3307 | 0.9621 |
| qsar-biodeg | `blind_decoder-attention` | plain-coh8 | 0.6685 | 0.9165 |
| qsar-biodeg | `blind_decoder-attention` | mixed-coh8 | 0.3291 | 0.9465 |
| qsar-biodeg | `blind_decoder-attention` | curriculum-coh8 | 0.3432 | 0.9304 |
| qsar-biodeg | `blind_similarity` | plain-coh2 | 0.3799 | 0.9380 |
| qsar-biodeg | `blind_similarity` | mixed-coh2 | 0.3409 | 0.9451 |
| qsar-biodeg | `blind_similarity` | curriculum-coh2 | 0.3383 | 0.9570 |
| qsar-biodeg | `blind_similarity` | plain-coh8 | 0.4032 | 0.9289 |
| qsar-biodeg | `blind_similarity` | mixed-coh8 | 0.3276 | 0.9455 |
| qsar-biodeg | `blind_similarity` | curriculum-coh8 | 0.3316 | 0.9372 |
| website_phishing | `vanilla` | plain-coh2 | 0.4225 | 0.8809 |
| website_phishing | `vanilla` | mixed-coh2 | 0.4293 | 0.8769 |
| website_phishing | `vanilla` | curriculum-coh2 | 0.4331 | 0.8504 |
| website_phishing | `vanilla` | plain-coh8 | 0.4376 | 0.8556 |
| website_phishing | `vanilla` | mixed-coh8 | 0.4321 | 0.8668 |
| website_phishing | `vanilla` | curriculum-coh8 | 0.4346 | 0.8447 |
| website_phishing | `decoder-alpha` | plain-coh2 | 0.4493 | 0.8958 |
| website_phishing | `decoder-alpha` | mixed-coh2 | 0.4339 | 0.8668 |
| website_phishing | `decoder-alpha` | curriculum-coh2 | 0.4474 | 0.8353 |
| website_phishing | `decoder-alpha` | plain-coh8 | 0.5097 | 0.8094 |
| website_phishing | `decoder-alpha` | mixed-coh8 | 0.4727 | 0.8175 |
| website_phishing | `decoder-alpha` | curriculum-coh8 | 0.4779 | 0.8290 |
| website_phishing | `decoder-attention` | plain-coh2 | 0.4899 | 0.8652 |
| website_phishing | `decoder-attention` | mixed-coh2 | 0.4206 | 0.8775 |
| website_phishing | `decoder-attention` | curriculum-coh2 | 0.4231 | 0.8820 |
| website_phishing | `decoder-attention` | plain-coh8 | 0.4622 | 0.8786 |
| website_phishing | `decoder-attention` | mixed-coh8 | 0.4361 | 0.8771 |
| website_phishing | `decoder-attention` | curriculum-coh8 | 0.4550 | 0.8620 |
| website_phishing | `blind_decoder-alpha` | plain-coh2 | 0.4565 | 0.8792 |
| website_phishing | `blind_decoder-alpha` | mixed-coh2 | 0.4422 | 0.8692 |
| website_phishing | `blind_decoder-alpha` | curriculum-coh2 | 0.4510 | 0.8516 |
| website_phishing | `blind_decoder-alpha` | plain-coh8 | 0.5122 | 0.7939 |
| website_phishing | `blind_decoder-alpha` | mixed-coh8 | 0.4449 | 0.7857 |
| website_phishing | `blind_decoder-alpha` | curriculum-coh8 | 0.4687 | 0.8023 |
| website_phishing | `blind_decoder-attention` | plain-coh2 | 0.3946 | 0.9059 |
| website_phishing | `blind_decoder-attention` | mixed-coh2 | 0.4043 | 0.8888 |
| website_phishing | `blind_decoder-attention` | curriculum-coh2 | 0.4322 | 0.9112 |
| website_phishing | `blind_decoder-attention` | plain-coh8 | 0.4401 | 0.8829 |
| website_phishing | `blind_decoder-attention` | mixed-coh8 | 0.4176 | 0.8967 |
| website_phishing | `blind_decoder-attention` | curriculum-coh8 | 0.4536 | 0.8560 |
| website_phishing | `blind_similarity` | plain-coh2 | 0.4199 | 0.8869 |
| website_phishing | `blind_similarity` | mixed-coh2 | 0.4159 | 0.8894 |
| website_phishing | `blind_similarity` | curriculum-coh2 | 0.4108 | 0.8717 |
| website_phishing | `blind_similarity` | plain-coh8 | 0.4922 | 0.8447 |
| website_phishing | `blind_similarity` | mixed-coh8 | 0.4443 | 0.8481 |
| website_phishing | `blind_similarity` | curriculum-coh8 | 0.4458 | 0.8386 |
