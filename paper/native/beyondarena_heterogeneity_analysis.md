# BeyondArena heterogeneity analysis

Source: [native_prior_results.md](native_prior_results.md). This report compares the final six-layer native canonical and curriculum NanoTabPFN checkpoints task by task. It does not re-evaluate the benchmark.

The source table contains 36 tasks: 27 binary (26 IID, one grouped) and nine multiclass (six IID, two grouped, one temporal). Grouped and temporal rows are too few for inferential claims.

Differences are **curriculum minus canonical**. For excess CE, negative favors curriculum; for accuracy gain and AUC, positive favors curriculum. The values are rounded to 0.001 in the source table; tie counts use an absolute tolerance of 0.0005.

| Slice | Tasks | Δ excess CE | CE W/T/L | Δ accuracy gain (pp) | Accuracy W/T/L | Δ AUC | AUC W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 36 | +0.0036 | 16/2/18 | -0.10 | 15/5/16 | -0.0014 | 13/5/18 |
| binary | 27 | +0.0049 | 11/2/14 | -0.11 | 12/4/11 | -0.0013 | 9/4/14 |
| multiclass | 9 | -0.0003 | 5/0/4 | -0.07 | 3/1/5 | -0.0017 | 4/1/4 |
| IID | 32 | +0.0021 | 15/2/15 | -0.07 | 15/4/13 | -0.0003 | 12/5/15 |
| grouped | 3 | +0.0187 | 1/0/2 | -0.27 | 0/1/2 | -0.0060 | 1/0/2 |
| temporal | 1 | +0.0070 | 0/0/1 | -0.60 | 0/0/1 | -0.0250 | 0/0/1 |
| binary × IID | 26 | +0.0028 | 11/2/13 | -0.10 | 12/4/10 | -0.0007 | 9/4/13 |
| binary × grouped | 1 | +0.0600 | 0/0/1 | -0.30 | 0/0/1 | -0.0180 | 0/0/1 |
| multiclass × IID | 6 | -0.0010 | 4/0/2 | +0.08 | 3/0/3 | +0.0017 | 3/1/2 |
| multiclass × grouped | 2 | -0.0020 | 1/0/1 | -0.25 | 0/1/1 | +0.0000 | 1/0/1 |
| multiclass × temporal | 1 | +0.0070 | 0/0/1 | -0.60 | 0/0/1 | -0.0250 | 0/0/1 |

W/T/L is the number of tasks favoring curriculum / tied at printed precision / favoring canonical.

## Interpretation

The all-task comparison tests whether the synthetic-prior change produces a broad real-data advantage. The IID/grouped/temporal rows instead describe where the existing task-level differences occur; they do not establish an effect of group shift because the groups contain only 32, three, and one tasks, respectively.

The task-level Spearman correlations below are exploratory and should not be used to select a favorable subgroup. They test only whether the observed curriculum-minus-canonical difference covaries with coarse dataset descriptors already present in the appendix.

| Metric difference | log(training rows) | features | majority fraction |
|---|---:|---:|---:|
| excess_cross_entropy | +0.302 | +0.252 | +0.194 |
| accuracy_gain | -0.076 | -0.225 | -0.092 |
| macro_ovr_auc | -0.248 | -0.133 | -0.187 |

The complementary merged-heart-site experiment is the stronger real-data group-information diagnostic: the hospital is highly recoverable from features in IID folds, adding its label changes little, and multiregime pretraining does not consistently alter that pattern. See [heart_sites_group_feature.md](heart_sites_group_feature.md).

## Scope

This report is descriptive rather than confirmatory. It uses one trained model per condition and values rounded in the source table; it cannot estimate training-seed uncertainty or prove that a benchmark split regime corresponds to latent regime-dependent label mechanisms.
