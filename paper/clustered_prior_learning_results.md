# Easy clustered multiregime prior: empirical learning report

Date: 2026-09-07.

The easy prior is empirically learnable. The existing small nanoTabPFN, without regime IDs, achieves **94.82% mean held-out accuracy** across 3 training seeds (range 94.33–95.78%). Shuffling its support labels reduces mean accuracy to **49.78%**. The oracle transformer averages 79.84%. A simple clustering-plus-classification learner reaches 98.54%.

## Setup

Two regimes sampled independently with probability 1/2. Three features: a regime cue x₁ = 3(2z−1) + ε with ε ~ N(0,1), and two independent standard-normal task features. Each episode draws a fresh random linear rule and an orthogonal second rule, with random handedness. Labels are deterministic within each regime; there is no added label noise. Regimes are balanced in expectation, not forced to have equal counts.

All final comparisons use the same 256 unseen episodes with 256 queries each (65,536 queries). Classical support curves use nested prefixes of a common 128-row support set. K-means fits only support features; its cluster IDs route support and queries to separate logistic regressions on the two task features. This baseline knows the feature roles, but never receives regime IDs or rule coefficients. The oracle uses true support/query regime IDs. The pooled comparator is a single linear logistic classifier on all three features, so it cannot represent general regime-by-feature interactions.

## Classical learning curve

| Support rows | Clustered | Oracle | Pooled linear | Shuffled labels |
|---:|---:|---:|---:|---:|
| 8 | 80.44% | 80.76% | 67.11% | 52.05% |
| 16 | 91.91% | 92.27% | 71.57% | 51.57% |
| 32 | 96.34% | 96.46% | 73.66% | 52.28% |
| 64 | 97.70% | 97.80% | 74.57% | 53.23% |
| 128 | 98.54% | 98.65% | 74.83% | 52.13% |

At 128 support rows, clustered accuracy has an approximate episode-level 95% CI of 98.41–98.67%. Query routing accuracy, after permutation alignment, is 99.861%. The oracle advantage is 0.111 percentage points.

## Neural learning

Existing NanoTabPFNModel trained from scratch on CPU: 2 layers, embedding size 32, 4 heads, MLP width 64; AdamW learning rate 0.001, weight decay 0.01, gradient clipping 1.0. Each run has 1,000 updates of 8 fresh episodes, 128 support rows and 32 training queries. Latent inputs contain only features and support labels; the oracle appends two one-hot regime features. Each arm uses training seeds [11, 12, 13], with matched episode streams. Checkpoint selection minimizes CE on 32 fixed validation episodes every 200 steps and at the final step. The final test namespace is separate from training and validation.

| Arm | Seed | Selected step | Test accuracy | Test CE | Shuffled-label accuracy |
|---|---:|---:|---:|---:|---:|
| Latent | 11 | 1000 | 94.33% | 0.1378 | 50.11% |
| Oracle | 11 | 1000 | 93.37% | 0.1600 | 49.57% |
| Latent | 12 | 1000 | 94.36% | 0.1354 | 48.90% |
| Oracle | 12 | 1000 | 51.41% | 0.6884 | 49.56% |
| Latent | 13 | 1000 | 95.78% | 0.1039 | 50.34% |
| Oracle | 13 | 1000 | 94.73% | 0.1273 | 50.66% |

Oracle seeds [12] remain below 60% test accuracy within this budget. These failed runs are retained; the neural oracle is not a consistently optimized reference. The successful latent runs establish feasibility, not universal training stability.

## Interpretation and limits

The classical curve establishes that unseen episode-specific rules can be learned with little routing loss on this prior. Neural validation curves and independent test scores measure whether the existing small transformer also acquires this ability. Shuffling support labels is a negative control for dependence on the support set, not a separate training arm.

Giving a finitely trained neural model extra oracle inputs does not guarantee a higher score: its optimization and input representation also change. The paired classical oracle provides the clearer routing comparison. Use this prior as a positive control before introducing overlap, irrelevant features, or more regimes; testing the slot architecture on it is a separate next experiment.

This is an empirical feasibility experiment, not a proof of slot recovery or real-table transfer. No slot-attention model was trained here. A pooled linear model is only a limited comparator; its failure does not establish an advantage over other nonlinear models. The feature roles, strong cluster cue, two-dimensional linear rules and zero label noise are deliberately favorable. Confidence intervals in JSON use episode means (normal approximation), not individual queries; neural runs share test episodes and must not be counted as independent test datasets. Three training seeds are a small stability check.

The regime cue is not perfectly deterministic: even with known generating rules, the latent Bayes error is 0.5 Φ(−3) ≈ 0.0675%, since orthogonal rules disagree on half the task inputs. The fitted oracle is not the Bayes oracle.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.clustered_prior_learning --output results/clustered_prior_learning --steps 1000 --seeds 11 12 13
.venv/bin/python scripts/report_clustered_prior_learning.py
.venv/bin/python -m unittest tests.test_clustered_prior_learning
```

Raw metrics and validation histories: `results/clustered_prior_learning/results.json`. Per-run selected checkpoints and metrics are saved beside it.

![Learning curves](../results/clustered_prior_learning/learning_curves.png)
