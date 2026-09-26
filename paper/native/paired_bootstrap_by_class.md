# Paired episode-bootstrap analysis

This analysis compares final checkpoints on the same score-hidden native synthetic **test** bank. Every bootstrap replicate resamples episodes independently within each factorial cell and then averages cell means. `candidate − canonical` is reported: negative cross-entropy and positive accuracy difference (equivalently, accuracy-gain difference) favor the candidate.

Bootstrap configuration: 5,000 percentile replicates, 95% intervals, random seed 20260924.

These intervals quantify variation across the fixed evaluation episodes only. They do **not** quantify variation across independent pretraining seeds, and their per-slice interpretation is descriptive rather than adjusted for multiple comparisons.

## Results

| Candidate | Slice | Episodes / cells | Metric | Canonical | Candidate | Candidate − canonical (CI) | Bootstrap fraction favoring candidate |
|---|---|---:|---|---:|---:|---:|---:|
| curriculum | Soft-gate multiregime, 3 classes | 960 / 120 | Cross-entropy (nats) | +0.8296 | +0.8167 | -0.0129 [-0.0146, -0.0112] | 100.00% |
| curriculum | Soft-gate multiregime, 3 classes | 960 / 120 | Accuracy (pp) | +62.39 pp | +63.18 pp | +0.79 pp [+0.67 pp, +0.92 pp] | 100.00% |
| curriculum | Soft-gate multiregime, 4 classes | 960 / 120 | Cross-entropy (nats) | +1.1013 | +1.0851 | -0.0162 [-0.0182, -0.0142] | 100.00% |
| curriculum | Soft-gate multiregime, 4 classes | 960 / 120 | Accuracy (pp) | +52.31 pp | +53.10 pp | +0.79 pp [+0.64 pp, +0.93 pp] | 100.00% |
| curriculum | Soft-gate multiregime, 5 classes | 960 / 120 | Cross-entropy (nats) | +1.3318 | +1.3105 | -0.0213 [-0.0231, -0.0195] | 100.00% |
| curriculum | Soft-gate multiregime, 5 classes | 960 / 120 | Accuracy (pp) | +44.87 pp | +45.81 pp | +0.94 pp [+0.80 pp, +1.08 pp] | 100.00% |
| curriculum | Persistent multiregime, 3 classes | 960 / 120 | Cross-entropy (nats) | +0.9813 | +0.9803 | -0.0010 [-0.0019, -0.0001] | 98.48% |
| curriculum | Persistent multiregime, 3 classes | 960 / 120 | Accuracy (pp) | +51.09 pp | +51.00 pp | -0.09 pp [-0.19 pp, +0.01 pp] | 5.04% |
| curriculum | Persistent multiregime, 4 classes | 960 / 120 | Cross-entropy (nats) | +1.2536 | +1.2532 | -0.0004 [-0.0016, +0.0008] | 72.22% |
| curriculum | Persistent multiregime, 4 classes | 960 / 120 | Accuracy (pp) | +42.19 pp | +42.23 pp | +0.04 pp [-0.07 pp, +0.14 pp] | 75.70% |
| curriculum | Persistent multiregime, 5 classes | 960 / 120 | Cross-entropy (nats) | +1.4583 | +1.4572 | -0.0010 [-0.0024, +0.0003] | 93.60% |
| curriculum | Persistent multiregime, 5 classes | 960 / 120 | Accuracy (pp) | +36.81 pp | +36.65 pp | -0.16 pp [-0.26 pp, -0.05 pp] | 0.10% |
| fixed | Soft-gate multiregime, 3 classes | 960 / 120 | Cross-entropy (nats) | +0.8296 | +0.8203 | -0.0093 [-0.0111, -0.0076] | 100.00% |
| fixed | Soft-gate multiregime, 3 classes | 960 / 120 | Accuracy (pp) | +62.39 pp | +62.94 pp | +0.55 pp [+0.42 pp, +0.69 pp] | 100.00% |
| fixed | Soft-gate multiregime, 4 classes | 960 / 120 | Cross-entropy (nats) | +1.1013 | +1.0867 | -0.0147 [-0.0167, -0.0128] | 100.00% |
| fixed | Soft-gate multiregime, 4 classes | 960 / 120 | Accuracy (pp) | +52.31 pp | +52.98 pp | +0.67 pp [+0.52 pp, +0.82 pp] | 100.00% |
| fixed | Soft-gate multiregime, 5 classes | 960 / 120 | Cross-entropy (nats) | +1.3318 | +1.3141 | -0.0176 [-0.0196, -0.0156] | 100.00% |
| fixed | Soft-gate multiregime, 5 classes | 960 / 120 | Accuracy (pp) | +44.87 pp | +45.65 pp | +0.79 pp [+0.65 pp, +0.92 pp] | 100.00% |
| fixed | Persistent multiregime, 3 classes | 960 / 120 | Cross-entropy (nats) | +0.9813 | +0.9811 | -0.0002 [-0.0013, +0.0009] | 65.00% |
| fixed | Persistent multiregime, 3 classes | 960 / 120 | Accuracy (pp) | +51.09 pp | +51.11 pp | +0.02 pp [-0.08 pp, +0.13 pp] | 67.06% |
| fixed | Persistent multiregime, 4 classes | 960 / 120 | Cross-entropy (nats) | +1.2536 | +1.2520 | -0.0015 [-0.0030, -0.0001] | 98.16% |
| fixed | Persistent multiregime, 4 classes | 960 / 120 | Accuracy (pp) | +42.19 pp | +42.34 pp | +0.15 pp [+0.04 pp, +0.26 pp] | 99.68% |
| fixed | Persistent multiregime, 5 classes | 960 / 120 | Cross-entropy (nats) | +1.4583 | +1.4559 | -0.0024 [-0.0039, -0.0009] | 99.84% |
| fixed | Persistent multiregime, 5 classes | 960 / 120 | Accuracy (pp) | +36.81 pp | +36.83 pp | +0.02 pp [-0.10 pp, +0.14 pp] | 63.72% |

Inputs:

- Canonical: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/original-large/seed-2402/v4_test/final.json`
- curriculum: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/rg_z-curriculum-large/seed-2402/v4_test/final.json`
- fixed: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/rg_z-fixed-large/seed-2402/v4_test/final.json`
