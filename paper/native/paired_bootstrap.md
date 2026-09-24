# Paired episode-bootstrap analysis

This analysis compares final checkpoints on the same score-hidden native synthetic **test** bank. Every bootstrap replicate resamples episodes independently within each factorial cell and then averages cell means. `candidate − canonical` is reported: negative cross-entropy and positive accuracy difference (equivalently, accuracy-gain difference) favor the candidate.

Bootstrap configuration: 5,000 percentile replicates, 95% intervals, random seed 20260924.

These intervals quantify variation across the fixed evaluation episodes only. They do **not** quantify variation across independent pretraining seeds, and their per-slice interpretation is descriptive rather than adjusted for multiple comparisons.

## Results

| Candidate | Slice | Episodes / cells | Metric | Canonical | Candidate | Candidate − canonical (CI) | Bootstrap fraction favoring candidate |
|---|---|---:|---|---:|---:|---:|---:|
| fixed | Multiregime multiclass | 5,760 / 720 | Cross-entropy (nats) | +1.1593 | +1.1517 | -0.0076 [-0.0083, -0.0069] | 100.00% |
| fixed | Multiregime multiclass | 5,760 / 720 | Accuracy (pp) | +48.28 pp | +48.65 pp | +0.37 pp [+0.32 pp, +0.42 pp] | 100.00% |
| fixed | Soft-gate multiregime multiclass | 2,880 / 360 | Cross-entropy (nats) | +1.0876 | +1.0737 | -0.0139 [-0.0150, -0.0127] | 100.00% |
| fixed | Soft-gate multiregime multiclass | 2,880 / 360 | Accuracy (pp) | +53.19 pp | +53.86 pp | +0.67 pp [+0.59 pp, +0.75 pp] | 100.00% |
| fixed | Persistent multiregime multiclass | 2,880 / 360 | Cross-entropy (nats) | +1.2310 | +1.2297 | -0.0014 [-0.0021, -0.0006] | 100.00% |
| fixed | Persistent multiregime multiclass | 2,880 / 360 | Accuracy (pp) | +43.36 pp | +43.43 pp | +0.07 pp [+0.00 pp, +0.13 pp] | 97.84% |
| fixed | Multiregime binary | 5,760 / 720 | Cross-entropy (nats) | +0.5403 | +0.5405 | +0.0003 [-0.0000, +0.0005] | 4.24% |
| fixed | Multiregime binary | 5,760 / 720 | Accuracy (pp) | +71.08 pp | +71.05 pp | -0.03 pp [-0.08 pp, +0.02 pp] | 12.86% |
| curriculum | Multiregime multiclass | 5,760 / 720 | Cross-entropy (nats) | +1.1593 | +1.1505 | -0.0088 [-0.0094, -0.0082] | 100.00% |
| curriculum | Multiregime multiclass | 5,760 / 720 | Accuracy (pp) | +48.28 pp | +48.66 pp | +0.38 pp [+0.34 pp, +0.43 pp] | 100.00% |
| curriculum | Soft-gate multiregime multiclass | 2,880 / 360 | Cross-entropy (nats) | +1.0876 | +1.0708 | -0.0168 [-0.0179, -0.0158] | 100.00% |
| curriculum | Soft-gate multiregime multiclass | 2,880 / 360 | Accuracy (pp) | +53.19 pp | +54.03 pp | +0.84 pp [+0.76 pp, +0.91 pp] | 100.00% |
| curriculum | Persistent multiregime multiclass | 2,880 / 360 | Cross-entropy (nats) | +1.2310 | +1.2302 | -0.0008 [-0.0015, -0.0002] | 99.40% |
| curriculum | Persistent multiregime multiclass | 2,880 / 360 | Accuracy (pp) | +43.36 pp | +43.29 pp | -0.07 pp [-0.13 pp, -0.01 pp] | 1.16% |
| curriculum | Multiregime binary | 5,760 / 720 | Cross-entropy (nats) | +0.5403 | +0.5405 | +0.0003 [+0.0000, +0.0005] | 1.72% |
| curriculum | Multiregime binary | 5,760 / 720 | Accuracy (pp) | +71.08 pp | +71.04 pp | -0.03 pp [-0.09 pp, +0.02 pp] | 11.08% |

Inputs:

- Canonical: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/original-large/seed-2402/v4_test/final.json`
- fixed: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/rg_z-fixed-large/seed-2402/v4_test/final.json`
- curriculum: `artifacts/cluster_sync/tfm_eval/native_regime_split/blind/rg_z-curriculum-large/seed-2402/v4_test/final.json`
