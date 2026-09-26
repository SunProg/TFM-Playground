# Training-time observed-ID comparison

This is an exploratory, descriptive comparison of the saved seed-2402,
step-10,000 small and medium checkpoints. The observed-ID configs enable
`v4_expose_regime_id`; the predictor therefore receives row-level regime IDs
during pretraining. This differs from the frozen-checkpoint diagnostic in
`regime_information_diagnostic.md`, which holds trained weights fixed and
changes only the evaluation-time input condition.

## Evaluation and uncertainty

The eight archived reports each contain 5,760 multiclass multiregime episodes
under hidden, shuffled-tag, and true-tag inputs. They record the same bank
path, and have identical episode IDs and episode metadata: 720 factorial
cells with eight episodes per cell. Hidden input is the primary comparison.
The observed-minus-original contrasts pair episodes, average cell means
equally, and use 5,000 bootstrap replicates that resample episodes within each
cell. The intervals quantify uncertainty over this fixed episode bank; they
do not estimate training-seed variation.

| Size | Observed-ID minus original, hidden CE (95% interval) | Accuracy (percentage points, 95% interval) |
|---|---:|---:|
| Small | +0.00493 [+0.00417, +0.00568] | -0.327 [-0.422, -0.239] |
| Medium | +0.00245 [+0.00089, +0.00397] | -0.867 [-0.963, -0.771] |

Absolute cross-entropy and accuracy for all three input conditions, including
the fixed-mixture and curriculum references, are in
[`appendix_observed_id_table.tex`](../iclr2027/appendix_observed_id_table.tex).
Machine-readable scores, interval endpoints, report checksums, and input
paths are in `observed_id_training_comparison.json`.

## Provenance limits

The reports record these model-source SHA-256 values:

- Original: `5ed1a46b2939aa0c8db1e3766a27a9cae9db6e36ac0a8b165471df7bcc14339c`
- Fixed-mixture and curriculum: `32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34`
- Observed-ID: `872ce8732d0e36a8f45da02f0d4fc560e25065b2250c0e703e689453ec5cdad1`

The reports do not record evaluator-source hashes. Their common bank path and
matching IDs/metadata establish alignment of the archived results, but the
bank file's content hash is unavailable, so byte-level identity is unverified.
The comparison uses the archived full-bank reports and is descriptive; it
does not isolate a causal effect of training with regime IDs.

Observed-ID reports and configs:

- `artifacts/cluster_sync/observed_id_training/observed-small/seed-2402/test.json`
- `artifacts/cluster_sync/observed_id_training/observed-medium/seed-2402/test.json`
- `artifacts/cluster_sync/observed_id_training/_provenance/observed-small-config.json`
- `artifacts/cluster_sync/observed_id_training/_provenance/observed-medium-config.json`

Original, fixed-mixture, and curriculum reports are under
`artifacts/cluster_sync/regime_information_checkpoint_compatible/`; their
configs are under `artifacts/cluster_sync/tfm_runs/native/`. The generated
summary checks each checkpoint's seed, step, size, and observed-ID exposure
setting, and verifies exact episode alignment across all reports.
