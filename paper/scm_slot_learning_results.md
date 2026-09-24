# Slot-attention models on the TabICL SCM prior

This is a short positive-control pilot, not a full hyperparameter study. The
prior is the constrained two-mechanism TabICL MLP-SCM family described in
[`scm_prior_learning_results.md`](scm_prior_learning_results.md). It uses two
nonlinear target mechanisms, 128 support rows, 256 query rows, cue separation
3, zero label noise and 50/50 regimes.

## Protocol

I trained the repository's `slot_head` and `slot_tabpfn` models from scratch on
CPU. Both use two slots, three slot iterations, a 32-dimensional embedding, 4
attention heads, two transformer layers and MLP width 64. `slot_head` applies
slot competition after the backbone representation; `slot_tabpfn` installs the
competition inside the backbone and decodes a per-query slot mixture.

Each arm ran 600 updates with batches of 8 fresh episodes, 128 support rows and
32 training queries. AdamW used learning rate 0.001, weight decay 0.01 and
gradient clipping 1. Checkpoints minimize cross-entropy on 32 fixed validation
episodes, evaluated at steps 0, 200, 400 and 600. There were two training seeds
(11 and 12). Each final score uses 128 separate test episodes, or 32,768 query
predictions per run. Shuffled-support-label scores reuse the same test episodes
after permuting each support label row.

## Results

| Model | Seed | Selected step | Test accuracy | Test CE | Shuffled support labels |
|---|---:|---:|---:|---:|---:|
| `slot_head` | 11 | 600 | **61.41%** | 0.6301 | 49.08% |
| `slot_head` | 12 | 600 | 49.55% | 0.6932 | 49.53% |
| `slot_tabpfn` | 11 | 600 | **63.11%** | 0.6433 | 49.49% |
| `slot_tabpfn` | 12 | 600 | 50.31% | 0.6931 | 50.31% |

Both architectures learned above chance for seed 11, and both seed-11 models
fell back to chance when support labels were shuffled. Seed 12 failed to learn
within the 600-step budget for either architecture. The pilot therefore shows
that the SCM prior supplies a usable training signal to the slot models, but
optimization is currently unstable and substantially below the 91.67% mean
obtained by the ordinary nanoTabPFN baseline after 1,000 updates.

The two slot models should not be ranked from these four runs. The experiment
uses a small budget, two seeds, no learning-rate sweep and no auxiliary regime
loss. The slot models' objective is the repository's slot mixture negative
log-likelihood; regime IDs remain hidden from the loss. The comparison is a
diagnostic of learnability, not evidence that the slot mechanism improves
prediction.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_slot_learning \
  --output results/scm_slot_learning \
  --steps 600 --seeds 11 12 \
  --models slot_head slot_tabpfn \
  --test-episodes 128 --validation-episodes 32

.venv/bin/python -m unittest tests.test_scm_slot_learning
```

Raw histories and checkpoints are in
`results/scm_slot_learning/`; the machine-readable summary is
`results/scm_slot_learning/results.json`.
