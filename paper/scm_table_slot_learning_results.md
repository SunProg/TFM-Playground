# Table-slot models on the TabICL SCM prior

This is the corrected SCM pilot using the table-aware models in
[`tfmplayground/models/table_slot.py`](../tfmplayground/models/table_slot.py).
The earlier `scm_slot_learning_results.md` measured `slot_regime.py` and
`slot_backbone.py`; it is not the table-slot result.

The prior is the constrained two-mechanism TabICL MLP-SCM family from
[`scm_regime_prior.py`](../tfmplayground/experiments/scm_regime_prior.py): two
nonlinear mechanisms, 128 support rows, 256 query rows, cue separation 3,
zero label noise and two balanced regimes. Regime IDs are hidden from the
latent models.

## Protocol

The models are constructed through the existing `build_model` path with
`table_slot_head` or `table_slot_backbone`. Both use `TableSlotModel` with the
default `cell_and_data` scope, so the table is processed by both feature-cell
and row-level competitions. They use two slots, three slot iterations, a
32-dimensional embedding, 4 attention heads, two transformer layers and MLP
width 64.

Each run trained from scratch for 600 updates on CPU, with batches of 8 fresh
episodes, 128 support rows and 32 training queries. AdamW used learning rate
0.001, weight decay 0.01 and gradient clipping 1. Checkpoints minimize
cross-entropy on 32 fixed validation episodes, evaluated at steps 0, 200, 400
and 600. There were two training seeds. Each final test score uses 128 separate
episodes, or 32,768 query predictions per run. The shuffled-label control
permutes each support label row on the same test episodes.

## Results

| Model | Seed | Selected step | Test accuracy | Test CE | Shuffled support labels |
|---|---:|---:|---:|---:|---:|
| `table_slot_head` | 11 | 600 | **92.01%** | 0.1913 | 50.17% |
| `table_slot_head` | 12 | 600 | 49.59% | 0.6933 | 49.48% |
| `table_slot_backbone` | 11 | 600 | 50.09% | 0.6931 | 50.09% |
| `table_slot_backbone` | 12 | 600 | 50.54% | 0.6932 | 50.54% |

`table_slot_head` learns the SCM prior strongly for seed 11 and uses the
support labels: shuffling them returns performance to chance. Seed 12 remains
at chance. `table_slot_backbone` remains at chance for both seeds in this
budget. The successful head run reaches the same range as the ordinary
nanoTabPFN baseline, but this is not evidence of an improvement because the
pilot has only two seeds and a shorter training budget.

The result points to a specific separation between the two table-slot paths:
the head placement can learn this positive-control prior, while the
in-backbone placement does not learn it under the current optimizer and
600-step budget. It does not establish that the backbone path is incapable of
learning; it identifies the next tuning target.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_slot_learning \
  --output results/scm_table_slot_learning \
  --steps 600 --seeds 11 12 \
  --models table_slot_head table_slot_backbone \
  --test-episodes 128 --validation-episodes 32

.venv/bin/python -m unittest tests.test_scm_slot_learning
```

Raw histories and checkpoints are in
`results/scm_table_slot_learning/`; the machine-readable summary is
`results/scm_table_slot_learning/results.json`.
