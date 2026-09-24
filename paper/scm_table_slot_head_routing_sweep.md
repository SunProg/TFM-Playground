# Table-slot head routing sweep on the SCM prior

This sweep uses `TableSlotModel(mode="head")` from
[`tfmplayground/models/table_slot.py`](../tfmplayground/models/table_slot.py).
It compares the four requested routing/compositing choices on the constrained
two-regime SCM prior. The sweep driver is
[`scm_table_slot_head_sweep.py`](../tfmplayground/experiments/scm_table_slot_head_sweep.py).

## Conditions

Every condition uses the same support reconstruction weight (1.0), the same
balanced-sharpness weight (0.05), the same architecture, episode stream and
training budget. The only changed settings are the query route and the
support reconstruction mixture.

| Condition | Query route | Support reconstruction mixture |
|---|---|---|
| `decoder_baseline` | learned decoder gate on the labelled-pass query embedding | slot attention `a[i,k]` |
| `decoder_alpha` | learned decoder gate on the labelled-pass query embedding | decoder alpha |
| `blind_decoder` | learned decoder gate on the label-blind query embedding | slot attention `a[i,k]` |
| `blind_similarity` | cosine similarity to blind support-slot centroids | slot attention `a[i,k]` |

The alpha setting must have a nonzero support reconstruction objective to have
an effect. The baseline and alpha cells therefore both train with the same
support reconstruction term; only its mixture weights differ. The similarity
route has no decoder alpha, so it remains paired with attention weighting.

## Protocol

The prior is the two-mechanism TabICL MLP-SCM family from
[`scm_regime_prior.py`](../tfmplayground/experiments/scm_regime_prior.py):
128 support rows, 256 test-query rows, two task features, cue separation 3,
zero label noise and balanced regimes. Regime IDs are hidden from the model.

Each run used two slots, three slot iterations, a 32-dimensional embedding,
four attention heads, two transformer layers and MLP width 64. AdamW used a
learning rate of 0.001, weight decay 0.01, gradient clipping at 1 and batches
of eight fresh episodes with 32 training queries. Each seed trained for 1,000
updates, matching the latent/oracle comparison protocol. The checkpoint is the
lowest validation cross-entropy among steps 0, 200, 400, 600, 800 and 1,000
on 32 fixed validation episodes. Final results use 128 test episodes (32,768
query predictions per run). The control shuffles each episode's support labels
before evaluation.

## Results

| Condition | Seed | Selected step | Test accuracy | Test CE | Shuffled-support accuracy |
|---|---:|---:|---:|---:|---:|
| `decoder_baseline` | 11 | 1,000 | **92.46%** | **0.1826** | 51.29% |
| `decoder_baseline` | 12 | 1,000 | **88.89%** | **0.2682** | 50.77% |
| `decoder_alpha` | 11 | 800 | 50.28% | 0.6931 | 50.10% |
| `decoder_alpha` | 12 | 800 | 50.31% | 0.6930 | 50.07% |
| `blind_decoder` | 11 | 200 | 49.97% | 0.6934 | 49.90% |
| `blind_decoder` | 12 | 1,000 | 49.89% | 0.6932 | 49.86% |
| `blind_similarity` | 11 | 1,000 | **72.73%** | **0.5315** | 51.56% |
| `blind_similarity` | 12 | 1,000 | **72.44%** | **0.5287** | 51.14% |

Across the two seeds, mean accuracy is 90.68% for `decoder_baseline`, 50.30%
for `decoder_alpha`, 49.93% for `blind_decoder` and 72.59% for
`blind_similarity`. The shuffled-label controls are at chance for every arm,
so the learned signal depends on the support labels rather than a fixed label
prior.

At the matched 1,000-step budget, the ordinary decoder baseline is the
strongest. Replacing the support reconstruction mixture with decoder alpha
still leaves both seeds at chance, so alpha is not a drop-in improvement for
this setup. The blind decoder also fails to learn, while blind similarity
recovers a stable but smaller signal and improves over its 600-step result.
These are two-seed pilot results; they identify the ranking and failure modes
for this prior rather than establish a general real-world ranking.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_table_slot_head_sweep \
  --output results/scm_table_slot_head_routing_sweep_1000 \
  --steps 1000 --seeds 11 12 \
  --test-episodes 128 --validation-episodes 32
```

The raw histories, selected checkpoints and machine-readable summary are in
`results/scm_table_slot_head_routing_sweep_1000/`.
