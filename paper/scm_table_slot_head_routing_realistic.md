# Table-slot head routing on a more realistic SCM profile

The controlled result remains in
[`scm_table_slot_head_routing_sweep.md`](scm_table_slot_head_routing_sweep.md).
This report keeps the same four `TableSlotModel(mode="head")` conditions and
the same 1,000-step protocol, but makes the synthetic data harder and closer
to a messy tabular setting. The sweep driver is
[`scm_table_slot_head_sweep.py`](../tfmplayground/experiments/scm_table_slot_head_sweep.py).

## Realistic profile

The profile changes the data difficulty while keeping the 128/256
support/query episode size fixed:

| Setting | Value |
|---|---:|
| Observed task covariates | 4 plus one regime cue |
| Latent regimes | 2 |
| Regime mixture | 65% / 35% |
| Cue separation | 1.5 |
| Mechanism depth | 2 nonlinear layers |
| Mechanism hidden width | 12 |
| Label noise | 5% |
| Calibration disagreement range | 20%–80% |

The cue is deliberately less separable than in the positive-control run, the
regimes are imbalanced, and labels contain measurement noise. This is still a
synthetic SCM stress test; it is not evidence from a real clinical, financial
or industrial table.

## Protocol

All four conditions use support reconstruction weight 1.0 and slot MI weight
0.05. Each seed trains for 1,000 updates with the same two-slot,
cell-and-data table-slot head, 32-dimensional backbone, four attention heads,
two transformer layers and MLP width 64. Checkpoints are selected by minimum
validation cross-entropy at steps 0, 200, 400, 600, 800 or 1,000. Final
metrics use 128 test episodes; the control shuffles each episode's support
labels before evaluation.

## Results

| Condition | Seed | Selected step | Test accuracy | Test CE | Shuffled-support accuracy |
|---|---:|---:|---:|---:|---:|
| `decoder_baseline` | 11 | 1,000 | **75.06%** | **0.5149** | 50.18% |
| `decoder_baseline` | 12 | 1,000 | **71.65%** | **0.5542** | 49.51% |
| `decoder_alpha` | 11 | 200 | 49.33% | 0.6933 | 49.34% |
| `decoder_alpha` | 12 | 200 | 50.10% | 0.6932 | 50.09% |
| `blind_decoder` | 11 | 400 | 50.43% | 0.6934 | 50.43% |
| `blind_decoder` | 12 | 400 | 50.20% | 0.6932 | 50.20% |
| `blind_similarity` | 11 | 800 | **57.49%** | **0.6718** | 50.66% |
| `blind_similarity` | 12 | 1,000 | **57.68%** | **0.6708** | 50.46% |

Mean accuracy is 73.35% for `decoder_baseline`, 49.71% for `decoder_alpha`,
50.31% for `blind_decoder` and 57.59% for `blind_similarity`. The support
shuffle returns accuracy to roughly 50% for every condition, so the signal is
using support labels.

The matched plain NanoTabPFN reference averages 53.69% (55.97% and 50.41%
across the two seeds). Its full breakdown is in
[`scm_nanotabpfn_realistic.md`](scm_nanotabpfn_realistic.md). The table-slot
baseline therefore improves over the plain reference in this pilot, while the
comparison should be read together with the fact that the table-slot run also
uses support reconstruction and slot-MI objectives.

The baseline remains the strongest route under this harder profile, but its
accuracy drops from 90.68% on the balanced, noiseless control to 73.35% here.
Blind similarity still learns a smaller signal, while alpha compositing and
the blind decoder remain at chance. The result is a more realistic stress test
of the routing mechanisms, not a claim that these rankings transfer unchanged
to real data.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_table_slot_head_sweep \
  --output results/scm_table_slot_head_routing_realistic_1000_v2 \
  --prior-profile realistic \
  --steps 1000 --seeds 11 12 \
  --test-episodes 128 --validation-episodes 32
```

Raw histories, checkpoints and the machine-readable summary are in
`results/scm_table_slot_head_routing_realistic_1000_v2/`.
