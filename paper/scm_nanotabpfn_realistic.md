# Plain NanoTabPFN baseline on the realistic SCM profile

This is the non-slot reference for the realistic table-slot routing sweep in
[`scm_table_slot_head_routing_realistic.md`](scm_table_slot_head_routing_realistic.md).
The runner is
[`scm_nanotabpfn_baseline.py`](../tfmplayground/experiments/scm_nanotabpfn_baseline.py).

The baseline uses the same realistic SCM profile, architecture, optimizer,
episode stream, two seeds, 1,000-step budget, validation schedule and
shuffled-support control as the table-slot runs. It is a plain
`NanoTabPFNModel`, trained with query cross-entropy only; it has no slot
reconstruction or slot-MI auxiliary objectives.

## Results

| Seed | Selected step | Test accuracy | Test CE | Shuffled-support accuracy |
|---:|---:|---:|---:|---:|
| 11 | 1,000 | **55.97%** | **0.6724** | 50.05% |
| 12 | 200 | 50.41% | 0.6931 | 50.41% |

Mean accuracy is **53.69%**. The support-label shuffle returns both runs to
chance, so the successful seed is using the in-context support labels. On this
profile, the table-slot decoder baseline averages 73.35% and
blind-similarity averages 57.59%, while alpha and blind-decoder routing remain
near chance. The table-slot baseline therefore adds predictive value over this
plain reference in this two-seed experiment, although it also carries the
support-reconstruction and MI objectives that the plain model does not.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_nanotabpfn_baseline \
  --output results/scm_nanotabpfn_realistic_1000 \
  --prior-profile realistic \
  --steps 1000 --seeds 11 12 \
  --test-episodes 128 --validation-episodes 32
```

Raw histories, checkpoints and the machine-readable summary are in
`results/scm_nanotabpfn_realistic_1000/`.
