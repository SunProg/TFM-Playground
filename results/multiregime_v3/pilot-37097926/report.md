# Multi-regime v3 pilot

Completed cells: 6/6.
Single training seed; intervals resample evaluation episodes, not training seeds.

| Cell | Model | Prior | Original NLL | Shared rule | Soft gate | Independent | Persistent |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | plain | original | 0.4748 | 0.6667 | 0.6705 | 0.6888 | 0.6747 |
| 1 | slot | original | 0.4761 | 0.6655 | 0.6704 | 0.6889 | 0.6737 |
| 2 | plain | fixed | 0.4764 | 0.6652 | 0.6698 | 0.6875 | 0.6735 |
| 3 | slot | fixed | 0.4783 | 0.6644 | 0.6697 | 0.6873 | 0.6730 |
| 4 | plain | curriculum | 0.4768 | 0.6646 | 0.6697 | 0.6873 | 0.6733 |
| 5 | slot | curriculum | 0.4781 | 0.6641 | 0.6697 | 0.6875 | 0.6727 |

## Paired comparisons

Negative delta favors the right cell.

| Left → right | Family | Delta NLL | Episode bootstrap 95% interval |
|---|---|---:|---|
| 0 → 1 | original | +0.0013 | [+0.0005, +0.0023] |
| 0 → 1 | shared_rule | -0.0012 | [-0.0021, -0.0002] |
| 0 → 1 | soft_gate | -0.0001 | [-0.0011, +0.0010] |
| 0 → 1 | independent | +0.0001 | [-0.0011, +0.0013] |
| 0 → 1 | persistent | -0.0011 | [-0.0023, -0.0000] |
| 2 → 3 | original | +0.0019 | [+0.0009, +0.0030] |
| 2 → 3 | shared_rule | -0.0008 | [-0.0019, +0.0001] |
| 2 → 3 | soft_gate | -0.0001 | [-0.0009, +0.0008] |
| 2 → 3 | independent | -0.0002 | [-0.0007, +0.0003] |
| 2 → 3 | persistent | -0.0006 | [-0.0016, +0.0001] |
| 4 → 5 | original | +0.0013 | [+0.0006, +0.0022] |
| 4 → 5 | shared_rule | -0.0005 | [-0.0012, +0.0002] |
| 4 → 5 | soft_gate | -0.0001 | [-0.0008, +0.0005] |
| 4 → 5 | independent | +0.0001 | [-0.0005, +0.0008] |
| 4 → 5 | persistent | -0.0006 | [-0.0013, -0.0001] |
| 0 → 2 | original | +0.0016 | [-0.0005, +0.0038] |
| 0 → 2 | shared_rule | -0.0015 | [-0.0036, +0.0002] |
| 0 → 2 | soft_gate | -0.0007 | [-0.0027, +0.0011] |
| 0 → 2 | independent | -0.0013 | [-0.0034, +0.0008] |
| 0 → 2 | persistent | -0.0012 | [-0.0034, +0.0007] |
| 0 → 4 | original | +0.0020 | [+0.0001, +0.0042] |
| 0 → 4 | shared_rule | -0.0021 | [-0.0043, -0.0003] |
| 0 → 4 | soft_gate | -0.0007 | [-0.0027, +0.0009] |
| 0 → 4 | independent | -0.0015 | [-0.0036, +0.0004] |
| 0 → 4 | persistent | -0.0014 | [-0.0039, +0.0005] |
| 1 → 3 | original | +0.0021 | [+0.0002, +0.0042] |
| 1 → 3 | shared_rule | -0.0011 | [-0.0034, +0.0010] |
| 1 → 3 | soft_gate | -0.0007 | [-0.0030, +0.0016] |
| 1 → 3 | independent | -0.0015 | [-0.0040, +0.0009] |
| 1 → 3 | persistent | -0.0007 | [-0.0030, +0.0013] |
| 1 → 5 | original | +0.0020 | [+0.0002, +0.0039] |
| 1 → 5 | shared_rule | -0.0015 | [-0.0036, +0.0004] |
| 1 → 5 | soft_gate | -0.0007 | [-0.0029, +0.0012] |
| 1 → 5 | independent | -0.0014 | [-0.0034, +0.0006] |
| 1 → 5 | persistent | -0.0010 | [-0.0031, +0.0009] |

## Matching checks

- pair_0_1_source_hash: True
- pair_0_1_initial_backbone_hash: True
- pair_0_1_evaluation_bank_hashes: True
- pair_0_1_training_stream_hash: True
- pair_2_3_source_hash: True
- pair_2_3_initial_backbone_hash: True
- pair_2_3_evaluation_bank_hashes: True
- pair_2_3_training_stream_hash: True
- pair_4_5_source_hash: True
- pair_4_5_initial_backbone_hash: True
- pair_4_5_evaluation_bank_hashes: True
- pair_4_5_training_stream_hash: True
- all_cells_evaluation_bank: True

Synthetic pilot results do not establish real-world transfer. Full metrics, grouping controls, and slot-recovery references are in each cell's result.json and evaluation.jsonl.
