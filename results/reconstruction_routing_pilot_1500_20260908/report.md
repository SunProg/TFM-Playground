# Reconstruction routing learning pilots

Two matched seeds per arm; 32 shared held-out SCM episodes, 64 query rows each.
These are small CPU learning pilots, not the full four-prior Slurm study.

| Scope | Arm | Accuracy | Query NLL | Embedding MSE | Slot std | Mask row std | Query gate std |
|---|---|---:|---:|---:|---:|---:|---:|
| data | masks | 50.00% | 0.6932 | 0.0036 | 0.0798 | 0.001208 | 8.988e-08 |
| data | values | 56.10% | 0.6727 | 0.4356 | 0.1596 | 0.001537 | 0.0001371 |
| cell_and_data | masks | 50.00% | 0.6934 | 0.0036 | 0.0594 | 0.0005163 | 1.138e-07 |
| cell_and_data | values | 58.86% | 0.6458 | 0.2121 | 0.0696 | 0.0005379 | 7.149e-05 |
| cell | masks | 50.00% | 0.6929 | 0.0464 | 0.3917 | 0.06811 | 0.005292 |
| cell | values | 50.00% | 0.6933 | 0.0015 | 0.6007 | 0.04375 | 0.001846 |

## Paired comparison

NLL difference is values minus masks (negative favors values). Average seeds per episode first,
then bootstrap the 32 paired episode differences. Intervals cover episode sampling, not training-seed uncertainty.

- data: -0.02050, paired episode bootstrap 95% interval [-0.02997, -0.01167].
- cell_and_data: -0.04760, paired episode bootstrap 95% interval [-0.07207, -0.02214].
- cell: +0.00035, paired episode bootstrap 95% interval [+0.00017, +0.00056].

## Per-seed diagnostics

| Scope | Arm | Seed | NLL | Accuracy | NLL with errors disabled | NLL with reversed errors | NLL with shuffled labels | Final sampled query-key gradient norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| data | masks | 11 | 0.6936 | 48.63% | 0.6936 | 0.6936 | 0.6936 | 2.57e-07 |
| data | values | 11 | 0.6526 | 60.84% | 0.6526 | 0.6525 | 0.7476 | 0.013 |
| cell_and_data | masks | 11 | 0.6940 | 48.63% | 0.6940 | 0.6940 | 0.6940 | 1.3e-08 |
| cell_and_data | values | 11 | 0.6940 | 48.63% | 0.6940 | 0.6940 | 0.6940 | 7.54e-06 |
| cell | masks | 11 | 0.6937 | 48.63% | 0.6937 | 0.6937 | 0.6937 | 4.25e-05 |
| cell | values | 11 | 0.6938 | 48.63% | 0.6938 | 0.6938 | 0.6938 | 2.78e-05 |
| data | masks | 12 | 0.6928 | 51.37% | 0.6928 | 0.6928 | 0.6928 | 1.03e-08 |
| data | values | 12 | 0.6928 | 51.37% | 0.6928 | 0.6928 | 0.6928 | 7.9e-07 |
| cell_and_data | masks | 12 | 0.6928 | 51.37% | 0.6928 | 0.6928 | 0.6928 | 3.18e-08 |
| cell_and_data | values | 12 | 0.5976 | 69.09% | 0.5974 | 0.5974 | 0.7996 | 0.0794 |
| cell | masks | 12 | 0.6922 | 51.37% | 0.6922 | 0.6922 | 0.6924 | 6.07e-05 |
| cell | values | 12 | 0.6928 | 51.37% | 0.6928 | 0.6928 | 0.6928 | 0.000407 |

## Plain TabPFN control

- Seed 11: NLL 0.6938, accuracy 48.63%.
- Seed 12: NLL 0.6620, accuracy 57.47%.

## Retrieved-context removal at inference

| Scope | Seed | Full NLL | Context removed NLL | Full accuracy | Context removed accuracy |
|---|---:|---:|---:|---:|---:|
| data | 11 | 0.6526 | 0.6608 | 60.84% | 60.11% |
| data | 12 | 0.6928 | 0.6928 | 51.37% | 51.37% |
| cell_and_data | 11 | 0.6940 | 0.6939 | 48.63% | 48.63% |
| cell_and_data | 12 | 0.5976 | 0.6111 | 69.09% | 67.63% |
| cell | 11 | 0.6938 | 0.6938 | 48.63% | 48.63% |
| cell | 12 | 0.6928 | 0.6928 | 51.37% | 51.37% |

Most gains survive removal of the direct context at inference. This ablation does not erase
changes learned by the shared backbone and slots during training, and does not isolate optimization effects.

## Interpretation limits

- No conclusion about real tables, long pretraining, or the four-prior sweep follows from this pilot.
- Error disabling/reversal is an inference ablation, not separately trained weighting controls.
- Cell values combines direct retrieval with query-conditioned cell aggregation; its effect is not isolated.
- Cell alignment uses support row zero as reference, not the proposed medoid. It is order sensitive.
- Both arms share a trainable target encoder. Low reconstruction MSE alone does not prove useful preservation.
- Global embedding variance does not establish row diversity; inspect target_row_std in the raw results.
- Neither arm replaces the previously submitted Slurm array.
