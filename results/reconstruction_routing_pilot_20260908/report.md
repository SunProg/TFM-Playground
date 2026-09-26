# Reconstruction routing learning pilots

Two matched seeds per arm; 32 shared held-out SCM episodes, 64 query rows each.
These are small CPU learning pilots, not the full four-prior Slurm study.

| Scope | Arm | Accuracy | Query NLL | Embedding MSE | Slot std | Mask row std | Query gate std |
|---|---|---:|---:|---:|---:|---:|---:|
| data | masks | 51.39% | 0.6929 | 0.0133 | 0.1255 | 0.001177 | 3.132e-07 |
| data | values | 51.56% | 0.6930 | 0.0171 | 0.1291 | 0.001413 | 1.336e-06 |
| cell_and_data | masks | 51.37% | 0.6928 | 0.0084 | 0.1358 | 0.001553 | 5.603e-07 |
| cell_and_data | values | 51.37% | 0.6928 | 0.0090 | 0.1298 | 0.001346 | 5.524e-06 |
| cell | masks | 52.05% | 0.6930 | 0.0783 | 0.3705 | 0.04085 | 0.0008444 |
| cell | values | 50.00% | 0.6934 | 0.1011 | 0.5299 | 0.04364 | 0.001731 |

## Paired comparison

NLL difference is values minus masks (negative favors values). Average seeds per episode first,
then bootstrap the 32 paired episode differences. Intervals cover episode sampling, not training-seed uncertainty.

- data: +0.00012, paired episode bootstrap 95% interval [-0.00008, +0.00031].
- cell_and_data: +0.00005, paired episode bootstrap 95% interval [-0.00030, +0.00039].
- cell: +0.00039, paired episode bootstrap 95% interval [+0.00015, +0.00065].

## Per-seed diagnostics

| Scope | Arm | Seed | NLL | Accuracy | NLL with errors disabled | NLL with reversed errors | NLL with shuffled labels | Final sampled query-key gradient norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| data | masks | 11 | 0.6930 | 51.37% | 0.6930 | 0.6930 | 0.6930 | 5.55e-08 |
| data | values | 11 | 0.6933 | 51.37% | 0.6933 | 0.6933 | 0.6933 | 4.56e-06 |
| cell_and_data | masks | 11 | 0.6927 | 51.37% | 0.6927 | 0.6927 | 0.6927 | 5.86e-08 |
| cell_and_data | values | 11 | 0.6927 | 51.37% | 0.6927 | 0.6927 | 0.6927 | 4.56e-05 |
| cell | masks | 11 | 0.6932 | 51.37% | 0.6932 | 0.6932 | 0.6931 | 4.53e-05 |
| cell | values | 11 | 0.6936 | 51.37% | 0.6936 | 0.6936 | 0.6936 | 0.0029 |
| data | masks | 12 | 0.6928 | 51.42% | 0.6928 | 0.6928 | 0.6928 | 5.18e-08 |
| data | values | 12 | 0.6928 | 51.76% | 0.6928 | 0.6928 | 0.6928 | 4.14e-05 |
| cell_and_data | masks | 12 | 0.6929 | 51.37% | 0.6929 | 0.6929 | 0.6929 | 8.5e-08 |
| cell_and_data | values | 12 | 0.6930 | 51.37% | 0.6930 | 0.6930 | 0.6930 | 4.89e-05 |
| cell | masks | 12 | 0.6928 | 52.73% | 0.6928 | 0.6928 | 0.6928 | 1.38e-05 |
| cell | values | 12 | 0.6931 | 48.63% | 0.6931 | 0.6931 | 0.6931 | 0.000478 |

## Plain TabPFN control

- Seed 11: NLL 0.6928, accuracy 51.37%.
- Seed 12: NLL 0.6928, accuracy 51.51%.

## Interpretation limits

- No conclusion about real tables, long pretraining, or the four-prior sweep follows from this pilot.
- Error disabling/reversal is an inference ablation, not separately trained weighting controls.
- Cell values combines direct retrieval with query-conditioned cell aggregation; its effect is not isolated.
- Cell alignment uses support row zero as reference, not the proposed medoid. It is order sensitive.
- Both arms share a trainable target encoder. Low reconstruction MSE alone does not prove useful preservation.
- Global embedding variance does not establish row diversity; inspect target_row_std in the raw results.
- Neither arm replaces the previously submitted Slurm array.
