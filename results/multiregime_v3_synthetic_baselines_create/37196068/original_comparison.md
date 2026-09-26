# CREATE original-family baseline verification

- Baseline JSON SHA256: `df181a08e5274e60692874403832b341794a675d0bf09643cb1087134538a76a`
- Pilot episode hashes: exact match (8/8)
- Model/episode rows: 64/64; model errors: 0

| model | log_loss | accuracy | pooled AUC | mean episode AUC |
|---|---:|---:|---:|---:|
| OURS: plain/original | 0.5149 | 0.770 | 0.8038 | — |
| logreg | 0.5575 | 0.750 | 0.7902 | 0.5073 |
| random_forest | 0.6140 | 0.711 | 0.7742 | 0.4657 |
| tabpfn-v2.2 | 0.5101 | 0.766 | 0.8175 | 0.5582 |
| tabpfn-v2.2-finetuned | 0.5089 | 0.766 | 0.8178 | 0.5557 |
| tabpfn-v2.6 | 0.5262 | 0.770 | 0.8131 | 0.5523 |
| tabpfn-v2.6-finetuned | 0.5256 | 0.770 | 0.8132 | 0.5500 |
| tabpfn-v3 | 0.5100 | 0.762 | 0.8178 | 0.5543 |
| tabpfn-v3-finetuned | 0.5300 | 0.750 | 0.8122 | 0.5329 |

AUC is pooled over all 256 query rows per model; mean episode AUC is retained as a diagnostic.
This is a comparison artifact only; it does not rewrite the paper table.
