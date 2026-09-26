# Before/after routing comparison

Same synthetic SCM prior, episode-seed schedule, 1,500 updates, two training seeds (11/12),
backbone width 24 and two layers, four slots, AdamW at 0.001, batch size 2.
Held-out evaluation: identical 32 tasks with 64 query rows each.

These are retrained small-model controls, not evaluations of the full-size Slurm checkpoints.

## Mean accuracy

| Model | data | cell_and_data | cell |
|---|---:|---:|---:|
| Original query-only | 66.80% | 66.72% | 72.19% |
| Original label-alpha | 72.12% | 65.97% | 73.17% |
| Embedding MSE, old gate | 58.89% | 59.67% | Not supported |
| New mask routing | 50.00% | 50.00% | 50.00% |
| New support retrieval | 56.10% | 58.86% | 50.00% |

## Mean query NLL

| Model | data | cell_and_data | cell |
|---|---:|---:|---:|
| Original query-only | 0.6169 | 0.6258 | 0.5599 |
| Original label-alpha | 0.5680 | 0.6120 | 0.5494 |
| Embedding MSE, old gate | 0.6551 | 0.6515 | Not supported |
| New mask routing | 0.6932 | 0.6934 | 0.6929 |
| New support retrieval | 0.6727 | 0.6458 | 0.6933 |

## Objectives and comparison limits

- Original query-only: existing TableSlotModel and decoder-alpha query gate, pure query NLL.
- Original label-alpha: same model, query NLL + alpha-composited support-label NLL + 0.05 slot MI.
- Embedding MSE, old gate: the initial modification, query NLL + embedding MSE, original query gate.
- New routing arms: query NLL + embedding MSE, with reconstruction-weighted support routing.
- Previous models are called directly through wrappers; no replacement slot encoder is substituted.
- Same seed does not guarantee identical head initialization when module construction order differs.
- Cell changes also include reconstruction, alignment, and (for retrieval) query-dependent cell weighting.
- All heads/backbones train from scratch. Two seeds are insufficient to establish robust rankings.
- This does not replace the missing matched retrieval-without-slots ablation.

## Per-seed results

| Model | Scope | Seed | Accuracy | Query NLL | Shuffled-label NLL |
|---|---|---:|---:|---:|---:|
| Original query-only | data | 11 | 66.26% | 0.6188 | 0.7954 |
| Original query-only | cell_and_data | 11 | 65.97% | 0.6335 | 0.7629 |
| Original query-only | cell | 11 | 73.97% | 0.5419 | 0.8116 |
| Original query-only | data | 12 | 67.33% | 0.6151 | 0.7693 |
| Original query-only | cell_and_data | 12 | 67.48% | 0.6181 | 0.8069 |
| Original query-only | cell | 12 | 70.41% | 0.5779 | 0.7903 |
| Original label-alpha | data | 11 | 71.00% | 0.5759 | 0.7690 |
| Original label-alpha | cell_and_data | 11 | 71.97% | 0.5655 | 0.8099 |
| Original label-alpha | cell | 11 | 72.51% | 0.5608 | 0.7834 |
| Original label-alpha | data | 12 | 73.24% | 0.5602 | 0.7777 |
| Original label-alpha | cell_and_data | 12 | 59.96% | 0.6584 | 0.7258 |
| Original label-alpha | cell | 12 | 73.83% | 0.5380 | 0.7737 |
| Embedding MSE, old gate | data | 11 | 66.41% | 0.6173 | 0.7635 |
| Embedding MSE, old gate | cell_and_data | 11 | 67.97% | 0.6103 | 0.7796 |
| Embedding MSE, old gate | data | 12 | 51.37% | 0.6928 | 0.6928 |
| Embedding MSE, old gate | cell_and_data | 12 | 51.37% | 0.6928 | 0.6928 |
| New mask routing | data | 11 | 48.63% | 0.6936 | 0.6936 |
| New mask routing | cell_and_data | 11 | 48.63% | 0.6940 | 0.6940 |
| New mask routing | cell | 11 | 48.63% | 0.6937 | 0.6937 |
| New mask routing | data | 12 | 51.37% | 0.6928 | 0.6928 |
| New mask routing | cell_and_data | 12 | 51.37% | 0.6928 | 0.6928 |
| New mask routing | cell | 12 | 51.37% | 0.6922 | 0.6924 |
| New support retrieval | data | 11 | 60.84% | 0.6526 | 0.7476 |
| New support retrieval | cell_and_data | 11 | 48.63% | 0.6940 | 0.6940 |
| New support retrieval | cell | 11 | 48.63% | 0.6938 | 0.6938 |
| New support retrieval | data | 12 | 51.37% | 0.6928 | 0.6928 |
| New support retrieval | cell_and_data | 12 | 69.09% | 0.5976 | 0.7996 |
| New support retrieval | cell | 12 | 51.37% | 0.6928 | 0.6928 |
