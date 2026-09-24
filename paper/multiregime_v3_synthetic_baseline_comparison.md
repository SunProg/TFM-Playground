# v3 pilot vs. sklearn vs. TabPFN, on a TabICL 2.1.1 synthetic eval bank

Date: 2026-09-14.

This is the corrected rerun requested after pinning `tabicl==2.1.1`. It uses
CREATE job `37203190` and evaluates every model on the same newly generated
bank. This is intentionally a **new bank**: the earlier pilot bank was
generated under TabICL 2.2.0, and TabICL 2.1.1 does not reproduce its episode
hashes. The old 2.2.0 comparison is therefore not mixed with these results.

## Evaluation design

- Five families: `original`, `shared_rule`, `soft_gate`, `independent`, and
  `persistent`.
- Eight independent episodes per family.
- Each episode has 128 support rows and 32 query rows.
- Pilot: all 21 completed cells in `runs/37113090` (seven model variants ×
  three modes: `original`, `fixed`, `curriculum`).
- Baselines: logistic regression, random forest, TabPFN v2.2, v2.6, and v3,
  each both in-context and per-episode finetuned.
- Primary cells below are `log_loss/accuracy`, aggregated over the 256 query
  rows in each family.

The query rows are not 256 independent task replicates: they are clustered in
eight SCM/task episodes. Use episode-level resampling for uncertainty.

## Results

| model | original | shared_rule | soft_gate | independent | persistent |
|---|---|---|---|---|---|
| OURS: plain/original | 0.3535/0.844 | 0.6458/0.617 | 0.6456/0.590 | 0.6379/0.645 | 0.6690/0.613 |
| OURS: plain/fixed | 0.3527/0.848 | 0.5488/0.746 | 0.6018/0.664 | 0.5691/0.719 | 0.5682/0.699 |
| OURS: plain/curriculum | 0.3496/0.859 | 0.5546/0.723 | 0.6094/0.648 | 0.5665/0.727 | 0.5659/0.707 |
| OURS: table_slot_head_decoder_baseline/original | 0.3490/0.840 | 0.5705/0.703 | 0.6087/0.637 | 0.5548/0.723 | 0.5888/0.668 |
| OURS: table_slot_head_decoder_baseline/fixed | 0.3489/0.848 | 0.5570/0.730 | 0.5924/0.676 | 0.5566/0.719 | 0.5797/0.691 |
| OURS: table_slot_head_decoder_baseline/curriculum | 0.3494/0.840 | 0.5575/0.719 | 0.5980/0.664 | 0.5569/0.703 | 0.5731/0.680 |
| OURS: table_slot_head_decoder_alpha/original | 0.3490/0.840 | 0.5705/0.703 | 0.6087/0.637 | 0.5548/0.723 | 0.5888/0.668 |
| OURS: table_slot_head_decoder_alpha/fixed | 0.3489/0.848 | 0.5570/0.730 | 0.5924/0.676 | 0.5566/0.719 | 0.5797/0.691 |
| OURS: table_slot_head_decoder_alpha/curriculum | 0.3494/0.840 | 0.5575/0.719 | 0.5980/0.664 | 0.5569/0.703 | 0.5731/0.680 |
| OURS: table_slot_head_blind_decoder/original | 0.3468/0.840 | 0.5711/0.707 | 0.6129/0.641 | 0.5512/0.715 | 0.5884/0.664 |
| OURS: table_slot_head_blind_decoder/fixed | 0.3472/0.852 | 0.5574/0.711 | 0.5951/0.676 | 0.5589/0.707 | 0.5746/0.691 |
| OURS: table_slot_head_blind_decoder/curriculum | 0.3503/0.836 | 0.5581/0.723 | 0.5948/0.664 | 0.5541/0.719 | 0.5707/0.703 |
| OURS: table_slot_head_blind_similarity/original | 0.3486/0.836 | 0.5724/0.699 | 0.6106/0.629 | 0.5548/0.723 | 0.5892/0.648 |
| OURS: table_slot_head_blind_similarity/fixed | 0.3485/0.852 | 0.5604/0.715 | 0.5934/0.672 | 0.5619/0.715 | 0.5722/0.699 |
| OURS: table_slot_head_blind_similarity/curriculum | 0.3498/0.844 | 0.5602/0.727 | 0.5952/0.672 | 0.5539/0.707 | 0.5707/0.684 |
| OURS: table_slot_backbone/original | 0.3475/0.836 | 0.5660/0.715 | 0.6203/0.633 | 0.5597/0.711 | 0.5944/0.672 |
| OURS: table_slot_backbone/fixed | 0.3496/0.840 | 0.5566/0.730 | 0.5960/0.664 | 0.5612/0.719 | 0.5822/0.680 |
| OURS: table_slot_backbone/curriculum | 0.3497/0.852 | 0.5546/0.727 | 0.6012/0.664 | 0.5618/0.727 | 0.5802/0.648 |
| OURS: table_slot_mufasa/original | 0.3493/0.844 | 0.5742/0.711 | 0.6168/0.629 | 0.5462/0.723 | 0.5864/0.695 |
| OURS: table_slot_mufasa/fixed | 0.3460/0.859 | 0.5539/0.730 | 0.6000/0.672 | 0.5614/0.719 | 0.5765/0.688 |
| OURS: table_slot_mufasa/curriculum | 0.3504/0.852 | 0.5594/0.715 | 0.5998/0.664 | 0.5636/0.715 | 0.5703/0.703 |
| logreg | 0.4057/0.848 | 0.5767/0.742 | 0.6600/0.629 | 0.5921/0.715 | 0.5910/0.672 |
| random_forest | 0.6041/0.840 | 0.6083/0.664 | 0.7198/0.648 | 0.5500/0.730 | 0.6001/0.680 |
| tabpfn-v2.2 | 0.3506/0.852 | 0.5642/0.707 | 0.5995/0.652 | 0.5606/0.727 | 0.5566/0.715 |
| tabpfn-v2.2-finetuned | 0.3533/0.852 | 0.5647/0.707 | 0.6001/0.656 | 0.5612/0.723 | 0.5551/0.723 |
| tabpfn-v2.6 | 0.3563/0.840 | 0.5825/0.734 | 0.6053/0.648 | 0.5508/0.715 | 0.5513/0.699 |
| tabpfn-v2.6-finetuned | 0.3577/0.852 | 0.5825/0.734 | 0.6054/0.645 | 0.5512/0.719 | 0.5523/0.703 |
| tabpfn-v3 | 0.3588/0.844 | 0.5741/0.707 | 0.5976/0.652 | 0.5595/0.734 | 0.5485/0.750 |
| tabpfn-v3-finetuned | 0.3857/0.840 | 0.5773/0.699 | 0.6023/0.664 | 0.5793/0.707 | 0.5648/0.734 |

## Interpretation

Best log loss by family in this rerun:

- `original`: OURS `table_slot_mufasa/fixed`, 0.3460; TabPFN v2.2 is close
  at 0.3506.
- `shared_rule`: OURS `plain/fixed`, 0.5488.
- `soft_gate`: OURS `table_slot_head_decoder_baseline/fixed` and
  `table_slot_head_decoder_alpha/fixed`, 0.5924.
- `independent`: OURS `table_slot_mufasa/original`, 0.5462; random forest is
  0.5500.
- `persistent`: TabPFN v3, 0.5485.

These are small-bank point estimates, not definitive rankings. In particular,
the eight episodes per family are the independent units, not the 256 query
rows.

## AUC and the pooled-versus-episode distinction

Both AUC estimands are now saved for every pilot cell and baseline family:

- `pooled_auc`: concatenate all query predictions in a family, then compute
  one AUC;
- `mean_episode_auc`: compute AUC separately within each episode, then average
  the eight values.

Because episodes have different SCMs, feature distributions, and class
balances, pooled AUC also compares examples across different tasks. It is not
mathematically invalid, but it is not the clean episodic estimand and can be
misleading. Use `mean_episode_auc` for the primary within-task diagnostic, and
report pooled AUC only as a secondary descriptive statistic. With 32 query
rows per episode, both remain noisy.

## Baseline repair and environment

The final baseline artifact contains 320/320 model-family/episode rows and no
errors. Three `original`, episode-4 finetuned TabPFN rows initially failed
because a support class had only one example, making a stratified internal
validation split undefined. Those rows were recomputed with
`validation_split_ratio=0.0`; this changes only the evaluation-time
finetuning validation setting, not the model checkpoint or the bank.

The recorded baseline environment is: NumPy 2.5.3, SciPy 1.18.1, PyTorch
2.9.0, scikit-learn 1.6.1, TabICL 2.1.1, and TabPFN 8.5.0.

Artifacts:

- [`evaluation_bank_manifest.json`](../results/multiregime_v3_tabicl211/37203190/evaluation_bank_manifest.json)
- [`pilot_metrics_full.json`](../results/multiregime_v3_tabicl211/37203190/pilot_metrics_full.json)
- [`baseline.json`](../results/multiregime_v3_tabicl211/37203190/baseline.json)
- [`baseline.environment.json`](../results/multiregime_v3_tabicl211/37203190/baseline.environment.json)
