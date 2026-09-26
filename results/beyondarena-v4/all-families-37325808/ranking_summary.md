# Multiregime-v4 NanoTabPFN on BeyondArena

Protocol: official Data Foundry containers and every official outer fold; complete training folds are used as context.

- Eligible tasks: classes 2–5, rows ≤ 10,000, raw features ≤ 30, classification only.
- Excluded: text and high-cardinality categorical tasks; target and grouped identifiers are removed from model inputs.
- Model selection: `final_checkpoint.pth` and lowest stored ordinary-prior validation CE (`best_own_val`).
- Aggregation: folds are averaged within task; models are ranked within each task, then task ranks are averaged equally.
- Metrics: excess CE (lower), macro OVR AUC (higher), and accuracy gain (higher).

Eligible tasks: 32 / manifest rows: 142
Selections: 47 (47 available)

## Best available selection by condition and metric

- `Grouped` / `accuracy_gain`: `tabpfn-v2.2` (published_default), score `0.100507`, tasks `3`, unsupported coverage `0.0%`
- `Grouped` / `excess_cross_entropy`: `tabpfn-v2.2` (published_default), score `-0.21729`, tasks `3`, unsupported coverage `0.0%`
- `Grouped` / `macro_ovr_auc`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (final), score `0.872652`, tasks `3`, unsupported coverage `0.0%`
- `IID` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.154061`, tasks `29`, unsupported coverage `0.0%`
- `IID` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.299401`, tasks `29`, unsupported coverage `0.0%`
- `IID` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.884576`, tasks `29`, unsupported coverage `0.0%`
- `all` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.149024`, tasks `32`, unsupported coverage `0.0%`
- `all` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.290235`, tasks `32`, unsupported coverage `0.0%`
- `all` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.881266`, tasks `32`, unsupported coverage `0.0%`
- `binary` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.112372`, tasks `24`, unsupported coverage `0.0%`
- `binary` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.201928`, tasks `24`, unsupported coverage `0.0%`
- `binary` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.863548`, tasks `24`, unsupported coverage `0.0%`
- `binary__Grouped` / `accuracy_gain`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/g_z-fixed-medium/seed-2402` (best_own_val), score `0.0708967`, tasks `1`, unsupported coverage `0.0%`
- `binary__Grouped` / `excess_cross_entropy`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-fixed-medium/seed-2402` (final), score `-0.113881`, tasks `1`, unsupported coverage `0.0%`
- `binary__Grouped` / `macro_ovr_auc`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (final), score `0.849312`, tasks `1`, unsupported coverage `0.0%`
- `binary__IID` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.115797`, tasks `23`, unsupported coverage `0.0%`
- `binary__IID` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.21574`, tasks `23`, unsupported coverage `0.0%`
- `binary__IID` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.869058`, tasks `23`, unsupported coverage `0.0%`
- `large_feat` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.137565`, tasks `5`, unsupported coverage `0.0%`
- `large_feat` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.237666`, tasks `5`, unsupported coverage `0.0%`
- `large_feat` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.924701`, tasks `5`, unsupported coverage `0.0%`
- `large_rows` / `accuracy_gain`: `tabpfn-v3` (published_default), score `0.130296`, tasks `2`, unsupported coverage `0.0%`
- `large_rows` / `excess_cross_entropy`: `tabpfn-v3` (published_default), score `-0.323998`, tasks `2`, unsupported coverage `0.0%`
- `large_rows` / `macro_ovr_auc`: `tabpfn-v3` (published_default), score `0.937048`, tasks `2`, unsupported coverage `0.0%`
- `large_rows__IID` / `accuracy_gain`: `tabpfn-v3` (published_default), score `0.130296`, tasks `2`, unsupported coverage `0.0%`
- `large_rows__IID` / `excess_cross_entropy`: `tabpfn-v3` (published_default), score `-0.323998`, tasks `2`, unsupported coverage `0.0%`
- `large_rows__IID` / `macro_ovr_auc`: `tabpfn-v3` (published_default), score `0.937048`, tasks `2`, unsupported coverage `0.0%`
- `medium_feat` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.124586`, tasks `15`, unsupported coverage `0.0%`
- `medium_feat` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.246601`, tasks `15`, unsupported coverage `0.0%`
- `medium_feat` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.881171`, tasks `15`, unsupported coverage `0.0%`
- `medium_rows` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.166563`, tasks `11`, unsupported coverage `0.0%`
- `medium_rows` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.325659`, tasks `11`, unsupported coverage `0.0%`
- `medium_rows` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.918133`, tasks `11`, unsupported coverage `0.0%`
- `medium_rows__Grouped` / `accuracy_gain`: `tabpfn-v2.2` (published_default), score `0.134512`, tasks `1`, unsupported coverage `0.0%`
- `medium_rows__Grouped` / `excess_cross_entropy`: `tabpfn-v2.2` (published_default), score `-0.457996`, tasks `1`, unsupported coverage `0.0%`
- `medium_rows__Grouped` / `macro_ovr_auc`: `tabpfn-v2.2` (published_default), score `0.977334`, tasks `1`, unsupported coverage `0.0%`
- `medium_rows__IID` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.17002`, tasks `10`, unsupported coverage `0.0%`
- `medium_rows__IID` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.313103`, tasks `10`, unsupported coverage `0.0%`
- `medium_rows__IID` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.91234`, tasks `10`, unsupported coverage `0.0%`
- `multi` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.258982`, tasks `8`, unsupported coverage `0.0%`
- `multi` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.555156`, tasks `8`, unsupported coverage `0.0%`
- `multi` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.934418`, tasks `8`, unsupported coverage `0.0%`
- `multi__Grouped` / `accuracy_gain`: `tabpfn-v2.2` (published_default), score `0.139617`, tasks `2`, unsupported coverage `0.0%`
- `multi__Grouped` / `excess_cross_entropy`: `tabpfn-v2.2` (published_default), score `-0.373689`, tasks `2`, unsupported coverage `0.0%`
- `multi__Grouped` / `macro_ovr_auc`: `tabpfn-v2.6` (published_default), score `0.909496`, tasks `2`, unsupported coverage `0.0%`
- `multi__IID` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.30074`, tasks `6`, unsupported coverage `0.0%`
- `multi__IID` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.620102`, tasks `6`, unsupported coverage `0.0%`
- `multi__IID` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.944062`, tasks `6`, unsupported coverage `0.0%`
- `small_feat` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.184347`, tasks `12`, unsupported coverage `0.0%`
- `small_feat` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.366681`, tasks `12`, unsupported coverage `0.0%`
- `small_feat` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.863285`, tasks `12`, unsupported coverage `0.0%`
- `small_rows` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.141783`, tasks `19`, unsupported coverage `0.0%`
- `small_rows` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.268281`, tasks `19`, unsupported coverage `0.0%`
- `small_rows` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.854423`, tasks `19`, unsupported coverage `0.0%`
- `small_rows__Grouped` / `accuracy_gain`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-fixed-medium/seed-2402` (final), score `0.0971661`, tasks `2`, unsupported coverage `0.0%`
- `small_rows__Grouped` / `excess_cross_entropy`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (final), score `-0.134318`, tasks `2`, unsupported coverage `0.0%`
- `small_rows__Grouped` / `macro_ovr_auc`: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (final), score `0.840084`, tasks `2`, unsupported coverage `0.0%`
- `small_rows__IID` / `accuracy_gain`: `tabicl-v2` (published_default), score `0.148521`, tasks `17`, unsupported coverage `0.0%`
- `small_rows__IID` / `excess_cross_entropy`: `tabicl-v2` (published_default), score `-0.290803`, tasks `17`, unsupported coverage `0.0%`
- `small_rows__IID` / `macro_ovr_auc`: `tabicl-v2` (published_default), score `0.862489`, tasks `17`, unsupported coverage `0.0%`

Unsupported full-fold evaluations have no metric score and are never replaced with a subsampled result.
