# BeyondArena ranking: final v4 plus published models

Comparison set: all v4 `final` selections plus published TabPFN v2.2/v2.6/v3 and TabICL v1/v2.
Models are ranked independently within each task after fold averaging; the reported rank is the mean task rank within this comparison set. Lower mean rank is better.

- Models: 26
- Conditions: 20
- Ranking rows: 1560

## Best model by condition and metric

### `Grouped`

- excess_cross_entropy: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283915/g_z-curriculum-small/seed-2402` (mean task rank `6`)
- macro_ovr_auc: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (mean task rank `6`)
- accuracy_gain: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-fixed-medium/seed-2402` (mean task rank `4`)

### `IID`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.931`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.034`)
- accuracy_gain: `tabpfn-v3` (mean task rank `2.655`)

### `all`

- excess_cross_entropy: `tabicl-v2` (mean task rank `2.688`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.531`)
- accuracy_gain: `tabpfn-v3` (mean task rank `3.188`)

### `binary`

- excess_cross_entropy: `tabicl-v2` (mean task rank `3`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `4`)
- accuracy_gain: `tabpfn-v3` (mean task rank `3.333`)

### `binary__Grouped`

- excess_cross_entropy: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-fixed-medium/seed-2402` (mean task rank `1`)
- macro_ovr_auc: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (mean task rank `1`)
- accuracy_gain: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/g_z-fixed-medium/seed-2402` (mean task rank `1`)

### `binary__IID`

- excess_cross_entropy: `tabicl-v2` (mean task rank `2.043`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.391`)
- accuracy_gain: `tabpfn-v3` (mean task rank `2.913`)

### `large_feat`

- excess_cross_entropy: `tabicl-v2` (mean task rank `6`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `4.6`)
- accuracy_gain: `tabicl-v2` (mean task rank `3`)

### `large_rows`

- excess_cross_entropy: `tabpfn-v3` (mean task rank `1`)
- macro_ovr_auc: `tabpfn-v3` (mean task rank `1`)
- accuracy_gain: `tabpfn-v3` (mean task rank `1`)

### `large_rows__IID`

- excess_cross_entropy: `tabpfn-v3` (mean task rank `1`)
- macro_ovr_auc: `tabpfn-v3` (mean task rank `1`)
- accuracy_gain: `tabpfn-v3` (mean task rank `1`)

### `medium_feat`

- excess_cross_entropy: `tabicl-v2` (mean task rank `2.333`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.333`)
- accuracy_gain: `tabpfn-v3` (mean task rank `3.4`)

### `medium_rows`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.909`)
- macro_ovr_auc: `tabpfn-v3` (mean task rank `2.091`)
- accuracy_gain: `tabpfn-v3` (mean task rank `2.909`)

### `medium_rows__Grouped`

- excess_cross_entropy: `tabpfn-v2.2` (mean task rank `1`)
- macro_ovr_auc: `tabpfn-v2.2` (mean task rank `1`)
- accuracy_gain: `tabpfn-v2.2` (mean task rank `1`)

### `medium_rows__IID`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.9`)
- macro_ovr_auc: `tabpfn-v3` (mean task rank `2`)
- accuracy_gain: `tabpfn-v3` (mean task rank `2.9`)

### `multi`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.75`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `2.125`)
- accuracy_gain: `tabicl-v2` (mean task rank `2.375`)

### `multi__Grouped`

- excess_cross_entropy: `tabpfn-v2.2` (mean task rank `1.5`)
- macro_ovr_auc: `tabpfn-v2.2` (mean task rank `2`)
- accuracy_gain: `tabpfn-v2.2` (mean task rank `2`)

### `multi__IID`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.5`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `1.667`)
- accuracy_gain: `tabpfn-v3` (mean task rank `1.667`)

### `small_feat`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.75`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.333`)
- accuracy_gain: `tabicl-v2` (mean task rank `2.25`)

### `small_rows`

- excess_cross_entropy: `tabicl-v2` (mean task rank `3.105`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `4.368`)
- accuracy_gain: `tabicl-v2` (mean task rank `3.053`)

### `small_rows__Grouped`

- excess_cross_entropy: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283915/g_z-curriculum-small/seed-2402` (mean task rank `4.5`)
- macro_ovr_auc: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/r_z-curriculum-medium/seed-2402` (mean task rank `5.5`)
- accuracy_gain: `seed-2402@/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/37283921/g_z-curriculum-medium/seed-2402` (mean task rank `2.5`)

### `small_rows__IID`

- excess_cross_entropy: `tabicl-v2` (mean task rank `1.824`)
- macro_ovr_auc: `tabicl-v2` (mean task rank `3.529`)
- accuracy_gain: `tabicl-v2, tabpfn-v3` (mean task rank `2.706`)

