# Native-TabICL-prior runs — results so far

Date: 2026-09-21 17:20. Companion to `paper/multiregime_v4_results.md` (rev. 6). Everything here is on the
**native prior** (fix of 2026-09-20; see §1) and the **native banks**; numbers are not comparable to the old
banks except through BeyondArena (same real data).

## 1. Why: the 2026-09-20 finding

The v4 production prior (`tfmplayground/experiments/multiregime_v4.py`, profile `production`) replaced TabICL's
label rule with a quantile calibration to a fixed class vector: every binary training task was 50/50 and every
K-class task 1/K, while the evaluation bank's binary cells are 0.1/0.3/0.5 and native TabICL's binary class
ratio is ~uniform on (0,1). Measured on the old dumps (20k episodes each): `original`, `r_z-*`, `g_z-*` binary
tasks 100 % in [0.50, 0.55) majority; the bank's binary episodes 66 % ≥ 0.8 majority; native TabICL 41 % ≥ 0.8.
The `original` control was therefore not native TabICL either (the native path was gated behind the
`tabicl_test` profile). This is the cause of the old study's "binary-imbalance collapse" on the bank and on the
≥ 0.8-majority BeyondArena datasets.

Fix (`hp_profile="native"`, now the default): TabICL's `DEFAULT_FIXED_HP` kept fixed, `DEFAULT_SAMPLED_HP`
sampled as TabICL does, native `Reg2Cls` labels (K=1 is bit-identical to `tabicl.prior.PriorDataset`, tested
for both mechanisms), one native label rule per regime for K>1 (`r_z`: per SCM score column; `g_z`:
independent draws on one score), TabICL's split sanity check on the selected labels, non-finite-X redraw.
`class_ratio` (bank cells) keeps the calibrated path. Dump class ratios after the fix: `original` binary
majority mean 0.766 / 45 % ≥ 0.8 (native TabICL 0.757 / 41 %); multiregime dumps 0.66 / 15 % pooled (per
regime the same spread as K=1; pooling K independent rules moves tasks toward balance). Multiclass is
imbalanced too (3-class majority 0.65, 5-class 0.50; 6–10 % of K=1 tasks realise fewer classes than requested).

## 2. Data and runs

* Dumps (`/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_paired_zmix5/`, job 37403290):
  `original` (K=1), `r_z-multiregime`, `g_z-multiregime`, 100k episodes each, same geometry as the old dumps,
  **z exposed in 50 % of the episodes** (per-episode `expose_z`). No shared-rule dumps: K=1 is the control.
* Banks (`…/tabicl_mix_scm_evaluation` blind, `…_expose_z`; jobs 37403291 / 37403364): same factorial cell grid
  and seeds as bank 37277969 on the native base prior; controlled binary cells keep the calibrated 0.1/0.3/0.5
  ratios, multiclass cells use native labels with every requested class present. Two validation episodes with
  non-finite X (native feature processing) were regenerated (job 37406225). Class-prior CE: validation 0.8360,
  test 0.8356; majority accuracy 0.596.
* Runs (`scripts/slurm/pretrain_multiregime_v4_native.sbatch`; `/scratch/users/k23139234/tfm_runs/native/`):
  `original`, `rg_z-fixed` (multiregime share **0.3**), `rg_z-curriculum` (ramp 0 → 0.5 over steps 1000–5000)
  × small / medium / large; 10k steps, **20 % warmup** (2000 steps), otherwise the v4 recipe; ordinary branch =
  `original.h5`, multiregime branch = r_z + g_z dumps round-robin; validation every 500 steps on both banks
  (`v4_validation/`, `v4_validation_zx/`); `latest_checkpoint.pth` every 50 steps.
  Status (23 Sep): **all nine runs complete.** small ×3 and medium ×3 on A30/A100; large original and
  rg_z-curriculum on A100 (20 h each, 37403309_2/_20); rg_z-fixed-large started on an A30 and was taken over by
  the A100 job 37424403_17 at step ~6100 on 22 Sep 13:08 (PREEMPT_JOB cancels the A30 twin and resumes from
  latest_checkpoint.pth), finishing at 22:06. Bank evals, BeyondArena and the per-regime breakdown are done for
  every run.
* Evals: every checkpoint on both banks (`runs_eval/native_bank{,_expose_z}/native/<run>/`), BeyondArena
  (`results/beyondarena-v4/nat-<run>-*`), published TabPFN/TabICL + conventional ML on the native banks
  (`/scratch/users/k23139234/tfm_eval/native_references/`, jobs 37422518 / 37422529, in progress).
* Figures: `figures/native/native_history.png` (bank / ordinary / training CE per size),
  `figures/native/native_<size>_{excess_ce,acc_gain}_{cells,regime_x_classes}.png` (500-step resolution,
  z-blind vs z-exposed bank, final test as diamonds).

## 3. Synthetic bank — final TEST, pooled

| run | test CE | excess CE | accuracy | AUC |
|---|---|---|---|---|
| nat-original-small | 0.8164 | -0.0192 | 0.6178 | 0.5781 |
| nat-rg_z-fixed-small | 0.8157 | -0.0199 | 0.6182 | 0.5783 |
| nat-rg_z-curriculum-small | 0.8161 | -0.0195 | 0.6179 | 0.5775 |
| nat-original-medium | 0.8095 | -0.0261 | 0.6304 | 0.6055 |
| nat-rg_z-fixed-medium | 0.8053 | -0.0303 | 0.6312 | 0.6056 |
| nat-rg_z-curriculum-medium | 0.8036 | -0.0320 | 0.6317 | 0.6059 |
| nat-original-large | 0.7811 | -0.0545 | 0.6385 | 0.6106 |
| nat-rg_z-fixed-large | 0.7802 | -0.0554 | 0.6393 | 0.6117 |
| nat-rg_z-curriculum-large | 0.7803 | -0.0553 | 0.6389 | 0.6114 |
| *reference: TabICL v1.1* | 0.7750 | -0.0606 | 0.6419 | 0.6143 |
| *reference: TabICL v1* | 0.7529 | -0.0827 | 0.6484 | 0.6225 |
| *reference: TabPFN v2.2 / v2.6 / v3* | 0.7459 / 0.7513 / 0.7465 | -0.090 / -0.084 / -0.089 | 0.651 / 0.650 / 0.651 | 0.623 / 0.622 / 0.624 |
| *reference: rf / logreg / catboost* | 0.8471 / 0.8557 / 0.9383 | +0.012 / +0.020 / +0.103 | 0.632 / 0.617 / 0.627 | 0.612 / 0.588 / 0.613 |

Old small runs on the old bank finished at excess CE +0.010 … +0.021 (worse than the prior after a mid-run
minimum of −0.02); old medium best −0.02. No late drift in any native run at small; at medium `original`
drifts up from step 3500 (0.804 → 0.810) while both rg_z runs hold (~0.804). Large: −0.024 test CE vs medium
for both families, no drift; the multiregime advantage shrinks to 0.001 pooled (0.004 at medium) but the large
runs are now within 0.006 of TabICL v1.1 and 0.03 of TabICL v1 / TabPFN on the same bank. The conventional
models never beat the class prior on likelihood (rf +0.012 is the closest); their accuracy is above the majority
baseline (0.596) but below every native run ≥ medium.

## 4. Synthetic bank — final TEST excess CE per cell (z-blind bank / z-exposed bank)

### small

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| all cells | -0.0191 | -0.0192 | -0.0199 | -0.0201 | -0.0195 | -0.0198 |
| binary, ratio 0.1 | -0.0074 | -0.0063 | -0.0076 | -0.0065 | -0.0075 | -0.0063 |
| binary, ratio 0.3 | -0.0183 | -0.0171 | -0.0194 | -0.0184 | -0.0195 | -0.0183 |
| binary, ratio 0.5 | -0.0185 | -0.0170 | -0.0204 | -0.0191 | -0.0207 | -0.0194 |
| 3 classes | -0.0290 | -0.0301 | -0.0300 | -0.0313 | -0.0294 | -0.0309 |
| 4 classes | -0.0247 | -0.0262 | -0.0254 | -0.0269 | -0.0243 | -0.0261 |
| 5 classes | -0.0167 | -0.0186 | -0.0164 | -0.0185 | -0.0154 | -0.0176 |
| K=1 (single rule) | -0.0242 | -0.0230 | -0.0246 | -0.0235 | -0.0242 | -0.0231 |
| K=2 shared rule | -0.0213 | -0.0198 | -0.0219 | -0.0204 | -0.0214 | -0.0200 |
| K=2 multiregime | -0.0083 | -0.0106 | -0.0094 | -0.0119 | -0.0101 | -0.0128 |
| K=3 shared rule | -0.0300 | -0.0290 | -0.0303 | -0.0295 | -0.0293 | -0.0284 |
| K=3 multiregime | -0.0093 | -0.0122 | -0.0107 | -0.0138 | -0.0103 | -0.0135 |
| K=4 shared rule | -0.0291 | -0.0282 | -0.0297 | -0.0289 | -0.0283 | -0.0275 |
| K=4 multiregime | -0.0059 | -0.0083 | -0.0072 | -0.0098 | -0.0070 | -0.0097 |
| multiregime K≥2, binary | +0.0067 | +0.0078 | +0.0055 | +0.0065 | +0.0049 | +0.0060 |
| multiregime K≥2, multiclass | -0.0224 | -0.0286 | -0.0238 | -0.0302 | -0.0234 | -0.0302 |

### medium (z-exposed test from the checkpoint evals; rg_z-fixed / rg_z-curriculum z-exposed pending)

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-curriculum (z-blind) |
|---|---|---|---|---|
| all cells | -0.0261 | -0.0264 | -0.0302 | -0.0320 |
| binary, ratio 0.1 | +0.0042 | +0.0040 | +0.0025 | +0.0031 |
| binary, ratio 0.3 | -0.0103 | -0.0103 | -0.0117 | -0.0124 |
| binary, ratio 0.5 | -0.0135 | -0.0132 | -0.0160 | -0.0182 |
| 3 classes | -0.0449 | -0.0454 | -0.0500 | -0.0539 |
| 4 classes | -0.0475 | -0.0477 | -0.0544 | -0.0563 |
| 5 classes | -0.0446 | -0.0457 | -0.0518 | -0.0541 |
| K=1 (single rule) | -0.0165 | -0.0153 | -0.0213 | -0.0227 |
| K=2 shared rule | -0.0144 | -0.0134 | -0.0193 | -0.0210 |
| K=2 multiregime | +0.0048 | +0.0032 | -0.0029 | -0.0059 |
| K=3 shared rule | -0.0610 | -0.0603 | -0.0615 | -0.0620 |
| K=3 multiregime | -0.0339 | -0.0368 | -0.0375 | -0.0400 |
| K=4 shared rule | -0.0616 | -0.0610 | -0.0621 | -0.0625 |
| K=4 multiregime | -0.0301 | -0.0339 | -0.0346 | -0.0374 |
| multiregime K≥2, binary | +0.0165 | +0.0161 | +0.0146 | +0.0142 |
| multiregime K≥2, multiclass | -0.0511 | -0.0560 | -0.0602 | -0.0654 |

### large (final TEST excess CE; all three runs, both banks)

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| all cells | -0.0544 | -0.0537 | -0.0554 | -0.0541 | -0.0553 | -0.0537 |
| binary, ratio 0.1 | -0.0133 | -0.0125 | -0.0130 | -0.0123 | -0.0121 | -0.0113 |
| binary, ratio 0.3 | -0.0364 | -0.0354 | -0.0360 | -0.0347 | -0.0353 | -0.0338 |
| binary, ratio 0.5 | -0.0407 | -0.0403 | -0.0409 | -0.0397 | -0.0412 | -0.0400 |
| 3 classes | -0.0779 | -0.0774 | -0.0791 | -0.0778 | -0.0803 | -0.0786 |
| 4 classes | -0.0802 | -0.0793 | -0.0817 | -0.0797 | -0.0819 | -0.0796 |
| 5 classes | -0.0781 | -0.0775 | -0.0815 | -0.0803 | -0.0808 | -0.0789 |
| K=1 (single rule) | -0.0501 | -0.0485 | -0.0487 | -0.0462 | -0.0487 | -0.0455 |
| K=2 shared rule | -0.0496 | -0.0480 | -0.0490 | -0.0465 | -0.0488 | -0.0458 |
| K=2 multiregime | -0.0313 | -0.0302 | -0.0337 | -0.0318 | -0.0344 | -0.0325 |
| K=3 shared rule | -0.0812 | -0.0807 | -0.0819 | -0.0810 | -0.0806 | -0.0797 |
| K=3 multiregime | -0.0524 | -0.0531 | -0.0568 | -0.0574 | -0.0571 | -0.0579 |
| K=4 shared rule | -0.0816 | -0.0812 | -0.0819 | -0.0811 | -0.0807 | -0.0800 |
| K=4 multiregime | -0.0512 | -0.0526 | -0.0560 | -0.0575 | -0.0566 | -0.0581 |
| multiregime K≥2, binary | -0.0034 | -0.0030 | -0.0032 | -0.0024 | -0.0032 | -0.0026 |
| multiregime K≥2, multiclass | -0.0838 | -0.0846 | -0.0914 | -0.0920 | -0.0926 | -0.0931 |
| multiregime, soft_gate | -0.0824 | -0.0838 | -0.0892 | -0.0909 | -0.0908 | -0.0926 |
| multiregime, persistent | -0.0049 | -0.0038 | -0.0054 | -0.0036 | -0.0050 | -0.0030 |

Reference models on the same bank (z-blind test): TabICL v1.1 -0.061, TabICL v1 -0.083, TabICL v2 -0.084,
TabPFN v2.6 -0.084, v3 -0.089, v2.2 -0.090; rf +0.012, logreg +0.020, catboost +0.103.

**rg_z-fixed matches the curriculum with 30 % multiregime exposure instead of a 0->50 % ramp**: pooled -0.0554 vs
-0.0553 (and the best final validation CE of the three, 0.7809), with the curriculum keeping a small edge on the
soft_gate cells (-0.0908 vs -0.0892) and rg_z-fixed slightly ahead on 5 classes (-0.0815 vs -0.0808) and on the
z-exposed bank at K=1/K=2 shared. Both beat original by 0.0009-0.0010 pooled and by 0.007-0.009 on multiregime
multiclass, and lose 0.0014 on K=1.

Large resolves the two medium failure modes: `multiregime K≥2, binary` goes from +0.015 (medium) to −0.003 —
better than TabICL v1.1 (+0.008) — and `persistent` from +0.013 to −0.005 (TabICL v1.1 0.000, TabPFN −0.02);
both cells were still positive at step 5000 and turned negative between steps 5000 and 8000, i.e. after the
cosine LR passed its mid-point. The multiregime prior's advantage at large is concentrated on the multiregime
cells (K≥2 multiclass −0.093 vs −0.084; K=3/4 multiregime −0.005 each) and slightly negative on K=1 / shared
(+0.001), net −0.001 pooled. The large runs beat TabICL v1.1 on binary ratio 0.3/0.5 and shared-rule binary and
trail it on multiclass (−0.08 vs −0.08…−0.12) and K=1 (−0.05 vs −0.062).

### Raw-scale interpretation: binary class ratio

Raw CE and accuracy are deliberately not used for the cross-ratio comparisons above: their trivial
class-prior floors change substantially with the minority-class ratio. For the intended binary rates,
the nominal prior values are:

| minority ratio | class-prior CE (nats) | majority accuracy | constant-score AUC |
|---:|---:|---:|---:|
| 0.1 | 0.325 | 0.900 | 0.500 |
| 0.3 | 0.611 | 0.700 | 0.500 |
| 0.5 | 0.693 | 0.500 | 0.500 |

Thus raw CE is *lower* and raw accuracy is *higher* on a 90/10 cell even for a model that learns
nothing beyond prevalence. This is a base-rate effect, not evidence that the imbalanced task is
easier or that the model has higher discriminative skill there.

For orientation, adding the medium test excess-CE values to the nominal binary entropy gives the
following **approximate** raw CEs. The exact comparator uses each episode's empirical support-class
frequency, so these reconstructions are not a replacement for raw per-episode reports.

| minority ratio | nat-original-medium CE | nat-rg_z-curriculum-medium CE |
|---:|---:|---:|
| 0.1 | 0.329 | 0.328 |
| 0.3 | 0.601 | 0.598 |
| 0.5 | 0.680 | 0.675 |

The saved medium accuracy-gain plot correspondingly puts raw accuracy at roughly 90%, 71%, and 58%
for ratios 0.1, 0.3, and 0.5: it mostly tracks the 90%, 70%, and 50% majority baselines. The raw
per-ratio accuracy and AUC reports were not retained in this checkout, so they should be regenerated
from the per-episode bank reports before quoting more precise values. Only pooled AUC is currently
available for these medium test runs: 0.6055 (`original`) and 0.6059 (`rg_z-curriculum`), both across
all cells and therefore not interpretable as a per-ratio AUC comparison.

### regime count × class condition (final TEST excess CE)

original-small (z-blind / z-exposed):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | -0.0150 / -0.0139 | -0.0358 / -0.0346 | -0.0339 / -0.0322 | -0.0277 / -0.0265 | -0.0193 / -0.0174 | -0.0135 / -0.0132 |
| K=2 shared | -0.0119 / -0.0107 | -0.0269 / -0.0254 | -0.0271 / -0.0254 | -0.0270 / -0.0253 | -0.0205 / -0.0186 | -0.0146 / -0.0137 |
| K=2 multiregime | +0.0049 / +0.0060 | +0.0084 / +0.0094 | +0.0081 / +0.0098 | -0.0301 / -0.0353 | -0.0235 / -0.0299 | -0.0173 / -0.0235 |
| K=3 shared | -0.0159 / -0.0147 | -0.0293 / -0.0275 | -0.0405 / -0.0393 | -0.0362 / -0.0352 | -0.0331 / -0.0330 | -0.0248 / -0.0243 |
| K=3 multiregime | +0.0054 / +0.0064 | +0.0072 / +0.0078 | +0.0054 / +0.0063 | -0.0289 / -0.0344 | -0.0265 / -0.0350 | -0.0183 / -0.0241 |
| K=4 shared | -0.0154 / -0.0139 | -0.0351 / -0.0339 | -0.0301 / -0.0289 | -0.0338 / -0.0332 | -0.0398 / -0.0389 | -0.0207 / -0.0206 |
| K=4 multiregime | +0.0061 / +0.0071 | +0.0064 / +0.0072 | +0.0081 / +0.0097 | -0.0214 / -0.0268 | -0.0206 / -0.0270 | -0.0139 / -0.0201 |

rg_z-fixed-small (z-blind / z-exposed):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | -0.0153 / -0.0142 | -0.0366 / -0.0356 | -0.0357 / -0.0344 | -0.0278 / -0.0266 | -0.0196 / -0.0176 | -0.0128 / -0.0125 |
| K=2 shared | -0.0122 / -0.0111 | -0.0279 / -0.0266 | -0.0292 / -0.0276 | -0.0277 / -0.0258 | -0.0203 / -0.0180 | -0.0142 / -0.0134 |
| K=2 multiregime | +0.0046 / +0.0056 | +0.0066 / +0.0074 | +0.0059 / +0.0075 | -0.0320 / -0.0377 | -0.0243 / -0.0307 | -0.0173 / -0.0236 |
| K=3 shared | -0.0156 / -0.0145 | -0.0305 / -0.0291 | -0.0424 / -0.0414 | -0.0367 / -0.0357 | -0.0330 / -0.0329 | -0.0240 / -0.0237 |
| K=3 multiregime | +0.0051 / +0.0059 | +0.0057 / +0.0061 | +0.0038 / +0.0045 | -0.0314 / -0.0370 | -0.0289 / -0.0376 | -0.0186 / -0.0249 |
| K=4 shared | -0.0155 / -0.0140 | -0.0356 / -0.0345 | -0.0319 / -0.0308 | -0.0342 / -0.0339 | -0.0410 / -0.0400 | -0.0199 / -0.0199 |
| K=4 multiregime | +0.0059 / +0.0068 | +0.0051 / +0.0059 | +0.0066 / +0.0080 | -0.0238 / -0.0294 | -0.0222 / -0.0288 | -0.0146 / -0.0212 |

rg_z-curriculum-small (z-blind / z-exposed):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | -0.0151 / -0.0140 | -0.0364 / -0.0353 | -0.0360 / -0.0347 | -0.0270 / -0.0259 | -0.0190 / -0.0172 | -0.0119 / -0.0116 |
| K=2 shared | -0.0119 / -0.0106 | -0.0278 / -0.0263 | -0.0294 / -0.0278 | -0.0272 / -0.0258 | -0.0191 / -0.0171 | -0.0129 / -0.0121 |
| K=2 multiregime | +0.0040 / +0.0052 | +0.0056 / +0.0066 | +0.0048 / +0.0064 | -0.0332 / -0.0391 | -0.0247 / -0.0318 | -0.0173 / -0.0242 |
| K=3 shared | -0.0149 / -0.0136 | -0.0301 / -0.0283 | -0.0418 / -0.0408 | -0.0355 / -0.0346 | -0.0311 / -0.0312 | -0.0222 / -0.0219 |
| K=3 multiregime | +0.0048 / +0.0058 | +0.0054 / +0.0060 | +0.0031 / +0.0039 | -0.0307 / -0.0366 | -0.0268 / -0.0360 | -0.0177 / -0.0243 |
| K=4 shared | -0.0146 / -0.0130 | -0.0351 / -0.0339 | -0.0316 / -0.0305 | -0.0328 / -0.0322 | -0.0384 / -0.0375 | -0.0175 / -0.0176 |
| K=4 multiregime | +0.0056 / +0.0067 | +0.0049 / +0.0059 | +0.0058 / +0.0073 | -0.0231 / -0.0291 | -0.0212 / -0.0282 | -0.0139 / -0.0209 |

original-medium (z-blind):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | +0.0042 | -0.0197 | -0.0213 | -0.0214 | -0.0197 | -0.0214 |
| K=2 shared | +0.0070 | -0.0080 | -0.0108 | -0.0268 | -0.0201 | -0.0277 |
| K=2 multiregime | +0.0319 | +0.0370 | +0.0326 | -0.0294 | -0.0206 | -0.0227 |
| K=3 shared | -0.0216 | -0.0412 | -0.0591 | -0.0785 | -0.0821 | -0.0837 |
| K=3 multiregime | +0.0074 | +0.0033 | -0.0013 | -0.0683 | -0.0795 | -0.0649 |
| K=4 shared | -0.0192 | -0.0488 | -0.0446 | -0.0739 | -0.1019 | -0.0812 |
| K=4 multiregime | +0.0093 | +0.0047 | +0.0064 | -0.0664 | -0.0724 | -0.0623 |

rg_z-curriculum-medium (z-blind):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | +0.0024 | -0.0229 | -0.0266 | -0.0308 | -0.0279 | -0.0304 |
| K=2 shared | +0.0051 | -0.0104 | -0.0181 | -0.0347 | -0.0303 | -0.0373 |
| K=2 multiregime | +0.0300 | +0.0346 | +0.0248 | -0.0462 | -0.0374 | -0.0414 |
| K=3 shared | -0.0214 | -0.0430 | -0.0614 | -0.0794 | -0.0823 | -0.0845 |
| K=3 multiregime | +0.0075 | +0.0021 | -0.0034 | -0.0787 | -0.0899 | -0.0774 |
| K=4 shared | -0.0196 | -0.0498 | -0.0469 | -0.0749 | -0.1019 | -0.0820 |
| K=4 multiregime | +0.0094 | +0.0037 | +0.0037 | -0.0796 | -0.0867 | -0.0750 |

large (z-exposed test for original-large pending its checkpoint eval):

original-large (z-blind / z-exposed):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | -0.0194 / — | -0.0539 / — | -0.0578 / — | -0.0600 / — | -0.0549 / — | -0.0545 / — |
| K=2 shared | -0.0159 / — | -0.0450 / — | -0.0497 / — | -0.0669 / — | -0.0599 / — | -0.0605 / — |
| K=2 multiregime | +0.0075 / — | -0.0025 / — | -0.0057 / — | -0.0724 / — | -0.0574 / — | -0.0574 / — |
| K=3 shared | -0.0308 / — | -0.0564 / — | -0.0744 / — | -0.1007 / — | -0.1071 / — | -0.1182 / — |
| K=3 multiregime | -0.0019 / — | -0.0082 / — | -0.0120 / — | -0.0917 / — | -0.1048 / — | -0.0961 / — |
| K=4 shared | -0.0287 / — | -0.0620 / — | -0.0596 / — | -0.0939 / — | -0.1289 / — | -0.1163 / — |
| K=4 multiregime | +0.0003 / — | -0.0056 / — | -0.0060 / — | -0.0948 / — | -0.1054 / — | -0.0960 / — |

rg_z-curriculum-large (z-blind / z-exposed):

| K / rule | bin 0.1 | bin 0.3 | bin 0.5 | 3 cls | 4 cls | 5 cls |
|---|---|---|---|---|---|---|
| K=1 | -0.0180 / -0.0168 | -0.0522 / -0.0494 | -0.0580 / -0.0561 | -0.0604 / -0.0566 | -0.0521 / -0.0475 | -0.0512 / -0.0466 |
| K=2 shared | -0.0141 / -0.0130 | -0.0426 / -0.0401 | -0.0507 / -0.0484 | -0.0670 / -0.0638 | -0.0576 / -0.0529 | -0.0610 / -0.0564 |
| K=2 multiregime | +0.0097 / +0.0108 | -0.0017 / -0.0004 | -0.0071 / -0.0051 | -0.0776 / -0.0758 | -0.0632 / -0.0599 | -0.0667 / -0.0646 |
| K=3 shared | -0.0304 / -0.0298 | -0.0561 / -0.0554 | -0.0748 / -0.0741 | -0.0997 / -0.0987 | -0.1060 / -0.1047 | -0.1168 / -0.1153 |
| K=3 multiregime | -0.0013 / -0.0010 | -0.0080 / -0.0080 | -0.0123 / -0.0124 | -0.0992 / -0.1005 | -0.1147 / -0.1163 | -0.1070 / -0.1093 |
| K=4 shared | -0.0284 / -0.0280 | -0.0614 / -0.0607 | -0.0595 / -0.0593 | -0.0932 / -0.0926 | -0.1283 / -0.1269 | -0.1137 / -0.1122 |
| K=4 multiregime | +0.0009 / +0.0010 | -0.0057 / -0.0060 | -0.0064 / -0.0063 | -0.1033 / -0.1051 | -0.1154 / -0.1182 | -0.1099 / -0.1143 |

### multiregime cells (K≥2) by mechanism × routing family

small:

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| r_z × soft_gate | -0.0217 | -0.0280 | -0.0234 | -0.0302 | -0.0234 | -0.0304 |
| g_z × soft_gate | -0.0189 | -0.0250 | -0.0203 | -0.0268 | -0.0203 | -0.0270 |
| r_z × persistent | +0.0018 | +0.0028 | +0.0008 | +0.0017 | +0.0006 | +0.0015 |
| g_z × persistent | +0.0073 | +0.0088 | +0.0064 | +0.0078 | +0.0061 | +0.0075 |
| soft_gate, binary | +0.0055 | +0.0065 | +0.0043 | +0.0051 | +0.0038 | +0.0048 |
| soft_gate, multiclass | -0.0460 | -0.0595 | -0.0480 | -0.0620 | -0.0475 | -0.0622 |
| persistent, binary | +0.0079 | +0.0092 | +0.0068 | +0.0078 | +0.0060 | +0.0072 |
| persistent, multiclass | +0.0012 | +0.0024 | +0.0005 | +0.0017 | +0.0007 | +0.0018 |
| r_z, binary | +0.0030 | +0.0041 | +0.0018 | +0.0027 | +0.0012 | +0.0023 |
| r_z, multiclass | -0.0229 | -0.0293 | -0.0244 | -0.0311 | -0.0240 | -0.0311 |
| g_z, binary | +0.0104 | +0.0115 | +0.0092 | +0.0102 | +0.0085 | +0.0097 |
| g_z, multiclass | -0.0219 | -0.0278 | -0.0231 | -0.0292 | -0.0227 | -0.0292 |
| shared rule, r_z | -0.0277 | -0.0266 | -0.0285 | -0.0274 | -0.0274 | -0.0263 |
| shared rule, g_z | -0.0248 | -0.0236 | -0.0251 | -0.0239 | -0.0243 | -0.0232 |

medium:

| cell | original (z-blind) | rg_z-curriculum (z-blind) |
|---|---|---|
| r_z × soft_gate | -0.0536 | -0.0656 |
| g_z × soft_gate | -0.0493 | -0.0616 |
| r_z × persistent | +0.0139 | +0.0091 |
| g_z × persistent | +0.0198 | +0.0157 |
| soft_gate, binary | +0.0120 | +0.0098 |
| soft_gate, multiclass | -0.1148 | -0.1371 |
| persistent, binary | +0.0210 | +0.0186 |
| persistent, multiclass | +0.0127 | +0.0063 |
| r_z, binary | +0.0117 | +0.0093 |
| r_z, multiclass | -0.0513 | -0.0658 |
| g_z, binary | +0.0213 | +0.0191 |
| g_z, multiclass | -0.0508 | -0.0650 |
| shared rule, r_z | -0.0454 | -0.0487 |
| shared rule, g_z | -0.0397 | -0.0427 |

large:

| cell | original (z-blind) | rg_z-curriculum (z-blind) | original (z-exposed) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|
| r_z × soft_gate | -0.0848 | -0.0933 | — | -0.0951 |
| g_z × soft_gate | -0.0800 | -0.0882 | — | -0.0900 |
| r_z × persistent | -0.0077 | -0.0078 | — | -0.0059 |
| g_z × persistent | -0.0020 | -0.0022 | — | -0.0002 |
| soft_gate, binary | -0.0086 | -0.0085 | — | -0.0083 |
| soft_gate, multiclass | -0.1562 | -0.1730 | — | -0.1769 |
| persistent, binary | +0.0017 | +0.0022 | — | +0.0032 |
| persistent, multiclass | -0.0114 | -0.0122 | — | -0.0093 |
| r_z, binary | -0.0077 | -0.0077 | — | -0.0070 |
| r_z, multiclass | -0.0848 | -0.0934 | — | -0.0940 |
| g_z, binary | +0.0008 | +0.0014 | — | +0.0019 |
| g_z, multiclass | -0.0828 | -0.0918 | — | -0.0921 |
| shared rule, r_z | -0.0726 | -0.0721 | — | -0.0704 |
| shared rule, g_z | -0.0648 | -0.0638 | — | -0.0620 |
| K=1, r_z | -0.0474 | -0.0464 | — | -0.0435 |
| K=1, g_z | -0.0527 | -0.0510 | — | -0.0475 |

### multiregime cells (K≥2) by mechanism × routing family × class kind

small (z-blind / z-exposed test bank):

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| r_z × soft_gate, binary | +0.0032 | +0.0042 | +0.0019 | +0.0027 | +0.0014 | +0.0024 |
| r_z × soft_gate, multiclass | -0.0465 | -0.0602 | -0.0487 | -0.0630 | -0.0482 | -0.0632 |
| r_z × persistent, binary | +0.0029 | +0.0040 | +0.0017 | +0.0027 | +0.0011 | +0.0022 |
| r_z × persistent, multiclass | +0.0007 | +0.0015 | -0.0001 | +0.0007 | +0.0002 | +0.0009 |
| g_z × soft_gate, binary | +0.0078 | +0.0088 | +0.0067 | +0.0075 | +0.0062 | +0.0071 |
| g_z × soft_gate, multiclass | -0.0455 | -0.0588 | -0.0473 | -0.0611 | -0.0467 | -0.0612 |
| g_z × persistent, binary | +0.0129 | +0.0143 | +0.0118 | +0.0129 | +0.0109 | +0.0122 |
| g_z × persistent, multiclass | +0.0017 | +0.0032 | +0.0010 | +0.0026 | +0.0012 | +0.0028 |

medium (z-blind):

| cell | original (z-blind) | rg_z-curriculum (z-blind) |
|---|---|---|
| r_z × soft_gate, binary | +0.0083 | +0.0063 |
| r_z × soft_gate, multiclass | -0.1154 | -0.1376 |
| r_z × persistent, binary | +0.0150 | +0.0123 |
| r_z × persistent, multiclass | +0.0128 | +0.0059 |
| g_z × soft_gate, binary | +0.0156 | +0.0134 |
| g_z × soft_gate, multiclass | -0.1142 | -0.1366 |
| g_z × persistent, binary | +0.0271 | +0.0248 |
| g_z × persistent, multiclass | +0.0126 | +0.0066 |

### multiregime cells by K × mechanism × routing family

small:

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| K=2, r_z × soft_gate | -0.0234 | -0.0296 | -0.0250 | -0.0318 | -0.0259 | -0.0328 |
| K=2, r_z × persistent | +0.0015 | +0.0023 | +0.0005 | +0.0015 | +0.0002 | +0.0010 |
| K=2, g_z × soft_gate | -0.0202 | -0.0262 | -0.0215 | -0.0278 | -0.0224 | -0.0291 |
| K=2, g_z × persistent | +0.0090 | +0.0112 | +0.0084 | +0.0104 | +0.0076 | +0.0096 |
| K=3, r_z × soft_gate | -0.0221 | -0.0289 | -0.0243 | -0.0313 | -0.0234 | -0.0306 |
| K=3, r_z × persistent | +0.0004 | +0.0012 | -0.0008 | -0.0002 | -0.0007 | +0.0001 |
| K=3, g_z × soft_gate | -0.0202 | -0.0268 | -0.0215 | -0.0283 | -0.0209 | -0.0281 |
| K=3, g_z × persistent | +0.0049 | +0.0058 | +0.0038 | +0.0046 | +0.0037 | +0.0046 |
| K=4, r_z × soft_gate | -0.0190 | -0.0250 | -0.0205 | -0.0269 | -0.0202 | -0.0269 |
| K=4, r_z × persistent | +0.0037 | +0.0049 | +0.0028 | +0.0040 | +0.0026 | +0.0038 |
| K=4, g_z × soft_gate | -0.0157 | -0.0216 | -0.0175 | -0.0238 | -0.0169 | -0.0233 |
| K=4, g_z × persistent | +0.0074 | +0.0085 | +0.0065 | +0.0075 | +0.0065 | +0.0076 |

medium (z-blind):

| cell | original (z-blind) | rg_z-curriculum (z-blind) |
|---|---|---|
| K=2, r_z × soft_gate | -0.0295 | -0.0425 |
| K=2, r_z × persistent | +0.0327 | +0.0241 |
| K=2, g_z × soft_gate | -0.0255 | -0.0390 |
| K=2, g_z × persistent | +0.0414 | +0.0336 |
| K=3, r_z × soft_gate | -0.0694 | -0.0797 |
| K=3, r_z × persistent | -0.0025 | -0.0050 |
| K=3, g_z × soft_gate | -0.0660 | -0.0755 |
| K=3, g_z × persistent | +0.0024 | +0.0003 |
| K=4, r_z × soft_gate | -0.0698 | -0.0825 |
| K=4, r_z × persistent | +0.0051 | +0.0033 |
| K=4, g_z × soft_gate | -0.0644 | -0.0778 |
| K=4, g_z × persistent | +0.0085 | +0.0073 |

large:

| cell | original (z-blind) | rg_z-curriculum (z-blind) | original (z-exposed) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|
| K=2, r_z × soft_gate | -0.0665 | -0.0733 | — | -0.0731 |
| K=2, r_z × persistent | -0.0027 | -0.0030 | — | +0.0002 |
| K=2, g_z × soft_gate | -0.0622 | -0.0672 | — | -0.0669 |
| K=2, g_z × persistent | +0.0062 | +0.0058 | — | +0.0098 |
| K=3, r_z × soft_gate | -0.0942 | -0.1031 | — | -0.1055 |
| K=3, r_z × persistent | -0.0152 | -0.0151 | — | -0.0141 |
| K=3, g_z × soft_gate | -0.0900 | -0.0992 | — | -0.1018 |
| K=3, g_z × persistent | -0.0104 | -0.0109 | — | -0.0102 |
| K=4, r_z × soft_gate | -0.0998 | -0.1102 | — | -0.1141 |
| K=4, r_z × persistent | -0.0069 | -0.0068 | — | -0.0059 |
| K=4, g_z × soft_gate | -0.0938 | -0.1052 | — | -0.1091 |
| K=4, g_z × persistent | -0.0045 | -0.0043 | — | -0.0034 |

Reading the large per-cell tables:

* **The multiregime prior's effect is entirely on multiregime cells and grows with capacity.** soft_gate multiclass
  −0.173 (rg_z-curriculum) vs −0.156 (original) at large, against −0.065 vs −0.051 at medium and −0.048 vs −0.046 at
  small; K=4 r_z × soft_gate −0.110 vs −0.100. On K=1 and shared-rule cells rg_z is 0.001 *worse* (−0.046 vs −0.047
  K=1 r_z), so the pooled gain stays small (−0.0553 vs −0.0544).
* **r_z beats g_z on every multiregime cell** (soft_gate −0.093 vs −0.088, multiclass −0.093 vs −0.092, binary
  −0.008 vs +0.001) but loses on K=1 (−0.046 vs −0.051) — consistent with r_z being the mechanism the multiregime
  dumps are built around (a separate SCM score column per regime), while g_z redraws rules on one score.
* **persistent routing is learnable at large.** Every K≥2 persistent cell is now ≤ 0 except K=2 g_z (+0.006):
  K=3 r_z −0.015, K=4 r_z −0.007, persistent multiclass −0.011. At small/medium these were all positive.
* **The K≥2 multiregime binary cells are the last positive ones** and only at K=2 (+0.008/+0.010 at ratio 0.1);
  from K=3 they are negative. Binary ratio 0.1 remains the hardest condition at every K.
* **Accuracy gain** follows the same pattern: rg_z-curriculum +0.070 vs +0.066 on multiregime soft_gate and
  +0.067 vs +0.063 on multiregime multiclass, identical elsewhere (pooled +0.0423 vs +0.0418).

### accuracy gain, small (z-blind / z-exposed)

| cell | original (z-blind) | original (z-exposed) | rg_z-fixed (z-blind) | rg_z-fixed (z-exposed) | rg_z-curriculum (z-blind) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|---|---|
| all cells | +0.0212 | +0.0216 | +0.0216 | +0.0222 | +0.0213 | +0.0218 |
| binary, ratio 0.1 | +0.0000 | -0.0001 | +0.0000 | -0.0000 | -0.0000 | -0.0001 |
| binary, ratio 0.3 | +0.0059 | +0.0047 | +0.0066 | +0.0057 | +0.0066 | +0.0056 |
| binary, ratio 0.5 | +0.0683 | +0.0675 | +0.0687 | +0.0685 | +0.0683 | +0.0674 |
| 3 classes | +0.0229 | +0.0241 | +0.0234 | +0.0251 | +0.0233 | +0.0246 |
| 4 classes | +0.0177 | +0.0193 | +0.0185 | +0.0202 | +0.0176 | +0.0196 |
| 5 classes | +0.0125 | +0.0141 | +0.0124 | +0.0140 | +0.0119 | +0.0135 |
| K=1 (single rule) | +0.0241 | +0.0237 | +0.0242 | +0.0238 | +0.0241 | +0.0236 |
| K=2 shared rule | +0.0228 | +0.0221 | +0.0233 | +0.0228 | +0.0224 | +0.0220 |
| K=2 multiregime | +0.0162 | +0.0180 | +0.0169 | +0.0192 | +0.0172 | +0.0190 |
| K=3 shared rule | +0.0265 | +0.0260 | +0.0267 | +0.0265 | +0.0263 | +0.0258 |
| K=3 multiregime | +0.0153 | +0.0175 | +0.0158 | +0.0185 | +0.0155 | +0.0181 |
| K=4 shared rule | +0.0257 | +0.0253 | +0.0261 | +0.0258 | +0.0254 | +0.0248 |
| K=4 multiregime | +0.0141 | +0.0164 | +0.0149 | +0.0173 | +0.0145 | +0.0169 |
| multiregime K≥2, binary | +0.0089 | +0.0082 | +0.0094 | +0.0091 | +0.0094 | +0.0086 |
| multiregime K≥2, multiclass | +0.0217 | +0.0265 | +0.0225 | +0.0277 | +0.0223 | +0.0276 |

Findings:
1. Every single-rule cell (K=1, shared) is below the prior, including binary ratio 0.1 / 0.3 where the old
   models collapsed — the prior fix works. Small: ratio 0.1 −0.008, 0.3 −0.019, 0.5 −0.021, 3/4/5 classes
   −0.03 / −0.025 / −0.016.
2. Multiregime multiclass cells are learned (small −0.02…−0.03; medium −0.06…−0.09, still improving at 10k),
   and only in `soft_gate` (regime visible in X; −0.046 small, −0.115/−0.137 medium). `persistent` (latent
   regime, not in X) stays at the prior for every model: no in-context learner can route without regime
   information. `r_z` (mechanism per regime) is slightly easier than `g_z` (same score, different cut).
3. z exposure helps exactly on the soft_gate multiregime cells (z is the soft-gate score): −0.005…−0.008 at
   small; no effect on K=1/shared cells; slightly *worse* on binary multiregime cells.
4. Remaining structural failure: **binary multiregime cells** stay above the prior (+0.004…+0.012 small,
   +0.005…+0.035 medium) and get worse after step 4000 — the model learns a rule-0 bias for two-class mixes.
   Capacity makes this worse, not better.
5. Multiregime pretraining: at small ≤ 0.002 (sign consistently right: rg_z-fixed ≥ rg_z-curriculum > original
   on every multiregime cell); at medium 0.012–0.022 on soft_gate multiclass for rg_z-curriculum vs original,
   plus less loss on the binary/persistent cells → 0.006 pooled.
6. Medium vs small: the medium trades K=1 binary calibration (worse than small) for multiregime multiclass skill.

### accuracy gain, large (z-blind / z-exposed)

| cell | original (z-blind) | rg_z-curriculum (z-blind) | original (z-exposed) | rg_z-curriculum (z-exposed) |
|---|---|---|---|---|
| all cells | +0.0418 | +0.0423 | — | +0.0423 |
| binary, ratio 0.1 | +0.0010 | +0.0010 | — | +0.0009 |
| binary, ratio 0.3 | +0.0169 | +0.0170 | — | +0.0166 |
| binary, ratio 0.5 | +0.0805 | +0.0802 | — | +0.0803 |
| 3 classes | +0.0524 | +0.0537 | — | +0.0535 |
| 4 classes | +0.0512 | +0.0523 | — | +0.0522 |
| 5 classes | +0.0489 | +0.0496 | — | +0.0500 |
| K=1 | +0.0409 | +0.0405 | — | +0.0401 |
| K≥2 shared | +0.0460 | +0.0459 | — | +0.0455 |
| K≥2 multiregime | +0.0383 | +0.0401 | — | +0.0407 |
| multiregime, binary | +0.0137 | +0.0133 | — | +0.0134 |
| multiregime, multiclass | +0.0630 | +0.0669 | — | +0.0680 |
| multiregime, soft_gate | +0.0658 | +0.0697 | — | +0.0712 |
| multiregime, persistent | +0.0109 | +0.0105 | — | +0.0103 |

## 5. BeyondArena (real data; same tasks as the old study, so directly comparable)

Pooled, final checkpoints (excess CE / accuracy gain / macro AUC):

| model | all (36) | binary (27) | multi (9) |
|---|---|---|---|
| nat-original-small | −0.147 / +0.079 / 0.805 | −0.115 / +0.066 / 0.794 | −0.246 / +0.116 / 0.838 |
| nat-rg_z-fixed-small | −0.150 / +0.081 / 0.804 | −0.117 / +0.069 / 0.792 | −0.249 / +0.118 / 0.839 |
| nat-rg_z-curriculum-small | −0.148 / +0.080 / 0.802 | −0.117 / +0.068 / 0.791 | −0.239 / +0.115 / 0.837 |
| nat-original-medium | −0.212 / +0.111 / 0.845 | −0.145 / +0.086 / 0.825 | −0.412 / +0.186 / 0.903 |
| nat-rg_z-fixed-medium | −0.210 / +0.110 / 0.843 | −0.144 / +0.086 / 0.824 | −0.408 / +0.184 / 0.902 |
| nat-rg_z-curriculum-medium | −0.209 / +0.111 / 0.843 | −0.142 / +0.085 / 0.823 | −0.407 / +0.187 / 0.902 |
| **nat-original-large** | **−0.224 / +0.120 / 0.852** | −0.153 / +0.094 / 0.833 | −0.437 / +0.200 / 0.908 |
| nat-rg_z-fixed-large | −0.221 / +0.120 / 0.850 | −0.150 / +0.093 / 0.832 | −0.436 / +0.202 / 0.905 |
| nat-rg_z-curriculum-large | −0.220 / +0.119 / 0.850 | −0.148 / +0.092 / 0.832 | −0.437 / +0.199 / 0.906 |
| rf (conventional best) | −0.225 / +0.127 / 0.852 | −0.158 / +0.097 / 0.836 | −0.425 / +0.217 / 0.901 |
| TabICL v2 | −0.273 / +0.141 / 0.872 | −0.194 / +0.112 / 0.854 | −0.509 / +0.230 / 0.929 |
| old original-small | +0.084 / −0.068 / 0.753 | −0.011 / +0.001 / 0.770 | +0.369 / −0.275 / 0.700 |
| old r_z-fixed-medium (best old) | −0.071 / +0.080 / 0.816 | −0.074 / +0.077 / 0.798 | −0.062 / +0.089 / 0.868 |
| old original-medium | −0.050 / +0.033 / 0.828 | −0.039 / +0.010 / 0.812 | −0.084 / +0.104 / 0.878 |
| TabPFN v3 | −0.267 / +0.140 / 0.871 | −0.188 / +0.110 / 0.852 | −0.503 / +0.228 / 0.929 |

Cross-model rank (42 models, 35 tasks, Temporal excluded; CE / acc / AUC): **nat-original-medium 9.8 / 10.9 /
15.8** — best of all our runs on every metric (old best CE rank 19.3 = r_z-fixed-medium; old original-medium
25.2 / 28.6 / 18.5), ahead of logreg (13.4) and catboost (17.3 CE), just behind rf (9.0 / 9.7 / 13.3), behind the
published models (tabicl-v2 3.5, tabpfn-v3 4.3). nat small runs 16.4–16.7 / 20.6–21.9 / 28.7–29.4.

Medium vs small on real data: −0.065 pooled excess CE (−0.031 binary, −0.166 multiclass), +0.033 accuracy gain,
+0.040 AUC; nat-original-medium beats the class prior on 25/27 binary (ties on heart_disease_va_long_beach
+0.000, parkinsons +0.004) and 9/9 multiclass datasets, and fixes both native-small failures (thyroid_discordant
−0.014, jm1 −0.039). It is within 0.05 CE of TabPFN v3 on 20/36 datasets; the big remaining gaps are the
small multiclass sets (ecoli −0.88 vs −1.04, maternal_health −0.41 vs −0.70, website_phishing −0.49 vs −0.68) and
homeq (−0.21 vs −0.47). On AUC it is 0.017 above the old original-medium and 0.026 below TabPFN v3.

Per-dataset (excess CE, 35+1 tasks): the native small runs beat the class prior on 25–26/27 binary and 9/9
multiclass datasets (old original-small 11/27 and 4/9; old original-large 16/27 and 4/9); the three skewed
multiclass datasets that broke every old model — cardiotocography (0.78 majority, old +0.39…+0.87),
hepatitis_c (0.88, old +1.5 small / +0.08 large), ghanas_indigenous (0.93, old +0.6…+1.3) — are now −0.07…−0.25.
Remaining native failures: thyroid_discordant (0.98 majority, +0.03…+0.08) and jm1 (nat-original only, +0.007).
On AUC the native small runs beat the old small (+0.02 binary, +0.14 multi) but are below the old
medium/large (−0.02 / −0.04): the prior fix bought calibration; discrimination is a capacity effect.
Full per-dataset CE and AUC tables: see the session tables (to be added to `multiregime_v4_tables.md` as T14).

Multiregime prior on real data at small: within 0.003 CE of original (rg_z-fixed best on every metric except
AUC). At medium (ranks over 44 models × 35 tasks, CE / acc / AUC): nat-rg_z-fixed-medium 12.0 / 14.3 / 17.6,
nat-original-medium 12.3 / 13.1 / 18.1, nat-rg_z-curriculum-medium 12.5 / 14.7 / 19.9 — the three within 0.003
pooled excess CE and ±0.02 per dataset. At large: **nat-original-large 10.8 / 10.5 / 15.0**,
nat-rg_z-curriculum-large 11.3 / 10.4 / 15.5 vs rf 10.6 / 11.2 / 14.5 — our large runs match the best conventional
model on likelihood (−0.224 / −0.220 vs rf −0.225) and AUC (0.852 / 0.850 vs 0.852), beat it on multiclass
(−0.437 vs −0.425 CE, AUC 0.907 vs 0.901) and on accuracy-gain rank, and trail the published models by ~0.05 CE
and 0.02 AUC. Size scaling on real data is much stronger than on the synthetic bank
(small −0.148 → medium −0.209 → large −0.224 pooled CE; AUC 0.802 → 0.843 → 0.852).

Multiregime at large is neutral to slightly negative, as at medium: original is 0.004 better pooled and 0.005 on
binary, identical on multiclass (−0.4372 vs −0.4374); rg_z-curriculum wins 15/36 datasets (the multiclass and
Grouped ones: dementia, orthopaedic, ecoli, hepatitis_c, horse_colic, heart_disease_hungary) and loses the large
imbalanced binaries (iranian_churn, tour_travels, homeq, churn) and parkinsons (+0.127 vs +0.067). parkinsons
(127 rows, Grouped, 0.76 majority) is the one dataset where capacity hurts: small/medium were negative, both
large runs are positive, and every published model is too (tabpfn-v3 +0.26, tabicl-v2 +0.12) — only rf (−0.118)
and the old runs do well there. nat-rg_z-fixed-large (37424398_58, finished 23 Sep 01:56) completes the trio: −0.221 / +0.120 / 0.850 pooled,
ranks 11.3 / 11.0 / 16.9, i.e. tied with original on CE and accuracy-gain rank and 0.001 ahead of the curriculum,
with the best multiclass accuracy gain of our runs (+0.2016). All three large runs sit within 0.004 pooled CE of
each other and of rf (−0.225), which is the clearest statement of the headline result: **the multiregime prior is
neutral on real data at every size**, while on the synthetic multiregime cells it is worth 0.007−0.009.

## 6. Open items

* Medium: rg_z-fixed finishing; per-checkpoint evals on both banks and BeyondArena for all three mediums queued.
* Large ×3 (A100 / A30), then the same evals.
* References on the native banks (TabPFN v2.2/v2.6/v3, TabICL v1/v1.1/v2, logreg/rf/hgb/xgboost/lightgbm/catboost).
* Binary multiregime cells: a separate diagnosis (is it the pooled-toward-balance class ratio of K-rule mixes,
  or the routing?) — candidate follow-up: per-regime class-ratio audit of the bank cells vs model predictions.
* Old-vs-new side-by-side section in `multiregime_v4_results.md`; T14 per-dataset tables.

## 7. Real-data group-exposure test (merged heart-disease sites)

The three UCI Heart Disease sites in BeyondArena share an identical 13-attribute schema, so merging them gives a
797-row real dataset with a known latent regime (the hospital) — the real-data counterpart of the z-blind /
z-exposed banks. Full tables: `paper/native/heart_sites_group_feature.md` (job 37449374). Headline: in-distribution
the group label is worthless (|Δ excess CE| ≤ 0.0045 for all 19 models; the site is 92 % recoverable from the
features), while under leave-one-site-out shift it helps the in-context models (tabpfn-v2.6 −0.019, tabpfn-v2.2 and
rf −0.014, nat-original-large −0.007) and destroys logreg (+0.074). |Δ AUC| ≤ 0.002 throughout: the effect is
calibration, not ranking — the same conclusion as the synthetic soft_gate (redundant z) versus persistent
(non-recoverable z) cells.

## A1. BeyondArena per-dataset tables (final checkpoints; bold = best in row)

Columns: nat-o / rgF / rgC = native-prior original / rg_z-fixed / rg_z-curriculum, S/M/L = small/medium/large;
old-o-M, old-o-L = old-prior original-medium / original-large; regime G = Grouped, T = Temporal; n = training rows,
d = features, maj = majority-class fraction. Tasks ordered binary-then-multiclass, IID first. All nine native runs
are present (jobs 37409166/84/75, 37409169/87, 37412390, 37409172, 37409190, 37424398).

### A1 — excess_cross_entropy

| task | cls | regime | n | d | maj | nat-o-S | nat-rgF-S | nat-rgC-S | nat-o-M | nat-rgF-M | nat-rgC-M | nat-o-L | nat-rgF-L | nat-rgC-L | old-o-M | old-o-L | tabpfn-v3 | tabicl-v2 | rf | logreg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bad_customer_detection | bin | IID | 1148 | 13 | 0.89 | -0.019 | -0.018 | -0.015 | -0.028 | -0.025 | -0.026 | -0.037 | **-0.038** | -0.036 | +0.436 | +0.348 | -0.038 | -0.037 | -0.030 | -0.025 |
| bank_customer_churn | bin | IID | 6666 | 10 | 0.80 | -0.075 | -0.079 | -0.080 | -0.137 | -0.133 | -0.129 | -0.152 | -0.152 | -0.150 | +0.073 | -0.028 | **-0.180** | -0.176 | -0.156 | -0.069 |
| blood_transfusion | bin | IID | 498 | 4 | 0.76 | -0.074 | -0.072 | **-0.076** | -0.065 | -0.067 | -0.064 | -0.067 | -0.069 | -0.068 | +0.103 | +0.063 | -0.072 | -0.074 | +0.116 | -0.069 |
| churn | bin | IID | 3333 | 19 | 0.86 | -0.076 | -0.075 | -0.067 | -0.136 | -0.131 | -0.133 | -0.166 | -0.161 | -0.161 | +0.046 | -0.004 | **-0.294** | -0.286 | -0.224 | -0.087 |
| credit_approval | bin | IID | 460 | 15 | 0.56 | -0.308 | -0.316 | -0.313 | -0.347 | -0.342 | -0.340 | -0.357 | -0.356 | -0.325 | -0.348 | -0.354 | -0.371 | **-0.374** | -0.346 | -0.334 |
| credit_g | bin | IID | 666 | 20 | 0.70 | -0.075 | -0.072 | -0.075 | -0.099 | -0.092 | -0.091 | -0.103 | -0.102 | -0.100 | +0.071 | +0.028 | -0.117 | **-0.121** | -0.109 | -0.067 |
| early_stage_diabetes_risk_pred | bin | IID | 167 | 16 | 0.69 | -0.297 | -0.288 | -0.285 | -0.349 | -0.351 | -0.337 | -0.349 | -0.346 | -0.347 | -0.314 | -0.317 | -0.404 | **-0.428** | -0.387 | -0.336 |
| ecommerce_shipping | bin | IID | 7332 | 10 | 0.60 | -0.124 | -0.130 | -0.129 | -0.154 | -0.157 | -0.154 | -0.162 | -0.163 | -0.164 | -0.116 | -0.089 | **-0.178** | -0.177 | -0.163 | -0.127 |
| fitness_club | bin | IID | 1000 | 6 | 0.70 | -0.118 | -0.122 | -0.123 | -0.139 | -0.140 | -0.141 | -0.140 | -0.142 | -0.140 | +0.016 | +0.006 | -0.147 | **-0.148** | -0.088 | -0.136 |
| hazelnut_spread_contaminant_de | bin | IID | 1600 | 30 | 0.50 | -0.220 | -0.225 | -0.227 | -0.314 | -0.309 | -0.305 | -0.387 | -0.378 | -0.377 | -0.336 | -0.423 | -0.604 | **-0.610** | -0.407 | -0.421 |
| heart_disease_cleveland | bin | IID | 202 | 13 | 0.54 | -0.261 | -0.259 | -0.259 | -0.287 | -0.289 | -0.287 | -0.268 | -0.264 | -0.272 | -0.276 | -0.268 | **-0.301** | -0.300 | -0.285 | -0.269 |
| heart_disease_hungary | bin | IID | 196 | 13 | 0.64 | -0.241 | -0.245 | -0.242 | -0.286 | -0.288 | **-0.294** | -0.264 | -0.265 | -0.274 | -0.254 | -0.216 | -0.283 | -0.285 | -0.244 | -0.274 |
| heart_disease_va_long_beach | bin | IID | 133 | 13 | 0.74 | -0.028 | -0.026 | **-0.031** | +0.000 | -0.000 | -0.013 | -0.011 | -0.003 | +0.001 | +0.103 | +0.060 | -0.019 | -0.008 | -0.024 | +0.059 |
| heart_failure_followup_surviva | bin | IID | 199 | 12 | 0.68 | -0.208 | -0.197 | -0.181 | -0.261 | -0.263 | -0.264 | -0.252 | -0.238 | -0.254 | -0.222 | -0.222 | -0.264 | **-0.274** | -0.262 | -0.198 |
| heloc | bin | IID | 6972 | 23 | 0.52 | -0.106 | -0.116 | -0.117 | -0.129 | -0.132 | -0.128 | -0.140 | -0.135 | -0.140 | -0.133 | -0.131 | **-0.150** | -0.149 | -0.139 | -0.127 |
| hepatitis_survival_prediction | bin | IID | 103 | 19 | 0.79 | -0.142 | -0.137 | -0.147 | -0.118 | -0.125 | -0.129 | -0.116 | -0.107 | -0.119 | -0.073 | -0.053 | -0.131 | -0.148 | **-0.168** | -0.006 |
| homeq_default_prediction | bin | IID | 3805 | 12 | 0.80 | -0.141 | -0.150 | -0.145 | -0.205 | -0.207 | -0.190 | -0.231 | -0.229 | -0.224 | +0.003 | -0.040 | **-0.468** | -0.432 | -0.293 | -0.220 |
| indian_liver_patient_dataset | bin | IID | 388 | 10 | 0.71 | -0.059 | -0.064 | -0.065 | -0.075 | -0.078 | -0.081 | -0.079 | -0.082 | -0.083 | +0.089 | +0.105 | -0.098 | **-0.100** | -0.079 | -0.081 |
| iranian_churn | bin | IID | 1900 | 13 | 0.84 | -0.183 | -0.190 | -0.185 | -0.230 | -0.223 | -0.220 | -0.263 | -0.257 | -0.252 | -0.049 | -0.074 | **-0.386** | -0.385 | -0.310 | -0.210 |
| jm1 | bin | IID | 7256 | 21 | 0.81 | +0.007 | -0.019 | -0.010 | -0.039 | -0.038 | -0.037 | -0.045 | -0.046 | -0.047 | +0.060 | +0.028 | **-0.078** | -0.076 | -0.034 | -0.044 |
| ljubljana_breast_cancer | bin | IID | 190 | 9 | 0.70 | -0.043 | -0.045 | -0.048 | -0.050 | -0.052 | -0.049 | -0.051 | -0.049 | -0.055 | +0.060 | +0.014 | -0.055 | **-0.057** | +0.025 | -0.022 |
| marketing_campaign | bin | IID | 1493 | 25 | 0.85 | -0.079 | -0.074 | -0.073 | -0.124 | -0.112 | -0.113 | -0.151 | -0.141 | -0.143 | +0.045 | -0.003 | -0.171 | **-0.189** | -0.142 | -0.134 |
| seismic_bumps | bin | IID | 1722 | 15 | 0.93 | -0.022 | -0.022 | -0.023 | -0.030 | -0.029 | -0.030 | -0.031 | **-0.033** | -0.032 | +0.044 | +0.008 | -0.032 | -0.032 | -0.001 | -0.025 |
| south_africa_coronary_heart_di | bin | IID | 308 | 9 | 0.65 | -0.083 | -0.079 | -0.081 | -0.090 | -0.091 | -0.091 | -0.086 | -0.087 | -0.087 | -0.013 | -0.033 | -0.098 | -0.103 | -0.069 | **-0.105** |
| thyroid_discordant | bin | IID | 2474 | 26 | 0.98 | +0.082 | +0.043 | +0.034 | -0.014 | -0.011 | -0.010 | -0.025 | -0.026 | -0.023 | +0.019 | +0.031 | -0.053 | **-0.054** | -0.044 | -0.010 |
| tour_travels_churn | bin | IID | 636 | 6 | 0.77 | -0.123 | -0.120 | -0.118 | -0.211 | -0.201 | -0.192 | -0.263 | -0.259 | -0.252 | +0.005 | -0.081 | **-0.346** | -0.341 | -0.294 | -0.134 |
| parkinsons_biomedical_voice_me | bin | G | 127 | 23 | 0.76 | -0.075 | -0.070 | -0.076 | +0.004 | -0.004 | +0.009 | +0.067 | +0.089 | +0.127 | -0.089 | +0.027 | +0.260 | +0.116 | **-0.118** | +0.041 |
| biomechanical_orthopaedic_pred | 3cls | IID | 206 | 6 | 0.48 | -0.406 | -0.409 | -0.407 | -0.627 | -0.637 | -0.641 | -0.669 | -0.673 | -0.674 | -0.539 | -0.565 | -0.701 | **-0.701** | -0.667 | -0.680 |
| ecoli_proteins | 3cls | IID | 218 | 6 | 0.44 | -0.563 | -0.560 | -0.548 | -0.879 | -0.839 | -0.855 | -0.892 | -0.895 | -0.898 | -0.497 | -0.524 | -1.038 | **-1.040** | -0.993 | -1.010 |
| hepatitis_c_prediction | 3cls | IID | 405 | 12 | 0.88 | -0.168 | -0.187 | -0.158 | -0.341 | -0.333 | -0.330 | -0.355 | -0.370 | -0.362 | -0.012 | +0.083 | -0.385 | **-0.385** | -0.326 | -0.285 |
| horse_colic_survival | 3cls | IID | 229 | 20 | 0.63 | -0.036 | -0.017 | -0.013 | **-0.205** | -0.201 | -0.202 | -0.191 | -0.191 | -0.198 | +0.114 | +0.188 | -0.189 | -0.204 | -0.183 | +0.091 |
| maternal_health_risk | 3cls | IID | 676 | 6 | 0.40 | -0.256 | -0.246 | -0.239 | -0.412 | -0.414 | -0.419 | -0.487 | -0.483 | -0.480 | -0.286 | -0.434 | -0.701 | **-0.709** | -0.639 | -0.298 |
| website_phishing | 3cls | IID | 902 | 9 | 0.52 | -0.272 | -0.270 | -0.271 | -0.489 | -0.473 | -0.474 | -0.554 | -0.551 | -0.542 | -0.235 | -0.199 | **-0.684** | -0.681 | -0.593 | -0.396 |
| cardiotocography | 3cls | G | 1254 | 22 | 0.78 | -0.212 | -0.224 | -0.209 | -0.359 | -0.369 | -0.352 | -0.403 | -0.404 | -0.399 | +0.230 | +0.871 | -0.439 | **-0.451** | -0.436 | -0.334 |
| dementia_prediction | 3cls | G | 250 | 8 | 0.56 | -0.228 | -0.249 | -0.236 | -0.282 | -0.303 | -0.293 | -0.289 | -0.289 | -0.297 | -0.137 | +0.256 | -0.251 | -0.269 | -0.253 | **-0.337** |
| ghanas_indigenous_intel | 3cls | T | 9933 | 10 | 0.93 | -0.072 | -0.079 | -0.068 | -0.115 | -0.105 | -0.099 | -0.095 | -0.069 | -0.088 | +0.610 | +0.697 | **-0.142** | -0.139 | +0.267 | -0.081 |

### A1 — accuracy_gain

| task | cls | regime | n | d | maj | nat-o-S | nat-rgF-S | nat-rgC-S | nat-o-M | nat-rgF-M | nat-rgC-M | nat-o-L | nat-rgF-L | nat-rgC-L | old-o-M | old-o-L | tabpfn-v3 | tabicl-v2 | rf | logreg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bad_customer_detection | bin | IID | 1148 | 13 | 0.89 | **+0.000** | **+0.000** | **+0.000** | **+0.000** | **+0.000** | **+0.000** | -0.000 | **+0.000** | **+0.000** | -0.555 | -0.531 | -0.001 | -0.001 | -0.001 | -0.001 |
| bank_customer_churn | bin | IID | 6666 | 10 | 0.80 | +0.011 | +0.011 | +0.011 | +0.054 | +0.048 | +0.050 | +0.058 | +0.057 | +0.058 | -0.082 | +0.014 | **+0.071** | +0.069 | +0.062 | +0.011 |
| blood_transfusion | bin | IID | 498 | 4 | 0.76 | +0.019 | +0.009 | +0.021 | +0.011 | +0.014 | +0.016 | +0.023 | **+0.034** | +0.033 | -0.182 | -0.066 | +0.025 | +0.029 | -0.018 | +0.011 |
| churn | bin | IID | 3333 | 19 | 0.86 | -0.006 | -0.005 | -0.004 | +0.024 | +0.022 | +0.024 | +0.048 | +0.045 | +0.044 | +0.000 | +0.002 | **+0.114** | +0.111 | +0.098 | +0.007 |
| credit_approval | bin | IID | 460 | 15 | 0.56 | +0.287 | +0.289 | +0.285 | +0.305 | +0.300 | +0.303 | +0.315 | +0.315 | +0.286 | +0.310 | +0.310 | +0.315 | +0.315 | **+0.322** | +0.307 |
| credit_g | bin | IID | 666 | 20 | 0.70 | +0.016 | +0.016 | +0.020 | +0.040 | +0.034 | +0.037 | +0.050 | +0.049 | +0.051 | -0.129 | -0.073 | +0.061 | **+0.061** | +0.054 | +0.018 |
| early_stage_diabetes_risk_pred | bin | IID | 167 | 16 | 0.69 | +0.180 | +0.172 | +0.170 | +0.181 | +0.178 | +0.169 | +0.198 | +0.191 | +0.191 | +0.174 | +0.180 | +0.215 | **+0.231** | +0.231 | +0.175 |
| ecommerce_shipping | bin | IID | 7332 | 10 | 0.60 | +0.062 | +0.069 | +0.073 | +0.076 | +0.081 | +0.077 | +0.076 | +0.076 | +0.083 | +0.071 | +0.075 | **+0.089** | +0.088 | +0.070 | +0.042 |
| fitness_club | bin | IID | 1000 | 6 | 0.70 | +0.076 | +0.075 | +0.076 | +0.073 | +0.073 | +0.077 | +0.074 | +0.076 | +0.078 | -0.005 | -0.028 | **+0.084** | +0.084 | +0.056 | +0.079 |
| hazelnut_spread_contaminant_de | bin | IID | 1600 | 30 | 0.50 | +0.282 | +0.292 | +0.294 | +0.337 | +0.340 | +0.334 | +0.369 | +0.364 | +0.362 | +0.347 | +0.385 | +0.468 | **+0.470** | +0.393 | +0.389 |
| heart_disease_cleveland | bin | IID | 202 | 13 | 0.54 | +0.265 | +0.258 | +0.254 | +0.281 | +0.279 | +0.281 | +0.273 | +0.273 | +0.283 | +0.276 | +0.267 | **+0.292** | +0.292 | +0.280 | +0.281 |
| heart_disease_hungary | bin | IID | 196 | 13 | 0.64 | +0.186 | +0.187 | +0.181 | **+0.204** | +0.199 | +0.198 | +0.202 | +0.203 | **+0.204** | +0.194 | +0.174 | +0.197 | +0.195 | +0.178 | +0.201 |
| heart_disease_va_long_beach | bin | IID | 133 | 13 | 0.74 | +0.007 | +0.003 | **+0.012** | -0.000 | +0.003 | +0.005 | +0.011 | +0.008 | +0.006 | -0.168 | -0.073 | +0.001 | -0.002 | +0.006 | -0.014 |
| heart_failure_followup_surviva | bin | IID | 199 | 12 | 0.68 | +0.136 | +0.123 | +0.103 | +0.153 | +0.153 | +0.161 | +0.147 | +0.145 | +0.154 | +0.149 | +0.134 | +0.165 | **+0.172** | +0.165 | +0.140 |
| heloc | bin | IID | 6972 | 23 | 0.52 | +0.177 | +0.183 | +0.184 | +0.195 | +0.197 | +0.195 | +0.198 | +0.199 | +0.199 | +0.194 | +0.192 | **+0.210** | +0.210 | +0.201 | +0.194 |
| hepatitis_survival_prediction | bin | IID | 103 | 19 | 0.79 | +0.034 | +0.036 | +0.038 | +0.049 | +0.048 | +0.043 | +0.047 | +0.043 | +0.042 | +0.019 | +0.004 | +0.041 | +0.043 | **+0.056** | +0.033 |
| homeq_default_prediction | bin | IID | 3805 | 12 | 0.80 | +0.044 | +0.052 | +0.052 | +0.078 | +0.077 | +0.067 | +0.096 | +0.101 | +0.099 | +0.026 | +0.040 | **+0.189** | +0.174 | +0.111 | +0.091 |
| indian_liver_patient_dataset | bin | IID | 388 | 10 | 0.71 | -0.030 | -0.027 | -0.025 | +0.001 | -0.003 | -0.001 | +0.001 | -0.008 | -0.006 | -0.182 | -0.160 | **+0.005** | +0.004 | -0.003 | +0.003 |
| iranian_churn | bin | IID | 1900 | 13 | 0.84 | +0.044 | +0.049 | +0.048 | +0.068 | +0.071 | +0.067 | +0.082 | +0.076 | +0.071 | -0.002 | +0.025 | **+0.138** | +0.138 | +0.112 | +0.054 |
| jm1 | bin | IID | 7256 | 21 | 0.81 | -0.050 | -0.013 | -0.031 | +0.001 | +0.002 | +0.002 | +0.004 | +0.003 | +0.005 | -0.046 | -0.006 | **+0.017** | +0.014 | +0.011 | +0.006 |
| ljubljana_breast_cancer | bin | IID | 190 | 9 | 0.70 | +0.005 | +0.004 | +0.005 | +0.020 | +0.024 | +0.018 | +0.034 | +0.036 | **+0.039** | -0.094 | -0.021 | +0.033 | +0.036 | +0.005 | +0.017 |
| marketing_campaign | bin | IID | 1493 | 25 | 0.85 | +0.009 | +0.013 | +0.014 | +0.024 | +0.020 | +0.020 | +0.038 | +0.033 | +0.035 | -0.039 | -0.011 | **+0.048** | +0.046 | +0.030 | +0.033 |
| seismic_bumps | bin | IID | 1722 | 15 | 0.93 | **+0.000** | **+0.000** | **+0.000** | **+0.000** | **+0.000** | -0.000 | **+0.000** | **+0.000** | **+0.000** | -0.012 | -0.001 | -0.000 | -0.000 | -0.002 | -0.003 |
| south_africa_coronary_heart_di | bin | IID | 308 | 9 | 0.65 | +0.053 | +0.045 | +0.045 | +0.058 | +0.057 | +0.059 | +0.053 | +0.058 | +0.058 | -0.028 | -0.005 | +0.064 | **+0.071** | +0.037 | +0.069 |
| thyroid_discordant | bin | IID | 2474 | 26 | 0.98 | -0.044 | -0.000 | +0.000 | +0.000 | +0.000 | -0.001 | +0.000 | +0.000 | +0.000 | -0.001 | +0.000 | +0.006 | **+0.007** | +0.002 | -0.002 |
| tour_travels_churn | bin | IID | 636 | 6 | 0.77 | +0.044 | +0.041 | +0.033 | +0.058 | +0.059 | +0.054 | +0.086 | +0.086 | +0.082 | -0.011 | +0.041 | +0.120 | **+0.120** | +0.112 | +0.054 |
| parkinsons_biomedical_voice_me | bin | G | 127 | 23 | 0.76 | -0.025 | -0.034 | -0.021 | +0.039 | +0.043 | +0.044 | +0.041 | **+0.045** | +0.038 | +0.033 | -0.023 | +0.010 | +0.034 | +0.039 | +0.024 |
| biomechanical_orthopaedic_pred | 3cls | IID | 206 | 6 | 0.48 | +0.233 | +0.237 | +0.229 | +0.351 | +0.342 | +0.350 | +0.364 | +0.371 | +0.367 | +0.348 | +0.341 | **+0.377** | +0.375 | +0.360 | +0.368 |
| ecoli_proteins | 3cls | IID | 218 | 6 | 0.44 | +0.210 | +0.202 | +0.204 | +0.388 | +0.362 | +0.375 | +0.396 | +0.392 | +0.392 | +0.263 | +0.323 | +0.444 | +0.443 | +0.440 | **+0.447** |
| hepatitis_c_prediction | 3cls | IID | 405 | 12 | 0.88 | +0.006 | +0.009 | +0.003 | +0.059 | +0.057 | +0.056 | +0.065 | +0.071 | +0.068 | +0.002 | +0.016 | +0.080 | **+0.081** | +0.059 | +0.063 |
| horse_colic_survival | 3cls | IID | 229 | 20 | 0.63 | +0.002 | -0.000 | +0.004 | +0.054 | +0.057 | +0.061 | +0.052 | +0.060 | **+0.065** | -0.105 | -0.161 | +0.060 | +0.058 | +0.053 | +0.019 |
| maternal_health_risk | 3cls | IID | 676 | 6 | 0.40 | +0.202 | +0.192 | +0.194 | +0.280 | +0.283 | +0.294 | +0.326 | +0.322 | +0.321 | +0.281 | +0.320 | +0.445 | **+0.450** | +0.425 | +0.216 |
| website_phishing | 3cls | IID | 902 | 9 | 0.52 | +0.247 | +0.251 | +0.245 | +0.313 | +0.314 | +0.310 | +0.344 | +0.346 | +0.339 | +0.125 | +0.173 | **+0.397** | +0.397 | +0.371 | +0.305 |
| cardiotocography | 3cls | G | 1254 | 22 | 0.78 | +0.023 | +0.025 | +0.024 | +0.077 | +0.085 | +0.071 | +0.101 | +0.103 | +0.096 | -0.095 | -0.088 | +0.129 | **+0.132** | +0.126 | +0.090 |
| dementia_prediction | 3cls | G | 250 | 8 | 0.56 | +0.123 | +0.148 | +0.137 | **+0.171** | +0.167 | +0.167 | +0.155 | +0.158 | +0.155 | +0.125 | +0.091 | +0.127 | +0.135 | +0.121 | +0.170 |
| ghanas_indigenous_intel | 3cls | T | 9933 | 10 | 0.93 | **+0.000** | **+0.000** | **+0.000** | -0.015 | -0.011 | -0.003 | -0.008 | -0.009 | -0.014 | -0.013 | -0.117 | -0.003 | -0.003 | -0.004 | -0.026 |

### A1 — macro_ovr_auc

| task | cls | regime | n | d | maj | nat-o-S | nat-rgF-S | nat-rgC-S | nat-o-M | nat-rgF-M | nat-rgC-M | nat-o-L | nat-rgF-L | nat-rgC-L | old-o-M | old-o-L | tabpfn-v3 | tabicl-v2 | rf | logreg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bad_customer_detection | bin | IID | 1148 | 13 | 0.89 | 0.693 | 0.689 | 0.678 | 0.736 | 0.734 | 0.735 | 0.745 | **0.751** | 0.745 | 0.704 | 0.703 | 0.747 | 0.746 | 0.728 | 0.705 |
| bank_customer_churn | bin | IID | 6666 | 10 | 0.80 | 0.787 | 0.791 | 0.790 | 0.843 | 0.841 | 0.836 | 0.848 | 0.850 | 0.847 | 0.839 | 0.841 | **0.875** | 0.871 | 0.851 | 0.753 |
| blood_transfusion | bin | IID | 498 | 4 | 0.76 | 0.755 | 0.757 | **0.758** | 0.739 | 0.741 | 0.735 | 0.746 | 0.742 | 0.739 | 0.746 | 0.740 | 0.753 | 0.753 | 0.682 | 0.751 |
| churn | bin | IID | 3333 | 19 | 0.86 | 0.841 | 0.833 | 0.825 | 0.880 | 0.875 | 0.878 | 0.894 | 0.890 | 0.891 | 0.861 | 0.879 | **0.933** | 0.931 | 0.916 | 0.823 |
| credit_approval | bin | IID | 460 | 15 | 0.56 | 0.907 | 0.910 | 0.908 | 0.925 | 0.924 | 0.923 | 0.933 | 0.933 | 0.931 | 0.927 | 0.934 | 0.937 | **0.939** | 0.934 | 0.920 |
| credit_g | bin | IID | 666 | 20 | 0.70 | 0.743 | 0.741 | 0.745 | 0.777 | 0.774 | 0.771 | 0.780 | 0.778 | 0.778 | 0.771 | 0.778 | 0.793 | **0.796** | 0.784 | 0.729 |
| early_stage_diabetes_risk_pred | bin | IID | 167 | 16 | 0.69 | 0.942 | 0.932 | 0.930 | 0.952 | 0.953 | 0.947 | 0.956 | 0.952 | 0.953 | 0.953 | 0.955 | 0.969 | **0.975** | 0.973 | 0.945 |
| ecommerce_shipping | bin | IID | 7332 | 10 | 0.60 | 0.731 | 0.734 | 0.734 | 0.740 | 0.740 | 0.740 | 0.743 | 0.738 | 0.739 | 0.738 | 0.742 | **0.747** | 0.746 | 0.743 | 0.721 |
| fitness_club | bin | IID | 1000 | 6 | 0.70 | 0.786 | 0.793 | 0.793 | 0.813 | 0.816 | 0.817 | 0.813 | 0.816 | 0.815 | 0.813 | 0.814 | **0.821** | 0.820 | 0.777 | 0.815 |
| hazelnut_spread_contaminant_de | bin | IID | 1600 | 30 | 0.50 | 0.877 | 0.877 | 0.877 | 0.913 | 0.910 | 0.910 | 0.943 | 0.939 | 0.940 | 0.924 | 0.955 | 0.994 | **0.995** | 0.958 | 0.953 |
| heart_disease_cleveland | bin | IID | 202 | 13 | 0.54 | 0.886 | 0.882 | 0.882 | 0.901 | 0.901 | 0.897 | 0.894 | 0.892 | 0.897 | 0.902 | 0.894 | **0.908** | 0.908 | 0.902 | 0.892 |
| heart_disease_hungary | bin | IID | 196 | 13 | 0.64 | 0.884 | 0.884 | 0.884 | 0.911 | 0.912 | **0.913** | 0.904 | 0.902 | 0.906 | 0.905 | 0.898 | 0.910 | 0.911 | 0.884 | 0.905 |
| heart_disease_va_long_beach | bin | IID | 133 | 13 | 0.74 | 0.682 | 0.679 | 0.684 | 0.659 | 0.652 | 0.663 | 0.656 | 0.658 | 0.650 | **0.685** | 0.660 | 0.676 | 0.665 | 0.674 | 0.656 |
| heart_failure_followup_surviva | bin | IID | 199 | 12 | 0.68 | 0.883 | 0.875 | 0.864 | 0.910 | 0.911 | 0.910 | 0.905 | 0.900 | 0.905 | 0.904 | 0.904 | 0.909 | **0.916** | 0.911 | 0.864 |
| heloc | bin | IID | 6972 | 23 | 0.52 | 0.762 | 0.773 | 0.774 | 0.784 | 0.785 | 0.781 | 0.792 | 0.790 | 0.792 | 0.789 | 0.793 | **0.802** | 0.802 | 0.792 | 0.781 |
| hepatitis_survival_prediction | bin | IID | 103 | 19 | 0.79 | 0.869 | 0.857 | 0.862 | 0.863 | 0.868 | 0.857 | 0.867 | 0.866 | 0.867 | 0.879 | 0.864 | 0.857 | 0.868 | **0.880** | 0.817 |
| homeq_default_prediction | bin | IID | 3805 | 12 | 0.80 | 0.846 | 0.858 | 0.853 | 0.905 | 0.909 | 0.898 | 0.924 | 0.919 | 0.919 | 0.908 | 0.915 | **0.999** | 0.996 | 0.961 | 0.902 |
| indian_liver_patient_dataset | bin | IID | 388 | 10 | 0.71 | 0.714 | 0.715 | 0.718 | 0.731 | 0.735 | 0.737 | 0.733 | 0.734 | 0.737 | 0.721 | 0.727 | 0.758 | **0.764** | 0.744 | 0.745 |
| iranian_churn | bin | IID | 1900 | 13 | 0.84 | 0.915 | 0.914 | 0.914 | 0.954 | 0.950 | 0.948 | 0.972 | 0.971 | 0.970 | 0.938 | 0.949 | **0.997** | 0.996 | 0.983 | 0.932 |
| jm1 | bin | IID | 7256 | 21 | 0.81 | 0.691 | 0.688 | 0.689 | 0.706 | 0.708 | 0.706 | 0.716 | 0.717 | 0.717 | 0.706 | 0.689 | **0.773** | 0.770 | 0.749 | 0.713 |
| ljubljana_breast_cancer | bin | IID | 190 | 9 | 0.70 | 0.715 | 0.717 | 0.720 | 0.709 | 0.711 | 0.709 | 0.702 | 0.707 | 0.713 | **0.721** | 0.716 | 0.710 | 0.712 | 0.661 | 0.685 |
| marketing_campaign | bin | IID | 1493 | 25 | 0.85 | 0.811 | 0.807 | 0.803 | 0.874 | 0.864 | 0.864 | 0.893 | 0.884 | 0.887 | 0.834 | 0.879 | 0.922 | **0.926** | 0.888 | 0.883 |
| seismic_bumps | bin | IID | 1722 | 15 | 0.93 | 0.746 | 0.749 | 0.749 | 0.770 | 0.766 | 0.770 | 0.772 | 0.777 | 0.775 | 0.742 | 0.692 | **0.782** | 0.779 | 0.747 | 0.754 |
| south_africa_coronary_heart_di | bin | IID | 308 | 9 | 0.65 | 0.746 | 0.741 | 0.743 | 0.758 | 0.759 | 0.759 | 0.750 | 0.751 | 0.752 | 0.755 | 0.760 | 0.765 | 0.770 | 0.727 | **0.771** |
| thyroid_discordant | bin | IID | 2474 | 26 | 0.98 | 0.580 | 0.547 | 0.543 | 0.857 | 0.837 | 0.843 | 0.906 | 0.911 | 0.907 | 0.521 | 0.414 | 0.987 | **0.989** | 0.976 | 0.822 |
| tour_travels_churn | bin | IID | 636 | 6 | 0.77 | 0.828 | 0.825 | 0.824 | 0.897 | 0.891 | 0.887 | 0.931 | 0.930 | 0.928 | 0.884 | 0.919 | **0.963** | 0.960 | 0.947 | 0.835 |
| parkinsons_biomedical_voice_me | bin | G | 127 | 23 | 0.76 | 0.818 | 0.809 | 0.811 | 0.778 | 0.776 | 0.783 | 0.772 | 0.755 | 0.754 | **0.840** | 0.767 | 0.708 | 0.737 | 0.802 | 0.754 |
| biomechanical_orthopaedic_pred | 3cls | IID | 206 | 6 | 0.48 | 0.874 | 0.879 | 0.883 | 0.945 | 0.946 | 0.947 | 0.953 | 0.954 | 0.954 | 0.935 | 0.940 | **0.960** | 0.960 | 0.950 | 0.957 |
| ecoli_proteins | 3cls | IID | 218 | 6 | 0.44 | 0.859 | 0.865 | 0.866 | 0.955 | 0.943 | 0.950 | 0.954 | 0.953 | 0.953 | 0.889 | 0.911 | 0.973 | **0.974** | 0.971 | 0.969 |
| hepatitis_c_prediction | 3cls | IID | 405 | 12 | 0.88 | 0.917 | 0.921 | 0.914 | 0.972 | 0.974 | 0.972 | 0.977 | 0.980 | 0.978 | 0.962 | 0.960 | 0.985 | **0.985** | 0.973 | 0.951 |
| horse_colic_survival | 3cls | IID | 229 | 20 | 0.63 | 0.719 | 0.718 | 0.724 | 0.805 | 0.800 | 0.803 | 0.797 | 0.802 | **0.807** | 0.784 | 0.769 | 0.798 | 0.806 | 0.794 | 0.743 |
| maternal_health_risk | 3cls | IID | 676 | 6 | 0.40 | 0.781 | 0.775 | 0.773 | 0.850 | 0.852 | 0.853 | 0.881 | 0.881 | 0.880 | 0.851 | 0.880 | 0.957 | **0.958** | 0.945 | 0.795 |
| website_phishing | 3cls | IID | 902 | 9 | 0.52 | 0.833 | 0.830 | 0.829 | 0.924 | 0.921 | 0.917 | 0.951 | 0.949 | 0.951 | 0.886 | 0.873 | **0.982** | 0.982 | 0.972 | 0.859 |
| cardiotocography | 3cls | G | 1254 | 22 | 0.78 | 0.880 | 0.881 | 0.868 | 0.948 | 0.950 | 0.948 | 0.960 | 0.960 | 0.958 | 0.943 | 0.929 | 0.974 | **0.976** | 0.970 | 0.946 |
| dementia_prediction | 3cls | G | 250 | 8 | 0.56 | 0.825 | 0.822 | 0.823 | 0.851 | 0.853 | 0.849 | 0.843 | 0.841 | 0.845 | 0.825 | 0.831 | 0.829 | 0.835 | 0.819 | **0.867** |
| ghanas_indigenous_intel | 3cls | T | 9933 | 10 | 0.93 | 0.857 | 0.858 | 0.852 | 0.875 | 0.877 | 0.873 | 0.853 | 0.825 | 0.828 | 0.827 | 0.735 | **0.905** | 0.888 | 0.718 | 0.780 |
