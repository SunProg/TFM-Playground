# Multi-regime v4: NanoTabPFN prior-scale study — results so far

Date: 2026-09-20 (rev. 6). Status: 15 baseline pretraining tasks complete
(10k steps each); Exp 2 complete; **Exp 1 complete for all 12 runs** (4
families × 3 sizes); Exp 3 complete; TabPFN v2.2/v2.6/v3 and TabICL
v1/v1.1/v2 baselines complete on the pre-repair bank; repaired-bank
re-evaluation of the 15 baseline runs complete (14/15 with final test —
g_z-curriculum-large's last file is being redone), repaired baseline re-runs
in progress (see §7); **BeyondArena real-data evaluation of all 38 models
complete (§6b: 27 runs, 5 published models, 6 conventional ML baselines,
with CD diagrams / Δ heatmaps / pairwise scatters / sign tests)**; mixed
r_z+g_z prior runs (12, blind and z-exposed) submitted (§7).

**Bank repair (2026-09-17 12:35).** The z-blind bank's 252 (validation) /
224 (test) episodes whose support features were all constant were replaced
(`*.h5.pre-repair-20260917T123924` backups kept; cell metadata unchanged).
Every number in this note that was computed on the z-blind bank is from the
pre-repair bank unless marked *repaired*. The affected episodes are 0.8 % of
the bank and were exactly the ones where TabPFN/TabICL could not fit and fell
back to the class prior, so pooled numbers move by < 0.005; a full
re-evaluation of every checkpoint and every baseline on the repaired bank is
finished for the 15 baseline runs (`runs_eval/repaired_bank/`, §3b) and is
being redone for the TabPFN/TabICL references (`runs_eval/tabpfn_baselines_repaired/`,
now on scratch): a disk-quota outage on hpc_home (21:37 on 09-18 → 09:39 on
09-19, caused by the project venv growing to 8.9 GB after a re-sync) zeroed
the first attempt's outputs.
Recomputed prior baselines on the repaired bank: validation prior CE 0.9615 /
majority acc 0.4681, test 0.9614 / 0.4680.
All tables are collected, numbered (T1–T13) and formatted uniformly in
`paper/multiregime_v4_tables.md`; figures for the real-data evaluation are in
`figures/beyondarena_v4/` (summary figures) and
`figures/beyondarena_v4/comparison/` (statistical comparison figures).
All numbers below are from the KCL CREATE cluster checkout
`/cephfs/volumes/hpc_home/k23139234/17f71989-c3b3-41d8-9315-37daa2fead54/repo/TFM-Playground-uncertainty`
(`${PROJECT_DIR}` below); figures in the local worktree under `figures/`.

## 1. Design

**Question.** Can a NanoTabPFN pretrained on the TabICL `mix_scm` prior, with
or without multi-regime episodes mixed in, learn to classify tables whose
label rule switches between 2–4 latent regimes (v4 `multiregime` rule_mode),
and does exposure to such episodes during pretraining help?

**Prior families (5).** `original` (plain mix_scm dump, no multiregime
episodes); `r_z-fixed`, `r_z-curriculum`, `g_z-fixed`, `g_z-curriculum`.
`r_z`: the SCM emits k candidate label columns and the regime selects one;
`g_z`: one score r(X) with regime-specific quantile maps. `fixed`: constant
multiregime ratio from step 0; `curriculum`: ratio ramps up over training.
Training data are pre-dumped episodes (`data/multiregime_v4/tabicl_mix_scm_paired/37277254/{original,r_z-shared,r_z-multiregime,g_z-shared,g_z-multiregime}.h5`,
100k episodes each, 1024 rows, 2–12 features, `expose_z_probability = 0`).

**Model sizes (3).** small / medium / large NanoTabPFN (feature-tokenised, so
input width is free). Same seed (2402) and init per size across families.

**Schedule.** 10 000 steps, warmup 500, cosine LR, checkpoint every 2000
(2000/4000/6000/8000/10000 + final), own-prior validation and shared v4-bank
validation every 500 steps, test bank once at the end on the final model.
SLURM arrays 37283915 (small, A30), 37283921 (medium, A30), 37283995 (large,
A30; tasks 2/5/8 = original / r_z-fixed / r_z-curriculum died at step ~5500 on a
disk-quota error and were resumed from checkpoint-004000 on A100 as
37289726/37289727/37289728). Run dirs:
`/scratch/users/k23139234/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale/<job>/<family>-<size>/seed-2402/`.

**Shared v4 bank** (`data/multiregime_v4/tabicl_mix_scm_evaluation/37277969/{validation,test}.h5`,
32 256 episodes per split). Factors: support_size {64,128,256,512} ×
num_features {2..24} × num_regimes {1,2,3,4} × num_classes {2..5} ×
class_ratio {0.1,0.3,0.5} × task_family {soft_gate, persistent} ×
mechanism_mode {r_z, g_z} × rule_mode {shared, multiregime}. `is_causal` is
sampled per episode and recorded as `effective_is_causal`; `expose_z = False`.
The bank is identical for every task, so v4 numbers are comparable across
families and sizes. The **own-prior validation loss** is *not* comparable
across families: `validate()` draws 16 fixed batches from the plain TabICL
mix_scm prior (seed+100000), which contains no regimes at all.

## 2. Metrics ("fair" = class-ratio aware)

Binary cells vary class_ratio (0.1/0.3/0.5) and multiclass cells vary
num_classes, so raw CE and accuracy have different floors per cell (the
class-prior predictor already scores CE 0.33 at ratio 0.1 vs 0.69 at 0.5).
Per episode we therefore report, against the trivial predictor that outputs
the support set's empirical class frequencies:

* **excess CE** = CE − class-prior CE. Lower is better; ≥ 0 means no better
  than the class prior.
* **accuracy gain** = accuracy − majority-class accuracy. Higher is better;
  ≤ 0 means no better than majority vote.
* **AUC** (OvR macro, renormalised over valid classes) where reports carry it.

Bank-wide prior baselines: validation prior CE 0.9560 / majority acc 0.4715;
test 0.9564 / 0.4712 (`runs_eval/prior_baselines/{validation,test}_prior.json`,
`scripts/compute_v4_bank_prior_baselines.py`). All pooling below is
episode-weighted (never a mean of cell means). Panels: `mr1..mr4` =
rule_mode multiregime at num_regimes 1..4; `shared` = rule_mode shared (all
regimes); `all` = every episode.

## 3. Headline result: final-checkpoint TEST scores

Excess CE (lower better) / accuracy gain (higher better), test bank.

| model | mr1 | mr2 | mr3 | mr4 | shared |
|---|---|---|---|---|---|
| **TabPFN v2.2** | −0.110 / +0.089 | **−0.035 / +0.039** | **−0.028 / +0.034** | **−0.020 / +0.027** | −0.104 / +0.087 |
| TabPFN v2.6 | −0.103 / +0.089 | −0.029 / +0.041 | −0.024 / +0.036 | −0.017 / +0.029 | −0.097 / +0.087 |
| TabPFN v3 | −0.109 / +0.087 | −0.036 / +0.038 | −0.029 / +0.033 | −0.021 / +0.025 | −0.103 / +0.085 |
| TabICL v1 (val) | −0.102 / +0.090 | −0.031 / +0.043 | −0.025 / +0.035 | −0.016 / +0.027 | −0.100 / +0.089 |
| TabICL v1.1 (val) | −0.072 / +0.085 | −0.009 / +0.041 | −0.015 / +0.037 | −0.006 / +0.029 | −0.078 / +0.085 |
| TabICL v2 (val) | −0.088 / +0.093 | −0.013 / +0.045 | −0.020 / +0.040 | −0.011 / +0.032 | −0.090 / +0.091 |
| original-small | −0.004 / +0.020 | +0.047 / −0.028 | +0.047 / −0.039 | +0.048 / −0.039 | +0.001 / +0.014 |
| r_z-fixed-small | +0.003 / +0.017 | +0.045 / −0.026 | +0.047 / −0.034 | +0.049 / −0.035 | +0.008 / +0.012 |
| r_z-curriculum-small | +0.010 / +0.006 | +0.053 / −0.036 | +0.054 / −0.044 | +0.056 / −0.046 | +0.015 / +0.002 |
| g_z-fixed-small | −0.009 / +0.025 | +0.028 / −0.016 | +0.030 / −0.024 | +0.032 / −0.026 | −0.005 / +0.020 |
| g_z-curriculum-small | −0.014 / +0.027 | +0.029 / −0.015 | +0.031 / −0.024 | +0.032 / −0.025 | −0.009 / +0.022 |
| original-medium | −0.018 / +0.033 | +0.058 / −0.027 | +0.055 / −0.047 | +0.061 / −0.053 | −0.014 / +0.027 |
| r_z-fixed-medium | −0.039 / +0.078 | +0.028 / **+0.031** | +0.021 / **+0.029** | +0.028 / **+0.023** | −0.036 / +0.075 |
| r_z-curriculum-medium | −0.019 / +0.048 | +0.051 / −0.005 | +0.045 / −0.017 | +0.052 / −0.021 | −0.015 / +0.042 |
| g_z-fixed-medium | −0.029 / +0.060 | +0.037 / +0.011 | +0.033 / −0.002 | +0.039 / −0.007 | −0.024 / +0.055 |
| g_z-curriculum-medium | −0.030 / +0.057 | +0.039 / +0.007 | +0.030 / −0.004 | +0.037 / −0.009 | −0.027 / +0.052 |
| original-large | −0.063 / +0.066 | +0.016 / +0.008 | +0.014 / −0.002 | +0.020 / −0.005 | −0.059 / +0.062 |
| r_z-fixed-large | −0.041 / +0.065 | +0.028 / +0.013 | +0.024 / +0.005 | +0.031 / +0.000 | −0.037 / +0.062 |
| r_z-curriculum-large | −0.063 / +0.072 | **+0.007** / +0.024 | **+0.006** / +0.017 | **+0.013** / +0.012 | −0.057 / +0.070 |
| g_z-fixed-large | −0.041 / +0.071 | +0.027 / +0.023 | +0.026 / +0.017 | +0.033 / +0.012 | −0.036 / +0.069 |
| g_z-curriculum-large | −0.045 / +0.058 | +0.025 / +0.006 | +0.027 / −0.005 | +0.034 / −0.011 | −0.038 / +0.054 |
| mr-only r_z-medium (Exp 2) | −0.018 / +0.074 | +0.033 / **+0.033** | +0.031 / **+0.029** | +0.037 / **+0.025** | −0.013 / +0.072 |
| mr-only r_z-small (Exp 2) | +0.062 / −0.105 | +0.086 / −0.142 | +0.090 / −0.150 | +0.092 / −0.153 | +0.068 / −0.111 |
| z-trained r_z-fixed-medium, z-exposed bank (Exp 1) | −0.047 / +0.079 | +0.018 / +0.034 | +0.012 / +0.032 | +0.019 / +0.026 | −0.044 / +0.076 |

Validation values track test within ±0.005 everywhere (TabICL rows are
validation; their test runs finished after the bank repair and are in the
repaired-bank set).

Findings:

1. **The multi-rule cells are learnable beyond the class prior** — TabPFN
   v2.2 reaches excess CE −0.035 / accuracy gain +0.039 at 2 regimes (about
   half of its single-rule gain, shrinking to −0.020 / +0.027 at 4 regimes).
2. **No NanoTabPFN run beats the class prior in likelihood on any multi-rule
   cell** (mr2–mr4 excess CE ≥ +0.006 for every model, every size). The best
   is r_z-curriculum-large at +0.007 / +0.006 / +0.013.
3. **Several medium/large r_z runs do beat majority vote in accuracy** on
   multi-rule cells: r_z-fixed-medium +0.031 / +0.029 / +0.023 and Exp 2's
   mr-only medium +0.033 / +0.029 / +0.025 are ~75–85 % of TabPFN's accuracy
   gain, while their excess CE is +0.02…+0.04. Their AUC is ~0.01 below TabPFN
   (r_z-fixed-large mr2/3/4: 0.552/0.542/0.532 vs 0.567/0.554/0.542). So on
   those runs the CE failure is largely **miscalibration** (over-confident
   wrong predictions), not absence of discriminative signal.
4. **Small models and `original`-medium end below majority vote** on multi-rule
   cells (accuracy gain −0.03…−0.05): for them both decision and calibration
   are bad.
5. Single-rule cells (mr1, shared): the gain over the prior grows with model
   size (small ≈ −0.01, medium ≈ −0.03, large ≈ −0.06 excess CE) but the
   large models still reach only ~55 % of TabPFN's −0.11.
6. `original` (never sees multiregime episodes) is as good as or better than
   the r_z/g_z families on single-rule cells at every size; the r_z families
   buy a small multi-rule accuracy gain at medium/large, g_z families do not.
   The per-family differences on multi-rule cells are within what one seed per
   cell can resolve (r_z-fixed-medium is the best medium run but r_z-fixed-large
   is not the best large run).
7. **The released TFMs agree on accuracy but not on likelihood.** All six
   references are within ±0.005 accuracy gain of each other on every panel
   (TabICL v2 nominally best: +0.045 / +0.040 / +0.032 at 2/3/4 regimes), but
   TabICL v1.1 / v2 have 3× smaller excess-CE gains on multi-rule cells than
   TabPFN v2.2 / v3 / TabICL v1 (−0.009…−0.013 vs −0.031…−0.039 at mr2). The
   accuracy-vs-CE split we see in our medium models is present, more mildly,
   in released models too.
8. **Best-own-val checkpoint vs final** (test, all 15 tasks): selecting the
   checkpoint by own-prior validation loss helps small/medium on multi-rule
   cells (e.g. original-small mr2 +0.011 at step 2000 vs +0.047 final;
   r_z-fixed-small +0.009 vs +0.045) but hurts large (original-large +0.033 at
   step 4000 vs +0.016 final; r_z-curriculum-large +0.034 vs +0.007) — the
   own-prior loss is not a usable model-selection signal for the v4 bank;
   select on the v4 validation bank directly.

## 3b. Repaired bank: final-checkpoint TEST scores (re-evaluated checkpoints)

Same layout as §3, from `runs_eval/repaired_bank/` (repaired bank, repaired
prior baselines: test prior CE 0.9614 / majority acc 0.4680). Figures:
`figures/repaired_bank/` (+ `multiclass_only/`, `regime_x_classes/`), with
the checkpoint-resolution (every 2000 steps) histories, step-0 init, and
test ◆.

| model | mr1 | mr2 | mr3 | mr4 | shared |
|---|---|---|---|---|---|
| original-small | −0.004 / +0.019 | +0.046 / −0.026 | +0.045 / −0.035 | +0.047 / −0.037 | +0.001 / +0.014 |
| r_z-fixed-small | +0.004 / +0.017 | +0.046 / −0.026 | +0.048 / −0.034 | +0.050 / −0.035 | +0.008 / +0.012 |
| r_z-curriculum-small | +0.011 / +0.006 | +0.053 / −0.036 | +0.055 / −0.044 | +0.057 / −0.046 | +0.015 / +0.002 |
| g_z-fixed-small | −0.009 / +0.024 | +0.029 / −0.015 | +0.031 / −0.024 | +0.033 / −0.025 | −0.005 / +0.020 |
| g_z-curriculum-small | −0.014 / +0.027 | +0.029 / −0.015 | +0.031 / −0.025 | +0.033 / −0.025 | −0.009 / +0.022 |
| original-medium | −0.018 / +0.033 | +0.056 / −0.026 | +0.053 / −0.043 | +0.060 / −0.051 | −0.014 / +0.027 |
| r_z-fixed-medium | −0.039 / +0.078 | +0.030 / +0.031 | +0.023 / +0.030 | +0.029 / +0.024 | −0.036 / +0.076 |
| r_z-curriculum-medium | −0.019 / +0.048 | +0.053 / −0.005 | +0.047 / −0.017 | +0.053 / −0.021 | −0.015 / +0.043 |
| g_z-fixed-medium | −0.029 / +0.061 | +0.039 / +0.011 | +0.034 / −0.002 | +0.040 / −0.007 | −0.024 / +0.056 |
| g_z-curriculum-medium | −0.030 / +0.058 | +0.040 / +0.007 | +0.032 / −0.004 | +0.038 / −0.009 | −0.027 / +0.053 |
| original-large | −0.063 / +0.066 | +0.014 / +0.011 | +0.012 / +0.002 | +0.019 / −0.004 | −0.059 / +0.063 |
| r_z-fixed-large | −0.042 / +0.065 | +0.030 / +0.013 | +0.026 / +0.005 | +0.033 / +0.000 | −0.038 / +0.062 |
| r_z-curriculum-large | −0.063 / +0.073 | +0.008 / +0.024 | +0.008 / +0.017 | +0.015 / +0.013 | −0.058 / +0.070 |
| g_z-fixed-large | −0.041 / +0.071 | +0.029 / +0.023 | +0.028 / +0.017 | +0.034 / +0.013 | −0.036 / +0.069 |
| g_z-curriculum-large | (final test being redone) | | | | |

Every value is within 0.003 of its pre-repair counterpart in §3; no
conclusion changes. The pre-repair and repaired numbers are kept in separate
directories and figure sets and are never mixed in one plot.

## 4. Training dynamics (pooled over all cells; see §4b for what drives it)

Figures: `figures/per_size_plots/fair_metrics/v4_{excess_cross_entropy,accuracy_gain}_per_regime_{small,medium,large}.png`
(6 panels each, all families, TabPFN reference lines, test ★, step-0 init
annotation); raw-CE versions in `figures/per_size_plots/`.

* **Step 0 → 500.** The random init scores excess CE ≈ +0.59…+0.64 on every
  panel (the medium init predicts a minority class: accuracy gain −0.39).
  After 500 steps every run is at +0.01…+0.02 on all panels — i.e. it has
  learned the class prior. Essentially all subsequent "learning" on multi-rule
  cells is movement around that level.
* **Small models: dip then rise.** All 5 families reach their best multi-rule
  excess CE at step 2000–3500 (+0.003…+0.014 at mr2), then rise monotonically
  to +0.03…+0.05 by 10k, and accuracy gain crosses below zero around step
  4000–6000. Best own-val checkpoints: original 2000, r_z-fixed 4000,
  r_z-curriculum 6000, g_z-fixed 6000, g_z-curriculum 10000.
* **Medium: no dip, flat/rising from the first checkpoint** for original and
  the curriculum runs (mr2 excess CE +0.015 at 500 → +0.05…+0.06 at 10k);
  r_z-fixed-medium is the exception, trending down after step 5500 to +0.026.
* **Large: late recovery.** original-large and r_z-curriculum-large decline
  from +0.045 (step 2000) to +0.013 / +0.003 (mr2) by 10k and were still
  improving when the cosine schedule ended. r_z-fixed-large has the deepest
  early collapse of the large runs (mr2 +0.071 at 2000, accuracy gain −0.027)
  and recovers to +0.025 by 8000.
* **soft_gate ≈ persistent**, and the pattern is the same for r_z and g_z
  mechanisms; worst at support_size 64.
* **Conditioning on support size** (`figures/per_size_plots/fair_metrics/support_{64,128,256,512}/`):
  on shared-rule cells the gain grows strongly with support (large: −0.016 at
  64 → −0.079 at 512), on multi-rule cells the best value is ≈ 0 at every
  support size for every model size — more context rows do not help identify
  the regime.

## 4b. Where the positive excess CE comes from: binary, imbalanced cells

Figures: `figures/per_size_plots/fair_metrics/regime_x_classes/v4_{,r_z_,g_z_}{excess_cross_entropy,accuracy_gain}_regime_x_classes_{small,medium,large}.png`
(4×4 grid, rows num_regimes 1–4, cols num_classes 2–5) and
`figures/per_size_plots/fair_metrics/multiclass_only/` (6-panel plots restricted
to num_classes ≥ 3).

* **On 3–5-class cells every model beats the class prior on every regime
  count, from step ~1500 on, monotonically** — no dip-then-rise. Large final
  (validation, num_classes ≥ 3): mr1 −0.13, mr2 −0.03…−0.04, mr3 −0.04, mr4
  −0.03, shared −0.13 vs TabPFN v2.2 −0.154 / −0.059 / −0.050 / −0.034 /
  −0.153. At 3–4 regimes the r_z/g_z large models reach TabPFN's level (mr4
  −0.030 vs −0.034); on accuracy they match or beat it (mr4c3: ours +0.06,
  TabPFN +0.054). On these cells the multiregime families also separate from
  `original`: original-large plateaus at −0.03 / −0.02 on mr3/mr4 while all four
  r_z/g_z families reach −0.04 / −0.03.
* **All of the positive excess CE is in the 2-class column**, and within it,
  in the imbalanced cells. Binary cells by class_ratio (validation, excess CE /
  accuracy gain):

  | model | rule | ratio 0.1 | ratio 0.3 | ratio 0.5 |
  |---|---|---|---|---|
  | TabPFN v2.2 | single | −0.035 / +0.005 | −0.071 / +0.035 | −0.079 / +0.112 |
  | TabPFN v2.2 | mr≥2 | −0.003 / 0.000 | −0.019 / +0.008 | −0.020 / +0.043 |
  | original-medium | single | +0.200 / −0.113 | +0.059 / −0.202 | −0.057 / +0.113 |
  | original-medium | mr≥2 | +0.262 / −0.152 | +0.119 / −0.298 | −0.006 / +0.049 |
  | r_z-fixed-medium | mr≥2 | +0.163 / −0.005 | +0.043 / −0.053 | −0.007 / +0.051 |
  | original-large | mr≥2 | +0.108 / −0.034 | +0.074 / −0.185 | −0.012 / +0.051 |

  At ratio 0.5 our models equal TabPFN (accuracy gain +0.113 vs +0.112
  single-rule, +0.049 vs +0.043 multi-rule). At ratio 0.1 / 0.3 they are far
  worse than the class prior *even on single-rule tables* (CE 0.52 where the
  prior scores 0.32; accuracy 11–30 points below majority vote), i.e. they
  systematically over-predict the minority class. Multi-rule cells make it
  worse only because the label is harder to predict, so the mis-set prior
  dominates. TabPFN itself gains almost nothing on imbalanced multi-rule
  binary cells (−0.003 at ratio 0.1) — those cells are near the ceiling.
* So the earlier "multi-rule collapse" is two separable things: (i) a
  **class-imbalance calibration defect** that exists on single-rule tables too
  and grows over training (the models become more confident in a wrong class
  prior), and (ii) a genuine but modest multi-rule discrimination gap to
  TabPFN on balanced/multiclass cells (~60 % of its excess-CE gain at 2
  regimes, ~100 % at 4). The training dumps' class-ratio distribution is the
  first thing to check for (i).

## 5. Experiments on the cause

### Exp 1 — expose z (regime routing score) as an input column

Bank: `/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_evaluation_expose_z/{validation,test}.h5`
(same 32 256 episodes; `X` is 25 wide, `input_width = num_features + 1`, the
continuous routing score sits at a random `z_column_index`; verified that
within an episode the regimes occupy disjoint z ranges). Prior baselines are
identical to the z-blind bank (`runs_eval/prior_baselines/expose_z_*`).
Training dumps with `expose_z_probability = 1.0`: `/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_paired_expose_z/r_z-{shared,multiregime}.h5`.
Runs: `runs_exp/expose_z/r_z-fixed-{small,medium}/seed-2402` (jobs 37295132
done, 37295134 running, step ~6500).

* z-trained small on the z-exposed bank: mr2 excess CE +0.009 at step 4000
  (best), +0.043 at 10k; test +0.044 / accuracy gain −0.026 — **the same
  curve as z-blind training** (`figures/per_size_plots/fair_metrics/*` family
  `r_z-fixed-exposez`).
* **All 12 runs complete** (4 families × 3 sizes; jobs 37295132/4, 37326476,
  37322052; run dirs `runs_exp/expose_z/<family>-<size>/seed-2402`, the
  large and the later small/medium ones physically on
  `/scratch/users/k23139234/tfm_runs/expose_z/`). Final-checkpoint test,
  **blind → exposed** (z-blind bank vs z-exposed bank; excess CE at mr2 / mr4,
  accuracy gain at mr2):

  | family | small | medium | large |
  |---|---|---|---|
  | r_z-fixed | +0.045/+0.049 (−0.026) → +0.044/+0.047 (−0.026) | +0.028/+0.028 (+0.031) → **+0.018/+0.019 (+0.034)** | +0.028/+0.031 (+0.013) → **+0.044/+0.054 (−0.012)** |
  | r_z-curriculum | +0.053/+0.056 (−0.036) → +0.035/+0.038 (−0.021) | +0.051/+0.052 (−0.005) → +0.035/+0.034 (+0.011) | +0.007/+0.013 (+0.024) → +0.008/+0.017 (+0.022) |
  | g_z-fixed | +0.028/+0.032 (−0.016) → +0.035/+0.039 (−0.019) | +0.037/+0.039 (+0.011) → +0.050/+0.057 (−0.011) | +0.027/+0.033 (+0.023) → +0.033/+0.043 (+0.009) |
  | g_z-curriculum | +0.029/+0.032 (−0.015) → +0.033/+0.036 (−0.020) | +0.039/+0.037 (+0.007) → +0.044/+0.045 (+0.011) | +0.025/+0.034 (+0.006) → +0.023/+0.032 (+0.009) |

  Restricted to num_classes ≥ 3 (excess CE mr2 / mr4, accuracy gain mr2),
  blind → exposed:

  | family | small | medium | large |
  |---|---|---|---|
  | r_z-fixed | −0.010/−0.005 (+0.044) → −0.009/−0.005 (+0.043) | −0.006/−0.017 (+0.063) → −0.010/−0.019 (+0.064) | −0.030/−0.030 (+0.071) → −0.036/−0.031 (+0.073) |
  | r_z-curriculum | −0.009/−0.005 (+0.045) → −0.009/−0.005 (+0.044) | −0.005/−0.020 (+0.062) → −0.008/−0.021 (+0.064) | −0.034/−0.030 (+0.071) → −0.036/−0.031 (+0.072) |
  | g_z-fixed | −0.010/−0.005 (+0.043) → −0.010/−0.005 (+0.043) | −0.012/−0.021 (+0.062) → −0.020/−0.023 (+0.064) | −0.033/−0.029 (+0.071) → −0.036/−0.032 (+0.074) |
  | g_z-curriculum | −0.012/−0.007 (+0.046) → −0.010/−0.006 (+0.045) | −0.005/−0.020 (+0.062) → −0.009/−0.021 (+0.063) | −0.032/−0.029 (+0.071) → −0.032/−0.031 (+0.073) |
  | *TabPFN v2.2* | | −0.059/−0.034 (+0.059) | |

  Reading, now with the full set:
  1. **On multiclass cells exposing z is a uniform, tiny help**: every one of
     the 12 exposed runs is within 0.008 excess CE and 0.003 accuracy gain of
     its blind twin, always on the better side. With a column that identifies
     the regime perfectly, the models gain ≤ 0.008 nats while the CE gap to
     TabPFN at 2 regimes is 0.02–0.05. **Decision quality is unaffected**, and
     medium/large models already beat TabPFN on multiclass multi-rule
     accuracy (large: +0.073 vs +0.059 at mr2, +0.060 vs +0.041 at mr4).
  2. **Pooled (binary-dominated) numbers move both ways**: the r_z families
     improve at small/medium (r_z-curriculum-small −0.018 excess CE at mr2,
     r_z-fixed-medium −0.010), the g_z families and **every large run except
     r_z-curriculum get worse** — r_z-fixed-large's pooled mr2–4 accuracy gain
     flips from ≈ 0 to −0.012…−0.038. The z column makes the imbalanced-binary
     calibration problem worse where it was already worst (large models,
     binary, class_ratio 0.1/0.3).
  3. Best-own-val vs final: on multiclass cells the final checkpoint beats the
     best-own-val checkpoint in every run (blind and exposed alike); the
     "early stopping helps" effect in §3 is purely a binary-cell effect.
* z-exposed-bank per-checkpoint evaluations of all 12 runs (37322053,
  37326477, 37310278/9) are queued/complete; they only duplicate the
  in-training curves in the shared `runs_eval/expose_z_bank/` tree. The z-exposed bank has **not** been repaired (it still contains
  the 252/224 degenerate episodes), so Exp 1 numbers are internally consistent
  but not strictly like-for-like with the repaired-bank set.
* **All 15 z-blind models scored on the z-exposed bank** (job 37302389,
  `runs_eval/expose_z_bank/`): every panel moves by ≤ 0.005 in both metrics,
  as often worse as better. E.g. original-medium mr2 +0.056 → +0.056,
  r_z-fixed-large +0.025 → +0.026, r_z-curriculum-large +0.003 → +0.004. The
  models ignore a column that perfectly determines the rule.

### Exp 2 — multiregime-only prior (ratio 1.0)

Runs: `runs_exp/multiregime_only/r_z-multiregime-{small,medium}/seed-2402`
(jobs 37294349 / 37294353, both complete). Figures:
`figures/exp2_multiregime_only/`.

* **Medium: two phases.** Steps 1500–6000 it sits in a bad basin — accuracy
  gain −0.09…−0.16 on *every* cell (systematically predicts a minority
  class), excess CE +0.05…+0.12. Between 6000 and 7500 it transitions sharply
  and by 10k reaches accuracy gain +0.033 / +0.029 / +0.025 at 2/3/4 regimes
  (equal to r_z-fixed-medium, ~80 % of TabPFN) with excess CE +0.03 and
  **still falling at 10k**. Single-rule cells end at the same level as the
  mixed-prior runs (mr1 −0.018 / +0.074), so nothing is lost by dropping the
  shared episodes.
* **Small: never escapes** the basin (accuracy gain −0.18 at 1500, −0.10…−0.15
  at 10k).
* Conclusion: the mixed prior is not the cause of the collapse; the
  multiregime-only prior can produce the same (weak) discriminator but needs
  ~7k steps to leave a bad solution, and is the run most worth extending.

### Exp 3 — evaluate checkpoints on the training dumps vs held-out dumps

Held-out dumps: `data/multiregime_v4/tabicl_mix_scm_paired_heldout/heldout_{r_z,g_z}-{multiregime,shared}.h5`
(4096 episodes each, seed 7777, job 37294334). Evaluation job 37294335;
outputs `runs_eval/nanotabpfn_tabicl_mix_scm_prior_scale/<job>/<task>/seed-2402/v4_dump_eval/`.

* Training-dump vs held-out CE/AUC agree within ~0.01 at every checkpoint
  (r_z-fixed-medium @10k, regimes 2/3/4: AUC 0.596/0.569/0.554 train vs
  0.582/0.559/0.545 held-out) — **no memorisation**.
* `original-medium`, never trained on multiregime episodes, scores the same on
  the held-out multiregime dumps (AUC 0.583/0.560/0.544) as r_z-fixed-medium.
  The 100k multiregime training episodes add nothing the model retains.
* Absolute AUC on the dumps' multi-rule episodes is only 0.53–0.60 for every
  model (regime 4 → 2), i.e. the exploitable signal is weak even where it is
  learned.

## 6. Interpretation

* Pooled over all cells the models never beat the class prior on multi-rule
  episodes, but §4b shows this is dominated by the imbalanced binary cells,
  where the models mis-set the class prior even on single-rule tables. On
  balanced / multiclass cells the multi-rule capability is largely there
  (TabPFN-level at 3–4 regimes in excess CE, matching it in accuracy).
* The remaining multi-rule gap is not in the data (TabPFN extracts more from
  the same bank; no memorisation; held-out = train), not in the prior mixture
  (Exp 2), and only marginally in regime observability (Exp 1: ~0.01 CE at
  medium, nothing at small; z-blind models ignore the column entirely).
* Where multi-rule signal is learned (medium/large r_z, mr-only medium) it
  shows up as accuracy/AUC, not likelihood — the probabilities are
  over-confident on multi-rule episodes. A per-regime temperature sweep on the
  saved checkpoints would separate the calibration gap from the
  discrimination gap cheaply.
* Size and steps are the visible levers: large > medium > small on multi-rule
  cells at every checkpoint, large and mr-only-medium were still improving at
  10k, and TabPFN (far larger, far longer trained) reaches −0.035.
* Priority order for the next round: (1) the class-imbalance calibration
  defect (check the training dumps' class-ratio distribution; it affects
  single-rule tables and is the largest single term in every pooled number);
  (2) select checkpoints on the v4 validation bank, not the own-prior loss;
  (3) longer budget for large / mr-only-medium.

## 6b. Real data: BeyondArena v4 evaluation

Jobs 37371347 (27 runs + 5 published), 37389908 / 37389920 / 37391280
(conventional ML baselines; array, one task per model;
`scripts/slurm/evaluate_beyondarena_v4_array_a30.sbatch`, evaluator
`tfmplayground/experiments/evaluate_multiregime_v4_beyondarena.py`).
Protocol: official Data Foundry containers, every official outer fold, the
complete training fold as context; eligible tasks = classification, 2–5
classes, ≤ 10 000 rows, ≤ 30 raw features, no text / high-cardinality
categoricals → **36 tasks** (27 binary, 9 multiclass; regimes IID 32 /
Grouped 3 / Temporal 1; rows small 19 / medium 11 / large 2; features small 14
/ medium 15 / large 7). Metrics per task (folds averaged, tasks equal weight):
excess CE, accuracy gain, macro OvR AUC — the same fair metrics as the
synthetic bank. Models: the 15 baseline runs and the 12 z-exposed runs
(`final_checkpoint.pth`; the evaluator also scores `best_own_val`), and
TabPFN v2.2 / v2.6 / v3, TabICL v1 / v2, and six conventional ML baselines
fitted per fold on the same preprocessed features with default
hyper-parameters (`logreg` = StandardScaler + LogisticRegression C=1; `rf` =
RandomForest 500 trees; `hgb` = sklearn HistGradientBoosting; `xgboost` 300
trees / lr 0.1 / depth 6; `lightgbm` 300 trees / lr 0.1; `catboost` 300
iterations / lr 0.1 / depth 6). Note: the first xgboost run silently failed
on 16 binary tasks because `XGBClassifier.fit` rewrites `self.objective` to
`multi:softprob` after a multiclass task and the evaluator reused one
estimator — fixed by building a fresh estimator per fold (37391280). Results:
`/scratch/users/k23139234/tfm_eval/beyondarena-v4/<mode>-<job>_<i>/`
(`rankings.csv`, `fold_metrics.csv`, …); tables from
`scripts/summarize_beyondarena_v4.py` (`--rank`, `--exclude-prefix`,
`--exclude-bucket`, `--sort-by`).

Figures: `figures/beyondarena_v4/beyondarena_{scores,rank,rank_scatter}_<condition>.png`
and `beyondarena_heatmap_rank.png` (`scripts/plot_beyondarena_v4.py`);
`figures/beyondarena_v4/comparison/` (`scripts/plot_beyondarena_v4_comparison.py`,
all 38 models, Temporal excluded): `cd_<metric>.png` (Friedman + Nemenyi
critical-difference diagrams), `delta_heatmap_<metric>_vs_<ref>.png`
(per-dataset Δ vs `tabpfn-v3`, vs the same-size `original`, vs the z-blind
twin, and vs each conventional baseline; Δ oriented so positive = better),
`pairwise_<metric>.png` / `pairwise_all_<metric>_vs_<ref>.png` (per-dataset
scatter vs a reference, every run), `delta_dist_<metric>.png` (Δ distribution
per run with sign tests), `main_table_<metric>.md`. Full tables T8–T13 in
`multiregime_v4_tables.md`.

### Scores (final checkpoints; excess CE / accuracy gain / AUC)

| model | all (36) | binary (27) | multi (9) |
|---|---|---|---|
| TabICL v2 | −0.273 / +0.141 / 0.872 | −0.194 / +0.112 / 0.854 | −0.509 / +0.230 / 0.929 |
| TabPFN v3 | −0.267 / +0.140 / 0.871 | −0.188 / +0.110 / 0.852 | −0.503 / +0.228 / 0.929 |
| TabPFN v2.2 | −0.265 / +0.138 / 0.866 | −0.186 / +0.107 / 0.848 | −0.500 / +0.230 / 0.922 |
| rf | −0.225 / +0.127 / 0.852 | −0.158 / +0.097 / 0.836 | −0.425 / +0.217 / 0.901 |
| logreg | −0.188 / +0.108 / 0.825 | −0.127 / +0.082 / 0.808 | −0.370 / +0.184 / 0.874 |
| catboost | −0.181 / +0.125 / 0.855 | −0.088 / +0.093 / 0.831 | −0.459 / +0.218 / 0.925 |
| xgboost | −0.126 / +0.118 / 0.843 | −0.060 / +0.087 / 0.819 | −0.325 / +0.213 / 0.914 |
| hgb | −0.113 / +0.118 / 0.843 | −0.091 / +0.088 / 0.821 | −0.177 / +0.209 / 0.910 |
| lightgbm | +0.101 / +0.117 / 0.838 | +0.121 / +0.086 / 0.816 | +0.040 / +0.211 / 0.907 |
| r_z-fixed-medium | **−0.071 / +0.080** / 0.816 | −0.074 / +0.077 / 0.798 | −0.062 / +0.089 / 0.868 |
| g_z-fixed-medium | −0.069 / +0.076 / 0.813 | −0.066 / +0.074 / 0.794 | −0.076 / +0.083 / 0.871 |
| g_z-curriculum-medium | −0.065 / +0.053 / 0.823 | −0.063 / +0.039 / 0.807 | −0.069 / +0.097 / 0.869 |
| original-medium | −0.050 / +0.033 / **0.828** | −0.039 / +0.010 / 0.812 | −0.084 / +0.104 / **0.878** |
| r_z-curriculum-large | −0.052 / +0.053 / 0.819 | −0.058 / +0.038 / 0.804 | −0.032 / +0.097 / 0.865 |
| g_z-fixed-large | −0.045 / +0.053 / 0.826 | −0.057 / +0.052 / 0.816 | −0.011 / +0.057 / 0.857 |
| original-large | −0.035 / +0.048 / 0.823 | −0.060 / +0.031 / 0.807 | +0.041 / +0.100 / 0.870 |
| zx-r_z-fixed-medium | −0.074 / +0.057 / 0.818 | −0.072 / +0.054 / 0.799 | −0.082 / +0.064 / 0.874 |
| zx-r_z-fixed-large | −0.016 / +0.044 / 0.828 | −0.023 / +0.031 / 0.817 | +0.004 / +0.081 / 0.861 |
| r_z-fixed-small | +0.031 / −0.011 / 0.771 | +0.006 / −0.001 / 0.775 | +0.109 / −0.040 / 0.759 |
| original-small | +0.084 / −0.068 / 0.753 | −0.011 / +0.001 / 0.770 | +0.369 / −0.275 / 0.700 |

### Cross-model mean rank (38 models, Temporal task excluded, 1 = best)

`all` (35 tasks), rank by excess CE / accuracy gain / AUC (Nemenyi CD = 10.3
at α = 0.05 for k = 38, N = 35; Friedman p < 1e-99 on every metric):

| model | CE | acc | AUC |
|---|---|---|---|
| TabICL v2 | 3.1 | 3.8 | 4.9 |
| TabPFN v3 | 3.8 | 3.9 | 5.9 |
| TabPFN v2.6 | 4.1 | 4.6 | 5.8 |
| TabPFN v2.2 | 4.7 | 4.4 | 6.9 |
| TabICL v1 | 5.1 | 4.9 | 6.8 |
| rf | 8.1 | 8.5 | 12.4 |
| logreg | 11.9 | 12.7 | 21.3 |
| catboost | 15.4 | 9.3 | 15.3 |
| zx-r_z-fixed-medium | 16.2 | 15.3 | 20.7 |
| r_z-fixed-medium | 16.3 | 14.5 | 21.9 |
| g_z-curriculum-medium | 18.5 | 19.4 | 20.7 |
| g_z-fixed-medium | 18.5 | 15.6 | 22.8 |
| hgb | 18.7 | 12.9 | 19.5 |
| zx-r_z-curriculum-medium | 19.1 | 17.9 | 17.5 |
| g_z-fixed-large | 19.3 | 19.6 | 16.8 |
| r_z-curriculum-large | 19.5 | 19.2 | 16.9 |
| xgboost | 20.9 | 12.8 | 19.3 |
| original-large | 20.9 | 21.8 | 17.7 |
| original-medium | 22.2 | 25.9 | 17.6 |
| zx-r_z-fixed-large | 23.2 | 23.2 | 15.1 |
| lightgbm | 27.4 | 13.9 | 21.0 |
| small runs (9) | 23.9–30.3 | 26.9–31.7 | 26.9–32.1 |

Internal ranking of our 27 runs (published models excluded), best of each
size / family per slice, AUC / CE / acc rank:

| slice | best on AUC | best on CE | best on accuracy | best `original` |
|---|---|---|---|---|
| all (35) | zx-r_z-fixed-large 7.5 | zx-r_z-fixed-medium 7.7 | r_z-fixed-medium 5.9 | original-medium AUC 9.7; original-large CE 12.2 |
| binary (27) | zx-r_z-fixed-large 8.1 | r_z-fixed-medium 6.4 | r_z-fixed-medium 4.3 | original-large CE 10.6 |
| multi (8) | zx-r_z-curriculum-large 5.1 | zx-r_z-curriculum-medium 7.8 | r_z-fixed-large 5.4 | original-medium CE 11.2; original-large 17.4 |
| Grouped (3) | zx-r_z-curriculum-medium 4.3 | g_z-curriculum-small 5.3 | zx-r_z-fixed-medium 2.7 | original-medium CE 8.3; original-large 25.7 (last) |
| IID (32) | zx-r_z-fixed-large 6.7 | zx-r_z-fixed-medium 7.7 | r_z-fixed-medium 6.1 | original-large CE 10.9 |

### Findings

1. **All medium and large runs beat the class prior on real tables**
   (excess CE −0.03…−0.07, accuracy gain +0.03…+0.08, AUC 0.81–0.83); all
   small runs are worse than the prior overall, driven by multiclass tasks
   (+0.11…+0.38) — the mirror image of the synthetic bank, where multiclass
   was the easy part and binary the problem.
2. **The published models are far ahead**: ranks 1–5 on every metric and
   every slice, ~8 rank positions clear of our best; in score terms −0.27 vs
   −0.07 excess CE, +0.14 vs +0.08 accuracy gain, 0.87 vs 0.83 AUC. The gap
   is smallest on small binary tables (our best at rank 8 vs 4) and widest
   on medium/large-row tables (14–15 vs 2): the published models exploit
   extra context rows, ours do not. On ≥ 16-feature tables the published
   models themselves rank worse (6–8), narrowing the gap.
3. **Multiregime pretraining transfers to real data.** Every multiregime
   medium run beats `original-medium` on likelihood and decisions (CE rank
   7.7–13.3 vs 13.3, accuracy 5.9–13.4 vs 16.0); at large, 5 of 8 beat
   `original-large` on CE and all 8 on accuracy. The benefit is in decisions
   and calibration, not ranking: `original` runs stay competitive on AUC
   (original-medium 7th of 27).
4. **Size**: medium is the best size on real data for likelihood/decisions
   (top 5 of 27 are all medium); large runs rank rows best (AUC 7.5–9.8) but
   calibrate worse (CE 10.7–14.8); small runs are last on everything.
5. **Which family**: r_z-fixed for binary / IID (CE 6.4, accuracy 4.3);
   r_z-curriculum (blind or z-exposed) for multiclass (top 3 on AUC, 7.8 on
   CE) and grouped splits; the 3 Grouped tasks — the closest real-data
   analogue of multiregime — are the one slice where several of our runs
   rank above the published models on likelihood (g_z-curriculum-small 8.7,
   r_z-curriculum-medium 10.3 vs TabPFN v2.2 10.7) and where every large run
   collapses (CE ranks 19–26; original-large last). Three tasks — direction
   only.
6. **z-exposed training** is a wash on real tables (no z column exists at
   evaluation): the zx twins sit within ±1 rank of their blind versions on
   `all`; zx-r_z-curriculum-medium is clearly better on multiclass (7.8 vs
   11.0 CE), zx-r_z-fixed-large clearly worse on likelihood (14.4 vs 12.3)
   while best on AUC. The zx small runs are the worst models overall. The
   per-dataset `delta_heatmap_*_vs_blind.png` figures show where the
   difference sits: on the ≥ 0.8-majority binary tasks the medium/large zx
   runs are slightly *worse* than their blind twins on excess CE (−0.06 to
   −0.17 on seismic_bumps / bad_customer_detection / churn / marketing_campaign);
   elsewhere |Δ| ≤ 0.05.
7. **Conventional ML baselines sit between the published models and our
   runs, and beat every one of our runs on likelihood.** Random forest
   (default 500 trees) is the strongest: rank 8.1 / 8.5 / 12.4 (CE / acc /
   AUC), −0.225 excess CE, 0.852 AUC — 2.5 CD-units behind TabICL v2 but ahead
   of every NanoTabPFN run on all three metrics (r_z-fixed-medium beats it on
   3/35 tasks for CE, 9/35 for AUC). Logistic regression ranks 11.9 on CE
   (−0.188) — better than all of ours — but only 21.3 on AUC, i.e. at parity
   with our medium/large runs on ranking quality (our large runs beat it on
   20–21/35 tasks, p ≈ 0.3–0.5); catboost is the best conventional model on
   AUC (0.855) and on multiclass (CE rank 6.2, AUC 7.6). hgb / xgboost are
   good on decisions (acc rank 12.8–12.9) but only mid-pack on likelihood
   (18.7 / 20.9), and lightgbm with default settings is *over-confident*:
   positive excess CE (+0.101 pooled, +0.121 on binary), CE rank 27.4 — below
   all our medium/large runs — while its accuracy gain (+0.117) and AUC (0.838)
   are as good as the other boosters. So on real tables our best runs beat
   the untuned boosters on calibration but not on discrimination, and trail
   rf / logreg on both likelihood and (for rf) AUC.
8. **Statistical picture (CD diagrams, T12).** With 38 models and 35 tasks
   the Nemenyi CD is 10.3 rank units, so only three groups are separable at
   α = 0.05: the published models (ranks 3–7) form one clique together with
   rf; a broad middle clique spans rf … our medium/large runs … the boosters;
   and the small runs plus lightgbm (CE) close the tail. No NanoTabPFN run is
   significantly different from any other on `all`; the pretraining effect
   is visible only in paired per-task tests: vs the same-size `original`,
   r_z-fixed-medium wins 25/35 on CE (sign test p = 0.017) but only 9/35 on
   AUC (p = 0.006 the other way); r_z-curriculum-large 19/35 CE (p = 0.74),
   18/35 AUC. Against TabPFN v3 our runs win 1–2/35 on CE and 4–6/35 on AUC
   (p < 1e-4 throughout).

### 6b-native. The native-prior runs on the same 36 tasks (2026-09-23)

The nine native-prior runs (`paper/native/native_prior_results.md`) were scored on the identical protocol, so the
old and new priors are directly comparable on real data. Final checkpoints, 36 tasks, excess CE / accuracy gain /
macro AUC:

| model | all (36) | binary (27) | multi (9) |
|---|---|---|---|
| nat-original-large | −0.224 / +0.120 / 0.852 | −0.153 / +0.094 / 0.833 | −0.437 / +0.200 / 0.908 |
| nat-rg_z-fixed-large | −0.221 / +0.120 / 0.850 | −0.150 / +0.093 / 0.832 | −0.436 / +0.202 / 0.905 |
| nat-rg_z-curriculum-large | −0.220 / +0.119 / 0.850 | −0.148 / +0.092 / 0.832 | −0.437 / +0.199 / 0.906 |
| nat-original-medium | −0.212 / +0.111 / 0.845 | −0.145 / +0.086 / 0.825 | −0.412 / +0.186 / 0.903 |
| nat-rg_z-fixed-medium | −0.210 / +0.110 / 0.843 | −0.144 / +0.086 / 0.824 | −0.408 / +0.184 / 0.902 |
| nat-rg_z-curriculum-medium | −0.209 / +0.111 / 0.843 | −0.142 / +0.085 / 0.823 | −0.407 / +0.187 / 0.902 |
| nat small ×3 | −0.147…−0.150 / +0.079…+0.081 / 0.802…0.805 | −0.115…−0.117 | −0.239…−0.249 |
| **old** original-medium | −0.050 / +0.033 / 0.828 | −0.039 / +0.010 / 0.812 | −0.084 / +0.104 / 0.878 |
| **old** original-large | −0.056 / +0.014 / 0.831 | — | — |
| **old** r_z-fixed-medium (best old) | −0.071 / +0.080 / 0.816 | −0.074 / +0.077 / 0.798 | −0.062 / +0.089 / 0.868 |
| rf | −0.225 / +0.127 / 0.852 | −0.158 / +0.097 / 0.836 | −0.425 / +0.217 / 0.901 |
| tabicl-v2 | −0.273 / +0.141 / 0.872 | −0.194 / +0.112 / 0.854 | −0.509 / +0.230 / 0.929 |

Mean rank over 22 models × 35 tasks (Temporal excluded), sorted by excess CE: tabicl-v2 3.23, tabpfn-v3 4.11,
tabpfn-v2.6 4.34, tabpfn-v2.2 4.86, tabicl-v1 5.17, **nat-rg_z-fixed-large 9.37, nat-original-large 9.51,
rf 9.54, nat-rg_z-curriculum-large 9.60**, nat medium 10.89–11.23, logreg 12.94, nat small 15.49–15.60,
catboost 15.7, xgboost 18.0, old r_z-fixed-medium 17.0, old original-large 18.41, old original-medium 18.99.

Two conclusions:

* **The prior fix is worth ~0.17 pooled excess CE on real data** — every native run beats every old-prior run,
  and the native large runs move from rank ~18 to ~9.5, level with a 500-tree random forest and roughly 0.05
  behind the published foundation models.
* **The multiregime prior remains neutral on real data at every size** (|Δ| ≤ 0.004 pooled vs the same-size
  original, no consistent sign), in contrast to the synthetic multiregime cells where it is worth 0.007–0.009 at
  large. Real tabular datasets in this benchmark carry no exploitable routing structure; see
  `paper/native/heart_sites_group_feature.md` for a real dataset where a latent group *is* present.

Figures: `figures/beyondarena_v4/comparison/{cd,delta_heatmap,pairwise,delta_dist,tables}/` regenerated with all
nine native runs (22-model panel).

## 7. Pending

* Repaired-bank references: TabPFN v2.2/v2.6 (37361542), TabICL v1/v1.1/v2
  (37361543), TabPFN v3 validation shard 3 + merge (37361104/5), v3 test
  shards 2–3 + merge (37326549–51); g_z-curriculum-large final test
  (37361544); mr-only pair (37344402). All outputs now on scratch
  (`runs_eval/tabpfn_baselines_repaired` → `/scratch/users/k23139234/tfm_eval/`).
  When these land: reference lines in `figures/repaired_bank/` and a §3b
  reference block.
* Storage: hpc_home 46.9/50 GB with an 8.9 GB venv; re-sync the venv with
  `UV_CACHE_DIR` on scratch and `UV_LINK_MODE=hardlink` once the queue is
  idle. scratch 191.5/200 GB.
* Decide whether to regenerate the z-exposed bank with the same repair
  (Exp 1 numbers are internally consistent but contain the 252/224 degenerate
  episodes).
* Cheap and informative, not started: per-regime temperature sweep on saved
  logits (calibration vs discrimination); class-ratio histogram of the
  training dumps (§4b hypothesis).
* BeyondArena follow-ups: `best_own_val` vs `final` on real data (both are in
  the result files); mr-only runs are not yet evaluated there (2 array
  tasks); Temporal/Grouped need more tasks before any claim. Conventional
  baselines are untuned defaults; a light HPO (or at least lightgbm's
  `min_child_samples`/learning rate) would make finding 7 fairer.
* **Mixed r_z + g_z prior (`rg_z-fixed`, `rg_z-curriculum`), submitted
  2026-09-20 13:19**: 12 runs = {blind, z-exposed} × {fixed, curriculum} ×
  {small, medium, large} (`scripts/slurm/pretrain_rg_z_mix.sbatch`; jobs
  37391372 small/medium on A30, 37391373 large on A100, est. start 21 Sep
  15:30). The multiregime share alternates micro-batches between the r_z and
  the g_z dump (`--v4-dump-path a,b` → `RoundRobinV4DumpLoader` in
  `pretrain_plain_nanotabpfn.py`); ordinary share alternates between the two
  paired shared dumps. Run dirs `/scratch/users/k23139234/tfm_runs/rg_z/` (blind, →
  `runs_exp/rg_z/`) and `…/tfm_runs/expose_z/rg_z-*` (zx). BeyondArena evals
  are queued with one `afterok` dependency per run (37391374–37391385, array
  specs 38–49). To do when they land: synthetic-bank test evals, add `rg_z`
  to the family lists of the summarize/plot scripts, redo ranks / CD / Δ
  figures and T8–T13.

## 8. Code

Repo (`scripts/`): `extract_v4_validation_cells.py` (per-cell aggregation with
prior baselines, step-0 and best-val merging), `plot_v4_per_regime_history.py`
(per-regime / per-support panels, any metric, reference lines),
`plot_pretraining_history.py`, `summarize_v4_report_excess.py` (stdlib, runs
on the login node), `compute_v4_bank_prior_baselines.py`,
`evaluate_checkpoint_on_v4_bank.py`, `evaluate_init_on_v4_bank.py`,
`evaluate_checkpoint_on_v4_dump.py`, `evaluate_tabpfn_versions_on_v4_bank.py`
(TabPFN v2.2/v2.6/v3 + TabICL v1/v1.1/v2), `slurm/evaluate_{tabpfn,tabicl,expose_z_bank}_*.sbatch`.
`merge_v4_bank_shards.py` (recombine sharded baseline runs),
`plot_v4_per_regime_history.py --panels regime_x_classes / --min-classes /
--mechanism / --support-size`, `summarize_v4_report_excess.py --json-output`
(reference lines for the plots).
`summarize_beyondarena_v4.py` (BeyondArena score tables and cross-model
ranks by task grid).
`slurm/`: `evaluate_beyondarena_v4_array_a30.sbatch`, `submit_beyondarena_v4.sh`,
`pretrain_expose_z_large_a100.sbatch`,
`pretrain_expose_z_small_medium_a30.sbatch`, `dump_expose_z_{r_z,g_z}_a30.sbatch`,
`evaluate_v4_bank_checkpoints_a30.sbatch` (repaired-bank re-evaluation),
`evaluate_expose_z_bank_a30.sbatch`.
`tfmplayground/experiments/multiregime_v4_evaluation.py` gained per-episode
AUC and the `expose_z` bank option (schema 5).
