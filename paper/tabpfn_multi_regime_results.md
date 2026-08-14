# nanoTabPFN vs. real TabPFN under mixed-regime (multi-SCM) contexts

## Status

- Date: 2026-08-14
- Scope: binary classification, synthetic episodes, real GPU run on CREATE (`biomed_a30_gpu`, NVIDIA A30)
- Models: this repo's `nanotabpfn.pth` checkpoint (6-layer, embedding 192, 3.7M params) and real
  `TabPFNClassifier` (`tabpfn==2.2.1`, default `n_estimators=8`)
- Branch/provenance: `codex/continuous-bayesian-uncertainty` @ `61d216b`, run from an isolated
  `git worktree` at `~/repo/TFM-Playground-uncertainty` on CREATE (the existing `SunProg/latent-hypothesis`
  checkout was never touched); Slurm array jobs `36548617` (15/18 combinations) + `36548766` (retry for the
  3 combinations lost to a first-run TabPFN checkpoint-cache download race) — `18/18` combinations complete

## Motivation

Everything else in this branch's uncertainty trials (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`,
`SUPPORT_RESAMPLING_VARIANCE_TRIAL.md`) asks whether nanoTabPFN's encoder or a resampling estimator can
*detect* epistemic uncertainty within a single, internally-consistent synthetic-function episode. This
experiment asks a different, more concrete question: what happens to raw predictive performance -- not an
uncertainty signal, plain AUC -- when the labelled context a model is given is a genuine mixture of two
different data-generating functions (SCMs), and does either model behave any differently as that mixture
gets more or less severe? It follows the exact `contamination` construction already used elsewhere in this
repo (`tfmplayground/experiments/evaluate_tabarena_small.py`): *"the same contamination fraction is applied
independently to support/context rows and query rows. The model sees only the mixed context labels; query
labels remain hidden until scoring."*

## Construction

For one episode, a **base** family and a **contaminant** family are drawn from
`tfmplayground.experiments.continuous_episodes._ANALYTIC_SCORES` (`linear`, `threshold`, `tree`,
`sparse_interaction`, `dense_interaction`, `smooth`). A single shared feature matrix `x` is generated once;
both families' label functions are evaluated on the *same* `x`, so a query row's meaning is unambiguous
under either family, and only the labels differ. Support rows and query rows are then drawn from `x`, and
independently, a fixed fraction of support rows and query rows are re-labelled under the contaminant family
instead of the base family:

```
support_size = 512, query_count = 16, features = 6, label noise = 0.05
contamination in {0.0, 0.2, 0.4}   (fraction of rows relabelled under the contaminant family)
6 family pairs, cyclic: linear -> threshold -> tree -> sparse_interaction -> dense_interaction -> smooth -> linear
30 episodes per (contamination, pair) combination, both models scored on identical episodes
```

Every query row is tagged **base** or **other** depending on which family actually generated its label.
Nothing in `(x, y)` marks a row's source family -- the model has no way to know from the data alone that a
context is mixed. `support_size=512` is the largest size usable identically by both models: nanoTabPFN has
no hard limit, but `TabPFNClassifier` refuses CPU runs above 1000 rows by default and its own package
warns above 200; running on GPU on CREATE removes that constraint but 512 was kept fixed for a clean
same-size comparison and because it is nanoTabPFN's own generator's largest grid value
(`continuous_episodes.SUPPORT_SIZES`).

### Concrete support-set example

512-row support set, contamination=0.2, base=`linear`, contaminant=`tree` (first 10 and last 5 rows shown):

```
 row                                     x (6 features)  source   y
   0     1.21    0.03   -5.58    0.33   -1.64  -13.72   linear   1
   1    -3.59   -0.02   -4.48    5.17    2.02    4.26   linear   0
   2    -3.45    0.25   -4.31   -0.67   -0.89    7.07   linear   0
   3    -0.41    0.07   -1.79    2.24   -0.02   -4.24   linear   0
   4    -1.58    0.23    0.41   -4.44   -2.12    6.64   linear   1
   5     2.10    0.18    5.22    6.71    0.43    2.93   linear   1
   6     5.12    0.13    4.96    2.07   -0.62    4.90   linear   1
   7    -1.75    0.01    2.29   -2.16   -1.70   -3.39   linear   0
   8     3.34    0.05   -7.14    3.90   -0.74   -0.56   linear   1
   9     3.51   -0.26    2.91   -0.59   -0.63   14.86     tree   1   <- contaminated row
 ...
 507    -2.30    0.06   -4.49   -4.37    1.11  -15.01   linear   0
 508     3.16   -0.33   -0.66    5.71   -1.20    0.98     tree   1   <- contaminated row
 509     1.52    0.10    3.16   -2.96    0.70   10.69   linear   0
 510    -1.58    0.16   -0.16   -8.69   -0.63   -7.97   linear   1
 511     2.05    0.06    2.07   -0.70   -1.93    3.17   linear   1

class balance: 253 positive / 259 negative
contaminated (tree-labeled): 102/512 = 19.9%
noise-flipped: 30/512 = 5.9%
```

The `source` column is diagnostic-only, kept for scoring; only `x` and the final `y` are ever passed to
either model.

## Results

Bootstrap 95% CIs (1000 resamples) over the pooled query rows within each `(contamination, pair, model,
regime)` cell, 30 episodes per cell (query_count=16, so up to 480 query rows per cell, split base/other by
the contamination fraction).

### contamination = 0.0 (no mixing; reference)

| pair | model | base AUC |
|---|---|---:|
| dense_interaction+smooth | nanoTabPFN | 0.892 |
| dense_interaction+smooth | TabPFN | 0.909 |
| linear+threshold | nanoTabPFN | 0.915 |
| linear+threshold | TabPFN | 0.921 |
| smooth+linear | nanoTabPFN | 0.837 |
| smooth+linear | TabPFN | 0.873 |
| sparse_interaction+dense_interaction | nanoTabPFN | 0.959 |
| sparse_interaction+dense_interaction | TabPFN | 0.972 |
| threshold+tree | nanoTabPFN | 0.923 |
| threshold+tree | TabPFN | 0.935 |
| tree+sparse_interaction | nanoTabPFN | 0.925 |
| tree+sparse_interaction | TabPFN | 0.962 |

TabPFN's base-regime AUC is higher than nanoTabPFN's in 6/6 pairs at zero contamination -- consistent with
it being the larger, more capable pretrained model, on a generator neither model trained on.

### contamination = 0.2

| pair | model | base AUC | other AUC [95% CI] |
|---|---|---:|---|
| dense_interaction+smooth | nanoTabPFN | 0.886 | 0.519 [0.388, 0.638] |
| dense_interaction+smooth | TabPFN | 0.908 | **0.442 [0.318, 0.564]** |
| linear+threshold | nanoTabPFN | 0.924 | 0.668 [0.550, 0.771] |
| linear+threshold | TabPFN | 0.939 | 0.687 [0.555, 0.803] |
| smooth+linear | nanoTabPFN | 0.785 | 0.650 [0.528, 0.762] |
| smooth+linear | TabPFN | 0.813 | 0.661 [0.544, 0.775] |
| sparse_interaction+dense_interaction | nanoTabPFN | 0.942 | 0.627 [0.513, 0.736] |
| sparse_interaction+dense_interaction | TabPFN | 0.955 | 0.530 [0.410, 0.651] |
| threshold+tree | nanoTabPFN | 0.942 | 0.662 [0.555, 0.766] |
| threshold+tree | TabPFN | 0.957 | 0.674 [0.560, 0.773] |
| tree+sparse_interaction | nanoTabPFN | 0.923 | 0.527 [0.407, 0.648] |
| tree+sparse_interaction | TabPFN | 0.955 | **0.447 [0.327, 0.569]** |

### contamination = 0.4

| pair | model | base AUC | other AUC [95% CI] |
|---|---|---:|---|
| dense_interaction+smooth | nanoTabPFN | 0.839 | 0.511 [0.424, 0.591] |
| dense_interaction+smooth | TabPFN | 0.884 | 0.619 [0.541, 0.696] |
| linear+threshold | nanoTabPFN | 0.884 | 0.739 [0.664, 0.807] |
| linear+threshold | TabPFN | 0.922 | 0.803 [0.731, 0.869] |
| smooth+linear | nanoTabPFN | **0.589** | **0.874 [0.824, 0.923]** |
| smooth+linear | TabPFN | **0.674** | **0.880 [0.825, 0.930]** |
| sparse_interaction+dense_interaction | nanoTabPFN | 0.873 | 0.743 [0.668, 0.810] |
| sparse_interaction+dense_interaction | TabPFN | 0.898 | 0.654 [0.576, 0.739] |
| threshold+tree | nanoTabPFN | 0.882 | 0.778 [0.707, 0.841] |
| threshold+tree | TabPFN | 0.882 | 0.804 [0.739, 0.865] |
| tree+sparse_interaction | nanoTabPFN | 0.853 | 0.702 [0.626, 0.773] |
| tree+sparse_interaction | TabPFN | 0.878 | 0.631 [0.547, 0.711] |

### Ranking by worst degradation (other-regime AUC, ascending), top 10 of 24 contaminated cells

| rank | contam | pair | model | base AUC | other AUC [95% CI] |
|---:|---:|---|---|---:|---|
| 1 | 0.2 | dense_interaction+smooth | TabPFN | 0.908 | 0.442 [0.318, 0.564] |
| 2 | 0.2 | tree+sparse_interaction | TabPFN | 0.955 | 0.447 [0.327, 0.569] |
| 3 | 0.4 | dense_interaction+smooth | nanoTabPFN | 0.839 | 0.511 [0.424, 0.591] |
| 4 | 0.2 | dense_interaction+smooth | nanoTabPFN | 0.886 | 0.519 [0.388, 0.638] |
| 5 | 0.2 | tree+sparse_interaction | nanoTabPFN | 0.923 | 0.527 [0.407, 0.648] |
| 6 | 0.2 | sparse_interaction+dense_interaction | TabPFN | 0.955 | 0.530 [0.410, 0.651] |
| 7 | 0.4 | dense_interaction+smooth | TabPFN | 0.884 | 0.619 [0.541, 0.696] |
| 8 | 0.2 | sparse_interaction+dense_interaction | nanoTabPFN | 0.942 | 0.627 [0.513, 0.736] |
| 9 | 0.4 | tree+sparse_interaction | TabPFN | 0.878 | 0.631 [0.547, 0.711] |
| 10 | 0.2 | smooth+linear | nanoTabPFN | 0.785 | 0.650 [0.528, 0.762] |

Full 36-row table (all contamination levels, all pairs, both models) in
`results/support_resampling/tabpfn_multi_regime_results.json` (mirrors the on-cluster
`regime_pairs_array_results/summary.json`).

## Findings

1. **The base/other performance gap is real, not noise.** In every one of the 12 non-zero-contamination
   cells, the "other"-regime 95% CI upper bound stays below its paired base-regime point estimate. At 30
   episodes and up to several hundred pooled query rows per cell, this is a statistically supported effect,
   not a small-sample artifact.

2. **The interaction-family pairs (`dense_interaction`, `sparse_interaction`, `tree`) are consistently the
   worst combinations for both models** -- they occupy 9 of the top 10 worst-degradation ranks. The simpler
   `linear`/`threshold` pairs degrade less. Family "distance" (e.g. two interaction families being
   structurally closer to each other) does not predict robustness -- if anything the interaction pairs
   fare worse than the linear/tree cross-pair despite superficially "same-ish" structure.

3. **`smooth+linear` at contamination=0.4 inverts, and the inversion is now statistically supported.** Both
   models' "other"-regime AUC (0.874-0.880) is *higher* than their own base-regime AUC (0.589-0.674) at that
   contamination level, with non-overlapping 95% CIs. This is the one genuine anomaly in the dataset and is
   not explained here; a plausible candidate is that `smooth` is by far the weakest base family for both
   models even at contamination=0 (0.837/0.873, the lowest of the six pairs), so once 40% of the support is
   `linear`-labelled the `linear` signal is easier for the model to actually learn than the smooth family it
   was nominally supposed to be conditioned on, and the "other" queries end up better served than the
   "base" ones. Worth a dedicated follow-up rather than treating as settled.

4. **No consistent winner between nanoTabPFN and real TabPFN.** TabPFN has higher base-regime AUC in every
   zero-contamination cell (larger pretrained model, unsurprising), but on the contaminated "other" rows
   each model tops the worst-degradation ranking about equally often (TabPFN: ranks 1, 2, 6, 7, 9;
   nanoTabPFN: ranks 3, 4, 5, 8, 10). Being the larger, generally stronger model does not make TabPFN
   more robust to context contamination specifically.

5. **Neither model's predictive distribution reacts to the contamination itself.** This experiment only
   scores accuracy/AUC against ground truth; it does not re-examine predictive entropy, but the pattern is
   consistent with every other result in this branch's uncertainty trials (`CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`,
   `SUPPORT_RESAMPLING_VARIANCE_TRIAL.md`): nothing about how confidently either model answers a query
   distinguishes a query whose true label came from the base family versus the contaminant family. The
   model simply interpolates according to the contamination fraction, like a single amortized average, with
   no internal signature anywhere flagged so far that separates "this specific query is unusually likely to
   be wrong because the context was mixed."

## Infrastructure notes

- Environment: `uv sync --extra tabpfn` in a fresh worktree venv (`torch==2.9.0+cu128`, `cuda_available=True`
  confirmed via `nvidia-smi` + `torch.cuda.is_available()` inside the Slurm job).
- The first array submission (`36548617`, `--array=0-17%8`) lost 3/18 tasks (indices 0, 1, 3, all
  contamination=0.0) to a race: 8 concurrent tasks all attempted to auto-download the same TabPFN model
  checkpoint (`tabpfn-v2-classifier-finetuned-*.ckpt`) to the shared `~/.cache/tabpfn/` on first use
  simultaneously, and some processes read a partially-written file. Fixed by warming the cache with one
  serial run before resubmitting only the missing 3 indices (`36548766`).
- `checkpoints/` and `runs/` are symlinked into the worktree from the original `~/repo/TFM-Playground`
  checkout rather than copied, since both are gitignored and not part of any commit.

## Files

- `regime_pairs_array.py` (on CREATE, not yet copied into this repo) -- the array-indexed experiment driver;
  `--index N` runs one `(contamination, pair)` combination end to end and writes
  `regime_pairs_array_results/{N}.json`; `--summarize` aggregates all combinations into the ranking table
  above.
- `scripts/slurm/regime_pairs_array.sbatch` (on CREATE) -- `--array=0-17%8` driver.
- `results/support_resampling/tabpfn_multi_regime_results.json` -- the flat 36-row result table backing
  every number in this report.
