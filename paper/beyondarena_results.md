# BeyondArena v4 results

## Scope

The evaluation uses the official BeyondArena containers and official outer folds.
Each complete training fold is used as context; oversized folds are recorded as
unsupported rather than subsampled.

Filters:

- Classification tasks only
- 2–5 classes
- Metadata row count ≤10,000
- Raw feature count ≤30
- Text and high-cardinality categorical tasks excluded
- Target and grouped-identifier columns excluded from model inputs
- Feature preprocessing fitted on each training fold only

Metrics:

- Excess cross-entropy relative to the training-fold class-prior predictor; lower is better
- Macro one-vs-rest AUC; higher is better
- Accuracy gain relative to the training-fold majority predictor; higher is better

## Ranking calculation

For each model, condition, and metric:

1. Average the metric across the official folds of each task. Folds therefore
   receive equal weight within a task.
2. Rank the models independently within each task using the task-level score.
   Rank 1 is lowest for excess cross-entropy and highest for AUC or accuracy
   gain. Ties use the minimum rank.
3. Average those per-task ranks across tasks. Each task therefore receives equal
   weight, regardless of its number of folds or held-out rows.

The `all` condition uses all 32 eligible tasks. Task-grid figures then report
the mean of these per-task-mean ranks across model identities grouped into a
family/size bin.

Comparison sets are kept explicit: the full `rankings.csv` compares 47
selections (21 v4 final, 21 v4 best-own-validation, and 5 published models).
The v4-only heatmaps recompute ranks within the 21 models of each checkpoint
policy, while the final-v4-plus-published comparison uses 26 models.

## Coverage

The full run was Slurm job `37325808` and completed successfully.

- Eligible tasks: 32
- Manifest rows inspected: 142
- Model selections: 47 available
  - 42 v4 selections: final and best-own-validation checkpoints
  - 5 published baselines: TabPFN v2.2, v2.6, v3; TabICL v1, v2
- Fold-metric rows: 462,903
- Evaluated rows: 462,903
- Error rows: 0
- Unsupported rows: 0

### Task grid

| Task grid | Tasks |
|---|---:|
| Binary × IID | 23 |
| Binary × Grouped | 1 |
| Multi × IID | 6 |
| Multi × Grouped | 2 |
| Temporal | 0 |
| Total | 32 |

## Final v4 plus published ranking

This comparison keeps only v4 `final` checkpoints and published TabPFN/TabICL
baselines. Ranks are recomputed within this 26-model comparison set, so rank 1
is the best possible per-task rank and lower mean task rank is better.

Across all eligible tasks:

| Metric | Best model by mean task rank | Mean task rank | Equal-task metric score |
|---|---|---:|
| Excess cross-entropy | TabICL v2 | 2.6875 | -0.290235 |
| Macro OVR AUC | TabICL v2 | 3.53125 | 0.881266 |
| Accuracy gain | TabPFN v3 | 3.1875 | 0.147515 |

The v4 branch/size groups are split as canonical, r_z-fixed, r_z-curriculum,
g_z-fixed, and g_z-curriculum; sizes are small, medium, and large.

## Result files

Full 47-selection run:

- [Task manifest](../results/beyondarena-v4/all-families-37325808/task_manifest.csv)
- [Fold metrics](../results/beyondarena-v4/all-families-37325808/fold_metrics.csv)
- [Rankings](../results/beyondarena-v4/all-families-37325808/rankings.csv)
- [Ranking summary](../results/beyondarena-v4/all-families-37325808/ranking_summary.md)
- [Checkpoint manifest](../results/beyondarena-v4/all-families-37325808/checkpoints.csv)
- [Run config](../results/beyondarena-v4/all-families-37325808/config.json)

Final-v4-plus-published comparison:

- [Recomputed rankings](../results/beyondarena-v4/all-families-37325808/final_published/rankings.csv)
- [Recomputed ranking summary](../results/beyondarena-v4/all-families-37325808/final_published/ranking_summary.md)

## Figures

- [Overall score comparison](../figures/beyondarena/beyondarena_overall_comparison.png)
- [Overall rank comparison](../figures/beyondarena/beyondarena_rank_comparison.png)
- [Family heatmaps](../figures/beyondarena/beyondarena_family_heatmaps.png)
- [Mean rank by checkpoint, family, and size](../figures/beyondarena/beyondarena_mean_rank_family_size.png)
- [Task-grid ranks: final](../figures/beyondarena/beyondarena_mean_rank_task_grid_final.png)
- [Task-grid ranks: best own validation](../figures/beyondarena/beyondarena_mean_rank_task_grid_best_own_val.png)
- [Task-grid ranks: final v4 plus published models](../figures/beyondarena/beyondarena_mean_rank_task_grid_all_models.png)
- [Final-v4-plus-published ranking plot](../figures/beyondarena/beyondarena_final_published_rank_comparison.png)
- [Baseline versus best-v4 comparison](../figures/beyondarena/beyondarena_baseline_comparison.png)

The final-v4-plus-published plot uses color for model family and hatch patterns
for v4 size; its legends identify both encodings.

## Reproduction scripts

- [BeyondArena evaluator](../tfmplayground/experiments/evaluate_multiregime_v4_beyondarena.py)
- [Ranking recomputation from saved fold metrics](../scripts/recompute_beyondarena_rankings.py)
- [Plot generator](../scripts/plot_beyondarena_comparisons.py)
- [Final-plus-published ranking generator](../scripts/make_beyondarena_final_published_ranking.py)
