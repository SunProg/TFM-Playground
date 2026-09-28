# Frozen TabICLv2 decoder experiments

These experiments load the released `tabicl-classifier-v2-20260212.ckpt`,
freeze its column, row, and ICL encoders, and train one prediction head:

- Slot Attention router with K = 4, 8, 16, or 32.
- Original TabICLv2 MLP architecture, initialized either from released weights
  or randomly. The current trajectory uses random initialization.
- TabPFNv3's `ManyClassDecoder` architecture, initialized randomly over
  TabICLv2 representations. This is not a pretrained TabPFNv3 model.

`tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder` subclasses
`tabicl.train._run.Trainer` from pinned TabICL 2.2.0. Upstream code supplies
synthetic task generation, loss, optimizer, scheduler, gradient accumulation,
and distributed training. The extension selects and freezes modules, records
metrics/provenance, and saves compact decoder checkpoints. Freezing means
`requires_grad=False`; the upstream training loop still uses training mode.

## Current Stage 1 trajectories

The current six runs use seed 2402: four slot counts, random MLP, and random
TabPFNv3 decoder. They continue from the released TabICLv2 backbone using
short Stage 1 tasks; they do not repeat the full three-stage pretraining.

| Setting | Current trajectory |
| --- | --- |
| GPU allocation | One A30 per training job |
| Steps | 10,000 |
| Checkpoint cadence | Every 500 steps, including step 500 |
| Batch / microbatch | 64 / 4 |
| Prior | Official `graph_scm` |
| Rows per synthetic task | Fixed 1,024 |
| Context fraction | 0.3–0.9 |
| Features / supported classes | Up to 100 / up to 10 |
| Optimizer / scheduler | Muon / `cosine_with_restarts` |
| Learning rate | 8e-4 |
| Warmup | 0.2, or 2,000 steps |
| Gradient clipping | 10.0 |

The older 500-step pilots have a different schedule. Keep those checkpoints
separate from the 10,000-step trajectories. The step-500 checkpoint from a
current trajectory is part of its evaluation history and must be included.

The Python entry point still defaults to the original Stage 3 continuation
recipe: 10,000 steps, batch 64, LR 2e-5, 1% warmup, clipping 1.0,
400–60,000 rows, and 79–81% context. The `*_trajectory.sbatch` scripts override
those values with the table above.

## Environment and submission on create

Run from `/users/k23139234/repo/TFM-Playground-slot` on `ssh create`.
The existing `.venv` supplies Torch and pretraining dependencies.
`scripts/slurm/setup_tabicl_slot_decoder_env.sbatch` installs TabICL 2.2.0
without dependencies into `.tabicl-slot-site`, preserving the shared
environment. It is only needed when that overlay is missing. TabPFNv3 also
requires the project's `tabpfn` extra (`tabpfn>=8.5,<9`).

For a new trajectory, explicitly set output directories; the scripts' fallback
directories retain older Stage 1 job-directory names. Example submissions
below create new jobs, so do not rerun them just to inspect existing experiments.

```bash
TRAJ_ROOT="$PWD/runs/tabicl_v2_slot_decoder/trajectory-stage1-10k-warmup20"
for k in 4 8 16 32; do
  sbatch --export="ALL,SLOT_COUNT=$k,SLOT_SEED=2402,SLOT_RUN_DIR=$TRAJ_ROOT/slot-k$k-seed2402" \
    scripts/slurm/train_tabicl_v2_slot_trajectory.sbatch
done
sbatch --export="ALL,MLP_SEED=2402,MLP_RUN_DIR=$TRAJ_ROOT/mlp-random-seed2402" \
  scripts/slurm/train_tabicl_v2_mlp_random_trajectory.sbatch
sbatch --export="ALL,V3_SEED=2402,V3_RUN_DIR=$TRAJ_ROOT/tabpfnv3-random-seed2402" \
  scripts/slurm/train_tabicl_v2_tabpfnv3_decoder_trajectory.sbatch
```

Each is a separate one-GPU job with no array concurrency cap. These commands
do not require eight GPUs for any individual run.

Historical workflows remain available: the `*_stage1.sbatch` and
`*stage1_pilot.sbatch` scripts are short pilots; the Stage 3
`train_tabicl_v2_slot_decoder_a30.sbatch` array and shell grids cover
K = 4/8/16 and seeds 2402/2403/2404. The GPU-scaling probe is a separate
resource experiment, not a prerequisite for the current one-GPU runs.

## Checkpoints and provenance

Current remote checkpoints follow this layout:

```text
/users/k23139234/repo/TFM-Playground-slot/
  runs/tabicl_v2_slot_decoder/trajectory-stage1-10k-warmup20/
    slot-k4-seed2402/step-500.ckpt
    slot-k8-seed2402/step-500.ckpt
    slot-k16-seed2402/step-500.ckpt
    slot-k32-seed2402/step-500.ckpt
    mlp-random-seed2402/step-500.ckpt
    tabpfnv3-random-seed2402/step-500.ckpt
    <variant>-seed2402/metrics.jsonl
    tabarena/step-<step>-<tag>/
```

The same layout extends through `step-10000.ckpt`. Each
`decoder_only_v1` checkpoint contains only the head's model weights plus
optimizer/scheduler state, step, model configuration, and provenance. The
frozen backbone is reconstructed from the released checkpoint, not duplicated
in each snapshot. Metadata records the base checkpoint path, SHA256 and
version, package version, initialization, seed, slot configuration, prior, and
schedule. Older full-model snapshots remain readable.

Inference needs access to the original base checkpoint or the official
download. The loader verifies the base configuration and SHA256 for compact
snapshots. Preserve the released checkpoint independently when archiving or
moving experiments. Generated checkpoints, result caches, environments and raw
Slurm logs are not source files and are not included in these commits.

## TabArena evaluation protocols

The evaluator requires TabArena revision
`06334097d539a5d494e56576cb973d09e251dc8c` and uses its task selection, splits,
metrics, result cache and leaderboard computation.

| Protocol | CLI selection | Scope used in these experiments |
| --- | --- | --- |
| Small comparison | Default | Six tasks: `lite & multiclass & tabpfn` |
| Expanded classification | `--classification-only` | 38 eligible binary/multiclass tasks, lite split 0 |
| Additional canonical splits | `--full`, optionally with `--classification-only` | Disables the lite filter; not the current 38-task history |

The 38-task evaluation has sometimes been called “full” in experiment
discussions. It means the expanded classification task set here, not every
TabArena problem type and split. Its limits are 100,000 training rows per fold,
2,000 features and 10 classes. The small protocol retains 10,000/500/10.
Expanded limits apply only to the new TabICL adapters; existing adapters keep
their original constraints.

Evaluate an available checkpoint directly, without a watcher:

```bash
PYTHONPATH="$PWD:$PWD/.tabicl-slot-site" .venv/bin/python \
  -m tfmplayground.experiments.evaluate_tabicl_v2_slot_decoder \
  --decoder "slot-k4-seed2402-step2000=$TRAJ_ROOT/slot-k4-seed2402/step-2000.ckpt" \
  --classification-only --skip-zero-shot \
  --output-dir "$TRAJ_ROOT/tabarena/step-2000-example" \
  --results-dir tabicl_v2_step2000_example
```

Use a fresh output directory. `protocol.json` identifies the exact subset,
revision, checkpoint hashes, human-readable model names and positional
TabArena configuration names. Use that mapping when joining histories;
`TabICLv2Decoder_c1` need not represent the same model in different runs.

Evaluate the released zero-shot model once per matching task/split protocol
and retain its cached per-task results. Omit `--skip-zero-shot` only for that
initial evaluation; later checkpoint jobs use `--skip-zero-shot`. An optional
`--zero-shot-checkpoint` points to the released checkpoint. Skipping the model
does not import a baseline from a different result cache: combine the saved
baseline and decoder task rows for a common comparison. A baseline on six
tasks cannot stand in for the 38-task baseline.

Outputs include the official leaderboard, per-task results with metadata, and
multiclass summaries by training size, class count and class imbalance. The
imbalance ratio is the largest divided by smallest class frequency in the
canonical dataset targets; it is descriptive metadata. Supply
`--imbalance-csv` to reuse it. The runner does not currently provide a separate
Brier/ECE calibration pipeline.

Compare Elo values only with the same tasks, reference models and aggregation.
Elo from independently assembled leaderboards can change even for unchanged
baseline predictions.

For remaining checkpoints, the trajectory submission script supports
`initial5` (K = 4/8/16, MLP, TabPFNv3; excludes K = 32) and `all6`.
A dependency submission covering every 500-step checkpoint is:

```bash
# Colon-separated IDs of the corresponding training jobs:
: "${TRAINING_JOB_IDS:?Set the relevant training job IDs first}"
for step in $(seq 500 500 10000); do
  sbatch -d "afterok:${TRAINING_JOB_IDS}" \
    --export="ALL,TRAJECTORY_ROOT=$TRAJ_ROOT,TRAJECTORY_STEP=$step,TRAJECTORY_VARIANTS=initial5,TRAJECTORY_TAG=classification-initial5,TRAJECTORY_CLASSIFICATION_ONLY=true,TRAJECTORY_INCLUDE_BASELINE=false" \
    scripts/slurm/evaluate_tabicl_v2_decoder_trajectory.sbatch
done
```

Submit only steps without existing results or pending evaluations. An
`afterok` dependency waits for successful completion of its training jobs;
to evaluate during training, submit an available checkpoint directly without
that dependency. The historical CPU watcher script is retained for
reproducibility and is not required for either workflow.

The trajectory evaluation script requests one A30 for one hour and currently
uses the saved imbalance table at
`results/tabicl_v2_slot_decoder/tabarena-lite-37581725/class_imbalance.csv`.
On another checkout, supply a valid cached table to the Python evaluator.
The older `evaluate_tabicl_v2_slot_decoder_tabarena.sbatch` references pilot
job directories and is also historical.

## Existing slot analysis and future work

`tfmplayground.experiments.analyze_tabicl_v2_slot_decoder` accepts repeated
`--checkpoint NAME=PATH` arguments and an `--output-dir`. It writes occupancy
and assignment entropy, hard-assignment ARI comparisons, and leave-one-slot-out
accuracy/log-loss and prediction changes.

This is exploratory analysis with its own smaller multiclass task filter.
Diagnostics currently come from the last internal inference forward; they are
not guaranteed to cover every ensemble member or query chunk. Pairwise
comparisons are grouped by task and K, so pass the same training step when
comparing seeds. The ablation renormalizes routing after masking a slot.
It does not establish semantic regimes.

Expanded diagnostics, explicit slot matching, permutation/input checks,
task-by-slot heatmaps, UMAP and ablation visualizations remain future work.
They are not implemented by this commit series.

## Focused verification

With TabICL 2.2.0 and the relevant optional dependencies installed:

```bash
OMP_NUM_THREADS=1 TABPFN_DISABLE_TELEMETRY=1 python -m pytest -q \
  tests/test_tabicl_slot_decoder.py \
  tests/test_tabicl_v2_slot_pretraining.py \
  tests/test_evaluate_tabicl_v2_slot_decoder.py
```

The CPU smoke test uses a tiny compatible backbone and a batch in the upstream
prior format. It exercises the official loss/optimization path, frozen
parameters, compact save/reload, provenance and inference. It does not
redownload the released model or launch the full TabArena suite.
