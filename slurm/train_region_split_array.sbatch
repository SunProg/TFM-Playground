#!/bin/bash
#SBATCH --job-name=region-split-1000
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/users/k23139234/repo/TFM-Playground/slurm/region-split-%A_%a.out
#SBATCH --error=/users/k23139234/repo/TFM-Playground/slurm/region-split-%A_%a.err

set -euo pipefail

REPO=/users/k23139234/repo/TFM-Playground
VENV=/users/k23139234/repo/ehr-text-multi-modal/.venv
PERSIST="$REPO/runs/region-split/array-1000"
SEEDS=(2402 22402 42402 52402 62402 72402 82402 92402)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"
RUN_ROOT="$SCRATCH_ROOT/tfm-region-split-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
SCRATCH_REPO="$RUN_ROOT/repo"
SCRATCH_OUT="$RUN_ROOT/output"

mkdir -p "$REPO/slurm" "$PERSIST" "$SCRATCH_REPO"
cp -a "$REPO/tfmplayground" "$REPO/pyproject.toml" "$SCRATCH_REPO/"
cp "$REPO/runs/region-split/tuning-20260812/lr-0p0001_final.pth" "$SCRATCH_REPO/source.pth"

export PYTHONPATH="$SCRATCH_REPO"
export TORCH_HOME="$RUN_ROOT/torch-cache"
export HF_HOME="$RUN_ROOT/huggingface-cache"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"

"$VENV/bin/python" -u -m tfmplayground.experiments.train_region_split_continuation \
  --source-checkpoint "$SCRATCH_REPO/source.pth" \
  --output-dir "$SCRATCH_OUT" \
  --device cuda \
  --seed "$SEED" \
  --start-step 400 \
  --end-step 1000 \
  --learning-rate 1e-4 \
  --batch-size 4 \
  --report-every 100 \
  --validation-episodes 64 \
  --ordinary-episode-fraction 0.35

mkdir -p "$PERSIST/task-${SLURM_ARRAY_TASK_ID}"
cp "$SCRATCH_OUT/trajectory.csv" "$PERSIST/task-${SLURM_ARRAY_TASK_ID}/"
cp "$SCRATCH_OUT/summary.json" "$PERSIST/task-${SLURM_ARRAY_TASK_ID}/"
cp "$SCRATCH_OUT/config.json" "$PERSIST/task-${SLURM_ARRAY_TASK_ID}/"
cp "$SCRATCH_OUT/adapter_step_1000.pth" "$PERSIST/task-${SLURM_ARRAY_TASK_ID}/"
