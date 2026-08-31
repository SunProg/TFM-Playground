#!/bin/bash -l
#SBATCH --job-name=detection-ceiling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# How well could any learner detect the contaminated rows of an episode?
#
# The prior as originally built has a ceiling of about 0.54, so every binding
# number this project reported was scored against a target nothing can reach.
# This grid finds which design, if any, is worth training on.
#
#   0-31   classification, support x features x contamination x num_classes
#   32-39  regression, independent rules (separation 0.0)
#   40-55  regression, separated rules (separation 0.5 and 0.9)
#   56-63  repeated labels (Dawid-Skene), 2 or 3 labels per row
#   64-71  anchor points, 10% or 25% of rows revealed as known-clean
#   72-79  gate and separation together -- mixture of experts proper
#
# Separation is the half of mixture-of-experts identifiability this prior
# never controlled: `regime_coherence` supplied the covariate-dependent gate
# and left component separation to chance, which is why the gate bought
# nothing.  32-39 is its zero point.
#
# This work is entirely CPU-bound -- SCM generation and scikit-learn, nothing
# on the device.  It runs on the gpu partition for queue availability and the
# wider nodes, so the requested GPU sits idle; move it back to `cpu` if that
# partition is free.
#
# The index-to-cell mapping lives in detection_ceiling.py, not here; run
# `python -m tfmplayground.experiments.detection_ceiling --list` to see it.
# Each task prints one JSON row to its .out file; collect with
#   grep -h '^{' slurm-detection-ceiling-<jobid>_*.out
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

GRID_SIZE="$("${PYTHON_BIN}" -c 'from tfmplayground.experiments.detection_ceiling import ceiling_cells as c; print(len(c()))')"
readonly GRID_SIZE
echo "grid_size=${GRID_SIZE} array_task_count=${SLURM_ARRAY_TASK_COUNT}"
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne "${GRID_SIZE}" ]]; then
    echo "NOTE: running ${SLURM_ARRAY_TASK_COUNT} of ${GRID_SIZE} grid cells (subset submission)"
fi
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.detection_ceiling --index "${SLURM_ARRAY_TASK_ID}"

echo "finished_at=$(date --iso-8601=seconds)"
