#!/bin/bash -l
#SBATCH --job-name=detection-ceiling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# How well could any learner detect the contaminated rows of an episode?
#
# The prior as originally built has a ceiling of about 0.54, so every binding
# number this project reported was scored against a target nothing can reach.
# This grid finds which design, if any, is worth training on -- and it is CPU
# work, so it does not belong on a GPU partition.
#
#   0-31   classification, support x features x contamination x num_classes
#   32-39  regression, same axes minus num_classes
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
