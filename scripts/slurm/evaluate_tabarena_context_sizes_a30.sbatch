#!/bin/bash -l
#SBATCH --job-name=tabarena-context-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Evaluate one or more named inference-compatible nanoTabPFN checkpoints at
# exact 5-fold support contexts of 512, 1024, and 2048 rows. Datasets that do
# not contain the required total rows are excluded for that context size.
set -euo pipefail

: "${CHECKPOINTS:?Set CHECKPOINTS to comma-separated name=checkpoint-path entries.}"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/results/tabarena_context_sweep/${SLURM_JOB_ID}}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

for support_size in 512 1024 2048; do
    srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
        --standalone-checkpoints "${CHECKPOINTS}" \
        --no-include-vanilla \
        --output-dir "${OUTPUT_ROOT}/support-${support_size}" \
        --device cuda \
        --support-size "${support_size}" \
        --require-full-subsample \
        --folds 5 \
        --repeats 10 \
        --no-include-sklearn \
        --no-include-tabpfn \
        --label-source real \
        --contamination 0.0 \
        --cache-directory "${PROJECT_DIR}/.cache/openml"
done
