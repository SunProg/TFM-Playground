#!/bin/bash -l
#SBATCH --job-name=vanilla-coh2-tabarena15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Post-hoc 15-task/30-predictor TabArena evaluation of the three vanilla-coh2
# checkpoints (job 36982768), matching the protocol the closure-loop arms
# were screened under -- one shared evaluation script, so numbers stay
# comparable across every arm regardless of when each was trained.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly CK="${PROJECT_DIR}/runs/slot_tabpfn/36982768"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/vanilla_coh2_tabarena15"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${PROJECT_DIR}/.cache/openml"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "plain=${CK}/plain-vanilla-coh2/final_checkpoint.pth,mixed=${CK}/mixed-vanilla-coh2/final_checkpoint.pth,curriculum=${CK}/curriculum-vanilla-coh2/final_checkpoint.pth" \
    --no-include-vanilla \
    --max-predictors 30 \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --subsample 2048 \
    --folds 5 \
    --repeats 10 \
    --no-include-sklearn \
    --no-include-tabpfn \
    --cache-directory "${PROJECT_DIR}/.cache/openml"

echo "finished_at=$(date --iso-8601=seconds)"
