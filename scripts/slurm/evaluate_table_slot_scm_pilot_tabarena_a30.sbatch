#!/bin/bash -l
#SBATCH --job-name=table-slot-tabarena
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%12
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One post-training TabArena-small evaluation per SCM sweep arm.  Submit with
# an afterok:TRAIN_ARRAY_%a dependency so an arm is evaluated only if its own
# matching training task completed successfully.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?TRAIN_ARRAY_JOB_ID is required}"
readonly MODES=(plain multiregime mixed curriculum)
readonly MODE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
readonly MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
readonly MODE="${MODES[${MODE_INDEX}]}"
readonly MODELS=(table_slot_head table_slot_backbone table_slot_mufasa)
readonly MODEL_TYPE="${MODELS[${MODEL_INDEX}]}"
readonly MODEL="${PROJECT_DIR}/runs/table_slot_scm/${TRAIN_ARRAY_JOB_ID}/${MODE}/${MODEL_TYPE}/seed-2402/final_checkpoint.pth"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/table_slot_scm/${TRAIN_ARRAY_JOB_ID}/${MODE}/${MODEL_TYPE}/seed-2402/tabarena-small"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${MODEL}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${MODEL_TYPE}=${MODEL}" --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" --device cuda --subsample 2048 --folds 5 --repeats 10 \
    --no-include-sklearn --no-include-tabpfn --label-source real --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"
