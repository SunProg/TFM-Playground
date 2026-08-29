#!/bin/bash -l
#SBATCH --job-name=tabarena-regime-final
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Compare all matched curriculum and plain seeds on identical 2,048-row,
# 5-fold x 10-repeat TabArena episodes. Submit with an afterok dependency on
# the two five-seed pretraining arrays.
set -euo pipefail

: "${CURRICULUM_ARRAY_JOB_ID:?Set CURRICULUM_ARRAY_JOB_ID.}"
: "${PLAIN_ARRAY_JOB_ID:?Set PLAIN_ARRAY_JOB_ID.}"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/regime_pretraining/${CURRICULUM_ARRAY_JOB_ID}-${PLAIN_ARRAY_JOB_ID}/tabarena-2048-r10"
readonly SEEDS=(2402 2403 2404 2405 2406)

cd "${PROJECT_DIR}"
declare -a CHECKPOINT_ENTRIES=()
for seed in "${SEEDS[@]}"; do
    CHECKPOINT_ENTRIES+=("curriculum-${seed}=${PROJECT_DIR}/runs/regime_pretraining/${CURRICULUM_ARRAY_JOB_ID}/curriculum/seed-${seed}/final_checkpoint.pth")
    CHECKPOINT_ENTRIES+=("plain-${seed}=${PROJECT_DIR}/runs/regime_pretraining/${PLAIN_ARRAY_JOB_ID}/plain/seed-${seed}/final_checkpoint.pth")
done
readonly CHECKPOINTS="$(IFS=,; echo "${CHECKPOINT_ENTRIES[*]}")"

mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${CHECKPOINTS}" \
    --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --subsample 2048 \
    --folds 5 \
    --repeats 10 \
    --no-include-sklearn \
    --no-include-tabpfn \
    --label-source real \
    --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"
