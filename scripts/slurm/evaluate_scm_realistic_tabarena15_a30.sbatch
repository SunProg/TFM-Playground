#!/bin/bash -l
#SBATCH --job-name=scm-tabarena15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly CHECKPOINT_ROOT="${PROJECT_DIR}/results/scm_realistic_5000_tabarena_checkpoints"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/scm_realistic_5000_tabarena15/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT_ROOT}/manifest.json"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

declare -a CHECKPOINT_ENTRIES=()
readonly MODELS=(
    decoder_baseline decoder_alpha blind_decoder blind_similarity
    nanotabpfn table_slot_backbone table_slot_mufasa
)
for model in "${MODELS[@]}"; do
    for seed in 11 12; do
        CHECKPOINT_ENTRIES+=(
            "${model}-seed-${seed}=${CHECKPOINT_ROOT}/${model}-seed-${seed}.pth"
        )
    done
done
readonly CHECKPOINTS="$(IFS=,; echo "${CHECKPOINT_ENTRIES[*]}")"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${CHECKPOINTS}" \
    --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --max-predictors 30 \
    --subsample 2048 \
    --folds 5 \
    --repeats 10 \
    --no-include-sklearn \
    --no-include-tabpfn \
    --label-source real \
    --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"

echo "finished_at=$(date --iso-8601=seconds)"
