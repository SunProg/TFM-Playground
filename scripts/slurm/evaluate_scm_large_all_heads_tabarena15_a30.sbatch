#!/bin/bash -l
#SBATCH --job-name=scm-large-tabarena15
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

: "${STUDY_ID:?Set STUDY_ID to the large table-slot training Slurm job id.}"

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly HEAD_ROOT="${PROJECT_DIR}/results/scm_table_slot_head_realistic_large_all_heads_prior_schedule_5000_slurm/${STUDY_ID}"
readonly NANO_ROOT="${PROJECT_DIR}/results/scm_nanotabpfn_realistic_large_prior_schedule_5000_slurm/37092903"
readonly CHECKPOINT_ROOT="${PROJECT_DIR}/results/scm_large_all_heads_tabarena_checkpoints/${STUDY_ID}"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/scm_large_all_heads_tabarena15/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -d "${HEAD_ROOT}"
test -d "${NANO_ROOT}"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/wrap_scm_checkpoints_for_tabarena.py \
    --head-root "${HEAD_ROOT}" \
    --nano-root "${NANO_ROOT}" \
    --output-root "${CHECKPOINT_ROOT}" \
    --embedding-size 192 \
    --num-attention-heads 6 \
    --mlp-hidden-size 768 \
    --num-layers 6 \
    --skip-variants

declare -a CHECKPOINT_ENTRIES=()
readonly MODELS=(
    decoder_baseline decoder_alpha blind_decoder blind_similarity nanotabpfn
)
for model in "${MODELS[@]}"; do
    for seed in 11 12; do
        CHECKPOINT_ENTRIES+=(
            "${model}-seed-${seed}=${CHECKPOINT_ROOT}/${model}-seed-${seed}.pth"
        )
    done
done
readonly CHECKPOINTS="$(IFS=,; echo "${CHECKPOINT_ENTRIES[*]}")"

echo "job=${SLURM_JOB_ID} study_id=${STUDY_ID} started_at=$(date --iso-8601=seconds)"
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
