#!/bin/bash -l
#SBATCH --job-name=scm-large-slot-grid
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-39%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# A capacity sweep of the table-slot decoder head.  All runs use the same
# large NanoTabPFN backbone and pilot training regimen as job 37093501.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SLOT_COUNTS=(2 4 8 16)
readonly ITERATIONS=(1 2 3 4 5)
readonly SEEDS=(11 12)
readonly CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
readonly SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
readonly SLOT_INDEX=$((CONFIG_INDEX / ${#ITERATIONS[@]}))
readonly ITERATION_INDEX=$((CONFIG_INDEX % ${#ITERATIONS[@]}))
readonly NUM_SLOTS="${SLOT_COUNTS[${SLOT_INDEX}]}"
readonly NUM_ITERATIONS="${ITERATIONS[${ITERATION_INDEX}]}"
readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${PROJECT_DIR}/results/scm_large_slot_count_iteration_sweep_5000_slurm/${STUDY_ID}/slots-${NUM_SLOTS}/iterations-${NUM_ITERATIONS}/seed-${SEED}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED} slots=${NUM_SLOTS} iterations=${NUM_ITERATIONS}"
echo "architecture=embedding-192 heads-6 mlp-768 layers-6"
echo "optimizer=AdamW peak-lr-1e-4 min-lr-1e-6 warmup-2000 cosine-decay"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_table_slot_head_sweep \
    --output "${RUN_DIR}" \
    --prior-profile realistic \
    --steps 5000 \
    --seeds "${SEED}" \
    --conditions decoder_baseline \
    --test-episodes 128 \
    --validation-episodes 32 \
    --embedding-size 192 \
    --num-attention-heads 6 \
    --mlp-hidden-size 768 \
    --num-layers 6 \
    --num-slots "${NUM_SLOTS}" \
    --num-slot-iterations "${NUM_ITERATIONS}" \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --device cuda \
    --require-cuda
