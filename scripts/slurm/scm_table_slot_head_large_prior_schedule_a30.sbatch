#!/bin/bash -l
#SBATCH --job-name=scm-table-slot-large-priorlr
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(11 12)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly CONDITION="decoder_baseline"
readonly RUN_DIR="${PROJECT_DIR}/results/scm_table_slot_head_realistic_large_prior_schedule_5000_slurm/${STUDY_ID}/${CONDITION}/seed-${SEED}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=${SEED}"
echo "condition=${CONDITION} architecture=embedding-192 heads-6 mlp-768 layers-6"
echo "optimizer=AdamW peak-lr-1e-4 min-lr-1e-6 warmup-2000 cosine-decay"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_table_slot_head_sweep \
    --output "${RUN_DIR}" \
    --prior-profile realistic \
    --steps 5000 \
    --seeds "${SEED}" \
    --conditions "${CONDITION}" \
    --test-episodes 128 \
    --validation-episodes 32 \
    --embedding-size 192 \
    --num-attention-heads 6 \
    --mlp-hidden-size 768 \
    --num-layers 6 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --device cuda \
    --require-cuda
