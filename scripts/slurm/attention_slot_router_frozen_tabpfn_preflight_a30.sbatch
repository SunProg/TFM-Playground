#!/bin/bash -l
#SBATCH --job-name=frozen-tabpfn-slot-preflight
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# First-ever GPU-scale run of AttentionSlotRouterFrozenTabPFN (frozen real
# TabPFN as the backbone, hooked for its per-column table, instead of a
# jointly-trained NanoTabPFNModel). Only CPU-scale (batch=2, 3 steps) has
# been verified so far; this is a short preflight to check wall-clock cost
# per step at realistic batch/support sizes before committing to a full
# multi-thousand-step run. scope=data only (the --prior scm branch ignores
# the mode axis entirely, so index 0 is the only meaningful cell to check).
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
readonly RUN_DIR="${PROJECT_DIR}/results/_attention_slot_router_frozen_tabpfn_preflight"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint at ${CHECKPOINT}" >&2; exit 3; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TABPFN_DISABLE_TELEMETRY=1

echo "job=${SLURM_JOB_ID} host=$(hostname) run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.attention_slot_router_frozen_tabpfn_scale \
    --index 0 \
    --output "${RUN_DIR}" \
    --device cuda \
    --prior scm \
    --tabpfn-device cuda \
    --tabpfn-model-path "${CHECKPOINT}" \
    --steps 200 \
    --hidden 768 \
    --support-size 128 --query-size 32 \
    --batch-size 4 --accumulate 2 \
    --warmup-steps 20 \
    --validation-interval 50 --validation-episodes 8 \
    --learning-rate 0.0001
