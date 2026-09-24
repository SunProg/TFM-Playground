#!/bin/bash -l
#SBATCH --job-name=scm-nano-priorlr
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

# Small plain NanoTabPFN control under the same schedule as the prior-family
# runs.  This isolates optimizer effects from the earlier fixed-1e-3 result.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(11 12)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${PROJECT_DIR}/results/scm_nanotabpfn_realistic_prior_schedule_5000_slurm/${STUDY_ID}/seed-${SEED}"

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
echo "architecture=embedding-32 heads-4 mlp-64 layers-2"
echo "optimizer=AdamW peak-lr-1e-4 min-lr-1e-6 warmup-2000 cosine-decay"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_nanotabpfn_baseline \
    --output "${RUN_DIR}" \
    --prior-profile realistic \
    --steps 5000 \
    --seeds "${SEED}" \
    --test-episodes 128 \
    --validation-episodes 32 \
    --embedding-size 32 \
    --num-attention-heads 4 \
    --mlp-hidden-size 64 \
    --num-layers 2 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --device cuda \
    --require-cuda
