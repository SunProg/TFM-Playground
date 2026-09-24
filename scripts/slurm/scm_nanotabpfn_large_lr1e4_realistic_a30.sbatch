#!/bin/bash -l
#SBATCH --job-name=scm-nanotabpfn-large-lr1e4
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
readonly RUN_DIR="${PROJECT_DIR}/results/scm_nanotabpfn_realistic_large_lr1e4_5000_slurm/${STUDY_ID}/seed-${SEED}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "model=nanotabpfn-large-lr1e4 seed=${SEED} run_dir=${RUN_DIR} host=$(hostname)"
echo "architecture=embedding-192 heads-6 mlp-768 layers-6 learning_rate=1e-4"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_nanotabpfn_baseline \
    --output "${RUN_DIR}" \
    --prior-profile realistic \
    --steps 5000 \
    --seeds "${SEED}" \
    --test-episodes 128 \
    --validation-episodes 32 \
    --embedding-size 192 \
    --num-attention-heads 6 \
    --mlp-hidden-size 768 \
    --num-layers 6 \
    --learning-rate 1e-4 \
    --device cuda \
    --require-cuda
