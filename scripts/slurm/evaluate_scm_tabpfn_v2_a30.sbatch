#!/bin/bash -l
#SBATCH --job-name=scm-tabpfn-v2
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
readonly OUTPUT="${PROJECT_DIR}/results/scm_tabpfn_v26_v3_realistic_test.json"
readonly V26_CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
readonly V3_CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${V26_CHECKPOINT}" || { echo "Missing ${V26_CHECKPOINT}" >&2; exit 3; }
test -f "${V3_CHECKPOINT}" || { echo "Missing ${V3_CHECKPOINT}" >&2; exit 3; }
mkdir -p "${PROJECT_DIR}/results"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" scripts/evaluate_scm_tabpfn_v2.py \
    --output "${OUTPUT}" \
    --prior-profile realistic \
    --episodes 128 \
    --namespace 3000000 \
    --device cuda \
    --random-state 0 \
    --versions v2.6 v3 \
    --v26-checkpoint "${V26_CHECKPOINT}" \
    --v3-checkpoint "${V3_CHECKPOINT}"
