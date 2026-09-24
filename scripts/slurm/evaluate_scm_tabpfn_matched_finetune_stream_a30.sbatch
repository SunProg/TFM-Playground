#!/bin/bash -l
#SBATCH --job-name=scm-tabpfn-fixed-paired
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/scm_tabpfn_fixed_weights_matched_finetune_stream"
readonly OUTPUT_PATH="${OUTPUT_DIR}/${SLURM_JOB_ID}.json"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" scripts/evaluate_scm_tabpfn_v2.py \
    --output "${OUTPUT_PATH}" \
    --prior-profile realistic \
    --episodes 128 \
    --namespace 4000000 \
    --device cuda \
    --random-state 0 \
    --versions v2.2 v2.6 v3 \
    --v26-checkpoint "${PROJECT_DIR}/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt" \
    --v3-checkpoint "${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt" \
    --skip-shuffled-support

echo "finished_at=$(date --iso-8601=seconds) output=${OUTPUT_PATH}"
