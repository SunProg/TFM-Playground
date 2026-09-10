#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-pilot
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_ROOT="${CODE_ROOT}/runs/${STUDY_ID}"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_ARRAY_JOB_ID}_${INDEX} source=${CODE_ROOT} output=${RUN_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The implementation gate is rerun on the allocated GPU and exact snapshot.
# Failure stops the cell before any training. Existing v2 gates are untouched.
srun uv run --project "${CODE_ROOT}" python -m tfmplayground.experiments.pretrain_multiregime_v3 \
    --preflight --device cuda --output "${RUN_ROOT}/gate-${INDEX}"
srun uv run --project "${CODE_ROOT}" python -m tfmplayground.experiments.pretrain_multiregime_v3 \
    --index "${INDEX}" --device cuda --output "${RUN_ROOT}/cell-${INDEX}" \
    --gate "${RUN_ROOT}/gate-${INDEX}/execution_gate.json"
