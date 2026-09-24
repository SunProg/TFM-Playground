#!/bin/bash -l
#SBATCH --job-name=table-slot-factorial-v3
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the CREATE environment Python}"
readonly GATE="${CODE_ROOT}/preflight.json"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${1:-}" == "preflight" ]]; then
  srun "${PYTHON_BIN}" tfmplayground/experiments/table_slot_factorial_v3.py \
    --device cuda --preflight --preflight-record "${GATE}"
else
  readonly RUN_DIR="${CODE_ROOT}/runs/${SLURM_ARRAY_JOB_ID}/task-${SLURM_ARRAY_TASK_ID}"
  srun "${PYTHON_BIN}" tfmplayground/experiments/table_slot_factorial_v3.py \
    --device cuda --index "${SLURM_ARRAY_TASK_ID}" --output "${RUN_DIR}" \
    --preflight-record "${GATE}"
fi
