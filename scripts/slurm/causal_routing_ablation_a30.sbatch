#!/bin/bash -l
#SBATCH --job-name=causal-routing-ablation
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the CREATE environment Python}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${CODE_ROOT}/runs/${STUDY_ID}/task-${SLURM_ARRAY_TASK_ID}"

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} code=${CODE_ROOT} output=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.causal_routing_ablation \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --output "${RUN_DIR}" \
  --device cuda \
  --width 192 --hidden 768 --layers 6 --heads 6 --steps "${STEPS:-5000}" \
  --support-size 128 --query-size 256 --batch-size 8 \
  --learning-rate 0.0001 --warmup-steps 2000 \
  --validation-interval "${VALIDATION_INTERVAL:-200}" \
  --validation-episodes "${VALIDATION_EPISODES:-32}" \
  --validation-query-size 256 \
  --test-episodes "${TEST_EPISODES:-128}" --test-query-size 256
