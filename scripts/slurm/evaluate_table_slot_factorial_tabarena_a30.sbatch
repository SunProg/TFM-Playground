#!/bin/bash -l
#SBATCH --job-name=table-slot-v3-tabarena
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the CREATE environment Python}"
readonly STUDY_ID="${STUDY_ID:?Set STUDY_ID to the TableSlot factorial training array job id}"
readonly RUN_ROOT="${CODE_ROOT}/runs/${STUDY_ID}"
readonly CHECKPOINT="${RUN_ROOT}/task-${SLURM_ARRAY_TASK_ID}/best.pth"
readonly OUTPUT_DIR="${CODE_ROOT}/tabarena_table_slot_factorial/${SLURM_ARRAY_JOB_ID}/task-${SLURM_ARRAY_TASK_ID}"
readonly OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/nanotabpfn.pth}"

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "${CHECKPOINT}"
test -f "${OFFICIAL_CHECKPOINT}"
mkdir -p "${CODE_ROOT}/.cache/openml"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} study=${STUDY_ID} checkpoint=${CHECKPOINT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Each evaluator task starts after its corresponding training task through
# aftercorr:${STUDY_ID}. It does not wait for the rest of the training array.
srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
  --official-checkpoint "${OFFICIAL_CHECKPOINT}" \
  --standalone-checkpoints "table_slot_factorial=${CHECKPOINT}" \
  --include-vanilla \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --max-predictors 30 \
  --subsample 2048 \
  --folds 5 \
  --repeats 10 \
  --no-include-sklearn \
  --no-include-tabpfn \
  --label-source real \
  --contamination 0.0 \
  --cache-directory "${CODE_ROOT}/.cache/openml"

echo "finished_at=$(date --iso-8601=seconds)"
