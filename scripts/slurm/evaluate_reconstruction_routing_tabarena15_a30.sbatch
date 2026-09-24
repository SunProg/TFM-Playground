#!/bin/bash -l
#SBATCH --job-name=recon-routing-tabarena15
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

readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the CREATE environment Python}"
readonly STUDY_ID="${STUDY_ID:?Set STUDY_ID to the reconstruction-routing training array job id}"
readonly CHECKPOINT_ROOT="${CODE_ROOT}/tabarena_reconstruction_checkpoints/${STUDY_ID}"
readonly OUTPUT_DIR="${CODE_ROOT}/tabarena_reconstruction15/${SLURM_JOB_ID}"

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${CODE_ROOT}/.cache/openml"

echo "job=${SLURM_JOB_ID} study=${STUDY_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${PYTHON_BIN}" scripts/wrap_reconstruction_checkpoints_for_tabarena.py \
  --run-root "${CODE_ROOT}/runs/${STUDY_ID}" \
  --output-root "${CHECKPOINT_ROOT}"
readonly CHECKPOINTS="$(cat "${CHECKPOINT_ROOT}/standalone_checkpoints.txt")"
test -n "${CHECKPOINTS}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
  --standalone-checkpoints "${CHECKPOINTS}" \
  --no-include-vanilla \
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
