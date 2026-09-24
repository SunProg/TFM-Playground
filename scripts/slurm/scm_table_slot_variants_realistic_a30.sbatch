#!/bin/bash -l
#SBATCH --job-name=scm-table-variants-5k
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly MODEL_TYPES=(table_slot_backbone table_slot_mufasa)
readonly SEEDS=(11 12)
readonly MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
readonly SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
readonly MODEL_TYPE="${MODEL_TYPES[${MODEL_INDEX}]}"
readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${PROJECT_DIR}/results/scm_table_slot_variants_realistic_5000_slurm/${STUDY_ID}/${MODEL_TYPE}/seed-${SEED}"

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
echo "model=${MODEL_TYPE} seed=${SEED} run_dir=${RUN_DIR} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_table_slot_variants \
    --output "${RUN_DIR}" \
    --prior-profile realistic \
    --model-types "${MODEL_TYPE}" \
    --steps 5000 \
    --seeds "${SEED}" \
    --test-episodes 128 \
    --validation-episodes 32 \
    --device cuda \
    --require-cuda
