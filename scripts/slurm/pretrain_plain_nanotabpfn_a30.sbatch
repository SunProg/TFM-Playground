#!/bin/bash -l
#SBATCH --job-name=plain-nanotabpfn
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Five matched ordinary-prior baseline seeds.  Do not alter the training flags
# for a later multiregime arm: the training prior must be the only difference.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(2402 2403 2404 2405 2406)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly STUDY_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn/${SLURM_ARRAY_JOB_ID}"
readonly RUN_DIR="${STUDY_DIR}/seed-${SEED}"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "array_job_id=${SLURM_ARRAY_JOB_ID} seed=${SEED} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed "${SEED}" \
    --max-steps 50000 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --validation-interval 5000 \
    --validation-batches 16 \
    --checkpoint-interval 10000 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --max-classes 2 \
    --prior-type mix_scm
