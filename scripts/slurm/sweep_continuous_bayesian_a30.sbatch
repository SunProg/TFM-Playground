#!/bin/bash -l
#SBATCH --job-name=cont-bayes-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-33%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Screening sweep for slot-free continuous Bayesian uncertainty.
#
# 34 configurations: 18 adapter-continuous, 6 frozen-continuous, 4 fully
# trainable uncertainty copies, and 6 Beta-adapter ablations.  The array runs
# eight tasks concurrently; do not lower that limit.

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.continuous_sweep --label "${TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.continuous_sweep --index "${TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS
readonly SWEEP_DIR="${PROJECT_DIR}/runs/continuous_bayesian/sweep-${SLURM_ARRAY_JOB_ID}"
readonly RUN_DIR="${SWEEP_DIR}/task-${TASK_ID}-${TRIAL_LABEL}"

mkdir -p "${SWEEP_DIR}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "trial_label=${TRIAL_LABEL}"
echo "trial_flags=${TRIAL_FLAGS}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

# --require-cuda asserts torch.cuda.is_available() inside the trainer.
srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_continuous_bayesian \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed 2402 \
    --batch-size 2 \
    --accumulate-gradients 2 \
    --validation-episodes 12 \
    --validation-support-size 128 \
    --validation-query-count 6 \
    ${TRIAL_FLAGS}

echo "finished_at=$(date --iso-8601=seconds)"
