#!/bin/bash -l
#SBATCH --job-name=multiregime-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-17%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Screening sweep for the multiregime backbone fine-tune.
#
# 18 configurations: learning rate x multiregime curriculum probability x
# support size (tfmplayground/experiments/multiregime_sweep.py). Eight
# concurrent tasks; do not lower that limit.
#
# After all tasks finish:
#   python -m tfmplayground.experiments.multiregime_sweep \
#       --summarize runs/multiregime_backbone/sweep-<ARRAY_JOB_ID> \
#       --output-dir results/multiregime_backbone/sweep

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

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.multiregime_sweep --label "${TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.multiregime_sweep --index "${TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS
readonly SWEEP_DIR="${PROJECT_DIR}/runs/multiregime_backbone/sweep-${SLURM_ARRAY_JOB_ID}"
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

srun "${PYTHON_BIN}" -m tfmplayground.experiments.finetune_multiregime_backbone \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --batch-size 2 \
    --accumulate-gradients 2 \
    --query-count 6 \
    --validation-episodes 16 \
    ${TRIAL_FLAGS}

echo "finished_at=$(date --iso-8601=seconds)"
