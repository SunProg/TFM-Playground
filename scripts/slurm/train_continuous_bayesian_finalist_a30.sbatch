#!/bin/bash -l
#SBATCH --job-name=cont-bayes-final
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Final training of one selected screening configuration with three seeds.
#
# Submit with the screening array index of the selected configuration, e.g.
#   SCREENING_INDEX=7 sbatch scripts/slurm/train_continuous_bayesian_finalist_a30.sbatch

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SCREENING_INDEX="${SCREENING_INDEX:?Set SCREENING_INDEX to the selected screening array index}"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.continuous_sweep --label "${SCREENING_INDEX}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.continuous_sweep \
    --index "${SCREENING_INDEX}" --final --seed "${SEED}")"
readonly TRIAL_LABEL TRIAL_FLAGS
readonly RUN_DIR="${PROJECT_DIR}/runs/continuous_bayesian/final-${TRIAL_LABEL}-seed${SEED}-${SLURM_ARRAY_JOB_ID}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "screening_index=${SCREENING_INDEX}"
echo "trial_label=${TRIAL_LABEL}"
echo "seed=${SEED}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_continuous_bayesian \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --batch-size 2 \
    --accumulate-gradients 2 \
    --validation-episodes 16 \
    --validation-support-size 128 \
    --validation-query-count 6 \
    ${TRIAL_FLAGS}

echo "finished_at=$(date --iso-8601=seconds)"
