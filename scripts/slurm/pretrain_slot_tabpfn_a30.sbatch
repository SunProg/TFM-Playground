#!/bin/bash -l
#SBATCH --job-name=slot-tabpfn-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Four slot-TabPFN pretraining arms, one per prior composition:
#   0 plain        single-regime only
#   1 multiregime  multiregime only
#   2 mixed        70% single + 30% multiregime
#   3 curriculum   single first, then ramp the multiregime share to 50%
#
# Every other training flag is identical across the arms and matches
# pretrain_plain_nanotabpfn_a30.sbatch, so the training prior is the only
# difference -- both within this sweep and against the vanilla baseline runs.
# The index-to-flags mapping lives in slot_tabpfn_sweep.py, not here.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --label "${SLURM_ARRAY_TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --index "${SLURM_ARRAY_TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS
readonly STUDY_DIR="${PROJECT_DIR}/runs/slot_tabpfn/${SLURM_ARRAY_JOB_ID}"
readonly RUN_DIR="${STUDY_DIR}/${TRIAL_LABEL}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "trial_label=${TRIAL_LABEL}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    ${TRIAL_FLAGS}

echo "finished_at=$(date --iso-8601=seconds)"
