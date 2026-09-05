#!/bin/bash -l
#SBATCH --job-name=slot-tabpfn-resume20k
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Resume one query-routing sweep cell (index = SLURM_ARRAY_TASK_ID) from its
# 5,000-step final_checkpoint.pth out to TARGET_MAX_STEPS.  SOURCE_STUDY_ID
# names the array job the initial 5k run was submitted under, so this reads
# the exact same run directory rather than starting a new one.
#
# For cells whose 5k run has not finished yet, submit this WITH
# --dependency=aftercorr:<SOURCE_STUDY_ID> so SLURM holds the task in PENDING
# (no GPU slot consumed) until the matching source task completes, rather
# than polling inside a live allocation.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SOURCE_STUDY_ID="${SOURCE_STUDY_ID:?SOURCE_STUDY_ID is required}"
readonly TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-20000}"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --label "${SLURM_ARRAY_TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --index "${SLURM_ARRAY_TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS

readonly RUN_DIR="${PROJECT_DIR}/runs/slot_tabpfn/${SOURCE_STUDY_ID}/${TRIAL_LABEL}"
readonly RESUME_CKPT="${RUN_DIR}/final_checkpoint.pth"

echo "task=${SLURM_ARRAY_TASK_ID} label=${TRIAL_LABEL} run_dir=${RUN_DIR} target_max_steps=${TARGET_MAX_STEPS}"
echo "started_at=$(date --iso-8601=seconds)"

if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "final_checkpoint.pth missing at ${RESUME_CKPT} -- source task has not completed" >&2
    exit 2
fi

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda --require-cuda \
    --resume-from "${RESUME_CKPT}" \
    ${TRIAL_FLAGS} --max-steps "${TARGET_MAX_STEPS}"

echo "finished_at=$(date --iso-8601=seconds)"
