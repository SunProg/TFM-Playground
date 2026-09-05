#!/bin/bash -l
#SBATCH --job-name=vanilla-coh2-rerun15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=44,46,47
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Re-run the three vanilla-coh2 cells (44 plain, 46 mixed, 47 curriculum) with
# the *15-task* TabArena protocol as the in-training validation, and with
# per-epoch checkpoints retained.
#
# Why: the original run (job 36982768) validated on --tabarena-max-predictors
# 10, which admits only 5 datasets, while the post-hoc comparison that ranked
# the three priors used 15.  The 5-task curve therefore cannot say whether the
# 15-task ranking is real or an artifact of which step got frozen as
# final_checkpoint.pth -- and the original --checkpoint-interval 10000 (larger
# than max_steps 5000) meant no intermediate weights survived to check.
#
# Two overrides are appended after ${TRIAL_FLAGS}; argparse takes the last
# occurrence, so these win over the sweep's own values without touching the
# frozen sweep grid:
#   --tabarena-max-predictors 30   the same 15 datasets the comparison used
#   --checkpoint-interval 500      one step-NNNNNN-checkpoint.pth per epoch,
#                                  so any step can be re-scored afterwards
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly DUMP_DIR="${PROJECT_DIR}/runs/multiregime_dump"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --label "${SLURM_ARRAY_TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --index "${SLURM_ARRAY_TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS

# These cells are all `--multiregime-dump none`, so the dump is not required;
# fail loudly rather than silently if that ever stops being true.
if [[ "${TRIAL_FLAGS}" != *"--multiregime-dump none"* && ! -d "${DUMP_DIR}" ]]; then
    echo "Missing multiregime dump at ${DUMP_DIR}; run dump_multiregime_episodes.sbatch first" >&2
    exit 2
fi
if [[ "${TRIAL_LABEL}" != *vanilla-coh2 ]]; then
    echo "Index ${SLURM_ARRAY_TASK_ID} is ${TRIAL_LABEL}, not a vanilla-coh2 cell" >&2
    exit 2
fi

readonly RUN_DIR="${PROJECT_DIR}/runs/slot_tabpfn/${STUDY_ID}/${TRIAL_LABEL}"
readonly ROLLING_CHECKPOINT="${RUN_DIR}/epoch-latest-checkpoint.pth"
RESUME_FLAGS=()
if [[ -f "${ROLLING_CHECKPOINT}" ]]; then
    echo "resuming_from=${ROLLING_CHECKPOINT}"
    RESUME_FLAGS=(--resume-from "${ROLLING_CHECKPOINT}")
fi
readonly RESUME_FLAGS

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "trial_label=${TRIAL_LABEL}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    "${RESUME_FLAGS[@]}" \
    ${TRIAL_FLAGS} \
    --tabarena-max-predictors 30 \
    --checkpoint-interval 500

echo "finished_at=$(date --iso-8601=seconds)"
