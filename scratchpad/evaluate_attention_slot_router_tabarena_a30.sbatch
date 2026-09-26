#!/bin/bash -l
#SBATCH --job-name=asr-tabarena
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# One post-training TabArena-15 evaluation for one AttentionSlotRouter
# training task. Submit once per task index with
# --dependency=afterok:<TRAIN_JOB_ID>_<TASK_INDEX> so evaluation only runs if
# that task's own training completed successfully -- each of the 9 cells is
# submitted and gated independently, not as a batch waiting on the whole
# array. --max-predictors 30 is the 15-dataset TabArena protocol (the default
# --max-predictors 10 is the narrower 5-dataset protocol other pilots use).
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly TRAIN_JOB_ID="${TRAIN_JOB_ID:?TRAIN_JOB_ID is required}"
readonly TASK_INDEX="${TASK_INDEX:?TASK_INDEX is required}"
readonly RUN_ROOT="${PROJECT_DIR}/runs/${TRAIN_JOB_ID}"
# Per-task subdirectory, not a shared wrapped/ across every concurrently
# running eval job: each task's wrap step used to overwrite the same
# manifest.json with only its own entry, so a concurrent task could read
# another task's (or a torn/empty) manifest depending on timing.
readonly WRAPPED_DIR="${PROJECT_DIR}/results/attention_slot_router_tabarena15/${TRAIN_JOB_ID}/wrapped/task-${TASK_INDEX}"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/attention_slot_router_tabarena15/${TRAIN_JOB_ID}/task-${TASK_INDEX}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -d "${RUN_ROOT}/task-${TASK_INDEX}"
# Only WRAPPED_DIR, not OUTPUT_DIR: evaluate_tabarena_small.py's own run()
# does output_dir.mkdir(parents=True, exist_ok=False) and refuses to start if
# it already exists -- pre-creating it here made every first submission fail.
mkdir -p "${WRAPPED_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON_BIN}" scripts/wrap_attention_slot_router_checkpoints_for_tabarena.py \
    --run-root "${RUN_ROOT}" --output-root "${WRAPPED_DIR}" --task "${TASK_INDEX}"

# Plain assignment, not `readonly ENTRY="$(...)"`: wrapping a command
# substitution directly in a `readonly`/`local` declaration masks its exit
# code from `set -e` (the declaration's own success is what's checked), so a
# failure here silently fell through to srun with an empty ENTRY instead of
# aborting the script.
ENTRY="$(python3 -c "import json; m=json.load(open('${WRAPPED_DIR}/manifest.json')); r=[x for x in m if x['task']==${TASK_INDEX}][0]; print(f\"{r['name']}={r['path']}\")")"
readonly ENTRY
test -n "${ENTRY}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${ENTRY}" --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" --device cuda --subsample 2048 --folds 5 --repeats 10 \
    --max-predictors 30 \
    --no-include-sklearn --no-include-tabpfn --label-source real --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"
