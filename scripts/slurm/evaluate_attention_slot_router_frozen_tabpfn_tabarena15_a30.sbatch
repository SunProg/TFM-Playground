#!/bin/bash -l
#SBATCH --job-name=frozen-tabpfn-slot-tabarena15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# TabArena-15 evaluation of the best checkpoint from the mix_scm training run
# (job 37203197: 10,000 steps, best validation NLL 0.347 at step 8750).
# --max-predictors 30 is the 15-dataset TabArena protocol (adapted from
# scratchpad/evaluate_attention_slot_router_tabarena_a30.sbatch, the
# equivalent pipeline for the NanoTabPFNModel-backed AttentionSlotRouter).
#
# The training run's own output directory is flat (no task-N nesting, since
# it was a single non-array job), but wrap_attention_slot_router_checkpoints_
# for_tabarena.py's directory scan expects run_root/task-<int>/ -- a symlink
# shim satisfies that without changing the wrap script's scanning logic or
# moving the actual run directory.
#
# tabpfn_device stays "cpu" during eval regardless of --device cuda:
# evaluate_tabarena_small.py's own call site never threads --device through
# to load_checkpoint_for_inference (see attention_slot_router_tabarena.py's
# build_attention_slot_router_frozen_tabpfn_inference docstring) -- real
# TabPFN's own inference is CPU-capable and fast enough at TabArena's
# small-protocol sizes (subsample <= 2048).
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TRAIN_RUN_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm"
readonly WRAP_ROOT="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm_tabarena15/wrap_root"
readonly WRAPPED_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm_tabarena15/wrapped"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm_tabarena15/eval"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${TRAIN_RUN_DIR}/best.pth" ]] || { echo "Missing ${TRAIN_RUN_DIR}/best.pth" >&2; exit 3; }
mkdir -p "${WRAP_ROOT}" "${WRAPPED_DIR}"
ln -sfn "${TRAIN_RUN_DIR}" "${WRAP_ROOT}/task-0"
# evaluate_tabarena_small.py's own run() does output_dir.mkdir(parents=True,
# exist_ok=False) and refuses to start if it already exists.
[[ -d "${OUTPUT_DIR}" ]] && { echo "Output dir already exists: ${OUTPUT_DIR}" >&2; exit 4; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TABPFN_DISABLE_TELEMETRY=1

echo "job=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${PYTHON_BIN}" scripts/wrap_attention_slot_router_checkpoints_for_tabarena.py \
    --run-root "${WRAP_ROOT}" --output-root "${WRAPPED_DIR}" --task 0 \
    --model-kind attention_slot_router_frozen_tabpfn

ENTRY="$("${PYTHON_BIN}" -c "import json; m=json.load(open('${WRAPPED_DIR}/manifest.json')); r=[x for x in m if x['task']==0][0]; print(f\"{r['name']}={r['path']}\")")"
readonly ENTRY
test -n "${ENTRY}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${ENTRY}" --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" --device cuda --subsample 2048 --folds 5 --repeats 10 \
    --max-predictors 30 \
    --no-include-sklearn --no-include-tabpfn --label-source real --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"
