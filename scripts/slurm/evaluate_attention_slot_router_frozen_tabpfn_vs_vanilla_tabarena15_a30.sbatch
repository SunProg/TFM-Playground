#!/bin/bash -l
#SBATCH --job-name=frozen-tabpfn-slot-vs-vanilla
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Same TabArena-15 protocol as
# evaluate_attention_slot_router_frozen_tabpfn_tabarena15_a30.sbatch (job
# 37226114), but adds --include-tabpfn so real vanilla TabPFN-v3 is scored
# in the SAME run (same eligible_tasks() selection, same fold seeds) for a
# direct, apples-to-apples AUC comparison against the mix_scm-trained slot
# router -- not a separately-seeded run.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
readonly WRAPPED_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm_tabarena15/wrapped"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm_tabarena15/eval_vs_vanilla"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint at ${CHECKPOINT}" >&2; exit 3; }
[[ -f "${WRAPPED_DIR}/manifest.json" ]] || { echo "Missing ${WRAPPED_DIR}/manifest.json -- run the wrap step first" >&2; exit 3; }
[[ -d "${OUTPUT_DIR}" ]] && { echo "Output dir already exists: ${OUTPUT_DIR}" >&2; exit 4; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TABPFN_DISABLE_TELEMETRY=1

echo "job=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

ENTRY="$("${PYTHON_BIN}" -c "import json; m=json.load(open('${WRAPPED_DIR}/manifest.json')); r=[x for x in m if x['task']==0][0]; print(f\"{r['name']}={r['path']}\")")"
readonly ENTRY
test -n "${ENTRY}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${ENTRY}" --no-include-vanilla \
    --include-tabpfn --tabpfn-model-path "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" --device cuda --subsample 2048 --folds 5 --repeats 10 \
    --max-predictors 30 \
    --no-include-sklearn --label-source real --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"
