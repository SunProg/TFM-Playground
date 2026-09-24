#!/bin/bash -l
#SBATCH --job-name=beyond-v4
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=beyond-v4-%x-%j.out
#SBATCH --error=beyond-v4-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${BEYONDARENA_PYTHON:-/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python}"
readonly RUN_ROOTS="${V4_RUN_ROOTS:-${V4_RUN_ROOT:-${PROJECT_DIR}/runs_exp/multiregime_only},${V4_EXPOSE_Z_RUN_ROOT:-${PROJECT_DIR}/runs_exp/expose_z}}"
readonly RESULT_ROOT="${BEYONDARENA_RESULT_ROOT:-${PROJECT_DIR}/results/beyondarena-v4}"
readonly DATA_CACHE="${BEYONDARENA_DATA_CACHE:-${PROJECT_DIR}/data-foundry-cache}"
readonly ROWS_MAX="${BEYONDARENA_ROWS_MAX:-11000}"
readonly TASK_NAMES_FILE="${BEYONDARENA_TASK_NAMES_FILE:-}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${RESULT_ROOT}" "${DATA_CACHE}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export TABPFN_V22_CHECKPOINT="${TABPFN_V22_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.2.ckpt}"

args=(
    --run-roots "${RUN_ROOTS}"
    --output-dir "${RESULT_ROOT}/${RUN_MODE:-full}-${SLURM_JOB_ID}"
    --data-cache "${DATA_CACHE}"
    --device cuda
    --query-chunk-size 128
    --num-mem-chunks 8
    --rows-max "${ROWS_MAX}"
)
if [[ -n "${SMOKE_TASK:-}" ]]; then
    args+=(--smoke-task "${SMOKE_TASK}")
fi
if [[ -n "${BASELINES:-}" ]]; then
    args+=(--baselines "${BASELINES}")
fi
if [[ -n "${TASK_NAMES_FILE}" ]]; then
    args+=(--task-names-file "${TASK_NAMES_FILE}")
fi

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_multiregime_v4_beyondarena "${args[@]}"
