#!/bin/bash -l
#SBATCH --job-name=tabicl-slot-scale
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TORCHRUN_BIN="${PROJECT_DIR}/.venv/bin/torchrun"
readonly SLOT_SITE_DIR="${PROJECT_DIR}/.tabicl-slot-site"
readonly NPROC_PER_NODE="${SLOT_NPROC_PER_NODE:?Set SLOT_NPROC_PER_NODE to the allocated GPU count}"
readonly STEPS="${SLOT_BENCH_STEPS:-12}"
readonly SLOT_COUNT="${SLOT_BENCH_SLOT_COUNT:-16}"
readonly SEED="${SLOT_BENCH_SEED:-2402}"
readonly RUN_ROOT="${SLOT_RUN_ROOT:-${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/scaling}"
readonly RUN_DIR="${RUN_ROOT}/world-${NPROC_PER_NODE}-steps-${STEPS}-job-${SLURM_JOB_ID}"

if [[ "${NPROC_PER_NODE}" -ge 8 ]]; then
    readonly N_JOBS=10
else
    readonly N_JOBS=16
fi

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${TORCHRUN_BIN}" ]] || { echo "Missing torchrun: ${TORCHRUN_BIN}" >&2; exit 2; }
[[ -d "${SLOT_SITE_DIR}/tabicl" ]] || { echo "Missing TabICL 2.2 overlay: ${SLOT_SITE_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${SLOT_SITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TABPFN_DISABLE_TELEMETRY=1

echo "BENCH_START job=${SLURM_JOB_ID} nproc=${NPROC_PER_NODE} n_jobs=${N_JOBS} steps=${STEPS} slots=${SLOT_COUNT} seed=${SEED} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

TRAIN_ARGS=(
    --head-kind slot
    --slot-count "${SLOT_COUNT}"
    --device cuda
    --wandb_log False
    --max_steps "${STEPS}"
    --batch_size 64
    --micro_batch_size 1
    --np_seed "${SEED}"
    --torch_seed "${SEED}"
    --n_jobs "${N_JOBS}"
    --checkpoint_dir "${RUN_DIR}"
)
START_SECONDS=$(date +%s)
srun "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder "${TRAIN_ARGS[@]}"
END_SECONDS=$(date +%s)
echo "BENCH_RESULT job=${SLURM_JOB_ID} nproc=${NPROC_PER_NODE} steps=${STEPS} total_seconds=$((END_SECONDS - START_SECONDS)) run_dir=${RUN_DIR}"
