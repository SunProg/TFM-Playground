#!/bin/bash -l
#SBATCH --job-name=tabicl-v2-slot
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=92
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One full-node 8-GPU run at a time; DDP splits the official global batch of
# 64 across ranks. TabICL's graph_scm prior and upstream training code remain
# unchanged.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TORCHRUN_BIN="${PROJECT_DIR}/.venv/bin/torchrun"
readonly SLOT_SITE_DIR="${PROJECT_DIR}/.tabicl-slot-site"
readonly NPROC_PER_NODE=8
readonly SLOT_COUNTS=(4 4 4 8 8 8 16 16 16)
readonly SEEDS=(2402 2403 2404 2402 2403 2404 2402 2403 2404)
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SLOT_COUNT="${SLOT_COUNTS[${INDEX}]}"
readonly SEED="${SEEDS[${INDEX}]}"
readonly MAX_STEPS="${SLOT_MAX_STEPS:-10000}"
readonly RUN_ROOT="${SLOT_RUN_ROOT:-${PROJECT_DIR}/runs/tabicl_v2_slot_decoder}"
readonly RUN_DIR="${RUN_ROOT}/study-${SLURM_ARRAY_JOB_ID}/slot-k${SLOT_COUNT}-seed${SEED}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python environment: ${PYTHON_BIN}" >&2; exit 2; }
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

echo "job=${SLURM_ARRAY_JOB_ID}_${INDEX} seed=${SEED} slots=${SLOT_COUNT} steps=${MAX_STEPS} ranks=${NPROC_PER_NODE} run_dir=${RUN_DIR}"
"${PYTHON_BIN}" -c 'import importlib.metadata, torch, tabicl; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "tabicl", importlib.metadata.version("tabicl"), tabicl.__file__, "gpus", torch.cuda.device_count())'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

TRAIN_ARGS=(
    --head-kind slot
    --slot-count "${SLOT_COUNT}"
    --device cuda
    --wandb_log False
    --max_steps "${MAX_STEPS}"
    --batch_size 64
    --micro_batch_size 1
    --np_seed "${SEED}"
    --torch_seed "${SEED}"
    --checkpoint_dir "${RUN_DIR}"
)
if [[ -n "${TABICL_BASE_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--base-checkpoint "${TABICL_BASE_CHECKPOINT}")
fi
srun "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder "${TRAIN_ARGS[@]}"
