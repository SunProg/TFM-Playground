#!/bin/bash -l
#SBATCH --job-name=tabicl-mlp-traj
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TORCHRUN_BIN="${PROJECT_DIR}/.venv/bin/torchrun"
readonly SLOT_SITE_DIR="${PROJECT_DIR}/.tabicl-slot-site"
readonly SEED="${MLP_SEED:-2402}"
readonly MAX_STEPS="${MLP_MAX_STEPS:-10000}"
readonly RUN_DIR="${MLP_RUN_DIR:-${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-mlp-random/job-${SLURM_JOB_ID}/mlp-random-seed${SEED}}"

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

echo "STAGE1_MLP_RANDOM job=${SLURM_JOB_ID} gpu_count=1 steps=${MAX_STEPS} seed=${SEED} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

TRAIN_ARGS=(
    --head-kind mlp
    --mlp-init random
    --device cuda
    --wandb_log False
    --max_steps "${MAX_STEPS}"
    --batch_size 64
    --micro_batch_size 4
    --np_seed "${SEED}"
    --torch_seed "${SEED}"
    --lr 8e-4
    --warmup_proportion 0.2
    --gradient_clipping 10.0
    --prior_type graph_scm
    --prior_device cpu
    --n_jobs 16
    --batch_size_per_gp 4
    --min_features 1
    --max_features 100
    --max_classes 10
    --fixed-seq-len 1024
    --log_seq_len False
    --seq_len_per_gp True
    --min_train_size 0.3
    --max_train_size 0.9
    --use_flash_attn3 False
    --save_temp_every 500
    --save_perm_every 500
    --checkpoint_dir "${RUN_DIR}"
)
START_SECONDS=$(date +%s)
srun "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=1 \
    -m tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder "${TRAIN_ARGS[@]}"
END_SECONDS=$(date +%s)
echo "STAGE1_MLP_RANDOM_RESULT job=${SLURM_JOB_ID} steps=${MAX_STEPS} total_seconds=$((END_SECONDS - START_SECONDS)) run_dir=${RUN_DIR}"
