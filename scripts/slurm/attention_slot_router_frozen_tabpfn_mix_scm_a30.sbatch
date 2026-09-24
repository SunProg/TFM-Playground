#!/bin/bash -l
#SBATCH --job-name=frozen-tabpfn-slot-mix-scm
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Real (non-preflight) training run of AttentionSlotRouterFrozenTabPFN on
# mix_scm (TabICL's own prior: randomly mlp_scm or tree_scm per episode).
# The preflight (job 37188759, --prior scm, 200 steps) already validated the
# mechanism works end-to-end on GPU; this run is scoped to actually see a
# convergence trend (10,000 steps, same batch/accumulate as the preflight --
# ~0.5s/step measured there, so ~1.5h total).
#
# --features 8 is load-bearing, not a tuning choice: real TabPFN's
# preprocessing drops constant (all-zero) columns per episode, and
# OriginalPrior zero-pads each episode's actual feature count up to
# max_features. Without min_features==max_features, different episodes in a
# batch could end up with different post-preprocessing column counts and
# crash FrozenTabPFNBackbone.encode_table's consistency guard -- see
# build_prior()'s comment in attention_slot_router_frozen_tabpfn_scale.py.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"
readonly RUN_DIR="${PROJECT_DIR}/results/attention_slot_router_frozen_tabpfn_mix_scm"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint at ${CHECKPOINT}" >&2; exit 3; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TABPFN_DISABLE_TELEMETRY=1

echo "job=${SLURM_JOB_ID} host=$(hostname) run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.attention_slot_router_frozen_tabpfn_scale \
    --index 0 \
    --output "${RUN_DIR}" \
    --device cuda \
    --prior mix_scm \
    --features 8 \
    --tabpfn-device cuda \
    --tabpfn-model-path "${CHECKPOINT}" \
    --steps 10000 \
    --hidden 768 \
    --support-size 128 --query-size 32 \
    --batch-size 4 --accumulate 2 \
    --warmup-steps 500 \
    --validation-interval 250 --validation-episodes 16 \
    --learning-rate 0.0001
