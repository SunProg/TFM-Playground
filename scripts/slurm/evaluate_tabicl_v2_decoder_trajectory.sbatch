#!/bin/bash -l
#SBATCH --job-name=tabicl-traj-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SLOT_SITE_DIR="${PROJECT_DIR}/.tabicl-slot-site"
readonly TRAJ_ROOT="${TRAJECTORY_ROOT:?TRAJECTORY_ROOT is required}"
readonly STEP="${TRAJECTORY_STEP:?TRAJECTORY_STEP is required}"
readonly VARIANT_SET="${TRAJECTORY_VARIANTS:-all6}"
readonly TAG="${TRAJECTORY_TAG:-all6}"
readonly INCLUDE_BASELINE="${TRAJECTORY_INCLUDE_BASELINE:-false}"
readonly ZERO_SHOT_CHECKPOINT="runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577418/slot-k4-seed2402/step-500.ckpt"
readonly IMBALANCE_CSV="${PROJECT_DIR}/results/tabicl_v2_slot_decoder/tabarena-lite-37581725/class_imbalance.csv"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${SLOT_SITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export TABPFN_DISABLE_TELEMETRY=1

case "${VARIANT_SET}" in
    mlp-only) VARIANT_LIST=(mlp-random) ;;
    slots2) VARIANT_LIST=(slot-k4 slot-k16) ;;
    slots3) VARIANT_LIST=(slot-k4 slot-k8 slot-k16) ;;
    initial4) VARIANT_LIST=(slot-k4 slot-k8 slot-k16 mlp-random) ;;
    initial5) VARIANT_LIST=(slot-k4 slot-k8 slot-k16 mlp-random tabpfnv3-random) ;;
    all6) VARIANT_LIST=(slot-k4 slot-k8 slot-k16 slot-k32 mlp-random tabpfnv3-random) ;;
    *) echo "Unknown trajectory variant set: ${VARIANT_SET}" >&2; exit 2 ;;
esac

EVAL_ARGS=()
for variant in "${VARIANT_LIST[@]}"; do
    checkpoint="${TRAJ_ROOT}/${variant}-seed2402/step-${STEP}.ckpt"
    test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 3; }
    EVAL_ARGS+=(--decoder "${variant}-seed2402-step${STEP}=${checkpoint}")
done
if [[ "${INCLUDE_BASELINE}" == "true" ]]; then
    :
else
    EVAL_ARGS+=(--skip-zero-shot)
fi
if [[ "${TRAJECTORY_CLASSIFICATION_ONLY:-false}" == "true" ]]; then
    EVAL_ARGS+=(--classification-only)
fi

echo "TabArena trajectory evaluation: step=${STEP}, tag=${TAG}, variants=${VARIANT_SET}, include_baseline=${INCLUDE_BASELINE}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabicl_v2_slot_decoder \
    "${EVAL_ARGS[@]}" \
    --output-dir "${TRAJ_ROOT}/tabarena/step-${STEP}-${TAG}" \
    --results-dir "tabicl_v2_decoder_trajectory_s2402_step_${STEP}_${TAG}" \
    --imbalance-csv "${IMBALANCE_CSV}"
