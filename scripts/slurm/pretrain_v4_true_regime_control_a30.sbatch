#!/bin/bash -l
#SBATCH --job-name=v4-true-regime
#SBATCH --partition=biomed_a100_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Sibling of pretrain_v4_hidden_regime_control_a30.sbatch: identical prior,
# curriculum, architecture, and seed, WITH --v4-expose-regime-id. Submitted
# alongside the hidden-regime control (not reusing the already-running
# v4-observed-regime jobs) so hidden vs. true can be compared step-for-step
# from the same start time rather than at mismatched step counts.
# Functionally the same training as pretrain_v4_observed_regime_large_a100.sbatch;
# this copy just writes into the true_regime_control/ namespace so it lines up
# 1:1 with hidden_regime_control/ under the same directory scheme.
# Set RESUME=1 to continue an interrupted run from its atomic 50-step latest
# checkpoint; this permits a short backfill allocation before the full run.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${V4_TRUE_REGIME_PYTHON:-/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python}"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4_native"
readonly PRIOR_DIR="${DATA}/tabicl_mix_scm_paired_zmix5"
readonly BANK="${DATA}/tabicl_mix_scm_evaluation/test.h5"
readonly MODEL_SIZE="${MODEL_SIZE:-large}"
case "${MODEL_SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
    *) echo "MODEL_SIZE must be small, medium, or large" >&2; exit 2 ;;
esac
readonly RUN_DIR="${V4_TRUE_REGIME_RUN_DIR:-${PROJECT_DIR}/artifacts/runs/true_regime_control/rg_z-curriculum-${MODEL_SIZE}/seed-2402}"
readonly DIAGNOSTIC_OUTPUT="${V4_TRUE_REGIME_EVAL:-${PROJECT_DIR}/artifacts/evaluations/true_regime_control/rg_z-curriculum-${MODEL_SIZE}/seed-2402/test.json}"
readonly RESUME="${RESUME:-0}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
for f in original r_z-multiregime g_z-multiregime; do test -f "${PRIOR_DIR}/${f}.h5"; done
test -f "${BANK}"
resume_args=()
if [[ "${RESUME}" == 1 ]]; then
    [[ -d "${RUN_DIR}" ]] || { echo "cannot resume missing run directory ${RUN_DIR}" >&2; exit 3; }
    if [[ -f "${RUN_DIR}/latest_checkpoint.pth" ]]; then
        latest="${RUN_DIR}/latest_checkpoint.pth"
    else
        latest="$(ls -1 "${RUN_DIR}"/checkpoint-*.pth 2>/dev/null | sort | tail -1 || true)"
    fi
    [[ -n "${latest}" ]] || { echo "cannot resume without a checkpoint under ${RUN_DIR}" >&2; exit 3; }
    resume_args=(--resume-checkpoint "${latest}")
else
    [[ ! -e "${RUN_DIR}" ]] || { echo "refusing to overwrite ${RUN_DIR}" >&2; exit 3; }
fi
mkdir -p "$(dirname "${RUN_DIR}")" "$(dirname "${DIAGNOSTIC_OUTPUT}")"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model_size=${MODEL_SIZE} layers=${LAYERS} resume=${RESUME} run_dir=${RUN_DIR}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    ${resume_args[@]+"${resume_args[@]}"} \
    --device cuda --require-cuda --seed 2402 --max-steps 10000 \
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 --warmup-proportion 0.2 \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 \
    --checkpoint-interval 2000 --latest-checkpoint-interval 50 --epoch-steps 500 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --prior-mode curriculum --multiregime-ratio 0.5 --multiregime-source v4 \
    --v4-shared-dump-path "${PRIOR_DIR}/original.h5" \
    --v4-dump-path "${PRIOR_DIR}/r_z-multiregime.h5,${PRIOR_DIR}/g_z-multiregime.h5" \
    --v4-expose-regime-id \
    --v4-regime-info-bank-path "${BANK}" \
    --v4-regime-info-interval 1000 \
    --v4-regime-info-cell-subsample-modulus 16

srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
    --checkpoint "${RUN_DIR}/final_checkpoint.pth" --bank "${BANK}" \
    --output "${DIAGNOSTIC_OUTPUT}" --device cuda --multiclass-only --max-episodes-per-forward 8
