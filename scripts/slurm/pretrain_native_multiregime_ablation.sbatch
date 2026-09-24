#!/bin/bash -l
#SBATCH --job-name=mr-ablation
#SBATCH --partition=biomed_a100_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Which part of the multiregime prior actually pays? Three medium runs identical to
# rg_z-fixed-medium (native dumps, 30 % multiregime, 10k steps, 20 % warmup) except that the
# MULTIREGIME branch is filtered; the ordinary K=1 branch is untouched.
#   0 soft_gate-only   drop persistent episodes (the routing regime is unidentifiable there,
#                      so they can only teach hedging)
#   1 multiclass-only  drop binary multiregime episodes (a K-regime binary task collapses into
#                      one staircase boundary, so there is no regime-conditional skill to learn)
#   2 both             soft_gate AND >=3 classes
# Control: rg_z-fixed-medium (same recipe, unfiltered).
# SCHEDULE=curriculum (env) switches the multiregime share from a constant 0.3 to the 0->0.5 ramp of
# rg_z-curriculum-medium, whose control is that run; the run directory gets the schedule in its name.
# SIZE=large (env) runs the large architecture instead of medium (control: rg_z-{fixed,curriculum}-large).
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4_native"
readonly PRIOR_DIR="${DATA}/tabicl_mix_scm_paired_zmix5"
readonly BANK_DIR="${DATA}/tabicl_mix_scm_evaluation"
readonly EXTRA_BANK="${DATA}/tabicl_mix_scm_evaluation_expose_z/validation.h5"
readonly -a TAGS=(softgate-only multiclass-only softgate-multiclass)
readonly TAG="${TAGS[${SLURM_ARRAY_TASK_ID}]}"
case "${SLURM_ARRAY_TASK_ID}" in
    0) readonly -a FILTER=(--v4-dump-family soft_gate) ;;
    1) readonly -a FILTER=(--v4-dump-min-classes 3) ;;
    2) readonly -a FILTER=(--v4-dump-family soft_gate --v4-dump-min-classes 3) ;;
esac
# RUN_SUFFIX lets a second submission (e.g. the A30 hedge) write to its own directory instead of
# colliding with the A100 copy of the same arm.
readonly RUN_SUFFIX="${RUN_SUFFIX:-}"
readonly SCHEDULE="${SCHEDULE:-fixed}"
readonly SIZE="${SIZE:-medium}"
case "${SIZE}" in
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
    *) echo "SIZE must be medium or large" >&2; exit 2 ;;
esac
case "${SCHEDULE}" in
    fixed)      readonly RATIO=0.3 ;;
    curriculum) readonly RATIO=0.5 ;;   # curriculum ramps 0 -> RATIO over the first half of training
    *) echo "SCHEDULE must be fixed or curriculum" >&2; exit 2 ;;
esac
readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/native_ablation/rg_z-${SCHEDULE}-${TAG}-${SIZE}${RUN_SUFFIX}/seed-2402"
readonly RESUME="${RESUME:-0}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
for f in original r_z-multiregime g_z-multiregime; do test -f "${PRIOR_DIR}/${f}.h5"; done
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5" && test -f "${EXTRA_BANK}"
mkdir -p "$(dirname "${RUN_DIR}")"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
resume_args=()
if [[ "${RESUME}" == 1 && -f "${RUN_DIR}/latest_checkpoint.pth" ]]; then
    resume_args=(--resume-checkpoint "${RUN_DIR}/latest_checkpoint.pth")
fi
echo "ablation=${TAG} size=${SIZE} schedule=${SCHEDULE} ratio=${RATIO} filter=${FILTER[*]} run_dir=${RUN_DIR} resume=${resume_args[*]:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda --require-cuda \
    --seed 2402 --max-steps 10000 \
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 --warmup-proportion 0.2 \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 \
    --v4-validation-bank-path "${BANK_DIR}/validation.h5" --v4-validation-interval 500 \
    --v4-extra-validation-bank-path "${EXTRA_BANK}" --v4-extra-validation-tag zx \
    --v4-test-bank-path "${BANK_DIR}/test.h5" \
    --checkpoint-interval 2000 --latest-checkpoint-interval 50 --epoch-steps 500 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" \
    --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --prior-mode "${SCHEDULE}" --multiregime-ratio "${RATIO}" --multiregime-source v4 \
    --v4-shared-dump-path "${PRIOR_DIR}/original.h5" \
    --v4-dump-path "${PRIOR_DIR}/r_z-multiregime.h5,${PRIOR_DIR}/g_z-multiregime.h5" \
    "${FILTER[@]}" ${resume_args[@]+"${resume_args[@]}"}
