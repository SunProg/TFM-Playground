#!/bin/bash -l
#SBATCH --job-name=orig-w02
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# `original` (plain mix_scm prior, dumped ordinary tasks) retrained with the
# 20 % warmup used by the rg_z runs, as their like-for-like reference. Same
# recipe as the v4 baselines (37283915/37283921/37283995) otherwise. Task id =
# size (0 small, 1 medium, 2 large; run large on biomed_a100_gpu):
#   sbatch --array=0,1 scripts/slurm/pretrain_original_warmup.sbatch
#   sbatch --partition=biomed_a100_gpu --array=2 scripts/slurm/pretrain_original_warmup.sbatch
# WARMUP_PROPORTION env (default 0.2) also names the run: original-w02-<size>.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4"
readonly WARMUP_PROPORTION="${WARMUP_PROPORTION:-0.2}"
readonly TAG="w$(printf '%s' "${WARMUP_PROPORTION}" | tr -d '.')"   # 0.2 -> w02
readonly -a SIZES=(small medium large)
readonly SIZE="${SIZES[${SLURM_ARRAY_TASK_ID}]}"
readonly FAMILY="original-${TAG}"
case "${SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
esac
readonly ORIGINAL_DUMP="${DATA}/tabicl_mix_scm_paired/37277254/original.h5"
readonly BANK_DIR="${DATA}/tabicl_mix_scm_evaluation/37277969"   # repaired bank
readonly EXTRA_BANK="${DATA}/tabicl_mix_scm_evaluation_expose_z/validation.h5"   # also scored, as v4_validation_zx_*
# Lives next to the rg_z runs (same warmup batch) so the eval scripts' rg_z group covers it.
readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/rg_z/${FAMILY}-${SIZE}/seed-2402"
readonly LINK_DIR="${PROJECT_DIR}/runs_exp/rg_z/${FAMILY}-${SIZE}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${ORIGINAL_DUMP}"
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5" && test -f "${EXTRA_BANK}"
mkdir -p "$(dirname "${RUN_DIR}")" "$(dirname "${LINK_DIR}")"
[[ -e "${LINK_DIR}" ]] || ln -s "$(dirname "${RUN_DIR}")" "${LINK_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
resume_args=()
if [[ "${RESUME:-0}" == 1 ]]; then   # continue from latest_checkpoint.pth (every 50 steps) or newest checkpoint-*.pth
    if [[ -f "${RUN_DIR}/latest_checkpoint.pth" ]]; then latest="${RUN_DIR}/latest_checkpoint.pth"; else latest="$(ls -1 "${RUN_DIR}"/checkpoint-*.pth 2>/dev/null | sort | tail -1 || true)"; fi
    [[ -n "${latest}" ]] && resume_args=(--resume-checkpoint "${latest}")
fi
echo "family=${FAMILY} size=${SIZE} run_dir=${RUN_DIR} warmup_proportion=${WARMUP_PROPORTION} resume=${resume_args[*]:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" --device cuda --require-cuda ${resume_args[@]+"${resume_args[@]}"} \
    --seed 2402 --max-steps 10000 \
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 --warmup-proportion "${WARMUP_PROPORTION}" \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 \
    --v4-validation-bank-path "${BANK_DIR}/validation.h5" --v4-validation-interval 500 \
    --v4-extra-validation-bank-path "${EXTRA_BANK}" --v4-extra-validation-tag zx \
    --v4-test-bank-path "${BANK_DIR}/test.h5" \
    --checkpoint-interval 2000 --latest-checkpoint-interval 50 --epoch-steps 500 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --prior-mode original --multiregime-source v4 --v4-shared-dump-path "${ORIGINAL_DUMP}"
