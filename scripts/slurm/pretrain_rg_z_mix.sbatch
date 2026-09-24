#!/bin/bash -l
#SBATCH --job-name=rg-z-mix
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Mixed r_z + g_z multiregime prior ("rg_z"): the multiregime share of every
# step alternates micro-batches between the r_z and the g_z dump (round-robin),
# and the ordinary share alternates between the two paired shared dumps.
# Every run also scores the *other* bank each validation (blind runs: the
# z-exposed bank as v4_validation_zx_*; zx runs: the z-blind bank as
# v4_validation_blind_*), so blind and z-exposed runs can be compared on one bank.
# 12 runs = {z-blind, z-exposed} x {fixed, curriculum} x {small, medium, large},
# same schedule/architecture as the v4 baselines (37283915/37283921/37283995)
# and the expose_z runs.  Task id = expose*6 + schedule*3 + size; submit the
# large tasks (2,5,8,11) on biomed_a100_gpu:
#   sbatch --array=0,1,3,4,6,7,9,10 scripts/slurm/pretrain_rg_z_mix.sbatch
#   sbatch --partition=biomed_a100_gpu --array=2,5,8,11 scripts/slurm/pretrain_rg_z_mix.sbatch
# WARMUP_PROPORTION (env, default -1 = the baselines' fixed 500 warmup steps) sets
# --warmup-proportion, e.g. --export=ALL,WARMUP_PROPORTION=0.2 for a 20 % warmup.
# RESUME=1 (env) continues an interrupted run from latest_checkpoint.pth (saved
# every 50 steps) or, failing that, its newest checkpoint-*.pth
# (model/optimizer/scheduler/RNG restored, history appended; the dump loaders
# restart from the first episode). Without a checkpoint the run starts fresh,
# which requires RUN_DIR not to exist.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4"
readonly WARMUP_PROPORTION="${WARMUP_PROPORTION:--1}"
readonly RESUME="${RESUME:-0}"
readonly -a EXPOSURES=(blind zx)
readonly -a SCHEDULES=(fixed curriculum)
readonly -a SIZES=(small medium large)
readonly EXPOSURE="${EXPOSURES[$((SLURM_ARRAY_TASK_ID / 6))]}"
readonly SCHEDULE="${SCHEDULES[$(((SLURM_ARRAY_TASK_ID % 6) / 3))]}"
readonly SIZE="${SIZES[$((SLURM_ARRAY_TASK_ID % 3))]}"
readonly FAMILY="rg_z-${SCHEDULE}"
case "${SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
esac
case "${EXPOSURE}" in
    blind)
        readonly DUMP_DIR="${DATA}/tabicl_mix_scm_paired/37277254"
        readonly BANK_DIR="${DATA}/tabicl_mix_scm_evaluation/37277969"   # repaired bank
        readonly EXTRA_BANK="${DATA}/tabicl_mix_scm_evaluation_expose_z/validation.h5" EXTRA_TAG=zx
        readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/rg_z/${FAMILY}-${SIZE}/seed-2402"
        readonly LINK_DIR="${PROJECT_DIR}/runs_exp/rg_z/${FAMILY}-${SIZE}"
        ;;
    zx)
        readonly DUMP_DIR="${DATA}/tabicl_mix_scm_paired_expose_z"
        readonly BANK_DIR="${DATA}/tabicl_mix_scm_evaluation_expose_z"
        readonly EXTRA_BANK="${DATA}/tabicl_mix_scm_evaluation/37277969/validation.h5" EXTRA_TAG=blind
        readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/expose_z/${FAMILY}-${SIZE}/seed-2402"
        readonly LINK_DIR="${PROJECT_DIR}/runs_exp/expose_z/${FAMILY}-${SIZE}"
        ;;
esac
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
for f in r_z-shared r_z-multiregime g_z-shared g_z-multiregime; do test -f "${DUMP_DIR}/${f}.h5"; done
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5" && test -f "${EXTRA_BANK}"
# pretrain_plain_nanotabpfn creates RUN_DIR itself (mkdir exist_ok=False), so only
# its parent may exist here.
mkdir -p "$(dirname "${RUN_DIR}")" "$(dirname "${LINK_DIR}")"
[[ -e "${LINK_DIR}" ]] || ln -s "$(dirname "${RUN_DIR}")" "${LINK_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
resume_args=()
if [[ "${RESUME}" == 1 ]]; then
    # latest_checkpoint.pth (written every 50 steps) is always at least as recent as any checkpoint-*.pth
    if [[ -f "${RUN_DIR}/latest_checkpoint.pth" ]]; then
        latest="${RUN_DIR}/latest_checkpoint.pth"
    else
        latest="$(ls -1 "${RUN_DIR}"/checkpoint-*.pth 2>/dev/null | sort | tail -1 || true)"
    fi
    if [[ -n "${latest}" ]]; then
        resume_args=(--resume-checkpoint "${latest}")
    else
        echo "RESUME=1 but no checkpoint under ${RUN_DIR}: starting fresh"
    fi
fi
echo "exposure=${EXPOSURE} family=${FAMILY} size=${SIZE} run_dir=${RUN_DIR} warmup_proportion=${WARMUP_PROPORTION} resume=${resume_args[*]:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" --device cuda --require-cuda ${resume_args[@]+"${resume_args[@]}"} \
    --seed 2402 --max-steps 10000 \
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 --warmup-proportion "${WARMUP_PROPORTION}" \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 \
    --v4-validation-bank-path "${BANK_DIR}/validation.h5" --v4-validation-interval 500 \
    --v4-extra-validation-bank-path "${EXTRA_BANK}" --v4-extra-validation-tag "${EXTRA_TAG}" \
    --v4-test-bank-path "${BANK_DIR}/test.h5" \
    --checkpoint-interval 2000 --latest-checkpoint-interval 50 --epoch-steps 500 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --prior-mode "${SCHEDULE}" --multiregime-ratio 0.5 --multiregime-source v4 \
    --v4-shared-dump-path "${DUMP_DIR}/r_z-shared.h5,${DUMP_DIR}/g_z-shared.h5" \
    --v4-dump-path "${DUMP_DIR}/r_z-multiregime.h5,${DUMP_DIR}/g_z-multiregime.h5"
