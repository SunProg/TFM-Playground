#!/bin/bash -l
#SBATCH --job-name=v4-native
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# The v4 prior-scale grid retrained on the NATIVE-TabICL-prior dumps
# (dump_multiregime_v4_native.sbatch) and validated/tested on the native
# controlled bank (build_multiregime_v4_bank_native.sbatch). Recipe identical
# to the 2026-09 baselines (37283915/37283921/37283995): 10k steps, 500-step
# warmup, same architecture per size. Task id = family*3 + size with
#   families: original r_z-fixed r_z-curriculum g_z-fixed g_z-curriculum rg_z-fixed rg_z-curriculum
#             (rg_z = r_z + g_z multiregime dumps round-robin, see pretrain_rg_z_mix.sbatch)
#   sizes:    small medium large   (large -> --partition=biomed_a100_gpu)
# e.g. original + rg_z, small/medium: sbatch --array=0,1,15,16,18,19 ...; large: --array=2,17,20
# WARMUP_PROPORTION env (default -1 = fixed 500 steps; 0.2 = 20 % warmup).
# MULTIREGIME_RATIO env (default 0.5): multiregime share for fixed / curriculum target.
# RESUME=1 continues from latest_checkpoint.pth (saved every 50 steps).
# PREEMPT_JOB=<jobid[_task]> (with RESUME=1): a faster-partition twin of a run
# already training elsewhere — on start it cancels that job, waits for it to
# exit, then resumes from the latest checkpoint it left behind.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4_native"
# Dumps expose z in 50 % of the episodes; the z-blind bank is the primary
# validation/test bank and the z-exposed bank is scored alongside
# (v4_validation_zx_* in history.jsonl).
readonly PRIOR_DIR="${DATA}/tabicl_mix_scm_paired_zmix5"
readonly BANK_DIR="${DATA}/tabicl_mix_scm_evaluation"
readonly EXTRA_BANK="${DATA}/tabicl_mix_scm_evaluation_expose_z/validation.h5"
readonly RESUME="${RESUME:-0}"
readonly WARMUP_PROPORTION="${WARMUP_PROPORTION:--1}"
readonly MULTIREGIME_RATIO="${MULTIREGIME_RATIO:-0.5}"   # multiregime share of the fixed schedule / curriculum target
readonly -a PRIOR_FAMILIES=(original r_z-fixed r_z-curriculum g_z-fixed g_z-curriculum rg_z-fixed rg_z-curriculum)
readonly -a MODEL_SIZES=(small medium large)
readonly PRIOR_FAMILY="${PRIOR_FAMILIES[$((SLURM_ARRAY_TASK_ID / 3))]}"
readonly MODEL_SIZE="${MODEL_SIZES[$((SLURM_ARRAY_TASK_ID % 3))]}"
case "${MODEL_SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
esac
readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/native/${PRIOR_FAMILY}-${MODEL_SIZE}/seed-2402"
readonly LINK_DIR="${PROJECT_DIR}/runs_exp/native/${PRIOR_FAMILY}-${MODEL_SIZE}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
# No paired shared-rule dumps: the ordinary branch of every family is the K=1
# original dump (a TabICL dump), the multiregime branch the family's multiregime dump.
for f in original r_z-multiregime g_z-multiregime; do test -f "${PRIOR_DIR}/${f}.h5"; done
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5" && test -f "${EXTRA_BANK}"
mkdir -p "$(dirname "${RUN_DIR}")" "$(dirname "${LINK_DIR}")"
[[ -e "${LINK_DIR}" ]] || ln -s "$(dirname "${RUN_DIR}")" "${LINK_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
if [[ -n "${PREEMPT_JOB:-}" ]]; then
    echo "preempting ${PREEMPT_JOB} (taking over its run dir)"
    scancel "${PREEMPT_JOB}" || true
    for _ in $(seq 1 60); do squeue -h -j "${PREEMPT_JOB}" 2>/dev/null | grep -q . || break; sleep 5; done
    sleep 10   # let the cancelled job's last checkpoint write settle
fi
resume_args=()
if [[ "${RESUME}" == 1 ]]; then
    if [[ -f "${RUN_DIR}/latest_checkpoint.pth" ]]; then latest="${RUN_DIR}/latest_checkpoint.pth"; else latest="$(ls -1 "${RUN_DIR}"/checkpoint-*.pth 2>/dev/null | sort | tail -1 || true)"; fi
    [[ -n "${latest}" ]] && resume_args=(--resume-checkpoint "${latest}")
fi
echo "prior=${PRIOR_FAMILY} size=${MODEL_SIZE} run_dir=${RUN_DIR} warmup_proportion=${WARMUP_PROPORTION} multiregime_ratio=${MULTIREGIME_RATIO} resume=${resume_args[*]:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

readonly -a COMMON_ARGS=(
    --output-dir "${RUN_DIR}"
    --device cuda --require-cuda
    --seed 2402 --max-steps 10000
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 --warmup-proportion "${WARMUP_PROPORTION}"
    --weight-decay 0.01 --gradient-clip 1.0
    --validation-interval 500 --validation-batches 16
    --v4-validation-bank-path "${BANK_DIR}/validation.h5" --v4-validation-interval 500
    --v4-extra-validation-bank-path "${EXTRA_BANK}" --v4-extra-validation-tag zx
    --v4-test-bank-path "${BANK_DIR}/test.h5"
    --checkpoint-interval 2000 --latest-checkpoint-interval 50 --epoch-steps 500
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}"
    --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}"
)

case "${PRIOR_FAMILY}" in
    original)
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn "${COMMON_ARGS[@]}" ${resume_args[@]+"${resume_args[@]}"} \
            --prior-mode original --multiregime-source v4 --v4-shared-dump-path "${PRIOR_DIR}/original.h5"
        ;;
    r_z-fixed|r_z-curriculum|g_z-fixed|g_z-curriculum)
        readonly MECHANISM_MODE="${PRIOR_FAMILY%%-*}"
        readonly SCHEDULE="${PRIOR_FAMILY#*-}"
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn "${COMMON_ARGS[@]}" ${resume_args[@]+"${resume_args[@]}"} \
            --prior-mode "${SCHEDULE}" --multiregime-ratio "${MULTIREGIME_RATIO}" --multiregime-source v4 \
            --v4-shared-dump-path "${PRIOR_DIR}/original.h5" \
            --v4-dump-path "${PRIOR_DIR}/${MECHANISM_MODE}-multiregime.h5"
        ;;
    rg_z-fixed|rg_z-curriculum)
        readonly SCHEDULE="${PRIOR_FAMILY#*-}"
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn "${COMMON_ARGS[@]}" ${resume_args[@]+"${resume_args[@]}"} \
            --prior-mode "${SCHEDULE}" --multiregime-ratio "${MULTIREGIME_RATIO}" --multiregime-source v4 \
            --v4-shared-dump-path "${PRIOR_DIR}/original.h5" \
            --v4-dump-path "${PRIOR_DIR}/r_z-multiregime.h5,${PRIOR_DIR}/g_z-multiregime.h5"
        ;;
    *) echo "Unknown prior family: ${PRIOR_FAMILY}" >&2; exit 2 ;;
esac
