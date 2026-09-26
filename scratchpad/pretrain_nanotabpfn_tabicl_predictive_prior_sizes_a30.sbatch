#!/bin/bash -l
#SBATCH --job-name=nano-tabicl-rz-gz
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One controlled-production K=1 original reference plus fixed-50% and
# curriculum-to-50% training for each multiregime construction (r_z and g_z),
# each at three capacities.
# Within each construction its shared/multiregime banks are paired: they differ
# only in Y_0 versus Y_Z.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly PRIOR_DUMP_JOB_ID="${PRIOR_DUMP_JOB_ID:?Set PRIOR_DUMP_JOB_ID when submitting this array.}"
readonly EVALUATION_JOB_ID="${EVALUATION_JOB_ID:?Set EVALUATION_JOB_ID when submitting this array.}"
readonly PRIOR_DIR="${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_paired/${PRIOR_DUMP_JOB_ID}"
readonly ORIGINAL_DUMP="${PRIOR_DIR}/original.h5"
readonly EVALUATION_DIR="${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_evaluation/${EVALUATION_JOB_ID}"
readonly VALIDATION_BANK="${EVALUATION_DIR}/validation.h5"
readonly TEST_BANK="${EVALUATION_DIR}/test.h5"

readonly -a PRIOR_FAMILIES=(original r_z-fixed r_z-curriculum g_z-fixed g_z-curriculum)
readonly -a MODEL_SIZES=(small medium large)
readonly PRIOR_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODEL_SIZES[@]}))
readonly SIZE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODEL_SIZES[@]}))
readonly PRIOR_FAMILY="${PRIOR_FAMILIES[PRIOR_INDEX]}"
readonly MODEL_SIZE="${MODEL_SIZES[SIZE_INDEX]}"

case "${MODEL_SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
    *) echo "Unknown model size: ${MODEL_SIZE}" >&2; exit 2 ;;
esac

readonly RUN_DIR="${PROJECT_DIR}/runs/nanotabpfn_tabicl_mix_scm_prior_scale/${SLURM_ARRAY_JOB_ID}/${PRIOR_FAMILY}-${MODEL_SIZE}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${ORIGINAL_DUMP}"
test -f "${PRIOR_DIR}/r_z-shared.h5"
test -f "${PRIOR_DIR}/r_z-multiregime.h5"
test -f "${PRIOR_DIR}/g_z-shared.h5"
test -f "${PRIOR_DIR}/g_z-multiregime.h5"
test -f "${VALIDATION_BANK}"
test -f "${TEST_BANK}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
echo "prior=${PRIOR_FAMILY} size=${MODEL_SIZE} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

readonly -a COMMON_ARGS=(
    --output-dir "${RUN_DIR}"
    --device cuda --require-cuda
    # Pilot: 1,000 updates at an effective batch of 128 tables. This retains
    # the real 1,024-row/C=5 geometry while fitting the A30 time budget.
    --seed 2402 --max-steps 1000
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 2000
    --weight-decay 0.01 --gradient-clip 1.0
    --validation-interval 5000 --validation-batches 16
    --v4-validation-bank-path "${VALIDATION_BANK}" --v4-validation-interval 1000
    --v4-test-bank-path "${TEST_BANK}"
    --checkpoint-interval 10000 --epoch-steps 500
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}"
    --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}"
)

case "${PRIOR_FAMILY}" in
    original)
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn "${COMMON_ARGS[@]}" \
            --prior-mode original --multiregime-source v4 --v4-shared-dump-path "${ORIGINAL_DUMP}"
        ;;
    r_z-fixed|r_z-curriculum|g_z-fixed|g_z-curriculum)
        readonly MECHANISM_MODE="${PRIOR_FAMILY%%-*}"
        readonly SCHEDULE="${PRIOR_FAMILY#*-}"
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn "${COMMON_ARGS[@]}" \
            --prior-mode "${SCHEDULE}" --multiregime-ratio 0.5 --multiregime-source v4 \
            --v4-shared-dump-path "${PRIOR_DIR}/${MECHANISM_MODE}-shared.h5" \
            --v4-dump-path "${PRIOR_DIR}/${MECHANISM_MODE}-multiregime.h5"
        ;;
    *) echo "Unknown prior family: ${PRIOR_FAMILY}" >&2; exit 2 ;;
esac
