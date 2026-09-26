#!/bin/bash -l
#SBATCH --job-name=nano-v4-scale
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

# Task layout, in array-index order:
# original/{small,medium,large,xlarge}, then fixed/{...}, then curriculum/{...}.
# Every run trains for the same 20k updates. The fixed share and curriculum
# endpoint are both 0.5; v4 validation is the same 2,304-episode factorial
# bank every 1,000 updates and the held-out bank runs once at the end.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly TRAIN_ENTRYPOINT="${PROJECT_DIR}/scratchpad/pretrain_plain_nanotabpfn_v4_prior_scale.py"
readonly DUMP_JOB_ID="37258624"
readonly EVALUATION_JOB_ID="37258425"
readonly DUMP_DIR="${PROJECT_DIR}/data/multiregime_v4/training_priors/${DUMP_JOB_ID}"
readonly ORIGINAL_DUMP="${DUMP_DIR}/original_mix_scm.h5"
readonly MULTIREGIME_DUMP="${DUMP_DIR}/multiregime_v4.h5"
readonly EVALUATION_DIR="${PROJECT_DIR}/data/multiregime_v4/evaluation/${EVALUATION_JOB_ID}"
readonly VALIDATION_BANK="${EVALUATION_DIR}/validation.h5"
readonly TEST_BANK="${EVALUATION_DIR}/test.h5"

readonly -a PRIOR_FAMILIES=(original fixed curriculum)
readonly -a MODEL_SIZES=(small medium large xlarge)
readonly PRIOR_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODEL_SIZES[@]}))
readonly SIZE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODEL_SIZES[@]}))
readonly PRIOR_FAMILY="${PRIOR_FAMILIES[PRIOR_INDEX]}"
readonly MODEL_SIZE="${MODEL_SIZES[SIZE_INDEX]}"

case "${MODEL_SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    large)  readonly EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
    xlarge) readonly EMBEDDING_SIZE=256 HEADS=8 HIDDEN_SIZE=1024 LAYERS=8 ;;
    *) echo "Unknown model size: ${MODEL_SIZE}" >&2; exit 2 ;;
esac

readonly RUN_DIR="${PROJECT_DIR}/runs/nanotabpfn_v4_prior_scale/${SLURM_ARRAY_JOB_ID}/${PRIOR_FAMILY}-${MODEL_SIZE}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${TRAIN_ENTRYPOINT}"
test -f "${ORIGINAL_DUMP}"
test -f "${VALIDATION_BANK}"
test -f "${TEST_BANK}"
if [[ "${PRIOR_FAMILY}" != "original" ]]; then
    test -f "${MULTIREGIME_DUMP}"
fi
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
echo "prior=${PRIOR_FAMILY} size=${MODEL_SIZE} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

readonly -a COMMON_ARGS=(
    --output-dir "${RUN_DIR}"
    --device cuda --require-cuda
    --seed 2402
    --max-steps 20000
    --micro-batch-size 8 --accumulate-gradients 4
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 2000
    --weight-decay 0.01 --gradient-clip 1.0
    --validation-interval 5000 --validation-batches 16
    --v4-validation-bank-path "${VALIDATION_BANK}" --v4-validation-interval 1000
    --v4-test-bank-path "${TEST_BANK}"
    --checkpoint-interval 10000
    --support-size 128 --query-size 32
    --min-features 2 --max-features 12 --max-classes 2 --prior-type mix_scm
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}"
    --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}"
    --prior-mode "${PRIOR_FAMILY}"
    --original-source dump --original-dump-path "${ORIGINAL_DUMP}"
)

if [[ "${PRIOR_FAMILY}" == "original" ]]; then
    srun "${PYTHON_BIN}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}"
else
    srun "${PYTHON_BIN}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
        --multiregime-ratio 0.5 --multiregime-source v4 --v4-dump-path "${MULTIREGIME_DUMP}"
fi
