#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-train
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-26%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# 0-2: standard latent; 3-5: oracle one-hot; 6-8: MUFASA unsupervised S=4;
# 9-17: auxiliary lambda {0.01,0.1,1}; 18-26: unsupervised S={1,2,3}.
# Submit 0-5 first. Submit 6-26 only after the benchmark gate, passing
# STUDY_ID and BENCHMARK_JOB_ID from the prerequisite submissions.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEEDS=(2402 2403 2404)

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

MODEL_TYPE=tabpfn
INPUT_MODE=latent
NUM_SLOTS=4
AUX_WEIGHT=0
SEED_INDEX=$((INDEX % 3))
LABEL=""
EXTRA_FLAGS=()

if (( INDEX < 3 )); then
    LABEL="standard-latent"
elif (( INDEX < 6 )); then
    LABEL="oracle-one-hot"
    INPUT_MODE=oracle_one_hot
elif (( INDEX < 9 )); then
    LABEL="mufasa-unsupervised-s4"
    MODEL_TYPE=mufasa_slot_tabpfn
elif (( INDEX < 18 )); then
    MODEL_TYPE=mufasa_slot_tabpfn
    LAMBDA_INDEX=$(((INDEX - 9) / 3))
    AUX_WEIGHTS=(0.01 0.1 1)
    AUX_WEIGHT="${AUX_WEIGHTS[${LAMBDA_INDEX}]}"
    LABEL="mufasa-supervised-lambda${AUX_WEIGHT}"
else
    MODEL_TYPE=mufasa_slot_tabpfn
    SLOT_INDEX=$(((INDEX - 18) / 3))
    SLOT_COUNTS=(1 2 3)
    NUM_SLOTS="${SLOT_COUNTS[${SLOT_INDEX}]}"
    LABEL="mufasa-unsupervised-s${NUM_SLOTS}"
fi

readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${PROJECT_DIR}/runs/multiregime_v2/${STUDY_ID}/${LABEL}/seed-${SEED}"
readonly PRIOR_DUMP="${PROJECT_DIR}/runs/multiregime_v2/prior-dump/seed-${SEED}"

if [[ ! -f "${PRIOR_DUMP}/manifest.json" ]]; then
    echo "Missing prior dump: ${PRIOR_DUMP}" >&2
    exit 2
fi

if [[ "${MODEL_TYPE}" != tabpfn ]]; then
    if [[ -z "${BENCHMARK_JOB_ID:-}" ]]; then
        echo "Slot jobs require BENCHMARK_JOB_ID." >&2
        exit 2
    fi
    GATE_PATH="${PROJECT_DIR}/results/multiregime_v2/benchmark-${BENCHMARK_JOB_ID}/seed-${SEED}/execution_gate.json"
    EXTRA_FLAGS=(--benchmark-gate-path "${GATE_PATH}")
fi

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} index=${INDEX}"
echo "label=${LABEL} seed=${SEED} run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v2 \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --model-type "${MODEL_TYPE}" \
    --input-mode "${INPUT_MODE}" \
    --seed "${SEED}" \
    --max-steps 10000 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --validation-interval 5000 \
    --validation-episodes 16 \
    --checkpoint-interval 10000 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --backend analytic \
    --difference-components coefficients \
    --label-noise 0 \
    --gate-strength 1 \
    --single-regime-source matched \
    --prior-dump "${PRIOR_DUMP}" \
    --num-slots "${NUM_SLOTS}" \
    --aux-regime-weight "${AUX_WEIGHT}" \
    --slot-layer-index 0 \
    --slot-layer-indices 2 3 4 5 \
    "${EXTRA_FLAGS[@]}"

echo "finished_at=$(date --iso-8601=seconds)"
