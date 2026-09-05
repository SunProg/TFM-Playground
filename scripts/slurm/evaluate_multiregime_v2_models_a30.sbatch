#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-20%6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Evaluate every slot arm on the same deterministic full locked grid.  The
# matched oracle-one-hot model is always included as the recovery control.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEEDS=(2402 2403 2404)
: "${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"

SEED_INDEX=$((INDEX % 3))
ARM_INDEX=$((INDEX / 3))
readonly SEED="${SEEDS[${SEED_INDEX}]}"
ARMS=(
    mufasa-unsupervised-s4
    mufasa-supervised-lambda0.01
    mufasa-supervised-lambda0.1
    mufasa-supervised-lambda1
    mufasa-unsupervised-s1
    mufasa-unsupervised-s2
    mufasa-unsupervised-s3
)
readonly ARM="${ARMS[${ARM_INDEX}]}"
readonly MODEL="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/${ARM}/seed-${SEED}/final_checkpoint.pth"
readonly ORACLE="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/oracle-one-hot/seed-${SEED}/final_checkpoint.pth"
readonly OUTPUT_DIR="${RESULTS_ROOT}/model-eval-${SLURM_ARRAY_JOB_ID}/${ARM}/seed-${SEED}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${MODEL}" "${ORACLE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing prerequisite: ${path}" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${CODE_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} arm=${ARM} seed=${SEED} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.benchmark_multiregime_v2 \
    --output-dir "${OUTPUT_DIR}" \
    --tabpfn-checkpoint "${MODEL}" \
    --oracle-one-hot-checkpoint "${ORACLE}" \
    --episodes-per-seed 32 \
    --query-size 256 \
    --bootstrap-samples 2000 \
    --cpu-workers 8 \
    --inference-batch-size 4 \
    --generation-workers 8 \
    --no-persist-episodes \
    --compress-predictions \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
