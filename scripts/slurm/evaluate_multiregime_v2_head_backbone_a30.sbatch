#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-head-backbone-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Evaluate slot-head and the pure slot-backbone-ce control on the locked v2 grid.
# Serial episode generation avoids the cluster's spawn-semaphore failure;
# model inference remains GPU-backed.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly GATE_ROOT="${GATE_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEEDS=(2402 2403 2404)
readonly SEED_INDEX=$((INDEX % 3))
readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly STUDY_ID="${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"
# The current locked benchmark gate is 36914949; callers may override this
# when a later gate is selected.  Keeping a default lets the already-submitted
# dependent array pick up the HGB-reuse path without replacement.
readonly BENCHMARK_JOB_ID="${BENCHMARK_JOB_ID:-36914949}"

MODEL_TYPE="slot-head"
LABEL="slot-head"
if (( INDEX >= 3 )); then
    MODEL_TYPE="slot-backbone"
    LABEL="slot-backbone-ce"
fi

readonly MODEL="${RUNS_ROOT}/multiregime_v2/${STUDY_ID}/${LABEL}/seed-${SEED}/final_checkpoint.pth"
readonly OUTPUT_DIR="${RESULTS_ROOT}/model-eval-${SLURM_ARRAY_JOB_ID}/${LABEL}/seed-${SEED}"
readonly CONTROL_RESULTS_DIR="${GATE_ROOT}/benchmark-${BENCHMARK_JOB_ID}/seed-${SEED}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${MODEL}" "${CONTROL_RESULTS_DIR}/episode_manifest.json" "${CONTROL_RESULTS_DIR}/episode_metrics.csv"; do
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

echo "job_id=${SLURM_JOB_ID} arm=${LABEL} seed=${SEED} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.benchmark_multiregime_v2 \
    --output-dir "${OUTPUT_DIR}" \
    --tabpfn-checkpoint "${MODEL}" \
    --model-name "${LABEL}" \
    --episodes-per-seed 32 \
    --query-size 256 \
    --bootstrap-samples 2000 \
    --cpu-workers 8 \
    --inference-batch-size 4 \
    --generation-workers 1 \
    --reuse-controls-from "${CONTROL_RESULTS_DIR}" \
    --no-persist-episodes \
    --compress-predictions \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
