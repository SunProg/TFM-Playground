#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-supervised-tabpfn-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Evaluate the supervised latent nanoTabPFN on the locked episodes.  HGB and
# oracle controls are reused from the completed benchmark gate, so this array
# only runs neural-model inference.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly GATE_ROOT="${GATE_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${INDEX}]}"
readonly STUDY_ID="${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"
readonly BENCHMARK_JOB_ID="${BENCHMARK_JOB_ID:-36914949}"
readonly LABEL="supervised-nanotabpfn-lambda0.1"
readonly MODEL="${RUNS_ROOT}/multiregime_v2/${STUDY_ID}/${LABEL}/seed-${SEED}/final_checkpoint.pth"
readonly CONTROL_RESULTS_DIR="${GATE_ROOT}/benchmark-${BENCHMARK_JOB_ID}/seed-${SEED}"
readonly OUTPUT_DIR="${RESULTS_ROOT}/model-eval-${SLURM_ARRAY_JOB_ID}/${LABEL}/seed-${SEED}"

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

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} index=${INDEX} seed=${SEED} output=${OUTPUT_DIR}"
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
