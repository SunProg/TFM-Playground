#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-cell
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-380%381
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One task evaluates one of the 381 locked benchmark cells.  Submit this
# script once per trained seed with TRAIN_SEED set to 2402, 2403, or 2404.
# Per-cell execution_gate.json is informational: aggregate gate checks require
# merging all cells and are intentionally not used as a Slurm failure condition.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TRAIN_STUDY_ID="${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"
readonly TRAIN_SEED="${TRAIN_SEED:?TRAIN_SEED is required (2402, 2403, or 2404)}"
readonly CELL_INDEX="${SLURM_ARRAY_TASK_ID}"
readonly GENERATION_WORKERS="${GENERATION_WORKERS:-8}"
readonly CPU_WORKERS="${CPU_WORKERS:-8}"
readonly STANDARD="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/standard-latent/seed-${TRAIN_SEED}/final_checkpoint.pth"
readonly ORACLE="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/oracle-one-hot/seed-${TRAIN_SEED}/final_checkpoint.pth"
readonly OUTPUT_DIR="${RESULTS_ROOT}/benchmark-sharded-${SLURM_ARRAY_JOB_ID}/seed-${TRAIN_SEED}/cell-${CELL_INDEX}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${STANDARD}" "${ORACLE}"; do
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

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} train_seed=${TRAIN_SEED} cell=${CELL_INDEX} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.benchmark_multiregime_v2 \
    --output-dir "${OUTPUT_DIR}" \
    --tabpfn-checkpoint "${STANDARD}" \
    --oracle-one-hot-checkpoint "${ORACLE}" \
    --episodes-per-seed 32 \
    --query-size 256 \
    --bootstrap-samples 2000 \
    --cpu-workers "${CPU_WORKERS}" \
    --inference-batch-size 4 \
    --generation-workers "${GENERATION_WORKERS}" \
    --no-persist-episodes \
    --compress-predictions \
    --cell-index "${CELL_INDEX}" \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
