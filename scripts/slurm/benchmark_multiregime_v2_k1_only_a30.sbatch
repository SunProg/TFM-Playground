#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-k1-eval
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

# Evaluate K=1-only standard nanoTabPFN checkpoints on the complete locked
# benchmark. This intentionally omits the oracle-one-hot checkpoint: the
# ablation asks how a K=1-only latent model transfers to multi-regime cells.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TRAIN_STUDY_ID="${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly STANDARD="${RUNS_ROOT}/multiregime_v2/${TRAIN_STUDY_ID}/k1-only-standard-latent/seed-${SEED}/final_checkpoint.pth"
readonly OUTPUT_DIR="${RESULTS_ROOT}/k1-only-benchmark-${SLURM_ARRAY_JOB_ID}/seed-${SEED}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${STANDARD}"; do
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

echo "job_id=${SLURM_JOB_ID} seed=${SEED} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

# The K=1 evaluation previously failed during Python spawn semaphore
# reconstruction on the cluster (before any model evaluation).  Serial
# generation avoids that transient /dev/shm SemLock failure; inference
# remains GPU-backed and parallelism is unchanged for model batches.
srun "${PYTHON_BIN}" -m tfmplayground.experiments.benchmark_multiregime_v2 \
    --output-dir "${OUTPUT_DIR}" \
    --tabpfn-checkpoint "${STANDARD}" \
    --episodes-per-seed 32 \
    --query-size 256 \
    --bootstrap-samples 2000 \
    --cpu-workers 8 \
    --inference-batch-size 4 \
    --generation-workers 1 \
    --no-persist-episodes \
    --compress-predictions \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
