#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-k1-prior
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# K=1-only prior stream for the matched ablation. The allocated GPU is unused.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly OUTPUT="${RUNS_ROOT}/multiregime_v2/prior-dump-k1-only/seed-${SEED}"

cd "${CODE_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite existing prior dump ${OUTPUT}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} seed=${SEED} output=${OUTPUT}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.multiregime_prior_dump \
    --output "${OUTPUT}" \
    --seed "${SEED}" \
    --max-steps 10000 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --backend analytic \
    --difference-components coefficients \
    --label-noise 0 \
    --gate-strength 1 \
    --curriculum-mode k1_only \
    --shard-episodes 4096
echo "finished_at=$(date --iso-8601=seconds)"
