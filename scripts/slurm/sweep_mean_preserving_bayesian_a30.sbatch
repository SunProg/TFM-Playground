#!/bin/bash -l
#SBATCH --job-name=bayes-nano-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID}"

# A compact 3 x 2 x 2 screening design. All trials use identical held-out
# seeds and data budgets, so validation losses are directly comparable.
readonly K_VALUES=(4 8 16)
readonly LR_VALUES=(0.00003 0.0001)
readonly TEMPERATURE_VALUES=(0.003 0.01)
readonly K_INDEX=$((TASK_ID / 4))
readonly REMAINDER=$((TASK_ID % 4))
readonly LR_INDEX=$((REMAINDER / 2))
readonly TEMPERATURE_INDEX=$((REMAINDER % 2))
readonly K="${K_VALUES[${K_INDEX}]}"
readonly LEARNING_RATE="${LR_VALUES[${LR_INDEX}]}"
readonly TEMPERATURE="${TEMPERATURE_VALUES[${TEMPERATURE_INDEX}]}"
readonly TRIAL_NAME="trial-${TASK_ID}-k${K}-lr${LEARNING_RATE}-tau${TEMPERATURE}"
readonly SWEEP_DIR="${PROJECT_DIR}/runs/bayesian_nanotabpfn/sweep-${SLURM_ARRAY_JOB_ID}"
readonly RUN_DIR="${SWEEP_DIR}/${TRIAL_NAME}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "task_id=${TASK_ID}"
echo "host=$(hostname)"
echo "trial_name=${TRIAL_NAME}"
echo "K=${K} learning_rate=${LEARNING_RATE} likelihood_temperature=${TEMPERATURE}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_bayesian_nanotabpfn \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}/training" \
    --device cuda \
    --seed 2402 \
    --num-hypotheses "${K}" \
    --query-count 8 \
    --batch-size 2 \
    --accumulate-gradients 4 \
    --support-size 512 \
    --frozen-steps 800 \
    --validation-interval 100 \
    --validation-episodes 12 \
    --ordinary-evaluation-batches 6 \
    --patience 5 \
    --learning-rate "${LEARNING_RATE}" \
    --likelihood-temperature "${TEMPERATURE}" \
    --min-features 1 \
    --max-features 16 \
    --candidate-pool-multiplier 4 \
    --max-candidate-attempts 64

echo "finished_at=$(date --iso-8601=seconds)"
