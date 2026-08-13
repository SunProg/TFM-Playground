#!/bin/bash -l
#SBATCH --job-name=bayes-nano-final
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly NUM_HYPOTHESES="${NUM_HYPOTHESES:-4}"
readonly LEARNING_RATE="${LEARNING_RATE:-0.00003}"
readonly LIKELIHOOD_TEMPERATURE="${LIKELIHOOD_TEMPERATURE:-0.01}"
readonly TRAINING_SEED="${TRAINING_SEED:-2402}"
readonly TRIAL_LABEL="${TRIAL_LABEL:-k${NUM_HYPOTHESES}-lr${LEARNING_RATE}-tau${LIKELIHOOD_TEMPERATURE}}"
readonly RUN_DIR="${PROJECT_DIR}/runs/bayesian_nanotabpfn/finalist-${TRIAL_LABEL}-${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "run_dir=${RUN_DIR}"
echo "num_hypotheses=${NUM_HYPOTHESES}"
echo "learning_rate=${LEARNING_RATE}"
echo "likelihood_temperature=${LIKELIHOOD_TEMPERATURE}"
echo "training_seed=${TRAINING_SEED}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The 5,000-step value is a safety ceiling; selection is driven by early
# stopping after ten 100-step checks without an improvement of at least 1e-4.
srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_bayesian_nanotabpfn \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}/training" \
    --device cuda \
    --seed "${TRAINING_SEED}" \
    --num-hypotheses "${NUM_HYPOTHESES}" \
    --query-count 8 \
    --batch-size 2 \
    --accumulate-gradients 4 \
    --support-size 512 \
    --frozen-steps 5000 \
    --validation-interval 100 \
    --validation-episodes 24 \
    --ordinary-evaluation-batches 12 \
    --patience 10 \
    --min-delta 0.0001 \
    --learning-rate "${LEARNING_RATE}" \
    --likelihood-temperature "${LIKELIHOOD_TEMPERATURE}" \
    --min-features 1 \
    --max-features 16 \
    --candidate-pool-multiplier 4 \
    --max-candidate-attempts 64

echo "finished_at=$(date --iso-8601=seconds)"
