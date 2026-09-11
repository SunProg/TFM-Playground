#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-synth-baselines
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly OUTPUT="${PROJECT_DIR}/results/multiregime_v3_synthetic_baselines/${SLURM_JOB_ID}.json"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# sklearn (logreg, random forest, fit on each episode's support set) and
# TabPFN v2.2/v2.6/v3, with and without per-episode finetuning, on the v3
# pilot's own held-out episode bank (identical episodes every v3 pilot cell
# is scored against, per evaluate_multiregime_v3_baselines.py's docstring).
srun uv run --project "${PROJECT_DIR}" python scripts/evaluate_multiregime_v3_baselines.py \
    --output "${OUTPUT}" \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
