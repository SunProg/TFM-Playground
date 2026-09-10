#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-acc-auc
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the pilot's runs/<STUDY_ID> directory}"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun uv run --project "${PROJECT_DIR}" python scripts/report_multiregime_v3_accuracy_auc.py \
    --run-root "${PROJECT_DIR}/${RUN_ROOT}" \
    --device cuda
