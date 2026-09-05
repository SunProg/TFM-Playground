#!/bin/bash -l
#SBATCH --job-name=mufasa-v2-select
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
: "${EVALUATION_JOB_ID:?EVALUATION_JOB_ID is required}"
readonly EVALUATION_ROOT="${PROJECT_DIR}/results/multiregime_v2/model-eval-${EVALUATION_JOB_ID}"
readonly OUTPUT="${EVALUATION_ROOT}/auxiliary_selection.json"

cd "${PROJECT_DIR}"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.select_multiregime_v2_auxiliary \
    --evaluation-root "${EVALUATION_ROOT}" \
    --output "${OUTPUT}"
