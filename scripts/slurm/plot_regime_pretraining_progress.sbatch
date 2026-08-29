#!/bin/bash -l
#SBATCH --job-name=plot-regime-progress
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${PLAIN_ARRAY_JOB_ID:?Set the plain pretraining array id.}"
: "${MULTIREGIME_ARRAY_JOB_ID:?Set the multiregime pretraining array id.}"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/regime_pretraining/${PLAIN_ARRAY_JOB_ID}-${MULTIREGIME_ARRAY_JOB_ID}"
readonly SEEDS=(2402 2403 2404 2405 2406)

cd "${PROJECT_DIR}"
declare -a ARGUMENTS=()
for seed in "${SEEDS[@]}"; do
    ARGUMENTS+=(--plain-run "${PROJECT_DIR}/runs/regime_pretraining/${PLAIN_ARRAY_JOB_ID}/plain/seed-${seed}")
    ARGUMENTS+=(--multiregime-run "${PROJECT_DIR}/runs/regime_pretraining/${MULTIREGIME_ARRAY_JOB_ID}/multiregime/seed-${seed}")
done
mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m tfmplayground.experiments.plot_regime_pretraining_progress \
    "${ARGUMENTS[@]}" \
    --output "${OUTPUT_DIR}/plain-vs-multiregime-progress.png"
