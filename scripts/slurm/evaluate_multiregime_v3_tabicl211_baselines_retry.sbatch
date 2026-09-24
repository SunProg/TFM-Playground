#!/bin/bash -l
#SBATCH --job-name=v3-tabicl211-baselines
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
readonly BANK_ROOT="${PROJECT_DIR}/results/multiregime_v3_tabicl211/37203190"
readonly PILOT_ROOT="${PROJECT_DIR}/runs/37113090"
readonly OUTPUT_ROOT="${PROJECT_DIR}/results/multiregime_v3_tabicl211/${SLURM_JOB_ID}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_ROOT}"
test -s "${BANK_ROOT}/evaluation_bank_manifest.json"
test -s "${PILOT_ROOT}/cell-0/config.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TABICL_VERSION="$(${PYTHON_BIN} -c 'import importlib.metadata; print(importlib.metadata.version("tabicl"))')"
test "${TABICL_VERSION}" = "2.1.1"
cp "${BANK_ROOT}/evaluation_bank_manifest.json" "${OUTPUT_ROOT}/evaluation_bank_manifest.json"

srun "${PYTHON_BIN}" scripts/evaluate_multiregime_v3_baselines.py \
    --output "${OUTPUT_ROOT}/baseline.json" \
    --environment-manifest "${OUTPUT_ROOT}/baseline.environment.json" \
    --pilot-config "${PILOT_ROOT}/cell-0/config.json" \
    --pilot-result "${OUTPUT_ROOT}/evaluation_bank_manifest.json" \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
