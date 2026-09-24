#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-tabicl211
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
readonly PILOT_ROOT="${PROJECT_DIR}/runs/37113090"
readonly PILOT_CONFIG="${PILOT_ROOT}/cell-0/config.json"
readonly PILOT_RESULT="${PILOT_ROOT}/cell-0/result.json"
readonly OUTPUT_ROOT="${PROJECT_DIR}/results/multiregime_v3_tabicl211/${SLURM_JOB_ID}"
readonly BANK_MANIFEST="${OUTPUT_ROOT}/evaluation_bank_manifest.json"
readonly PILOT_ACCURACY_AUC="${OUTPUT_ROOT}/pilot_accuracy_auc.json"
readonly BASELINE_OUTPUT="${OUTPUT_ROOT}/baseline.json"
readonly BASELINE_ENVIRONMENT="${OUTPUT_ROOT}/baseline.environment.json"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_ROOT}"
test -s "${PILOT_CONFIG}"
test -s "${PILOT_RESULT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TABICL_VERSION="$(${PYTHON_BIN} -c 'import importlib.metadata; print(importlib.metadata.version("tabicl"))')"
if [[ "${TABICL_VERSION}" != "2.1.1" ]]; then
    echo "expected tabicl==2.1.1, found ${TABICL_VERSION}" >&2
    exit 1
fi
echo "tabicl=${TABICL_VERSION}"

"${PYTHON_BIN}" scripts/make_multiregime_v3_tabicl211_bank.py \
    --pilot-config "${PILOT_CONFIG}" \
    --pilot-result "${PILOT_RESULT}" \
    --output "${BANK_MANIFEST}"

# All completed pilot cells, including the six table-slot families beyond the
# nine rows used in the earlier comparison note.
"${PYTHON_BIN}" scripts/report_multiregime_v3_accuracy_auc.py \
    --run-root "${PILOT_ROOT}" \
    --output "${PILOT_ACCURACY_AUC}" \
    --bank-manifest "${BANK_MANIFEST}" \
    --device cuda \
    --indices $(seq 0 20)

# sklearn and TabPFN v2.2/v2.6/v3, with and without per-episode finetuning,
# on exactly the same tabicl 2.1.1 bank.
srun "${PYTHON_BIN}" scripts/evaluate_multiregime_v3_baselines.py \
    --output "${BASELINE_OUTPUT}" \
    --environment-manifest "${BASELINE_ENVIRONMENT}" \
    --pilot-config "${PILOT_CONFIG}" \
    --pilot-result "${BANK_MANIFEST}" \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
