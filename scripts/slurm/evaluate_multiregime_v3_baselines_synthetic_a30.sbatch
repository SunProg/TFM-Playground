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
readonly PILOT_ROOT="${PROJECT_DIR}/runs/37113090"
readonly PILOT_CONFIG="${PILOT_ROOT}/cell-0/config.json"
readonly PILOT_RESULT="${PILOT_ROOT}/cell-0/result.json"
readonly ENVIRONMENT_MANIFEST="${PROJECT_DIR}/results/multiregime_v3_synthetic_baselines/${SLURM_JOB_ID}.environment.json"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"
test -s "${PILOT_CONFIG}"
test -s "${PILOT_RESULT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

TABICL_VERSION="$(${PROJECT_DIR}/.venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("tabicl"))')"
if [[ "${TABICL_VERSION}" != "2.1.1" ]]; then
    echo "expected tabicl==2.1.1, found ${TABICL_VERSION}" >&2
    exit 1
fi
echo "tabicl=${TABICL_VERSION}"

# sklearn (logreg, random forest, fit on each episode's support set) and
# TabPFN v2.2/v2.6/v3, with and without per-episode finetuning, on the
# CREATE pilot's original family. Invoke the existing
# environment directly so uv cannot silently replace tabicl with another
# resolver-selected version during the run.
srun "${PROJECT_DIR}/.venv/bin/python" scripts/evaluate_multiregime_v3_baselines.py \
    --output "${OUTPUT}" \
    --environment-manifest "${ENVIRONMENT_MANIFEST}" \
    --pilot-config "${PILOT_CONFIG}" \
    --pilot-result "${PILOT_RESULT}" \
    --families original \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
