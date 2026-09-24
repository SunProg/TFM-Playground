#!/bin/bash -l
#SBATCH --job-name=scm-tabpfn-finetune
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly VERSIONS=(v2.2 v2.6 v3)
readonly VERSION="${VERSIONS[${SLURM_ARRAY_TASK_ID}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly OUTPUT="${PROJECT_DIR}/results/scm_tabpfn_official_finetune_realistic/${STUDY_ID}/${VERSION}.json"
readonly V26_CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
readonly V3_CHECKPOINT="${PROJECT_DIR}/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT}")"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT_ARGS=()
if [[ "${VERSION}" == "v2.6" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "${V26_CHECKPOINT}")
elif [[ "${VERSION}" == "v3" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "${V3_CHECKPOINT}")
fi

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} version=${VERSION} output=${OUTPUT}"
echo "finetuning=5-epochs lr-1e-5 scheduler-on n-estimators-finetune-2"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" scripts/finetune_scm_tabpfn_official.py \
    --version "${VERSION}" \
    "${CHECKPOINT_ARGS[@]}" \
    --output "${OUTPUT}" \
    --prior-profile realistic \
    --episodes 128 \
    --namespace 4_000_000 \
    --epochs 5 \
    --learning-rate 1e-5 \
    --validation-split-ratio 0.2 \
    --early-stopping-patience 2 \
    --n-estimators-finetune 2 \
    --n-estimators-validation 2 \
    --n-estimators-final-inference 8 \
    --device cuda
