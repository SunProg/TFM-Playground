#!/bin/bash -l
#SBATCH --job-name=bayes-nano-k8
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
readonly RUN_NAME="create-k8-a30-${SLURM_JOB_ID}"
readonly RUN_DIR="${PROJECT_DIR}/runs/bayesian_nanotabpfn/${RUN_NAME}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "project_dir=${PROJECT_DIR}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi

"${PYTHON_BIN}" -c 'import json, platform, torch; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))'
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_bayesian_nanotabpfn \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}/training" \
    --device cuda \
    --seed 2402 \
    --num-hypotheses 8 \
    --query-count 8 \
    --batch-size 2 \
    --accumulate-gradients 4 \
    --support-size 512 \
    --frozen-steps 3000 \
    --validation-interval 100 \
    --validation-episodes 24 \
    --ordinary-evaluation-batches 12 \
    --patience 15 \
    --learning-rate 0.0001 \
    --likelihood-temperature 0.01 \
    --min-features 1 \
    --max-features 16 \
    --candidate-pool-multiplier 4 \
    --max-candidate-attempts 64

echo "finished_at=$(date --iso-8601=seconds)"
