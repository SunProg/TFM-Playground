#!/bin/bash -l
#SBATCH --job-name=multiregime-backbone
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Fine-tunes the nanoTabPFN backbone itself (every parameter, not a frozen-mean
# adapter head) on a curriculum that includes multiregime episodes.  See
# tfmplayground/experiments/finetune_multiregime_backbone.py's module
# docstring for why this is the one thing not yet tried after three prior
# frozen-backbone architectures (slots, particle filter, resampling) and one
# frozen-backbone training pilot all failed to find a usable multi-regime
# signal.

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

readonly RUN_DIR="${PROJECT_DIR}/runs/backbone_finetune_pilot/gpu-seed${SEED}-${SLURM_ARRAY_JOB_ID}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "seed=${SEED}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.finetune_multiregime_backbone \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed "${SEED}" \
    --batch-size 2 \
    --accumulate-gradients 2 \
    --max-steps 3000 \
    --validation-interval 100 \
    --patience 10 \
    --validation-episodes 16 \
    --support-size 128 \
    --query-count 6 \
    --learning-rate 0.00001 \
    --multiregime-probability 0.30

echo "finished_at=$(date --iso-8601=seconds)"
