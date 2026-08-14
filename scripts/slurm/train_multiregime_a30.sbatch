#!/bin/bash -l
#SBATCH --job-name=multiregime-train
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

# Training pilot for the "multiregime" curriculum condition (see
# tfmplayground/experiments/continuous_episodes.py:sample_multiregime_episode
# and the plan at can-we-evaluate-the-federated-torvalds.md): the same
# frozen-mean/trainable-uncertainty-head model as every other condition in
# this trial, trained on a curriculum that includes row-level mixed-regime
# episodes.  Three seeds, no screening grid -- this is a first pilot, not a
# selected finalist configuration.

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

readonly RUN_DIR="${PROJECT_DIR}/runs/multiregime_pilot/gpu-seed${SEED}-${SLURM_ARRAY_JOB_ID}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "seed=${SEED}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.train_continuous_bayesian \
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
    --validation-support-size 128 \
    --validation-query-count 6 \
    --uncertainty-mode adapters \
    --context-mode deepsets \
    --ambiguous-probability 0.30 \
    --identifiable-probability 0.20 \
    --noisy-probability 0.15 \
    --paired-probability 0.15 \
    --multiregime-probability 0.20

echo "finished_at=$(date --iso-8601=seconds)"
