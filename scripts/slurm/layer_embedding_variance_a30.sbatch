#!/bin/bash -l
#SBATCH --job-name=layer-variance
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Per-layer nanoTabPFN embedding variance under bootstrap resampling of the
# support set: 3 synthetic held-out-family conditions (ambiguous/identifiable/
# noisy) plus all 20 official TabArena binary tasks, official checkpoint.
# See tfmplayground/experiments/layer_embedding_variance.py.

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

readonly OUTPUT_DIR="${PROJECT_DIR}/results/layer_embedding_variance/${SLURM_JOB_ID}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.layer_embedding_variance \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --members 32 \
    --episodes-per-condition 20 \
    --support-size 128 \
    --query-count 6 \
    --context-size 1024 \
    --num-mem-chunks 8 \
    --tabarena-query-cap 256

echo "finished_at=$(date --iso-8601=seconds)"
