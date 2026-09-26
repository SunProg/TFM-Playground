#!/bin/bash -l
#SBATCH --job-name=nano-v4-c5-preflight
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One forward/backward/optimizer update at the largest row geometry used by
# the pretraining prior. Gradient accumulation does not retain activations
# between micro-batches, so micro_batch_size=4 is the relevant VRAM quantity.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"

case "${SLURM_ARRAY_TASK_ID}" in
    0) readonly MODEL_SIZE="small" EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    1) readonly MODEL_SIZE="medium" EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
    2) readonly MODEL_SIZE="large" EMBEDDING_SIZE=192 HEADS=6 HIDDEN_SIZE=768 LAYERS=6 ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "model=${MODEL_SIZE} support=512 query=512 classes=5 micro_batch=4"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.preflight_nanotabpfn_v4_memory \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" \
    --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --micro-batch-size 4 --support-size 512 --query-size 512 --num-features 12 --num-classes 5
