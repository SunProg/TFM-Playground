#!/bin/bash -l
#SBATCH --job-name=recon-routing-realistic-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the CREATE environment Python}"
readonly STUDY_ID="${STUDY_ID:?Set STUDY_ID to the training array job id}"
readonly RUN_ROOT="${CODE_ROOT}/runs/${STUDY_ID}"
readonly OUTPUT="${CODE_ROOT}/realistic_regime_metrics_${STUDY_ID}.json"

cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "study=${STUDY_ID} run_root=${RUN_ROOT} output=${OUTPUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" scripts/evaluate_reconstruction_routing_regime.py \
  --run-root "${RUN_ROOT}" \
  --output "${OUTPUT}" \
  --prior-profile realistic \
  --episodes 128 --episode-start 3000000 --query-size 256 \
  --device cuda
