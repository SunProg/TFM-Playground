#!/bin/bash -l
#SBATCH --job-name=eval-plain-all-epochs
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Accuracy + AUROC across all 19 epoch checkpoints of the unpadded plain-
# nanoTabPFN mix_scm run (37195771), all scored against the identical locked
# episode set (evaluate_plain_nanotabpfn.py generates episodes once, before
# any checkpoint loads) -- a true apples-to-apples trend across training.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_matched/37195771/seed-2402"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/plain_nanotabpfn_mix_scm_eval/37195771-all-epochs"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

declare -a CHECKPOINT_ARGS=()
for f in "${RUN_DIR}"/epoch-*-checkpoint.pth; do
  epoch="$(basename "$f" | sed -E 's/epoch-0*([0-9]+)-checkpoint.pth/\1/')"
  CHECKPOINT_ARGS+=(--checkpoint "epoch-${epoch}=${f}")
done
echo "evaluating ${#CHECKPOINT_ARGS[@]} checkpoint args (/2 = checkpoints)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_plain_nanotabpfn \
    "${CHECKPOINT_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" --device cuda \
    --ordinary-episodes 100 --multiregime-episodes 0
