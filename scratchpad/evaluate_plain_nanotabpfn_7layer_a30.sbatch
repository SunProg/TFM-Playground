#!/bin/bash -l
#SBATCH --job-name=eval-plain-7layer
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Accuracy + AUROC for the capacity-matched 7-layer plain-nanoTabPFN run
# (37203123), using evaluate_plain_nanotabpfn.py's own pre-existing
# ordinary/mix_scm evaluation path -- same as
# evaluate_plain_nanotabpfn_unpadded_a30.sbatch (job 37199211, 6-layer run
# 37195771), just pointed at the 7-layer checkpoint directory. This script's
# own live validate() only tracks loss, no AUC/accuracy, hence this separate
# eval pass against the latest saved checkpoint.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_7layer/37203123/seed-2402"
readonly CHECKPOINT="$(ls -t "${RUN_DIR}"/epoch-*-checkpoint.pth | head -1)"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/plain_nanotabpfn_mix_scm_eval/37203123-7layer"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -n "${CHECKPOINT}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_plain_nanotabpfn \
    --checkpoint "7layer_37203123=${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" --device cuda \
    --ordinary-episodes 100 --multiregime-episodes 0
