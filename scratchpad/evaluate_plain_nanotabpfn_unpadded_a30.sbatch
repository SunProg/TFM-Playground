#!/bin/bash -l
#SBATCH --job-name=eval-plain-unpadded
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Accuracy + AUROC for the unpadded plain-nanoTabPFN mix_scm run (37195771),
# using evaluate_plain_nanotabpfn.py's own pre-existing ordinary/mix_scm
# evaluation path (make_prior() from pretrain_plain_nanotabpfn.py, unmodified)
# -- multiregime-episodes=0 since that code path hits an unrelated tabicl
# version-drift bug (Reg2Cls KeyError: 'regression') in continuous_episodes.py,
# irrelevant to what we're evaluating here.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_matched/37195771/seed-2402"
readonly CHECKPOINT="$(ls -t "${RUN_DIR}"/epoch-*-checkpoint.pth | head -1)"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/plain_nanotabpfn_mix_scm_eval/37195771"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -n "${CHECKPOINT}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "checkpoint=${CHECKPOINT} output=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_plain_nanotabpfn \
    --checkpoint "unpadded_37195771=${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" --device cuda \
    --ordinary-episodes 100 --multiregime-episodes 0
