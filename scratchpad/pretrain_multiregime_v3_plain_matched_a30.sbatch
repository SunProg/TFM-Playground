#!/bin/bash -l
#SBATCH --job-name=v3-plain-matched
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Plain nanoTabPFN baseline (KINDS[0]="plain", index 0/1/2 = original/fixed/
# curriculum mode), re-run at the same budget as the AttentionSlotRouter
# scope x mode grid (20000 steps, validation every 500, 32 validation
# episodes) instead of PilotConfig's own defaults (10000 steps, every 5000,
# 8 episodes) -- so the two are checkpoint-for-checkpoint comparable instead
# of only meeting at 3 sparse points.
set -euo pipefail
readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_ROOT="${CODE_ROOT}/runs/${STUDY_ID}"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
cd "${CODE_ROOT}"
test -x "${PYTHON_BIN}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_ARRAY_JOB_ID}_${INDEX} source=${CODE_ROOT} output=${RUN_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v3 \
    --preflight --device cuda --output "${RUN_ROOT}/gate-${INDEX}" \
    --steps 20000 --validation-interval 500 --validation-episodes 32
srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v3 \
    --index "${INDEX}" --device cuda --output "${RUN_ROOT}/cell-${INDEX}" \
    --gate "${RUN_ROOT}/gate-${INDEX}/execution_gate.json" \
    --steps 20000 --validation-interval 500 --validation-episodes 32
