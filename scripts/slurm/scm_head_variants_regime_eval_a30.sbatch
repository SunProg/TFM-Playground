#!/bin/bash -l
#SBATCH --job-name=scm-heads-prior-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly STUDY_ID="${STUDY_ID:?Set STUDY_ID to the completed head training array id}"
readonly ROOT="${PROJECT_DIR}/results/scm_table_slot_head_realistic_prior_schedule_5000_slurm/${STUDY_ID}"
readonly OUTPUT="${PROJECT_DIR}/results/scm_realistic_5000_small_head_prior_schedule_regime_stratified.json"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_head_variants_regime_eval \
    --root "${ROOT}" \
    --output "${OUTPUT}" \
    --episodes 128 \
    --device cuda \
    --require-cuda
