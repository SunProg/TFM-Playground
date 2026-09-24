#!/bin/bash -l
#SBATCH --job-name=scm-nano-prior-eval
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
readonly STUDY_ID="${STUDY_ID:?Set STUDY_ID to the completed small NanoTabPFN array id}"
readonly NANO_ROOT="${PROJECT_DIR}/results/scm_nanotabpfn_realistic_prior_schedule_5000_slurm/${STUDY_ID}"
readonly OUTPUT="${PROJECT_DIR}/results/scm_realistic_5000_small_nano_prior_schedule_regime_stratified.json"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_capacity_regime_eval \
    --nano-root "${NANO_ROOT}" \
    --head-root "${NANO_ROOT}" \
    --skip-head \
    --output "${OUTPUT}" \
    --embedding-size 32 \
    --num-attention-heads 4 \
    --mlp-hidden-size 64 \
    --num-layers 2 \
    --episodes 128 \
    --device cuda \
    --require-cuda
