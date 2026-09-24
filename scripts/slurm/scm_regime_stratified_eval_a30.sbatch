#!/bin/bash -l
#SBATCH --job-name=scm-regime-eval
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
readonly RESULTS_ROOT="${PROJECT_DIR}/results"
readonly OUTPUT="${RESULTS_ROOT}/scm_realistic_5000_regime_stratified.json"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.scm_regime_stratified_eval \
    --head-root "${RESULTS_ROOT}/scm_table_slot_head_routing_realistic_5000_slurm/37062078" \
    --nano-root "${RESULTS_ROOT}/scm_nanotabpfn_realistic_5000_slurm/37050453" \
    --variant-root "${RESULTS_ROOT}/scm_table_slot_variants_realistic_5000_slurm/37050501" \
    --output "${OUTPUT}" \
    --device cuda \
    --require-cuda \
    --episodes 128
