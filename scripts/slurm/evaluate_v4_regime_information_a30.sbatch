#!/bin/bash -l
#SBATCH --job-name=v4-regime-info
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#
# Frozen-checkpoint regime-information diagnostic on the native score-hidden
# test bank.  Array entries are the three primary six-layer final checkpoints;
# every entry compares hidden, shuffled-tag, and true-tag inputs over all
# 5,760 multiclass K>=2 multiregime episodes.  No pretraining occurs here.
# INVALID for native checkpoints: no --model-source-path, so they load into this
# checkout's attention and mispredict. Use evaluate_v4_single_regime_controls_a30.sbatch.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${V4_REGIME_INFO_PYTHON:-/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python}"
readonly RUN_ROOT="${V4_NATIVE_RUN_ROOT:-/scratch/users/k23139234/tfm_runs/native}"
readonly BANK="${V4_NATIVE_BANK:-/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation/test.h5}"
readonly OUTPUT_ROOT="${V4_REGIME_INFO_OUTPUT:-${PROJECT_DIR}/artifacts/evaluations/regime_information_native}"
readonly MODELS=(original-large rg_z-fixed-large rg_z-curriculum-large)
readonly MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CHECKPOINT="${RUN_ROOT}/${MODEL_NAME}/seed-2402/final_checkpoint.pth"
readonly OUTPUT="${OUTPUT_ROOT}/${MODEL_NAME}/seed-2402/test.json"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
test -f "${BANK}"
mkdir -p "$(dirname "${OUTPUT}")"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name --format=csv,noheader

srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
    --checkpoint "${CHECKPOINT}" \
    --bank "${BANK}" \
    --output "${OUTPUT}" \
    --device cuda --multiclass-only --max-episodes-per-forward 8
