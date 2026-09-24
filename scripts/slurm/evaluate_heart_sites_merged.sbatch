#!/bin/bash -l
#SBATCH --job-name=heart-sites
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=heart-sites-%j.out
#SBATCH --error=heart-sites-%j.err
# Merged UCI Heart Disease sites (Cleveland + Hungary + VA Long Beach, one schema):
# every model scored with the hospital hidden vs given as a feature, on identical
# folds, under two protocols (repeated stratified K-fold; leave-one-site-out).
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly SNAPSHOT="/scratch/users/k23139234/beyondarena-cache/data-foundry-cache/BeyondArena/datasets--TabArena--BeyondArena/snapshots/2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"
readonly OUT_DIR="${OUT_DIR:-/scratch/users/k23139234/tfm_eval/heart_sites}"
readonly RUN_ROOTS="${RUN_ROOTS:-/scratch/users/k23139234/tfm_runs/native}"   # real paths: rglob does not follow the runs_exp symlinks
readonly BASELINES="${BASELINES:-tabpfn-v2.2 tabpfn-v2.6 tabpfn-v3 tabicl-v1 tabicl-v2 rf logreg hgb xgboost lightgbm catboost}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"; test -d "${SNAPSHOT}"
mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/users/k23139234/beyondarena-cache/data-foundry-cache
export TABPFN_V22_CHECKPOINT="${TABPFN_V22_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.2.ckpt}"
nvidia-smi --query-gpu=name --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_heart_sites_merged \
    --datasets-root "${SNAPSHOT}" \
    --run-roots "${RUN_ROOTS}" \
    --baselines ${BASELINES} \
    --output-dir "${OUT_DIR}" \
    --device cuda --splits 5 --repeats 2 --seed 0
