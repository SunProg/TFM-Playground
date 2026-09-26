#!/bin/bash -l
#SBATCH --job-name=heart-adapt
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=heart-adapt-%j.out
#SBATCH --error=heart-adapt-%j.err
# Observed-site evaluation with 20 stratified labelled examples from the
# target site in support. True and shuffled-support tags have equal width.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly SNAPSHOT="/scratch/users/k23139234/beyondarena-cache/data-foundry-cache/BeyondArena/datasets--TabArena--BeyondArena/snapshots/2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"
readonly RUN_ROOTS="/scratch/users/k23139234/tfm_runs/native"
readonly OUT_DIR="${OUT_DIR:-/scratch/users/k23139234/tfm_eval/heart_sites_adaptation_20shot}"
readonly BASELINES="tabpfn-v2.2 tabpfn-v2.6 tabpfn-v3 tabicl-v1 tabicl-v2 rf logreg hgb xgboost lightgbm catboost"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -d "${SNAPSHOT}"
test -d "${RUN_ROOTS}"
mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/users/k23139234/beyondarena-cache/data-foundry-cache
export TABPFN_V22_CHECKPOINT="${TABPFN_V22_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.2.ckpt}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_heart_sites_adaptation \
    --datasets-root "${SNAPSHOT}" \
    --run-roots "${RUN_ROOTS}" \
    --baselines ${BASELINES} \
    --output-dir "${OUT_DIR}" \
    --device cuda --shots 20 --repeats 5 --seed 0
