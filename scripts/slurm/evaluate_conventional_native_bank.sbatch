#!/bin/bash -l
#SBATCH --job-name=conv-native-bank
#SBATCH --partition=cpu
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Conventional ML baselines (default hyper-parameters, fresh estimator per
# episode) on the NATIVE controlled bank: logreg rf hgb xgboost lightgbm catboost x validation/test.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly BANK_DIR="${BANK_DIR:-/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation}"
readonly OUT_DIR="${OUT_DIR:-/scratch/users/k23139234/tfm_eval/native_references}"
readonly -a MODELS=(logreg rf hgb xgboost lightgbm catboost)
readonly -a SPLITS=(validation test)
readonly MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}"
readonly SPLIT="${SPLITS[$((SLURM_ARRAY_TASK_ID % 2))]}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"; test -f "${BANK_DIR}/${SPLIT}.h5"; mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "model=${MODEL} split=${SPLIT} out=${OUT_DIR}"
srun "${PYTHON_BIN}" scratchpad/evaluate_tabpfn_versions_on_v4_bank.py \
    --bank "${BANK_DIR}/${SPLIT}.h5" --versions "${MODEL}" --output-dir "${OUT_DIR}" --device cpu
