#!/bin/bash -l
#SBATCH --job-name=tabpfn-v4-bank
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Real TabPFN v2.2 / v2.6 / v3 in-context baselines on the v4 validation and
# test banks (one array task per version x split). Same per-episode protocol
# and report layout as the pretrained NanoTabPFN evaluations.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly BANK_DIR="${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_evaluation/37277969"
readonly OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs_eval/tabpfn_baselines}"
readonly VERSIONS=(v2.2 v2.6 v3)
readonly SPLITS=(validation test)
readonly VERSION="${VERSIONS[$((SLURM_ARRAY_TASK_ID / 2))]}"
readonly SPLIT="${SPLITS[$((SLURM_ARRAY_TASK_ID % 2))]}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${BANK_DIR}/${SPLIT}.h5"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
echo "version=${VERSION} split=${SPLIT}"
nvidia-smi --query-gpu=name --format=csv,noheader
srun "${PYTHON_BIN}" scratchpad/evaluate_tabpfn_versions_on_v4_bank.py \
    --bank "${BANK_DIR}/${SPLIT}.h5" --versions "${VERSION}" \
    --output-dir "${OUT_DIR}" --device cuda --n-estimators 8
