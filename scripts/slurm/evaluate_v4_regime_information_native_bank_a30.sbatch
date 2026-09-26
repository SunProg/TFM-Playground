#!/bin/bash -l
#SBATCH --job-name=v4-info-mechanism
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#
# Frozen-checkpoint regime-information evaluation for all nine primary native
# models. Per-episode output retains mechanism_mode so r_z and g_z can be
# summarized separately. This does not retrain any model.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${V4_REGIME_INFO_PYTHON:-/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python}"
readonly RUN_ROOT="${V4_NATIVE_RUN_ROOT:-/scratch/users/k23139234/tfm_runs/native}"
readonly BANK="${V4_NATIVE_BANK:-/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation/test.h5}"
readonly OUTPUT_ROOT="${V4_REGIME_INFO_OUTPUT:-${PROJECT_DIR}/artifacts/evaluations/regime_information_by_mechanism}"
readonly ORIGINAL_MODEL_SOURCE="${V4_REGIME_INFO_ORIGINAL_MODEL_SOURCE:-/users/k23139234/repo/TFM-Playground-slot/tfmplayground/models/nanotabpfn.py}"
readonly MIXED_MODEL_SOURCE="${V4_REGIME_INFO_MIXED_MODEL_SOURCE:-/users/k23139234/repo/TFM-Playground-uncertainty/tfmplayground/models/nanotabpfn.py}"
readonly MODELS=(
  original-small original-medium original-large
  rg_z-fixed-small rg_z-fixed-medium rg_z-fixed-large
  rg_z-curriculum-small rg_z-curriculum-medium rg_z-curriculum-large
)
readonly MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CHECKPOINT="${RUN_ROOT}/${MODEL_NAME}/seed-2402/final_checkpoint.pth"
readonly HIDDEN_REFERENCE="${RUN_ROOT}/${MODEL_NAME}/seed-2402/v4_test/final.json"
readonly OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/seed-2402"

if [[ "${MODEL_NAME}" == original-* ]]; then
  readonly MODEL_SOURCE="${ORIGINAL_MODEL_SOURCE}"
  readonly EXPECTED_MODEL_SOURCE_SHA256="5ed1a46b2939aa0c8db1e3766a27a9cae9db6e36ac0a8b165471df7bcc14339c"
else
  readonly MODEL_SOURCE="${MIXED_MODEL_SOURCE}"
  readonly EXPECTED_MODEL_SOURCE_SHA256="32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34"
fi

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
test -f "${BANK}"
test -f "${MODEL_SOURCE}"
test -f "${HIDDEN_REFERENCE}"
readonly ACTUAL_MODEL_SOURCE_SHA256="$(sha256sum "${MODEL_SOURCE}" | awk '{print $1}')"
if [[ "${ACTUAL_MODEL_SOURCE_SHA256}" != "${EXPECTED_MODEL_SOURCE_SHA256}" ]]; then
  echo "Model source hash mismatch for ${MODEL_NAME}: expected ${EXPECTED_MODEL_SOURCE_SHA256}, got ${ACTUAL_MODEL_SOURCE_SHA256}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "model=${MODEL_NAME} checkpoint=${CHECKPOINT} source=${MODEL_SOURCE}"

readonly PROBE="${OUTPUT_DIR}/monitor_probe_seed2402.json"
if [[ ! -s "${PROBE}" ]]; then
  srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
      --checkpoint "${CHECKPOINT}" --bank "${BANK}" \
      --model-source-path "${MODEL_SOURCE}" \
      --reference-hidden-report "${HIDDEN_REFERENCE}" \
      --output "${PROBE}" --device cuda --multiclass-only \
      --information hidden shuffled true --seed 2402 --episode-in-cell 0 \
      --cell-subsample-modulus 16 --cell-subsample-remainder 0 \
      --bootstrap-replicates 200 --max-episodes-per-forward 1
fi

readonly REPORT="${OUTPUT_DIR}/test.json"
if [[ ! -s "${REPORT}" ]]; then
  srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
      --checkpoint "${CHECKPOINT}" --bank "${BANK}" \
      --model-source-path "${MODEL_SOURCE}" \
      --output "${REPORT}" --device cuda --multiclass-only \
      --information hidden shuffled true --seed 20260924 \
      --bootstrap-replicates 5000 --max-episodes-per-forward 8
fi
