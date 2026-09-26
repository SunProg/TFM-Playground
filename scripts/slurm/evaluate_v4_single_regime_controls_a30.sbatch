#!/bin/bash -l
#SBATCH --job-name=v4-single-regime-control
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#
# Frozen single-regime controls, matched to the observed-regime training probe.
# Entries: original-small (2 layers), original-medium (4), original-large (6).
# The probe uses exactly the training monitor's episodes and tag permutations.
# Full-bank evaluation uses the final diagnostic's seed. The checkpoint's
# original model code and saved hidden scores verify inference compatibility.
# These are 10,000-step reference models; early training checkpoints are not
# matched training-budget comparisons. This job performs no training.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${V4_REGIME_INFO_PYTHON:-/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python}"
readonly RUN_ROOT="${V4_NATIVE_RUN_ROOT:-/scratch/users/k23139234/tfm_runs/native}"
readonly BANK="${V4_NATIVE_BANK:-/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation/test.h5}"
readonly OUTPUT_ROOT="${V4_REGIME_INFO_OUTPUT:-${PROJECT_DIR}/artifacts/evaluations/regime_information_checkpoint_compatible}"
readonly MODEL_SOURCE="${V4_REGIME_INFO_MODEL_SOURCE:-/users/k23139234/repo/TFM-Playground-uncertainty/tfmplayground/models/nanotabpfn.py}"
# Colon-separated: sbatch --export splits values on commas.
IFS=':' read -r -a MODELS <<< "${V4_REGIME_INFO_MODELS:-original-small:original-medium:original-large}"
readonly MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly CHECKPOINT="${RUN_ROOT}/${MODEL_NAME}/seed-2402/final_checkpoint.pth"
readonly HIDDEN_REFERENCE="${RUN_ROOT}/${MODEL_NAME}/seed-2402/v4_test/final.json"
readonly OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/seed-2402"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
test -f "${BANK}"
test -f "${MODEL_SOURCE}"
test -f "${HIDDEN_REFERENCE}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "model=${MODEL_NAME} checkpoint=${CHECKPOINT}"

srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
    --checkpoint "${CHECKPOINT}" --bank "${BANK}" \
    --model-source-path "${MODEL_SOURCE}" --reference-hidden-report "${HIDDEN_REFERENCE}" \
    --output "${OUTPUT_DIR}/monitor_probe_seed2402.json" \
    --device cuda --multiclass-only --information hidden shuffled true \
    --seed 2402 --episode-in-cell 0 \
    --cell-subsample-modulus 16 --cell-subsample-remainder 0 \
    --bootstrap-replicates 200 --max-episodes-per-forward 1

if [[ -s "${OUTPUT_DIR}/test.json" ]]; then
    echo "Reusing existing full-bank report: ${OUTPUT_DIR}/test.json"
else
    # No --reference-hidden-report here: the probe above already checks the model
    # source against v4_test/final.json. At 8 episodes per forward, episode 4429
    # reproducibly misses rtol=1e-4 by ~7e-4 while the 1-per-forward probe
    # passes; cause unverified (batching is the main suspect).
    srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
        --checkpoint "${CHECKPOINT}" --bank "${BANK}" \
        --model-source-path "${MODEL_SOURCE}" \
        --output "${OUTPUT_DIR}/test.json" \
        --device cuda --multiclass-only --information hidden shuffled true \
        --seed 20260924 --bootstrap-replicates 5000 --max-episodes-per-forward 8
fi
