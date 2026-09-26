#!/bin/bash -l
#SBATCH --job-name=v4-single-regime-steps
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
# Evaluate saved single-regime checkpoints on the exact 42-episode training
# monitor: seed 2402, episode 0, cell hash modulus 16, all three tag conditions.
# Verify one complete cell against each checkpoint's saved validation report.
# Final 10k-step probes already exist; this array supplies 2k/4k/6k/8k controls.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${V4_REGIME_INFO_PYTHON:-/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python}"
readonly RUN_ROOT="${V4_NATIVE_RUN_ROOT:-/scratch/users/k23139234/tfm_runs/native}"
readonly BANK_ROOT="/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation"
readonly MODEL_SOURCE="${V4_REGIME_INFO_MODEL_SOURCE:-/scratch/users/k23139234/TFM-Playground-slot-attention/tfmplayground/models/nanotabpfn.py}"
readonly MODELS=(original-small original-medium original-large)
readonly MODEL_NAME="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
readonly RUN_DIR="${RUN_ROOT}/${MODEL_NAME}/seed-2402"
readonly OUTPUT_DIR="${PROJECT_DIR}/artifacts/evaluations/regime_information_checkpoint_compatible/${MODEL_NAME}/seed-2402/checkpoint_probes"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${MODEL_SOURCE}"
test -f "${BANK_ROOT}/test.h5"
test -f "${BANK_ROOT}/validation.h5"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name --format=csv,noheader

# Match the available small-model 4k probe first; the other runs approach 2k.
steps=(002000 004000 006000 008000)
if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then
    steps=(004000 002000 006000 008000)
fi
for step in "${steps[@]}"; do
    checkpoint="${RUN_DIR}/checkpoint-${step}.pth"
    reference="${RUN_DIR}/v4_validation/step-${step}.json"
    test -f "${checkpoint}"
    test -f "${reference}"
    echo "model=${MODEL_NAME} checkpoint_step=${step}"

    srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
        --checkpoint "${checkpoint}" --model-source-path "${MODEL_SOURCE}" \
        --bank "${BANK_ROOT}/validation.h5" --reference-hidden-report "${reference}" \
        --output "${OUTPUT_DIR}/step-${step}-compatibility.json" \
        --device cuda --multiclass-only --information hidden \
        --max-episodes 8 --max-episodes-per-forward 1 --bootstrap-replicates 1

    srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information.py \
        --checkpoint "${checkpoint}" --model-source-path "${MODEL_SOURCE}" \
        --bank "${BANK_ROOT}/test.h5" \
        --output "${OUTPUT_DIR}/step-${step}.json" \
        --device cuda --multiclass-only --information hidden shuffled true \
        --seed 2402 --episode-in-cell 0 \
        --cell-subsample-modulus 16 --cell-subsample-remainder 0 \
        --bootstrap-replicates 200 --max-episodes-per-forward 1
done
