#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-tabarena15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the completed pilot run under runs/STUDY_ID}"
readonly CHECKPOINT_ROOT="${PROJECT_DIR}/${RUN_ROOT}/tabarena_checkpoints"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/multiregime_v3_tabarena15/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Retrofit the six pilot checkpoints (written before checkpoints carried an
# inference-ready "architecture" block) into standalone TabArena-loadable
# files. A no-op to re-run: wrap() always overwrites its output path.
uv run --project "${PROJECT_DIR}" python scripts/wrap_multiregime_v3_checkpoints_for_tabarena.py \
    --run-root "${PROJECT_DIR}/${RUN_ROOT}" \
    --output-root "${CHECKPOINT_ROOT}"

declare -a CHECKPOINT_ENTRIES=()
for entry in plain-original slot-original plain-fixed slot-fixed plain-curriculum slot-curriculum; do
    CHECKPOINT_ENTRIES+=("${entry}=${CHECKPOINT_ROOT}/multiregime_v3-${entry}.pth")
done
readonly CHECKPOINTS="$(IFS=,; echo "${CHECKPOINT_ENTRIES[*]}")"

echo "job=${SLURM_JOB_ID} run_root=${RUN_ROOT} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun uv run --project "${PROJECT_DIR}" python -m tfmplayground.experiments.evaluate_tabarena_small \
    --standalone-checkpoints "${CHECKPOINTS}" \
    --no-include-vanilla \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --max-predictors 30 \
    --subsample 2048 \
    --folds 5 \
    --repeats 10 \
    --no-include-sklearn \
    --no-include-tabpfn \
    --label-source real \
    --contamination 0.0 \
    --cache-directory "${PROJECT_DIR}/.cache/openml"

echo "finished_at=$(date --iso-8601=seconds)"
