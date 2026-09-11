#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-cells-tabarena
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
readonly RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the pilot run under runs/STUDY_ID}"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/multiregime_v3_tabarena15/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Cells from this run save an inference-ready "architecture" block natively
# (pretrain_multiregime_v3.inference_architecture), so no wrap step is
# needed -- just point evaluate_tabarena_small at every finished checkpoint.
# Only cells whose run finished (checkpoint.pth exists) are included, so
# this can be rerun as more cells complete.
declare -a CHECKPOINT_ENTRIES=()
for checkpoint in "${PROJECT_DIR}/${RUN_ROOT}"/cell-*/checkpoint.pth; do
    [ -e "${checkpoint}" ] || continue
    cell_dir="$(dirname "${checkpoint}")"
    name="$(basename "${cell_dir}")"
    CHECKPOINT_ENTRIES+=("${name}=${checkpoint}")
done
if [ "${#CHECKPOINT_ENTRIES[@]}" -eq 0 ]; then
    echo "No checkpoint.pth files found under ${RUN_ROOT}/cell-*/" >&2
    exit 1
fi
readonly CHECKPOINTS="$(IFS=,; echo "${CHECKPOINT_ENTRIES[*]}")"

echo "job=${SLURM_JOB_ID} run_root=${RUN_ROOT} cells=${#CHECKPOINT_ENTRIES[@]} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
