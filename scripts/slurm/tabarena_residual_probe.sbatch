#!/bin/bash -l
#SBATCH --job-name=tabarena-residual-probe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Cross-fitted residual scoring over the TabArena binary tables, swept across
# expert families.  Asks whether real tables carry a second, covariate-
# dependent label rule at all -- the question every synthetic regime result in
# this project assumes the answer to.
#
# CPU only: the probe fits logistic regressions and small gradient boosters,
# never a transformer, so it must not consume GPU allocation.  `max_predictors`
# 30 selects fifteen binary datasets rather than the five the pretraining
# evaluation uses at its default of 10.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/tabarena_residual_probe/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.tabarena_residual_probe \
    --output-dir "${OUTPUT_DIR}" \
    --max-predictors 30 \
    --subsample 2048 \
    --inner-folds 5 \
    --outer-folds 5 \
    --hard-quantile 0.3 \
    --expert-families linear shallow hgb \
    --seed 2402

echo "finished_at=$(date --iso-8601=seconds)"
