#!/bin/bash -l
#SBATCH --job-name=multiregime-v3-baselines
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
readonly OUTPUT_ROOT="${PROJECT_DIR}/results/multiregime_v3_tabarena15_baselines/${SLURM_JOB_ID}"
readonly V22_CHECKPOINT="/users/k23139234/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
readonly V26_CHECKPOINT="/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
readonly V3_CHECKPOINT="/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Same seed (default 2402), same subsample/folds/repeats/max-predictors as
# the v3 pilot's own TabArena job (evaluate_multiregime_v3_tabarena_a30.sbatch)
# -- eligible_tasks() and RepeatedStratifiedKFold are both seeded from these,
# so this scores the identical test set.
readonly SHARED_FLAGS=(
    --no-include-vanilla
    --device cuda
    --max-predictors 30
    --subsample 2048
    --folds 5
    --repeats 10
    --no-include-tabpfn
    --label-source real
    --contamination 0.0
    --cache-directory "${PROJECT_DIR}/.cache/openml"
)

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# sklearn baselines (logistic regression, random forest); no TabPFN here.
srun uv run --project "${PROJECT_DIR}" python -m tfmplayground.experiments.evaluate_tabarena_small \
    "${SHARED_FLAGS[@]}" --include-sklearn --no-include-tabpfn \
    --output-dir "${OUTPUT_ROOT}/sklearn"

for entry in "v2.2=${V22_CHECKPOINT}" "v2.6=${V26_CHECKPOINT}" "v3=${V3_CHECKPOINT}"; do
    name="${entry%%=*}"
    checkpoint="${entry#*=}"
    echo "tabpfn ${name} checkpoint=${checkpoint}"
    srun uv run --project "${PROJECT_DIR}" python -m tfmplayground.experiments.evaluate_tabarena_small \
        "${SHARED_FLAGS[@]}" --no-include-sklearn --include-tabpfn --tabpfn-model-path "${checkpoint}" \
        --output-dir "${OUTPUT_ROOT}/tabpfn-${name}"
done

echo "finished_at=$(date --iso-8601=seconds)"
