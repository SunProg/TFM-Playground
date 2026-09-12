#!/bin/bash -l
#SBATCH --job-name=tabpfn-versions-tabarena15
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly OUTPUT_ROOT="${PROJECT_DIR}/results/tabpfn_versions_tabarena15/${SLURM_JOB_ID}"
readonly V26_CHECKPOINT="/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt"
readonly V3_CHECKPOINT="/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# tabarena15 protocol: max_predictors=30 -> 15 eligible binary datasets,
# subsample=2048, 5-fold x 10-repeat CV, real labels -- same settings as
# every other TabArena job in this project (sklearn baselines, v3 pilot
# cells), so directly comparable. v2.2 uses the package's own cached
# default (no explicit checkpoint needed on this host); v2.6/v3 need the
# local checkpoint files to bypass the gated hosted-weights download.
srun uv run --project "${PROJECT_DIR}" python scripts/evaluate_tabpfn_versions_tabarena.py \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    --cache-directory "${PROJECT_DIR}/.cache/openml" \
    --checkpoint-v26 "${V26_CHECKPOINT}" \
    --checkpoint-v3 "${V3_CHECKPOINT}" \
    --max-predictors 30 \
    --subsample 2048 \
    --folds 5 \
    --repeats 10 \
    --seed 2402

echo "finished_at=$(date --iso-8601=seconds)"
