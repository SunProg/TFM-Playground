#!/bin/bash -l
#SBATCH --job-name=nanotabpfn-regime
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Set PRIOR_MODE to plain, multiregime, or curriculum.
# Both arms have identical model, optimizer, update budget, geometry, and
# epoch-level TabArena-small evaluation; only the synthetic training prior differs.
set -euo pipefail

: "${PRIOR_MODE:?Set PRIOR_MODE to plain, multiregime, or curriculum when submitting.}"
case "${PRIOR_MODE}" in plain|multiregime|curriculum) ;; *) echo "Invalid PRIOR_MODE=${PRIOR_MODE}" >&2; exit 2 ;; esac
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(2402 2403 2404 2405 2406)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly RUN_DIR="${PROJECT_DIR}/runs/regime_pretraining/${SLURM_ARRAY_JOB_ID}/${PRIOR_MODE}/seed-${SEED}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'from torch.utils.tensorboard import SummaryWriter; print("tensorboard=available")'
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed "${SEED}" \
    --prior-mode "${PRIOR_MODE}" \
    --multiregime-contamination 0.30 \
    --max-steps 10000 \
    --epoch-steps 500 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --validation-interval 1000 \
    --validation-batches 16 \
    --checkpoint-interval 10000 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --max-classes 2 \
    --prior-type mix_scm \
    --tabarena-every-epoch \
    --tabarena-folds 5 \
    --tabarena-repeats 10 \
    --tabarena-subsample 2048 \
    --tabarena-cache-directory "${PROJECT_DIR}/.cache/openml" \
    --tensorboard
