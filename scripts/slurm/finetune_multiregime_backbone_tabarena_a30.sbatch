#!/bin/bash -l
#SBATCH --job-name=multiregime-backbone-tabarena
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Fine-tune on both TabICL SCM families. Multi-regime episodes are balanced
# across MLP--MLP, tree--tree, and MLP--tree label-function mixtures on shared
# covariates; checkpoints are then tested on clean, held-out TabArena-small.

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
# Five multi-regime step frequencies x three independent seeds.
readonly PROBABILITIES=(0.10 0.20 0.30 0.40 0.50)
readonly SEEDS=(2402 2403 2404)
readonly PROBABILITY_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
readonly SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
readonly MULTIREGIME_PROBABILITY="${PROBABILITIES[${PROBABILITY_INDEX}]}"
readonly PROBABILITY_PERCENT=$((10 * (PROBABILITY_INDEX + 1)))
readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly RUN_DIR="${PROJECT_DIR}/runs/backbone_finetune_all_scm_sweep/p${PROBABILITY_PERCENT}-seed${SEED}-${SLURM_ARRAY_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m tfmplayground.experiments.finetune_multiregime_backbone \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed "${SEED}" \
    --batch-size 2 \
    --accumulate-gradients 2 \
    --max-steps 3000 \
    --validation-interval 100 \
    --patience 10 \
    --validation-episodes 24 \
    --support-size 128 \
    --query-count 6 \
    --learning-rate 0.00001 \
    --multiregime-probability "${MULTIREGIME_PROBABILITY}"
