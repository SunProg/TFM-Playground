#!/bin/bash -l
#SBATCH --job-name=tabarena-finetuned-backbone
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Evaluate vanilla nanoTabPFN and the fifteen all-SCM full-backbone fine-tunes
# from the 10--50% multi-regime-frequency sweep on the clean, held-out
# TabArena-small protocol: real labels, no contamination, 200-row subsamples,
# 5 folds, and 20 repeats.

set -euo pipefail

: "${FINETUNE_ARRAY_JOB_ID:?Submit with FINETUNE_ARRAY_JOB_ID=<array job id>.}"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly FINETUNE_DIR="${PROJECT_DIR}/runs/backbone_finetune_all_scm_sweep"
readonly OUTPUT_DIR="${PROJECT_DIR}/runs/tabarena_small/finetuned-backbone-all-scm-sweep-${FINETUNE_ARRAY_JOB_ID}-${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
readonly PROBABILITY_PERCENTS=(10 20 30 40 50)
readonly SEEDS=(2402 2403 2404)
declare -a BACKBONE_CHECKPOINTS=()
for probability_percent in "${PROBABILITY_PERCENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        checkpoint="${FINETUNE_DIR}/p${probability_percent}-seed${seed}-${FINETUNE_ARRAY_JOB_ID}/backbone.pth"
        test -f "${checkpoint}"
        BACKBONE_CHECKPOINTS+=("p${probability_percent}-seed${seed}=${checkpoint}")
    done
done
readonly BACKBONE_CHECKPOINT_ARGUMENT="$(IFS=,; echo "${BACKBONE_CHECKPOINTS[*]}")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabarena_small \
    --official-checkpoint checkpoints/nanotabpfn.pth \
    --backbone-checkpoints "${BACKBONE_CHECKPOINT_ARGUMENT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --subsample 200 \
    --folds 5 \
    --repeats 20 \
    --label-source real \
    --contamination 0.0 \
    --no-include-sklearn \
    --no-include-tabpfn
