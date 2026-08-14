#!/bin/bash -l
#SBATCH --job-name=multiregime-eval
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Standalone multiregime evaluation: query_count=32, contamination swept at
# {0.0, 0.2, 0.4}, seven mixture sources (mlp_scm, tree_scm, and the five
# analytic family pairs from paper/tabpfn_multi_regime_results.md), baseline
# checkpoint vs every completed backbone fine-tune side by side. Single job,
# not an array -- all models evaluated in one process.
#
# Edit FINETUNED_CHECKPOINTS below to point at the fine-tune run directories
# to compare (defaults to the three seeds from job 36552366).

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

readonly OUTPUT_DIR="${PROJECT_DIR}/results/multiregime_backbone/eval-${SLURM_JOB_ID}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_multiregime_backbone \
    --baseline-checkpoint checkpoints/nanotabpfn.pth \
    --finetuned-checkpoint seed2402=runs/backbone_finetune_pilot/gpu-seed2402-36552366/backbone.pth \
    --finetuned-checkpoint seed2403=runs/backbone_finetune_pilot/gpu-seed2403-36552366/backbone.pth \
    --finetuned-checkpoint seed2404=runs/backbone_finetune_pilot/gpu-seed2404-36552366/backbone.pth \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --query-count 32 \
    --support-size 128 \
    --episodes-per-cell 20

echo "finished_at=$(date --iso-8601=seconds)"
