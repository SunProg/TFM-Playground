#!/bin/bash -l
#SBATCH --job-name=posterior-slot-large-5k
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Same two-slot mixed-prior task as the 1k MPS pilot.  Only backbone capacity,
# effective batch size, validation precision, and training duration are scaled.
# Task 0 has no cross-fit auxiliary; task 1 uses lambda=0.1 and two folds.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly VARIANTS=("no-crossfit" "crossfit-lambda0.1")
readonly CROSSFIT_WEIGHTS=("0.0" "0.1")
readonly VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
readonly CROSSFIT_WEIGHT="${CROSSFIT_WEIGHTS[${SLURM_ARRAY_TASK_ID}]}"
readonly RUN_DIR="${PROJECT_DIR}/results/posterior_attention_large_5000_slurm/${STUDY_ID}/${VARIANT}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python at ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "variant=${VARIANT} crossfit_weight=${CROSSFIT_WEIGHT} run_dir=${RUN_DIR}"
echo "architecture=embedding-192 heads-6 mlp-768 layers-6 slots-2 slot-heads-4"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda --require-cuda --seed 2402 \
    --max-steps 5000 --micro-batch-size 8 --accumulate-gradients 4 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 2000 \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 --validation-episodes 32 \
    --epoch-steps 500 --checkpoint-interval 5000 --no-tensorboard --no-tabarena-every-epoch \
    --support-size 16 --query-size 8 --min-features 2 --max-features 6 \
    --embedding-size 192 --num-attention-heads 6 --mlp-hidden-size 768 --num-layers 6 \
    --num-slots 2 --num-slot-iterations 3 --slot-attention-heads 4 \
    --table-slot-layer-indices 3 4 5 --table-slot-scope cell_and_data \
    --model-kind table_slot_head --query-routing-mode posterior_attention \
    --posterior-crossfit-weight "${CROSSFIT_WEIGHT}" --posterior-crossfit-folds 2 \
    --prior-type mix_scm --prior-mode mixed --multiregime-share 0.3 --multiregime-contamination 0.3 \
    --slot-mi-weight 0 --support-reconstruction-weight 0 --embedding-reconstruction-weight 0
