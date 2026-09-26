#!/bin/bash -l
#SBATCH --job-name=plain-7layer
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Capacity-matched control: same prior/budget as pretrain_plain_nanotabpfn_matched_a30.sbatch
# (job 37195771, 6-layer, 3,711,362 params) but with num_layers=7 (4,304,834
# params -- the closest integer-layer match, from below, to
# AttentionSlotRouter's own live param count of 4,451,905 excluding its dead
# unused backbone decoder; see paper/attention_slot_router.md). Tests whether
# AttentionSlotRouter's marginal-rate-shortcut susceptibility is really about
# its slot-retrieval routing head specifically, or just relative undercapacity
# of the 6-layer plain backbone -- if this 7-layer plain model also reaches
# strong AUC on mix_scm (as the 6-layer one already does), capacity alone
# isn't the explanation.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_7layer/${SLURM_JOB_ID}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
echo "job_id=${SLURM_JOB_ID} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed 2402 \
    --max-steps 20000 \
    --num-layers 7 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --validation-interval 500 \
    --validation-batches 8 \
    --checkpoint-interval 500 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --max-classes 2 \
    --prior-type mix_scm
