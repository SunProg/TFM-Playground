#!/bin/bash -l
#SBATCH --job-name=plain-7layer-curriculum
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Same 7-layer capacity-matched control as pretrain_plain_nanotabpfn_7layer_a30.sbatch
# (job 37203123, --prior-mode plain, pure mix_scm), but --prior-mode
# curriculum -- 0% multiregime for the first 10% of steps, then a linear ramp
# to a 50% multiregime share by 50% of steps, held there for the rest (see
# multiregime_probability()'s docstring in pretrain_plain_nanotabpfn.py).
# Completes the 7-layer capacity-matched grid alongside plain/multiregime.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_7layer_curriculum/${SLURM_JOB_ID}/seed-2402"
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
    --prior-mode curriculum \
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
