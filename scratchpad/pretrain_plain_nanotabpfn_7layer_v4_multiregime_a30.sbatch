#!/bin/bash -l
#SBATCH --job-name=plain-7layer-v4-multiregime
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Same 7-layer capacity-matched control as the legacy-source
# pretrain_plain_nanotabpfn_7layer_multiregime_a30.sbatch (job 37208633), but
# --multiregime-source v4 -- the multiregime share is drawn from
# multiregime_v4.py's real mlp_scm/tree_scm-grounded prior (dump job
# 37233157: soft_gate/persistent families, TabICL's own random-row class
# ratio instead of always-50/50, per-episode-sampled feature count) instead
# of continuous_episodes.py's naive hand-built generator. --prior-mode
# itself (multiregime_probability's ramp shape) is unchanged: constant 100%
# multiregime share from step 1.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly RUN_DIR="${PROJECT_DIR}/runs/plain_nanotabpfn_7layer_v4_multiregime/${SLURM_JOB_ID}/seed-2402"
readonly V4_DUMP_DIR="${PROJECT_DIR}/data/multiregime_v4/37233157"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -d "${V4_DUMP_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
echo "job_id=${SLURM_JOB_ID} run_dir=${RUN_DIR} v4_dump_dir=${V4_DUMP_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --seed 2402 \
    --max-steps 20000 \
    --num-layers 7 \
    --prior-mode multiregime \
    --multiregime-source v4 \
    --v4-dump-path "${V4_DUMP_DIR}" \
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
