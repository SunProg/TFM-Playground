#!/bin/bash -l
#SBATCH --job-name=mr-episode-dump
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Pre-generate the multiregime episode dump consumed by pretrain_slot_tabpfn
# via --multiregime-dump.
#
# Building these episodes is pure CPU work and was the multiregime arm's
# bottleneck: 39 against 85 steps/min, roughly 0.83 s per step spent generating
# 8 episodes, about 11.5 h of a 21.5 h run.  Reading them back costs ~0.09 ms
# per episode instead.  This runs on the cpu partition so it does not consume
# GPU allocation.
#
# 16 shards x 6250 episodes = 100k.  The training arm draws 50k steps x 4
# accumulation x 8 = 1.6M episodes, so the dump is cycled ~16 times: the
# diversity of fresh generation is what is traded away for the speed.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly EPISODES="${EPISODES:-100000}"
readonly DUMP_DIR="${PROJECT_DIR}/runs/multiregime_dump"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${DUMP_DIR}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID} shard=${SLURM_ARRAY_TASK_ID} episodes_total=${EPISODES}"
echo "dump_dir=${DUMP_DIR}"
echo "started_at=$(date --iso-8601=seconds)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_multiregime_episodes \
    --output "${DUMP_DIR}/shard-$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").h5" \
    --episodes "${EPISODES}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --batch-size 8 \
    --support-size 128 \
    --query-count 32 \
    --max-features 12 \
    --contamination 0.3 \
    --noise 0.0 \
    --seed 2402

echo "finished_at=$(date --iso-8601=seconds)"
