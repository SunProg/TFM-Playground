#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-k1-train
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Standard nanoTabPFN trained only on matched K=1 episodes.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly PRIOR_DUMP="${RUNS_ROOT}/multiregime_v2/prior-dump-k1-only/seed-${SEED}"
readonly RUN_DIR="${RUNS_ROOT}/multiregime_v2/${STUDY_ID}/k1-only-standard-latent/seed-${SEED}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${PRIOR_DUMP}/manifest.json"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing prerequisite: ${path}" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} seed=${SEED} run_dir=${RUN_DIR} prior_dump=${PRIOR_DUMP}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v2 \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --model-type tabpfn \
    --input-mode latent \
    --curriculum-mode k1_only \
    --seed "${SEED}" \
    --max-steps 10000 \
    --micro-batch-size 8 \
    --accumulate-gradients 4 \
    --learning-rate 0.0001 \
    --min-learning-rate 0.000001 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --validation-interval 5000 \
    --validation-episodes 16 \
    --checkpoint-interval 10000 \
    --support-size 128 \
    --query-size 32 \
    --min-features 2 \
    --max-features 12 \
    --backend analytic \
    --difference-components coefficients \
    --label-noise 0 \
    --gate-strength 1 \
    --single-regime-source matched \
    --prior-dump "${PRIOR_DUMP}"

echo "finished_at=$(date --iso-8601=seconds)"
