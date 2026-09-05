#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-supervised-tabpfn-aux
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# The existing λ=0.1 array is the middle point.  This companion array runs
# the two remaining auxiliary weights (0.01 and 1) for all three seeds.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEEDS=(2402 2403 2404)
readonly AUX_WEIGHTS=(0.01 1)
readonly SEED_INDEX=$((INDEX % 3))
readonly AUX_INDEX=$((INDEX / 3))
readonly SEED="${SEEDS[${SEED_INDEX}]}"
readonly AUX_WEIGHT="${AUX_WEIGHTS[${AUX_INDEX}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly BENCHMARK_JOB_ID="${BENCHMARK_JOB_ID:?BENCHMARK_JOB_ID is required}"
readonly LABEL="supervised-nanotabpfn-lambda${AUX_WEIGHT}"
readonly RUN_DIR="${RUNS_ROOT}/multiregime_v2/${STUDY_ID}/${LABEL}/seed-${SEED}"
readonly PRIOR_DUMP="${RUNS_ROOT}/multiregime_v2/prior-dump/seed-${SEED}"
readonly GATE_PATH="${PROJECT_DIR}/results/multiregime_v2/benchmark-${BENCHMARK_JOB_ID}/seed-${SEED}/execution_gate.json"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${PRIOR_DUMP}/manifest.json" "${GATE_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing prerequisite: ${path}" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${CODE_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} index=${INDEX}"
echo "model_type=supervised_tabpfn seed=${SEED} aux_regime_weight=${AUX_WEIGHT} run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v2 \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --model-type supervised_tabpfn \
    --input-mode latent \
    --curriculum-mode mixed \
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
    --prior-dump "${PRIOR_DUMP}" \
    --num-slots 4 \
    --aux-regime-weight "${AUX_WEIGHT}" \
    --benchmark-gate-path "${GATE_PATH}"

echo "finished_at=$(date --iso-8601=seconds)"
