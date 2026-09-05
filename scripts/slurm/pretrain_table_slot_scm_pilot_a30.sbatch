#!/bin/bash -l
#SBATCH --job-name=table-slot-scm
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Twelve independent feature--datapoint table-slot pilots: every placement is
# crossed with plain, multiregime-only, mixed, and curriculum SCM training.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/runs}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly BENCHMARK_JOB_ID="${BENCHMARK_JOB_ID:?BENCHMARK_JOB_ID is required}"
readonly GATE_PATH="${PROJECT_DIR}/results/multiregime_v2/benchmark-${BENCHMARK_JOB_ID}/seed-2402/execution_gate.json"
readonly MODES=(plain multiregime mixed curriculum)
readonly MODELS=(table_slot_head table_slot_backbone table_slot_mufasa)
readonly MODE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
readonly MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
readonly MODE="${MODES[${MODE_INDEX}]}"
readonly MODEL_TYPE="${MODELS[${MODEL_INDEX}]}"
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"
readonly RUN_DIR="${RUNS_ROOT}/table_slot_scm/${STUDY_ID}/${MODE}/${MODEL_TYPE}/seed-2402"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${GATE_PATH}"; do
    [[ -e "${path}" ]] || { echo "Missing prerequisite: ${path}" >&2; exit 2; }
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${CODE_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} mode=${MODE} model=${MODEL_TYPE} seed=2402"
echo "run_dir=${RUN_DIR} validation_interval=500 tabarena_interval=500 progress_interval=50"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_multiregime_v2 \
    --output-dir "${RUN_DIR}" --device cuda --require-cuda --model-type "${MODEL_TYPE}" \
    --seed 2402 --max-steps 5000 --micro-batch-size 8 --accumulate-gradients 4 \
    --validation-interval 500 --validation-episodes 16 --tabarena-interval 500 \
    --tabarena-folds 5 --tabarena-repeats 10 --tabarena-subsample 2048 --progress-interval 50 \
    --support-size 64 --query-size 32 --backend tabicl_scm --difference-components coefficients \
    --single-regime-source matched --curriculum-mode "${MODE}" --num-slots 4 --slot-layer-indices 3 4 5 \
    --aux-regime-weight 0 --benchmark-gate-path "${GATE_PATH}"
