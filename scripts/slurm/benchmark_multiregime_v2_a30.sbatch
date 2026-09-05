#!/bin/bash -l
#SBATCH --job-name=multiregime-v2-gate
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Full locked analytic, mechanism, and SCM benchmark for one trained seed.
# A failed execution gate makes the array task fail, keeping dependent slot
# jobs queued but ineligible to run.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly CODE_ROOT="${CODE_ROOT:-${PROJECT_DIR}}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/multiregime_v2}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SEEDS=(2402 2403 2404)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
: "${TRAIN_STUDY_ID:?TRAIN_STUDY_ID is required}"
readonly STANDARD="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/standard-latent/seed-${SEED}/final_checkpoint.pth"
readonly ORACLE="${PROJECT_DIR}/runs/multiregime_v2/${TRAIN_STUDY_ID}/oracle-one-hot/seed-${SEED}/final_checkpoint.pth"
readonly OUTPUT_DIR="${RESULTS_ROOT}/benchmark-${SLURM_ARRAY_JOB_ID}/seed-${SEED}"

cd "${CODE_ROOT}"
for path in "${PYTHON_BIN}" "${STANDARD}" "${ORACLE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing prerequisite: ${path}" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${CODE_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} seed=${SEED} output=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds) host=$(hostname)"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.benchmark_multiregime_v2 \
    --output-dir "${OUTPUT_DIR}" \
    --tabpfn-checkpoint "${STANDARD}" \
    --oracle-one-hot-checkpoint "${ORACLE}" \
    --episodes-per-seed 32 \
    --query-size 256 \
    --bootstrap-samples 2000 \
    --cpu-workers 8 \
    --inference-batch-size 4 \
    --generation-workers 8 \
    --no-persist-episodes \
    --compress-predictions \
    --device cuda

"${PYTHON_BIN}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p,sort_keys=True)); raise SystemExit(0 if p["passed"] else 3)' \
    "${OUTPUT_DIR}/execution_gate.json"

echo "finished_at=$(date --iso-8601=seconds)"
