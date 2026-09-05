#!/bin/bash -l
#SBATCH --job-name=slot-ui-full
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Full-sample (uncapped, up to ~10,000 rows) PCA/t-SNE of the exact u_i the
# support competition reads, across all 15 TabArena tasks, for the three
# table_slot_head baselines (job 36961662) and the six closure-loop arms
# (job 36978907: reconstruction-only and reconstruction+MI, three priors
# each).  See tfmplayground/experiments/slot_assignment_distributions.py.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly TASKS_CSV="${PROJECT_DIR}/artifacts/slot-assignment-distributions/tabarena-15/tasks.csv"
readonly BASE_CK="${PROJECT_DIR}/artifacts/slot-assignment-distributions/checkpoints"
readonly CLOSURE_CK="${PROJECT_DIR}/runs/slot_tabpfn/36978907"
readonly OUTPUT_DIR="${PROJECT_DIR}/artifacts/slot-assignment-distributions/tabarena-15-full-sample"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${TASKS_CSV}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_JOB_ID} started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.slot_assignment_distributions \
    --checkpoint "plain=baseline=${BASE_CK}/plain-table_slot_head-s4-coh2/final_checkpoint.pth" \
    --checkpoint "mixed=baseline=${BASE_CK}/mixed-table_slot_head-s4-coh2/final_checkpoint.pth" \
    --checkpoint "curriculum=baseline=${BASE_CK}/curriculum-table_slot_head-s4-coh2/final_checkpoint.pth" \
    --checkpoint "plain=rec=${CLOSURE_CK}/plain-table_slot_head-s4-coh2-rec1/final_checkpoint.pth" \
    --checkpoint "mixed=rec=${CLOSURE_CK}/mixed-table_slot_head-s4-coh2-rec1/final_checkpoint.pth" \
    --checkpoint "curriculum=rec=${CLOSURE_CK}/curriculum-table_slot_head-s4-coh2-rec1/final_checkpoint.pth" \
    --checkpoint "plain=rec_mi=${CLOSURE_CK}/plain-table_slot_head-s4-coh2-rec1-mi0.05/final_checkpoint.pth" \
    --checkpoint "mixed=rec_mi=${CLOSURE_CK}/mixed-table_slot_head-s4-coh2-rec1-mi0.05/final_checkpoint.pth" \
    --checkpoint "curriculum=rec_mi=${CLOSURE_CK}/curriculum-table_slot_head-s4-coh2-rec1-mi0.05/final_checkpoint.pth" \
    --tasks-csv "${TASKS_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --full-sample \
    --num-mem-chunks 32 \
    --folds 5 \
    --seed 2402 \
    --device cuda

echo "finished_at=$(date --iso-8601=seconds)"
