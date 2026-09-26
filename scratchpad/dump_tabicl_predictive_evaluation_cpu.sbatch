#!/bin/bash -l
#SBATCH --job-name=dump-tabicl-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# The factorial validation/test bank spans r_z and g_z, their shared and
# multiregime rule modes, K=1..4, and all requested geometry/task slices.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_evaluation/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.multiregime_v4_evaluation \
    --validation-output "${OUTPUT_DIR}/validation.h5" \
    --test-output "${OUTPUT_DIR}/test.h5" \
    --validation-seed 900001 --test-seed 1900001 \
    --episodes-per-cell 8 \
    --support-sizes 64,128,256,512 --num-features 2,4,8,12,16,24 --num-regimes 1,2,3,4 \
    --num-classes 2,3,4,5 \
    --class-ratios 0.1,0.3,0.5 --task-families soft_gate,persistent \
    --mechanism-modes r_z,g_z --rule-modes shared,multiregime \
    --query-size 256 --calibration-size 256 --min-samples-per-regime 32 \
    --pad-features 24 --mlp-probability 0.7
