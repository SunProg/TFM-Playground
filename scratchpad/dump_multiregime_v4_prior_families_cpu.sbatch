#!/bin/bash -l
#SBATCH --job-name=dump-v4-priors
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Build the fixed factorial validation/test banks for every v4 prior-task
# family. Both banks have the same cells and different seed roots:
# support={32,64,128}, features={2,4,8,12}, K={1,2,3,4},
# positive ratio={0.1,0.5,0.9}, family={soft_gate,persistent}; every cell has
# eight independent episodes with 256 query rows. K=1 is the single-rule
# control. The writer rejects an existing target, and the job ID makes every
# submitted bank directory distinct.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/evaluation/${SLURM_JOB_ID}"
readonly VALIDATION_OUTPUT="${OUTPUT_DIR}/validation.h5"
readonly TEST_OUTPUT="${OUTPUT_DIR}/test.h5"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "validation=${VALIDATION_OUTPUT} test=${TEST_OUTPUT}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.multiregime_v4_evaluation \
    --validation-output "${VALIDATION_OUTPUT}" \
    --test-output "${TEST_OUTPUT}" \
    --validation-seed 900001 \
    --test-seed 1900001 \
    --episodes-per-cell 8 \
    --support-sizes 32,64,128 \
    --num-features 2,4,8,12 \
    --num-regimes 1,2,3,4 \
    --class-ratios 0.1,0.5,0.9 \
    --task-families soft_gate,persistent \
    --query-size 256 \
    --calibration-size 256 \
    --min-samples-per-regime 8 \
    --mlp-probability 0.7
