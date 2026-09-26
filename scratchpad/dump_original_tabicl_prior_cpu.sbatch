#!/bin/bash -l
#SBATCH --job-name=dump-tabicl-original
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Native TabICL reference bank. This is deliberately separate from the V4
# production priors, whose formerly fixed controls are sampled.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/original_tabicl/${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_original_tabicl_prior \
    --output "${OUTPUT_DIR}/original.h5" \
    --episodes 100000 --batch-size 4 --batch-size-per-gp 4 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm --seed 2402
