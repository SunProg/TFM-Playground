#!/bin/bash -l
#SBATCH --job-name=dump-tabicl-matched
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Generate the ordinary control with TabICL's native mix_scm/Reg2Cls prior.
# The existing v4 dump supplies only group geometry (rows, split, and feature
# count); it never supplies SCMs, labels, or samples to this control.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly V4_DUMP_JOB_ID="37258624"
readonly V4_SOURCE="${PROJECT_DIR}/data/multiregime_v4/training_priors/${V4_DUMP_JOB_ID}/multiregime_v4.h5"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/matched_original/${SLURM_JOB_ID}"
readonly OUTPUT="${OUTPUT_DIR}/tabicl_mix_scm_matched_geometry.h5"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${V4_SOURCE}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "geometry_source=${V4_SOURCE}"
echo "output=${OUTPUT}"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_tabicl_mix_scm_matched_geometry \
    --geometry-source "${V4_SOURCE}" \
    --output "${OUTPUT}" \
    --batch-size 8 --max-classes 2 --prior-type mix_scm --seed 2402
