#!/bin/bash -l
#SBATCH --job-name=dump-tabicl-paired
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# The K=1 production control and every paired multiregime construction share
# one geometry, SCM distribution, and controlled C=2--5 label policy. The
# fixed/curriculum runs draw their ordinary branch from this K=1 file.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_paired/${SLURM_ARRAY_JOB_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) readonly MECHANISM_MODE="r_z" RULE_MODE="shared" MIN_REGIMES=1 MAX_REGIMES=1 OUTPUT_NAME="original.h5" ;;
    1) readonly MECHANISM_MODE="r_z" RULE_MODE="shared" MIN_REGIMES=2 MAX_REGIMES=4 OUTPUT_NAME="r_z-shared.h5" ;;
    2) readonly MECHANISM_MODE="r_z" RULE_MODE="multiregime" MIN_REGIMES=2 MAX_REGIMES=4 OUTPUT_NAME="r_z-multiregime.h5" ;;
    3) readonly MECHANISM_MODE="g_z" RULE_MODE="shared" MIN_REGIMES=2 MAX_REGIMES=4 OUTPUT_NAME="g_z-shared.h5" ;;
    4) readonly MECHANISM_MODE="g_z" RULE_MODE="multiregime" MIN_REGIMES=2 MAX_REGIMES=4 OUTPUT_NAME="g_z-multiregime.h5" ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

readonly OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "mechanism_mode=${MECHANISM_MODE} rule_mode=${RULE_MODE} output=${OUTPUT}"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_multiregime_v4_episodes \
    --output "${OUTPUT}" \
    --episodes 100000 --generation-group-size 4 \
    --min-features 2 --max-features 12 \
    --min-classes 2 --max-classes 5 \
    --min-instances 1024 --max-instances 1024 \
    --min-train-fraction 0.1 --max-train-fraction 0.9 \
    --min-regimes "${MIN_REGIMES}" --max-regimes "${MAX_REGIMES}" --min-samples-per-regime 32 \
    --calibration-size 256 --pad-features 13 --mlp-probability 0.7 \
    --mechanism-mode "${MECHANISM_MODE}" --rule-mode "${RULE_MODE}" --seed 2402
