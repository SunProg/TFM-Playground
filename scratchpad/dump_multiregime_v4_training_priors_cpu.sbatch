#!/bin/bash -l
#SBATCH --job-name=dump-v4-train
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Array task 0 writes the v4 multiregime branch.  Task 1 writes the ordinary
# TabICL mix_scm branch in the repository-standard PriorDumpDataLoader layout.
# A default fixed-ratio run (0.5) streams one of these two files for each
# training episode; curriculum uses the exact same pair with a changing coin.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/data/multiregime_v4/training_priors/${SLURM_ARRAY_JOB_ID}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        readonly OUTPUT="${OUTPUT_DIR}/multiregime_v4.h5"
        echo "multiregime_v4=${OUTPUT}"
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_multiregime_v4_episodes \
            --output "${OUTPUT}" \
            --episodes 100000 \
            --generation-group-size 8 \
            --min-features 4 --max-features 10 \
            --min-instances 64 --max-instances 256 \
            --min-train-fraction 0.1 --max-train-fraction 0.9 \
            --min-regimes 2 --max-regimes 4 \
            --min-samples-per-regime 8 \
            --calibration-size 256 \
            --pad-features 12 \
            --mlp-probability 0.7 \
            --seed 2402
        ;;
    1)
        readonly OUTPUT="${OUTPUT_DIR}/original_mix_scm.h5"
        echo "original_mix_scm=${OUTPUT}"
        srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_original_tabicl_prior \
            --output "${OUTPUT}" \
            --episodes 100000 \
            --batch-size 8 \
            --support-size 128 --query-size 32 \
            --min-features 2 --max-features 12 \
            --max-classes 2 \
            --prior-type mix_scm \
            --seed 2402
        ;;
    *)
        echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac
