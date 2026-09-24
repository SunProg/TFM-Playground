#!/bin/bash -l
#SBATCH --job-name=dump-expose-z-r_z
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Exp 1 (expose_z control): the r_z paired training dumps regenerated with
# the routing score appended as an extra feature column in every episode
# (expose_z_probability=1.0). Same seed/geometry as the 37277254 dumps, so
# X and labels are identical up to the extra column and a column shuffle.
# CPU-bound; the GPU is only requested because the partition requires one.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly OUTPUT_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_paired_expose_z"
case "${SLURM_ARRAY_TASK_ID}" in
    0) readonly RULE_MODE="shared" ;;
    1) readonly RULE_MODE="multiregime" ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
readonly OUTPUT="${OUTPUT_DIR}/r_z-${RULE_MODE}.h5"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "rule_mode=${RULE_MODE} expose_z=1.0 output=${OUTPUT}"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_multiregime_v4_episodes \
    --output "${OUTPUT}" \
    --episodes 100000 --generation-group-size 4 \
    --min-features 2 --max-features 12 \
    --min-classes 2 --max-classes 5 \
    --min-instances 1024 --max-instances 1024 \
    --min-train-fraction 0.1 --max-train-fraction 0.9 \
    --min-regimes 2 --max-regimes 4 --min-samples-per-regime 32 \
    --calibration-size 256 --pad-features 13 --mlp-probability 0.7 \
    --mechanism-mode r_z --rule-mode "${RULE_MODE}" --seed 2402 \
    --expose-z-probability 1.0
