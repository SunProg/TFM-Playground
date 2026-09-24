#!/bin/bash -l
#SBATCH --job-name=dump-v4-native
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0,2,4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Native-TabICL-prior version of the paired v4 training dumps (fix for the
# 2026-09-20 finding that the ``production`` profile calibrated every binary
# task to 50/50): same geometry, seed and pairing as dump job 37277254, but
# ``--hp-profile native`` (TabICL fixed HPs + native Reg2Cls labels). Written
# to a new directory; the old dumps are untouched. Only original (K=1) and
# the two multiregime dumps are needed (array 0,2,4): K=1 is the control, so
# the paired shared-rule dumps are not generated.
#   EXPOSE_Z_PROBABILITY (default 0.5): per-episode probability that the
#   routing score z is appended as an input column (stored per episode as
#   expose_z / z_column_index). 0 = all blind, 1 = all exposed; the directory
#   name carries the setting (…_paired_zmix50 / _paired / _paired_expose_z).
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly EXPOSE_Z_PROBABILITY="${EXPOSE_Z_PROBABILITY:-0.5}"
case "${EXPOSE_Z_PROBABILITY}" in
    0|0.0) readonly SUFFIX="" ;;
    1|1.0) readonly SUFFIX="_expose_z" ;;
    *)     readonly SUFFIX="_zmix$(printf '%s' "${EXPOSE_Z_PROBABILITY}" | sed 's/^0\.//; s/^\.//')" ;;
esac
readonly OUTPUT_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_paired${SUFFIX}"

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
[[ -e "${OUTPUT}" ]] && { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "mechanism_mode=${MECHANISM_MODE} rule_mode=${RULE_MODE} expose_z_probability=${EXPOSE_Z_PROBABILITY} output=${OUTPUT}"
srun "${PYTHON_BIN}" -m tfmplayground.experiments.dump_multiregime_v4_episodes \
    --output "${OUTPUT}" \
    --episodes 100000 --generation-group-size 4 \
    --min-features 2 --max-features 12 \
    --min-classes 2 --max-classes 5 \
    --min-instances 1024 --max-instances 1024 \
    --min-train-fraction 0.1 --max-train-fraction 0.9 \
    --min-regimes "${MIN_REGIMES}" --max-regimes "${MAX_REGIMES}" --min-samples-per-regime 32 \
    --calibration-size 256 --pad-features 13 --mlp-probability 0.7 \
    --mechanism-mode "${MECHANISM_MODE}" --rule-mode "${RULE_MODE}" --seed 2402 \
    --expose-z-probability "${EXPOSE_Z_PROBABILITY}" --hp-profile native
"${PYTHON_BIN}" scripts/report_v4_dump_class_ratios.py --h5 "${OUTPUT}" --episodes 20000
