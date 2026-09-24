#!/bin/bash -l
#SBATCH --job-name=bank-v4-native
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Controlled factorial validation/test bank on the native TabICL base prior:
# same cell grid and seeds as bank 37277969 (class_ratio 0.1/0.3/0.5 binary
# cells keep calibrated labels; multiclass cells use native Reg2Cls labels with
# every requested class present). EXPOSE_Z=1 builds the z-exposed twin.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly EXPOSE_Z="${EXPOSE_Z:-0}"
readonly OUTPUT_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation$([[ ${EXPOSE_Z} == 1 ]] && echo _expose_z)"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"
[[ -e "${OUTPUT_DIR}/validation.h5" ]] && { echo "refusing to overwrite ${OUTPUT_DIR}" >&2; exit 3; }
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
extra=()
PAD=24
[[ "${EXPOSE_Z}" == 1 ]] && { extra=(--expose-z); PAD=25; }   # + the z column
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
    --pad-features "${PAD}" --mlp-probability 0.7 --hp-profile native ${extra[@]+"${extra[@]}"}
for split in validation test; do
    "${PYTHON_BIN}" scripts/report_v4_dump_class_ratios.py --h5 "${OUTPUT_DIR}/${split}.h5" --episodes 40000
    "${PYTHON_BIN}" scratchpad/compute_v4_bank_prior_baselines.py --bank "${OUTPUT_DIR}/${split}.h5" --output "${OUTPUT_DIR}/${split}_prior.json"
done
