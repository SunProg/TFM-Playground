#!/bin/bash -l
#SBATCH --job-name=eval-v4-ckpts
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-52
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Re-evaluate every saved checkpoint of a training run on a v4 bank (both
# splits). Written for the 2026-09-17 bank repair (the 252/224 all-constant
# episodes were replaced), which made the in-training v4 histories stale;
# the checkpoints are the only way to re-score those runs. One array task per
# run; task 0 also scores the step-0 (init) model for each size, which is
# shared by all families of that size.
#
# Env overrides: BANK_DIR (default: the repaired z-blind bank), OUT_ROOT
# (default runs_eval/repaired_bank). Outputs mirror the run layout so
# extract_v4_validation_cells.py --base "${OUT_ROOT}" --eval-base "${OUT_ROOT}"
# works unchanged:
#   <OUT_ROOT>/<group>/<task>/seed-2402/v4_validation/step-XXXXXX.json  (every checkpoint, + step-000000 init)
#   <OUT_ROOT>/<group>/<task>/seed-2402/v4_test/step-XXXXXX.json        (every checkpoint)
#   <OUT_ROOT>/<group>/<task>/seed-2402/v4_test/final.json              (final_checkpoint.pth)
# Existing outputs are skipped, so the array can be re-run to fill gaps.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly BANK_DIR="${BANK_DIR:-${PROJECT_DIR}/data/multiregime_v4/tabicl_mix_scm_evaluation/37277969}"
readonly OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/runs_eval/repaired_bank}"
readonly RUNS="/scratch/users/k23139234/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale"
readonly SPECS=(
  37283915:original-small 37283915:r_z-fixed-small 37283915:r_z-curriculum-small 37283915:g_z-fixed-small 37283915:g_z-curriculum-small
  37283921:original-medium 37283921:r_z-fixed-medium 37283921:r_z-curriculum-medium 37283921:g_z-fixed-medium 37283921:g_z-curriculum-medium
  37283995:original-large 37283995:r_z-fixed-large 37283995:r_z-curriculum-large 37283995:g_z-fixed-large 37283995:g_z-curriculum-large
  multiregime_only:r_z-multiregime-small multiregime_only:r_z-multiregime-medium
  # 17-22: mixed r_z+g_z prior runs (pretrain_rg_z_mix.sbatch, z-blind)
  rg_z:rg_z-fixed-small rg_z:rg_z-fixed-medium rg_z:rg_z-fixed-large
  rg_z:rg_z-curriculum-small rg_z:rg_z-curriculum-medium rg_z:rg_z-curriculum-large
  # 23-25: original retrained with the rg_z runs' 20 % warmup (pretrain_original_warmup.sbatch)
  rg_z:original-w02-small rg_z:original-w02-medium rg_z:original-w02-large
  # 26-37: the z-EXPOSED runs scored on the z-BLIND bank (no z column at evaluation)
  expose_z:r_z-fixed-small expose_z:r_z-fixed-medium expose_z:r_z-fixed-large
  expose_z:r_z-curriculum-small expose_z:r_z-curriculum-medium expose_z:r_z-curriculum-large
  expose_z:g_z-fixed-small expose_z:g_z-fixed-medium expose_z:g_z-fixed-large
  expose_z:g_z-curriculum-small expose_z:g_z-curriculum-medium expose_z:g_z-curriculum-large
  # 38-43: the z-exposed rg_z runs on the z-blind bank
  expose_z:rg_z-fixed-small expose_z:rg_z-fixed-medium expose_z:rg_z-fixed-large
  expose_z:rg_z-curriculum-small expose_z:rg_z-curriculum-medium expose_z:rg_z-curriculum-large
  # 44-52: native-prior runs (pretrain_multiregime_v4_native.sbatch); use BANK_DIR/OUT_ROOT for the native banks
  native:original-small native:original-medium native:original-large
  native:rg_z-fixed-small native:rg_z-fixed-medium native:rg_z-fixed-large
  native:rg_z-curriculum-small native:rg_z-curriculum-medium native:rg_z-curriculum-large
)
readonly SPEC="${SPECS[${SLURM_ARRAY_TASK_ID}]}"
readonly GROUP="${SPEC%%:*}"
readonly TASK="${SPEC##*:}"
if [[ "${GROUP}" =~ ^[0-9]+$ ]]; then
  RUN_DIR="${RUNS}/${GROUP}/${TASK}/seed-2402"
else
  RUN_DIR="${PROJECT_DIR}/runs_exp/${GROUP}/${TASK}/seed-2402"
fi
readonly OUT_DIR="${OUT_ROOT}/${GROUP}/${TASK}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5"
test -d "${RUN_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "task=${TASK} run_dir=${RUN_DIR} bank=${BANK_DIR} out=${OUT_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "${OUT_DIR}/v4_validation" "${OUT_DIR}/v4_test"

evaluate() {  # checkpoint split output
  if [[ -f "$3" ]]; then echo "skip $3"; return; fi
  echo "$2 <- $1"
  srun "${PYTHON_BIN}" scratchpad/evaluate_checkpoint_on_v4_bank.py \
    --checkpoint "$1" --bank "${BANK_DIR}/$2.h5" --output "$3" --device cuda --require-cuda
}

for ckpt in "${RUN_DIR}"/checkpoint-*.pth; do
  [[ -f "${ckpt}" ]] || continue
  step="$(basename "${ckpt}" .pth)"; step="${step#checkpoint-}"
  evaluate "${ckpt}" validation "${OUT_DIR}/v4_validation/step-${step}.json"
  evaluate "${ckpt}" test "${OUT_DIR}/v4_test/step-${step}.json"
done
if [[ -f "${RUN_DIR}/final_checkpoint.pth" ]]; then
  evaluate "${RUN_DIR}/final_checkpoint.pth" test "${OUT_DIR}/v4_test/final.json"
else
  echo "no final checkpoint yet for ${TASK}"
fi

# step-0 (init) model per size: same seed + architecture for every family of a size
if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then
  for spec in 37283915:small 37283921:medium 37283995:large; do
    job="${spec%%:*}"; size="${spec##*:}"
    out="${OUT_ROOT}/${job}/original-${size}/seed-2402/v4_validation/step-000000.json"
    if [[ -f "${out}" ]]; then echo "skip ${out}"; continue; fi
    mkdir -p "$(dirname "${out}")"
    echo "init ${size} -> ${out}"
    srun "${PYTHON_BIN}" scratchpad/evaluate_init_on_v4_bank.py \
      --config "${RUNS}/${job}/original-${size}/seed-2402/config.json" \
      --bank "${BANK_DIR}/validation.h5" --output "${out}" --device cuda
  done
fi
