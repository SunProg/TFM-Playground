#!/bin/bash -l
#SBATCH --job-name=eval-expose-z-bank
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Evaluate checkpoints (z-blind runs, and the expose_z-trained runs) on the z-EXPOSED v4 bank (routing score appended
# as an extra input column). One array task per training run: every saved
# checkpoint-XXXXXX.pth goes against validation.h5, the final checkpoint also
# against test.h5. Outputs mirror the training-run layout under
# runs_eval/expose_z_bank/<group>/<task>/seed-2402/{v4_validation,v4_test}/ so
# extract_v4_validation_cells.py --base runs_eval/expose_z_bank works unchanged.
# Existing outputs are skipped, so re-running the array after a task finishes
# only evaluates what is new.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly BANK_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_evaluation_expose_z"
readonly RUNS="/scratch/users/k23139234/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale"
readonly OUT_ROOT="${PROJECT_DIR}/runs_eval/expose_z_bank"
# group:task:run_dir  (group = the training job id, or the experiment name)
readonly SPECS=(
  37283915:original-small 37283915:r_z-fixed-small 37283915:r_z-curriculum-small 37283915:g_z-fixed-small 37283915:g_z-curriculum-small
  37283921:original-medium 37283921:r_z-fixed-medium 37283921:r_z-curriculum-medium 37283921:g_z-fixed-medium 37283921:g_z-curriculum-medium
  37283995:original-large 37283995:r_z-fixed-large 37283995:r_z-curriculum-large 37283995:g_z-fixed-large 37283995:g_z-curriculum-large
  multiregime_only:r_z-multiregime-small multiregime_only:r_z-multiregime-medium
  expose_z:r_z-fixed-small expose_z:r_z-fixed-medium
  expose_z:r_z-fixed-large expose_z:r_z-curriculum-large expose_z:g_z-fixed-large expose_z:g_z-curriculum-large
  expose_z:r_z-curriculum-small expose_z:r_z-curriculum-medium expose_z:g_z-fixed-small expose_z:g_z-fixed-medium
  expose_z:g_z-curriculum-small expose_z:g_z-curriculum-medium
  # 29-34: mixed r_z+g_z prior runs, z-exposed (pretrain_rg_z_mix.sbatch)
  expose_z:rg_z-fixed-small expose_z:rg_z-fixed-medium expose_z:rg_z-fixed-large
  expose_z:rg_z-curriculum-small expose_z:rg_z-curriculum-medium expose_z:rg_z-curriculum-large
)
readonly SPEC="${SPECS[${SLURM_ARRAY_TASK_ID}]}"
readonly GROUP="${SPEC%%:*}"
readonly TASK="${SPEC##*:}"
if [[ "${GROUP}" == multiregime_only || "${GROUP}" == expose_z ]]; then
  RUN_DIR="${PROJECT_DIR}/runs_exp/${GROUP}/${TASK}/seed-2402"
else
  RUN_DIR="${RUNS}/${GROUP}/${TASK}/seed-2402"
fi
readonly OUT_DIR="${OUT_ROOT}/${GROUP}/${TASK}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5"
test -d "${RUN_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "task=${TASK} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "${OUT_DIR}/v4_validation" "${OUT_DIR}/v4_test"

for ckpt in "${RUN_DIR}"/checkpoint-*.pth; do
  [[ -f "${ckpt}" ]] || continue
  step="$(basename "${ckpt}" .pth)"; step="${step#checkpoint-}"
  out="${OUT_DIR}/v4_validation/step-${step}.json"
  if [[ -f "${out}" ]]; then echo "skip ${out}"; continue; fi
  echo "validation <- ${ckpt}"
  srun "${PYTHON_BIN}" scratchpad/evaluate_checkpoint_on_v4_bank.py \
    --checkpoint "${ckpt}" --bank "${BANK_DIR}/validation.h5" --output "${out}" --device cuda --require-cuda
done

final="${RUN_DIR}/final_checkpoint.pth"
if [[ -f "${final}" ]]; then
  out="${OUT_DIR}/v4_test/final.json"
  if [[ -f "${out}" ]]; then echo "skip ${out}"; else
    echo "test <- ${final}"
    srun "${PYTHON_BIN}" scratchpad/evaluate_checkpoint_on_v4_bank.py \
      --checkpoint "${final}" --bank "${BANK_DIR}/test.h5" --output "${out}" --device cuda --require-cuda
  fi
else
  echo "no final checkpoint yet for ${TASK}; test evaluation deferred"
fi
