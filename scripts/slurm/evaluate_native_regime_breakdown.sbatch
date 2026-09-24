#!/bin/bash -l
#SBATCH --job-name=regime-split
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Re-score the FINAL checkpoint of each native run on both native banks and both
# splits with the per-regime breakdown added on 2026-09-22 (regime_cross_entropy /
# _accuracy / _positive_rate per routing regime inside every K>=2 episode, plus the
# by_regime_position summary).  The existing eval JSONs predate those fields.
# One array task per run; submit in the order large -> medium -> small via --array.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DATA="/scratch/users/k23139234/tfm_data/multiregime_v4_native"
readonly OUT_ROOT="${OUT_ROOT:-/scratch/users/k23139234/tfm_eval/native_regime_split}"
readonly RUNS="${RUNS:-/scratch/users/k23139234/tfm_runs/native}"
readonly SPECS=(
  original-large rg_z-fixed-large rg_z-curriculum-large          # 0-2
  original-medium rg_z-fixed-medium rg_z-curriculum-medium       # 3-5
  original-small rg_z-fixed-small rg_z-curriculum-small          # 6-8
  # 9-18 (RUNS=/scratch/users/k23139234/tfm_runs/native_ablation): the multiregime-prior ablation arms
  rg_z-fixed-softgate-only-medium rg_z-fixed-multiclass-only-medium rg_z-fixed-softgate-multiclass-medium
  rg_z-curriculum-softgate-only-medium rg_z-curriculum-multiclass-only-medium rg_z-curriculum-softgate-multiclass-medium
  rg_z-fixed-softgate-multiclass-large rg_z-curriculum-softgate-multiclass-large
  rg_z-fixed-multiclass-only-large rg_z-curriculum-multiclass-only-large                          # 17-18
)
readonly TASK="${SPECS[${SLURM_ARRAY_TASK_ID}]}"
readonly RUN_DIR="${RUNS}/${TASK}/seed-2402"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"; test -d "${RUN_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "task=${TASK} run_dir=${RUN_DIR} out_root=${OUT_ROOT}"
nvidia-smi --query-gpu=name --format=csv,noheader
test -f "${RUN_DIR}/final_checkpoint.pth" || { echo "no final checkpoint for ${TASK}" >&2; exit 1; }

for bank_tag in blind zx; do
  case "${bank_tag}" in
    blind) BANK_DIR="${DATA}/tabicl_mix_scm_evaluation" ;;
    zx)    BANK_DIR="${DATA}/tabicl_mix_scm_evaluation_expose_z" ;;
  esac
  for split in validation test; do
    out="${OUT_ROOT}/${bank_tag}/${TASK}/seed-2402/v4_${split}/final.json"
    if [[ -f "${out}" ]]; then echo "skip ${out}"; continue; fi
    mkdir -p "$(dirname "${out}")"
    echo "${bank_tag}/${split} <- final_checkpoint.pth"
    srun "${PYTHON_BIN}" scratchpad/evaluate_checkpoint_on_v4_bank.py \
      --checkpoint "${RUN_DIR}/final_checkpoint.pth" --bank "${BANK_DIR}/${split}.h5" \
      --output "${out}" --device cuda --require-cuda
  done
done
