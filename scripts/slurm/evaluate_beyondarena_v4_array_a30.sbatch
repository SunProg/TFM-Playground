#!/bin/bash -l
#SBATCH --job-name=beyond-v4
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-69
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=beyond-v4-%x-%A_%a.out
#SBATCH --error=beyond-v4-%x-%A_%a.err
# BeyondArena v4 evaluation, one array task per model: 15 baseline pretraining
# runs (final + best-own-val checkpoints are selected by the evaluator), the 12
# z-exposed runs, the 5 published baselines and 5 conventional baselines
# (logreg / rf / hgb / xgboost / lightgbm / catboost, default hyper-parameters). Same evaluator/settings as
# evaluate_multiregime_v4_beyondarena_a30.sbatch; results land in
# results/beyondarena-v4/<mode>-<jobid>_<task> (symlinked to scratch) and the
# shared dataset cache is reused.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${BEYONDARENA_PYTHON:-/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python}"
readonly RESULT_ROOT="${BEYONDARENA_RESULT_ROOT:-${PROJECT_DIR}/results/beyondarena-v4}"
readonly DATA_CACHE="${BEYONDARENA_DATA_CACHE:-${PROJECT_DIR}/data-foundry-cache}"
readonly ROWS_MAX="${BEYONDARENA_ROWS_MAX:-11000}"
readonly R="/scratch/users/k23139234/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale"
readonly ZX="${PROJECT_DIR}/runs_exp/expose_z"
readonly RG="${PROJECT_DIR}/runs_exp/rg_z"
readonly NAT="${PROJECT_DIR}/runs_exp/native"
readonly ABL="/scratch/users/k23139234/tfm_runs/native_ablation"   # multiregime-prior ablation arms (2026-09-23)
# mode:run_root[:baseline]
readonly SPECS=(
  original-small:${R}/37283915/original-small       r_z-fixed-small:${R}/37283915/r_z-fixed-small
  r_z-curriculum-small:${R}/37283915/r_z-curriculum-small g_z-fixed-small:${R}/37283915/g_z-fixed-small
  g_z-curriculum-small:${R}/37283915/g_z-curriculum-small
  original-medium:${R}/37283921/original-medium     r_z-fixed-medium:${R}/37283921/r_z-fixed-medium
  r_z-curriculum-medium:${R}/37283921/r_z-curriculum-medium g_z-fixed-medium:${R}/37283921/g_z-fixed-medium
  g_z-curriculum-medium:${R}/37283921/g_z-curriculum-medium
  original-large:${R}/37283995/original-large       r_z-fixed-large:${R}/37283995/r_z-fixed-large
  r_z-curriculum-large:${R}/37283995/r_z-curriculum-large g_z-fixed-large:${R}/37283995/g_z-fixed-large
  g_z-curriculum-large:${R}/37283995/g_z-curriculum-large
  zx-r_z-fixed-small:${ZX}/r_z-fixed-small           zx-r_z-curriculum-small:${ZX}/r_z-curriculum-small
  zx-g_z-fixed-small:${ZX}/g_z-fixed-small           zx-g_z-curriculum-small:${ZX}/g_z-curriculum-small
  zx-r_z-fixed-medium:${ZX}/r_z-fixed-medium         zx-r_z-curriculum-medium:${ZX}/r_z-curriculum-medium
  zx-g_z-fixed-medium:${ZX}/g_z-fixed-medium         zx-g_z-curriculum-medium:${ZX}/g_z-curriculum-medium
  zx-r_z-fixed-large:${ZX}/r_z-fixed-large           zx-r_z-curriculum-large:${ZX}/r_z-curriculum-large
  zx-g_z-fixed-large:${ZX}/g_z-fixed-large           zx-g_z-curriculum-large:${ZX}/g_z-curriculum-large
  tabpfn-v2.2:${R}/37283915/original-small:tabpfn-v2.2
  tabpfn-v2.6:${R}/37283915/original-small:tabpfn-v2.6
  tabpfn-v3:${R}/37283915/original-small:tabpfn-v3
  tabicl-v1:${R}/37283915/original-small:tabicl-v1
  tabicl-v2:${R}/37283915/original-small:tabicl-v2
  logreg:${R}/37283915/original-small:logreg     rf:${R}/37283915/original-small:rf
  hgb:${R}/37283915/original-small:hgb           xgboost:${R}/37283915/original-small:xgboost
  lightgbm:${R}/37283915/original-small:lightgbm
  catboost:${R}/37283915/original-small:catboost
  # 38-49: mixed r_z+g_z prior runs (pretrain_rg_z_mix.sbatch), blind then z-exposed
  rg_z-fixed-small:${RG}/rg_z-fixed-small             rg_z-fixed-medium:${RG}/rg_z-fixed-medium
  rg_z-fixed-large:${RG}/rg_z-fixed-large             rg_z-curriculum-small:${RG}/rg_z-curriculum-small
  rg_z-curriculum-medium:${RG}/rg_z-curriculum-medium rg_z-curriculum-large:${RG}/rg_z-curriculum-large
  zx-rg_z-fixed-small:${ZX}/rg_z-fixed-small          zx-rg_z-fixed-medium:${ZX}/rg_z-fixed-medium
  zx-rg_z-fixed-large:${ZX}/rg_z-fixed-large          zx-rg_z-curriculum-small:${ZX}/rg_z-curriculum-small
  zx-rg_z-curriculum-medium:${ZX}/rg_z-curriculum-medium zx-rg_z-curriculum-large:${ZX}/rg_z-curriculum-large
  # 50-52: original with 20 % warmup (pretrain_original_warmup.sbatch)
  original-w02-small:${RG}/original-w02-small original-w02-medium:${RG}/original-w02-medium original-w02-large:${RG}/original-w02-large
  # 53-61: native-prior runs (mixed-z dumps, 20 % warmup; rg_z-fixed mr 0.3)
  nat-original-small:${NAT}/original-small nat-original-medium:${NAT}/original-medium nat-original-large:${NAT}/original-large
  nat-rg_z-fixed-small:${NAT}/rg_z-fixed-small nat-rg_z-fixed-medium:${NAT}/rg_z-fixed-medium nat-rg_z-fixed-large:${NAT}/rg_z-fixed-large
  nat-rg_z-curriculum-small:${NAT}/rg_z-curriculum-small nat-rg_z-curriculum-medium:${NAT}/rg_z-curriculum-medium nat-rg_z-curriculum-large:${NAT}/rg_z-curriculum-large
  # 62-69: multiregime-prior ablations — the multiregime branch restricted to soft_gate and/or >=3 classes
  abl-fixed-softgate-only-medium:${ABL}/rg_z-fixed-softgate-only-medium
  abl-fixed-multiclass-only-medium:${ABL}/rg_z-fixed-multiclass-only-medium
  abl-fixed-softgate-multiclass-medium:${ABL}/rg_z-fixed-softgate-multiclass-medium
  abl-curri-softgate-only-medium:${ABL}/rg_z-curriculum-softgate-only-medium
  abl-curri-multiclass-only-medium:${ABL}/rg_z-curriculum-multiclass-only-medium
  abl-curri-softgate-multiclass-medium:${ABL}/rg_z-curriculum-softgate-multiclass-medium
  abl-fixed-softgate-multiclass-large:${ABL}/rg_z-fixed-softgate-multiclass-large
  abl-curri-softgate-multiclass-large:${ABL}/rg_z-curriculum-softgate-multiclass-large
)
readonly SPEC="${SPECS[${SLURM_ARRAY_TASK_ID}]}"
IFS=: read -r MODE RUN_ROOT BASELINE <<< "${SPEC}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -d "${RUN_ROOT}"
mkdir -p "${RESULT_ROOT}" "${DATA_CACHE}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export TABPFN_V22_CHECKPOINT="${TABPFN_V22_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.2.ckpt}"
echo "task=${SLURM_ARRAY_TASK_ID} mode=${MODE} run_root=${RUN_ROOT} baseline=${BASELINE:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader
args=(
    --run-roots "${RUN_ROOT}"
    --output-dir "${RESULT_ROOT}/${MODE}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    --data-cache "${DATA_CACHE}"
    --device cuda
    --query-chunk-size 128
    --num-mem-chunks 8
    --rows-max "${ROWS_MAX}"
)
[[ -n "${SMOKE_TASK:-}" ]] && args+=(--smoke-task "${SMOKE_TASK}")
[[ -n "${TASK_NAMES_FILE:-}" ]] && args+=(--task-names-file "${TASK_NAMES_FILE}")
[[ -n "${BASELINE:-}" ]] && args+=(--baselines "${BASELINE}")
srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_multiregime_v4_beyondarena "${args[@]}"
