#!/bin/bash -l
#SBATCH --job-name=v4-regime-query-rank
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/scratch/users/k23139234/TFM-Playground-slot-attention/.venv/bin/python"
readonly RUN_ROOT="/scratch/users/k23139234/tfm_runs/native"
readonly BANK="/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation/test.h5"
readonly OUTPUT_ROOT="${PROJECT_DIR}/artifacts/evaluations/regime_information_query_ranked"
readonly MODEL_NAMES=(original-large rg_z-fixed-large rg_z-curriculum-large)
readonly MODEL_NAME="${MODEL_NAMES[${SLURM_ARRAY_TASK_ID}]}"
readonly CHECKPOINT="${RUN_ROOT}/${MODEL_NAME}/seed-2402/final_checkpoint.pth"
readonly OUTPUT="${OUTPUT_ROOT}/${MODEL_NAME}/seed-2402/test_query_by_regime.json"

case "${MODEL_NAME}" in
    original-large)
        readonly MODEL_SOURCE="/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/TFM-Playground-slot-attention/tfmplayground/models/nanotabpfn.py"
        readonly HIDDEN_REFERENCE="${RUN_ROOT}/${MODEL_NAME}/seed-2402/v4_test/final.json"
        ;;
    rg_z-fixed-large|rg_z-curriculum-large)
        readonly MODEL_SOURCE="/cephfs/volumes/hpc_home/k23139234/17f71989-c3b3-41d8-9315-37daa2fead54/repo/TFM-Playground-uncertainty/tfmplayground/models/nanotabpfn.py"
        readonly HIDDEN_REFERENCE="${RUN_ROOT}/${MODEL_NAME}/seed-2402/v4_test/final.json"
        ;;
    *)
        echo "Unexpected model: ${MODEL_NAME}" >&2
        exit 2
        ;;
esac

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
test -f "${MODEL_SOURCE}"
test -f "${HIDDEN_REFERENCE}"
test -f "${BANK}"
test ! -e "${OUTPUT}"
mkdir -p "$(dirname "${OUTPUT}")"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "model=${MODEL_NAME} checkpoint=${CHECKPOINT}"
echo "model_source=${MODEL_SOURCE}"
nvidia-smi --query-gpu=name --format=csv,noheader

srun "${PYTHON_BIN}" scripts/evaluate_v4_regime_information_query_level.py \
    --checkpoint "${CHECKPOINT}" \
    --bank "${BANK}" \
    --model-source-path "${MODEL_SOURCE}" \
    --reference-hidden-report "${HIDDEN_REFERENCE}" \
    --output "${OUTPUT}" \
    --device cuda \
    --information hidden shuffled true \
    --seed 20260924 \
    --max-episodes-per-forward 1 \
    --bootstrap-replicates 5000 \
    --query-regime-metrics
