#!/bin/bash -l
#SBATCH --job-name=tabicl-v2-decoder-tabarena
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Official TabArena-v0.1 lite multiclass comparison for slot and random-MLP runs.
# Submitted with afterok dependencies on the remaining Stage-1 jobs.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly OUTPUT_DIR="${PROJECT_DIR}/results/tabicl_v2_slot_decoder/tabarena-lite-${SLURM_JOB_ID}"
readonly RESULTS_DIR="tabicl_v2_slot_decoder_tabarena_lite_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
mkdir -p "${PROJECT_DIR}/results/tabicl_v2_slot_decoder" "${PROJECT_DIR}/.cache/openml"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/.tabicl-slot-site${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export TABPFN_DISABLE_TELEMETRY=1

declare -a DECODERS=()
add_decoder() {
    local name="$1"
    local checkpoint="$2"
    test -f "${checkpoint}" || { echo "Missing completed checkpoint: ${checkpoint}" >&2; exit 3; }
    DECODERS+=(--decoder "${name}=${checkpoint}")
}
add_decoder slot-k4-seed2402 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577418/slot-k4-seed2402/step-500.ckpt"
add_decoder slot-k4-seed2403 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577436/slot-k4-seed2403/step-500.ckpt"
add_decoder slot-k4-seed2404 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577440/slot-k4-seed2404/step-500.ckpt"
add_decoder slot-k8-seed2402 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577442/slot-k8-seed2402/step-500.ckpt"
add_decoder slot-k8-seed2403 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577443/slot-k8-seed2403/step-500.ckpt"
add_decoder slot-k8-seed2404 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577444/slot-k8-seed2404/step-500.ckpt"
add_decoder slot-k16-seed2402 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot-resumed/k16-seed2402/step-500.ckpt"
add_decoder slot-k16-seed2403 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577448/slot-k16-seed2403/step-500.ckpt"
add_decoder slot-k16-seed2404 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577450/slot-k16-seed2404/step-500.ckpt"
add_decoder mlp-random-seed2402 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-mlp-random/job-37580198/mlp-random-seed2402/step-500.ckpt"
add_decoder mlp-random-seed2403 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-mlp-random/job-37580199/mlp-random-seed2403/step-500.ckpt"
add_decoder mlp-random-seed2404 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-mlp-random/job-37580209/mlp-random-seed2404/step-500.ckpt"
add_decoder tabpfnv3-random-seed2402 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-tabpfnv3-random/job-37580523/tabpfnv3-random-seed2402/step-500.ckpt"
add_decoder tabpfnv3-random-seed2403 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-tabpfnv3-random/job-37580524/tabpfnv3-random-seed2403/step-500.ckpt"
add_decoder tabpfnv3-random-seed2404 "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-tabpfnv3-random/job-37580525/tabpfnv3-random-seed2404/step-500.ckpt"

readonly ZERO_SHOT_CHECKPOINT="$("${PYTHON_BIN}" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["slot_decoder"]["base_checkpoint_path"])' "${PROJECT_DIR}/runs/tabicl_v2_slot_decoder/stage1-pilot/job-37577418/slot-k4-seed2402/step-500.ckpt")"
test -f "${ZERO_SHOT_CHECKPOINT}"

echo "TabArena job=${SLURM_JOB_ID}; zero-shot baseline plus ${#DECODERS[@]} decoder checkpoints; output=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun "${PYTHON_BIN}" -m tfmplayground.experiments.evaluate_tabicl_v2_slot_decoder     "${DECODERS[@]}"     --zero-shot-checkpoint "${ZERO_SHOT_CHECKPOINT}"     --output-dir "${OUTPUT_DIR}"     --results-dir "${RESULTS_DIR}"
