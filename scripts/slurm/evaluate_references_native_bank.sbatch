#!/bin/bash -l
#SBATCH --job-name=ref-native-bank
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Published TabPFN (v2.2 / v2.6 / v3) and TabICL (v1 / v1.1 / v2) on the NATIVE
# controlled bank (validation + test), same per-episode protocol as the runs.
# v3 is sharded into 4 (merge with scratchpad/merge_v4_bank_shards.py).
#   0-1 v2.2 val/test | 2-3 v2.6 | 4-11 v3 shard s x split | 12-17 tabicl v1, v1.1, v2 x split
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly BANK_DIR="${BANK_DIR:-/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_evaluation}"
readonly OUT_DIR="${OUT_DIR:-/scratch/users/k23139234/tfm_eval/native_references}"
readonly -a SPLITS=(validation test)
t="${SLURM_ARRAY_TASK_ID}"
shard=()
if   (( t < 2 ));  then VERSION=v2.2; SPLIT="${SPLITS[$t]}"
elif (( t < 4 ));  then VERSION=v2.6; SPLIT="${SPLITS[$((t-2))]}"
elif (( t < 12 )); then VERSION=v3; s=$(( (t-4) / 2 )); SPLIT="${SPLITS[$(( (t-4) % 2 ))]}"; shard=(--start $(( s * 8064 )) --stop $(( (s + 1) * 8064 )))
else readonly -a TI=(tabicl-v1 tabicl-v1.1 tabicl-v2); VERSION="${TI[$(( (t-12) / 2 ))]}"; SPLIT="${SPLITS[$(( (t-12) % 2 ))]}"
fi
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"; test -f "${BANK_DIR}/${SPLIT}.h5"; mkdir -p "${OUT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
export TABPFN_V22_CHECKPOINT="${TABPFN_V22_CHECKPOINT:-/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.2.ckpt}"
echo "version=${VERSION} split=${SPLIT} shard=${shard[*]:-none} out=${OUT_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader
srun "${PYTHON_BIN}" scratchpad/evaluate_tabpfn_versions_on_v4_bank.py \
    --bank "${BANK_DIR}/${SPLIT}.h5" --versions "${VERSION}" \
    --output-dir "${OUT_DIR}" --device cuda --n-estimators 8 ${shard[@]+"${shard[@]}"}
