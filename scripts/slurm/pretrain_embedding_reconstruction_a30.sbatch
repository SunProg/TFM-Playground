#!/bin/bash -l
#SBATCH --job-name=slot-embedding-priors
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

# Submit from a source snapshot; reuse the cluster environment explicitly.
readonly CODE_ROOT="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the cluster environment Python}"
readonly MODES=(plain multiregime mixed curriculum)
readonly MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
readonly RUN_DIR="${CODE_ROOT}/runs/${SLURM_ARRAY_JOB_ID}/${MODE}/seed-2402"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} mode=${MODE} code=${CODE_ROOT} output=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" --device cuda --require-cuda \
    --model-kind table_slot_head --table-slot-scope data --num-slots 4 \
    --embedding-reconstruction-weight 1.0 --support-reconstruction-weight 0.0 --slot-mi-weight 0.0 \
    --query-routing-mode decoder --prior-mode "${MODE}" --prior-type mix_scm \
    --multiregime-share 0.30 --multiregime-dump none \
    --regime-coherence 2.0 --multiregime-contamination 0.15 \
    --min-features 4 --max-features 4 --max-classes 3 \
    --support-size 128 --query-size 32 --seed 2402 --max-steps 5000 \
    --micro-batch-size 8 --accumulate-gradients 4 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 2000 \
    --validation-interval 500 --validation-batches 16 --validation-episodes 8 \
    --epoch-steps 500 --checkpoint-interval 5000 --no-tabarena-every-epoch
