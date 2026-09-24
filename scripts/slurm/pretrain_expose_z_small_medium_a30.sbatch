#!/bin/bash -l
#SBATCH --job-name=expose-z-sm
#SBATCH --partition=biomed_a30_gpu
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
# Exp 1 at small/medium size for the families not covered by the first pair of
# runs (r_z-fixed-small/-medium): r_z-curriculum, g_z-fixed, g_z-curriculum,
# pretrained on z-EXPOSED dumps and evaluated on the z-exposed v4 bank. Same
# schedule/architecture as the small/medium baselines (37283915/37283921). Requires the r_z / g_z z-exposed dumps under
# ${DUMP_DIR} (dump_expose_z_{r_z,g_z}_a30.sbatch).
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly DUMP_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_paired_expose_z"
readonly BANK_DIR="/scratch/users/k23139234/tfm_data/multiregime_v4/tabicl_mix_scm_evaluation_expose_z"
readonly -a FAMILIES=(r_z-curriculum g_z-fixed g_z-curriculum)
readonly -a SIZES=(small medium)
readonly FAMILY="${FAMILIES[$((SLURM_ARRAY_TASK_ID / 2))]}"
readonly SIZE="${SIZES[$((SLURM_ARRAY_TASK_ID % 2))]}"
readonly MECHANISM="${FAMILY%%-*}"
readonly SCHEDULE="${FAMILY#*-}"
case "${SIZE}" in
    small)  readonly EMBEDDING_SIZE=64 HEADS=4 HIDDEN_SIZE=256 LAYERS=2 ;;
    medium) readonly EMBEDDING_SIZE=128 HEADS=4 HIDDEN_SIZE=512 LAYERS=4 ;;
esac
# Checkpoints/history go to scratch (hpc_home is at its 50 GB quota); a symlink
# under runs_exp/ keeps the runs_exp/expose_z/<task>/seed-2402 layout for the
# evaluation and extraction scripts.
readonly RUN_DIR="/scratch/users/k23139234/tfm_runs/expose_z/${FAMILY}-${SIZE}/seed-2402"
readonly LINK_DIR="${PROJECT_DIR}/runs_exp/expose_z/${FAMILY}-${SIZE}"
cd "${PROJECT_DIR}"
test -x "${PYTHON_BIN}"
test -f "${DUMP_DIR}/${MECHANISM}-shared.h5" && test -f "${DUMP_DIR}/${MECHANISM}-multiregime.h5"
test -f "${BANK_DIR}/validation.h5" && test -f "${BANK_DIR}/test.h5"
# pretrain_plain_nanotabpfn creates RUN_DIR itself (mkdir exist_ok=False), so only
# its parent may exist here.
mkdir -p "$(dirname "${RUN_DIR}")" "$(dirname "${LINK_DIR}")"
[[ -e "${LINK_DIR}" ]] || ln -s "$(dirname "${RUN_DIR}")" "${LINK_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "family=${FAMILY} size=${SIZE} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_plain_nanotabpfn \
    --output-dir "${RUN_DIR}" --device cuda --require-cuda \
    --seed 2402 --max-steps 10000 \
    --micro-batch-size 4 --batch-size-per-gp 4 --accumulate-gradients 32 \
    --learning-rate 0.0001 --min-learning-rate 0.000001 --warmup-steps 500 \
    --weight-decay 0.01 --gradient-clip 1.0 \
    --validation-interval 500 --validation-batches 16 \
    --v4-validation-bank-path "${BANK_DIR}/validation.h5" --v4-validation-interval 500 \
    --v4-test-bank-path "${BANK_DIR}/test.h5" \
    --checkpoint-interval 2000 --epoch-steps 500 \
    --min-rows 1024 --max-rows 1024 --min-train-size 0.1 --max-train-size 0.9 \
    --min-features 2 --max-features 12 --max-classes 5 --prior-type mix_scm \
    --embedding-size "${EMBEDDING_SIZE}" --num-attention-heads "${HEADS}" --mlp-hidden-size "${HIDDEN_SIZE}" --num-layers "${LAYERS}" \
    --prior-mode "${SCHEDULE}" --multiregime-ratio 0.5 --multiregime-source v4 \
    --v4-shared-dump-path "${DUMP_DIR}/${MECHANISM}-shared.h5" \
    --v4-dump-path "${DUMP_DIR}/${MECHANISM}-multiregime.h5"
