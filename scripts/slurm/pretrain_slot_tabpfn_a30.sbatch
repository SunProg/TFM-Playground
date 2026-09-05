#!/bin/bash -l
#SBATCH --job-name=slot-tabpfn-sweep
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-218
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# The slot-TabPFN screening grid: 4 prior compositions x model variant.
#
#                            plain   multiregime   mixed(70/30)   curriculum
#   slot head       K=2        0          1            2              3
#                   K=3        4          5            6              7
#                   K=4        8          9           10             11
#   vanilla                   12         13           14             15
#   slot_backbone   K=2       16         17           18             19
#                   K=3       20         21           22             23
#                   K=4       24         25           26             27
#                   K=5..8    28 .. 43
#   coherent regime (regime_coherence 2.0, 10 epochs)
#                   vanilla   44         45           46             47
#                   K=2       48         49           50             51
#                   K=3       52         53           54             55
#   learnable design (3 classes, 4 features, 15% contamination, 10 epochs)
#     vanilla                 88         89           90             91
#     slot head K=2           92         93           94             95
#     slot_backbone K=2       96         97           98             99
#     slot_mixture  K=2      100        101          102            103
#   positive control (coherence 8.0, learnable design) -- multiregime/mixed only
#     slot head / backbone / mixture      104-105  106-107  108-109
#   slots after feature attention -- matched to controls 106-109
#     backbone / mixture                  110-111  112-113
#   table slots (K=4, coherence 2.0, 10 epochs), both competitions
#     head / backbone / mufasa           114-117  118-121  122-125
#   table-slot scope ablation: cell 126-137, data 138-149
#     head / backbone / mufasa, four priors each, in that order
#   readable design (3 classes, 512 support, 15-dataset TabArena) -- no
#   multiregime arm, so three priors per cell rather than four
#     cell_and_data 150-158, cell 159-167, data 168-176, vanilla 177-179
#   closure objective (support-label reconstruction) -- matched to 114/116/117
#     reconstruction only            180  181  182   (plain/mixed/curriculum)
#     reconstruction + MI 0.05       183  184  185
#   query routing, at the passing closure weights, crossed with every scope
#     blind_decoder    186-194, blind_similarity 195-203
#     (cell_and_data / cell / data, three priors each, in that order)
#   compositing rule, on the learnable design -- see the note at the bottom
#     alpha    blind_decoder 204-205, decoder 206-207
#     baseline blind_decoder 208-209, decoder 210-211
#     (mixed / curriculum; `plain` is the control below)
#   compositing negative control -- `plain`, matched to 204-211
#     alpha 212-213, baseline 214-215
#   blind_similarity on the learnable design (plain/mixed/curriculum) 216-218
#
#   compatibility (coherence 2.0, 10 epochs) -- what a slot's claim is scored by
#     likelihood    K=2       56         57           58             59
#                   K=3       60         61           62             63
#     additive      K=2       64         65           66             67
#                   K=3       68         69           70             71
#
# Blocks are appended, never reordered, so an in-flight array keeps its
# index-to-arm mapping when the grid is extended.  Indices 0-43 all run at
# regime_coherence 0, where the contaminated rows are a uniform random subset;
# 44-55 change that assignment to a feature-coherent group and are the only
# cells that generate episodes rather than streaming the dump, which holds
# coherence-0 episodes.  Their run directories carry a `-coh2` suffix so they
# cannot resume from, or be confused with, their coherence-0 counterparts.
# Indices 110-113 use the same positive-control task as 106-109 but move each
# layer's slot competition before datapoint attention.  Their directories carry
# an `-after-feature` suffix.
#
# 88-99 are the only cells trained on a task whose achievable detection AUC is
# not chance.  detection_ceiling.py puts every earlier design at ~0.505, so a
# model number on those could never be read: "the model failed" and "nothing
# can succeed" produce the same value.  These run at 0.742 instead, with one
# label per row and the regime still latent, so discovery is still required.
# TabArena stays on: at max_classes=3 the prior mixes 2- and 3-class episodes,
# so the model learns to use its first two outputs for binary tables.
#
# 56-71 change the model rather than the task: the competition scores
# `log p(y | x, slot)` instead of a dot product, holding the coherent task
# fixed so the compatibility function is the only thing varying.  Their
# directories carry a `-likelihood` / `-additive` suffix for the same reason,
# and they generate episodes too.  Indices 48-55 are their matched
# dot-product controls, 44-47 the vanilla ones.
#
# 180-185 change the objective rather than the task or the placement.  Every
# cell above trains the head on the query mixture NLL alone, which reads only
# the slot logits and the query gate -- the support competition carries no
# gradient at all, so one slot claiming every row costs nothing, and
# `purity - base` has been exactly zero throughout.  These add the tabular
# reading of Slot Attention's mask-weighted reconstruction: the assignment must
# explain the labels it claims.  Their directories carry a `-rec1` / `-rec1-mi`
# suffix, and 114/116/117 are the matched baseline at identical settings.
#
# 186-203 change neither the task nor the objective but the *gate*: every cell
# above routes a query row through the learned mask channel on the labelled-pass
# embedding, which has already attended over real support labels through
# ordinary self-attention before the competition runs, so a query can be
# predicted well without the slots contributing anything.
#
# 204-211 change the compositing rule.  Every cell above -- closure and routing
# alike -- weights the support reconstruction by `a[i,k]` and discards the
# decoder's alpha channel, while the query side gates on that discarded channel
# and never sees `a`: two routing quantities, one of them trained.  Locatello
# has no such split -- the decoder emits content and an unnormalized alpha from
# one pathway, the alphas are softmaxed across slots, and that softmax *is* the
# mixture weight the reconstruction is scored through.  `--reconstruction-mixture
# alpha` composites that way, so `L_rec` and the query mixture become the same
# expression evaluated on rows that do and do not carry a target.  Their
# directories carry an `-alpha` suffix and 208-211 are the matched baseline at
# identical settings.
#
# 212-215 are their negative control.  `plain`'s multiregime share is 0.00 at
# every step, so those cells never train on a mixture -- which is the point.
# Compositing by alpha does one thing unrelated to mixtures: it puts `L_rec`
# gradient on the mask head, which the `a[i,k]` weighting never did.  A gap on
# `mixed` alone cannot separate "alpha splits regimes better" from "training the
# mask head is good for the model"; a matched pair on `plain` can.  Appended
# rather than folded in, because 204-211 were already running.
#
# These are the only cells after 103 that run on the learnable design, and that
# is deliberate: `detection_ceiling` puts the coherence-2.0 design every block
# from 104 on uses at ~0.505 achievable, so a flat number there cannot be read
# -- "the fix did nothing" and "nothing could have worked here" produce the same
# value.  Screening a routing fix against an unmeasurable task would repeat the
# mistake 88-103 exist to correct.
#
# 216-218 put `blind_similarity` on the learnable design.  195-203 already
# screen that gate, but on the coherence-2.0 design at ~0.505 achievable, where
# a flat result cannot be read.  These sit alongside 208-215 so all three gates
# are comparable at baseline compositing.  Baseline only: the mode replaces the
# decoder's alpha, so there is no alpha for the compositing axis to act on.
#
# Run a subset with `sbatch --array=<range>`, which overrides the directive
# above; the log below prints the array count against the grid size so a partial
# submission is visible rather than silently skipping cells.
#
# Every other training flag is identical across the arms and matches
# pretrain_plain_nanotabpfn_a30.sbatch, so the training prior is the only
# difference -- both within this sweep and against the vanilla baseline runs.
# The index-to-flags mapping lives in slot_tabpfn_sweep.py, not here.
set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
# Pre-generated by dump_multiregime_episodes.sbatch; streaming these instead
# of generating them is what keeps the multiregime arm inside the time limit.
readonly DUMP_DIR="${PROJECT_DIR}/runs/multiregime_dump"
# Run output lives under runs/slot_tabpfn/<STUDY_ID>/<label>. Defaults to a
# fresh directory per submission (SLURM_ARRAY_JOB_ID); to continue a failed
# task from its last checkpoint instead of restarting at step 0, resubmit
# with the earlier job's id, e.g.:
#   STUDY_ID=<earlier_array_job_id> sbatch --array=<failed_indices> pretrain_slot_tabpfn_a30.sbatch
readonly STUDY_ID="${STUDY_ID:-${SLURM_ARRAY_JOB_ID}}"

cd "${PROJECT_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable Python environment at ${PYTHON_BIN}" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
# The large-support cells sit close to the card's limit, where a fragmented
# allocator fails on an allocation the free memory could otherwise satisfy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRIAL_LABEL="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --label "${SLURM_ARRAY_TASK_ID}")"
TRIAL_FLAGS="$("${PYTHON_BIN}" -m tfmplayground.experiments.slot_tabpfn_sweep --index "${SLURM_ARRAY_TASK_ID}")"
readonly TRIAL_LABEL TRIAL_FLAGS
if [[ "${TRIAL_FLAGS}" != *"--multiregime-dump none"* && ! -d "${DUMP_DIR}" ]]; then
    echo "Missing multiregime dump at ${DUMP_DIR}; run dump_multiregime_episodes.sbatch first" >&2
    exit 2
fi
readonly STUDY_DIR="${PROJECT_DIR}/runs/slot_tabpfn/${STUDY_ID}"
readonly RUN_DIR="${STUDY_DIR}/${TRIAL_LABEL}"
readonly ROLLING_CHECKPOINT="${RUN_DIR}/epoch-latest-checkpoint.pth"
RESUME_FLAGS=()
if [[ -f "${ROLLING_CHECKPOINT}" ]]; then
    echo "resuming_from=${ROLLING_CHECKPOINT}"
    RESUME_FLAGS=(--resume-from "${ROLLING_CHECKPOINT}")
fi
readonly RESUME_FLAGS

GRID_SIZE="$("${PYTHON_BIN}" -c 'from tfmplayground.experiments.slot_tabpfn_sweep import screening_configurations as c; print(len(c()))')"
readonly GRID_SIZE
echo "grid_size=${GRID_SIZE} array_task_count=${SLURM_ARRAY_TASK_COUNT}"
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne "${GRID_SIZE}" ]]; then
    echo "NOTE: running ${SLURM_ARRAY_TASK_COUNT} of ${GRID_SIZE} grid cells (subset submission)"
fi
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "trial_label=${TRIAL_LABEL}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" -c "import torch; print(f'cuda_version={torch.version.cuda} torch={torch.__version__}')"

srun "${PYTHON_BIN}" -m tfmplayground.experiments.pretrain_slot_tabpfn \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --require-cuda \
    --multiregime-dump "${DUMP_DIR}" \
    "${RESUME_FLAGS[@]}" \
    ${TRIAL_FLAGS}

echo "finished_at=$(date --iso-8601=seconds)"
