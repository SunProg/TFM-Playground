#!/usr/bin/env bash
set -euo pipefail

# Run the complete K x seed grid. Set NPROC_PER_NODE to the available GPU count;
# the upstream Trainer handles DDP and adjusts per-process batch sizes.
checkpoint_root="${CHECKPOINT_ROOT:-runs/tabicl_v2_slot_decoder}"
nproc_per_node="${NPROC_PER_NODE:-1}"
launcher=(torchrun --standalone "--nproc_per_node=${nproc_per_node}" -m tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder)
base_args=(
  --device cuda
  --wandb_log False
  --max_steps 10000
  --batch_size 64
  --micro_batch_size 1
)

if [[ -n "${TABICL_BASE_CHECKPOINT:-}" ]]; then
  base_args+=(--base-checkpoint "$TABICL_BASE_CHECKPOINT")
fi

for slot_count in 4 8 16; do
  for seed in 2402 2403 2404; do
    output_dir="${checkpoint_root}/slot-k${slot_count}-seed${seed}"
    "${launcher[@]}" "${base_args[@]}" \
      --head-kind slot \
      --slot-count "$slot_count" \
      --np_seed "$seed" \
      --torch_seed "$seed" \
      --checkpoint_dir "$output_dir"
  done
done
