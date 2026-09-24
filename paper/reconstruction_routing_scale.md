# Larger reconstruction-routing study

Designed to test the undercapacity/undertraining explanation for the small pilots.
Submission target: ssh create, biomed_a30_gpu. See the saved submission record for
job IDs; this document by itself does not mean a job has been submitted.

- Backbone: width 192, 6 layers, 6 attention heads, hidden width 768, four slots.
- Budget: 20,000 optimizer updates; batch 8 accumulated four times (effective 32).
- Context: 128 support rows and 32 training queries; validation has 64 queries.
- Prior: the same two-regime SCM family as the pilots, cue separation 1.5,
  regime probability 0.65, label noise 0.05, two task features plus the cue.
- Optimizer: AdamW, learning rate 1e-4, 2,000 warmup steps, cosine decay to 1%.
- Validation: every 500 updates on 32 fixed episodes shared across all arms.
- Scopes: data, cell_and_data, cell; seeds 11 and 12.
- Arms: original label-alpha (query + support NLL + 0.05 MI); embedding MSE with
  original decoder gate; embedding MSE with added support retrieval.
- Baseline: a vanilla NanoTabPFN with the same width, depth, hidden size, heads,
  optimizer, context, and training budget, but no slots, reconstruction auxiliary,
  or support retrieval path. One task per seed is included for a matched control.
- Embedding MSE weights: 0.1 and 1.0. No extra MI in embedding arms.
- Cell has no historical embedding-MSE/original-gate implementation, so this
  unsupported combination is omitted rather than invented. Including the two
  vanilla seeds, the full grid contains 28 tasks.
- Each task requests one GPU, 64 GB RAM, four CPUs, 24 hours; maximum four tasks
  concurrently. Outputs and atomically written latest/best checkpoints live in
  the immutable submission snapshot under runs/<array-id>/task-<index>.
- Training streams fresh episodes with a deterministic step-based seed schedule.
  Matching tasks see identical episodes. Resume restores optimizer state and
  continues the same schedule; STUDY_ID can target a previous output directory.
- Log query/auxiliary backbone-gradient norms at validation steps, validation
  query NLL/accuracy, reconstruction metrics, and peak CUDA allocated memory.

The larger context, batch, learning rate schedule, width and depth all change
relative to the small pilot. This tests a larger training regime, not an isolated
causal effect of width versus steps. The study does not establish benefits across
the separate four-prior mixture sweep or real tabular datasets.
