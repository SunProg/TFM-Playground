# TableSlotModel factorial experiment

This experiment uses **TableSlotModel, mode=head**, with all three existing scopes. It never constructs the rejected `CausalRoutingModel`. The switch numbering is specific to this new experiment.

| Switch | Intervention |
|---|---|
| 1 | Give the query decoder blind row representations, produced with support labels replaced by their episode mean. Retain slots from the labelled pass. Both class prediction and routing use the blind row representation and labelled slots. |
| 2 | Randomly hold out 32 of 128 support rows per step. Build a separate context and slots from the remaining 96 rows, then predict the held-out labels. Detached posterior responsibilities supervise the gate and per-slot class predictions, with auxiliary weight 1. |
| 3 | Reduce routing temperature linearly from 2 to 1 over the first 2,000 steps. Simultaneously decay the weight on KL(uniform || episode-mean query usage) from 0.01 to zero. Evaluation uses temperature 1. |

All eight conditions are included: **none, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}**. Crossing these with `data`, `cell_and_data`, and `cell`, and original/fixed/curriculum priors gives **72 table-slot runs**, plus **3 vanilla NanoTabPFN controls**.

The unchanged baseline uses query cross-entropy only, matching the canonical v3 head baseline. `reconstruction_mixture=attention` is retained, but support self-reconstruction and embedding reconstruction both have weight zero. There is no alpha-reconstruction arm. Query routing remains the existing soft decoder-mask mixture; this sweep does not replace it with a new cross-attention decoder.

## Training

All runs start from scratch with seed 11 and 5,000 optimizer steps. Architecture: 6 transformer layers, embedding width 192, hidden width 768, 6 heads, 4 slots, and 3 Slot Attention iterations. Batch size is 8 without gradient accumulation. Each training episode has 128 support and 32 query rows.

AdamW uses peak learning rate 1e-4, weight decay 0.01, gradient clipping at 1, 2,000 warmup steps, then cosine decay toward 1e-6. These are step-matched comparisons. Switch 1 adds a blind forward and switch 2 adds a held-out forward/backward, so compute is not matched.

The canonical `pretrain_multiregime_v3.training_episode` sampler supplies the stream:

- **original:** actual TabICL `mix_scm`, without nuisance group-code columns in training.
- **fixed:** 50% original and 50% v3. Within v3, shared-rule/soft-gate/independent/persistent probabilities are 20/30/20/30%.
- **curriculum:** original only for the first 10% of steps; linearly increase v3 probability to 50% by 40% of steps; hold there afterward.

Mixed modes give original and synthetic episodes the same-width nuisance group-code block. Maximum ordinary features: 12; group-code width: 5; realized groups: 1–5. Synthetic regime counts are 2, 3, or 4 with probabilities 50/30/20%. No regime IDs or true query labels enter model inputs or routing targets.

Backbone initialization and episode streams match across conditions within each prior. Each step resets model RNG independently, so extra forwards cannot advance later steps' random draws. Checkpoints record model class, scope, switches, source hash, initial backbone hash, cumulative episode-stream hash, and realized family counts.

| Model | Parameters |
|---|---:|
| Vanilla NanoTabPFN | 3,711,362 |
| TableSlot, data | 4,935,750 |
| TableSlot, cell_and_data | 5,640,583 |
| TableSlot, cell | 4,861,638 |

Switches add no learned parameters. Counts include the inherited backbone decoder, which the table-slot head does not use.

## Held-out supervision

For held-out row i and slot k, let g be routing probability and p the probability assigned to the held-out label by that slot. The responsibility is `stopgrad(softmax(log g + log p))`. Gate cross-entropy and responsibility-weighted expert cross-entropy are added with unit weight. Their combined gradient equals the held-out marginal NLL gradient. Their numeric sum includes responsibility entropy and is not predictive NLL; the actual held-out marginal NLL is logged separately.

Only the 96 context labels enter the backbone and slot construction. Held-out labels enter the loss only. Each forward uses its own slot identities; slots from different held-out splits are never combined. The split changes with a dedicated deterministic seed each step, shared across arms.

## Evaluation

Validation runs every 500 steps on the same 8 episodes per family, each with **128 support and 128 query rows**. Families: original, shared-rule, soft-gate, independent, and persistent. This gives 40 validation episodes and 5,120 query predictions per checkpoint. Original validation episodes include nuisance codes, matching the canonical mixed-family evaluation bank.

An independent test bank contains 32 episodes per family: 160 episodes and **20,480 query predictions**. Its generation seeds are disjoint from training and validation. After training, both the final 5,000-step model and the checkpoint selected by lowest overall validation NLL are evaluated on this bank. No checkpoint is selected by test performance.

Metrics: query cross-entropy/NLL in nats, accuracy, and episode-level binary AUC. AUC excludes subsets with only one class and reports eligible episode counts. Summary metrics give each eligible episode equal weight. Major regimes are those tied for the largest support count; the remaining regimes are minor. Major/minor metrics apply only to synthetic families.

Routing diagnostics: normalized query-gate entropy, across-query gate standard deviation, hard-slot counts and dominant-slot fraction, effective soft slot count, per-slot prediction disagreement, and support-assignment entropy. Mean-gate and uniform-gate interventions report their change in query NLL. These distinguish nearly uniform soft routing with constant argmax from genuine one-slot domination.

## Interpretation and verification

The `cell` scope retains its historical per-row slot averaging; this sweep does not add cross-row alignment. Blind query rows retain the mean support label, so switch 1 removes individual support-label information from the row input, not every label statistic. No switch guarantees regime specialization.

This is a single-seed factorial screen. Compare paired episode metrics within each scope/prior, and estimate main effects and interactions. Additional training seeds are needed to confirm promising conditions. A 15-task TabArena follow-up is not included in this submission.

Local verification: **76 tests passed**, covering existing table-slot behavior, all scope/switch forward/backward cases, information-path and held-out leakage checks, responsibility-gradient equivalence, temperature-1 compatibility, checkpoint round trips, and the opt-in slot-useful loss. A small real-prior preflight and two-step fixed/all-switch run completed training, validation, checkpointing, and independent-test evaluation. A separate full-size GPU preflight gates the training array.

## Slot-useful-loss pilot

The submitted factorial leaves switch 2 unchanged. An opt-in pilot now adds the
older slot-useful objective without changing the array indices: support labels
are reconstructed through the decoder's alpha mask with weight 1.0, optionally
plus the balanced-sharpness term at weight 0.05. Alpha is used so the support
loss trains the same routing quantity used for query rows. The command-line
axis is `--slot-useful-loss {reconstruction,reconstruction_mi}`.

On the medium MPS smoke architecture (width 64, hidden 256, four layers, 1,000
steps), fix 1+3 plus reconstruction improved the eight-episode-per-family test
NLL from 0.6399 to 0.6366 and accuracy from 0.6270 to 0.6402. Adding MI reduced
support assignment entropy from 0.9371 to 0.8722 but worsened test NLL to 0.6642.
Query gates remained nearly uniform in all three arms (normalized entropy
0.9983--0.9988), so the useful effect is predictive slot training rather than
hard query routing. This is a promising single-seed pilot, not a full-scale
result; the reconstruction-only arm is the one to carry forward.

Array ordering: original 0–24, fixed 25–49, curriculum 50–74. Within each block: data offsets 0–7, cell_and_data 8–15, cell 16–23, vanilla 24. Each scope uses switch order none, 1, 2, 3, 1+2, 1+3, 2+3, 1+2+3.

Remote snapshot on CREATE: `/cephfs/volumes/hpc_data_usr/k23139234/e769188c-13fe-48c6-acb8-28c71b6704fe/TFM-Playground-slot-attention/submissions/table-slot-factorial-v3-20260910-224145/code`.

Tested source SHA-256: `2acf88e4bc6058046316ab461382616c0bb54e2eec115bf5102ed76e3b21c02d`.

Submitted on CREATE: **GPU preflight 37113448** and **training array 37113451**, indices `0-74`, on `biomed_a30_gpu`. The initial four-task concurrency cap was removed at the user's request using `ArrayTaskThrottle=0`. The array requires successful completion of the preflight (`afterok:37113448`) and cancels if that dependency fails. Each training task requests one GPU, four CPUs, 64 GB RAM, and up to 24 hours. The preflight has a 30-minute limit.

The complete task mapping and configuration are saved in `paper/table_slot_factorial_v3_submission.json`. Training outputs live under the snapshot's `runs/37113451/task-<index>/` directory.

The linked TabArena evaluator is `scripts/slurm/evaluate_table_slot_factorial_tabarena_a30.sbatch` (job **37113502**). Its array uses `aftercorr:37113451`, so evaluator task `i` starts after training task `i` succeeds; it does not wait for the other training tasks. It scores that task's `best.pth` against the shared vanilla checkpoint using the established real-label protocol, with vanilla included in each output.
