# v3 pilot: table_slot generalizes zero-shot to unseen families; plain nanoTabPFN doesn't

Date: 2026-09-11

## Status

Run `runs/37113090` (21-cell table_slot sweep, `pretrain_multiregime_v3.py`,
10,000 steps, effective batch 32, embedding=192/heads=6/mlp=768/layers=6),
in progress. This note covers the first 9 completed cells: `plain` and the
`table_slot_head_decoder_baseline`/`table_slot_head_decoder_alpha`
conditions, each crossed with `original`/`fixed`/`curriculum` training mode.

## The finding

Under `original`-mode training, `mixture_probability` is always 0 -- the
model trains exclusively on the unmodified TabICL `mix_scm` prior and never
sees the four v3-generated families (`shared_rule`, `soft_gate`,
`independent`, `persistent`) during training. Both `plain` and `table_slot`
cells share this: zero training exposure to those four families. Yet on
held-out episodes from them:

| family | plain/original | table_slot(baseline\|alpha)/original | delta (log_loss) |
|---|---|---|---|
| shared_rule | 0.6458 / 0.6172 | 0.5705 / 0.7031 | table_slot −0.075 |
| soft_gate | 0.6456 / 0.5898 | 0.6087 / 0.6367 | table_slot −0.037 |
| independent | 0.6379 / 0.6445 | 0.5548 / 0.7227 | table_slot −0.083 |
| persistent | 0.6690 / 0.6133 | 0.5888 / 0.6680 | table_slot −0.080 |

(format: log_loss / accuracy)

**table_slot is consistently and meaningfully better across all four unseen
families**, despite neither model training on them. This is zero-shot
generalization from the slot mechanism itself, not from extra data
exposure -- both models see identical training data under this mode.

Compare `fixed`-mode, where both models *do* see all five families during
training:

| family | plain/fixed | table_slot/fixed | delta (log_loss) |
|---|---|---|---|
| shared_rule | 0.5488 / 0.7461 | 0.5570 / 0.7305 | plain −0.008 |
| soft_gate | 0.6018 / 0.6641 | 0.5924 / 0.6758 | table_slot −0.009 |
| independent | 0.5691 / 0.7188 | 0.5566 / 0.7188 | table_slot −0.013 |
| persistent | 0.5682 / 0.6992 | 0.5797 / 0.6914 | plain −0.012 |

Once plain nanoTabPFN gets direct training exposure to the mixture, the gap
closes to near-nothing and flips sign depending on family -- no consistent
winner either way.

**Reading**: plain nanoTabPFN needs to see the mixture during training to
handle it; table_slot generalizes to it zero-shot, closing most of the gap
to what plain only reaches after direct exposure.

## Caveats

- **Single seed.** No error bars; the deltas above are point estimates from
  one training run per cell.
- **Effectively one independent table_slot data point so far.**
  `table_slot_head_decoder_baseline` and `table_slot_head_decoder_alpha`
  are bit-identical checkpoints under this pilot: `reconstruction_mixture`
  (their only difference) only affects an auxiliary support-reconstruction
  target that the training loop never uses -- the loss is plain query
  cross-entropy (`pretrain_multiregime_v3.py`, one `F.nll_loss` call, no
  reconstruction term). Same seed, same data stream, same optimizer, so
  same gradients throughout training. Confirmed identical to 4 decimal
  places across all 5 families, both cells.
- **Not yet checked**: whether this zero-shot advantage is specific to
  `mode="head"` decoder-gated routing, or holds for `blind_decoder`,
  `blind_similarity`, `backbone`, and `mufasa` too -- those cells are still
  running/queued as of this note.
- **`slot_gate_entropy` is ~0.9998-0.9999 (near-maximal/uniform) in every
  completed slot cell so far**, for every family. The query gate is not
  confidently routing to specific slots despite the zero-shot advantage
  above -- whatever table_slot is doing better than plain here, it is not
  visible as confident slot specialization by this metric. Worth
  reconciling: the mixture-over-slots prediction can still be more accurate
  than a single-head prediction even with a near-uniform gate, if the
  underlying slot decoders themselves specialize in a way this entropy
  metric doesn't capture, or the benefit comes from the model architecture
  (extra slot-attention capacity/pooling) rather than genuine regime
  discrimination.

## Source data

Raw `result.json` for cells 0-8 of `runs/37113090`, fetched via `scp` from
the CREATE checkout. Full log_loss/brier/accuracy/ece_10 per family per
cell was reported in-conversation; this note captures only the
original-vs-fixed zero-shot comparison and its caveats.
