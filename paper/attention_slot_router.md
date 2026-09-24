# AttentionSlotRouter

Date: 2026-09-13

Source: `tfmplayground/models/attention_slot_router.py`

## What it is

Two-hop attention retrieval: query row -> slots -> support labels, with no MLP
anywhere in the decoder. It replaces both halves of `ReconstructionRouter`'s MLP
usage (`_SlotDecoder`, used once for support-row reconstruction and again for
query classification) with the same attention-retrieval idiom TabPFN-3's
many-class decoder uses,

$$p_m = \mathrm{softmax}_n(q_m \cdot k_n) \, @ \, y_n,$$

applied twice in sequence rather than once.

## Forward pass

1. **Encode the table.** `NanoTabPFNModel.encode_table((x, support_y.float()), split)`
   produces row embeddings. Query rows are label-blind (mean-padded by
   `TargetEncoder`); support rows keep their real per-row labels. No separate
   masked pass is needed for query rows because the encoder already does this.
2. **Adapt into slots.** `TableSlotAdapter` (scope-dependent: `data`,
   `cell_and_data`, or `cell`) produces `state.slots` and `state.pooled_rows`.
   For `cell`/`cell_and_data` scope the adapter rewrites the table with the
   cell-level competition's own output *before* pooling rows from it, so
   `rows` must come from `state.pooled_rows`, not an independent
   `table.mean(2)` over the raw table -- otherwise `row_query` and `slot_key`
   would live in inconsistent spaces (this was worse for `cell`/`cell_and_data`
   in the first local pilots, before the fix).
3. **Support rows attend to slots -> responsibilities.**
   `inner_similarity = row_query[:, :split] @ slot_key.T / sqrt(width)`,
   softmaxed over slots.
4. **Each slot's value = responsibility-weighted average of real support
   labels.** `slot_value = einsum("bsk,bsc->bkc", slot_weight, y_onehot)`,
   where `slot_weight` is `responsibilities` renormalized per slot (denominator
   `responsibilities.sum(dim=1)`).
5. **Query rows attend to slots -> prediction, directly.**
   `outer_similarity = row_query[:, split:] @ slot_key.T / sqrt(width)`,
   softmaxed to `alpha`; `prediction = einsum("bqk,bkc->bqc", alpha,
   slot_value)`. There is no separate `gate`/`slot_logits`/`log_gate`
   marginalization step -- the second attention's output already *is* the
   class-probability prediction.

One projection (`row_query`) is shared between support and query rows: a row's
blind representation asks the same question of the slots ("which of you do I
belong to / look like") regardless of which side of the support/query split it
sits on -- there is no structural reason for two independently-learned notions
of row-to-slot fit.

## Diagnostics (`self.last`, logged every forward pass)

| key | definition | measures |
|---|---|---|
| `support_nll` / `support_accuracy` | leave-one-out re-prediction of each support row's own label (row `i` excluded from its own slot's value before scoring, the way leave-one-out kNN excludes a point from its own neighborhood) | genuine generalization on the support set, not self-match |
| `mask_row_std` | `responsibilities.std(dim=1).mean()` | how much different *support rows* get routed differently across slots -- low value means every row's responsibility distribution looks alike regardless of content |
| `slot_std` | `slots.std(dim=1).mean()` | how distinct the slot representations are from each other -- low value means slots have collapsed to a shared representation |
| `query_gate_std` | `alpha.std(dim=1).mean()` | same as `mask_row_std` but for query rows |

`mask_row_std`/`query_gate_std` are the more direct read on whether routing
itself is content-sensitive: a model can have well-differentiated slots
(`slot_std` high) while every row still gets a near-identical responsibility
distribution over them (`mask_row_std`/`query_gate_std` low) -- in which case
`prediction` converges to approximately the same slot-value mixture for every
query regardless of its actual features, i.e. every query predicted as
roughly the support set's marginal class rate. This is a cheap, low-dimensional
degenerate solution specific to the slot-retrieval design (a bare
classification transformer has no equivalent shortcut available at the same
representational cost), and is the leading hypothesis for why
`AttentionSlotRouter` trained under `mode="original"` (pure `mix_scm`, severe
per-episode class imbalance, std=0.291 in positive fraction) gets stuck near
chance AUC despite reaching ~74% accuracy "for free" via the marginal-rate
shortcut, while a bare `NanoTabPFNModel` backbone trained on the *same* prior
and batch structure does not get stuck.

## Related classes in the same file

- `AttentionSlotRouterFrozenTabPFN` -- same retrieval logic, but the backbone
  is a frozen real TabPFN model (`FrozenTabPFNBackbone`) instead of a
  jointly-trained `NanoTabPFNModel`. Only `adapter`, `row_query`, and
  `slot_key` receive gradients. See `tabpfn_frozen.py`'s module docstring for
  the column-count caveat this backbone swap introduces (TabPFN groups pairs
  of raw feature columns into one embedding column).

## No separate architecture writeup elsewhere

There is no standalone doc for this model outside this file and the module's
own docstring/inline comments (`tfmplayground/models/attention_slot_router.py`,
top-of-file + inline comments through `forward()`). This file exists so the
architecture and its diagnostics have one place to point to outside the code
itself.
