# Support embedding reconstruction

`pretrain_slot_tabpfn` supports `--embedding-reconstruction-weight` (default 0).
For example, add these flags to a training invocation:

```text
--model-kind table_slot_head --table-slot-scope data \
--embedding-reconstruction-weight 1.0 --support-reconstruction-weight 0.0
```

The objective is query-label NLL plus the weighted embedding MSE. Optional
support-label NLL and slot MI remain independently configurable. This feature
supports `data` and `cell_and_data` head scopes, not cell-only or backbone modes.

The target is the pooled support representation immediately before datapoint
Slot Attention and before its residual rewrite. For `cell_and_data`, this is
after the cell adapter. Targets are detached. The feature decoder predicts
one continuous embedding and mask per support position and slot; softmax masks
combine the embeddings before MSE over batch, rows, and coordinates.

The decoder sees slots and sinusoidal row addresses, never target embeddings
or encoder assignment weights. The same addresses are added to the slot encoder
inputs. This is an explicit experimental architectural change: arbitrary table
row order now affects slot formation. Randomizing support row order during
training and evaluating row permutations are useful robustness checks. This
is not the original position-free table-slot architecture.

A zero weight omits the feature decoder and addresses, preserving the old model.
Enabled checkpoint metadata restores both. Ordinary inference skips feature
decoding but retains addresses, because those are part of the trained encoder.
The alpha/attention setting continues to affect only support-label reconstruction.

Training logs `embedding_reconstruction_mse`; validation reports
`support_embedding_reconstruction_mse` with the existing aggregate statistics.
The target backbone is still trained by query prediction and the slot path:
stop-gradient is not a frozen teacher and does not guarantee absence of collapse.
Track query performance and embedding variance as well as MSE. This change does
not establish improved query routing or predictive accuracy.
