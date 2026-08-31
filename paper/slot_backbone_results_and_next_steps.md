# In-backbone slot attention: results, interpretation, and what to try next

## Status

- Date: 2026-08-31
- Slurm arrays `36864564` (K=2,3,4) and `36864955` (K=5,6,7,8), 28 cells, all COMPLETED
- Branch `exp/slot-attention` at `bafc2d6` and later
- Companions: [`slot_attention_prior_composition_results.md`](slot_attention_prior_composition_results.md)
  (the head variant), [`slot_sweep_vanilla_baseline_metrics.md`](slot_sweep_vanilla_baseline_metrics.md)
  (matched controls), [`slot_attention_implementation_corrections.md`](slot_attention_implementation_corrections.md)
  (defects)

## What was run

Slot attention inside every transformer layer rather than on top of the finished
representation. After each layer's datapoint attention the slots read the
support rows, compete for them, and a learned share of every row state is
reconstructed from them:

```python
mix = torch.sigmoid(self.slot_mix)          # initialized to 0.5
updated = (1 - mix) * target + mix * reconstruction
```

This replaced an earlier zero-initialized `tanh` gate that the optimizer ignored
entirely, leaving twelve runs measuring plain nanoTabPFN (see the corrections
document). 4 prior compositions x 7 slot counts, 10,000 steps, seed 2402.

## Results

`purity - base` is the honest binding quantity; purity alone rises with K
mechanically. `bindAUC` is scored on identifiable rows only, so its ceiling is
1.0 rather than 0.75.

| K | mr_ce range | bindAUC range | purity − base | attH range |
|---|---|---|---|---|
| 2 | 0.6815–0.6859 | 0.506–0.518 | **+0.0000** | 0.91–0.97 |
| 3 | 0.6813–0.6850 | 0.515–0.532 | **+0.0000** | 0.88–0.97 |
| 4 | 0.6807–0.6849 | 0.528–0.533 | **+0.0000** | 0.86–0.94 |
| 5 | 0.6804–0.6838 | 0.525–0.549 | **+0.0000** | 0.81–0.98 |
| 6 | 0.6804–0.6851 | 0.530–0.546 | **+0.0000** | 0.88–0.96 |
| 7 | 0.6811–0.6847 | 0.529–0.548 | +0.0000 to +0.0006 | 0.92–0.98 |
| 8 | 0.6814–0.6854 | 0.535–0.546 | +0.0000 to +0.0006 | 0.93–0.97 |

**The slot path engaged.** `sigmoid(slot_mix)` finished at 0.493 to 0.497 in
every layer of every arm, against 0.5 at initialization. Slots really were
carrying half of each row state; this is not a repeat of the dead-gate no-op.

**Binding did not happen.** `purity - base` is zero to four decimal places in 26
of 28 cells: every support row's argmax still lands on a single slot. `mr_ce` is
0.680 to 0.686 against chance at ln 2 = 0.6931 throughout.

### The apparent K trend is an artifact of the metric

`bindAUC` appears to rise from ~0.51 at K=2 to ~0.54 at K=8. It does not. The
statistic takes the *best-separating* slot, `max over slots of max(auc, 1-auc)`,
so its null grows with K. A permutation null with random attention, 30%
positives, 200 draws per K:

| K | null mean | null p95 | observed |
|---|---|---|---|
| 2 | 0.5150 | 0.5382 | 0.5120 |
| 3 | 0.5253 | 0.5454 | 0.5250 |
| 4 | 0.5289 | 0.5490 | 0.5290 |
| 5 | 0.5300 | 0.5506 | 0.5330 |
| 6 | 0.5331 | 0.5549 | 0.5410 |
| 7 | 0.5331 | 0.5470 | 0.5410 |
| 8 | 0.5355 | 0.5535 | 0.5410 |

**Observed sits at the null mean and below the null p95 at every K.** There is no
binding signal at any slot count. Not weak -- absent. Any future report of this
statistic must quote the null for its K, not 0.5.

## Interpretation

### Why one slot takes everything

Slot attention's competition scores compatibility between an element and a slot.
In vision that is a dot product between a pixel's features -- colour plus a
`SoftPositionEmbed` coordinate grid, applied before the image is flattened to a
set -- and a slot's query. It works because **a pixel's own features determine
which object it belongs to**.

Here they do not. Contamination is assigned by

```python
pos = rng.choice(support_size, size=n_contam_support, replace=False)
support_labels[pos] = labels_other[support_indices][pos]
```

over row *indices*, never consulting `x`. So `P(contaminated | x) = 0.3` for
every `x`. The attention logits are a function of the row embedding, and that
embedding carries no regime information, so whichever slot has the larger
projection wins every row. One-slot-takes-all is the *correct* answer to the
compatibility function we gave it.

### The disagreement region: where structure does exist

The above is about contamination. **Identifiability is a different quantity and
it is feature-structured.** `f_A(x) != f_B(x)` is a deterministic function of
`x`, and measured over 19,200 rows the two label functions agree on 0.507 of
them. Decomposed:

- `P(contaminated | x) = 0.3`, uniform -- no structure.
- `1[f_A(x) != f_B(x)]` -- a real region of feature space, learnable from `x`.
- Inside that region, `contaminated <=> y = f_B(x)`; the label resolves it.
- Outside it, relabelling changed nothing observable and the row is
  uninformative by construction. That is the 50% of contaminated rows that cap
  an oracle at AUC 0.75 on the unrestricted metric.

So a similarity-derived positional encoding cannot identify contamination, but it
could identify *where contamination is detectable*. That is a useful intermediate
the current design never estimates, and it is the one place a positional
encoding could earn its keep. An earlier claim in this work that there is "no
feature-space structure to encode" was too strong and is corrected here.

### The structural mismatch

Object discovery works because objects are spatially contiguous: "which pixels
belong together" genuinely is a positional question. Regimes here are randomly
interleaved by design -- the sampler's own docstring says "nothing in
`(support_x, support_y, query_x)` marks which rows came from which family."

What distinguishes a contaminated row is that its label disagrees with what the
*majority* label function predicts at its `x`. That is a residual against a
hypothesis inferred globally from the other rows, not a property the row carries
alone. Slot attention assumes group membership is inferable from an element's own
features plus competition; that assumption is violated by construction.

## What to try next

### A. Positive control -- do this first

Make contamination feature-dependent (contaminate a half-space rather than
`rng.choice` over indices) and run the existing code unchanged. This is a task
slot attention *should* solve.

It is the only experiment that separates two explanations we currently cannot
distinguish: the implementation is correct and the task violates its inductive
bias, versus the implementation is subtly broken and would fail on object
discovery too. Everything below is worth less until this is answered. Note it
changes the task and is a control, not a proposed method.

### B. Capacity probe with supervision

Run once *with* slot supervision -- as an upper bound, not a method. If slots
told which regime to take still cannot separate the rows, the architecture
cannot represent the split and no unsupervised scheme will find it. Cheap, and
it bounds everything else.

### C. Likelihood compatibility (the principled fix)

Replace the compatibility function so the competition scores what the task
actually distinguishes:

```
current:    logits[n,k] = <k(h_n), q(slot_k)> / sqrt(E)
proposed:   logits[n,k] = log p(y_n | x_n, slot_k)
```

Everything else in the algorithm is unchanged -- softmax over slots, weighted
mean, GRU update, T iterations. Queries, having no label, get no responsibility
and are predicted by the mixture `sum_k pi_k p(y | x_q, slot_k)` with
`pi_k = sum_n a_hat[n,k]`. That is the honest Bayesian answer and it retires the
invented query gate, which has now been got wrong twice.

Under this compatibility two slots that drift to different hypotheses explain
different rows, so competition sustains a split instead of collapsing it.

`hypothesis.py` already computed a per-(row, slot) compatibility as
`row_log_evidence`, but slots were built by one cross-attention pass first and
evidence computed afterwards, so the loop never closed.

### D. Slot initialization

Cannot fix anything on its own: with a compatibility that carries no regime
information, better initialization changes which slot wins, not whether the split
is findable. It becomes load-bearing *given* C, where the symmetric fixed point
is real.

### E. Learner-achievable ceiling

An oracle knowing both label functions caps at AUC 0.75 unrestricted, 1.0 on
identifiable rows. A *learner* must infer both functions from 128 rows at 30%
contamination, and that ceiling is unknown. If it is near 0.55 then nothing built
here will look impressive and we would be chasing noise. Worth computing with an
exact-Bayes baseline over the two known candidates before investing in C.

Suggested order: **A, then B, then C with D, with E alongside.**

## A caution on the EM framing

Proposal C is EM-*shaped*: responsibilities `r[n,k] ∝ pi_k p(y_n | x_n, theta_k)`
are an E-step, and the GRU update plays the M-step. The analogy justifies the
choice of compatibility function and nothing more. It is not EM:

- The T iterations compute slot **activations** for one episode. Backprop updates
  GRU **weights** shared across episodes. The network learns to *perform* an
  EM-shaped update; it does not run EM.
- EM increases likelihood monotonically by construction. The GRU update carries
  no such guarantee.
- EM's E-step is exact and fixed during the M-step. Here responsibilities are
  differentiable and gradient flows through the softmax, so the E-step is trained
  too. Different algorithm.
- EM optimizes observed-data likelihood; this optimizes query cross entropy after
  a fixed T. Nothing forces the iterations to converge; T=3 is a hyperparameter.

The loop is fully unrolled with no detachment, so GRU weights accumulate gradient
recurrently from all T steps -- which is why the update is a GRU rather than a
plain residual.

**Risk specific to C:** with likelihood compatibility the decoder sits both
inside the loop and in the final prediction, so gradient can reduce loss by
making the decoder confidently wrong purely to sharpen responsibilities --
optimizing routing rather than prediction. Detaching the responsibilities, closer
to a true E-step, would block that path and should be tested as a variant rather
than assumed.

## Limitations

- Single seed (2402), 10,000 steps.
- No `competitive=False` ablation on the in-backbone variant.
- Nothing here learned the multiregime task, so these compare models that all
  failed the actual objective.
- The claim that regime is unpredictable from `x` rests on the generator source,
  where `pos = rng.choice(support_size, ...)` never consults `x`. That is a proof
  rather than an estimate, so no empirical probe is reported and none is owed.
