# Coherent regimes and likelihood compatibility: two interventions, both null

## Status

- Date: 2026-08-31
- Branch `exp/slot-attention`, commits `7d7a6cc` (coherence), `f659a72` (compatibility),
  `c39ea58` (compatibility moved onto the coherent task)
- Slurm `36874344` (coherence, dot product) and `36874566` (likelihood/additive),
  both stopped early by hand once the result was clear
- Companion: [`slot_backbone_results_and_next_steps.md`](slot_backbone_results_and_next_steps.md),
  which proposed both interventions

## What was tested

Two independent changes, each attacking a different candidate explanation for
28 cells of flat binding.

### 1. The task: `regime_coherence`

Previously the contaminated rows were a uniform random subset drawn by
`rng.choice` over row indices, so the regime tag was statistically independent
of the features: `P(contaminated | x) = 0.3` for every `x`, and a mechanism that
scores a row against a group had nothing to score.

`regime_coherence` draws them by Gumbel-top-k against a standardized projection
onto a per-episode hyperplane, so they become a feature-coherent group while a
single row's regime stays undetermined by its own features. The hyperplane is
drawn once per episode and shared by support and query rows, so structure
inferred from the support transfers. The count is held exactly, so coherence
cannot confound itself with the contamination rate.

Measured with a logistic probe fitted on support rows and scored on query rows:

| coherence | tag AUC | contamination rate |
|---|---|---|
| 0.0 | 0.506 | 0.297 |
| 1.0 | 0.744 | 0.297 |
| **2.0** (used) | **0.891** | 0.297 |
| 8.0 | 0.980 | 0.297 |

The first row also settles empirically what the generator source already proved,
so the linear probe recorded as owed in the companion document is discharged.

Coherence was deliberately not pushed higher. In the limit the latent variable
disappears and the episode collapses to a single piecewise label function that
needs no mixture at all.

### 2. The model: `slot_compatibility`

    dot          <k(h_n), q(s_k)> / sqrt(E)         Locatello's own
    likelihood   log p(y_n | x_n, s_k)
    additive     dot + softplus(scale) * log p

Written as a bilinear classifier, so the likelihood is still a dot product
against the slot -- with the row's label selecting which of `max_classes` keys
is used, minus the log-partition over classes. The subtraction is the substance:
the bare dot product lets a slot claim rows by growing its norm without ever
predicting them correctly, and the partition term removes that freedom because
inflating a slot inflates its own normaliser too.

## Results

All at `regime_coherence` 2.0. Runs were cancelled at the steps shown.

| run | K | step | mr_ce | bindAUC | purity − base | attH | tabAUC |
|---|---|---|---|---|---|---|---|
| **dot** (`36874344`) | | | | | | | |
| plain | 2 | 4929 | 0.6648 | 0.5234 | +0.0000 | 0.931 | 0.7413@4500 |
| mixed | 2 | 3918 | 0.6654 | 0.5271 | +0.0000 | 0.973 | 0.7475@3500 |
| curriculum | 2 | 3824 | 0.6607 | 0.5214 | +0.0000 | 0.923 | 0.7436@3500 |
| multiregime | 2 | 2643 | 0.6934 | 0.5128 | +0.0000 | 0.988 | 0.4459@2500 |
| **likelihood** (`36874566`) | | | | | | | |
| plain | 2 | 3966 | 0.6710 | 0.5319 | +0.0000 | 0.979 | 0.7375@3500 |
| mixed | 2 | 3141 | 0.6670 | 0.5147 | +0.0000 | 0.944 | 0.7471@3000 |
| curriculum | 2 | 3171 | 0.6706 | 0.5182 | +0.0000 | 0.934 | 0.7346@3000 |
| multiregime | 2 | 2216 | 0.6940 | 0.5122 | +0.0000 | 0.960 | 0.4633@2000 |
| plain | 3 | 3979 | 0.6708 | 0.5369 | +0.0000 | 0.963 | 0.7427@3500 |
| mixed | 3 | 2499 | 0.6776 | 0.5352 | +0.0000 | 0.971 | 0.7238@2000 |
| multiregime | 3 | 1905 | 0.6934 | 0.5261 | +0.0000 | 0.960 | 0.4448@1500 |

Vanilla controls, completed at 5000: plain 0.7149, mixed 0.7099,
curriculum 0.7277, multiregime 0.5615.

### Binding: null under both interventions

Permutation nulls -- K=2 mean 0.5150 / p95 0.5382; K=3 mean 0.5253 / p95 0.5454.
**Every cell sits below its own p95.** `purity - base` is exactly zero
throughout: every support row's argmax still lands on one slot. `attH` is
0.92--0.99, so the attention is near *uniform* -- it is not even a sharp
one-slot-takes-all, it is nobody claiming anything.

| changed | binding |
|---|---|
| the task (coherence 2.0) | flat |
| the compatibility (log p) | flat |
| both together | flat |

Any future report of `bindAUC` must quote the null for its K, never 0.5.

## The diagnosis

**The in-backbone variant's loss never mentions slots.** From `slot_batch_loss`:

```python
output = model(support_x, support_y, query_x)
if isinstance(output, SlotRegimePrediction):
    return slot_regime_loss(output, target)      # mixture NLL -- head variant only
logits = output[..., :2]
return F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1).long())
```

`slot_backbone` returns a plain tensor, so it takes the second branch: one cross
entropy on the final decoder output. The slots are an internal representation
the gradient passes through. Nothing in the objective decomposes over `k`, so
nothing rewards using two slots rather than one.

This subsumes the earlier "no bottleneck" framing, which was too weak. In object
discovery one slot cannot reconstruct a whole image, so leaving a region
unexplained costs loss. Here it is not that the pressure is small -- there is no
*term*. Both interventions changed how slots compete for rows while the training
signal remained indifferent to who won. A perfect compatibility function still
receives no gradient telling it to split, because the loss cannot see a split.

Neither existing variant had both required properties:

| | competition before rows are mixed | loss decomposes over slots |
|---|---|---|
| `slot_regime` (head) | no -- reads the finished representation | yes |
| `slot_backbone` | yes | **no** |
| `slot_backbone_mixture` (new) | yes | yes |

## The unexpected result: TabArena

Slots do not bind to regimes. Slots **do** improve real tables. The two findings
are unrelated, and the second was not what this work set out to measure.

Coherence-0 study, `slot_backbone` K=2 (`36864564`) against matched vanilla
(`36860987`), 10k steps, 20 matched checkpoints:

| prior | mean Δ AUC | Δ after step 1500 | positive | slot final | vanilla final |
|---|---|---|---|---|---|
| plain | +0.0073 | +0.0081 | 17/18 | 0.7379 | 0.7298 |
| mixed | +0.0152 | +0.0213 | 17/18 | 0.7312 | 0.7188 |
| curriculum | +0.0149 | +0.0166 | 17/18 | 0.7412 | 0.7295 |
| multiregime | −0.0028 | −0.0083 | 11/18 | 0.7275 | 0.7179 |

At coherence 2.0 the same comparison is 14/14 positive after step 1500.

Per dataset the gain is mostly **threshold behaviour, not ranking**. On
Bank_Customer_Churn, ROC AUC differs by 0.016 but recall differs 4.6-fold
(0.2309 against 0.0501) and f1 3.8-fold (0.3589 against 0.0945): vanilla
collapses onto the majority class at specificity 0.9994, and the slot arm keeps
some minority recall. Same on E-Commerce, specificity 0.6838 against 0.1376.
`cross_entropy` and `brier` favour the slot arm in 9 of 11 comparable cells.

That is consistent with the dissociation rather than against it. A low-rank
support summary broadcast to every row is a regularizer, and damping
majority-class collapse is the kind of thing a regularizer does. It has nothing
to do with separating regimes.

**Two caveats before this is believed.** `SCREENING_SEED = 2402` in both
studies, so the replication is across task and budget but *not* across seed, and
checkpoints within a run are correlated -- 17/18 is not 18 independent trials.
And `slot_backbone` adds parameters, so the gain could be capacity rather than
competition.

## A second unplanned finding

Coherent multiregime-*only* training destroys real-table transfer:

| multiregime arm | TabArena AUC |
|---|---|
| coherence 0 | 0.7275 |
| coherence 2.0 | **0.4460** -- below chance |

Both slot and vanilla collapse together, so it is the prior rather than the
model. Per-dataset they become degenerate constant predictors -- precision
0.0000 with recall 0.0000, or recall 1.0000 with specificity 0.0000. Coherence
2.0 teaches a feature-dependent label flip that does not exist in real tables,
and a pure diet of it is actively harmful. Mixed and curriculum are unaffected
because they still see ordinary tables.

This is a constraint on the coherent prior, not a bug: it should never be used
at share 1.0.

## Plan

### Now: the mixture readout (`slot_backbone_mixture`)

Slots come from the deepest layer that ran; a per-slot decoder emits class
logits and a routing logit from one pathway (the alpha-mask design the head
variant already uses, so routing cannot drift from decoder competence); the
loss is `slot_regime_loss`, `logsumexp` over `k`, permutation invariant, no
matching anywhere.

Sweep indices 72--87: both compatibilities × K ∈ {2, 3} × four priors, at
coherence 2.0, 10 epochs. Both compatibilities are carried across because the
loss and the score are separate claims, and this is the first time either has
been tested with an objective that can see the split.

### If that is also null

The supervised capacity probe becomes the discriminator: train once *with* slot
supervision, as an upper bound rather than a method. If slots that are told
which regime to take still cannot separate the rows, the blocker is the
architecture and no unsupervised scheme will find it. Specifically it would test
whether `write_back` -- which broadcasts K slots to all rows through one
`MultiheadAttention` -- can carry a per-row distinction at all.

### Regardless of the outcome: confirm the TabArena gain

1. Three seeds, `plain` prior, slot against vanilla. `FINAL_SEEDS = (2402, 2403,
   2404)` already exists for this.
2. `competitive=False` ablation. Identical parameters, softmax over inputs
   instead of slots. Separates "competition helps" from "more parameters help",
   and `--no-competitive-slots` is already wired.

This is currently the only positive result in the project, and it rests on one
seed. It should be settled before anything is written up.

## Limitations

- Single seed (2402) everywhere.
- The coherence and compatibility runs were cancelled between steps 1905 and
  4929, so none reached 5000. Binding was flat from the first validation onward
  in every one, so this is unlikely to change the null, but the TabArena numbers
  are mid-curve and not final.
- The K=3 dot-product cells at coherence 2.0 never ran, so the K=3 likelihood
  cells have no matched dot control at that coherence.
- `additive` produced no results at all; every cell was cancelled before
  starting.
