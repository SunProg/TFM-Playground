# Revised plan: reconstruction-weighted query-to-support routing

Status: isolated learning-pilot implementations and results now exist (see below);
full pretraining integration is not submitted. Existing array
37071623 remains the data-scope embedding-MSE experiment with decoder-alpha
query routing. This plan supersedes the proposal to infer slot responsibilities
from each slot's individual embedding MSE.

## Objective and interpretation

Keep query-label NLL plus support embedding MSE. Add explicit query-to-support
attention whose values are the embedding decoder's support-slot masks. This
connects reconstruction to query routing. Reconstruction error measures fidelity
to a support embedding, not proven label reliability or generalization confidence.

Do not use softmax of negative individual slot reconstruction errors as
responsibilities: the current loss trains the mixed output, allowing slot
predictions to compensate for one another.

## Routing equations and initial settings

For support row i, let h_i be its detached embedding target and let the embedding
decoder produce v_ik and alpha_ik = softmax_k(mask_ik). Retain both the masks and
the mixed reconstruction hhat_i = sum_k alpha_ik v_ik in the model output.
Compute e_i = mean_d (hhat_id - h_id)^2 before reducing over rows or batches.

Use label-blind pre-adapter backbone row embeddings b for similarity, so added
row addresses and slot residual write-back are not direct query/key inputs:

    similarity_qi = Q(b_q) dot K(b_i) / sqrt(d)
    A0_qi = softmax_i(similarity_qi)
    u_i = stopgrad(e_i / (e_i + mean_support(e) + epsilon))
    A1_qi = softmax_i(similarity_qi - beta * u_i)
    A_qi = (1 - eta) * A0_qi + eta * A1_qi
    R_ik = stopgrad(alpha_ik)
    w_qk = sum_i A_qi * R_ik
    p(y_q) = sum_k w_qk * p_k(y_q | query, slot_k)

Start with beta=0.5 and eta=0.25, fixed and serialized. u is in [0,1]; the
bounded penalty limits relative reweighting, and the residual attention ensures
A_qi >= 0.75 * A0_qi. Equal errors leave A unchanged. eta=0 is the exact
similarity-only control. Hyperparameters may subsequently be chosen using
validation query NLL, never final test results.

Detach both u and R for the initial experiment. With non-degenerate support
masks and differing expert predictions, query NLL can train Q/K. It also trains
the existing class decoder and shared slots; MSE trains the embedding decoder and
slot encoder. Detachment blocks a direct query-loss path through reliability
scores, but shared representations can still change those scores indirectly.
Do not claim that masks are calibrated responsibilities or latent regime IDs.

Compute support reconstruction at inference for the new routing modes as well
as during training. Do not accidentally use the existing inference shortcut
that skips embedding decoding. No query labels enter any routing calculation.
The additional blind backbone pass and reconstruction decoder have measurable
runtime/memory cost; record both.

## Scope-specific implementation

### data

Reconstruct pooled support rows before datapoint Slot Attention's residual
rewrite. Decoder masks already index the episode's data slots, so R maps
directly to the class decoder's slot axis.

### cell_and_data

Keep the existing pooled-row reconstruction after cell processing, through
final data slots. Route with those data-slot masks. This first experiment does
not add a second cell reconstruction loss. Preserve the existing objective so
routing comparisons within this scope remain controlled.

### cell

Add a cell decoder that reconstructs each support row's original encoded cells
from that row's cell slots plus column addresses, with no target-copy input.
Add matching addresses to cell-slot inputs. Reconstruction targets are the
pre-cell-adapter embeddings; query cells are excluded from reconstruction loss.

Per-row cell slots have no guaranteed shared ordering. Align their slot sets
before indexwise episode aggregation. Prototype: choose a support-row slot-set
medoid under pairwise optimal-matching cosine cost, then use Hungarian matching
to align each support row to that reference. Apply the SAME permutation to slot
vectors and reconstruction masks. Handle ties deterministically; measure cost
and tie frequency. This is computationally more expensive and alignment quality
must be assessed rather than treated as established semantic correspondence.

For cell reconstruction masks alpha_ick, aggregate after alignment:

    R_ik = mean_c aligned(alpha_ick)
    e_i = mean_c,d (hhat_icd - h_icd)^2

R remains normalized over slots. Feed aligned support-slot means into the class
decoder. Keep all cell routing arms on this same alignment/decoder architecture,
including the decoder-alpha baseline, to isolate routing effects. Compare raw
MSE only within a scope: cell and pooled-row targets differ.

## Verification gates

1. Expose un-reduced errors/masks and confirm their reductions reproduce MSE.
2. Verify attention and slot weights are finite, nonnegative, and normalized;
   eta=0 exactly reproduces similarity-only routing, including gradients.
3. Verify equal-error invariance, bounded influence, and query-specific weights.
4. Check query-loss gradients reach Q/K but not the detached error/mask branches;
   reconstruction gradients still reach its decoder and slot encoder.
5. Check missing required reconstruction raises a clear error, and checkpoint
   round trips preserve routing, scope, beta, eta, and normal inference behavior.
6. Test the aggregation/alignment layer under independent slot permutations and
   support-row permutations. Full-model row-order robustness is an empirical
   check because existing row addresses intentionally break permutation symmetry.
7. Run minority-regime stress tests: higher reconstruction error must not be
   described as lower label reliability without supporting measurements.
8. Shuffle error scores while retaining masks, then shuffle masks separately,
   as distinct evaluation ablations to identify which signal helps.
9. Report query NLL/accuracy, slot usage/diversity, attention concentration,
   subgroup performance, embedding variance, and per-row reconstruction/label
   error association. Include runtime and memory.

## Experiment grid and execution order

Three scopes (data, cell_and_data, cell) x three matched routing arms
(decoder alpha, similarity-only support routing, bounded-error support routing)
x four priors (plain mix_scm, multiregime, mixed 70/30, curriculum): 36 runs.
One initial matched seed; repeat promising comparisons with additional seeds.
Use the existing 5,000-step task design and embedding-loss weight for screening.

Implement and test data first, then cell_and_data, then cell reconstruction and
alignment. Do not launch a full grid until those gates pass and a small learning
smoke test is finite for each scope. Submit isolated snapshots through ssh create
on biomed_a30_gpu. Record job IDs and do not overwrite existing submissions.


## Numerical verification results

Reproducible command:

```text
.venv/bin/python scripts/verification/verify_reconstruction_routing.py
```

Nine algebra checks pass: normalization, finite/nonnegative weights, the 75%
attention floor, detached-score gradients, exact eta-zero values and gradients,
equal-error invariance, joint support-permutation invariance, slot-permutation
equivariance, and the ability to produce distinct weights for distinct queries.
These check a standalone formula prototype, not a production implementation.

Two degeneracies were reproduced and must be additional implementation gates:

- If R is identical for every support row, w is identical for every query,
  regardless of relevance or errors. The routing gradient is zero up to floating
  point noise (2.8e-8 in the check). Track variation of masks across support rows
  and actual query-loss gradients into Q/K. Finite losses alone are insufficient.
  Uniform mask usage across an episode does not establish this variation.
- Cell-mask averaging discards column structure. Different per-cell partitions
  can yield identical row responsibilities. Alignment fixes index correspondence
  but cannot restore information lost by averaging. The cell-only aggregation
  remains a prototype, not a validated equivalent of data-slot routing.

Before the full grid, require a short learning pilot for each scope that checks
these degeneracies, expert prediction diversity, and held-out query performance.
Keep decoder-alpha available as the baseline; do not silently introduce a fallback
into the candidate arms. If cell averages are uninformative, revise that scope to
preserve query-conditioned cell information, and verify that alternative before
submitting cell routing comparisons. All three scopes remain in the intended
study, but the 36-run grid is not yet ready for launch.

Support-row permutation checks here apply to aggregation with fixed embeddings
and jointly permuted scores/masks. They do not establish whole-model permutation
invariance, nor do these checks verify cell-slot Hungarian alignment, checkpoint
integration, or generalization. Those production checks remain outstanding.


## Completed learning pilots

Implemented both variants in `tfmplayground/models/reconstruction_routing.py` and
ran `tfmplayground.experiments.reconstruction_routing_pilot` on CPU for every scope
with seeds 11/12, first 300 and then 1,500 optimizer updates. Training uses fresh
SCM episodes, support 24/query 32/batch 2, and evaluation uses 32 shared held-out
episodes with 64 queries. A same-width plain TabPFN control was also trained.
This is one overlapping-cue, noisy synthetic prior, not the four-prior sweep.

Primary results and context ablations:
`results/reconstruction_routing_pilot_1500_20260908/report.md`.
The augmented path improved mean accuracy for data and cell_and_data, with only
one successful seed per scope. Cell-only stayed near chance. Slot variance was
nonzero; mask-only query gates nevertheless varied very little in data scopes.
Disabling reconstruction error weighting at inference made negligible differences.
These results do not establish a benefit from the error penalty or robust gains
across seeds. Low MSE was not a sufficient indicator of predictive learning.

Pilot deviations are explicit: cell alignment references support row zero rather
than a medoid; the augmented cell arm jointly adds cell-conditioned aggregation
and value retrieval. The pilot is isolated from TableSlotModel pretraining and
has its own state-dict checkpoints. Forty focused tests pass, including gradient
isolation, alignment, checkpoint restoration, and the earlier reconstruction paths.
The full Slurm experiment grid remains unsubmitted.


## Earlier-model comparison

The missing earlier-model baselines have now been retrained at the same pilot
budget and on the identical episode schedule. See
`results/reconstruction_routing_before_after_20260908.md`.
Original query-only and label-alpha TableSlotModel variants outperform both new
routing variants in mean accuracy in every scope. The original label-alpha
objective includes support-label NLL at weight 1 and slot MI at weight 0.05.
The intermediate embedding-MSE/original-gate variant also loses accuracy versus
the original label objectives. This reverses any interpretation that the new
retrieval path is already an improvement over the previous model: its observed
advantage was only over the new mask-only route. Two small-model seeds still
limit generalization of this finding. Do not promote the modifications on the
basis of these pilots; retain the earlier model as the reference.
