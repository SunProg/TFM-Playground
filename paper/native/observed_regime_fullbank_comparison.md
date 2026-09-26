# Observed-regime checkpoints: full-bank comparison

This is a descriptive, single-seed frozen-checkpoint comparison for the
observed-regime diagnostic. It is not a new training-seed experiment and does
not establish a causal effect of adding regime IDs.

## Evaluation

Compare `original`, `rg_z-fixed`, and `rg_z-curriculum` at the same size
(small: 2 layers; medium: 4; large: 6), seed 2402, and checkpoint step
10,000. Each report evaluates the same 5,760 multiclass multiregime episodes
from the existing `test.h5` bank (2,880 `persistent`, 2,880 `soft_gate`).
The bank is evaluation data used by this diagnostic, not an untouched final
test set. For each episode, the evaluator scores hidden, shuffled-tag, and
true-tag inputs. Reported differences below are paired **candidate minus
original**; negative CE and positive accuracy differences favor the candidate.
Accuracy differences are percentage points (pp).

The canonical original reports load the original model implementation. The
fixed and curriculum checkpoints were trained/evaluated with a different
attention calculation (support self-attention outputs rather than the
original query attention over input support representations). The comparison
therefore has an implementation confound and cannot attribute differences
solely to regime-ID training or curriculum. It also uses one pretraining seed;
the numbers describe these checkpoints and episodes, not seed-level
uncertainty. Figure A1 adds paired, cell-stratified episode-bootstrap
intervals for candidate-minus-original differences. Those intervals quantify
variation across this finite evaluation bank only; they do not estimate
pretraining-seed uncertainty or support seed-general significance claims.

The recorded model-source SHA-256 is
`5ed1a46b2939aa0c8db1e3766a27a9cae9db6e36ac0a8b165471df7bcc14339c` for the
original reports and
`32662bf83a70f72c09067c8c0e2a861232168f6dcec92e56157cd41ce94c7e34` for the
fixed/curriculum reports.

## Paired differences from original

Each cell is `CE difference / accuracy difference`. H = hidden membership,
S = shuffled membership tags, T = true membership tags.

### Persistent routing

| Size | Variant | H (CE / acc pp) | S (CE / acc pp) | T (CE / acc pp) |
|---|---|---:|---:|---:|
| Small | Fixed | −0.00073 / +0.03 | −0.00076 / +0.06 | −0.00210 / +0.15 |
| Small | Curriculum | −0.00049 / −0.01 | −0.00041 / −0.01 | −0.00077 / ≈0.00 |
| Medium | Fixed | −0.00463 / +0.03 | −0.00418 / +0.06 | −0.01091 / +0.41 |
| Medium | Curriculum | −0.00642 / +0.10 | −0.00655 / +0.14 | −0.01922 / +0.80 |
| Large | Fixed | −0.00136 / +0.07 | −0.00130 / +0.10 | −0.01110 / +0.50 |
| Large | Curriculum | −0.00079 / −0.07 | −0.00075 / −0.02 | −0.01252 / +0.45 |

### Soft-gate routing

| Size | Variant | H (CE / acc pp) | S (CE / acc pp) | T (CE / acc pp) |
|---|---|---:|---:|---:|
| Small | Fixed | −0.00199 / +0.13 | −0.00215 / +0.15 | −0.00183 / +0.01 |
| Small | Curriculum | −0.00143 / +0.13 | −0.00147 / +0.12 | −0.00065 / −0.04 |
| Medium | Fixed | −0.01370 / +0.61 | −0.01452 / +0.65 | −0.00979 / +0.29 |
| Medium | Curriculum | −0.02226 / +1.04 | −0.02387 / +1.15 | −0.01662 / +0.50 |
| Large | Fixed | −0.01387 / +0.67 | −0.01556 / +0.75 | −0.01165 / +0.42 |
| Large | Curriculum | −0.01680 / +0.84 | −0.01878 / +0.94 | −0.01297 / +0.41 |

## Interpretation for the outline

Across these checkpoints, persistent-routing variants are close to the
original under hidden and shuffled membership, while their CE advantage is
larger under true membership at medium and large sizes. For soft-gate,
fixed and curriculum variants have a larger advantage over the original under
hidden or shuffled membership than under true membership at medium and large
sizes; the small-model differences are much smaller. This routing-dependent
pattern is **consistent with**, but does not demonstrate, the hypothesis that
soft-gate models can infer useful routing information from `X` when membership
is hidden, with less need for that inference when the true tag is supplied.
The evaluation does not directly measure regime recovery, and the code-path
confound prevents attributing the pattern specifically to regime-ID training.

The appropriate main-text wording is therefore descriptive: “On the shared
evaluation bank, the relative performance pattern differed by task family:
persistent-routing models showed a larger advantage with true regime
membership, whereas soft-gate models showed a larger advantage when membership
was hidden or shuffled. This pattern is consistent with, but does not establish,
input-based regime inference in soft-gate tasks.”

## Report locations

The underlying full-bank reports are
`artifacts/evaluations/regime_information_checkpoint_compatible/{original,rg_z-fixed,rg_z-curriculum}-{small,medium,large}/seed-2402/test.json`.
Compatibility details and the separate 42-episode checkpoint probe are in
[`regime_information_diagnostic.md`](regime_information_diagnostic.md). The
earlier 42-episode probe is monitoring/validation data; it is not a substitute
for this full-bank comparison.
