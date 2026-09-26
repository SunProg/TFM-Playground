# Regime-information diagnostic pilot

This is a frozen-checkpoint diagnostic, not a main-paper result yet. It tests
whether access to a query row's realized regime membership changes prediction
on the native synthetic bank. It does not retrain a model, and it does not
estimate uncertainty across pretraining seeds.

## Compatibility audit and canonical controls (24 September 2026)

The six-layer diagnostic array `37500513` completed, but its reports under
`artifacts/evaluations/regime_information_native/` are **not valid checkpoint
comparisons**. The native checkpoints were trained using query attention over
the input support representations; the diagnostic checkout instead uses the
support self-attention output (`x_left`) as keys and values. Loading the same
state dictionary does not preserve the model's predictions across this change.
For `original-large`, episode 2504 has saved CE **0.81174135**; the original
implementation reproduces **0.81174147**, while the diagnostic implementation
gives **2.83717203**. Audit the earlier pilot's model-source compatibility before
reusing its results below.

Array **37503813** evaluates frozen single-regime final checkpoints (seed 2402,
10,000 steps) on A30: `_0` = two layers, `_1` = four layers, `_2` = six layers.
Each task first reproduces the training monitor's exact probe (seed 2402,
episode 0 in a deterministic one-in-16 sample of cells), then evaluates all
5,760 multiclass multiregime episodes (tag seed 20260924). Both stages compare
hidden, shuffled, and true tags. The evaluator loads the checkpoint's original
model source, records its SHA-256, and checks hidden-condition CE and accuracy
against the saved native evaluation. Reports are written separately under
`artifacts/evaluations/regime_information_checkpoint_compatible/original-<size>/seed-2402/`.

The ongoing observed-regime runs use the newer attention calculation. Thus the
new canonical controls provide valid reference predictions, but comparing them
with the observed-regime runs does **not** isolate regime-ID training: the
attention calculation also differs. A matched untagged training control, or
observed-regime training with the original attention calculation, is needed for
that causal attribution. Early probe checkpoints also have a smaller training
budget than these 10,000-step references.

## Setup

- **Checkpoint:** local copy of the native-prior `rg_z` curriculum,
  multiclass-only, four-layer NanoTabPFN (`e128-l4-h4-m512`), seed 2402,
  step 9,950. This is a supporting ablation checkpoint, not the primary
  six-layer result reported in the abstract.
- **Episodes:** a deterministic one-in-eight hash sample of factorial cells
  from the score-hidden native test bank. The retained cells contain 648
  multiclass, K>=2, `rule_mode=multiregime` episodes: 304 `soft_gate` and
  344 `persistent`.
- **Information conditions:**
  - `hidden`: original feature matrix;
  - `shuffled`: append a regime-tag column but shuffle tag values independently
    within the support and query partitions; and
  - `true`: append the realized regime tag.

  Tag values are randomly renamed independently per episode in both tag
  conditions. Thus a tag's numeric identity cannot carry a global convention;
  only support/query correspondence can convey membership. The true tag is
  privileged diagnostic information, not a deployable input.
- **Intervals:** 5,000 cell-stratified episode bootstraps. They quantify
  finite-episode variation in this sample only.

Source output: [cell-sampled JSON](../../artifacts/local_mps/regime_information/curriculum_multiclass_only_medium_cell_sample8.json).

## Results

| Routing | Comparison | CE difference (candidate − reference) | 95% interval | Accuracy difference | 95% interval |
|---|---|---:|---:|---:|---:|
| Persistent | shuffled − hidden | −0.0314 | [−0.0353, −0.0277] | −0.03 pp | [−0.07, +0.01] pp |
| Persistent | true − hidden | −0.0577 | [−0.0700, −0.0458] | +0.01 pp | [−0.04, +0.05] pp |
| Persistent | **true − shuffled** | **−0.0263** | **[−0.0376, −0.0148]** | **+0.03 pp** | **[−0.03, +0.10] pp** |
| Soft-gate | shuffled − hidden | −0.0259 | [−0.0306, −0.0213] | +0.05 pp | [−0.02, +0.14] pp |
| Soft-gate | true − hidden | −0.0359 | [−0.0475, −0.0250] | +0.01 pp | [−0.08, +0.11] pp |
| Soft-gate | true − shuffled | −0.0100 | [−0.0214, +0.0006] | −0.04 pp | [−0.10, +0.01] pp |

The shuffled-tag condition itself improves cross-entropy, so hidden-versus-true
does not isolate regime information: it also changes the input representation.
The appropriate contrast is **true minus shuffled**. On this supporting model,
the additional value of true membership is clear for `persistent` routing and
smaller, not resolved from zero by this sample, for `soft_gate` routing. This
is consistent with the hypothesis that unobserved membership is a bottleneck
for persistent tasks.

The result needs the same diagnostic run for the three primary six-layer
checkpoints before it can appear in the paper. That A30 recipe is
[`evaluate_v4_regime_information_a30.sbatch`](../../scripts/slurm/evaluate_v4_regime_information_a30.sbatch).
