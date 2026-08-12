# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.
It contains a much smaller and simpler implementation of the TabPFNv2 architecture (nanoTabPFN) as well as a training loop, multiple interfaces to load prior data and an evaluation pipeline. We are planning to rapidly extend the repository with more features, prior interfaces and architectures.
It is supposed to be a good starting point for students and researchers that are interested in learning about how Tabular foundation models work under the hood.

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

We offer the same interface as TabPFN:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Our Code

`tfmplayground/models/nanotabpfn.py` contains the implementation of the architecture in less than 300 lines of code. `tfmplayground/train.py` implements a simple training loop in under 200 lines and `tfmplayground/external_priors/` provides an interface to publicly available priors form other repositories as well as a dataloader for loading HDF5 dumps.
We will release multiple dumps of different scales soon. We also offer an interface where you can provide your own get\_batch function.

### Pretrain your own small nanoTabPFN
First we download 100k pre-generated datasets with 50 datapoints, 3 features and up to 3 classes each from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5).

Then you can run:
```
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 --priordump 50x3_3_100k_classification.h5
```
This should take less than 5 min on a modern NVIDIA GPU (around 10 minutes on Macbook M4 Pro GPU and around 40 min on M4 Pro CPU).

We also offer a pre-generated dataset containing 1.28M tables with 50 datapoints and 3 features each for regression [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5).

You can pretrain on it using `python pretrain_regressor.py`.

#### Step by Step Explanation (Classifier)

First we import our Architecture, Prior interface and training loop, etc.
```python
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss
```
then we instantiate our model and loss criterion:
```python
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=10,
)
criterion = CrossEntropyLoss()
```
then we instantiate our prior:
```python
device = get_default_device()
prior = PriorDumpDataLoader(filename='50x3_3_100k_classification.h5', num_steps=25, batch_size=50, device=device)
```
and finally train our model:
```python
trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=80,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)
```

### Creating your own datasets
Check out [tfmplayground.external_priors](https://github.com/automl/TFM-Playground/tree/main/tfmplayground/external_priors) to create your own data using publicly available priors.

You can use `tfmplayground.external_priors` as a command-line-tool to pre-generate data from a prior, e.g. via
```
python -m tfmplayground.external_priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```
which can afterwards be loaded via
```python
from tfmplayground.external_priors import PriorDumpDataLoader
prior = PriorDumpDataLoader('tabicl_4k_50x3.h5', num_steps=20, batch_size=4, device='cpu')
```
You can also just let it create the data on-the-fly via:
```python
from tfmplayground.external_priors import TabICLPriorDataLoader
prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_max=50,
    min_features=3,
    max_features=3,
    max_num_classes=3,
    device='cpu'
)
```
You can check out `next(iter(prior))` if you want to see an example batch.

Check out `prior_visualization.ipynb` for some more examples.

### Supported Priors

- [TabICL](https://github.com/soda-inria/tabicl) (Classification)
- [TICL](https://github.com/microsoft/ticl) (Regression, Classification)

## Hypothesis-collapse experiment

The hypothesis-collapse experiment tests whether nanoTabPFN's individually calibrated predictions correspond to a
coherent joint posterior over several queries. It compares nanoTabPFN with the exact Bayesian predictor and an
independent-marginal oracle on synthetic tasks with two unresolved latent functions.

Install the plotting dependency and run the default experiment:

```bash
pip install -e '.[experiments]'
python -m tfmplayground.experiments.hypothesis_collapse
```

By default, nanoTabPFN uses the official pretrained classifier checkpoint, downloading it into `checkpoints/` if it is
not already cached. A local checkpoint and output directory can be selected explicitly:

```bash
python -m tfmplayground.experiments.hypothesis_collapse \
    --checkpoint /path/to/nanotabpfn_classifier.pth \
    --output-dir runs/hypothesis_collapse/my-run \
    --trials 32 \
    --query-counts 2 3 4 \
    --evidence-counts 0 1 2 4 8
```

The output directory must not already exist. Each run writes:

- `config.json`: complete run configuration plus checkpoint hash and architecture.
- `trial_metrics.csv`: marginal and joint metrics for every task and model.
- `joint_probabilities.csv`: canonical and reverse-order probability for every binary label vector.
- `summary.csv`: grouped means, standard deviations, standard errors, and 95% intervals.

## Prior-generated bimodal latent filter

To train the separate K=2 latent-hypothesis experiment on paired TabICL `mix_scm`
tasks, run:

```bash
PYTHONPATH=. python -m tfmplayground.experiments.train_prior_bimodal_filter \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir runs/prior_bimodal_filter/k2
```

The generator evaluates two independently sampled SCM task networks on the same
feature matrix, keeps pairs with compatible initial support and disagreeing
stream/query labels, and exposes candidate-task metadata only for evaluation.
The model itself receives only the support, stream, and query tensors. Use
`--no-use-diversity` for the particle-diversity ablation.

To use the downloaded HDF5 prior instead:

```bash
PYTHONPATH=. python -m tfmplayground.experiments.train_h5_prior_bimodal_filter \
    --checkpoint checkpoints/nanotabpfn.pth \
    --output-dir runs/h5_prior_bimodal_filter/k2
```

- PNG and PDF figures for metric sweeps and representative ambiguous joint distributions.

Use `--no-plots` to omit figures, or `--models exact independent` for a checkpoint-free baseline run. The main collapse
signature is low marginal error together with high joint divergence or incoherent probability mass. The full
experimental rationale is documented in
[`nanoTabPFN_hypothesis_collapse_diagnostic_plan.md`](nanoTabPFN_hypothesis_collapse_diagnostic_plan.md).

### Larger-scale robustness experiment

An additional entry point tests whether the result persists with more independent trials and larger in-context support
sets. Its defaults are 64 trials, support sizes 16/64/128, query counts 2/4, and evidence counts 0/2/8/16/64/128. The
extended evidence sweep includes a condition whose evidence count matches each common support-set size:

```bash
python -m tfmplayground.experiments.hypothesis_collapse_large_scale
```

This sweep varies support-set scale instead of increasing the query count because exact joint recovery is exponential
in the number of queries. Results include `common_support_size` in every CSV grouping, and multi-scale figures include
the support size in their filenames (for example, `metric_sweep_m4_n128.png`). Any standard experiment option can be
overridden; for a quick checkpoint-free smoke run:

```bash
python -m tfmplayground.experiments.hypothesis_collapse_large_scale \
    --trials 2 --common-support-sizes 16 64 --query-counts 2 \
    --evidence-counts 0 2 --models exact independent --no-plots
```

### Coherent-hypothesis fine-tuning

The coherent-hypothesis research model reuses the official nanoTabPFN backbone and compares the unchanged checkpoint,
a consistency-fine-tuned checkpoint, and a two-slot task-posterior head. The default curriculum alternates controlled
ambiguity tasks with on-the-fly binary TabICL `mix_scm` tasks:

```bash
python -m tfmplayground.experiments.train_coherent_hypotheses \
    --output-dir runs/coherent_hypotheses/my-run
```

Training defaults to 300 consistency steps, 100 slot-head steps with the backbone frozen, and 300 steps with the final
two transformer blocks unfrozen. Use `--stage consistency` or `--stage slots --consistency-checkpoint PATH` to run one
stage. For a controlled-only smoke run without TabICL sampling:

```bash
python -m tfmplayground.experiments.train_coherent_hypotheses \
    --output-dir runs/coherent_hypotheses/smoke \
    --controlled-only --batch-size 1 --accumulate-gradients 1 \
    --consistency-steps 1 --slot-frozen-steps 1 --slot-unfrozen-steps 1 \
    --validation-interval 1 --evaluation-trials 1 --ordinary-evaluation-batches 1
```

Each run records both checkpoints, learning curves, held-out collapse metrics, ordinary-prior balanced accuracy, and an
`acceptance.json` report for the four-query 128-support/128-evidence thresholds.

### Non-leaking hypothesis-weight correction

The correction experiment starts from the consistency checkpoint and estimates its two hypothesis weights from
out-of-fold predictive label likelihoods. A support label is never visible in the context used to score that label.
Training uses only observed support and query rows and labels; generator evidence masks, latent identities, posterior
targets, and the known noise rate are not passed to either correction model.

```bash
python -m tfmplayground.experiments.train_coherent_correction \
    --output-dir runs/coherent_correction/my-run
```

The default trains cross-fitting for 100 frozen-backbone and 300 partially unfrozen steps. It evaluates every
acceptance gate immediately and, on any failure, trains a fresh two-state variational head from the same consistency
checkpoint for 100 frozen plus 500 partially unfrozen steps. Use `--no-fallback-on-failure` to suppress that behavior,
or `--evidence-model variational` to request the fallback directly. Each run keeps separate checkpoints and learning
curves and writes `selection.json` with all gate failures and the selected final model.

## TabArena task-posterior adapter

`NanoTabPFNTaskPosteriorAdapter` is the TabArena-facing particle model. Its four
particles are zero-initialized residual corrections to the pretrained logits,
so a fresh adapter is numerically identical to vanilla nanoTabPFN. The default
`iid_set` mode constructs task slots from the complete labeled context and is
invariant to context ordering. Classification supports 2–10 classes; regression
is out of scope.

The scikit-learn interface uses at most 1,024 stratified context rows and averages
four deterministic context/permutation ensembles:

```python
from tfmplayground import TaskPosteriorClassifier

classifier = TaskPosteriorClassifier(
    model="runs/task_posterior/adapter.pth",
    particle_count=4,
    context_size=1024,
    context_ensembles=4,
    random_state=0,
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
```

Paired-prior training should use
`contrastive_episode_objective` from
`tfmplayground.experiments.train_task_posterior_adapter`; candidate stream and
query labels are matched directly to particle tasks, while ordinary episodes
teach a canonical zero-correction hypothesis. Promotion gates, including the
paired bootstrap against 0.621 and the 0.002 no-harm tolerance, live in
`tfmplayground.experiments.task_posterior_acceptance`.

Official evaluation is delegated to a pinned TabArena checkout instead of the
legacy OpenML loop:

```bash
python -m tfmplayground.experiments.evaluate_task_posterior_tabarena \
    --checkpoint runs/task_posterior/adapter.pth \
    --results-dir runs/tabarena-lite/raw \
    --output-dir runs/tabarena-lite/report
```

The default evaluates canonical split 0 of every compatible classification
dataset using TabArena's task-specific metrics and Elo aggregation. Add `--full`
only after the Lite promotion gate passes. The older
`evaluate_integrated_tabarena` entry point is retained solely as a custom binary
diagnostic and its artifacts are explicitly marked non-official.

## Particle-specific environment benchmark

Particle claims are evaluated separately from IID TabArena. The reusable
protocol in `tfmplayground.experiments.particle_benchmark` generates stable,
A-B, A-B-A, and A-B-C-A binary streams from distinct latent functions evaluated
on the same feature rows. Every method must implement `predict_proba(x)` and
`update(x, y)`: the evaluator commits all predictions for a batch before it
reveals that batch's labels. It reports prequential log loss, AUC, balanced
accuracy, Brier score, calibration error, oracle regret, regime identification,
runtime, memory, recovery delay, and recurrence gain.

`tabpfn_context_baselines` constructs the required cumulative, sliding-window,
exponentially weighted, retrieval, safe single-residual, and oracle contexts
from caller-provided sklearn-compatible TabPFN factories. An adaptive particle
checkpoint can enter the same evaluator through
`AdaptiveParticleOnlineClassifier`. The filter now supports a configurable
Markov `transition_probability`, and both particle decoders support bounded
residual logits. The zero transition and unbounded-residual defaults preserve
older checkpoints.

The companion `tfmplayground.experiments.environment_adaptation` module keeps
official training and future periods separate, tunes only on rolling windows
inside training, and implements frozen few-shot evaluation for m in
`{0, 8, 32, 128}`. Grouped data uses a deterministic seeded within-group order.
These outputs are a derived online/few-shot analysis compatible with loaded
BeyondArena data; they are not the standard BeyondArena leaderboard.

## K=2 batch-causal scratch/warm comparison

Run the paired health pilot first; full training is refused unless both pilot
arms pass their causal, finite-gradient, reload, and held-out-loss checks:

```bash
python -m tfmplayground.experiments.train_particle_regime_comparison \
  --phase pilot --initialization all
python -m tfmplayground.experiments.train_particle_regime_comparison \
  --phase full --initialization all
```

Full runs write three paired seeds below `runs/particle_regime_comparison/full`,
including synthetic metrics, promotion gates, and warm-minus-scratch reports.
BeyondArena evaluation is a separate derived protocol (not the standard
leaderboard) and requires the pinned optional dependency:

```bash
pip install -e '.[beyondarena]'
python -m tfmplayground.experiments.evaluate_particle_regime_beyondarena \
  runs/particle_regime_comparison/full/2402/scratch/selected_checkpoint.pth
```
