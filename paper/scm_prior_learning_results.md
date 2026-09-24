# Learnable multiregime prior based on TabICL SCM

Experiment date: 2026-09-07.

Small nanoTabPFN models trained from scratch reach **91.67% mean held-out accuracy** across 3 seeds (range 90.91–92.56%). Shuffling support labels reduces the mean to **50.56%**. Clustering plus nonlinear classifiers reaches 94.74%, versus 94.83% with observed regime IDs.

## SCM construction

The sampler uses the installed TabICL `MLPSCM` and `XSampler` implementations. It uses predictive mode (`is_causal=False`): the observed task features are Gaussian root causes, and two independently initialized nonlinear MLP mechanisms map those same causes to scores. It is a constrained subset of the SCM family; the unrestricted TabICL hyperparameter distribution and hidden-node feature selection are not sampled.

For each fresh episode:

1. Draw two TabICL MLP mechanisms with `num_layers=2`, `hidden_dim=8`, tanh activations, initialization standard deviation 0.5, zero weight dropout and zero mechanism noise. TabICL appends an output block in predictive mode, so this means an input linear layer, one hidden tanh/linear block and one tanh/output block.
2. Draw 512 independent calibration cause vectors. Threshold each mechanism at its calibration median. Accept a pair only if its calibration labels disagree on 25–75% of rows; reject degenerate outputs. This explicitly conditions the mechanism prior. Neither support nor query rows enter selection.
3. Draw fresh support/query causes u ~ N(0,I₂), and independent regime IDs z ~ Bernoulli(0.5). Append a cue c = d(2z−1) + ε, ε ~ N(0,1), with default d=3.
4. Set y = 1[f_z(u) > t_z]. The observed input is (c,u); latent learners receive only support features, support labels and query features.

The generator returns the existing `RegimeEpisode` type, including separate diagnostic regime IDs, counterfactual probabilities and exact posterior gate probabilities. Only two regimes are active; `max_regimes` controls padding, not the number of mechanisms. Calling `latent_inputs()` excludes those diagnostics. `oracle_inputs()` appends the regime one-hot. The sampler restores the caller's CPU Torch RNG state and records configuration, mechanism seeds, hashes, thresholds and rejection attempts.

## Results

Evaluation uses 256 fresh episodes, each with 128 support and 256 query rows (65,536 query predictions per model). The neural models use the existing nanoTabPFN architecture with embedding size 32, 4 attention heads, 2 transformer blocks and MLP width 64. Each arm trains for 1,000 steps on batches of 8 fresh episodes with 32 queries each. AdamW: learning rate 0.001, weight decay 0.01, gradient clipping 1. Select checkpoints by CE on 32 separate validation episodes every 200 steps. Training seeds 11, 12 and 13 have paired episode streams between latent and oracle arms. A seed audit found 24,000 distinct training episode seeds and no train/validation/test seed overlap.

| Neural arm | Seed | Selected step | Test accuracy | Test CE | Shuffled labels |
|---|---:|---:|---:|---:|---:|
| Latent | 11 | 1000 | 90.91% | 0.2166 | 50.58% |
| Oracle | 11 | 1000 | 89.08% | 0.2571 | 51.40% |
| Latent | 12 | 1000 | 91.55% | 0.2043 | 50.77% |
| Oracle | 12 | 600 | 49.75% | 0.6932 | 49.75% |
| Latent | 13 | 1000 | 92.56% | 0.1775 | 50.32% |
| Oracle | 13 | 1000 | 74.88% | 0.5028 | 49.60% |

Oracle seed 12 stalled near chance (49.75%) within the fixed training budget. This failed run is retained. No optimization sweep or longer-budget rescue was performed.

| Classical comparator | Test accuracy | Approximate episode-level 95% CI |
|---|---:|---:|
| K-means + per-cluster RBF SVM | 94.74% | 94.50–94.98% |
| True z + per-regime RBF SVM | 94.83% | 94.59–95.06% |
| Pooled RBF SVM | 93.63% | 93.36–93.90% |
| Clustered, shuffled labels | 50.29% | 49.09–51.49% |
| Known-mechanism Bayes, hidden z | 99.92% | 99.90–99.94% |

K-means fits only support features. Per-cluster RBF classifiers use the task features, with support-only scaling and fixed C=10 and gamma='scale'; the pooled RBF classifier uses all features. Classical learners know the cue's feature index. Known-mechanism Bayes has privileged access to the two generating functions and thresholds, but marginalizes query regime using P(z|c). It is a ceiling comparator, not a fitted learner.

## Cue separation control

Only cue separation changes: task mechanisms, thresholds, root causes, regimes and labels are identical across these paired test conditions. The oracle scores therefore stay fixed. These controls fit classical learners; neural models were trained only at separation 3.

| Cue separation | Clustered RBF | Oracle RBF | Known-mechanism Bayes |
|---:|---:|---:|---:|
| 0 | 73.08% | 94.83% | 75.00% |
| 1 | 84.77% | 94.83% | 92.05% |
| 3 | 94.74% | 94.83% | 99.92% |

At zero separation, regime is independent of the observed features. When the two deterministic mechanisms disagree, even a learner knowing both functions cannot resolve the sampled regime. The roughly 75% ceiling follows from disagreement on roughly half the rows. Increasing separation supplies the missing information.

## Use and scope

```python
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode

config = SCMRegimeConfig(cue_separation=3.0, task_features=2)
episode = sample_scm_regime_episode(config, seed=42)
support_x, support_y, query_x = episode.latent_inputs()
# logits = model(support_x, support_y, query_x)
```

Start with this measured default. `cue_separation` controls routing difficulty; `task_features`, `num_layers` and `hidden_dim` control mechanism complexity. `label_noise` and `regime_probability` control label flips and regime balance. Calibration disagreement bounds condition how different the mechanisms are. Other settings require their own learning checks. This does not establish learning by the slot architecture, recovery of causal graphs, or transfer to real tables. A trained oracle neural model is not a mathematical upper bound: finite optimization can fail despite the additional inputs. All runs, including weak ones, are retained. CIs treat episodes as units; the three neural seeds share test episodes.

## Reproduce

```sh
.venv/bin/python -m tfmplayground.experiments.scm_prior_learning --output results/scm_prior_learning
.venv/bin/python -m tfmplayground.experiments.scm_prior_learning --output results/scm_prior_learning_cue0 --cue-separation 0 --classical-only
.venv/bin/python -m tfmplayground.experiments.scm_prior_learning --output results/scm_prior_learning_cue1 --cue-separation 1 --classical-only
.venv/bin/python scripts/report_scm_prior_learning.py
.venv/bin/python -m unittest tests.test_scm_regime_prior tests.test_clustered_prior_learning
```

The experiment requires fresh output directories to prevent overwriting results. Metrics, validation histories and selected neural checkpoints are under `results/scm_prior_learning/`; the controls are in their corresponding directories.

![SCM learning and cue controls](../results/scm_prior_learning/learning_curves.png)
