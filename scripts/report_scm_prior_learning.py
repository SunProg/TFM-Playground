"""Render measured SCM-prior results and cue controls without retraining."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/scm_prior_learning"
    result = json.loads((output / "results.json").read_text())
    config = result["config"]
    if len(result["neural"]) != 2 * len(config["seeds"]):
        raise ValueError("Training is not complete.")
    controls = [json.loads((root / f"results/scm_prior_learning_cue{d}/results.json").read_text()) for d in (0, 1)]
    controls.append(result)
    latent = [r for r in result["neural"] if not r["oracle"]]
    scores = [100 * r["test"]["accuracy"]["mean"] for r in latent]
    shuffled = [100 * r["test_shuffled_support_labels"]["accuracy"]["mean"] for r in latent]
    metrics = result["classical"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for oracle, color, label in [(False, "#2563eb", "Latent nanoTabPFN"), (True, "#d97706", "Oracle nanoTabPFN")]:
        for i, run in enumerate(r for r in result["neural"] if r["oracle"] == oracle):
            axes[0].plot(
                [h["step"] for h in run["history"]],
                [100 * h["accuracy"]["mean"] for h in run["history"]],
                color=color,
                alpha=0.7,
                label=label if i == 0 else None,
            )
    axes[0].set(xlabel="Training steps", ylabel="Validation accuracy (%)", title="Learning fresh SCM mechanisms")
    for key, label in [
        ("clustered_rbf", "Clustered RBF classifiers"),
        ("oracle_rbf", "Oracle RBF classifiers"),
        ("known_mechanisms_bayes", "Known-mechanism Bayes"),
    ]:
        axes[1].plot(
            [0, 1, 3], [100 * c["classical"]["metrics"][key]["mean"] for c in controls], marker="o", label=label
        )
    axes[1].set(xlabel="Regime cue separation", ylabel="Test accuracy (%)", title="Controlled routing difficulty")
    for ax in axes:
        ax.set_ylim(45, 101)
        ax.axhline(50, color="gray", linestyle=":", linewidth=1)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Learnable multiregime prior based on TabICL SCM",
        "",
        "Experiment date: 2026-09-07.",
        "",
        f"Small nanoTabPFN models trained from scratch reach **{np.mean(scores):.2f}% mean held-out accuracy** "
        f"across {len(scores)} seeds (range {min(scores):.2f}–{max(scores):.2f}%). "
        f"Shuffling support labels reduces the mean to **{np.mean(shuffled):.2f}%**. "
        f"Clustering plus nonlinear classifiers reaches {100 * metrics['clustered_rbf']['mean']:.2f}%, "
        f"versus {100 * metrics['oracle_rbf']['mean']:.2f}% with observed regime IDs.",
        "",
        "## SCM construction",
        "",
        "The sampler uses the installed TabICL `MLPSCM` and `XSampler` implementations. "
        "It uses predictive mode (`is_causal=False`): the observed task features are Gaussian root causes, "
        "and two independently initialized nonlinear MLP mechanisms map those same causes to scores. "
        "It is a constrained subset of the SCM family; the unrestricted TabICL hyperparameter distribution "
        "and hidden-node feature selection are not sampled.",
        "",
        "For each fresh episode:",
        "",
        "1. Draw two TabICL MLP mechanisms with `num_layers=2`, `hidden_dim=8`, tanh activations, "
        "initialization standard deviation 0.5, zero weight dropout and zero mechanism noise. "
        "TabICL appends an output block in predictive mode, so this means an input linear layer, "
        "one hidden tanh/linear block and one tanh/output block.",
        "2. Draw 512 independent calibration cause vectors. Threshold each mechanism at its calibration median. "
        "Accept a pair only if its calibration labels disagree on 25–75% of rows; reject degenerate outputs. "
        "This explicitly conditions the mechanism prior. Neither support nor query rows enter selection.",
        "3. Draw fresh support/query causes u ~ N(0,I₂), and independent regime IDs z ~ Bernoulli(0.5). "
        "Append a cue c = d(2z−1) + ε, ε ~ N(0,1), with default d=3.",
        "4. Set y = 1[f_z(u) > t_z]. The observed input is (c,u); latent learners receive only "
        "support features, support labels and query features.",
        "",
        "The generator returns the existing `RegimeEpisode` type, including separate diagnostic regime IDs, "
        "counterfactual probabilities and exact posterior gate probabilities. "
        "Only two regimes are active; `max_regimes` controls padding, not the number of mechanisms. "
        "Calling `latent_inputs()` excludes those diagnostics. `oracle_inputs()` appends the regime one-hot. "
        "The sampler restores the caller's CPU Torch RNG state and records configuration, mechanism seeds, "
        "hashes, thresholds and rejection attempts.",
        "",
        "## Results",
        "",
        f"Evaluation uses {config['test_episodes']} fresh episodes, each with 128 support and 256 query rows "
        f"({256 * config['test_episodes']:,} query predictions per model). "
        "The neural models use the existing nanoTabPFN architecture with embedding size 32, 4 attention heads, "
        "2 transformer blocks and MLP width 64. Each arm trains for 1,000 steps on batches of 8 fresh episodes "
        "with 32 queries each. AdamW: learning rate 0.001, weight decay 0.01, gradient clipping 1. "
        "Select checkpoints by CE on 32 separate validation episodes every 200 steps. "
        "Training seeds 11, 12 and 13 have paired episode streams between latent and oracle arms. "
        "A seed audit found 24,000 distinct training episode seeds and no train/validation/test seed overlap.",
        "",
        "| Neural arm | Seed | Selected step | Test accuracy | Test CE | Shuffled labels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["neural"]:
        lines.append(
            f"| {'Oracle' if r['oracle'] else 'Latent'} | {r['seed']} | {r['selected_step']} | "
            f"{100 * r['test']['accuracy']['mean']:.2f}% | {r['test']['cross_entropy']['mean']:.4f} | "
            f"{100 * r['test_shuffled_support_labels']['accuracy']['mean']:.2f}% |"
        )
    for run in result["neural"]:
        if run["test"]["accuracy"]["mean"] < 0.6:
            lines += [
                "",
                f"{'Oracle' if run['oracle'] else 'Latent'} seed {run['seed']} stalled near chance "
                f"({100 * run['test']['accuracy']['mean']:.2f}%) within the fixed training budget. "
                "This failed run is retained. No optimization sweep or longer-budget rescue was performed.",
            ]
    lines += ["", "| Classical comparator | Test accuracy | Approximate episode-level 95% CI |", "|---|---:|---:|"]
    for key, label in [
        ("clustered_rbf", "K-means + per-cluster RBF SVM"),
        ("oracle_rbf", "True z + per-regime RBF SVM"),
        ("pooled_rbf", "Pooled RBF SVM"),
        ("shuffled_clustered_rbf", "Clustered, shuffled labels"),
        ("known_mechanisms_bayes", "Known-mechanism Bayes, hidden z"),
    ]:
        metric = metrics[key]
        lines.append(
            f"| {label} | {100 * metric['mean']:.2f}% | {100 * metric['ci95'][0]:.2f}–{100 * metric['ci95'][1]:.2f}% |"
        )
    lines += [
        "",
        "K-means fits only support features. Per-cluster RBF classifiers use the task features, "
        "with support-only scaling and fixed C=10 and gamma='scale'; the pooled RBF classifier uses "
        "all features. Classical learners know the cue's feature index. "
        "Known-mechanism Bayes has privileged access to the two generating functions and thresholds, "
        "but marginalizes query regime using P(z|c). It is a ceiling comparator, not a fitted learner.",
        "",
        "## Cue separation control",
        "",
        "Only cue separation changes: task mechanisms, thresholds, root causes, regimes and labels "
        "are identical across these paired test conditions. The oracle scores therefore stay fixed. "
        "These controls fit classical learners; neural models were trained only at separation 3.",
        "",
        "| Cue separation | Clustered RBF | Oracle RBF | Known-mechanism Bayes |",
        "|---:|---:|---:|---:|",
    ]
    for c in controls:
        m = c["classical"]["metrics"]
        lines.append(
            f"| {c['config']['prior']['cue_separation']:g} | {100 * m['clustered_rbf']['mean']:.2f}% | "
            f"{100 * m['oracle_rbf']['mean']:.2f}% | {100 * m['known_mechanisms_bayes']['mean']:.2f}% |"
        )
    lines += [
        "",
        "At zero separation, regime is independent of the observed features. "
        "When the two deterministic mechanisms disagree, even a learner knowing both functions cannot "
        "resolve the sampled regime. The roughly 75% ceiling follows from disagreement on roughly half "
        "the rows. Increasing separation supplies the missing information.",
        "",
        "## Use and scope",
        "",
        "```python",
        "from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode",
        "",
        "config = SCMRegimeConfig(cue_separation=3.0, task_features=2)",
        "episode = sample_scm_regime_episode(config, seed=42)",
        "support_x, support_y, query_x = episode.latent_inputs()",
        "# logits = model(support_x, support_y, query_x)",
        "```",
        "",
        "Start with this measured default. `cue_separation` controls routing difficulty; "
        "`task_features`, `num_layers` and `hidden_dim` control mechanism complexity. "
        "`label_noise` and `regime_probability` control label flips and regime balance. "
        "Calibration disagreement bounds condition how different the mechanisms are. "
        "Other settings require their own learning checks. This does not establish learning by the "
        "slot architecture, recovery of causal graphs, or transfer to real tables. "
        "A trained oracle neural model is not a mathematical upper bound: finite optimization can fail "
        "despite the additional inputs. All runs, including weak ones, are retained. "
        "CIs treat episodes as units; the three neural seeds share test episodes.",
        "",
        "## Reproduce",
        "",
        "```sh",
        ".venv/bin/python -m tfmplayground.experiments.scm_prior_learning --output results/scm_prior_learning",
        ".venv/bin/python -m tfmplayground.experiments.scm_prior_learning "
        "--output results/scm_prior_learning_cue0 --cue-separation 0 --classical-only",
        ".venv/bin/python -m tfmplayground.experiments.scm_prior_learning "
        "--output results/scm_prior_learning_cue1 --cue-separation 1 --classical-only",
        ".venv/bin/python scripts/report_scm_prior_learning.py",
        ".venv/bin/python -m unittest tests.test_scm_regime_prior tests.test_clustered_prior_learning",
        "```",
        "",
        "The experiment requires fresh output directories to prevent overwriting results. "
        "Metrics, validation histories and selected neural checkpoints are under "
        "`results/scm_prior_learning/`; the controls are in their corresponding directories.",
        "",
        "![SCM learning and cue controls](../results/scm_prior_learning/learning_curves.png)",
        "",
    ]
    report = root / "paper/scm_prior_learning_results.md"
    report.write_text("\n".join(lines))
    print(report)


if __name__ == "__main__":
    main()
