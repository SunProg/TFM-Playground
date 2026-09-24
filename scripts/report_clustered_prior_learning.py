"""Render the saved clustered-prior experiment without rerunning training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/clustered_prior_learning/results.json"))
    parser.add_argument("--report", type=Path, default=Path("paper/clustered_prior_learning_results.md"))
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    config = results["config"]
    expected_runs = 2 * len(config["seeds"])
    if len(results["neural"]) != expected_runs:
        raise ValueError(f"Expected {expected_runs} completed runs; training is not complete.")
    classical = results["classical"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    counts = [int(n) for n in classical]
    for key, label in [
        ("clustered", "K-means + two classifiers"),
        ("oracle", "Oracle + two classifiers"),
        ("pooled", "One pooled classifier"),
        ("clustered_shuffled_labels", "Shuffled support labels"),
    ]:
        means = np.array([classical[str(n)][key]["mean"] for n in counts])
        ci = np.array([classical[str(n)][key]["ci95"] for n in counts])
        axes[0].plot(counts, 100 * means, marker="o", label=label)
        axes[0].fill_between(counts, 100 * ci[:, 0], 100 * ci[:, 1], alpha=0.12)
    axes[0].set(xlabel="Support rows", ylabel="Held-out query accuracy (%)", title="The prior is learnable")
    axes[0].legend(fontsize=8, loc="center right", bbox_to_anchor=(0.98, 0.36))
    for oracle, color, label in [(False, "#2563eb", "Latent nanoTabPFN"), (True, "#d97706", "Oracle nanoTabPFN")]:
        runs = [run for run in results["neural"] if run["oracle"] == oracle]
        for i, run in enumerate(runs):
            axes[1].plot(
                [h["step"] for h in run["history"]],
                [100 * h["accuracy"]["mean"] for h in run["history"]],
                color=color,
                alpha=0.7,
                label=label if i == 0 else None,
            )
    axes[1].set(
        xlabel="Training steps",
        ylabel="Validation query accuracy (%)",
        title=f"Fresh episodes; {len(config['seeds'])} training seeds",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(50, color="gray", linestyle=":", linewidth=1)
        ax.set_ylim(40, 101)
        ax.grid(alpha=0.15)
    figure = args.results.parent / "learning_curves.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    latent = [run for run in results["neural"] if not run["oracle"]]
    oracle = [run for run in results["neural"] if run["oracle"]]
    latent_scores = [100 * run["test"]["accuracy"]["mean"] for run in latent]
    oracle_scores = [100 * run["test"]["accuracy"]["mean"] for run in oracle]
    shuffled_scores = [100 * run["test_shuffled_support_labels"]["accuracy"]["mean"] for run in latent]
    lines = [
        "# Easy clustered multiregime prior: empirical learning report",
        "",
        "Date: 2026-09-07.",
        "",
        f"The easy prior is empirically learnable. The existing small nanoTabPFN, without regime IDs, "
        f"achieves **{np.mean(latent_scores):.2f}% mean held-out accuracy** across {len(latent)} training seeds "
        f"(range {min(latent_scores):.2f}–{max(latent_scores):.2f}%). "
        f"Shuffling its support labels reduces mean accuracy to **{np.mean(shuffled_scores):.2f}%**. "
        f"The oracle transformer averages {np.mean(oracle_scores):.2f}%. "
        f"A simple clustering-plus-classification learner reaches {100 * classical['128']['clustered']['mean']:.2f}%.",
        "",
        "## Setup",
        "",
        "Two regimes sampled independently with probability 1/2. Three features: "
        "a regime cue x₁ = 3(2z−1) + ε with ε ~ N(0,1), and two independent standard-normal task features. "
        "Each episode draws a fresh random linear rule and an orthogonal second rule, with random handedness. "
        "Labels are deterministic within each regime; there is no added label noise. "
        "Regimes are balanced in expectation, not forced to have equal counts.",
        "",
        f"All final comparisons use the same {config['test_episodes']} unseen episodes with 256 queries each "
        f"({config['test_episodes'] * 256:,} queries). "
        "Classical support curves use nested prefixes of a common 128-row support set. K-means fits only support "
        "features; its cluster IDs route support and queries to separate logistic regressions "
        "on the two task features. "
        "This baseline knows the feature roles, but never receives regime IDs or rule coefficients. "
        "The oracle uses true support/query regime IDs. The pooled comparator is a single linear logistic classifier "
        "on all three features, so it cannot represent general regime-by-feature interactions.",
        "",
        "## Classical learning curve",
        "",
        "| Support rows | Clustered | Oracle | Pooled linear | Shuffled labels |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, row in classical.items():
        values = [
            f"{100 * row[key]['mean']:.2f}%" for key in ("clustered", "oracle", "pooled", "clustered_shuffled_labels")
        ]
        lines.append(f"| {n} | " + " | ".join(values) + " |")
    row = classical["128"]
    lo, hi = [100 * v for v in row["clustered"]["ci95"]]
    lines += [
        "",
        f"At 128 support rows, clustered accuracy has an approximate episode-level 95% CI of "
        f"{lo:.2f}–{hi:.2f}%. Query routing accuracy, after permutation alignment, is "
        f"{100 * row['routing_accuracy']['mean']:.3f}%. "
        f"The oracle advantage is {100 * (row['oracle']['mean'] - row['clustered']['mean']):.3f} percentage points.",
        "",
        "## Neural learning",
        "",
        "Existing NanoTabPFNModel trained from scratch on CPU: 2 layers, embedding size 32, 4 heads, "
        "MLP width 64; AdamW learning rate 0.001, weight decay 0.01, gradient clipping 1.0. "
        f"Each run has {config['steps']:,} updates of 8 fresh episodes, 128 support rows and 32 training queries. "
        "Latent inputs contain only features and support labels; the oracle appends two one-hot regime features. "
        f"Each arm uses training seeds {config['seeds']}, with matched episode streams. "
        f"Checkpoint selection minimizes CE on {config['validation_episodes']} fixed validation episodes "
        "every 200 steps and at the final step. "
        "The final test namespace is separate from training and validation.",
        "",
        "| Arm | Seed | Selected step | Test accuracy | Test CE | Shuffled-label accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in results["neural"]:
        lines.append(
            f"| {'Oracle' if run['oracle'] else 'Latent'} | {run['seed']} | {run['selected_step']} | "
            f"{100 * run['test']['accuracy']['mean']:.2f}% | "
            f"{run['test']['cross_entropy']['mean']:.4f} | "
            f"{100 * run['test_shuffled_support_labels']['accuracy']['mean']:.2f}% |"
        )
    stalled = [run["seed"] for run in oracle if run["test"]["accuracy"]["mean"] < 0.6]
    if stalled:
        lines += [
            "",
            f"Oracle seeds {stalled} remain below 60% test accuracy within this budget. "
            "These failed runs are retained; the neural oracle is not a consistently optimized reference. "
            "The successful latent runs establish feasibility, not universal training stability.",
        ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The classical curve establishes that unseen episode-specific rules can be learned with little routing "
        "loss on this prior. Neural validation curves and independent test scores measure whether the existing "
        "small transformer also acquires this ability. Shuffling support labels is a negative control for "
        "dependence on the support set, not a separate training arm.",
        "",
        "Giving a finitely trained neural model extra oracle inputs does not guarantee a higher score: "
        "its optimization and input representation also change. The paired classical oracle provides the clearer "
        "routing comparison. Use this prior as a positive control before introducing overlap, irrelevant features, "
        "or more regimes; testing the slot architecture on it is a separate next experiment.",
        "",
        "This is an empirical feasibility experiment, not a proof of slot recovery or real-table transfer. "
        "No slot-attention model was trained here. A pooled linear model is only a limited comparator; "
        "its failure does not establish an advantage over other nonlinear models. "
        "The feature roles, strong cluster cue, two-dimensional linear rules and zero label noise are "
        "deliberately favorable. Confidence intervals in JSON use episode means (normal approximation), "
        "not individual queries; neural runs share test episodes and must not be counted as independent "
        "test datasets. Three training seeds are a small stability check.",
        "",
        "The regime cue is not perfectly deterministic: even with known generating rules, "
        "the latent Bayes error is 0.5 Φ(−3) ≈ 0.0675%, since orthogonal rules disagree on half the task inputs. "
        "The fitted oracle is not the Bayes oracle.",
        "",
        "## Reproduce",
        "",
        "```sh",
        ".venv/bin/python -m tfmplayground.experiments.clustered_prior_learning "
        "--output results/clustered_prior_learning --steps 1000 --seeds 11 12 13",
        ".venv/bin/python scripts/report_clustered_prior_learning.py",
        ".venv/bin/python -m unittest tests.test_clustered_prior_learning",
        "```",
        "",
        "Raw metrics and validation histories: `results/clustered_prior_learning/results.json`. "
        "Per-run selected checkpoints and metrics are saved beside it.",
        "",
        "![Learning curves](../results/clustered_prior_learning/learning_curves.png)",
        "",
    ]
    args.report.write_text("\n".join(lines))
    print(args.report.resolve())
    print(figure.resolve())


if __name__ == "__main__":
    main()
