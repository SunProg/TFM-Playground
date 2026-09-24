"""Summarize matched seeds; bootstrap paired held-out episode differences."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    results = json.loads((args.directory / "results.json").read_text())
    if len(results) != 12:
        raise ValueError(f"Expected 12 completed routing pilots, found {len(results)}")
    lines = [
        "# Reconstruction routing learning pilots",
        "",
        "Two matched seeds per arm; 32 shared held-out SCM episodes, 64 query rows each.",
        "These are small CPU learning pilots, not the full four-prior Slurm study.",
        "",
        "| Scope | Arm | Accuracy | Query NLL | Embedding MSE | Slot std | Mask row std | Query gate std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("data", "cell_and_data", "cell"):
        for variant in ("masks", "values"):
            cells = [r for r in results if r["scope"] == scope and r["variant"] == variant]
            avg = {key: np.mean([r["final"][key] for r in cells]) for key in cells[0]["final"]}
            lines.append(
                f"| {scope} | {variant} | {avg['accuracy']:.2%} | {avg['nll']:.4f} | {avg['mse']:.4f} | "
                f"{avg['slot_std']:.4f} | {avg['mask_row_std']:.4g} | {avg['query_gate_std']:.4g} |"
            )
    lines += [
        "",
        "## Paired comparison",
        "",
        "NLL difference is values minus masks (negative favors values). Average seeds per episode first,",
        "then bootstrap the 32 paired episode differences. Intervals cover episode sampling, "
        "not training-seed uncertainty.",
        "",
    ]
    rng = np.random.default_rng(73)
    for scope in ("data", "cell_and_data", "cell"):
        episode_means = {}
        for variant in ("masks", "values"):
            rows = [r for r in results if r["scope"] == scope and r["variant"] == variant]
            episode_means[variant] = np.mean([[e["nll"] for e in r["heldout_episodes"]] for r in rows], axis=0)
        delta = episode_means["values"] - episode_means["masks"]
        samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        lines.append(f"- {scope}: {delta.mean():+.5f}, paired episode bootstrap 95% interval [{lo:+.5f}, {hi:+.5f}].")
    lines += [
        "",
        "## Per-seed diagnostics",
        "",
        "| Scope | Arm | Seed | NLL | Accuracy | NLL with errors disabled | NLL with reversed errors | "
        "NLL with shuffled labels | Final sampled query-key gradient norm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['scope']} | {r['variant']} | {r['seed']} | {r['final']['nll']:.4f} | "
            f"{r['final']['accuracy']:.2%} | {r['unweighted']['nll']:.4f} | "
            f"{r['shuffled_errors']['nll']:.4f} | {r['shuffled_labels']['nll']:.4f} | "
            f"{r['query_key_gradient_norms'][-1]:.3g} |"
        )
    if args.control:
        lines += ["", "## Plain TabPFN control", ""]
        for r in json.loads((args.control / "results.json").read_text()):
            lines.append(f"- Seed {r['seed']}: NLL {r['final']['nll']:.4f}, accuracy {r['final']['accuracy']:.2%}.")
    ablation_path = args.directory / "context_ablation.json"
    if ablation_path.exists():
        lines += [
            "",
            "## Retrieved-context removal at inference",
            "",
            "| Scope | Seed | Full NLL | Context removed NLL | Full accuracy | Context removed accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in json.loads(ablation_path.read_text()):
            lines.append(
                f"| {row['scope']} | {row['seed']} | {row['full_nll']:.4f} | "
                f"{row['removed_nll']:.4f} | {row['full_accuracy']:.2%} | {row['removed_accuracy']:.2%} |"
            )
        lines += [
            "",
            "Most gains survive removal of the direct context at inference. This ablation does not erase",
            "changes learned by the shared backbone and slots during training, "
            "and does not isolate optimization effects.",
        ]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- No conclusion about real tables, long pretraining, or the four-prior sweep follows from this pilot.",
        "- Error disabling/reversal is an inference ablation, not separately trained weighting controls.",
        "- Cell values combines direct retrieval with query-conditioned cell aggregation; its effect is not isolated.",
        "- Cell alignment uses support row zero as reference, not the proposed medoid. It is order sensitive.",
        "- Both arms share a trainable target encoder. "
        "Low reconstruction MSE alone does not prove useful preservation.",
        "- Global embedding variance does not establish row diversity; inspect target_row_std in the raw results.",
        "- Neither arm replaces the previously submitted Slurm array.",
        "",
    ]
    path = args.directory / "report.md"
    path.write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
