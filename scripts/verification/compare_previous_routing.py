"""Compare the earlier TableSlotModel objectives with the new routing pilots."""

import json
from pathlib import Path

import numpy as np

root = Path("results")
sources = {
    "Original query-only": ("reconstruction_routing_previous_query_1500_20260908", "query_only", 6),
    "Original label-alpha": ("reconstruction_routing_previous_alpha_1500_20260908", "label_alpha", 6),
    "Embedding MSE, old gate": ("reconstruction_routing_previous_embedding_1500_20260908", "embedding_mse", 4),
    "New mask routing": ("reconstruction_routing_pilot_1500_20260908", "masks", 12),
    "New support retrieval": ("reconstruction_routing_pilot_1500_20260908", "values", 12),
}
collected = {}
prior = None
for label, (directory, variant, count) in sources.items():
    rows = json.loads((root / directory / "results.json").read_text())
    assert len(rows) == count, f"Incomplete {label}: {len(rows)}"
    manifest = json.loads((root / directory / "manifest.json").read_text())
    if prior is None:
        prior = manifest["prior"]
    assert manifest["prior"] == prior and manifest["steps"] == 1500
    assert manifest["seeds"] == [11, 12]
    collected[label] = [r for r in rows if r["variant"] == variant]

lines = [
    "# Before/after routing comparison",
    "",
    "Same synthetic SCM prior, episode-seed schedule, 1,500 updates, two training seeds (11/12),",
    "backbone width 24 and two layers, four slots, AdamW at 0.001, batch size 2.",
    "Held-out evaluation: identical 32 tasks with 64 query rows each.",
    "",
    "These are retrained small-model controls, not evaluations of the full-size Slurm checkpoints.",
    "",
]
for metric, title in (("accuracy", "Mean accuracy"), ("nll", "Mean query NLL")):
    lines += [f"## {title}", "", "| Model | data | cell_and_data | cell |", "|---|---:|---:|---:|"]
    for label, rows in collected.items():
        values = []
        for scope in ("data", "cell_and_data", "cell"):
            scoped = [r for r in rows if r["scope"] == scope]
            if not scoped:
                values.append("Not supported")
            else:
                avg = np.mean([r["final"][metric] for r in scoped])
                values.append(f"{avg:.2%}" if metric == "accuracy" else f"{avg:.4f}")
        lines.append("| " + " | ".join([label] + values) + " |")
    lines.append("")
lines += [
    "## Objectives and comparison limits",
    "",
    "- Original query-only: existing TableSlotModel and decoder-alpha query gate, pure query NLL.",
    "- Original label-alpha: same model, query NLL + alpha-composited support-label NLL + 0.05 slot MI.",
    "- Embedding MSE, old gate: the initial modification, query NLL + embedding MSE, original query gate.",
    "- New routing arms: query NLL + embedding MSE, with reconstruction-weighted support routing.",
    "- Previous models are called directly through wrappers; no replacement slot encoder is substituted.",
    "- Same seed does not guarantee identical head initialization when module construction order differs.",
    "- Cell changes also include reconstruction, alignment, and (for retrieval) query-dependent cell weighting.",
    "- All heads/backbones train from scratch. Two seeds are insufficient to establish robust rankings.",
    "- This does not replace the missing matched retrieval-without-slots ablation.",
    "",
    "## Per-seed results",
    "",
    "| Model | Scope | Seed | Accuracy | Query NLL | Shuffled-label NLL |",
    "|---|---|---:|---:|---:|---:|",
]
for label, rows in collected.items():
    for r in rows:
        lines.append(
            f"| {label} | {r['scope']} | {r['seed']} | {r['final']['accuracy']:.2%} | "
            f"{r['final']['nll']:.4f} | {r['shuffled_labels']['nll']:.4f} |"
        )
path = root / "reconstruction_routing_before_after_20260908.md"
path.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:29]))
print(path)
