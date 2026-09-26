import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

raw = sys.stdin.read()
marker = "@@DATA@@"
if marker not in raw:
    raise SystemExit("Remote aggregate marker missing; no plot generated")
data = json.loads(raw[raw.index(marker) + len(marker):].strip())
rows = data["rows"]
sizes = ["small", "medium"]
families = ["persistent", "soft_gate"]
metrics = ["ce_improvement", "accuracy_improvement_pp"]
labels = {"persistent": "Persistent", "soft_gate": "Soft-gate"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6), constrained_layout=True)
for row_idx, family in enumerate(families):
    fam_rows = [r for r in rows if r["family"] == family]
    for size_idx, size in enumerate(sizes):
        sr = [r for r in fam_rows if r["size"] == size]
        class_vals = sorted({r["num_classes"] for r in sr})
        regime_vals = sorted({r["num_regimes"] for r in sr})
        for metric_idx, metric in enumerate(metrics):
            ax = axes[row_idx, size_idx * 2 + metric_idx]
            matrix = np.full((len(class_vals), len(regime_vals)), np.nan)
            annotations = {}
            for r in sr:
                i = class_vals.index(r["num_classes"])
                j = regime_vals.index(r["num_regimes"])
                matrix[i, j] = r[metric]
                annotations[(i, j)] = r["n"]
            valid = matrix[np.isfinite(matrix)]
            if not len(valid):
                ax.set_visible(False)
                continue
            vmax = max(float(np.max(np.abs(valid))), 1e-6)
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(regime_vals)), regime_vals)
            ax.set_yticks(range(len(class_vals)), class_vals)
            ax.set_xlabel("Number of regimes")
            if size_idx == 0 and metric_idx == 0:
                ax.set_ylabel(f"{labels[family]}\nNumber of classes")
            title_metric = "CE improvement" if metric == "ce_improvement" else "Accuracy improvement (pp)"
            ax.set_title(f"{size.title()} · {title_metric}")
            for (i, j), n in annotations.items():
                value = matrix[i, j]
                value_text = f"{value:+.3f}" if metric == "ce_improvement" else f"{value:+.1f}"
                ax.text(j, i, f"{value_text}\n(n={n})", ha="center", va="center", fontsize=7,
                        color="black" if abs(value) < 0.72 * vmax else "white")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("lower CE is better" if metric == "ce_improvement" else "higher accuracy is better", fontsize=7)

fig.suptitle(
    "Observed-regime vs original model · full-bank paired comparison\n"
    "True-membership condition; positive values favor observed model. "
    "2,880 episodes per family (360 paired cells).",
    fontsize=12,
)
fig.text(
    0.5, -0.012,
    "CE improvement = original CE − observed CE; accuracy improvement = observed − original. "
    "Original checkpoints use legacy attention code; implementation remains a confound.",
    ha="center", fontsize=8,
)
out = Path("monitoring-plots/observed-regime-fullbank-original-comparison.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out.resolve())
