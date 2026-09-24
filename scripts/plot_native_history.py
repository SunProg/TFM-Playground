"""Training histories of the native-prior runs (history.jsonl): bank CE (z-blind and z-exposed), ordinary-prior
validation CE, and rolling training CE, one column per size, prior-CE reference line.

    python scripts/plot_native_history.py --root <dir containing native/<run>/seed-2402/history.jsonl> --output-dir figures/native
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"original": "#1f77e6", "rg_z-fixed": "#7b3fe4", "rg_z-curriculum": "#00a3c4",
          # multiregime-prior ablation arms (the multiregime branch filtered)
          "rg_z-fixed-softgate-only": "#e377c2", "rg_z-fixed-multiclass-only": "#d62728",
          "rg_z-fixed-softgate-multiclass": "#8c564b",
          "rg_z-curriculum-softgate-only": "#98df8a", "rg_z-curriculum-multiclass-only": "#2ca02c",
          "rg_z-curriculum-softgate-multiclass": "#17becf"}
PRIOR_CE = 0.8360  # native z-blind validation bank, class-prior CE (episode mean)


def load(p: Path):
    rows = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); rows[r["step"]] = r
    return [rows[s] for s in sorted(rows)]


def rolling(v, w=100):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0)); return np.concatenate([np.full(w - 1, np.nan), (c[w:] - c[:-w]) / w])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--groups", nargs="+", default=["native"],
                    help="subdirectories of --root to plot, e.g. native native_ablation")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.output_dir) / "history"; out.mkdir(parents=True, exist_ok=True)
    sizes = [s for s in a.sizes if any(list(root.glob(f"{g}/*-{s}/seed-2402/history.jsonl")) for g in a.groups)]
    fig, axes = plt.subplots(4, len(sizes), figsize=(7 * len(sizes), 18), squeeze=False)
    for j, size in enumerate(sizes):
        paths = [q for g in a.groups for q in sorted(root.glob(f"{g}/*-{size}/seed-2402/history.jsonl"))]
        for p in paths:
            fam = p.parts[-3][: -len(size) - 1]; rows = load(p); c = COLORS.get(fam, "gray")
            for key, ax, ls, lab in (("v4_validation_query_cross_entropy", axes[0][j], "-", "z-blind bank"),
                                     ("v4_validation_zx_query_cross_entropy", axes[0][j], "--", "z-exposed bank"),
                                     ("validation_query_cross_entropy", axes[1][j], "-", None)):
                pts = [(r["step"], r[key]) for r in rows if key in r and np.isfinite(r[key])]
                if pts:
                    ax.plot(*zip(*pts), ls=ls, marker="o" if ls == "-" else "s", ms=3, lw=1.6, color=c,
                            label=f"{fam}" + (f" ({lab})" if lab else "") + f" — step {pts[-1][0]}: {pts[-1][1]:.4f}")
            steps = [r["step"] for r in rows]
            axes[2][j].plot(steps, rolling([r["query_cross_entropy"] for r in rows]), lw=1.4, color=c, label=fam)
            mr = [(r["step"], r["multiregime_probability"]) for r in rows if "multiregime_probability" in r]
            axes[3][j].plot(*zip(*mr), lw=1.4, color=c, label=fam)
        axes[0][j].axhline(PRIOR_CE, color="black", lw=0.8, ls=":", label=f"class-prior CE {PRIOR_CE}")
        axes[0][j].set_title(f"{size} — v4 bank validation CE (native banks)"); axes[0][j].legend(fontsize=7)
        axes[1][j].set_title(f"{size} — ordinary-prior validation CE (trainer's fixed native stream)"); axes[1][j].legend(fontsize=7)
        axes[2][j].set_title(f"{size} — training query CE (rolling mean, 100 steps)"); axes[2][j].legend(fontsize=7)
        axes[3][j].set_title(f"{size} — multiregime share of training batches"); axes[3][j].legend(fontsize=7); axes[3][j].set_ylim(-0.02, 0.6)
        for i in range(4):
            axes[i][j].set_xlabel("step"); axes[i][j].grid(alpha=0.3); axes[i][j].set_xlim(0, 10000)
    fig.suptitle("Native-TabICL-prior runs (mixed-z dumps, 20 % warmup; rg_z-fixed: multiregime share 0.3, rg_z-curriculum: ramp to 0.5)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / f"native_history{a.tag}.png"; fig.savefig(p, dpi=110); print(p)


if __name__ == "__main__":
    main()
