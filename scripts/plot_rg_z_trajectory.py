"""Training trajectory of the mixed r_z+g_z ("rg_z") runs against the single-prior runs of the same size.

Three rows per size: v4-bank validation CE (shared bank: blind runs on the repaired z-blind bank, zx runs on the
z-exposed bank), ordinary-prior validation CE (the trainer's fixed validation stream: 16 freshly generated mix_scm
batches with seed 2402+100000, identical for every run, so comparable across families), and rolling-mean training query CE. rg_z runs are drawn as thick lines, the references thin.

    python scripts/plot_rg_z_trajectory.py --root <dir with the history.jsonl tree> --output-dir figures/rg_z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"original": "#1f77e6", "r_z-fixed": "#ff5a2e", "r_z-curriculum": "#12b07a", "g_z-fixed": "#f0a000",
          "g_z-curriculum": "#f67ba8", "rg_z-fixed": "#7b3fe4", "rg_z-curriculum": "#00a3c4"}
BASE_JOB = {"small": "37283915", "medium": "37283921", "large": "37283995"}
PRIOR_CE = {"blind": 0.9615, "zx": None}


def load(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["step"]] = r  # last record per step wins (restarts)
    return [rows[s] for s in sorted(rows)]


def runs(root: Path, size: str, exposure: str) -> dict[str, list[dict]]:
    out = {}
    if exposure == "blind":
        for fam in ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]:
            p = root / "nanotabpfn_tabicl_mix_scm_prior_scale" / BASE_JOB[size] / f"{fam}-{size}" / "seed-2402" / "history.jsonl"
            if p.is_file():
                out[fam] = load(p)
        for fam in ["rg_z-fixed", "rg_z-curriculum"]:
            p = root / "rg_z" / f"{fam}-{size}" / "seed-2402" / "history.jsonl"
            if p.is_file():
                out[fam] = load(p)
            p = root / "rg_z_cancelled_warmup500" / "rg_z" / f"{fam}-{size}" / "seed-2402" / "history.jsonl"
            if p.is_file():
                out[fam + " (warmup 500, cancelled)"] = load(p)
    else:
        for fam in ["r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum", "rg_z-fixed", "rg_z-curriculum"]:
            p = root / "expose_z" / f"{fam}-{size}" / "seed-2402" / "history.jsonl"
            if p.is_file():
                out[fam] = load(p)
        for fam in ["rg_z-fixed", "rg_z-curriculum"]:
            p = root / "rg_z_cancelled_warmup500" / "expose_z" / f"{fam}-{size}" / "seed-2402" / "history.jsonl"
            if p.is_file():
                out[fam + " (warmup 500, cancelled)"] = load(p)
    return out


def style(fam: str):
    """(color, linewidth, alpha, zorder, linestyle) — rg_z thick; cancelled warmup-500 rg_z thick dashed."""
    base = fam.split(" ")[0]
    if fam.endswith("cancelled)"):
        return COLORS[base], 2.2, 0.55, 2, "--"
    if base.startswith("rg_z"):
        return COLORS[base], 2.6, 1.0, 3, "-"
    return COLORS[base], 1.2, 0.75, 1, "-"


def rolling(v, w=100):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    r = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), r])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for exposure in ["blind", "zx"]:
        sizes = [s for s in a.sizes if runs(root, s, exposure)]
        fig, axes = plt.subplots(3, len(sizes), figsize=(7 * len(sizes), 15), squeeze=False)
        for j, size in enumerate(sizes):
            data = runs(root, size, exposure)
            ax_v, ax_o, ax_t = axes[0][j], axes[1][j], axes[2][j]
            for fam, rows in data.items():
                color, lw, alpha, z, ls = style(fam)
                rg = fam.startswith("rg_z") and ls == "-"
                vs = [(r["step"], r["v4_validation_query_cross_entropy"]) for r in rows if "v4_validation_query_cross_entropy" in r]
                if vs:
                    ax_v.plot(*zip(*vs), marker="o" if rg else None, ms=4, lw=lw, alpha=alpha, color=color, zorder=z, ls=ls,
                              label=f"{fam} (step {vs[-1][0]}: {vs[-1][1]:.4f})")
                ov = [(r["step"], r["validation_query_cross_entropy"]) for r in rows if "validation_query_cross_entropy" in r]
                if ov:
                    ax_o.plot(*zip(*ov), marker="o" if rg else None, ms=4, lw=lw, alpha=alpha, color=color, zorder=z, ls=ls,
                              label=f"{fam} (step {ov[-1][0]}: {ov[-1][1]:.4f})")
                steps = [r["step"] for r in rows]
                ax_t.plot(steps, rolling([r["query_cross_entropy"] for r in rows]), lw=lw, alpha=alpha, color=color, zorder=z, ls=ls, label=fam)
            if PRIOR_CE[exposure]:
                ax_v.axhline(PRIOR_CE[exposure], color="black", lw=0.8, ls="--", label=f"class-prior CE {PRIOR_CE[exposure]}")
            bank = "repaired z-blind bank" if exposure == "blind" else "z-exposed bank"
            ax_v.set_title(f"{size} — v4 validation CE ({bank})")
            ax_v.set_xlabel("step"); ax_v.set_ylabel("validation query CE"); ax_v.grid(alpha=0.3)
            allv = [r["v4_validation_query_cross_entropy"] for rows in data.values() for r in rows if "v4_validation_query_cross_entropy" in r and r["step"] > 0]
            if allv:
                ax_v.set_ylim(min(allv) - 0.01, max(allv) + 0.01)
            ax_v.legend(fontsize=7)
            ax_o.set_title(f"{size} — ordinary-prior validation CE (trainer's fixed stream, same for every run)")
            ax_o.set_xlabel("step"); ax_o.set_ylabel("ordinary validation query CE"); ax_o.grid(alpha=0.3)
            ax_o.legend(fontsize=7)
            ax_t.set_title(f"{size} — training query CE (rolling mean, 100 steps)")
            ax_t.set_xlabel("step"); ax_t.set_ylabel("query CE"); ax_t.grid(alpha=0.3)
            ax_t.set_xlim(0, 10000)
            ax_t.legend(fontsize=7)
        tag = "z-blind" if exposure == "blind" else "z-exposed (zx-)"
        fig.suptitle(f"Mixed r_z+g_z prior (rg_z, thick solid = 20 % warmup; thick dashed = cancelled 500-step-warmup attempt) vs single-prior runs (500-step warmup) — {tag} training")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = out / f"rg_z_trajectory_{exposure}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(p)


if __name__ == "__main__":
    main()
