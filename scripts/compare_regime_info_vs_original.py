"""Compare all live v4 regime-exposure training arms against the original (no-multiregime) model.

Reads the local mirror produced by ``sync_observed_regime_monitor.sh``
(``artifacts/cluster_sync/observed_regime_monitor/``): each live arm's
``regime_information_progress.jsonl`` (the true/shuffled/hidden probe, logged
every ~1000 steps) plus the original model's fixed checkpoint probes at steps
2000/4000/6000/8000. The live arms are:

* ``observed_regime``            -- multiregime data, regime ID always exposed.
* ``hidden_regime_control``      -- multiregime data, regime ID never exposed.
* ``probability_regime_control`` -- per-episode Bernoulli exposure (exposed
                                     and hidden land on *different* episodes).
* ``paired_regime_control``      -- every episode trained both exposed and
                                     hidden (the same episode, duplicated).

For each model size, all arms that have reached a probe step are compared at
the highest step common to all of them (each arm's own history is searched
for its most recent probe at-or-below that step), against the original
model's checkpoint at the same-or-nearest-lower of its four fixed steps. This
answers "is regime-ID curriculum training teaching the model to use the
column more than training alone would" across every intervention we've tried,
not just one.

Appends one entry per (size, reference step) to a running JSONL log (so the
comparison's evolution over training is visible) and (re)writes a
human-readable Markdown summary of the latest snapshot.

    python scripts/compare_regime_info_vs_original.py \\
        --monitor-dir artifacts/cluster_sync/observed_regime_monitor \\
        --report artifacts/cluster_sync/observed_regime_monitor/comparison.md \\
        --log artifacts/cluster_sync/observed_regime_monitor/comparison_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ORIGINAL_STEPS = (2000, 4000, 6000, 8000)
SIZE_RE = re.compile(r"(small|medium|large)")
# Display/report order: no exposure -> confounded partial exposure -> full
# exposure -> the same-episode paired design (see module docstring).
ARM_ORDER = (
    "hidden_regime_control",
    "probability_regime_control",
    "observed_regime",
    "paired_regime_control",
)


def _overall(report: dict) -> dict[str, dict]:
    return {row["information"]: row for row in report["overall"]}


def _paired(report: dict) -> dict[tuple[str, str, str], dict]:
    return {(row["task_family"], row["information"], row["reference"]): row for row in report["paired_differences"]}


def _size_of(path: Path) -> str | None:
    match = SIZE_RE.search(path.as_posix())
    return match.group(1) if match else None


def load_live_series(monitor_dir: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """``{(arm, size): {step: probe_report}}`` for every synced live run."""
    series: dict[tuple[str, str], dict[int, dict]] = {}
    for progress_path in sorted(monitor_dir.glob("runs/*/*/*/regime_information_progress.jsonl")):
        # .../runs/<arm>/<run_name>/seed-*/regime_information_progress.jsonl
        arm = progress_path.parents[2].name
        size = _size_of(progress_path.parents[1])
        if size is None:
            print(f"skipping {progress_path}: could not infer model size")
            continue
        rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        by_step = series.setdefault((arm, size), {})
        for row in rows:
            by_step[int(row["step"])] = row
    return series


def load_original_series(monitor_dir: Path, size: str) -> dict[int, dict]:
    by_step = {}
    for step in ORIGINAL_STEPS:
        path = monitor_dir / "original_probes" / f"original-{size}" / "seed-2402" / "checkpoint_probes" / f"step-{step:06d}.json"
        if path.exists():
            by_step[step] = json.loads(path.read_text())
    return by_step


def nearest_step_at_or_below(series: dict[int, dict], target: int) -> int | None:
    eligible = [s for s in series if s <= target]
    return max(eligible) if eligible else None


def arm_summary(report: dict, step: int) -> dict:
    overall = _overall(report)
    paired = _paired(report)
    families = sorted({fam for (fam, info, ref) in paired if info == "true" and ref == "hidden"})
    return {
        "step": step,
        "overall": {
            cond: {"acc": overall[cond]["query_accuracy"], "ce": overall[cond]["query_cross_entropy"]}
            for cond in ("hidden", "shuffled", "true")
            if cond in overall
        },
        "true_minus_hidden_by_family": {
            fam: {
                "ce_gap": paired[(fam, "true", "hidden")]["cross_entropy_difference"],
                "acc_gap": paired[(fam, "true", "hidden")]["accuracy_difference"],
            }
            for fam in families
        },
    }


def build_snapshots(monitor_dir: Path) -> list[dict]:
    live = load_live_series(monitor_dir)
    sizes = sorted({size for (_, size) in live})
    snapshots = []
    for size in sizes:
        live_for_size = {arm: steps for (arm, s), steps in live.items() if s == size}
        if not live_for_size:
            continue
        reference_step = min(max(steps) for steps in live_for_size.values())
        arms: dict[str, dict] = {}
        for arm, steps in live_for_size.items():
            step = nearest_step_at_or_below(steps, reference_step)
            if step is not None:
                arms[arm] = arm_summary(steps[step], step)
        original_series = load_original_series(monitor_dir, size)
        original_step = nearest_step_at_or_below(original_series, reference_step)
        if original_step is not None:
            arms["original"] = arm_summary(original_series[original_step], original_step)
        if arms:
            snapshots.append({"size": size, "reference_step": reference_step, "arms": arms})
    return snapshots


def render_markdown(snapshots: list[dict], generated_at: str) -> str:
    lines = ["# Regime-exposure arms vs. original model — regime-info comparison", "", f"_generated {generated_at}_", ""]
    if not snapshots:
        lines.append("No live regime-exposure runs found (nothing synced in the staleness window).")
        return "\n".join(lines) + "\n"
    arm_display = ("original",) + ARM_ORDER
    for snap in snapshots:
        present = [arm for arm in arm_display if arm in snap["arms"]]
        lines.append(f"## {snap['size']} (reference step {snap['reference_step']})")
        lines.append("")
        lines.append("Each arm's own step (may lag the reference step if it hasn't probed that far yet):")
        lines.append("")
        lines.append("| arm | step |")
        lines.append("|---|---|")
        for arm in present:
            lines.append(f"| {arm} | {snap['arms'][arm]['step']} |")
        lines.append("")
        lines.append("Overall accuracy / cross-entropy by information condition:")
        lines.append("")
        lines.append("| information | " + " | ".join(present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for cond in ("hidden", "shuffled", "true"):
            cells = []
            for arm in present:
                row = snap["arms"][arm]["overall"].get(cond)
                cells.append(f"{row['acc']:.4f} / {row['ce']:.4f}" if row else "-")
            lines.append(f"| {cond} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("true − hidden gap by task family (ΔCE / Δacc):")
        lines.append("")
        families = sorted({fam for arm in present for fam in snap["arms"][arm]["true_minus_hidden_by_family"]})
        lines.append("| family | " + " | ".join(present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for fam in families:
            cells = []
            for arm in present:
                gap = snap["arms"][arm]["true_minus_hidden_by_family"].get(fam)
                cells.append(f"{gap['ce_gap']:.4f} / {gap['acc_gap']:.4f}" if gap else "-")
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    snapshots = build_snapshots(args.monitor_dir)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for snapshot in snapshots:
        snapshot["generated_at"] = generated_at

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(snapshots, generated_at))
    print(f"wrote {args.report}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as f:
        for snapshot in snapshots:
            f.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(f"appended {len(snapshots)} snapshot(s) to {args.log}")


if __name__ == "__main__":
    main()
