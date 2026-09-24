"""Episode-weighted fair metrics for v4-bank reports, one row per report.

Takes any number of report JSONs that carry ``per_episode`` rows with
``episode_id`` (training-loop ``v4_validation/step-*.json``, ``v4_test/*.json``,
``evaluate_checkpoint_on_v4_bank.py`` output, or the TabPFN/TabICL baselines)
plus the matching ``compute_v4_bank_prior_baselines.py`` file, and prints
excess cross entropy (CE − class-prior CE; lower is better, ≥0 = no better
than the prior) and accuracy gain (acc − majority accuracy) pooled over:

    mr1 mr2 mr3 mr4  = rule_mode=multiregime at num_regimes=1..4
    shared           = rule_mode=shared (all regimes)
    all              = every episode

Stdlib only, so it runs unchanged on the cluster login node:

    python scripts/summarize_v4_report_excess.py --baselines validation_prior.json \\
        runs_eval/tabpfn_baselines/tabpfn_v2.2__validation.json ...
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

PANELS = ("mr1", "mr2", "mr3", "mr4", "shared", "all")


def panel_of(episode: dict) -> list[str]:
    """Pooled panels (mr1..mr4 / shared / all) plus the num_regimes x num_classes cells
    (``mr2c3`` = multiregime, 2 regimes, 3 classes; ``sharedc3`` = shared rule, 3 classes)."""
    panels = ["all"]
    classes = int(episode["num_classes"])
    if episode["rule_mode"] == "shared":
        panels += ["shared", f"sharedc{classes}"]
    else:
        regimes = int(episode["num_regimes"])
        panels += [f"mr{regimes}", f"mr{regimes}c{classes}"]
    return panels


def summarize(
    report: dict,
    baselines: dict[str, dict],
    metric: str,
    support_size: int | None = None,
    mechanism: str | None = None,
    min_classes: int | None = None,
    feature_counts: list[int] | None = None,
) -> dict[str, float | None]:
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for episode in report["per_episode"]:
        base = baselines.get(str(episode["episode_id"]))
        if base is None or (support_size is not None and int(episode["support_size"]) != support_size):
            continue
        if mechanism is not None and episode["mechanism_mode"] != mechanism:
            continue
        if min_classes is not None and int(episode["num_classes"]) < min_classes:
            continue
        if feature_counts is not None and int(episode["num_features"]) not in feature_counts:
            continue
        if metric == "excess_ce":
            value = episode["query_cross_entropy"] - base["prior_cross_entropy"]
        elif metric == "auc":
            value = episode.get("query_auc")
            if value is None or value != value:  # degenerate episode (single class in the query)
                continue
        else:
            value = episode["query_accuracy"] - base["majority_accuracy"]
        for panel in panel_of(episode):
            total[panel] += value
            count[panel] += 1
    return {p: (total[p] / count[p] if count[p] else None) for p in list(PANELS) + sorted(k for k in count if k not in PANELS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--baselines", required=True, help="prior-baseline JSON for the bank these reports were scored on")
    parser.add_argument("--metric", choices=("excess_ce", "acc_gain", "auc"), default="excess_ce")
    parser.add_argument("--label-depth", type=int, default=3, help="path components (from the right) used as the row label")
    parser.add_argument(
        "--json-output",
        default=None,
        help="also write {label: {panel: value}} for both metrics, consumable by plot_v4_per_regime_history.py --reference",
    )
    parser.add_argument("--support-size", type=int, default=None, help="restrict to episodes with this support_size")
    parser.add_argument("--mechanism", choices=("r_z", "g_z"), default=None, help="restrict to one mechanism_mode")
    parser.add_argument("--min-classes", type=int, default=None, help="drop episodes with num_classes below this")
    parser.add_argument("--feature-counts", type=int, nargs="+", default=None, help="keep only episodes with these num_features")
    args = parser.parse_args()

    with open(args.baselines) as f:
        baselines = json.load(f)["episodes"]

    print(f"{'report':58s} " + " ".join(f"{p:>8s}" for p in PANELS) + f"   ({args.metric})")
    collected: dict[str, dict[str, dict[str, float | None]]] = {}
    for path in args.reports:
        try:
            with open(path) as f:
                report = json.load(f)
        except Exception as error:  # noqa: BLE001
            print(f"{path}: unreadable ({error})")
            continue
        label = "/".join(os.path.normpath(path).split(os.sep)[-args.label_depth :])
        row = summarize(report, baselines, args.metric, args.support_size, args.mechanism, args.min_classes, args.feature_counts)
        print(f"{label:58s} " + " ".join(f"{row[p]:+8.4f}" if row[p] is not None else f"{'-':>8s}" for p in PANELS))
        if args.json_output:
            collected[label] = {
                "excess_cross_entropy": summarize(report, baselines, "excess_ce", args.support_size, args.mechanism, args.min_classes, args.feature_counts),
                "accuracy_gain": summarize(report, baselines, "acc_gain", args.support_size, args.mechanism, args.min_classes, args.feature_counts),
            }
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(collected, f, indent=1)


if __name__ == "__main__":
    main()
