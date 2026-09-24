"""Merge sharded ``evaluate_tabpfn_versions_on_v4_bank.py`` reports into one.

Shards are written as ``<stem>__<split>__<start>-<stop>.json``; this
concatenates their ``per_episode`` rows (checking the ranges tile the bank
without gaps or overlaps), recomputes ``overall`` and every ``by_*`` summary
with the same helpers the evaluator uses, and writes ``<stem>__<split>.json``.

    python scripts/merge_v4_bank_shards.py --output runs_eval/tabpfn_baselines/tabpfn_v3__validation.json \\
        runs_eval/tabpfn_baselines/tabpfn_v3__validation__*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tfmplayground.experiments.multiregime_v4_evaluation import FACTORS, _nanmean, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-episodes", type=int, default=None, help="fail unless the merged count equals this")
    args = parser.parse_args()

    reports = [json.load(open(p)) for p in args.shards]
    reports.sort(key=lambda r: r["episode_range"][0])
    for a, b in zip(reports, reports[1:]):
        if a["episode_range"][1] != b["episode_range"][0]:
            raise SystemExit(f"shards do not tile: {a['episode_range']} then {b['episode_range']}")
    head = reports[0]
    for r in reports[1:]:
        for key in ("split", "model", "checkpoint", "n_estimators"):
            if r.get(key) != head.get(key):
                raise SystemExit(f"shard mismatch on {key}: {r.get(key)!r} vs {head.get(key)!r}")

    per_episode = [row for r in reports for row in r["per_episode"]]
    if args.expected_episodes is not None and len(per_episode) != args.expected_episodes:
        raise SystemExit(f"merged {len(per_episode)} episodes, expected {args.expected_episodes}")

    merged = {
        **{k: head[k] for k in ("split", "model", "checkpoint", "n_estimators")},
        "episodes": len(per_episode),
        "prior_fallback_episodes": int(sum(r.get("prior_fallback", False) for r in per_episode)),
        "episode_range": [reports[0]["episode_range"][0], reports[-1]["episode_range"][1]],
        "merged_from": [str(p) for p in args.shards],
        "overall": {
            "query_cross_entropy": float(np.mean([r["query_cross_entropy"] for r in per_episode])),
            "query_accuracy": float(np.mean([r["query_accuracy"] for r in per_episode])),
            "query_auc": _nanmean(r["query_auc"] for r in per_episode),
        },
        "per_episode": per_episode,
        "by_cell": _summary(per_episode, FACTORS),
        "by_num_regimes": _summary(per_episode, ("num_regimes",)),
        "by_num_classes": _summary(per_episode, ("num_classes",)),
        "by_support_size": _summary(per_episode, ("support_size",)),
        "by_rule_mode": _summary(per_episode, ("rule_mode",)),
        "by_mechanism_mode": _summary(per_episode, ("mechanism_mode",)),
        "by_task_family": _summary(per_episode, ("task_family",)),
    }
    out = Path(args.output)
    out.write_text(json.dumps(merged, indent=1) + "\n")
    o = merged["overall"]
    print(f"{merged['model']} {merged['split']}: {merged['episodes']} episodes, loss={o['query_cross_entropy']:.4f} "
          f"acc={o['query_accuracy']:.4f} auc={o['query_auc']:.4f} -> {out}")


if __name__ == "__main__":
    main()
