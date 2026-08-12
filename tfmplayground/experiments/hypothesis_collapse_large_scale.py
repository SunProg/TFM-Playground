"""Run the hypothesis-collapse diagnostic at larger support and trial scales.

This entry point keeps query counts modest because recovering a complete binary
joint distribution costs O(2**m), and instead scales the number of independent
trials and the number of in-context support rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from tfmplayground.experiments.hypothesis_collapse import build_parser as build_base_parser
from tfmplayground.experiments.hypothesis_collapse import config_from_args, run_experiment, write_artifacts

LARGE_SCALE_TRIALS = 64
LARGE_SCALE_QUERY_COUNTS = (2, 4)
LARGE_SCALE_EVIDENCE_COUNTS = (0, 2, 8, 16, 64, 128)
LARGE_SCALE_SUPPORT_SIZES = (16, 64, 128)


def build_parser():
    parser = build_base_parser()
    parser.description = __doc__
    parser.set_defaults(
        trials=LARGE_SCALE_TRIALS,
        query_counts=list(LARGE_SCALE_QUERY_COUNTS),
        evidence_counts=list(LARGE_SCALE_EVIDENCE_COUNTS),
        common_support_sizes=list(LARGE_SCALE_SUPPORT_SIZES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(Path("runs") / "hypothesis_collapse_large_scale" / timestamp)
    config = config_from_args(args)
    trial_metrics, joint_probabilities, summary, metadata = run_experiment(config)
    output_dir = write_artifacts(
        config=config,
        trial_metrics=trial_metrics,
        joint_probabilities=joint_probabilities,
        summary=summary,
        metadata={**metadata, "experiment": "large_support_scale"},
    )
    print(f"Wrote large-scale hypothesis-collapse experiment artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
