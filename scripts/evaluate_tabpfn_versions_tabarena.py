#!/usr/bin/env python3
"""Real TabPFN v2.2/v2.6/v3 on the tabarena15 protocol, one run per version.

evaluate_tabarena_small.py's own --include-tabpfn path can only build a
TabPFNClassifier from the package's "auto" default, which resolves through
a gated hosted-weights download that fails non-interactively -- and its
committed HEAD has no --tabpfn-model-path flag to bypass that (that flag
exists only in another session's uncommitted local edit to that file, so
this deliberately does not depend on it or touch that file).

Reuses evaluate_tabarena_small.run() entirely unmodified for dataset
loading, fold splitting, and metric computation/CSV writing -- only its
module-level _build_tabpfn function is monkeypatched per version, so the
output format (fold_metrics.csv, per_dataset.csv, overall.csv) matches
every other TabArena result in this project exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tfmplayground.experiments.evaluate_tabarena_small as ets

VERSIONS = ("v2.2", "v2.6", "v3")


def make_build_tabpfn(version: str, checkpoint: str | None):
    def _build_tabpfn(config: ets.SmallTabArenaConfig):
        import os

        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        model_version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[version]
        kwargs = dict(
            device=config.device,
            random_state=0,
            n_estimators=8,
            softmax_temperature=0.9,
            balance_probabilities=False,
            average_before_softmax=False,
            fit_mode="fit_preprocessors",
            show_progress_bar=False,
        )
        if checkpoint:
            return TabPFNClassifier(model_path=checkpoint, **kwargs)
        return TabPFNClassifier.create_default_for_version(model_version, **kwargs)

    return _build_tabpfn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--checkpoint-v22", default=None, help="Omit to use the package's cached v2.2 default.")
    parser.add_argument("--checkpoint-v26", required=True)
    parser.add_argument("--checkpoint-v3", required=True)
    parser.add_argument("--max-predictors", type=int, default=30)
    parser.add_argument("--subsample", type=int, default=2048)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--versions", nargs="*", choices=VERSIONS, default=list(VERSIONS))
    args = parser.parse_args()

    checkpoints = {"v2.2": args.checkpoint_v22, "v2.6": args.checkpoint_v26, "v3": args.checkpoint_v3}

    for version in args.versions:
        ets._build_tabpfn = make_build_tabpfn(version, checkpoints[version])
        output_dir = args.output_root / f"tabpfn-{version}"
        print(f"=== {version} -> {output_dir} ===", flush=True)
        ets.run(
            ets.SmallTabArenaConfig(
                include_vanilla=False,
                include_sklearn=False,
                include_tabpfn=True,
                output_dir=str(output_dir),
                device=args.device,
                cache_directory=args.cache_directory,
                max_predictors=args.max_predictors,
                subsample=args.subsample,
                folds=args.folds,
                repeats=args.repeats,
                label_source="real",
                contamination=0.0,
                seed=args.seed,
            )
        )


if __name__ == "__main__":
    main()
