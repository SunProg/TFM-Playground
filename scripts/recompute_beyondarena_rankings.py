"""Recompute BeyondArena ranking tables from an existing fold-metrics run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import (
    BeyondArenaTask,
    ModelSelection,
    aggregate_rankings,
)


def _tuple_columns(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    return tuple(item for item in str(value).split(";") if item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    manifest = pd.read_csv(run_dir / "task_manifest.csv")
    eligible = manifest[manifest["status"] == "eligible"]
    tasks = [
        BeyondArenaTask(
            name=row.task_name,
            uuid=row.uuid,
            checksum=row.checksum,
            rows=int(row.rows),
            raw_features=int(row.raw_features),
            classes=int(row.classes),
            problem_type=row.problem_type,
            regime=row.regime,
            target_column=row.target_column,
            group_columns=_tuple_columns(row.group_columns),
            text_features=_tuple_columns(row.text_features),
            high_cardinality_features=_tuple_columns(row.high_cardinality_features),
            container=None,
            time_column=None if pd.isna(row.time_column) else str(row.time_column),
        )
        for row in eligible.itertuples(index=False)
    ]

    old_rankings = pd.read_csv(run_dir / "rankings.csv")
    if "z_expose" not in old_rankings.columns:
        old_rankings["z_expose"] = False
    selection_columns = [
        "model_identity",
        "family",
        "size",
        "checkpoint_policy",
        "checkpoint_path",
        "selection_status",
        "selection_reason",
        "validation_ce",
        "run_dir",
        "candidate_count",
        "model_kind",
        "z_expose",
    ]
    selections = []
    for row in old_rankings[selection_columns].drop_duplicates().itertuples(index=False):
        checkpoint_path = None if pd.isna(row.checkpoint_path) else Path(row.checkpoint_path)
        selections.append(
            ModelSelection(
                model_identity=row.model_identity,
                family=row.family,
                size=row.size,
                checkpoint_policy=row.checkpoint_policy,
                checkpoint_path=checkpoint_path,
                selection_status=row.selection_status,
                selection_reason=None if pd.isna(row.selection_reason) else row.selection_reason,
                validation_ce=None if pd.isna(row.validation_ce) else float(row.validation_ce),
                run_dir=Path(row.run_dir),
                candidate_count=int(row.candidate_count),
                model_kind=row.model_kind,
                z_expose=bool(row.z_expose),
            )
        )

    fold_rows = pd.read_csv(run_dir / "fold_metrics.csv").to_dict("records")
    rankings = aggregate_rankings(tasks, selections, fold_rows)
    rankings.to_csv(run_dir / "rankings.csv", index=False)
    print(f"Wrote {len(rankings)} ranking rows to {run_dir / 'rankings.csv'}")


if __name__ == "__main__":
    main()
