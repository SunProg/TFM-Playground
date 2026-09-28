"""Compare trained TabICLv2 slot/MLP heads with zero-shot TabICLv2 on TabArena.

Uses TabArena's canonical task metadata, splits, metrics, and aggregation. The
runner writes per-task metrics and subgroup summaries for the multiclass suite.
An optional dataset-to-imbalance CSV adds class-imbalance strata when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TABARENA_COMMIT = "06334097d539a5d494e56576cb973d09e251dc8c"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("slot_decoder")
    if not isinstance(metadata, dict):
        raise ValueError(f"Not a slot-decoder/MLP-control checkpoint: {path}")
    return metadata


def _compute_imbalance_csv(context, output_path: Path) -> Path:
    """Compute descriptive max/min class-frequency ratios for canonical tasks."""
    import openml
    import pandas as pd

    task_metadata = context.task_metadata_collection.to_dataframe()
    task_metadata = task_metadata.loc[
        task_metadata["problem_type"].isin(("binary", "multiclass")), ["dataset_name", "task_id_str"]
    ].drop_duplicates("dataset_name")
    rows = []
    for record in task_metadata.itertuples(index=False):
        row: dict[str, Any] = {"dataset": str(record.dataset_name), "imbalance_ratio": float("nan")}
        try:
            task = openml.tasks.get_task(int(record.task_id_str), download_splits=False)
            _, target = task.get_X_and_y(dataset_format="dataframe")
            if isinstance(target, pd.DataFrame):
                if target.shape[1] != 1:
                    raise ValueError("Expected one classification target column.")
                target = target.iloc[:, 0]
            frequencies = target.value_counts(dropna=True)
            if not frequencies.empty and frequencies.min() > 0:
                row["imbalance_ratio"] = float(frequencies.max() / frequencies.min())
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    imbalance_path = output_path / "class_imbalance.csv"
    pd.DataFrame(rows).to_csv(imbalance_path, index=False)
    return imbalance_path


def _add_subgroup_summaries(context, task_results, output_path: Path, imbalance_csv: str | None) -> None:
    import pandas as pd

    results = task_results.copy()
    grid = context.task_metadata_collection.task_grid()[["dataset", "split", "max_train_rows", "n_classes"]].rename(
        columns={"split": "arena_split"}
    )
    results = results.merge(grid, left_on=["dataset", "fold"], right_on=["dataset", "arena_split"], how="left")

    strata = {
        "multiclass": lambda frame: frame["problem_type"] == "multiclass",
        "multiclass_train_le_2k": lambda frame: (
            (frame["problem_type"] == "multiclass") & (frame["max_train_rows"] <= 2_000)
        ),
        "multiclass_train_gt_2k": lambda frame: (
            (frame["problem_type"] == "multiclass") & (frame["max_train_rows"] > 2_000)
        ),
        "multiclass_classes_3_5": lambda frame: (
            (frame["problem_type"] == "multiclass") & frame["n_classes"].between(3, 5)
        ),
        "multiclass_classes_6_10": lambda frame: (
            (frame["problem_type"] == "multiclass") & frame["n_classes"].between(6, 10)
        ),
    }
    if imbalance_csv:
        imbalance = pd.read_csv(imbalance_csv)
        required = {"dataset", "imbalance_ratio"}
        if not required.issubset(imbalance.columns):
            raise ValueError("Imbalance CSV must contain dataset and imbalance_ratio columns.")
        if imbalance["dataset"].duplicated().any():
            raise ValueError("Imbalance CSV must contain one row per dataset.")
        results = results.merge(imbalance[["dataset", "imbalance_ratio"]], on="dataset", how="left")
        results["imbalance_stratum"] = pd.cut(
            results["imbalance_ratio"],
            bins=[0, 2, 5, float("inf")],
            labels=["ratio_le_2", "ratio_2_5", "ratio_gt_5"],
            include_lowest=True,
        ).astype("string")
        for category in ("ratio_le_2", "ratio_2_5", "ratio_gt_5"):
            strata[f"multiclass_{category}"] = lambda frame, category=category: (
                (frame["problem_type"] == "multiclass") & (frame["imbalance_stratum"] == category)
            )

    aggregate_rows = []
    metric_columns = [column for column in ("metric_error", "aux_metric_error") if column in results]
    model_column = "method" if "method" in results else "framework"
    for stratum, mask_function in strata.items():
        subset = results.loc[mask_function(results)]
        if not subset.empty:
            summary = subset.groupby(model_column, dropna=False)[metric_columns].agg(["mean", "count"])
            summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
            summary.insert(0, "stratum", stratum)
            aggregate_rows.append(summary.reset_index())
    results.to_csv(output_path / "task_metrics_with_metadata.csv", index=False)
    if aggregate_rows:
        pd.concat(aggregate_rows, ignore_index=True).to_csv(output_path / "stratified_metrics.csv", index=False)


def run_official_tabarena(
    *,
    decoders: list[tuple[str, str]],
    output_dir: str,
    results_dir: str,
    zero_shot_checkpoint: str | None = None,
    lite: bool = True,
    debug_mode: bool = False,
    imbalance_csv: str | None = None,
    skip_imbalance_analysis: bool = False,
    skip_zero_shot: bool = False,
    classification_only: bool = False,
):
    try:
        from tabarena.benchmark.experiment import ModelConstraints, TabArenaV0pt1ExperimentBundle
        from tabarena.contexts import TabArenaContext
    except ImportError as error:
        raise ImportError("Official evaluation requires the pinned TabArena environment.") from error

    import tabarena

    from tfmplayground.experiments.evaluate_task_posterior_tabarena import _installed_tabarena_revision
    from tfmplayground.tabarena_model import TabArenaTabICLDecoderModel, TabArenaTabICLZeroShotModel

    installed_revision = _installed_tabarena_revision(tabarena.__file__)
    if installed_revision != TABARENA_COMMIT:
        raise RuntimeError(
            f"Expected TabArena commit {TABARENA_COMMIT}, found {installed_revision or 'unverifiable install'}."
        )
    if not decoders:
        raise ValueError("Provide at least one trained decoder checkpoint with --decoder NAME=CHECKPOINT.")

    output_path = Path(output_dir)
    model_protocol = []
    decoder_configs = []
    decoder_names = set()
    for name, checkpoint_name in decoders:
        if name in decoder_names:
            raise ValueError(f"Duplicate decoder name: {name!r}.")
        decoder_names.add(name)
        checkpoint_path = Path(checkpoint_name).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Decoder checkpoint does not exist: {checkpoint_path}")
        metadata = _metadata(checkpoint_path)
        model_protocol.append(
            {
                "name": name,
                "tabarena_config": f"TabICLv2Decoder_c{len(decoder_configs) + 1}",
                "checkpoint": str(checkpoint_path),
                "sha256": _digest(checkpoint_path),
                "head": metadata,
            }
        )
        decoder_configs.append({"model": str(checkpoint_path)})

    models = []
    if decoder_configs:
        decoder_generator = TabArenaTabICLDecoderModel.config_generator()
        decoder_generator.manual_configs = decoder_configs
        models.append((decoder_generator, 0))

    if not skip_zero_shot:
        zero_shot_path = Path(zero_shot_checkpoint).resolve() if zero_shot_checkpoint else None
        if zero_shot_path is not None and not zero_shot_path.is_file():
            raise FileNotFoundError(f"Released TabICLv2 checkpoint does not exist: {zero_shot_path}")
        if zero_shot_checkpoint is not None:
            zero_generator = TabArenaTabICLZeroShotModel.config_generator()
            zero_generator.manual_configs = [{"model": str(zero_shot_path)}]
            models.append((zero_generator, 0))
            model_protocol.append(
                {
                    "name": "zero_shot_tabicl_v2",
                    "tabarena_config": "TabICLv2ZeroShot_c1",
                    "checkpoint": str(zero_shot_path),
                    "sha256": _digest(zero_shot_path),
                }
            )
        else:
            zero_generator = TabArenaTabICLZeroShotModel.config_generator()
            zero_generator.manual_configs = [{}]
            models.append((zero_generator, 0))
            model_protocol.append(
                {
                    "name": "zero_shot_tabicl_v2",
                    "tabarena_config": "TabICLv2ZeroShot_c1",
                    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
                    "sha256": None,
                }
            )
    output_path.mkdir(parents=True, exist_ok=False)
    context = TabArenaContext()
    if imbalance_csv is None and not skip_imbalance_analysis:
        imbalance_csv = str(_compute_imbalance_csv(context, output_path))
    subset = ["classification"] if classification_only else ["multiclass", "tabpfn"]
    if lite:
        subset.insert(0, "lite")
    protocol = {
        "tabarena_commit": TABARENA_COMMIT,
        "tabarena_installed_revision": installed_revision,
        "suite": "TabArena-v0.1",
        "subset": ("lite & classification" if lite else "classification") if classification_only else ("lite & multiclass & tabpfn" if lite else "multiclass & tabpfn"),
        "models": model_protocol,
        "imbalance_csv": str(Path(imbalance_csv).resolve()) if imbalance_csv else None,
        "aggregation": "official TabArena task metrics/Elo; detailed per-task metrics and size/class strata also saved",
        "regression": "out_of_scope",
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    constraints = ModelConstraints(
        max_n_samples_train_per_fold=100_000 if classification_only else 10_000,
        max_n_features=2_000 if classification_only else 500,
        max_n_classes=10,
        regression_support=False,
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=models,
        outer_experiments=True,
        custom_model_constraints={
            TabArenaTabICLDecoderModel.ag_key: constraints,
            TabArenaTabICLZeroShotModel.ag_key: constraints,
        },
    ).build_experiments()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[TabICLv2 decoder] ",
        debug_mode=debug_mode,
    )
    leaderboard, task_results = context.compare(
        output_dir=output_path / "leaderboard",
        subset=subset,
        return_results=True,
        plot=False,
    )
    _add_subgroup_summaries(context, task_results, output_path, imbalance_csv)
    return leaderboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Trained slot or frozen-MLP checkpoint; repeat to compare runs/seeds.",
    )
    parser.add_argument("--zero-shot-checkpoint", help="Optional local released TabICLv2 checkpoint.")
    parser.add_argument("--skip-zero-shot", action="store_true", help="Do not evaluate the built-in TabICLv2 zero-shot model.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--imbalance-csv", help="Optional CSV with dataset,imbalance_ratio for imbalance strata.")
    parser.add_argument(
        "--skip-imbalance-analysis",
        action="store_true",
        help="Skip loading canonical OpenML targets to calculate class-imbalance strata.",
    )
    parser.add_argument("--full", action="store_true", help="Use the non-lite task split subset.")
    parser.add_argument("--classification-only", action="store_true", help="Evaluate all TabArena-v0.1 classification tasks with TabICLv2-scale row/feature limits.")
    parser.add_argument("--debug-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decoders = []
    for item in args.decoder:
        if "=" not in item:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {item!r}.")
        decoders.append(tuple(item.split("=", maxsplit=1)))
    leaderboard = run_official_tabarena(
        decoders=decoders,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        zero_shot_checkpoint=args.zero_shot_checkpoint,
        lite=not args.full,
        debug_mode=args.debug_mode,
        imbalance_csv=args.imbalance_csv,
        skip_imbalance_analysis=args.skip_imbalance_analysis,
        skip_zero_shot=args.skip_zero_shot,
        classification_only=args.classification_only,
    )
    print(leaderboard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
