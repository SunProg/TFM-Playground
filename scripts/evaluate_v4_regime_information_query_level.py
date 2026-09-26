"""Diagnose how frozen NanoTabPFN checkpoints use regime information.

The primary v4 bank keeps the realized regime assignment only as evaluation
metadata.  This script compares three inputs on *the same held-out episodes*:

``hidden``
    The original feature matrix.  This reproduces the score-hidden evaluation.
``shuffled``
    A regime-tag column whose values are independently shuffled within support
    and query rows.  It has the true tag's cardinality and marginal counts but
    contains no membership information.
``true``
    A regime-tag column with the realized membership.  Tag names are randomly
    relabelled independently for each episode, so a model can only use the tag
    through support/query correspondence within that episode.

The true-tag condition is privileged diagnostic information, not a deployable
test setting.  It distinguishes difficulty in identifying a query's regime
from difficulty in applying a regime-specific rule.  No model is retrained.

Example:

    python scripts/evaluate_v4_regime_information.py \\
        --checkpoint runs/native/rg_z-curriculum-large/seed-2402/final_checkpoint.pth \\
        --bank data/multiregime_v4/tabicl_mix_scm_evaluation/test.h5 \\
        --output paper/native/regime_information_curriculum_large.json --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.multiregime_v4_evaluation import FACTORS, MultiregimeV4EvaluationBank, _episode_auc
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


InformationCondition = Literal["hidden", "shuffled", "true"]
CONDITIONS: tuple[InformationCondition, ...] = ("hidden", "shuffled", "true")


def _tag_column(
    support_regime: torch.Tensor,
    query_regime: torch.Tensor,
    metadata: list[dict[str, Any]],
    condition: InformationCondition,
    *,
    seed: int,
) -> torch.Tensor | None:
    """Return an episode-wise tag column, or ``None`` for the hidden condition.

    Randomly renaming true tags prevents their numeric identities from being a
    global convention.  Shuffling separately in support and query keeps both
    splits' tag-frequency vectors unchanged while breaking membership.
    """
    if condition == "hidden":
        return None
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown information condition: {condition!r}")

    tags = torch.empty(
        (support_regime.shape[0], support_regime.shape[1] + query_regime.shape[1]),
        dtype=torch.float32,
        device=support_regime.device,
    )
    for batch_index, row in enumerate(metadata):
        episode_seed = int(row["episode_seed"])
        num_regimes = int(row["num_regimes"])
        rng = np.random.default_rng(np.random.SeedSequence((seed, episode_seed)))
        rename = torch.as_tensor(rng.permutation(num_regimes), device=support_regime.device)
        support = rename[support_regime[batch_index]]
        query = rename[query_regime[batch_index]]
        if condition == "shuffled":
            support = support[torch.as_tensor(rng.permutation(len(support)), device=support_regime.device)]
            query = query[torch.as_tensor(rng.permutation(len(query)), device=query_regime.device)]
        tags[batch_index] = torch.cat((support, query)).float()
    return tags


def _input_with_information(
    batch: dict[str, Any], condition: InformationCondition, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_x = batch["support_x"]
    query_x = batch["query_x"]
    tags = _tag_column(
        batch["support_regime"], batch["query_regime"], batch["metadata"], condition, seed=seed
    )
    if tags is None:
        return support_x, batch["support_y"], query_x
    support_rows = support_x.shape[1]
    support_x = torch.cat((support_x, tags[:, :support_rows, None]), dim=-1)
    query_x = torch.cat((query_x, tags[:, support_rows:, None]), dim=-1)
    return support_x, batch["support_y"], query_x


def _summary(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "episodes": len(group),
                "query_cross_entropy": float(np.mean([row["query_cross_entropy"] for row in group])),
                "query_accuracy": float(np.mean([row["query_accuracy"] for row in group])),
                "query_auc": (
                    float(np.nanmean([row["query_auc"] for row in group]))
                    if not np.all(np.isnan([row["query_auc"] for row in group]))
                    else float("nan")
                ),
            }
        )
    return result


def _paired_differences(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    """Return cell-stratified paired differences between information conditions."""
    by_episode: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_episode[(int(row["episode_id"]), str(row["task_family"]))][str(row["information"])] = row
    output: list[dict[str, Any]] = []
    for information, reference in (("shuffled", "hidden"), ("true", "hidden"), ("true", "shuffled")):
        grouped: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for (_, family), values in by_episode.items():
            if reference in values and information in values:
                grouped[family][int(values[information]["cell_id"])].append(
                    (
                        values[information]["query_cross_entropy"] - values[reference]["query_cross_entropy"],
                        values[information]["query_accuracy"] - values[reference]["query_accuracy"],
                    )
                )
        for family, by_cell in sorted(grouped.items()):
            differences = [difference for values in by_cell.values() for difference in values]
            ce, accuracy = np.asarray(differences, dtype=float).T
            family_seed = sum((index + 1) * ord(character) for index, character in enumerate(family))
            rng = np.random.default_rng(np.random.SeedSequence((seed, family_seed, len(output))))
            ce_draws = np.empty(replicates, dtype=float)
            accuracy_draws = np.empty(replicates, dtype=float)
            cells = [np.asarray(values, dtype=float) for _, values in sorted(by_cell.items())]
            for draw in range(replicates):
                resampled_means = [values[rng.integers(len(values), size=len(values))].mean(axis=0) for values in cells]
                ce_draws[draw], accuracy_draws[draw] = np.mean(resampled_means, axis=0)
            output.append(
                {
                    "information": information,
                    "reference": reference,
                    "task_family": family,
                    "episodes": len(differences),
                    "cells": len(cells),
                    "cross_entropy_difference": float(ce.mean()),
                    "cross_entropy_difference_95_interval": [
                        float(np.quantile(ce_draws, 0.025)),
                        float(np.quantile(ce_draws, 0.975)),
                    ],
                    "accuracy_difference": float(accuracy.mean()),
                    "accuracy_difference_95_interval": [
                        float(np.quantile(accuracy_draws, 0.025)),
                        float(np.quantile(accuracy_draws, 0.975)),
                    ],
                }
            )
    return output


def evaluate(
    model: torch.nn.Module,
    bank_path: str | Path,
    *,
    device: str,
    information: tuple[InformationCondition, ...],
    multiclass_only: bool,
    multiregime_only: bool,
    episode_in_cell: int | None,
    cell_subsample_modulus: int | None,
    cell_subsample_remainder: int,
    max_episodes: int | None,
    seed: int,
    max_episodes_per_forward: int | None,
    with_auc: bool,
    query_regime_metrics: bool,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    selected = 0
    with MultiregimeV4EvaluationBank(bank_path, device=device) as bank, torch.no_grad():
        if episode_in_cell is not None and not 0 <= episode_in_cell < bank.episodes_per_cell:
            raise ValueError(
                f"episode_in_cell must lie in [0, {bank.episodes_per_cell - 1}] for this evaluation bank"
            )
        if cell_subsample_modulus is not None and cell_subsample_modulus < 1:
            raise ValueError("cell_subsample_modulus must be positive")
        if cell_subsample_modulus is not None and not 0 <= cell_subsample_remainder < cell_subsample_modulus:
            raise ValueError("cell_subsample_remainder must lie in [0, cell_subsample_modulus)")
        for cell_start in range(0, len(bank), bank.episodes_per_cell):
            cell_id = cell_start // bank.episodes_per_cell
            if cell_subsample_modulus is not None:
                cell_draw = int(np.random.SeedSequence((seed, cell_id)).generate_state(1, dtype=np.uint32)[0])
                if cell_draw % cell_subsample_modulus != cell_subsample_remainder:
                    continue
            start = cell_start if episode_in_cell is None else cell_start + episode_in_cell
            cell_limit = bank.episodes_per_cell if episode_in_cell is None else 1
            cell = bank.cell_batch(start, max_episodes=cell_limit)
            cell_metadata = cell["metadata"]
            first = cell_metadata[0]
            eligible = (
                int(first["num_regimes"]) >= 2
                and (not multiregime_only or str(first["rule_mode"]) == "multiregime")
                and (not multiclass_only or int(first["num_classes"]) >= 3)
            )
            if not eligible:
                continue
            if max_episodes is not None and selected >= max_episodes:
                break
            if max_episodes is not None and selected + len(cell_metadata) > max_episodes:
                cell = bank.cell_batch(cell_start, max_episodes=max_episodes - selected)
                cell_metadata = cell["metadata"]
            batch_size = len(cell_metadata) if max_episodes_per_forward is None else max_episodes_per_forward
            for start in range(0, len(cell_metadata), batch_size):
                stop = min(start + batch_size, len(cell_metadata))
                subbatch = {
                    key: value[start:stop] if isinstance(value, torch.Tensor) else value[start:stop]
                    for key, value in cell.items()
                }
                for condition in information:
                    support_x, support_y, query_x = _input_with_information(subbatch, condition, seed=seed)
                    logits = model(support_x, support_y, query_x)
                    target = subbatch["query_y"]
                    per_query_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
                    ).reshape_as(target)
                    probabilities = logits.softmax(dim=-1) if with_auc else None
                    for index, metadata in enumerate(subbatch["metadata"]):
                        classes = int(metadata["num_classes"])
                        query_by_regime = None
                        if query_regime_metrics:
                            query_regimes = subbatch["query_regime"][index]
                            predictions = logits[index].argmax(dim=-1)
                            query_by_regime = []
                            for regime_id in torch.unique(query_regimes, sorted=True):
                                mask = query_regimes == regime_id
                                query_by_regime.append(
                                    {
                                        "regime_id": int(regime_id.item()),
                                        "queries": int(mask.sum().item()),
                                        "query_cross_entropy": float(per_query_loss[index][mask].mean().cpu()),
                                        "query_accuracy": float((predictions[mask] == target[index][mask]).float().mean().cpu()),
                                    }
                                )
                        rows.append(
                            {
                                **metadata,
                                "information": condition,
                                "query_cross_entropy": float(per_query_loss[index].mean().cpu()),
                                "query_accuracy": float((logits[index].argmax(dim=-1) == target[index]).float().mean().cpu()),
                                **({"query_by_regime": query_by_regime} if query_by_regime is not None else {}),
                                "query_auc": (
                                    _episode_auc(probabilities[index], target[index], classes)
                                    if with_auc
                                    else float("nan")
                                ),
                            }
                        )
            selected += len(cell_metadata)
    model.train(was_training)
    return {
        "episodes_per_condition": selected,
        "information_conditions": list(information),
        "multiclass_only": multiclass_only,
        "multiregime_only": multiregime_only,
        "episode_in_cell": episode_in_cell,
        "cell_subsample_modulus": cell_subsample_modulus,
        "cell_subsample_remainder": cell_subsample_remainder if cell_subsample_modulus is not None else None,
        "with_auc": with_auc,
        "query_regime_metrics": query_regime_metrics,
        "tag_encoding": "one scalar tag column; identifiers randomly relabelled per episode",
        "shuffle_control": "tag values shuffled independently within support and query rows",
        "per_episode": rows,
        "overall": _summary(rows, ("information",)),
        "by_task_family": _summary(rows, ("information", "task_family")),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "scope": "finite-episode uncertainty only; does not estimate pretraining-seed uncertainty",
        },
        "paired_differences": _paired_differences(rows, replicates=bootstrap_replicates, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-source-path",
        help="load NanoTabPFNModel from the Python file used to train this checkpoint",
    )
    parser.add_argument(
        "--reference-hidden-report",
        help="verify hidden-condition scores against this checkpoint's saved native evaluation report",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multiclass-only", action="store_true")
    parser.add_argument(
        "--include-shared-rule",
        action="store_true",
        help="also score K>=2 shared-rule controls; the default isolates actual multiregime label selection",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-in-cell",
        type=int,
        default=None,
        help="evaluate one fixed episode index from every factorial cell; useful for a balanced pilot",
    )
    parser.add_argument(
        "--cell-subsample-modulus",
        type=int,
        default=None,
        help="retain a deterministic 1/N sample of factorial cells, while keeping every episode in retained cells",
    )
    parser.add_argument("--cell-subsample-remainder", type=int, default=0)
    parser.add_argument("--max-episodes-per-forward", type=int, default=None)
    parser.add_argument(
        "--with-auc",
        action="store_true",
        help="also calculate per-episode AUC; disabled by default because CE and accuracy are the diagnostic metrics",
    )
    parser.add_argument(
        "--query-regime-metrics",
        action="store_true",
        help="include per-query CE and accuracy grouped by the realized query regime",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--information", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    if args.max_episodes_per_forward is not None and args.max_episodes_per_forward < 1:
        raise ValueError("--max-episodes-per-forward must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_class = NanoTabPFNModel
    model_source = Path(inspect.getfile(NanoTabPFNModel)).resolve()
    if args.model_source_path is not None:
        model_source = Path(args.model_source_path).resolve()
        spec = importlib.util.spec_from_file_location("nanotabpfn_checkpoint_source", model_source)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load model source: {model_source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_class = module.NanoTabPFNModel
    model = model_class(**state["architecture"]).to(args.device)
    model.load_state_dict(state["model"])
    report = evaluate(
        model,
        args.bank,
        device=args.device,
        information=tuple(args.information),
        multiclass_only=args.multiclass_only,
        multiregime_only=not args.include_shared_rule,
        episode_in_cell=args.episode_in_cell,
        cell_subsample_modulus=args.cell_subsample_modulus,
        cell_subsample_remainder=args.cell_subsample_remainder,
        max_episodes=args.max_episodes,
        seed=args.seed,
        max_episodes_per_forward=args.max_episodes_per_forward,
        with_auc=args.with_auc,
        query_regime_metrics=args.query_regime_metrics,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    report.update({
        "source_checkpoint": str(args.checkpoint),
        "source_step": int(state["step"]),
        "bank_path": str(Path(args.bank).resolve()),
        "model_source_path": str(model_source),
        "model_source_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
    })
    if args.reference_hidden_report is not None:
        reference = json.loads(Path(args.reference_hidden_report).read_text())
        hidden = [row for row in report["per_episode"] if row["information"] == "hidden"]
        if not hidden:
            raise ValueError("Hidden-reference verification requires hidden-condition episodes")
        if "per_episode" in reference:
            by_episode = {int(row["episode_id"]): row for row in reference["per_episode"]}
            comparisons = [(row, by_episode[int(row["episode_id"])]) for row in hidden]
            reference_aggregation = "episode"
        elif "by_cell" in reference:
            # Intermediate reports may retain only factorial-cell summaries.
            # Compare complete cells; a partial cell is not the same reference.
            def cell_key(row):
                return tuple(row[field] for field in FACTORS)

            by_cell = {cell_key(row): row for row in reference["by_cell"]}
            comparisons = []
            for row in _summary(hidden, FACTORS):
                expected = by_cell[cell_key(row)]
                if row["episodes"] != expected["episodes"]:
                    raise ValueError(
                        f"Cell-reference verification needs all {expected['episodes']} episodes "
                        f"in cell {cell_key(row)}, received {row['episodes']}"
                    )
                comparisons.append((row, expected))
            reference_aggregation = "cell"
        else:
            raise ValueError("Reference report must contain per_episode or by_cell scores")
        max_ce_error = 0.0
        for row, expected in comparisons:
            for metric in ("query_cross_entropy", "query_accuracy"):
                if not np.isclose(row[metric], expected[metric], rtol=1e-4, atol=1e-5):
                    raise ValueError(
                        f"Checkpoint compatibility check failed: "
                        f"reference={reference_aggregation} episode={row.get('episode_id', 'cell mean')} "
                        f"{metric}={row[metric]}, saved={expected[metric]}"
                    )
            max_ce_error = max(max_ce_error, abs(row["query_cross_entropy"] - expected["query_cross_entropy"]))
        report["hidden_reference_validation"] = {
            "reference_path": str(args.reference_hidden_report),
            "episodes_checked": len(hidden),
            "reference_aggregation": reference_aggregation,
            "reference_units_checked": len(comparisons),
            "max_cross_entropy_absolute_error": max_ce_error,
            "passed": True,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["by_task_family"]:
        print(
            f"{row['information']:8s} {row['task_family']:10s} n={row['episodes']:5d} "
            f"CE={row['query_cross_entropy']:.5f} acc={100 * row['query_accuracy']:.2f}%"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
