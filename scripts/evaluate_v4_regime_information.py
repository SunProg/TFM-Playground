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
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.experiments.multiregime_v4_evaluation import MultiregimeV4EvaluationBank, _episode_auc
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
                "query_auc": float(np.nanmean([row["query_auc"] for row in group])),
            }
        )
    return result


def _paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return true/shuffled-minus-hidden episode-paired differences by family."""
    by_episode: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_episode[(int(row["episode_id"]), str(row["task_family"]))][str(row["information"])] = row
    output: list[dict[str, Any]] = []
    for information in ("shuffled", "true"):
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (_, family), values in by_episode.items():
            if "hidden" in values and information in values:
                grouped[family].append(
                    (
                        values[information]["query_cross_entropy"] - values["hidden"]["query_cross_entropy"],
                        values[information]["query_accuracy"] - values["hidden"]["query_accuracy"],
                    )
                )
        for family, differences in sorted(grouped.items()):
            ce, accuracy = np.asarray(differences, dtype=float).T
            output.append(
                {
                    "information": information,
                    "reference": "hidden",
                    "task_family": family,
                    "episodes": len(differences),
                    "cross_entropy_difference": float(ce.mean()),
                    "accuracy_difference": float(accuracy.mean()),
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
    max_episodes: int | None,
    seed: int,
    max_episodes_per_forward: int | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    selected = 0
    with MultiregimeV4EvaluationBank(bank_path, device=device) as bank, torch.no_grad():
        for cell_start in range(0, len(bank), bank.episodes_per_cell):
            cell = bank.cell_batch(cell_start, max_episodes=bank.episodes_per_cell)
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
                    probabilities = logits.softmax(dim=-1)
                    for index, metadata in enumerate(subbatch["metadata"]):
                        classes = int(metadata["num_classes"])
                        rows.append(
                            {
                                **metadata,
                                "information": condition,
                                "query_cross_entropy": float(per_query_loss[index].mean().cpu()),
                                "query_accuracy": float((logits[index].argmax(dim=-1) == target[index]).float().mean().cpu()),
                                "query_auc": _episode_auc(probabilities[index], target[index], classes),
                            }
                        )
            selected += len(cell_metadata)
    model.train(was_training)
    return {
        "episodes_per_condition": selected,
        "information_conditions": list(information),
        "multiclass_only": multiclass_only,
        "multiregime_only": multiregime_only,
        "tag_encoding": "one scalar tag column; identifiers randomly relabelled per episode",
        "shuffle_control": "tag values shuffled independently within support and query rows",
        "per_episode": rows,
        "overall": _summary(rows, ("information",)),
        "by_task_family": _summary(rows, ("information", "task_family")),
        "paired_minus_hidden": _paired_differences(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multiclass-only", action="store_true")
    parser.add_argument(
        "--include-shared-rule",
        action="store_true",
        help="also score K>=2 shared-rule controls; the default isolates actual multiregime label selection",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-forward", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--information", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    if args.max_episodes_per_forward is not None and args.max_episodes_per_forward < 1:
        raise ValueError("--max-episodes-per-forward must be positive")

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = NanoTabPFNModel(**state["architecture"]).to(args.device)
    model.load_state_dict(state["model"])
    report = evaluate(
        model,
        args.bank,
        device=args.device,
        information=tuple(args.information),
        multiclass_only=args.multiclass_only,
        multiregime_only=not args.include_shared_rule,
        max_episodes=args.max_episodes,
        seed=args.seed,
        max_episodes_per_forward=args.max_episodes_per_forward,
    )
    report.update({"source_checkpoint": str(args.checkpoint), "source_step": int(state["step"])})
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
