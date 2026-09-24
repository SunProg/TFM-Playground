"""Extract per-cell v4 validation/test metrics from pretrain_plain_nanotabpfn runs.

Walks ``<base>/<job_id>/<family>-<size>/seed-<seed>/v4_validation/step-*.json``
(and, if present, ``v4_test/final.json``) for every family/size combination and
aggregates each report's ``per_episode`` rows down to ``(mechanism_mode,
rule_mode, num_regimes, num_classes, support_size, num_features,
effective_is_causal)`` cells (plain per-episode mean; each row already
weights one episode). Aggregating from ``per_episode`` rather than the
report's own ``by_cell`` is required to condition on ``effective_is_causal``,
since ``by_cell`` does not carry it. Output is one compact JSON file consumed
by ``plot_v4_mechanism_rule_grid.py``.

Run this where the run directories are visible (e.g. via ssh on the cluster,
or against a locally mounted/copied runs tree):

    python scripts/extract_v4_validation_cells.py \\
        --base /scratch/users/<user>/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale \\
        --job small=37283915 --job medium=37283921 --job large=37283995 \\
        --output cells_history.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

FAMILIES = ["original", "r_z-fixed", "r_z-curriculum", "g_z-fixed", "g_z-curriculum"]
CELL_KEYS = (
    "mechanism_mode",
    "rule_mode",
    "num_regimes",
    "num_classes",
    "support_size",
    "num_features",
    "effective_is_causal",
)
METRIC_KEYS = ("query_cross_entropy", "query_accuracy", "query_auc", "excess_cross_entropy", "accuracy_gain")


def _attach_prior_baselines(episodes: list[dict], baselines: dict[str, dict] | None) -> None:
    """Add excess_cross_entropy / accuracy_gain per episode from precomputed class-prior baselines.

    ``baselines`` is the ``episodes`` mapping written by
    ``compute_v4_bank_prior_baselines.py`` (keyed by episode id). Raw CE and
    accuracy have different floors across class_ratio / num_classes cells, so
    the prior-subtracted versions are the ones that are comparable across cells.
    """
    if not baselines:
        return
    for episode in episodes:
        base = baselines.get(str(episode.get("episode_id")))
        if base is None:
            continue
        episode["excess_cross_entropy"] = episode["query_cross_entropy"] - base["prior_cross_entropy"]
        episode["accuracy_gain"] = episode["query_accuracy"] - base["majority_accuracy"]


def _aggregate_episodes(episodes: list[dict]) -> dict[tuple, dict[str, float]]:
    """Per-episode mean of METRIC_KEYS grouped by CELL_KEYS (NaN-safe, e.g. missing/degenerate AUC).

    Each result also carries ``episodes``, the raw episode count for that
    exact cell. This is required for correctly weighting any *further*
    marginalization downstream (e.g. averaging several cells together to get
    a coarser breakdown) — a plain mean across cells of unequal size is
    biased; the fix is a weighted mean using these counts, not a mean of
    means.
    """
    agg: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple, int] = defaultdict(int)
    for episode in episodes:
        key = tuple(episode[k] for k in CELL_KEYS)
        counts[key] += 1
        for metric in METRIC_KEYS:
            value = episode.get(metric)  # older reports (pre-AUC) simply lack query_auc
            if value is not None and value == value:  # excludes NaN
                agg[key][metric].append(value)
    return {
        key: {
            **{metric: sum(values) / len(values) for metric, values in metrics.items() if values},
            "episodes": counts[key],
        }
        for key, metrics in agg.items()
    }


def _load_report(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None  # tolerate a file still being written


def _cell_records(report: dict, baselines: dict[str, dict] | None, **extra) -> list[dict]:
    episodes = report.get("per_episode", [])
    _attach_prior_baselines(episodes, baselines)
    return [{**extra, **dict(zip(CELL_KEYS, key)), **metrics} for key, metrics in _aggregate_episodes(episodes).items()]


def extract_history(
    base: str,
    jobs: dict[str, str],
    seed: int,
    baselines: dict[str, dict] | None = None,
    eval_base: str | None = None,
) -> dict[str, list[dict]]:
    """Per-step cell metrics from the training runs' periodic v4_validation reports.

    ``eval_base`` optionally points at the offline-evaluation tree (same
    ``<job>/<task>/seed-<seed>`` layout) holding ``v4_validation/step-000000.json``
    from ``evaluate_init_on_v4_bank.py``. The step-0 model depends only on the
    size's architecture and seed, so one such report is shared by all five
    families of that size.
    """
    out: dict[str, list[dict]] = {}
    for size, job_id in jobs.items():
        init_records: list[dict] = []
        if eval_base is not None:
            init_paths = glob.glob(f"{eval_base}/{job_id}/*-{size}/seed-{seed}/v4_validation/step-000000.json")
            init_report = _load_report(init_paths[0]) if init_paths else None
            if init_report is not None:
                init_records = _cell_records(init_report, baselines, step=0)
        for family in FAMILIES:
            task = f"{family}-{size}"
            pattern = f"{base}/{job_id}/{task}/seed-{seed}/v4_validation/step-*.json"
            records = [dict(r) for r in init_records]
            for path in sorted(glob.glob(pattern)):
                try:
                    with open(path) as f:
                        report = json.load(f)
                except Exception:
                    continue  # tolerate a checkpoint file still being written
                step = int(os.path.basename(path).split("-")[1].split(".")[0])
                episodes = report.get("per_episode", [])
                _attach_prior_baselines(episodes, baselines)
                for key, metrics in _aggregate_episodes(episodes).items():
                    records.append({"step": step, **dict(zip(CELL_KEYS, key)), **metrics})
            if records:
                out[task] = records
    return out


def extract_test(
    base: str, jobs: dict[str, str], seed: int, baselines: dict[str, dict] | None = None
) -> dict[str, list[dict]]:
    """Same cell aggregation, but for the one-shot final v4_test report (if a task finished)."""
    out: dict[str, list[dict]] = {}
    for size, job_id in jobs.items():
        for family in FAMILIES:
            task = f"{family}-{size}"
            report = _load_report(f"{base}/{job_id}/{task}/seed-{seed}/v4_test/final.json")
            if report is not None:
                out[task] = _cell_records(report, baselines)
    return out


def extract_best_val_test(
    eval_base: str, jobs: dict[str, str], seed: int, baselines: dict[str, dict] | None = None
) -> dict[str, list[dict]]:
    """Test cells for the best-own-validation checkpoint (``v4_test/best_val_stepXXXXXX.json``
    written by ``evaluate_checkpoint_on_v4_bank.py``); each record carries that ``step``."""
    out: dict[str, list[dict]] = {}
    for size, job_id in jobs.items():
        for family in FAMILIES:
            task = f"{family}-{size}"
            paths = sorted(glob.glob(f"{eval_base}/{job_id}/{task}/seed-{seed}/v4_test/best_val_step*.json"))
            if not paths:
                continue
            report = _load_report(paths[-1])
            if report is None:
                continue
            step = int(os.path.basename(paths[-1]).removeprefix("best_val_step").split(".")[0])
            out[task] = _cell_records(report, baselines, step=step)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Directory holding <job_id>/<family>-<size>/... run dirs")
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        metavar="SIZE=JOB_ID",
        help="Repeatable, e.g. --job small=37283915 --job medium=37283921 --job large=37283995",
    )
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help=f"family names to look for (task = <family>-<size>); default {FAMILIES}",
    )
    parser.add_argument("--output", required=True, help="Validation-history output JSON path")
    parser.add_argument(
        "--test-output",
        default=None,
        help="Optional output JSON path for final v4_test cell metrics (only tasks that completed)",
    )
    parser.add_argument("--prior-baselines", default=None, help="compute_v4_bank_prior_baselines.py output for the validation bank")
    parser.add_argument("--test-prior-baselines", default=None, help="same, for the test bank")
    parser.add_argument(
        "--eval-base",
        default=None,
        help="Offline-evaluation tree (runs_eval/...) with step-000000.json init reports and best_val_step*.json test reports",
    )
    parser.add_argument("--best-val-test-output", default=None, help="Output JSON for best-own-val checkpoint test cells")
    args = parser.parse_args()

    def _load_baselines(path):
        if not path:
            return None
        with open(path) as f:
            return json.load(f)["episodes"]

    val_baselines = _load_baselines(args.prior_baselines)
    test_baselines = _load_baselines(args.test_prior_baselines)

    jobs = dict(item.split("=", 1) for item in args.job)
    if args.families:
        FAMILIES[:] = args.families

    history = extract_history(args.base, jobs, args.seed, val_baselines, args.eval_base)
    with open(args.output, "w") as f:
        json.dump(history, f)
    print(f"wrote {sum(len(v) for v in history.values())} history records across {len(history)} tasks to {args.output}")

    if args.test_output:
        test = extract_test(args.base, jobs, args.seed, test_baselines)
        with open(args.test_output, "w") as f:
            json.dump(test, f)
        print(f"wrote test-set cells for {len(test)} completed tasks to {args.test_output}")

    if args.best_val_test_output:
        if args.eval_base is None:
            parser.error("--best-val-test-output requires --eval-base")
        best = extract_best_val_test(args.eval_base, jobs, args.seed, test_baselines)
        with open(args.best_val_test_output, "w") as f:
            json.dump(best, f)
        print(f"wrote best-val test cells for {len(best)} tasks to {args.best_val_test_output}")


if __name__ == "__main__":
    main()
