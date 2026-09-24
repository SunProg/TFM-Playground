"""Cross-model table from per-model BeyondArena v4 result directories.

Each ``evaluate_beyondarena_v4_array_a30.sbatch`` task writes
``results/beyondarena-v4/<mode>-<jobid>_<task>/rankings.csv`` for one model
(NanoTabPFN run: ``final`` + ``best_own_val`` checkpoints; published baseline:
``published_default``). This script pools them and prints, per condition, one
row per (model, checkpoint policy) with excess CE / accuracy gain / macro OvR
AUC (task-equal-weighted scores from the evaluator). Stdlib only.

    python scripts/summarize_beyondarena_v4.py --root results/beyondarena-v4 --job 37371347 \\
        --conditions all binary multi IID --markdown
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

METRICS = ("excess_cross_entropy", "accuracy_gain", "macro_ovr_auc")
BASELINE_PREFIXES = ("tabpfn", "tabicl", "logreg", "rf", "hgb", "xgboost", "lightgbm", "catboost")


def result_dirs(root: str, jobs: str):
    """(mode, dir) for every per-model result dir of the comma-separated job ids."""
    for job in jobs.split(","):
        for d in sorted(glob.glob(os.path.join(root, f"*-{job}_*"))):
            if not os.path.exists(os.path.join(d, "fold_metrics.csv")):
                continue  # task still running
            yield os.path.basename(d).rsplit(f"-{job}_", 1)[0], d


def load(root: str, job: str) -> list[dict]:
    rows = []
    for mode, d in result_dirs(root, job):
        with open(os.path.join(d, "rankings.csv")) as f:
            for r in csv.DictReader(f):
                # baseline tasks also carry the dummy run root's nanotabpfn rows; keep only the baseline itself
                if mode.startswith(BASELINE_PREFIXES) and r["model_kind"] == "nanotabpfn":
                    continue
                r["mode"] = mode
                rows.append(r)
    return rows


def label(r: dict) -> str:
    if r["model_kind"] != "nanotabpfn":
        return r["model_identity"]
    return r["mode"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True, help="SLURM array job id(s), comma-separated, whose result dirs to pool")
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--policy", choices=("final", "best_own_val", "both"), default="both")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--exclude-prefix", nargs="*", default=[], help="drop models whose name starts with any of these (e.g. zx-)")
    parser.add_argument("--exclude-bucket", nargs="*", default=[], help="drop tasks carrying any of these bucket labels (e.g. Temporal)")
    parser.add_argument("--sort-by", choices=METRICS, default="excess_cross_entropy", help="metric whose mean rank orders the --rank table")
    parser.add_argument("--json-output", default=None, help="also dump {condition: {model: {...}}} (scores or mean ranks) to this file")
    parser.add_argument("--per-task", nargs="*", default=None, metavar="MODEL",
                        help="per-dataset table (folds averaged) for these models; empty list = all models")
    parser.add_argument("--metric", choices=METRICS, default="excess_cross_entropy", help="metric for --per-task")
    parser.add_argument("--per-task-json", default=None,
                        help="dump {metrics: {metric: {task: {model: fold-mean}}}, tasks: {task: meta}} for every model to this file")
    parser.add_argument(
        "--rank",
        action="store_true",
        help="cross-model ranking: per task (folds averaged) rank every model on each metric, then average ranks over tasks",
    )
    args = parser.parse_args()
    if args.rank:
        cross_model_rank(args)
        return
    if args.per_task_json:
        per_task_json(args)
        return
    if args.per_task is not None:
        per_task(args)
        return

    rows = load(args.root, args.job)
    dump: dict[str, dict] = {}
    for cond in args.conditions:
        table: dict[tuple[str, str], dict[str, str]] = {}
        for r in rows:
            if r["condition"] != cond:
                continue
            key = (label(r), r["checkpoint_policy"])
            table.setdefault(key, {"tasks": r["task_count"]})[r["metric"]] = r["score"]
        keys = [k for k in table if args.policy == "both" or k[1] in (args.policy, "published_default")]
        # order: nanotabpfn runs (as named), then baselines
        keys.sort(key=lambda k: (k[0].startswith(BASELINE_PREFIXES), k[0], k[1]))
        print(f"\n### condition = {cond}")
        if args.markdown:
            print("| model | checkpoint | tasks | excess CE | acc gain | macro AUC |")
            print("|---|---|---|---|---|---|")
        else:
            print(f"{'model':28s} {'ckpt':16s} {'tasks':>5s} {'excessCE':>9s} {'accGain':>9s} {'AUC':>7s}")
        for k in keys:
            v = table[k]
            vals = [float(v.get(m, "nan")) for m in METRICS]
            dump.setdefault(cond, {})[f"{k[0]}|{k[1]}"] = {"tasks": int(v["tasks"]), **dict(zip(METRICS, vals))}
            if args.markdown:
                print(f"| {k[0]} | {k[1]} | {v['tasks']} | {vals[0]:+.4f} | {vals[1]:+.4f} | {vals[2]:.4f} |")
            else:
                print(f"{k[0]:28s} {k[1]:16s} {v['tasks']:>5s} {vals[0]:+9.4f} {vals[1]:+9.4f} {vals[2]:7.4f}")
    if args.json_output:
        import json
        with open(args.json_output, "w") as f:
            json.dump(dump, f, indent=1)


def _fold_scores(args):
    """model -> task -> metric -> [fold values]; plus task metadata."""
    from collections import defaultdict
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta: dict[str, dict] = {}
    for mode, d in result_dirs(args.root, args.job):
        with open(os.path.join(d, "fold_metrics.csv")) as f:
            for r in csv.DictReader(f):
                if r.get("status") not in ("evaluated", "ok") or r.get("condition", "all") != "all":
                    continue
                if mode.startswith(BASELINE_PREFIXES):
                    if r["model_kind"] == "nanotabpfn":
                        continue
                    model = r["model_identity"]
                elif r["checkpoint_policy"] == (args.policy if args.policy != "both" else "final"):
                    model = mode
                else:
                    continue
                m0 = meta.setdefault(r["task_name"], {"classes": r["problem_bucket"], "rows": r["row_bucket"] or "-", "feat": r["feature_bucket"], "regime": r["regime"], "train_rows": r["train_rows"], "features": r["raw_train_features"], "majority": []})
                if r.get("majority_accuracy"):
                    m0["majority"].append(float(r["majority_accuracy"]))
                for m in METRICS:
                    if r[m] != "":
                        scores[model][r["task_name"]][m].append(float(r[m]))
    return scores, meta


def per_task_json(args) -> None:
    import json
    from statistics import mean
    scores, meta = _fold_scores(args)
    out = {"metrics": {m: {} for m in METRICS}, "tasks": {}}
    for t, md in meta.items():
        out["tasks"][t] = {**{k: v for k, v in md.items() if k != "majority"}, "majority": mean(md["majority"]) if md["majority"] else None}
        for m in METRICS:
            out["metrics"][m][t] = {model: mean(scores[model][t][m]) for model in scores if scores[model][t][m]}
    with open(args.per_task_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(scores)} models, {len(meta)} tasks -> {args.per_task_json}")


def per_task(args) -> None:
    from statistics import mean
    scores, meta = _fold_scores(args)
    models = args.per_task or sorted(scores)
    missing = [m for m in models if m not in scores]
    if missing:
        raise SystemExit(f"unknown models: {missing}; available: {sorted(scores)}")
    tasks = sorted(meta, key=lambda t: (meta[t]["classes"], meta[t]["regime"], t))
    if args.exclude_bucket:
        tasks = [t for t in tasks if not (set(args.exclude_bucket) & {meta[t]["classes"], meta[t]["rows"], meta[t]["feat"], meta[t]["regime"]})]
    label = {"excess_cross_entropy": "excess CE", "accuracy_gain": "accuracy gain", "macro_ovr_auc": "macro AUC"}[args.metric]
    fmt = (lambda v: f"{v:.3f}") if args.metric == "macro_ovr_auc" else (lambda v: f"{v:+.3f}")
    print(f"\n### per dataset — {label} (folds averaged; final checkpoints)")
    print("| task | classes | regime | train rows | feats | majority frac | " + " | ".join(models) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(models))
    for t in tasks:
        cells = []
        for m in models:
            v = scores[m][t][args.metric]
            cells.append(fmt(mean(v)) if v else "-")
        md = meta[t]
        maj = f"{mean(md['majority']):.2f}" if md["majority"] else "-"
        print(f"| {t} | {md['classes']} | {md['regime']} | {md['train_rows']} | {md['features']} | {maj} | " + " | ".join(cells) + " |")


def cross_model_rank(args) -> None:
    """Rank all models against each other on the same tasks.

    The per-run rankings.csv only ranks a run's own checkpoints, so this
    re-derives ranks from fold_metrics.csv: task score = mean over folds
    (status == evaluated), models ranked per task per metric (1 = best; lower excess
    CE, higher accuracy gain / AUC), then mean rank over the tasks of the
    condition. Only tasks scored by every model are used.
    """
    from collections import defaultdict
    from statistics import mean

    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    task_bucket: dict[str, set[str]] = {}  # task -> its bucket labels (problem, rows, features, regime)
    for mode, d in result_dirs(args.root, args.job):
        with open(os.path.join(d, "fold_metrics.csv")) as f:
            for r in csv.DictReader(f):
                # one row per (fold, condition label); keep the fold once via the "all" label
                if r.get("status") not in ("evaluated", "ok") or r.get("condition", "all") != "all":
                    continue
                if mode.startswith(BASELINE_PREFIXES):
                    if r["model_kind"] == "nanotabpfn":
                        continue
                    model = r["model_identity"]
                elif r["checkpoint_policy"] == (args.policy if args.policy != "both" else "final"):
                    model = mode
                else:
                    continue
                task_bucket[r["task_name"]] = {r["problem_bucket"], r["row_bucket"] or "unbucketed_rows", r["feature_bucket"], r["regime"]}
                for m in METRICS:
                    if r[m] != "":  # AUC can be blank on degenerate folds
                        scores[model][r["task_name"]][m].append(float(r[m]))
    models = sorted(m for m in scores if not any(m.startswith(p) for p in args.exclude_prefix))
    if not models:
        raise SystemExit("no fold_metrics rows found")
    dump: dict[str, dict] = {}
    common = sorted(set(scores[models[0]]).intersection(*(set(scores[m]) for m in models[1:])))
    for cond in args.conditions:
        # a condition is "all" or one or more bucket labels joined by "__" (e.g. binary__small_rows, multi__IID)
        wanted = set() if cond == "all" else set(cond.split("__"))
        tasks = [t for t in common if wanted <= task_bucket[t] and not (set(args.exclude_bucket) & task_bucket[t])]
        if not tasks:
            print(f"\n### condition = {cond}: no tasks"); continue
        ranks: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for t in tasks:
            for m in METRICS:
                vals = {model: mean(scores[model][t][m]) for model in models if scores[model][t][m]}
                if len(vals) < len(models):
                    continue
                order = sorted(vals, key=lambda k: vals[k] if m == "excess_cross_entropy" else -vals[k])
                for i, model in enumerate(order, start=1):
                    ranks[model][m].append(i)
        table = sorted(models, key=lambda k: mean(ranks[k][args.sort_by]))
        print(f"\n### cross-model mean rank, condition = {cond} ({len(tasks)} tasks, {len(models)} models; 1 = best)")
        hdr = "| model | rank excess CE | rank acc gain | rank AUC |"
        print(hdr + "\n|---|---|---|---|" if args.markdown else f"{'model':28s} {'rk excessCE':>12s} {'rk accGain':>11s} {'rk AUC':>8s}")
        for model in table:
            r = [mean(ranks[model][m]) for m in METRICS]
            dump.setdefault(cond, {})[model] = {"tasks": len(tasks), **{m: r[i] for i, m in enumerate(METRICS)}}
            if args.markdown:
                print(f"| {model} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} |")
            else:
                print(f"{model:28s} {r[0]:12.1f} {r[1]:11.1f} {r[2]:8.1f}")
    if args.json_output:
        import json
        with open(args.json_output, "w") as f:
            json.dump(dump, f, indent=1)


if __name__ == "__main__":
    main()
