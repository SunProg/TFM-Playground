"""Cross-fitted residual scoring for latent regimes in the TabArena tables.

Every regime result in this project so far is measured on a synthetic prior
whose contamination we planted ourselves.  That leaves the prior question
unanswered: do *real* tables carry more than one label function at all?  If
they do not, an architecture that discovers regimes has nothing to discover
outside the synthetic setting, and the slot work is a study of a prior.

The detector is the conventional one and deliberately not a model from this
repo: cross-fit a strong learner on ``(x, y)``, score each row by its
out-of-fold loss, and ask whether the rows the majority rule gets wrong form a
*covariate-dependent* group rather than noise.  Cross-fitting is not optional --
an in-fold fit memorizes the hard rows and every score below becomes vacuous.

Three statistics, in increasing strength:

``residual_gap``
    Mean out-of-fold loss of the hardest ``hard_quantile`` of rows minus the
    rest.  Always positive by construction; it calibrates the other two.
``gate_auc``
    Can a second learner predict *from the features alone* which rows the
    majority rule gets wrong?  This is the covariate-dependent gating that
    finite mixtures of regressions need to be identifiable at all.  At 0.5 the
    hard rows are label noise, not a regime, and no mixture model can help.
    Cross-fitted: an in-fold gate memorizes its own targets and reports 1.0 on
    pure label noise, which is how this probe first read.  Note that a high
    ``gate_auc`` is necessary and not sufficient: on a single-regime table the
    hard rows are the ones near the decision boundary, which are perfectly
    covariate-predictable (measured 0.85 on a planted single-regime table).
    ``routed_gain`` is the statistic that separates the cases.
``routed_gain``
    Held-out log-loss improvement of a two-expert routed predictor over the
    pooled model.  The experts, the gate and the routing threshold are all fit
    inside the training half of an outer split and scored on the untouched
    half, so a table that merely has noisy rows scores zero rather than
    positive.  This is what a 1990s mixture of regressions would buy here.

    The experts are **linear**, and that is the point rather than a shortcut.
    Heterogeneity is only ever defined against a model class: a gradient
    boosting pooled model absorbs a covariate-gated regime into ``p(y | x)``
    directly, leaving routing nothing to recover, and the measured gain is
    then ~0 whether or not the table has regimes.  Latent-class regression
    asks the answerable question instead -- do the *coefficients* differ
    across a partition of the rows?  ``pooled_hgb_log_loss`` is reported
    alongside so the linear model's headroom is visible.

A positive ``routed_gain`` with ``gate_auc`` near 0.5 is contradictory and
means the outer split leaked; the two are reported together for that reason.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from openml.config import set_root_cache_directory
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

from tfmplayground.experiments.evaluate_tabarena_small import SmallTabArenaConfig, _subsample, eligible_tasks
from tfmplayground.interface import get_feature_preprocessor


@dataclass
class ResidualProbeConfig:
    output_dir: str = "results/tabarena_residual_probe"
    max_predictors: int = 30
    subsample: int = 2_048
    #: Folds for the inner cross-fit that produces the out-of-fold losses.
    inner_folds: int = 5
    #: Outer splits.  Everything -- experts, gate, threshold -- is fit inside
    #: the training half of one of these and scored on the other half.
    outer_folds: int = 5
    #: Fraction of rows treated as the minority candidate.  0.3 matches the
    #: synthetic prior's contamination so the two settings are comparable.
    hard_quantile: float = 0.3
    #: Expert families to sweep.  See ``EXPERT_FAMILIES``.
    expert_families: tuple[str, ...] = ("linear", "shallow", "hgb")
    seed: int = 2402
    cache_directory: str | None = None
    max_iter: int = 200
    max_leaf_nodes: int = 15


#: Expert families, ordered by capacity.  Heterogeneity is only ever defined
#: against a model class -- a rule a flexible learner can express itself is not
#: a second regime to that learner -- so the family is the axis this probe
#: sweeps rather than a setting it fixes.  ``hgb`` doubles as the flexible
#: reference each family is reported against.
EXPERT_FAMILIES = ("linear", "shallow", "hgb")

#: The capacity-matched pooled control for each family.  Two experts plus a
#: gate hold more parameters than one expert, so a routed predictor beats its
#: own pooled model whenever that model is *underfitting* -- no second rule
#: required.  Measured: the ``shallow`` family scored +0.089 on a table with a
#: single planted rule, purely on capacity.  ``routed_gain_matched`` scores the
#: routed predictor against one pooled model of comparable capacity instead,
#: and is the statistic to read; ``routed_gain`` is kept beside it because the
#: gap between the two *is* the capacity effect.
_MATCHED = {"linear": "linear_matched", "shallow": "shallow_matched", "hgb": "hgb_matched"}


def _fit(x: np.ndarray, y: np.ndarray, config: ResidualProbeConfig, family: str = "linear"):
    """A fitted model, or the constant to use when one class is absent."""
    if family not in EXPERT_FAMILIES and family not in set(_MATCHED.values()):
        raise ValueError(f"family must be one of {EXPERT_FAMILIES}, got {family!r}.")
    if np.unique(y).size < 2:
        return float((y.sum() + 1) / (len(y) + 2))
    if family == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, random_state=config.seed
        ).fit(x, y)
    if family == "shallow":
        # Enough capacity for a curved boundary, not enough to absorb a second
        # rule: the case where routing should pay if regimes exist at all.
        return HistGradientBoostingClassifier(
            max_iter=30, max_leaf_nodes=4, random_state=config.seed
        ).fit(x, y)
    if family == "linear_matched":
        # Degree-2 features: what a two-expert linear mixture can already
        # express with one model.  See ``_MATCHED``.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=config.seed),
            ).fit(x, y)
    if family == "shallow_matched":
        return HistGradientBoostingClassifier(
            max_iter=60, max_leaf_nodes=8, random_state=config.seed
        ).fit(x, y)
    if family == "hgb_matched":
        return HistGradientBoostingClassifier(
            max_iter=2 * config.max_iter, max_leaf_nodes=2 * config.max_leaf_nodes, random_state=config.seed
        ).fit(x, y)
    with warnings.catch_warnings():
        # scipy raises OptimizeWarning about an `iprint` option lbfgs no longer
        # reads; it is a version skew between sklearn and scipy, not a fit
        # problem, and it would otherwise print once per fold per table.
        warnings.simplefilter("ignore")
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.seed)
        ).fit(x, y)


def _predict(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(x), model)
    return model.predict_proba(x)[:, 1]


def _row_loss(probability: np.ndarray, y: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))


def cross_fitted_probability(
    x: np.ndarray, y: np.ndarray, config: ResidualProbeConfig, family: str = "linear"
) -> np.ndarray:
    """Out-of-fold probability for every row.  Never an in-fold prediction."""
    probability = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=config.inner_folds, shuffle=True, random_state=config.seed)
    for train_index, test_index in splitter.split(x, y):
        model = _fit(x[train_index], y[train_index], config, family)
        probability[test_index] = _predict(model, x[test_index])
    return probability


def cross_fitted_losses(
    x: np.ndarray, y: np.ndarray, config: ResidualProbeConfig, family: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probability and per-row loss for every row."""
    probability = cross_fitted_probability(x, y, config, family)
    return probability, _row_loss(probability, y)


def probe_table(
    x: np.ndarray, y: np.ndarray, config: ResidualProbeConfig, family: str = "linear"
) -> dict[str, Any]:
    """Score one table for a second regime, as seen by one expert family."""
    outer = StratifiedKFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)
    pooled_losses, routed_losses, hgb_losses, matched_losses = [], [], [], []
    gate_aucs, gaps, hard_rates = [], [], []
    for train_index, test_index in outer.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        # Everything below is fit on the training half only.
        _, train_loss = cross_fitted_losses(x_train, y_train, config, family)
        threshold = float(np.quantile(train_loss, 1.0 - config.hard_quantile))
        hard = train_loss > threshold
        gaps.append(float(train_loss[hard].mean() - train_loss[~hard].mean()) if hard.any() else 0.0)
        hard_rates.append(float(hard.mean()))
        if hard.sum() < 10 or (~hard).sum() < 10 or np.unique(hard).size < 2:
            continue

        # Is "hard" predictable from the features?  A gate that cannot be
        # learned from x is noise, whatever its residuals look like.  Scored
        # out of fold: an in-fold gate reports 1.0 even on random labels.
        gate_oof = cross_fitted_probability(x_train, hard.astype(int), config, "hgb")
        gate_aucs.append(float(roc_auc_score(hard.astype(int), gate_oof)))
        gate = _fit(x_train, hard.astype(int), config, "hgb")
        gate_test = _predict(gate, x_test)

        majority = _fit(x_train[~hard], y_train[~hard], config, family)
        minority = _fit(x_train[hard], y_train[hard], config, family)
        pooled = _fit(x_train, y_train, config, family)
        pooled_matched = _fit(x_train, y_train, config, _MATCHED[family])
        pooled_hgb = _fit(x_train, y_train, config, "hgb")

        mixture = gate_test * _predict(minority, x_test) + (1.0 - gate_test) * _predict(majority, x_test)
        pooled_losses.append(float(log_loss(y_test, np.clip(_predict(pooled, x_test), 1e-7, 1 - 1e-7), labels=[0, 1])))
        routed_losses.append(float(log_loss(y_test, np.clip(mixture, 1e-7, 1 - 1e-7), labels=[0, 1])))
        hgb_losses.append(
            float(log_loss(y_test, np.clip(_predict(pooled_hgb, x_test), 1e-7, 1 - 1e-7), labels=[0, 1]))
        )
        matched_losses.append(
            float(log_loss(y_test, np.clip(_predict(pooled_matched, x_test), 1e-7, 1 - 1e-7), labels=[0, 1]))
        )

    if not routed_losses:
        return {"family": family, "pooled_log_loss": None, "routed_log_loss": None, "routed_gain": None,
                "routed_gain_matched": None, "pooled_matched_log_loss": None,
                "pooled_hgb_log_loss": None, "gate_auc": None, "residual_gap": None,
                "hard_rate": None, "reason": "insufficient_rows"}
    pooled_mean, routed_mean = float(np.mean(pooled_losses)), float(np.mean(routed_losses))
    return {
        "family": family,
        "pooled_log_loss": pooled_mean,
        "routed_log_loss": routed_mean,
        # Positive means routing helped on held-out rows.
        "routed_gain": pooled_mean - routed_mean,
        # The statistic to read: routing against one model of the same budget.
        "routed_gain_matched": float(np.mean(matched_losses)) - routed_mean,
        "pooled_matched_log_loss": float(np.mean(matched_losses)),
        "routed_gain_std": float(np.std(np.array(pooled_losses) - np.array(routed_losses))),
        "pooled_hgb_log_loss": float(np.mean(hgb_losses)) if hgb_losses else None,
        "gate_auc": float(np.mean(gate_aucs)) if gate_aucs else None,
        "residual_gap": float(np.mean(gaps)),
        "hard_rate": float(np.mean(hard_rates)),
        "folds": len(routed_losses),
        "reason": None,
    }


def run(config: ResidualProbeConfig) -> Path:
    if config.cache_directory is not None:
        set_root_cache_directory(config.cache_directory)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))

    tasks = eligible_tasks(
        SmallTabArenaConfig(max_predictors=config.max_predictors, subsample=config.subsample, seed=config.seed)
    )
    print(f"eligible datasets: {len(tasks)}", flush=True)
    rows = []
    for task_info in tasks:
        task = openml.tasks.get_task(task_info["task_id"], download_splits=False)
        dataset = task.get_dataset(download_data=False)
        frame, target, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        labels = LabelEncoder().fit_transform(target)
        frame, labels = _subsample(frame, labels, config.subsample, config.seed)
        features = np.asarray(get_feature_preprocessor(frame).fit_transform(frame), dtype=np.float64)
        for family in config.expert_families:
            result = probe_table(features, np.asarray(labels), config, family)
            row = {"task_id": task_info["task_id"], "dataset": task_info["dataset"],
                   "predictors": task_info["predictors"], "rows": len(labels), **result}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_dataset.csv", index=False)
    scored = frame[frame["routed_gain"].notna()]
    summary: dict[str, Any] = {
        "datasets": int(frame["task_id"].nunique()),
        "scored_cells": int(len(scored)),
        "by_family": {},
    }
    for family in config.expert_families:
        group = scored[scored["family"] == family]
        if not len(group):
            continue
        summary["by_family"][family] = {
            "mean_routed_gain": float(group["routed_gain"].mean()),
            "mean_routed_gain_matched": float(group["routed_gain_matched"].mean()),
            "datasets_with_positive_gain": int((group["routed_gain"] > 0).sum()),
            "datasets_with_positive_matched_gain": int((group["routed_gain_matched"] > 0).sum()),
            "datasets": int(len(group)),
            "mean_gate_auc": float(group["gate_auc"].mean()),
            "mean_residual_gap": float(group["residual_gap"].mean()),
            "mean_pooled_log_loss": float(group["pooled_log_loss"].mean()),
            "mean_pooled_hgb_log_loss": float(group["pooled_hgb_log_loss"].mean()),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    defaults = ResidualProbeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-predictors", type=int, default=defaults.max_predictors)
    parser.add_argument("--subsample", type=int, default=defaults.subsample)
    parser.add_argument("--inner-folds", type=int, default=defaults.inner_folds)
    parser.add_argument("--outer-folds", type=int, default=defaults.outer_folds)
    parser.add_argument("--hard-quantile", type=float, default=defaults.hard_quantile)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--cache-directory", default=defaults.cache_directory)
    parser.add_argument(
        "--expert-families", nargs="+", default=list(defaults.expert_families), choices=EXPERT_FAMILIES
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    arguments = vars(build_parser().parse_args(argv))
    arguments["expert_families"] = tuple(arguments["expert_families"])
    return run(ResidualProbeConfig(**arguments))


if __name__ == "__main__":
    print(main())
