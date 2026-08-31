"""How well could *any* learner detect the contaminated rows of an episode?

Every binding number this project reports is scored against chance 0.5 and an
oracle ceiling of 1.0.  The oracle knows both label functions and reads the tag
straight off the label.  A learner knows neither and has to infer them from the
support set -- and that ceiling turns out to be about 0.54 on the prior as
originally built, which means every null result was measured against a target
nothing could reach.  This module measures the reachable target instead, and it
is cheap enough (CPU, minutes) that it should be run *before* any training run
that will be judged against it.

The detector is deliberately the strongest simple thing rather than a model
from this repo: cross-fit the support set's own ``(x, y)`` relationship, then
score each row by how badly the out-of-fold fit misses its label.  The majority
process dominates the fit, so minority-process rows are the ones it gets wrong.
Cross-fitting is not optional -- an in-fold fit memorizes the contaminated rows
and the score becomes meaningless.

It is also never given the tags.  A probe fitted on true tags answers "is the
contamination learnable *with supervision*", which is a different question and
the one that made ``regime_coherence`` look like a fix when it changed nothing
the model could use.

Two knobs matter, and they pull against each other:

``num_classes``
    Two independent rules collide on a row with probability about
    ``1 / num_classes``, and a collided row carries the tag while being
    observationally identical to a clean row.  More classes means more of the
    contamination is identifiable at all.
``features`` (and support size)
    More classes also splits the row budget more ways, so each label function
    gets harder to infer.  ``clean_fit_accuracy`` reports that directly:
    detection cannot beat how well the fit recovers the majority rule.

``target="regression"`` escapes the trade-off.  Continuous rules collide with
probability zero, so identifiability is 1.0, and no row budget is split.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from tfmplayground.experiments.continuous_episodes import (
    _contaminated_positions,
    _regime_direction,
    _scm_candidates,
    _seed_all,
)
from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    _rng_state,
    _sample_params,
    _set_rng_state,
)

#: Classification cells, then regression cells.  Append only, so a live array
#: keeps its index-to-cell mapping.
CLASS_COUNTS = (2, 3, 4, 5)
FEATURE_COUNTS = (12, 4)
CONTAMINATIONS = (0.30, 0.15)
SUPPORT_SIZES = (128, 512)
#: How far apart the two label rules are pushed, for continuous targets.
#:
#: Mixture-of-experts identifiability needs two things: a covariate-dependent
#: gate, *and* well-separated components.  `regime_coherence` supplied the gate
#: and separation was left to chance -- two independent SCM draws are however
#: far apart they happen to land -- which is why the gate bought nothing.  This
#: is the missing half, as a correlation: 0.0 leaves the rules independent (the
#: behaviour every earlier measurement used) and 1.0 makes the second rule the
#: negation of the first.
SEPARATIONS = (0.0, 0.5, 0.9)
#: Labels per row for the Dawid-Skene mechanism.  Given the true label the
#: annotators' responses are conditionally independent, and that assumption is
#: what makes the confusion matrices and the label prior identifiable.  This is
#: the one mechanism that makes an unidentifiable per-instance problem
#: identifiable *by construction* rather than by making instances easier.
REPEAT_COUNTS = (2, 3)
#: Fraction of the episode's rows revealed as known-clean.  The label-noise literature's
#: anchor-point assumption: instances known to belong to a class with
#: probability one.  Cheapest to bolt on, weakest guarantee.
ANCHOR_FRACTIONS = (0.10, 0.25)
#: Gate strength for the combined cells.  Mixture-of-experts identifiability
#: needs a covariate-dependent gate *and* separated components; every earlier
#: measurement had at most one of the two.
GATE_COHERENCE = 2.0
EPISODES = 60
SEED = 11


def ceiling_cells() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [
        {
            "target": "classification",
            "num_classes": num_classes,
            "features": features,
            "contamination": contamination,
            "support": support,
        }
        for support in SUPPORT_SIZES
        for features in FEATURE_COUNTS
        for contamination in CONTAMINATIONS
        for num_classes in CLASS_COUNTS
    ]
    grid += [
        {
            "target": "regression",
            "num_classes": 0,
            "features": features,
            "contamination": contamination,
            "support": support,
            "separation": 0.0,
        }
        for support in SUPPORT_SIZES
        for features in FEATURE_COUNTS
        for contamination in CONTAMINATIONS
    ]
    # Separated-component cells, appended so the indices above keep their
    # meaning while the array that is using them is still running.  Separation
    # is the half of mixture-of-experts identifiability that this prior never
    # controlled; the block above is its zero point.
    grid += [
        {
            "target": "regression",
            "num_classes": 0,
            "features": features,
            "contamination": contamination,
            "support": support,
            "separation": separation,
        }
        for separation in SEPARATIONS[1:]
        for support in SUPPORT_SIZES
        for features in FEATURE_COUNTS
        for contamination in CONTAMINATIONS
    ]
    # The remaining mechanisms, each on the *hardest* baseline -- two classes,
    # twelve features -- where the ceiling is 0.505.  Anything that rescues that
    # cell is doing real work rather than riding an easier task.
    grid += [
        {
            "target": "classification",
            "mechanism": "repeated",
            "num_classes": 2,
            "features": 12,
            "contamination": contamination,
            "support": support,
            "repeats": repeats,
        }
        for repeats in REPEAT_COUNTS
        for support in SUPPORT_SIZES
        for contamination in CONTAMINATIONS
    ]
    grid += [
        {
            "target": "classification",
            "mechanism": "anchors",
            "num_classes": 2,
            "features": 12,
            "contamination": contamination,
            "support": support,
            "anchor_fraction": anchor_fraction,
        }
        for anchor_fraction in ANCHOR_FRACTIONS
        for support in SUPPORT_SIZES
        for contamination in CONTAMINATIONS
    ]
    # Gate and separation together, which is the mixture-of-experts construction
    # proper.  `regime_coherence` alone moved the ceiling from 0.540 to 0.507.
    grid += [
        {
            "target": "regression",
            "num_classes": 0,
            "features": 12,
            "contamination": contamination,
            "support": support,
            "separation": separation,
            "coherence": GATE_COHERENCE,
        }
        for separation in SEPARATIONS[1:]
        for support in SUPPORT_SIZES
        for contamination in CONTAMINATIONS
    ]
    return grid


def _separate(values: list[torch.Tensor], separation: float) -> list[torch.Tensor]:
    """Push every rule after the first toward the negation of the first.

    Standardize, project out the component along rule A, then recombine at a
    target correlation of ``-separation``.  At 0.0 the rules are returned
    untouched, so the default reproduces independent draws exactly.
    """
    if separation <= 0.0:
        return values
    base = values[0]
    anchor = (base - base.mean()) / base.std().clamp_min(1e-6)
    separated = [base]
    for value in values[1:]:
        centred = (value - value.mean()) / value.std().clamp_min(1e-6)
        orthogonal = centred - (centred * anchor).mean() * anchor
        orthogonal = orthogonal / orthogonal.std().clamp_min(1e-6)
        residual = float(np.sqrt(max(0.0, 1.0 - separation**2)))
        separated.append(-separation * anchor + residual * orthogonal)
    return separated


def _continuous_candidates(
    rows: int, features: int, rng: np.random.Generator, separation: float = 0.0, max_attempts: int = 16
):
    """Two independent SCM label functions on one shared table, undiscretized.

    ``_scm_candidates`` runs the SCM's continuous output through ``Reg2Cls``.
    The regression target is that output before discretization, so nothing new
    has to be generated -- the same episode is simply read one step earlier.
    """
    from tabicl.prior._dataset import SCMPrior
    from tabicl.prior._mlp_scm import MLPSCM
    from tabicl.prior._reg2cls import Reg2Cls

    config = PriorBimodalConfig(
        initial_support_count=max(2, rows // 2),
        stream_count=0,
        query_count=1,
        min_features=features,
        max_features=features,
        device="cpu",
        prior_type="mlp_scm",
    )
    prior = SCMPrior(
        batch_size=1,
        min_features=features,
        max_features=features,
        max_classes=2,
        min_seq_len=rows,
        max_seq_len=rows + 1,
        min_train_size=max(2, rows // 2),
        max_train_size=max(3, rows // 2 + 1),
        prior_type="mlp_scm",
        n_jobs=1,
        device="cpu",
    )
    outer = _rng_state()
    try:
        for _ in range(max_attempts):
            _seed_all(int(rng.integers(0, 2**31 - 1)))
            params = _sample_params(prior, config, rows, features)
            models = []
            for _ in range(2):
                _seed_all(int(rng.integers(0, 2**31 - 1)))
                with torch.no_grad():
                    models.append(MLPSCM(**params))
            _seed_all(int(rng.integers(0, 2**31 - 1)))
            with torch.no_grad():
                shared = models[0].xsampler.sample()
            processed_x, values = None, []
            for model in models:
                with torch.no_grad():
                    value = model.layers(shared)
                    if value.shape[-1] == 1:
                        value = value.squeeze(-1)
                    # Reg2Cls also standardizes the features; its x is reused so
                    # the regression and classification cells see the same table.
                    x_value, _ = Reg2Cls(params)(shared.clone(), value)
                if processed_x is None:
                    processed_x = x_value.float()
                values.append(value.reshape(-1).float())
            values = _separate(values, separation)
            stacked = torch.stack(values).cpu().numpy().astype(np.float64)
            x = processed_x.reshape(-1, features).cpu().numpy().astype(np.float64)
            if np.isfinite(x).all() and np.isfinite(stacked).all() and stacked.std(axis=1).min() > 1e-6:
                # Standardize each rule so contamination is measured in target
                # standard deviations rather than whatever scale the SCM landed on.
                stacked = (stacked - stacked.mean(axis=1, keepdims=True)) / stacked.std(axis=1, keepdims=True)
                return x[:rows], stacked[:, :rows]
    finally:
        _set_rng_state(outer)
    raise RuntimeError("Could not draw finite continuous candidates.")


def _classification_misfit(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold ``1 - p(observed label)``."""
    score = np.zeros(len(y))
    counts = np.bincount(y)
    usable = folds if counts[counts > 0].min() >= folds else max(2, int(counts[counts > 0].min()))
    for train, test in StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed).split(x, y):
        if len(set(y[train])) < 2:
            score[test] = 0.5
            continue
        model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(x[train], y[train])
        probability = model.predict_proba(x[test])
        index = {label: position for position, label in enumerate(model.classes_)}
        score[test] = [
            1.0 - probability[row, index[label]] if label in index else 1.0
            for row, label in enumerate(y[test])
        ]
    return score


def _regression_misfit(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold absolute residual."""
    score = np.zeros(len(y))
    for train, test in KFold(n_splits=folds, shuffle=True, random_state=seed).split(x):
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15).fit(x[train], y[train])
        score[test] = np.abs(y[test] - model.predict(x[test]))
    return score


def measure(cell: dict[str, Any], episodes: int = EPISODES, seed: int = SEED) -> dict[str, Any]:
    """Run one grid cell and return its summary row."""
    rng = np.random.default_rng(seed)
    support, features = int(cell["support"]), int(cell["features"])
    contamination, regression = float(cell["contamination"]), cell["target"] == "regression"
    mechanism, coherence = cell.get("mechanism", "plain"), float(cell.get("coherence", 0.0))
    restricted, unrestricted, identifiable, clean_fit, separation = [], [], [], [], []

    for _ in range(episodes):
        try:
            if regression:
                x, rules = _continuous_candidates(
                    support + 8, features, rng, separation=float(cell.get("separation", 0.0))
                )
            else:
                x, rules = _scm_candidates(
                    "mlp_scm", 2, support + 8, features, rng, num_classes=int(cell["num_classes"])
                )
        except RuntimeError:
            continue
        x, base, other = x[:support], rules[0][:support], rules[1][:support]
        count = int(round(support * contamination))
        # The gate: at coherence 0 this is a uniform subset, which is what every
        # earlier measurement used.  Above 0 the relabelled rows concentrate on
        # one side of a per-episode hyperplane.
        direction = _regime_direction(rng, x.shape[1], coherence)

        def draw():
            """One draw of the mixture: which rows take the second rule."""
            chosen = _contaminated_positions(rng, x, count, coherence, direction)
            labels, flag = base.copy(), np.zeros(support, dtype=int)
            labels[chosen] = other[chosen]
            flag[chosen] = 1
            return labels, flag

        y, tag = draw()
        if len(set(tag)) < 2:
            continue
        distinct = np.ones(support, dtype=bool) if regression else (base != other)
        scored = np.ones(support, dtype=bool)

        if mechanism == "repeated":
            # Dawid-Skene.  Extra labels for the same row, each an independent
            # draw from the same mixture -- conditionally independent given the
            # true label, which is the assumption that buys identifiability.  A
            # contaminated primary label disagrees with the others exactly where
            # the two rules differ, so the votes localize it without any fit.
            votes = [draw()[0] for _ in range(int(cell["repeats"]) - 1)]
            score = np.mean([(vote != y).astype(float) for vote in votes], axis=0)
            clean_fit.append(float(np.mean(score[tag == 0] < 0.5)))
        elif mechanism == "anchors":
            # Anchor points.  A fraction of the clean rows is revealed as clean,
            # so the majority rule can be fitted on data with no contamination
            # in it at all -- which is the estimation problem that caps every
            # other mechanism.  Anchors are excluded from scoring: their status
            # was given, not inferred.
            clean = np.flatnonzero(tag == 0)
            # A fraction of the *episode*, not of the clean rows: "10% of rows
            # are known clean" is the assumption an annotator can actually make.
            anchor_count = min(len(clean), int(round(support * float(cell["anchor_fraction"]))))
            if anchor_count < 8:
                continue
            anchors = rng.choice(clean, size=anchor_count, replace=False)
            scored[anchors] = False
            labels = y.astype(int)
            if len(set(labels[anchors])) < 2:
                continue
            model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15)
            model.fit(x[anchors], labels[anchors])
            probability = model.predict_proba(x)
            position = {label: column for column, label in enumerate(model.classes_)}
            score = np.array(
                [
                    1.0 - probability[row, position[label]] if label in position else 1.0
                    for row, label in enumerate(labels)
                ]
            )
            clean_fit.append(float(np.mean(score[(tag == 0) & scored] < 0.5)))
        elif regression:
            separation.append(float(np.mean(np.abs(base - other))))
            score = _regression_misfit(x, y)
            clean_fit.append(float(np.mean(score[tag == 0] < np.median(score))))
        else:
            if len(set(y)) < 2:
                continue
            score = _classification_misfit(x, y.astype(int))
            clean_fit.append(float(np.mean(score[tag == 0] < 0.5)))

        identifiable.append(float(distinct[tag == 1].mean()))
        if len(set(tag[scored])) < 2:
            continue
        unrestricted.append(float(roc_auc_score(tag[scored], score[scored])))
        keep = scored & (distinct | (tag == 0))
        if len(set(tag[keep])) > 1:
            restricted.append(float(roc_auc_score(tag[keep], score[keep])))

    def summary(values: list[float]) -> dict[str, float]:
        array = np.array(values, dtype=float)
        if not array.size:
            return {"mean": float("nan"), "ci": float("nan"), "n": 0}
        return {
            "mean": float(array.mean()),
            "ci": float(1.96 * array.std() / np.sqrt(array.size)),
            "n": int(array.size),
        }

    identifiable_mean = float(np.mean(identifiable)) if identifiable else float("nan")
    return {
        **cell,
        "restricted": summary(restricted),
        "unrestricted": summary(unrestricted),
        "identifiable": identifiable_mean,
        "clean_fit_accuracy": float(np.mean(clean_fit)) if clean_fit else float("nan"),
        # What a perfect detector reaches when the collided positives are left in.
        "unrestricted_ceiling": identifiable_mean + (1.0 - identifiable_mean) * 0.5,
        "rule_separation": float(np.mean(separation)) if separation else None,
        "episodes": episodes,
        "seed": seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, help="Run one cell of the grid and print its JSON row.")
    parser.add_argument("--list", action="store_true", help="Print the grid and exit.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> str:
    arguments = build_parser().parse_args(argv)
    cells = ceiling_cells()
    if arguments.list:
        return json.dumps(cells, indent=2)
    if arguments.index is None:
        raise SystemExit("Pass --index or --list.")
    if not 0 <= arguments.index < len(cells):
        raise IndexError(f"Array index {arguments.index} is outside 0..{len(cells) - 1}.")
    return json.dumps(measure(cells[arguments.index], episodes=arguments.episodes, seed=arguments.seed))


if __name__ == "__main__":
    print(main())
