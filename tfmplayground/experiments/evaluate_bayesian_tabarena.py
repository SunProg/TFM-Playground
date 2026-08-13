"""Evaluate static Bayesian nanoTabPFN on binary TabArena tasks.

The local metric helpers are useful for smoke tests and reports.  The official
runner delegates task loading, split selection, and scoring to the pinned
TabArena checkout, exactly as the existing TabArena protocol does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tfmplayground.bayesian_interface import BayesianNanoTabPFNClassifier, VanillaNanoTabPFNClassifier
from tfmplayground.experiments.evaluate_task_posterior_tabarena import TABARENA_COMMIT, _installed_tabarena_revision
from tfmplayground.tabarena_model import TabArenaBayesianNanoTabPFNModel, TabArenaVanillaNanoTabPFNModel


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= left) & ((confidence <= right) if right == 1 else (confidence < right))
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def static_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    predictive_entropy: np.ndarray | None = None,
    mutual_information: np.ndarray | None = None,
    epistemic_variance: np.ndarray | None = None,
    posterior_entropy: np.ndarray | None = None,
    posterior_weights: np.ndarray | None = None,
    effective_sample_size: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute the requested binary quality, calibration, and uncertainty metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2 or len(y_true) != len(probabilities):
        raise ValueError("Expected y_true=(n,) and binary probabilities=(n, 2).")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise ValueError("probabilities must be finite and row-normalized.")
    positive = np.clip(probabilities[:, 1], 1e-12, 1 - 1e-12)
    nll = -np.log(np.where(y_true == 1, positive, 1 - positive)).mean()
    brier = np.mean((positive - y_true) ** 2)
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score

        accuracy = float(accuracy_score(y_true, probabilities.argmax(1)))
        roc_auc = float(roc_auc_score(y_true, positive))
    except ValueError:
        accuracy, roc_auc = float((probabilities.argmax(1) == y_true).mean()), None
    order = np.argsort(-np.maximum(positive, 1 - positive))
    coverage = max(1, int(0.8 * len(y_true)))
    selective_risk = float(1 - (probabilities.argmax(1)[order[:coverage]] == y_true[order[:coverage]]).mean())
    uncertainty_error_auroc = None
    uncertainty_aurc = None
    if mutual_information is not None:
        uncertainty = np.asarray(mutual_information, dtype=float)
        if uncertainty.shape != y_true.shape or not np.isfinite(uncertainty).all():
            raise ValueError("mutual_information must be finite and align with y_true.")
        errors = (probabilities.argmax(1) != y_true).astype(int)
        if np.unique(errors).size == 2:
            from sklearn.metrics import roc_auc_score

            uncertainty_error_auroc = float(roc_auc_score(errors, uncertainty))
        retained = np.argsort(uncertainty)
        cumulative_risk = np.cumsum(errors[retained]) / np.arange(1, len(errors) + 1)
        uncertainty_aurc = float(cumulative_risk.mean())
    result: dict[str, float | None] = {
        "nll": float(nll),
        "brier": float(brier),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "ece": _ece(y_true, probabilities),
        "selective_risk_at_80_coverage": selective_risk,
        "predictive_entropy": None if predictive_entropy is None else float(np.mean(predictive_entropy)),
        "epistemic_uncertainty": None if mutual_information is None else float(np.mean(mutual_information)),
        "epistemic_variance": None if epistemic_variance is None else float(np.mean(epistemic_variance)),
        "uncertainty_error_auroc": uncertainty_error_auroc,
        "uncertainty_aurc": uncertainty_aurc,
        "posterior_entropy": None if posterior_entropy is None else float(np.mean(posterior_entropy)),
        "effective_sample_size": (
            None if effective_sample_size is None else float(np.mean(effective_sample_size))
        ),
    }
    if posterior_weights is not None:
        weights = np.asarray(posterior_weights, dtype=float)
        entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(axis=-1)
        result["effective_hypothesis_count"] = float(np.exp(entropy).mean())
        result["posterior_collapse_fraction"] = float((weights.max(axis=-1) > 0.95).mean())
    else:
        result["effective_hypothesis_count"] = None
        result["posterior_collapse_fraction"] = None
    return result


def compare_uncertainty_to_vanilla_entropy(
    y_true: np.ndarray,
    bayesian_probabilities: np.ndarray,
    vanilla_probabilities: np.ndarray,
    bayesian_mutual_information: np.ndarray,
    *,
    material_tolerance: float = 0.01,
) -> dict[str, float | bool | None]:
    """Compare Bayesian MI against vanilla predictive entropy at a fixed mean."""
    if material_tolerance < 0:
        raise ValueError("material_tolerance must be non-negative.")
    vanilla = np.asarray(vanilla_probabilities, dtype=float)
    vanilla_entropy = -(vanilla * np.log(np.clip(vanilla, 1e-12, 1))).sum(axis=1)
    bayesian = static_metrics(
        y_true,
        bayesian_probabilities,
        mutual_information=bayesian_mutual_information,
    )
    baseline = static_metrics(y_true, vanilla_probabilities, mutual_information=vanilla_entropy)
    bayes_auroc = bayesian["uncertainty_error_auroc"]
    base_auroc = baseline["uncertainty_error_auroc"]
    bayes_aurc = bayesian["uncertainty_aurc"]
    base_aurc = baseline["uncertainty_aurc"]
    improves_auroc = None if bayes_auroc is None or base_auroc is None else bool(bayes_auroc > base_auroc)
    improves_aurc = None if bayes_aurc is None or base_aurc is None else bool(bayes_aurc < base_aurc)
    no_material_auroc_harm = (
        None
        if bayes_auroc is None or base_auroc is None
        else bool(bayes_auroc >= base_auroc - material_tolerance)
    )
    no_material_aurc_harm = (
        None
        if bayes_aurc is None or base_aurc is None
        else bool(bayes_aurc <= base_aurc + material_tolerance)
    )
    accepted = (
        None
        if None in {improves_auroc, improves_aurc, no_material_auroc_harm, no_material_aurc_harm}
        else bool(
            (improves_auroc or improves_aurc)
            and no_material_auroc_harm
            and no_material_aurc_harm
        )
    )
    return {
        "bayesian_error_auroc": bayes_auroc,
        "vanilla_entropy_error_auroc": base_auroc,
        "bayesian_aurc": bayes_aurc,
        "vanilla_entropy_aurc": base_aurc,
        "improves_error_auroc": improves_auroc,
        "improves_aurc": improves_aurc,
        "no_material_error_auroc_harm": no_material_auroc_harm,
        "no_material_aurc_harm": no_material_aurc_harm,
        "accepted": accepted,
    }


def posterior_collapse_diagnostics(weights: np.ndarray, threshold: float = 0.95) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or not np.isfinite(weights).all() or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("weights must be finite normalized rows with shape (n, hypotheses).")
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1)
    return {
        "posterior_entropy": float(entropy.mean()),
        "effective_hypothesis_count": float(np.exp(entropy).mean()),
        "collapse_fraction": float((weights.max(1) >= threshold).mean()),
    }


def evaluate_uncertainty_split(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    checkpoint: str,
    vanilla_checkpoint: str = "checkpoints/nanotabpfn.pth",
    context_size: int = 1024,
    random_state: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate quality and uncertainty on one untouched binary test split.

    Both classifiers receive the same deterministic context indices. Test
    labels are transformed only after both prediction calls have completed.
    """
    bayesian = BayesianNanoTabPFNClassifier(
        checkpoint,
        context_size=context_size,
        random_state=random_state,
        device=device,
    ).fit(X_train, y_train)
    vanilla = VanillaNanoTabPFNClassifier(
        vanilla_checkpoint,
        context_size=context_size,
        random_state=random_state,
        device=device,
    ).fit(X_train, y_train)
    bayesian_probabilities = bayesian.predict_proba(X_test)
    vanilla_probabilities = vanilla.predict_proba(X_test)
    encoded_y = bayesian.label_encoder_.transform(np.asarray(y_test))
    diagnostics = bayesian.last_diagnostics_
    bayesian_metrics = static_metrics(
        encoded_y,
        bayesian_probabilities,
        predictive_entropy=diagnostics["predictive_entropy"],
        mutual_information=diagnostics["mutual_information"],
        epistemic_variance=diagnostics["epistemic_variance"],
        posterior_entropy=np.broadcast_to(diagnostics["posterior_entropy"], encoded_y.shape),
        posterior_weights=np.broadcast_to(
            diagnostics["posterior_weights"],
            (len(encoded_y), diagnostics["posterior_weights"].shape[-1]),
        ),
        effective_sample_size=np.broadcast_to(diagnostics["effective_sample_size"], encoded_y.shape),
    )
    vanilla_metrics = static_metrics(encoded_y, vanilla_probabilities)
    uncertainty_comparison = compare_uncertainty_to_vanilla_entropy(
        encoded_y,
        bayesian_probabilities,
        vanilla_probabilities,
        diagnostics["mutual_information"],
    )
    return {
        "bayesian": bayesian_metrics,
        "vanilla": vanilla_metrics,
        "uncertainty_comparison": uncertainty_comparison,
        "max_mean_probability_difference": float(
            np.max(np.abs(bayesian_probabilities - vanilla_probabilities))
        ),
        "context_indices_identical": bool(
            np.array_equal(bayesian._context_indices(), vanilla._context_indices())
        ),
        "query_labels_used_for_construction": False,
    }


def run_official_tabarena(
    *, checkpoint: str, output_dir: str, results_dir: str, lite: bool = True, debug_mode: bool = False,
    compare_vanilla: bool = True, vanilla_checkpoint: str = "checkpoints/nanotabpfn.pth",
):
    try:
        import tabarena
        from tabarena.benchmark.experiment import ModelConstraints, TabArenaV0pt1ExperimentBundle
        from tabarena.contexts import TabArenaContext
    except ImportError as error:
        raise ImportError("Official evaluation requires the pinned TabArena checkout.") from error
    installed_revision = _installed_tabarena_revision(tabarena.__file__)
    if installed_revision != TABARENA_COMMIT:
        raise RuntimeError(f"Expected TabArena commit {TABARENA_COMMIT}, found {installed_revision or 'unknown'}.")
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    vanilla_checkpoint_path = Path(vanilla_checkpoint).expanduser().resolve()
    if compare_vanilla and not vanilla_checkpoint_path.is_file():
        raise FileNotFoundError(vanilla_checkpoint_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "tabarena_commit": TABARENA_COMMIT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "query_labels_used_for_construction": False,
        "test_partition": "official untouched test partition",
        "binary_only": True,
        "context_selection": "identical deterministic random subset with random_state=0",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    constraints = ModelConstraints(
        max_n_samples_train_per_fold=10_000, max_n_features=500, max_n_classes=2, regression_support=False
    )
    generators = []
    model_classes = (
        [TabArenaBayesianNanoTabPFNModel, TabArenaVanillaNanoTabPFNModel]
        if compare_vanilla
        else [TabArenaBayesianNanoTabPFNModel]
    )
    for model_class in model_classes:
        generator = model_class.config_generator()
        model_checkpoint = (
            checkpoint_path if model_class is TabArenaBayesianNanoTabPFNModel else vanilla_checkpoint_path
        )
        generator.manual_configs = [{"model": str(model_checkpoint)}]
        generators.append((generator, 0))
    experiments = TabArenaV0pt1ExperimentBundle(
        models=generators,
        outer_experiments=True,
        custom_model_constraints={
            TabArenaBayesianNanoTabPFNModel.ag_key: constraints,
            TabArenaVanillaNanoTabPFNModel.ag_key: constraints,
        },
    ).build_experiments()
    subset = ["classification", "tabpfn"]
    if lite:
        subset.insert(0, "lite")
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments, expname=results_dir, subset=subset, new_result_prefix="[New] ", debug_mode=debug_mode
    )
    return context.compare(output_dir=output, subset=subset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vanilla-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--no-vanilla", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(
        run_official_tabarena(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            lite=not args.full,
            debug_mode=args.debug_mode,
            compare_vanilla=not args.no_vanilla,
            vanilla_checkpoint=args.vanilla_checkpoint,
        ).to_markdown(index=False)
    )
