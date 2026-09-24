"""TabICL-matched SCM episodes with two multiregime constructions.

The default ``native`` HP profile *is* TabICL's prior: ``DEFAULT_FIXED_HP``
kept fixed (``balanced=False``, ``cat_prob=0.2``, ``permute_labels=True``, …)
and ``DEFAULT_SAMPLED_HP`` sampled per group exactly as ``HpSamplerList`` does,
with native MLP/tree SCMs and native ``Reg2Cls`` feature preprocessing *and*
label assignment (``MulticlassAssigner`` random rank/value boundaries).  K=1
episodes are therefore TabICL datasets, and multiregime episodes apply that
same native label rule once per regime.  ``class_ratio`` (evaluation-bank
cells only) switches an episode to the calibrated label path, which fixes the
class vector by quantile.  ``production`` is the deprecated 2026-09 profile
that additionally randomised TabICL's fixed controls and calibrated *every*
label to a uniform class vector (all binary tasks 50/50); it is kept only to
reproduce the dumps made with it.  ``tabicl_test`` is an alias of ``native``
kept for the equivalence test.  In every profile, ``is_causal`` remains a
sampled MLP-SCM hyperparameter; TreeSCM keeps its native forced-predictive
behavior.

``r_z`` gives the SCM ``num_regimes`` candidate regression columns:

    X -> [Y_0, ..., Y_(k-1)]

``soft_gate`` derives its latent selector from the observed ``X`` without
adding a separate SCM output.  ``persistent`` uses a latent group selector.
``g_z`` instead draws one ordinary TabICL SCM score ``r(X)`` and applies
prevalence-normalized regime-specific quantile maps to that shared score.
Thus it changes only the mapping from the shared score to a class label:

    X -> r(X) -> g_Z(r)

For either construction, a paired shared-rule control always uses rule zero;
the multiregime task uses the selected rule.  Calls with the same seed are
paired in their observed features and available candidate rules.  Under the
``native`` profile both constructions route ``K=1`` through the native TabICL
generation path, so it is an exact original-prior control rather than an
approximation.

Generation follows TabICL's group/subgroup idea. A ``V4GenerationGroup``
samples table length, support/query split, feature count, regime count, SCM
backend, and SCM hyperparameters once. The regime count is bounded by
``support_size // min_samples_per_regime``. Member episodes have independent
seeds, rows, and SCM weights, but share those group-level characteristics.
This lets a dump loader form rectangular model batches while training sees a
distribution of feature widths, table sizes, and regime counts.

The ``persistent`` family remains a deliberately latent control: its row
groups select one fixed candidate label function. Group identities are not
included in ``X`` and therefore do not disclose a query row's regime.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from tabicl.prior._dataset import SCMPrior
from tabicl.prior._hp_sampling import HpSamplerList
from tabicl.prior._mlp_scm import MLPSCM
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior._reg2cls import MulticlassAssigner, Reg2Cls, permute_classes, standard_scaling
from tabicl.prior._tree_scm import TreeSCM

FAMILIES = ("soft_gate", "persistent")
RULE_MODES = ("shared", "multiregime")
MECHANISM_MODES = ("r_z", "g_z")
HP_PROFILES = ("native", "production", "tabicl_test")
#: Attempts at drawing an episode whose selected labels pass TabICL's split sanity check.
_NATIVE_LABEL_ATTEMPTS = 64
DEFAULT_REGIME_COUNT_WEIGHTS = {2: 0.5, 3: 0.3, 4: 0.2}


def _mlp_probability(mix_probs: tuple[float, float]) -> float:
    if len(mix_probs) != 2 or min(mix_probs) < 0 or sum(mix_probs) <= 0:
        raise ValueError("mix_probs must contain two non-negative weights with positive sum.")
    return mix_probs[0] / sum(mix_probs)


@contextmanager
def _temporary_global_seed(seed: int):
    """Seed TabICL's global samplers without perturbing the caller's RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**63 - 1))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _sample_fixed_hparams(
    seed: int,
    *,
    mix_probs: tuple[float, float],
    hp_profile: Literal["native", "production", "tabicl_test"],
) -> dict[str, Any]:
    """TabICL's fixed controls (``native``/``tabicl_test``) or the deprecated sampled ones.

    ``production`` (deprecated) does not import TabICL's ``DEFAULT_FIXED_HP``
    values into a dataset: it samples the controls that TabICL normally holds
    constant and records their realised values.
    """
    if hp_profile in ("native", "tabicl_test"):
        return {**DEFAULT_FIXED_HP, "mix_probs": tuple(mix_probs)}
    if hp_profile != "production":
        raise ValueError(f"Unknown HP profile {hp_profile!r}.")
    rng = np.random.default_rng(seed)
    return {
        "mix_probs": tuple(mix_probs),
        "tree_model": "xgboost",
        "tree_depth_lambda": float(rng.uniform(0.25, 1.25)),
        "tree_n_estimators_lambda": float(rng.uniform(0.25, 1.25)),
        "balanced": bool(rng.integers(2)),
        "multiclass_ordered_prob": float(rng.uniform(0.0, 1.0)),
        "cat_prob": float(rng.uniform(0.0, 0.5)),
        "max_categories": int(rng.integers(2, 13)),
        "scale_by_max_features": bool(rng.integers(2)),
        "permute_features": bool(rng.integers(2)),
        "permute_labels": bool(rng.integers(2)),
    }


def _sample_tabicl_mix_scm_hparams(
    seed: int,
    *,
    num_features: int,
    num_outputs: int,
    num_classes: int = 2,
    mix_probs: tuple[float, float],
    hp_profile: Literal["native", "production", "tabicl_test"] = "native",
) -> dict[str, Any]:
    """Sample SCM settings: TabICL's sampled HPs plus the profile's fixed controls."""
    with _temporary_global_seed(seed):
        sampled = HpSamplerList(DEFAULT_SAMPLED_HP, device="cpu").sample()
        sampled = {name: value() if callable(value) else value for name, value in sampled.items()}
    return {
        **_sample_fixed_hparams(seed, mix_probs=mix_probs, hp_profile=hp_profile),
        **sampled,
        "num_features": num_features,
        "num_outputs": num_outputs,
        "num_classes": num_classes,
        "max_features": num_features,
        "device": "cpu",
    }


def _target_class_probabilities(num_classes: int, class_ratio: float | None) -> np.ndarray:
    """Return the class vector shared by every regime in a production episode.

    Binary evaluation cells may specify their positive-class probability.
    Multiclass episodes deliberately use a uniform vector: requested
    cardinality is therefore not merely a label vocabulary, but has one
    controlled marginal distribution in every regime.
    """
    if class_ratio is not None:
        if num_classes != 2:
            raise ValueError("class_ratio controls are defined only for binary episodes.")
        return np.asarray((1.0 - class_ratio, class_ratio), dtype=np.float64)
    return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)


def _controlled_label_grid(
    raw_rule_scores: np.ndarray,
    *,
    calibration_scores: np.ndarray,
    calibration_indices: np.ndarray,
    calibration_z: np.ndarray,
    tie_breaker: np.ndarray,
    num_regimes: int,
    class_probabilities: np.ndarray,
    mechanism_mode: str,
) -> np.ndarray:
    """Calibrate every selector/rule pair to one shared class vector.

    ``calibration_z`` is never exposed to the model.  It estimates the score
    CDF conditional on a routing regime.  Applying that CDF to support and
    query rows makes their *generative* ``P(Y | Z=z)`` identical.  A circular
    quantile shift keeps ``g_z`` rules distinct while preserving the vector;
    for ``r_z`` it complements the different SCM score columns.
    """
    if raw_rule_scores.ndim != 2 or calibration_scores.ndim != 2:
        raise ValueError("raw_rule_scores must have shape (rows, candidate_scores).")
    if calibration_scores.shape[1] != raw_rule_scores.shape[1]:
        raise ValueError("calibration_scores must have the same score width as raw_rule_scores.")
    if calibration_indices.ndim != 1 or len(calibration_indices) != len(calibration_scores):
        raise ValueError("calibration_indices must identify every label-reference row.")
    if tie_breaker.ndim != 1 or len(tie_breaker) != len(raw_rule_scores):
        raise ValueError("tie_breaker must supply one deterministic value per generated row.")
    if calibration_z.ndim != 1 or len(calibration_z) != len(calibration_scores):
        raise ValueError("calibration_z must assign every label-reference row to a routing regime.")
    if not np.all((calibration_z >= 0) & (calibration_z < num_regimes)):
        raise ValueError("calibration_z contains an invalid regime index.")
    if mechanism_mode == "g_z":
        score_columns = np.zeros(num_regimes, dtype=np.int64)
    elif mechanism_mode == "r_z":
        if raw_rule_scores.shape[1] != num_regimes:
            raise ValueError("r_z requires one raw score column per regime.")
        score_columns = np.arange(num_regimes)
    else:
        raise ValueError(f"Unknown mechanism_mode {mechanism_mode!r}.")

    boundaries = np.cumsum(class_probabilities, dtype=np.float64)[:-1]
    labels = np.empty((len(raw_rule_scores), num_regimes, num_regimes), dtype=np.int64)
    # Tree-SCM outputs may contain substantial exact-score ties. A deterministic
    # hash of observed X breaks only those ties, allowing a score rule to meet
    # its requested class vector without adding unobserved label noise.
    stable_scores = raw_rule_scores.astype(np.float64, copy=True)
    for column in range(stable_scores.shape[1]):
        unique = np.unique(stable_scores[:, column])
        gaps = np.diff(unique)
        positive_gaps = gaps[gaps > 0]
        scale = (
            float(positive_gaps.min()) / 4.0
            if len(positive_gaps)
            else max(1.0, float(np.abs(unique).max(initial=0.0))) * 1e-6
        )
        stable_scores[:, column] += scale * (tie_breaker - 0.5)
    calibration_stable_scores = stable_scores[calibration_indices]
    for selector in range(num_regimes):
        selector_rows = calibration_z == selector
        if selector_rows.sum() < 2:
            raise RuntimeError("Each routing regime needs at least two calibration rows.")
        for rule, column in enumerate(score_columns):
            reference = np.sort(calibration_stable_scores[selector_rows, column])
            quantiles = np.searchsorted(reference, stable_scores[:, column], side="right") / len(reference)
            shifted_quantiles = np.mod(quantiles + rule / num_regimes, 1.0)
            labels[:, selector, rule] = np.searchsorted(boundaries, shifted_quantiles, side="right")
    return labels


def _observed_x_tie_breaker(x: np.ndarray) -> np.ndarray:
    """A deterministic, observed-X hash used only to resolve exact score ties."""
    weights = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=np.float64))
    projection = np.nan_to_num(x, copy=False).astype(np.float64) @ weights
    return np.mod(np.sin(projection * 12.9898) * 43_758.5453, 1.0)


def _class_counts_by_routing_regime(
    labels: np.ndarray, routing_z: np.ndarray, *, num_regimes: int, num_classes: int
) -> list[list[int]]:
    """Record all class counts, including zeroes, for an audit-friendly cell."""
    return [
        np.bincount(labels[routing_z == regime], minlength=num_classes).astype(int).tolist()
        for regime in range(num_regimes)
    ]


@dataclass(frozen=True)
class _NativeBaseDraw:
    """One accepted native TabICL SCM draw, retaining its pre-label score."""

    x: torch.Tensor
    y: torch.Tensor
    raw_x: torch.Tensor
    raw_y: torch.Tensor
    d: int
    params: dict[str, Any]
    model: torch.nn.Module
    prior_type: str


class _CapturingSCMPrior(SCMPrior):
    """Native ``SCMPrior`` with a read-only capture point before Reg2Cls.

    ``generate_dataset`` deliberately follows TabICL's implementation line by
    line.  The only additions preserve the unlabelled SCM score and apply a
    sanity-check permutation to it as well.  Consequently the returned X/y
    tensors for the first rule are bit-identical to native ``SCMPrior``.
    """

    captured: _NativeBaseDraw | None = None
    #: When set, overrides the class count TabICL samples per dataset (the
    #: episode draws it first with the same 50 % binary / uniform mixture).
    forced_num_classes: int | None = None

    @staticmethod
    def _sanity_check_with_raw(
        prior: SCMPrior,
        x: torch.Tensor,
        y: torch.Tensor,
        raw_x: torch.Tensor,
        raw_y: torch.Tensor,
        train_size: int,
    ) -> bool:
        def valid(labels: torch.Tensor) -> bool:
            if train_size <= 0 or train_size >= labels.shape[0]:
                return False
            train_classes = torch.unique(labels[:train_size])
            test_classes = torch.unique(labels[train_size:])
            return set(train_classes.tolist()) == set(test_classes.tolist()) and len(train_classes) >= 2

        if valid(y[0]):
            return True
        # This is TabICL's ``sanity_check`` loop with the identical RNG draw.
        # Moving raw values by that draw is needed only to keep g_z(r) aligned.
        for _ in range(10):
            permutation = torch.randperm(y.shape[1])
            permuted_y = y[0, permutation]
            if valid(permuted_y):
                x[0], y[0] = x[0, permutation], permuted_y
                raw_x.copy_(raw_x[permutation])
                raw_y.copy_(raw_y[permutation])
                return True
        return False

    @torch.no_grad()
    def generate_dataset(self, params: dict[str, Any]):  # type: ignore[override]
        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:  # pragma: no cover - protected by native SCMPrior.get_prior
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        if self.forced_num_classes is not None:
            params = {**params, "num_classes": int(self.forced_num_classes)}
        while True:
            model = prior_cls(**params)
            raw_x, raw_y = model()
            # r_z draws ``num_outputs`` score columns at once; column 0 is the
            # native score that Reg2Cls labels, the others are the extra rules.
            if raw_y.ndim == 2:
                raw_y = raw_y if raw_y.shape[1] > 1 else raw_y[:, 0]
            native_score = raw_y[:, 0] if raw_y.ndim == 2 else raw_y
            # Reg2Cls may convert feature columns in-place.  Preserve the SCM
            # output before that native operation for the additional rules.
            rule_x, rule_y = raw_x.clone(), raw_y.clone()
            x, y = Reg2Cls(params)(raw_x, native_score)

            x_batch, y_batch = x.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)
            x_batch, d = self.delete_unique_features(x_batch, d)
            if (d > 0).all() and self._sanity_check_with_raw(
                self, x_batch, y_batch, rule_x, rule_y, params["train_size"]
            ):
                # An in-place tensor copy keeps the raw score columns aligned
                # with any sanity-check permutation (2-D rule_y is a view-safe copy).
                self.captured = _NativeBaseDraw(
                    x=x_batch.squeeze(0).clone(),
                    y=y_batch.squeeze(0).clone(),
                    raw_x=rule_x.clone(),
                    raw_y=rule_y.clone(),
                    d=int(d.squeeze(0)),
                    params=dict(params),
                    model=model,
                    prior_type=params["prior_type"],
                )
                return x_batch.squeeze(0), y_batch.squeeze(0), d.squeeze(0)


def _draw_native_tabicl_base(
    seed: int,
    *,
    num_features: int,
    rows: int,
    support_size: int,
    mix_probs: tuple[float, float],
    hp_profile: Literal["native", "production", "tabicl_test"] = "native",
    shared_hparams: dict[str, Any] | None = None,
    shared_prior_type: str | None = None,
    num_classes: int | None = None,
    num_outputs: int = 1,
) -> _NativeBaseDraw:
    """Draw one native SCM task, including its native acceptance loop.

    Grouped V4 draws pass fully realised shared HPs, just as native TabICL
    shares a group's sampled HPs.  Standalone draws sample the normal native
    hierarchy; ``tabicl_test`` is the exact-default comparison path.
    """
    if rows <= support_size:
        raise ValueError("rows must exceed support_size.")
    if shared_hparams is None:
        fixed_hp = {**_sample_fixed_hparams(seed, mix_probs=mix_probs, hp_profile=hp_profile), "num_outputs": num_outputs}
        sampled_hp: dict[str, Any] = DEFAULT_SAMPLED_HP
        prior_type = "mix_scm"
    else:
        fixed_hp = {**shared_hparams, "num_outputs": num_outputs}
        sampled_hp = {}
        prior_type = shared_prior_type or "mix_scm"
    with _temporary_global_seed(seed):
        prior = _CapturingSCMPrior(
            batch_size=1,
            batch_size_per_gp=1,
            batch_size_per_subgp=1,
            min_features=num_features,
            max_features=num_features,
            max_classes=2 if num_classes is None else num_classes,
            min_seq_len=rows,
            max_seq_len=rows + 1,
            min_train_size=support_size,
            max_train_size=support_size + 1,
            prior_type=prior_type,
            fixed_hp=fixed_hp,
            sampled_hp=sampled_hp,
            n_jobs=1,
            device="cpu",
        )
        prior.forced_num_classes = num_classes
        prior.get_batch()
    if prior.captured is None:  # pragma: no cover - defensive only
        raise RuntimeError("Native TabICL draw completed without a captured dataset.")
    return prior.captured


def _native_label_rule(score: torch.Tensor, hparams: dict[str, Any]) -> torch.Tensor:
    """TabICL's ``Reg2Cls`` target path only: standardise, assign classes, permute labels.

    This is exactly what ``Reg2Cls.forward`` does to ``y`` (random rank/value
    boundaries via ``MulticlassAssigner``); the feature processing is skipped
    because every rule of an episode shares the one native processed ``X``.
    """
    y = standard_scaling(score.reshape(-1, 1)).reshape(-1)
    num_classes = int(hparams["num_classes"])
    if num_classes == 2 and hparams.get("balanced", False):
        y = (y > y.median()).long()  # BalancedBinarize
    else:
        y = MulticlassAssigner(
            num_classes, mode=hparams["multiclass_type"], ordered_prob=hparams["multiclass_ordered_prob"]
        )(y)
    if hparams.get("permute_labels", True):
        y = permute_classes(y)
    return y.float()


def _rank_threshold_rule(score: torch.Tensor, rule: int, num_regimes: int, existing: list[torch.Tensor]) -> torch.Tensor:
    """Last-resort deterministic rule: a rank threshold that differs from every existing rule."""
    ranks = torch.argsort(torch.argsort(score))
    boundary = max(1, min(len(ranks) - 1, (rule * len(ranks)) // num_regimes))
    label = (ranks >= boundary).to(existing[0].dtype)
    if any(torch.equal(label, other) for other in existing):
        label = 1 - label
    return label


def _tabicl_label_rules_from_native_base(
    base: _NativeBaseDraw, num_regimes: int, *, mechanism_mode: str = "g_z"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native rule zero plus one native label rule per additional regime.

    ``g_z``: every rule is an independent ``Reg2Cls`` draw on the one native
    score ``r(X)``.  ``r_z``: rule ``j`` is the native label rule applied to
    the SCM's ``j``-th score column (``base.raw_y`` is then ``(rows, K)``).
    A resample avoids a degenerate regime whose sampled rule coincides with an
    already available one; the rank threshold is the bounded fallback.
    """
    labels = [base.y]
    scores = base.raw_y if base.raw_y.ndim == 2 else base.raw_y[:, None]
    for rule in range(1, num_regimes):
        score = scores[:, rule] if mechanism_mode == "r_z" else scores[:, 0]
        for _attempt in range(32):
            label = _native_label_rule(score.clone(), base.params)
            if not any(torch.equal(label, existing) for existing in labels):
                labels.append(label)
                break
        else:
            labels.append(_rank_threshold_rule(score, rule, num_regimes, labels))
    return base.x, torch.stack(labels, dim=1)


def _split_is_valid(support_y: np.ndarray, query_y: np.ndarray) -> bool:
    """TabICL's ``sanity_check`` condition: both halves hold the same >= 2 classes."""
    support_classes, query_classes = set(np.unique(support_y).tolist()), set(np.unique(query_y).tolist())
    return support_classes == query_classes and len(support_classes) >= 2


def _jsonable(value: Any) -> Any:
    """Convert TabICL HP objects and modules into lossless audit metadata."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    return repr(value)


def _scm_metadata(model: torch.nn.Module, hparams: dict[str, Any], prior_type: str) -> dict[str, Any]:
    """Record both requested TabICL HPs and the instantiated SCM's settings."""
    metadata: dict[str, Any] = {
        "prior_type": prior_type,
        "sampled_is_causal": bool(hparams["is_causal"]),
        "effective_is_causal": bool(model.is_causal),
        "requested_num_layers": int(hparams["num_layers"]),
        "effective_num_layers": int(model.num_layers),
        "requested_hidden_dim": int(hparams["hidden_dim"]),
        "effective_hidden_dim": int(model.hidden_dim),
        "effective_num_causes": int(model.num_causes),
        "tabicl_hparams": _jsonable(hparams),
    }
    if prior_type == "mlp_scm":
        metadata["effective_mlp_activations"] = [
            type(layer[0]).__name__
            for layer in model.layers
            if isinstance(layer, torch.nn.Sequential)
        ]
    else:
        tree_layers = []
        for layer in model.layers:
            tree_layer = layer[0] if isinstance(layer, torch.nn.Sequential) else layer
            tree_layers.append(
                {
                    "model": type(tree_layer.model).__name__,
                    "max_depth": int(tree_layer.max_depth),
                    "n_estimators": int(tree_layer.n_estimators),
                    "out_dim": int(tree_layer.out_dim),
                }
            )
        metadata["effective_tree_layers"] = tree_layers
    return metadata


@dataclass(frozen=True)
class V4GenerationGroup:
    """Shared task-distribution parameters for independent member episodes."""

    num_features: int
    support_size: int
    query_size: int
    num_regimes: int
    min_samples_per_regime: int
    prior_type: str
    hparams: dict[str, Any]
    hp_profile: Literal["native", "production", "tabicl_test"] = "native"

    @property
    def rows(self) -> int:
        return self.support_size + self.query_size


def sample_generation_group_v4(
    rng: np.random.Generator,
    *,
    min_features: int,
    max_features: int,
    min_instances: int,
    max_instances: int,
    min_train_fraction: float,
    max_train_fraction: float,
    num_regimes: int | None = None,
    min_regimes: int = 2,
    max_regimes: int = 4,
    min_samples_per_regime: int = 32,
    mix_probs: tuple[float, float] = (0.7, 0.3),
    hp_profile: Literal["native", "production", "tabicl_test"] = "native",
    fixed_support_size: int | None = None,
    fixed_query_size: int | None = None,
) -> V4GenerationGroup:
    """Sample one TabICL-style group with a variable table width and length.

    Bounds for features and instances are inclusive. Supplying both fixed
    split sizes bypasses fraction sampling and makes every group use that exact
    table geometry; otherwise the support fraction is sampled uniformly from
    ``[min_train_fraction, max_train_fraction)``. When
    ``num_regimes`` is omitted, its weighted draw is truncated at
    ``support_size // min_samples_per_regime``. A supplied fixed count is
    subject to the same cap (except the single-regime compatibility case).
    """
    if not 1 <= min_features <= max_features:
        raise ValueError("Feature bounds must satisfy 1 <= min_features <= max_features.")
    if not 2 <= min_instances <= max_instances:
        raise ValueError("Instance bounds must satisfy 2 <= min_instances <= max_instances.")
    if not 0 < min_train_fraction <= max_train_fraction < 1:
        raise ValueError("Train fractions must satisfy 0 < min <= max < 1.")
    if num_regimes is not None and num_regimes < 1:
        raise ValueError("num_regimes must be positive.")
    if not 1 <= min_regimes <= max_regimes:
        raise ValueError("Sampled regime bounds must satisfy 1 <= min_regimes <= max_regimes.")
    if min_samples_per_regime < 1:
        raise ValueError("min_samples_per_regime must be positive.")
    if (fixed_support_size is None) != (fixed_query_size is None):
        raise ValueError("fixed_support_size and fixed_query_size must be supplied together.")
    if fixed_support_size is not None:
        if fixed_support_size < 1 or fixed_query_size is None or fixed_query_size < 1:
            raise ValueError("Fixed support and query sizes must both be positive.")
        fixed_rows = fixed_support_size + fixed_query_size
        if not min_instances <= fixed_rows <= max_instances:
            raise ValueError("Fixed support/query sizes must sum within the instance bounds.")

    # A soft gate has roughly 1/K support rows in each quantile bin. Therefore
    # K is bounded by the support evidence, not the full table length.
    for _attempt in range(64):
        if fixed_support_size is None:
            rows = int(rng.integers(min_instances, max_instances + 1))
            train_fraction = float(rng.uniform(min_train_fraction, max_train_fraction))
            support_size = int(np.clip(int(rows * train_fraction), 1, rows - 1))
            query_size = rows - support_size
        else:
            support_size, query_size = fixed_support_size, fixed_query_size
            rows = support_size + query_size
        regime_cap = support_size // min_samples_per_regime
        if num_regimes is not None:
            sampled_num_regimes = num_regimes
            if sampled_num_regimes == 1 or sampled_num_regimes <= regime_cap:
                break
            continue
        candidates = np.arange(min_regimes, min(max_regimes, regime_cap) + 1)
        if len(candidates) == 0:
            continue
        weights = np.asarray([DEFAULT_REGIME_COUNT_WEIGHTS.get(int(k), 1.0) for k in candidates], dtype=float)
        sampled_num_regimes = int(rng.choice(candidates, p=weights / weights.sum()))
        break
    else:
        raise ValueError("Could not draw enough support rows for the requested regime count.")

    num_features = int(rng.integers(min_features, max_features + 1))
    prior_type = "mlp_scm" if rng.random() < _mlp_probability(mix_probs) else "tree_scm"
    hparams = _sample_tabicl_mix_scm_hparams(
        int(rng.integers(2**31)),
        num_features=num_features,
        num_outputs=sampled_num_regimes,
        mix_probs=mix_probs,
        hp_profile=hp_profile,
    )
    return V4GenerationGroup(
        num_features=num_features,
        support_size=support_size,
        query_size=query_size,
        num_regimes=sampled_num_regimes,
        min_samples_per_regime=min_samples_per_regime,
        prior_type=prior_type,
        hparams=hparams,
        hp_profile=hp_profile,
    )


@dataclass(frozen=True)
class V4Episode:
    support_x: torch.Tensor  # (1, support_size, padded_features)
    support_y: torch.Tensor  # (1, support_size)
    query_x: torch.Tensor  # (1, query_size, padded_features)
    query_y: torch.Tensor  # (1, query_size)
    support_z: np.ndarray  # realized regime; diagnostic only
    query_z: np.ndarray  # realized regime; diagnostic only
    d: int  # unpadded feature count; adds one when expose_z=True
    num_classes: int  # target cardinality requested for this episode
    prior_type: str  # SCM backend; diagnostic only
    scm_metadata: dict[str, Any]  # requested and effective TabICL SCM details
    z_column_index: int | None = None  # diagnostic only


def sample_episode_v4(
    seed: int,
    *,
    family: str,
    min_features: int,
    max_features: int,
    num_regimes: int,
    support_size: int | None = None,
    query_size: int | None = None,
    # Retained for CLI/dump compatibility. Controlled production labels use
    # the full generated episode as their native-style rank reference.
    calibration_size: int = 256,
    pad_features: int | None = None,
    num_classes: int | None = None,
    max_classes: int = 2,
    min_samples_per_regime: int | None = None,
    num_groups: int = 5,
    active_groups: int | None = None,
    mix_probs: tuple[float, float] = (0.7, 0.3),
    class_ratio: float | None = None,
    expose_z: bool = False,
    rule_mode: Literal["shared", "multiregime"] = "multiregime",
    mechanism_mode: Literal["r_z", "g_z"] = "r_z",
    hp_profile: Literal["native", "production", "tabicl_test"] = "native",
    group: V4GenerationGroup | None = None,
    require_num_classes: bool = False,
) -> V4Episode:
    """Draw one member episode.

    Passing ``group`` uses its shared SCM configuration. Without one, this
    retains the old standalone API: feature count and SCM hparams are sampled
    for this episode and ``support_size``/``query_size`` must be supplied.
    ``require_num_classes`` (evaluation-bank cells) makes native labels redraw
    until every requested class is present in both halves; training dumps
    keep TabICL's weaker condition (>= 2 classes, same set in both halves).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown v4 family {family!r}; use one of {FAMILIES}.")
    if rule_mode not in RULE_MODES:
        raise ValueError("rule_mode must be shared or multiregime.")
    if mechanism_mode not in MECHANISM_MODES:
        raise ValueError(f"mechanism_mode must be one of {MECHANISM_MODES}.")
    if hp_profile not in HP_PROFILES:
        raise ValueError(f"hp_profile must be one of {HP_PROFILES}.")
    if num_regimes < 1:
        raise ValueError("num_regimes must be positive.")
    if max_classes < 2:
        raise ValueError("max_classes must be at least two.")
    if num_classes is not None and not 2 <= num_classes <= max_classes:
        raise ValueError("num_classes must lie in [2, max_classes].")
    if calibration_size < 1:
        raise ValueError("calibration_size must be positive.")
    if class_ratio is not None and not 0 < class_ratio < 1:
        raise ValueError("class_ratio must lie strictly between 0 and 1.")
    if active_groups is None:
        active_groups = num_groups
    if not 1 <= active_groups <= num_groups:
        raise ValueError("active_groups must lie in [1, num_groups].")

    rng = np.random.default_rng(seed)
    # Match TabICL's class-cardinality mixture: half of the datasets are
    # explicitly binary; the other half draw uniformly from the permitted
    # cardinalities (including two).  The draw happens before either paired
    # label mode, so shared and multiregime calls have identical vocabularies.
    if num_classes is None:
        num_classes = 2 if max_classes == 2 or rng.random() <= 0.5 else int(rng.integers(2, max_classes + 1))
    if group is None:
        if support_size is None or query_size is None:
            raise ValueError("support_size and query_size are required without a generation group.")
        if support_size < 1 or query_size < 1:
            raise ValueError("Need at least one support and one query row.")
        num_features = int(rng.integers(min_features, max_features + 1))
        prior_type = "mlp_scm" if rng.random() < _mlp_probability(mix_probs) else "tree_scm"
        hparams = _sample_tabicl_mix_scm_hparams(
            int(rng.integers(2**31)),
            num_features=num_features,
            num_outputs=num_regimes,
            num_classes=num_classes,
            mix_probs=mix_probs,
            hp_profile=hp_profile,
        )
    else:
        if support_size is not None or query_size is not None:
            raise ValueError("A generation group owns support_size and query_size.")
        num_features = group.num_features
        support_size = group.support_size
        query_size = group.query_size
        prior_type = group.prior_type
        hparams = {**group.hparams, "num_classes": num_classes}
        if min_samples_per_regime is None:
            min_samples_per_regime = group.min_samples_per_regime
        hp_profile = group.hp_profile
        if num_regimes != group.num_regimes:
            raise ValueError("num_regimes must match the generation group's sampled regime count.")
        if hparams.get("num_features") != num_features or hparams.get("num_outputs") != num_regimes:
            raise ValueError("Generation-group hparams do not match this episode's dimensions.")

    k = num_regimes
    if min_samples_per_regime is None:
        min_samples_per_regime = 32
    if min_samples_per_regime < 1:
        raise ValueError("min_samples_per_regime must be positive.")
    if k > 1 and support_size < k * min_samples_per_regime:
        raise ValueError("Support size cannot provide min_samples_per_regime evidence for every regime.")
    if family == "persistent" and active_groups < k:
        raise ValueError("persistent tasks need at least one latent group per regime.")

    if pad_features is None:
        pad_features = max_features + 1
    real_features = num_features + int(expose_z)
    if pad_features < real_features:
        raise ValueError("pad_features is too small for this episode.")

    rows = support_size + query_size
    sup = slice(0, support_size)
    qry = slice(support_size, rows)

    def _route(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assign every support/query row to a routing regime from the processed X (and rng for persistent)."""
        routing_scores = x[:, 0]
        if family == "persistent":
            # Each latent group has a persistent regime identity. Seed the support
            # allocation with min_samples_per_regime examples of every identity so
            # the configured evidence floor is realised, not only used as a cap.
            group_z = np.concatenate((np.arange(k), rng.integers(k, size=active_groups - k)))
            rng.shuffle(group_z)
            representatives = np.array([np.flatnonzero(group_z == regime)[0] for regime in range(k)])
            support_groups = np.concatenate(
                (
                    np.repeat(representatives, min_samples_per_regime),
                    rng.integers(active_groups, size=support_size - k * min_samples_per_regime),
                )
            )
            rng.shuffle(support_groups)
            query_groups = rng.integers(active_groups, size=query_size)
            return group_z[support_groups], group_z[query_groups]
        # A feature-derived selector makes multiregime structure potentially
        # discoverable without inserting an unobserved SCM output into X. Rank
        # bins on support rows guarantee the configured evidence floor; query
        # rows use the corresponding observed-score boundaries.
        support_scores = routing_scores[sup]
        if k > 1:
            ordered_support = np.argsort(support_scores, kind="stable")
            routing_support_z = np.empty(support_size, dtype=np.int64)
            routing_support_z[ordered_support] = np.arange(support_size) * k // support_size
            boundaries = support_scores[ordered_support][np.arange(1, k) * support_size // k]
        else:
            routing_support_z = np.zeros(support_size, dtype=np.int64)
            boundaries = np.array([])
        routing_query_z = np.clip(np.searchsorted(boundaries, routing_scores[qry]), 0, k - 1)
        return routing_support_z, routing_query_z

    # Native labels (default): the episode is a TabICL dataset — native SCM
    # draw, native Reg2Cls features and labels, native split sanity check —
    # and each additional regime applies TabICL's label rule once more (g_z: on
    # the same score; r_z: on its own SCM score column).  Only a requested
    # class_ratio (evaluation-bank cells) or the deprecated production profile
    # switches to the calibrated label grid below.
    native_base = class_ratio is None and hp_profile != "production"
    if native_base:
        # The native draw enforces TabICL's split sanity check on rule 0.  The
        # multiregime selection mixes rules, so it is re-checked here; if it
        # fails, the additional rules are redrawn (X and rule 0 kept) and, as a
        # last resort, the base is redrawn with the next seed.  Both checks are
        # made for the multiregime selection regardless of rule_mode, so the
        # shared and multiregime episodes of one seed accept the same base and
        # rules and stay paired.  Evaluation cells additionally require every
        # requested class in both halves.
        accepted = False
        for base_attempt in range(_NATIVE_LABEL_ATTEMPTS):
            base = _draw_native_tabicl_base(
                seed + base_attempt * 7_919,
                num_features=num_features,
                rows=rows,
                support_size=support_size,
                mix_probs=mix_probs,
                hp_profile=hp_profile,
                shared_hparams=hparams if group is not None else None,
                shared_prior_type=prior_type if group is not None else None,
                num_classes=num_classes,
                num_outputs=k if mechanism_mode == "r_z" else 1,
            )
            x = base.x.detach().cpu().numpy()
            if not np.isfinite(x).all():
                continue  # native feature processing can emit NaN on degenerate columns; redraw
            routing_support_z, routing_query_z = _route(x)
            rule0 = base.y.numpy()

            def _ok(support_labels: np.ndarray, query_labels: np.ndarray) -> bool:
                return _split_is_valid(support_labels, query_labels) and (
                    not require_num_classes or np.unique(support_labels).size == num_classes
                )

            if not _ok(rule0[sup], rule0[qry]):
                continue
            for _rule_attempt in range(8):
                # The additional rules use TabICL's global samplers; seed them from the
                # episode seed so labels are a deterministic function of (seed, attempt).
                with _temporary_global_seed((seed + base_attempt * 7_919) * 8 + _rule_attempt + 1):
                    x_tensor, labels = _tabicl_label_rules_from_native_base(base, k, mechanism_mode=mechanism_mode)
                candidate_labels = labels.detach().cpu().numpy().astype(np.int64)
                mr_support = candidate_labels[sup][np.arange(support_size), routing_support_z]
                mr_query = candidate_labels[qry][np.arange(query_size), routing_query_z]
                if _ok(mr_support, mr_query):
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            raise RuntimeError(
                f"Could not draw a native episode with valid labels in both rule modes within "
                f"{_NATIVE_LABEL_ATTEMPTS} base draws at seed {seed}."
            )
        prior_type, hparams = base.prior_type, base.params
        scm_metadata = _scm_metadata(base.model, hparams, prior_type)
        raw_rule_scores = base.raw_y.detach().cpu().numpy()
        raw_rule_scores = raw_rule_scores if raw_rule_scores.ndim == 2 else raw_rule_scores[:, None]
        class_probabilities = None
        calibration_rows = 0
    else:
        # r_z(X) has one SCM score per regime; g_z(X) has one shared score
        # whose copies receive distinct prevalence-preserving label maps.
        # As in native rank-based Reg2Cls, thresholds are fitted on the whole
        # generated episode (support plus query); no query labels are exposed
        # to the model, and support/query then share the same fixed rule.
        class_probabilities = _target_class_probabilities(num_classes, class_ratio)
        calibration_rows = 0
        rows = calibration_rows + support_size + query_size
        torch.manual_seed(seed % (2**63 - 1))
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        model_cls = MLPSCM if prior_type == "mlp_scm" else TreeSCM
        score_hparams = {**hparams, "num_outputs": 1 if mechanism_mode == "g_z" else k}
        for _attempt in range(32):
            with torch.no_grad():
                model = model_cls(seq_len=rows, **score_hparams)
                x_tensor, candidates = model()
            if torch.isfinite(x_tensor).all() and torch.isfinite(candidates).all():
                break
        else:
            raise RuntimeError(f"Could not draw a finite r_z SCM batch within 32 attempts at seed {seed}.")

        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(1)
        expected_scores = 1 if mechanism_mode == "g_z" else k
        if candidates.ndim != 2 or candidates.shape[1] != expected_scores:
            raise RuntimeError("SCM output has an unexpected number of candidate score columns.")
        scm_metadata = _scm_metadata(model, score_hparams, prior_type)
        # Retain TabICL's observed-X preprocessing while replacing its
        # unconstrained labels with controlled class probabilities below.
        x_tensor, _ = Reg2Cls(hparams)(x_tensor, candidates[:, 0])
        raw_rule_scores = candidates.detach().cpu().numpy()
        sup = slice(calibration_rows, calibration_rows + support_size)
        qry = slice(calibration_rows + support_size, rows)

    x = x_tensor.detach().cpu().numpy()
    if not native_base:
        routing_support_z, routing_query_z = _route(x)

    selected_rules_support = routing_support_z if rule_mode == "multiregime" else np.zeros_like(routing_support_z)
    selected_rules_query = routing_query_z if rule_mode == "multiregime" else np.zeros_like(routing_query_z)
    if native_base:
        support_y = candidate_labels[sup][np.arange(support_size), selected_rules_support]
        query_y = candidate_labels[qry][np.arange(query_size), selected_rules_query]
    else:
        reference_indices = np.concatenate(
            (np.arange(sup.start, sup.stop), np.arange(qry.start, qry.stop))
        )
        calibration_z = np.concatenate((routing_support_z, routing_query_z))
        label_grid = _controlled_label_grid(
            raw_rule_scores,
            calibration_scores=raw_rule_scores[reference_indices],
            calibration_indices=reference_indices,
            calibration_z=calibration_z,
            tie_breaker=_observed_x_tie_breaker(x),
            num_regimes=k,
            class_probabilities=class_probabilities,
            mechanism_mode=mechanism_mode,
        )
        support_y = label_grid[sup][np.arange(support_size), routing_support_z, selected_rules_support]
        query_y = label_grid[qry][np.arange(query_size), routing_query_z, selected_rules_query]

    if rule_mode == "shared":
        support_z = np.zeros_like(routing_support_z)
        query_z = np.zeros_like(routing_query_z)
    else:
        support_z, query_z = routing_support_z, routing_query_z

    if expose_z:
        routing_scores = x[:, 0]  # the soft-gate routing score is the first processed feature
        d = num_features + 1
        column_order = rng.permutation(d)
        z_column_index = int(np.where(column_order == num_features)[0][0])
        real_support_x = np.concatenate((x[sup], routing_scores[sup, None]), axis=1)[:, column_order]
        real_query_x = np.concatenate((x[qry], routing_scores[qry, None]), axis=1)[:, column_order]
    else:
        d = num_features
        z_column_index = None
        real_support_x, real_query_x = x[sup], x[qry]

    scm_metadata.update(
        {
            "family": family,
            "rule_mode": rule_mode,
            "mechanism_mode": mechanism_mode,
            "hp_profile": hp_profile,
            "native_k1_equivalence": bool(k == 1 and native_base and base_attempt == 0),
            "native_base_attempt": base_attempt if native_base else None,
            "num_regimes": k,
            "num_classes": num_classes,
            "candidate_num_classes": (
                [int(np.unique(rule).size) for rule in candidate_labels.T]
                if native_base
                else [int(np.unique(label_grid[:, :, rule]).size) for rule in range(k)]
            ),
            "observed_support_num_classes": int(np.unique(support_y).size),
            "observed_query_num_classes": int(np.unique(query_y).size),
            "label_prior_controlled": not native_base,
            "target_class_probabilities": None if class_probabilities is None else class_probabilities.tolist(),
            "class_probability_scope": (
                "native_reg2cls" if native_base else "conditional_on_routing_regime"
            ),
            "label_reference": "native_reg2cls" if native_base else "full_episode_by_routing_regime",
            "label_reference_rows_by_routing_regime": (
                None
                if native_base
                else np.bincount(
                    np.concatenate((routing_support_z, routing_query_z)), minlength=k
                ).astype(int).tolist()
            ),
            "deprecated_calibration_size_argument": calibration_size,
            "support_class_counts_by_routing_regime": _class_counts_by_routing_regime(
                support_y,
                routing_support_z,
                num_regimes=k,
                num_classes=num_classes,
            ),
            "query_class_counts_by_routing_regime": _class_counts_by_routing_regime(
                query_y,
                routing_query_z,
                num_regimes=k,
                num_classes=num_classes,
            ),
        }
    )

    support_x = np.zeros((support_size, pad_features), dtype=np.float32)
    query_x = np.zeros((query_size, pad_features), dtype=np.float32)
    support_x[:, :d] = real_support_x
    query_x[:, :d] = real_query_x

    return V4Episode(
        support_x=torch.from_numpy(support_x)[None],
        support_y=torch.from_numpy(support_y.astype(np.float32))[None],
        query_x=torch.from_numpy(query_x)[None],
        query_y=torch.from_numpy(query_y.astype(np.float32))[None],
        support_z=support_z,
        query_z=query_z,
        d=d,
        num_classes=num_classes,
        prior_type=prior_type,
        scm_metadata=scm_metadata,
        z_column_index=z_column_index,
    )
