"""Does resampling the SCM every episode make regime identification unlearnable?

Earlier work on the multi-regime prior (``detection_ceiling.py``,
``representation_probe.py``) traced several negative results to
identifiability: at ``regime_coherence=0.0`` the regime tag is provably
independent of ``x`` by construction, and even the oracle ceiling on the
contamination-detection framing was only about 0.54.  Those issues are
orthogonal to a different question: even once ``regime_coherence > 0`` makes
``Z`` a real function of ``X``, every multiregime episode redraws a brand-new
SCM (``_scm_candidates`` samples fresh parameters and instantiates a fresh
``MLPSCM``/``TreeSCM`` every call).  If the regime-defining rule
``g^{(e)}: X -> Z`` has no stable form across episodes, an in-context learner
has to solve a harder joint problem -- infer which rule this episode uses,
then apply it -- than a model fit to one dataset ever does.

This measures that axis directly, holding coherence fixed and varying only how
much the *rule generating Z* changes from one episode to the next:

``fixed``              the same deterministic rule every episode
                        (``Z = 1[x0 > 0]``): the sanity check.  If a model
                        fails here, the diversity ladder is not the problem --
                        look at the architecture or training loop instead.
``single_family``      family fixed (``mlp_scm``), instance (graph and
                        coefficients) resampled every episode.
``restricted_family``  family drawn each episode from a 2-member subset.
``resampled``           family drawn each episode from the full set (both
                        families plus their cross-family combination) --
                        today's actual multiregime training distribution.

At each rung, a small nanoTabPFN is trained from scratch to predict the regime
tag ``Z`` in context (support ``(x, z)`` -> query ``z``), then compared against
a per-episode ``HistGradientBoostingClassifier`` fit fresh on that episode's
own support set -- the "conventional ML only has to fit one dataset" baseline.
If ``fixed``/``single_family`` track the RF baseline but ``restricted_family``/
``resampled`` collapse toward chance while RF stays high, episode-level SCM
diversity -- not residual unidentifiability -- is the bottleneck.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.continuous_episodes import (
    SCM_FAMILIES,
    TRAIN_REGIME,
    ContinuousEpisode,
    EpisodeRegime,
    sample_scm_multiregime_episode,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed

Rung = Literal["fixed", "single_family", "restricted_family", "resampled"]
RUNGS: tuple[Rung, ...] = ("fixed", "single_family", "restricted_family", "resampled")

#: Two of the three multiregime sources, leaving the cross-family combination
#: out -- the missing rung between "one family" and "everything".
_RESTRICTED_SOURCES: tuple[str, ...] = SCM_FAMILIES
#: All three sources ``multiregime_batch`` (pretrain_plain_nanotabpfn.py)
#: draws from: both single families plus their cross-family combination.
_FULL_SOURCES: tuple[str | tuple[str, str], ...] = (*SCM_FAMILIES, ("mlp_scm", "tree_scm"))


def fixed_rule_episode(
    rng: np.random.Generator,
    *,
    batch_size: int,
    support_size: int,
    query_count: int,
    features: int = 4,
) -> ContinuousEpisode:
    """The sanity-check rung: ``Z = 1[x0 > 0]``, identical every episode.

    Only ``x`` varies from call to call; the rule itself never does.  Shaped
    like ``random_label_episode`` but with a fixed, learnable rule instead of
    independent noise -- a model that cannot learn this cannot learn any rung.
    """
    support_x = rng.normal(size=(batch_size, support_size, features)).astype(np.float32)
    query_x = rng.normal(size=(batch_size, query_count, features)).astype(np.float32)
    # Support labels feed the model's target encoder, which means-pools them,
    # so -- exactly like every other episode builder in this repo -- they must
    # be float; only the query target (a cross-entropy class index) is int.
    support_z = (support_x[:, :, 0] > 0).astype(np.float32)
    query_z = (query_x[:, :, 0] > 0).astype(np.int64)
    placeholder_query = np.full((batch_size, 1, query_count), 0.5, dtype=np.float32)
    placeholder_support = np.full((batch_size, 1, support_size), 0.5, dtype=np.float32)
    return ContinuousEpisode(
        torch.from_numpy(support_x),
        torch.from_numpy(support_z),
        torch.from_numpy(query_x),
        torch.from_numpy(query_z),
        torch.from_numpy(placeholder_query),
        torch.from_numpy(placeholder_support),
        torch.ones(batch_size, 1),
        0.0,
        "fixed_rule_diagnostic",
        "fixed",
    )


def _regime_target(episode: ContinuousEpisode) -> ContinuousEpisode:
    """Swap the model's label for the diagnostic-only regime tag.

    ``ContinuousEpisode`` is a plain dataclass, so this is a direct field
    substitution: nothing about ``continuous_episodes.py`` needs to change to
    train against ``Z`` instead of ``Y``.
    """
    return dataclasses.replace(
        episode,
        # Support labels feed the model's target encoder, which means-pools
        # them, so -- like every other episode builder here -- they must be
        # float; the query target only needs to be an integer class index.
        support_y=episode.support_regime_source.float(),
        query_y=episode.query_regime_source.long(),
    )


def sample_rung_episode(
    rung: Rung,
    rng: np.random.Generator,
    *,
    batch_size: int,
    support_size: int,
    query_count: int,
    features: int,
    contamination: float,
    regime_coherence: float,
) -> ContinuousEpisode:
    if rung == "fixed":
        return fixed_rule_episode(
            rng, batch_size=batch_size, support_size=support_size, query_count=query_count, features=features
        )
    if rung == "single_family":
        family: str | tuple[str, str] = "mlp_scm"
    elif rung == "restricted_family":
        family = str(rng.choice(_RESTRICTED_SOURCES))
    elif rung == "resampled":
        sources = _FULL_SOURCES
        family = sources[int(rng.integers(len(sources)))]
    else:
        raise ValueError(f"Unknown rung {rung!r}")
    # TRAIN_REGIME draws a random feature count per episode (2-12); pin it so
    # every rung is compared at the same table width.
    regime = EpisodeRegime(TRAIN_REGIME.families, features, features, TRAIN_REGIME.imbalance_range, TRAIN_REGIME.scale_exponent)
    episode = sample_scm_multiregime_episode(
        rng,
        regime=regime,
        family=family,
        batch_size=batch_size,
        support_size=support_size,
        query_count=query_count,
        noise=0.0,
        contamination=contamination,
        regime_coherence=regime_coherence,
        num_classes=2,
    )
    return _regime_target(episode)


def query_loss(model: NanoTabPFNModel, episode: ContinuousEpisode) -> torch.Tensor:
    logits = model(episode.support_x, episode.support_y, episode.query_x)
    classes = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, classes), episode.query_y.reshape(-1).long())


def train_rung(
    rung: Rung,
    *,
    steps: int,
    batch_size: int,
    support_size: int,
    query_count: int,
    features: int,
    contamination: float,
    regime_coherence: float,
    seed: int,
    device: str = "cpu",
) -> NanoTabPFNModel:
    set_randomness_seed(seed)
    model = NanoTabPFNModel(
        num_layers=3,
        embedding_size=64,
        num_attention_heads=4,
        mlp_hidden_size=256,
        num_outputs=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(steps):
        episode = sample_rung_episode(
            rung,
            rng,
            batch_size=batch_size,
            support_size=support_size,
            query_count=query_count,
            features=features,
            contamination=contamination,
            regime_coherence=regime_coherence,
        ).to(device)
        optimizer.zero_grad()
        loss = query_loss(model, episode)
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def evaluate_rung(
    model: NanoTabPFNModel,
    rung: Rung,
    *,
    episodes: int,
    batch_size: int,
    support_size: int,
    query_count: int,
    features: int,
    contamination: float,
    regime_coherence: float,
    seed: int,
    device: str = "cpu",
) -> dict[str, float]:
    """In-context nanoTabPFN AUC vs. a per-episode RF fit, on held-out episodes.

    The RF is fit fresh on each episode's own support set and scored on that
    same episode's query set -- no cross-episode generalization required, so
    it is the "conventional ML only has to fit one dataset" reference point.
    """
    rng = np.random.default_rng(seed)
    model_scores, rf_scores = [], []
    for _ in range(episodes):
        episode = sample_rung_episode(
            rung,
            rng,
            batch_size=batch_size,
            support_size=support_size,
            query_count=query_count,
            features=features,
            contamination=contamination,
            regime_coherence=regime_coherence,
        )
        logits = model(episode.support_x.to(device), episode.support_y.to(device), episode.query_x.to(device))
        probs = torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()
        for batch in range(episode.support_x.shape[0]):
            query_tag = episode.query_y[batch].numpy()
            if len(set(query_tag.tolist())) < 2:
                continue
            model_scores.append(roc_auc_score(query_tag, probs[batch]))
            support_x = episode.support_x[batch].numpy()
            support_tag = episode.support_y[batch].numpy()
            if len(set(support_tag.tolist())) < 2:
                continue
            rf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(support_x, support_tag)
            rf_probs = rf.predict_proba(episode.query_x[batch].numpy())[:, list(rf.classes_).index(1)]
            rf_scores.append(roc_auc_score(query_tag, rf_probs))
    return {
        "model_auc": float(np.mean(model_scores)) if model_scores else float("nan"),
        "rf_auc": float(np.mean(rf_scores)) if rf_scores else float("nan"),
        "n_model": len(model_scores),
        "n_rf": len(rf_scores),
    }


def run_ladder(
    *,
    steps: int,
    train_batch_size: int,
    eval_episodes: int,
    eval_batch_size: int,
    support_size: int,
    query_count: int,
    features: int,
    contamination: float,
    regime_coherence: float,
    seed: int,
    device: str = "cpu",
) -> list[dict]:
    rows = []
    for rung in RUNGS:
        model = train_rung(
            rung,
            steps=steps,
            batch_size=train_batch_size,
            support_size=support_size,
            query_count=query_count,
            features=features,
            contamination=contamination,
            regime_coherence=regime_coherence,
            seed=seed,
            device=device,
        )
        metrics = evaluate_rung(
            model,
            rung,
            episodes=eval_episodes,
            batch_size=eval_batch_size,
            support_size=support_size,
            query_count=query_count,
            features=features,
            contamination=contamination,
            regime_coherence=regime_coherence,
            seed=seed + 1,
            device=device,
        )
        row = {
            "rung": rung,
            "steps": steps,
            "regime_coherence": regime_coherence,
            "gap": metrics["rf_auc"] - metrics["model_auc"],
            **metrics,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--contamination", type=float, default=0.3)
    parser.add_argument("--regime-coherence", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    arguments = build_parser().parse_args(argv)
    return run_ladder(
        steps=arguments.steps,
        train_batch_size=arguments.train_batch_size,
        eval_episodes=arguments.eval_episodes,
        eval_batch_size=arguments.eval_batch_size,
        support_size=arguments.support_size,
        query_count=arguments.query_size,
        features=arguments.features,
        contamination=arguments.contamination,
        regime_coherence=arguments.regime_coherence,
        seed=arguments.seed,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
