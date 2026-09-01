"""Does the representation the slot competition reads carry the regime at all?

Every mechanism tried in this project scores a support row's state ``h_n``
against a slot: the dot product, the label likelihood, the mixture readout.  All
of them are downstream of ``h_n``.  So if ``h_n`` does not contain the regime
distinction, no compatibility function, no loss and no slot count can recover
it, and every negative result so far has a single cause that none of those
experiments could see.

This measures that directly and supervised.  Take a trained checkpoint, run a
validation episode, pull the support row states out of the deepest slot layer,
and cross-fit a classifier ``h_n -> regime tag``.

    high AUC   the information is there; the failure is the competition or the
               objective, which is a fixable and much sharper target
    AUC ~ 0.5  the representation does not carry the distinction, and nothing
               downstream of it ever could

It is deliberately given the tags, which is exactly what a model must never
receive.  **This is a diagnostic upper bound, never a method**, and its number
is not a model result -- reporting it as one would be the supervision defect
this project removed from the losses in the first place.

Two references are printed alongside, because the probe number means nothing
without them:

``raw features``  the same probe on ``x`` instead of ``h_n``.  The backbone is
                  supposed to *add* information; if it does not beat this, it
                  has not.
``achievable``    what `detection_ceiling` measures for the same design, which
                  is what an unsupervised detector can reach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tfmplayground.experiments.continuous_episodes import TRAIN_REGIME, sample_scm_multiregime_episode
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_backbone import install_slot_layers

PROBES = {
    "linear": lambda: LogisticRegression(max_iter=2000),
    "boosted": lambda: HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15),
}


def load_backbone(path: str | Path) -> tuple[NanoTabPFNModel, dict[str, Any]]:
    """Rebuild whichever variant produced this checkpoint, as a bare backbone.

    The mixture variant stores its decoder outside the backbone, so its state
    dictionary is prefixed; only the backbone's own weights are needed here.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    architecture = state["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        # The head variant writes `backbone_num_outputs`, since its own output
        # width is the mixture's, not the backbone's.
        num_outputs=architecture.get("num_outputs", architecture.get("backbone_num_outputs", 2)),
    )
    # The head variant carries `num_slots` too but builds its slots outside the
    # backbone, so only the in-backbone kinds need layers installed.
    if architecture.get("model_kind") in ("slot_backbone", "slot_backbone_mixture"):
        install_slot_layers(
            backbone,
            num_slots=architecture["num_slots"],
            num_slot_iterations=architecture.get("num_slot_iterations", 3),
            competitive_slots=architecture.get("competitive_slots", True),
            compatibility=architecture.get("slot_compatibility", "dot"),
            max_classes=architecture.get("max_classes", 2),
        )
    weights = state["model"]
    prefix = "backbone."
    if any(key.startswith(prefix) for key in weights):
        weights = {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)}
    backbone.load_state_dict(weights, strict=False)
    return backbone.eval(), architecture


def cross_fit_auc(features: np.ndarray, tag: np.ndarray, probe: str, folds: int = 5, seed: int = 0) -> float:
    """Supervised AUC for ``features -> tag``, predicted out of fold.

    Cross-fitting is not optional even here: an in-fold fit on 192-dimensional
    states memorizes 128 rows outright and reports 1.0 regardless of content.
    """
    if len(set(tag)) < 2:
        return float("nan")
    score = np.zeros(len(tag))
    for train, test in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, tag):
        model = PROBES[probe]().fit(features[train], tag[train])
        score[test] = model.predict_proba(features[test])[:, list(model.classes_).index(1)]
    return float(roc_auc_score(tag, score))


@torch.no_grad()
def probe_checkpoint(path: str | Path, *, episodes: int = 8, seed: int = 404) -> dict[str, Any]:
    backbone, architecture = load_backbone(path)
    config = torch.load(str(path), map_location="cpu", weights_only=False).get("training_config", {})
    features = int(config.get("max_features", 12))
    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {f"{where}_{probe}": [] for where in ("state", "raw") for probe in PROBES}

    for _ in range(episodes):
        episode = sample_scm_multiregime_episode(
            rng,
            regime=TRAIN_REGIME,
            family="mlp_scm",
            batch_size=4,
            support_size=int(config.get("support_size", 128)),
            query_count=int(config.get("query_size", 32)),
            noise=0.0,
            contamination=float(config.get("multiregime_contamination", 0.3)),
            regime_coherence=float(config.get("regime_coherence", 0.0)),
            num_classes=int(config.get("max_classes", 2)),
        )
        split = episode.support_x.shape[1]
        table = torch.cat((episode.support_x, episode.query_x), dim=1)
        encoded = backbone.encode_table((table, episode.support_y), train_test_split_index=split)
        # The target column at support rows: exactly what the competition reads.
        states = encoded[:, :split, -1, :]
        for batch in range(states.shape[0]):
            tag = episode.support_regime_source[batch].numpy().astype(int)
            for probe in PROBES:
                results[f"state_{probe}"].append(cross_fit_auc(states[batch].numpy(), tag, probe))
                results[f"raw_{probe}"].append(cross_fit_auc(episode.support_x[batch].numpy(), tag, probe))

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.array([v for v in values if not np.isnan(v)])
        if not array.size:
            return {"mean": float("nan"), "ci": float("nan"), "n": 0}
        return {
            "mean": float(array.mean()),
            "ci": float(1.96 * array.std() / np.sqrt(array.size)),
            "n": int(array.size),
        }

    return {
        "checkpoint": str(path),
        "model_kind": architecture.get("model_kind", "vanilla" if "num_slots" not in architecture else "slot"),
        "max_classes": int(config.get("max_classes", 2)),
        "features": features,
        "contamination": float(config.get("multiregime_contamination", 0.3)),
        **{name: summarize(values) for name, values in results.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=404)
    return parser


def main(argv: list[str] | None = None) -> str:
    """Print each checkpoint's row as it finishes.

    Accumulating and printing once at the end means a run killed partway --
    which is what a detached remote process does when its session ends --
    produces nothing at all, discarding every checkpoint already probed.
    """
    arguments = build_parser().parse_args(argv)
    lines = []
    for path in arguments.checkpoints:
        line = json.dumps(probe_checkpoint(path, episodes=arguments.episodes, seed=arguments.seed))
        print(line, flush=True)
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
