"""Standalone multiregime evaluation: bigger query set, per-family breakdown, baseline vs. fine-tuned.

Decoupled from `finetune_multiregime_backbone.py`'s training loop, which only
ever reported its multiregime diagnostics as a training-time side effect:
`query_count=6` (too coarse -- `round(query_count * contamination)` collapses
`contamination in (0.1, 0.5)` to just {0, 1, 2, 3} contaminated query rows),
one aggregate number per run, and only ever on one mixture source (`mlp_scm`
in training, `tree_scm` in validation).

This script fixes all three: a much larger query set, contamination swept at
fixed controlled levels rather than left to the training-time random draw, a
per-mixture-source breakdown (both SCM families plus the five analytic family
pairs from `paper/tabpfn_multi_regime_results.md`, so the backbone fine-tune
is directly comparable to that table), and the untouched baseline checkpoint
evaluated side by side with every fine-tuned checkpoint in the same run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from tfmplayground.experiments.continuous_episodes import (
    ANALYTIC_FAMILIES,
    HELDOUT_REGIME,
    TRAIN_REGIME,
    ContinuousEpisode,
    EpisodeRegime,
    _build_multiregime_item,
    _stack,
    sample_scm_multiregime_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

#: The five analytic family pairs inspected by hand in
#: paper/tabpfn_multi_regime_results.md, reused here so the backbone
#: fine-tune's numbers land in the same table.
ANALYTIC_SOURCE_PAIRS = (
    ("linear", "tree"),
    ("linear", "dense_interaction"),
    ("threshold", "smooth"),
    ("sparse_interaction", "dense_interaction"),
    ("smooth", "tree"),
)
CONTAMINATIONS = (0.0, 0.2, 0.4)


def _fixed_pair_episode(
    rng: np.random.Generator,
    *,
    base_family: str,
    other_family: str,
    regime: EpisodeRegime,
    batch_size: int,
    support_size: int,
    query_count: int,
    noise: float | None,
    contamination: float,
    device: str,
) -> ContinuousEpisode:
    """Like sample_multiregime_episode, but for one caller-chosen family pair
    rather than a randomly drawn one -- needed for a controlled per-pair sweep.
    """
    noise = float(rng.choice((0.0, 0.05, 0.1))) if noise is None else noise
    effective_noise = max(noise, 1e-3)
    features = int(rng.integers(regime.min_features, regime.max_features + 1))
    items = [
        _build_multiregime_item(
            regime,
            rng,
            base_family=base_family,
            other_family=other_family,
            support_size=support_size,
            query_count=query_count,
            noise=effective_noise,
            features=features,
            contamination=contamination,
        )
        for _ in range(batch_size)
    ]
    episode = ContinuousEpisode(
        _stack(items, "support_x"),
        _stack(items, "support_y"),
        _stack(items, "query_x"),
        _stack(items, "query_y"),
        _stack(items, "candidate_query_positive"),
        _stack(items, "candidate_support_positive"),
        _stack(items, "posterior"),
        effective_noise,
        "multiregime",
        f"{base_family}+{other_family}",
        {"support_size": support_size, "query_count": query_count, "features": features, "contamination": contamination},
        _stack(items, "query_regime_source"),
    )
    return episode.to(device)


def _make_source(name: str) -> Callable[..., ContinuousEpisode]:
    """One callable per mixture source, sharing a uniform (rng, regime, ..., contamination) signature."""
    if name == "mlp_scm":

        def source(rng, *, regime, batch_size, support_size, query_count, contamination, device):
            del regime  # fixed to TRAIN_REGIME for this source, not the caller's choice
            return sample_scm_multiregime_episode(
                rng,
                regime=TRAIN_REGIME,
                batch_size=batch_size,
                support_size=support_size,
                query_count=query_count,
                contamination=contamination,
                device=device,
            )

        return source
    if name == "tree_scm":

        def source(rng, *, regime, batch_size, support_size, query_count, contamination, device):
            del regime  # fixed to HELDOUT_REGIME for this source, not the caller's choice
            return sample_scm_multiregime_episode(
                rng,
                regime=HELDOUT_REGIME,
                batch_size=batch_size,
                support_size=support_size,
                query_count=query_count,
                contamination=contamination,
                device=device,
            )

        return source
    base_family, other_family = name.split("+")
    if base_family not in ANALYTIC_FAMILIES or other_family not in ANALYTIC_FAMILIES:
        raise ValueError(f"Unknown mixture source {name!r}.")

    def source(rng, *, regime, batch_size, support_size, query_count, contamination, device):
        return _fixed_pair_episode(
            rng,
            base_family=base_family,
            other_family=other_family,
            regime=regime,
            batch_size=batch_size,
            support_size=support_size,
            query_count=query_count,
            noise=None,
            contamination=contamination,
            device=device,
        )

    return source


SOURCE_NAMES = ("mlp_scm", "tree_scm", *(f"{a}+{b}" for a, b in ANALYTIC_SOURCE_PAIRS))


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 4404
    device: str = "cpu"
    query_count: int = 32
    support_size: int = 128
    episodes_per_cell: int = 20


def bootstrap_auc_acc(
    p: np.ndarray, y: np.ndarray, *, iterations: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, dict[str, float] | None]:
    """Nonparametric bootstrap over pooled query rows; same construction as regime_pairs_array.py."""
    if np.unique(y).size < 2 or len(y) < 4:
        return {"auc": None, "accuracy": None}
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, accs = [], []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        if np.unique(yb).size < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
        accs.append(accuracy_score(yb, (pb >= 0.5).astype(int)))
    tail = (1 - level) / 2

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values)
        return {"mean": float(array.mean()), "lower": float(np.quantile(array, tail)), "upper": float(np.quantile(array, 1 - tail))}

    return {"auc": summarize(aucs), "accuracy": summarize(accs)}


@torch.no_grad()
def predict_positive(model: NanoTabPFNModel, episode: ContinuousEpisode) -> np.ndarray:
    logits = model(episode.support_x, episode.support_y, episode.query_x)[..., :2]
    return logits.softmax(dim=-1)[..., 1].cpu().numpy()


def evaluate_cell(
    model: NanoTabPFNModel,
    source: Callable[..., ContinuousEpisode],
    regime: EpisodeRegime,
    contamination: float,
    config: EvaluationConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    positive_all, label_all, regime_source_all = [], [], []
    for episode_index in range(config.episodes_per_cell):
        episode = source(
            rng,
            regime=regime,
            batch_size=1,
            support_size=config.support_size,
            query_count=config.query_count,
            contamination=contamination,
            device=config.device,
        )
        positive = predict_positive(model, episode)[0]
        positive_all.append(positive)
        label_all.append(episode.query_y[0].cpu().numpy())
        regime_source_all.append(episode.query_regime_source[0].cpu().numpy())
        del episode_index
    positive_all = np.concatenate(positive_all)
    label_all = np.concatenate(label_all)
    regime_source_all = np.concatenate(regime_source_all)
    base_mask, other_mask = regime_source_all == 0, regime_source_all == 1
    return {
        "n_base": int(base_mask.sum()),
        "n_other": int(other_mask.sum()),
        "base": bootstrap_auc_acc(positive_all[base_mask], label_all[base_mask], seed=seed),
        "other": bootstrap_auc_acc(positive_all[other_mask], label_all[other_mask], seed=seed),
    }


def load_model(checkpoint_path: str, device: str) -> NanoTabPFNModel:
    """Load either a plain checkpoint (init_model_from_state_dict_file's format)
    or a fine-tuned backbone.pth (torch.save({"model": state_dict, ...}))."""
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "architecture" in raw:
        return init_model_from_state_dict_file(checkpoint_path).to(device).eval()
    reference = init_model_from_state_dict_file("checkpoints/nanotabpfn.pth")
    reference.load_state_dict(raw["model"])
    return reference.to(device).eval()


def run_evaluation(
    *, baseline_checkpoint: str, finetuned_checkpoints: dict[str, str], config: EvaluationConfig, output_dir: str
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = {"baseline": load_model(baseline_checkpoint, config.device)}
    for name, path in finetuned_checkpoints.items():
        models[name] = load_model(path, config.device)

    rows: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        source = _make_source(source_name)
        regime = HELDOUT_REGIME if source_name in ("tree_scm",) else TRAIN_REGIME
        for contamination in CONTAMINATIONS:
            for model_name, model in models.items():
                seed = config.seed + hash((source_name, contamination, model_name)) % 10_000
                cell = evaluate_cell(model, source, regime, contamination, config, seed)
                row = {"source": source_name, "contamination": contamination, "model": model_name, **cell}
                rows.append(row)
                base_auc = (cell["base"]["auc"] or {}).get("mean")
                other_auc = (cell["other"]["auc"] or {}).get("mean")
                print(
                    f"{source_name:>22} c={contamination:.1f} {model_name:>10} "
                    f"base_auc={base_auc} other_auc={other_auc}",
                    flush=True,
                )

    (output / "results.json").write_text(json.dumps(rows, indent=2))
    _print_ranking(rows)
    return output.resolve()


def _print_ranking(rows: list[dict[str, Any]]) -> None:
    def other_auc(row: dict[str, Any]) -> float:
        value = (row["other"]["auc"] or {}).get("mean")
        return value if value is not None else 1.0

    ranked = sorted((row for row in rows if row["contamination"] > 0), key=other_auc)
    print(f"\n=== RANKING (by other-regime AUC, worst degradation first), {len(ranked)} cells ===")
    print(f"{'rank':>4} {'source':>22} {'contam':>7} {'model':>10} {'base AUC':>9} {'other AUC':>25}")
    for rank, row in enumerate(ranked, start=1):
        base = (row["base"]["auc"] or {}).get("mean")
        other = row["other"]["auc"]
        base_str = f"{base:.3f}" if base is not None else "n/a"
        other_str = f"{other['mean']:.3f} [{other['lower']:.3f},{other['upper']:.3f}]" if other else "n/a"
        print(f"{rank:4d} {row['source']:>22} {row['contamination']:7.1f} {row['model']:>10} {base_str:>9} {other_str:>25}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = EvaluationConfig()
    parser.add_argument("--baseline-checkpoint", default="checkpoints/nanotabpfn.pth")
    parser.add_argument("--finetuned-checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--query-count", type=int, default=defaults.query_count)
    parser.add_argument("--support-size", type=int, default=defaults.support_size)
    parser.add_argument("--episodes-per-cell", type=int, default=defaults.episodes_per_cell)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    finetuned = dict(item.split("=", 1) for item in arguments.finetuned_checkpoint)
    config = EvaluationConfig(
        seed=arguments.seed,
        device=arguments.device,
        query_count=arguments.query_count,
        support_size=arguments.support_size,
        episodes_per_cell=arguments.episodes_per_cell,
    )
    print(
        "Wrote multiregime backbone evaluation to "
        + str(
            run_evaluation(
                baseline_checkpoint=arguments.baseline_checkpoint,
                finetuned_checkpoints=finetuned,
                config=config,
                output_dir=arguments.output_dir,
            )
        )
    )
