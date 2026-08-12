"""Episodes whose bimodality is true by construction, not by rejection sampling.

The paired-bimodal family in :mod:`prior_bimodal_episodes` searches a dump for two
records whose labels happen to agree on the support.  Measurement showed that
produces neither ambiguity nor resolvability: the alternative's feature relevance
correlates -0.09 with the observed task's (there is no second mode to infer, only
a label vector declared to be one), what little is predictable is a selection
artifact of the disagreement filter, and the candidates differ on a mean of 1.47
of 4 query rows so the identification metric is near-powerless.

This family constructs the pair instead.  Pick a region of feature space; both
candidates follow the same base rule outside it and diverge inside it::

                       outside R              inside R
    candidate A        y_base                 y_base
    candidate B        y_base                 arm-specific

      region_flip      y_base                 y_base XOR 1     (inverted)
      second_scm       y_base                 y_other          (independent SCM)

    prior  block  <- rows OUTSIDE R    candidates agree exactly
    stream block  <- rows INSIDE  R    the evidence that separates them
    query  block  <- rows INSIDE  R    what tracking the separation buys

The prior block therefore carries *zero* information about the divergence, so the
posterior over candidates after it is genuinely uniform, and the stream block
carries exactly the evidence that resolves it.  Neither property is searched for;
both hold by construction.

Two arms exist because the shape of the alternative is itself a confound.
``region_flip`` makes every row inside the region discriminate, but its
alternative is always "the same rule, inverted", which a model could learn as a
fixed pattern.  ``second_scm`` draws the inside-region labels from an
independently sampled SCM over the *same* feature matrix, removing that pattern
at the cost of needing the on-the-fly prior.  Results that agree across both arms
are about the model; results that do not are about the generator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    _candidate_dataset,
    _rng_state,
    _sample_params,
    corrupted_record_index,
)
from tfmplayground.experiments.structural_latents import (
    StructuralLatentSchema,
    structural_feature_mask,
    structural_latent_vector,
)
from tfmplayground.experiments.train_sequential_latent_filter import SequentialEpisodeBatch

ALTERNATIVE_MODES = ("region_flip", "second_scm")
SOURCES = ("h5", "tabicl")


@dataclass(frozen=True)
class RegionSplitConfig:
    alternative_mode: str = "region_flip"
    source: str = "h5"
    initial_support_count: int = 48
    stream_count: int = 16
    query_count: int = 16
    min_features: int = 2
    max_features: int = 5
    split_quantile_range: tuple[float, float] = (0.35, 0.65)
    max_record_attempts: int = 256
    device: str = "cpu"
    prior_type: str = "mix_scm"
    compute_structural_latents: bool = True
    sequence_length: int = 512

    def validate(self) -> None:
        if self.alternative_mode not in ALTERNATIVE_MODES:
            raise ValueError(f"alternative_mode must be one of {ALTERNATIVE_MODES}.")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}.")
        if self.alternative_mode == "second_scm" and self.source != "tabicl":
            # A dump record's labels are not a function of another record's
            # features.  Pairing them is exactly the flaw that produced an
            # alternative uncorrelated with everything observable.
            raise ValueError("second_scm requires source='tabicl' so both functions share one feature matrix.")
        if self.initial_support_count < 4 or self.stream_count < 1 or self.query_count < 1:
            raise ValueError("Block sizes must be positive and the prior block needs at least four rows.")
        low, high = self.split_quantile_range
        if not 0.0 < low <= high < 1.0:
            raise ValueError("split_quantile_range must satisfy 0 < low <= high < 1.")
        if self.max_record_attempts < 1:
            raise ValueError("max_record_attempts must be positive.")
        if not 1 <= self.min_features <= self.max_features:
            raise ValueError("Feature bounds are invalid.")

    @property
    def inside_rows_needed(self) -> int:
        return self.stream_count + self.query_count


def _region_mask(x: np.ndarray, feature: int, quantile: float) -> np.ndarray:
    threshold = float(np.quantile(x[:, feature], quantile))
    return x[:, feature] > threshold


def _usable(mask: np.ndarray, labels: np.ndarray, config: RegionSplitConfig) -> bool:
    """Both sides must be big enough and the prior block must not be single-class."""
    outside = int((~mask).sum())
    inside = int(mask.sum())
    if outside < config.initial_support_count or inside < config.inside_rows_needed:
        return False
    return len(np.unique(labels[~mask])) == 2


def _base_task_from_dump(path: str, config: RegionSplitConfig, rng: np.random.Generator):
    import h5py

    corrupted = corrupted_record_index(path)
    with h5py.File(path, "r") as handle:
        records = int(handle["X"].shape[0])
        feature_counts = np.asarray(handle["num_features"][...], dtype=np.int64)
        for _ in range(config.max_record_attempts):
            index = int(rng.integers(records))
            if index in corrupted:
                continue
            features = int(feature_counts[index])
            if not config.min_features <= features <= config.max_features:
                continue
            x = np.asarray(handle["X"][index, :, :features], dtype=np.float32)
            y = np.asarray(handle["y"][index], dtype=np.int64)
            if not np.isfinite(x).all() or len(np.unique(y)) != 2:
                continue
            yield x, y, None
    return


def _base_task_from_prior(prior, config: RegionSplitConfig, rng: np.random.Generator):
    """Yield ``(x, y_base, y_other)`` with both labelings over the same features."""
    episode_config = PriorBimodalConfig(
        initial_support_count=config.initial_support_count,
        stream_count=config.stream_count,
        query_count=config.query_count,
        min_features=config.min_features,
        max_features=config.max_features,
        device=config.device,
        prior_type=config.prior_type,
    )
    for _ in range(config.max_record_attempts):
        features = int(rng.integers(config.min_features, config.max_features + 1))
        params_a = _sample_params(prior, episode_config, config.sequence_length, features)
        params_b = copy.deepcopy(params_a)
        state = _rng_state()
        x_a, y_a = _candidate_dataset(params_a, seed_state=state)
        x_b, y_b = _candidate_dataset(params_b, seed_state=state)
        if not torch.equal(x_a, x_b):
            # Some prior implementations keep extra process-local sampling state;
            # without an identical feature matrix the two labelings are not
            # comparable and the episode would be invalid.
            continue
        yield x_a.numpy(), y_a.numpy(), y_b.numpy()


def _alternative_labels(
    y_base: np.ndarray, y_other: np.ndarray | None, mask: np.ndarray, config: RegionSplitConfig
) -> np.ndarray:
    alternative = y_base.copy()
    if config.alternative_mode == "region_flip":
        alternative[mask] = 1 - y_base[mask]
    else:
        assert y_other is not None
        alternative[mask] = y_other[mask]
    return alternative


def _build_episode(x: np.ndarray, y_base: np.ndarray, y_other: np.ndarray | None, config, rng):
    """Return one episode, or ``None`` when this base task cannot support one."""
    features = x.shape[1]
    feature = int(rng.integers(features))
    quantile = float(rng.uniform(*config.split_quantile_range))
    mask = _region_mask(x, feature, quantile)
    if not _usable(mask, y_base, config):
        return None

    alternative = _alternative_labels(y_base, y_other, mask, config)
    # Only rows where the candidates actually differ can separate them.  Under
    # region_flip that is every inside row; under second_scm the two SCMs agree
    # on roughly half of them by chance.  Selecting on rows biases which rows are
    # shown, never which tasks are paired, so it cannot reintroduce the
    # task-level selection artifact this family exists to remove.
    inside = np.flatnonzero(mask & (alternative != y_base))
    outside = np.flatnonzero(~mask)
    if len(inside) < config.inside_rows_needed or len(outside) < config.initial_support_count:
        return None

    prior_rows = rng.choice(outside, size=config.initial_support_count, replace=False)
    inside_rows = rng.choice(inside, size=config.inside_rows_needed, replace=False)
    stream_rows = inside_rows[: config.stream_count]
    query_rows = inside_rows[config.stream_count :]
    rows = np.concatenate((prior_rows, stream_rows, query_rows))

    candidates = np.stack((y_base[rows], alternative[rows]))
    true_task = int(rng.integers(2))
    return x[rows], candidates, true_task, feature, quantile


@torch.no_grad()
def _identify(backbone, context_x, context_y, query_x, candidate_query_y) -> torch.Tensor:
    """Which candidate does the backbone's own query distribution favour?

    Deliberately duplicated rather than imported from
    ``run_task_posterior_local_evaluation``: that module imports this one to wire
    up the episode source, so sharing would create a cycle.
    """
    log_probabilities = backbone(context_x, context_y.float(), query_x)[..., :2].log_softmax(-1)
    scores = [
        log_probabilities.gather(-1, candidate_query_y[:, k].long()[..., None]).squeeze(-1).sum(1) for k in range(2)
    ]
    return torch.stack(scores, dim=-1).argmax(-1)


def region_split_acceptance(
    config: RegionSplitConfig,
    backbone,
    rng: np.random.Generator,
    *,
    episodes: int = 512,
    batch_size: int = 8,
    path: str | None = None,
) -> dict[str, float]:
    """Measure that the episodes are ambiguous *and* resolvable.

    The old family assumed both and was wrong about both: identification from the
    full context was 0.531 and from the support alone 0.514, so the evidence never
    separated the candidates.  Nobody checked, and every downstream result was
    built on episodes no filter could help with.  This runs before any training.

    ``prior_only_identification`` should sit at chance - the prior block is drawn
    from outside the region and carries no information about the divergence.
    ``prior_stream_identification`` should sit well above it, since the stream
    block is exactly the evidence that separates the two.
    """
    backbone.eval()
    prior_correct, updated_correct, discriminating, seen = [], [], [], 0
    support_gap, stream_gap, query_gap = [], [], []
    while seen < episodes:
        current = min(batch_size, episodes - seen)
        batch = generate_region_split_episodes(config, rng, batch_size=current, path=path)
        seen += current
        support_gap.append((batch.candidate_support_y[:, 0] != batch.candidate_support_y[:, 1]).float().mean(-1))
        stream_gap.append((batch.candidate_stream_y[:, 0] != batch.candidate_stream_y[:, 1]).float().mean(-1))
        query_gap.append((batch.candidate_query_y[:, 0] != batch.candidate_query_y[:, 1]).float().mean(-1))
        discriminating.append((batch.candidate_query_y[:, 0] != batch.candidate_query_y[:, 1]).sum(-1).float())

        prior = _identify(
            backbone, batch.initial_support_x, batch.initial_support_y, batch.query_x, batch.candidate_query_y
        )
        updated = _identify(
            backbone,
            torch.cat((batch.initial_support_x, batch.stream_x), dim=1),
            torch.cat((batch.initial_support_y, batch.stream_y), dim=1),
            batch.query_x,
            batch.candidate_query_y,
        )
        prior_correct.append((prior == batch.candidate_task).float())
        updated_correct.append((updated == batch.candidate_task).float())

    report = {
        "episodes": float(seen),
        "support_disagreement": float(torch.cat(support_gap).mean()),
        "stream_disagreement": float(torch.cat(stream_gap).mean()),
        "query_disagreement": float(torch.cat(query_gap).mean()),
        "discriminating_query_rows": float(torch.cat(discriminating).mean()),
        "prior_only_identification": float(torch.cat(prior_correct).mean()),
        "prior_stream_identification": float(torch.cat(updated_correct).mean()),
    }
    report["identification_gain"] = report["prior_stream_identification"] - report["prior_only_identification"]
    report["ambiguous"] = float(abs(report["prior_only_identification"] - 0.5) < 0.05)
    report["resolvable"] = float(report["identification_gain"] > 0.05)
    return report


def generate_region_split_episodes(
    config: RegionSplitConfig,
    rng: np.random.Generator,
    *,
    batch_size: int,
    path: str | None = None,
) -> SequentialEpisodeBatch:
    """Generate a batch of constructed-bimodal episodes."""
    config.validate()
    if config.source == "h5" and path is None:
        raise ValueError("source='h5' requires a prior dump path.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if config.source == "tabicl":
        from tabicl.prior._dataset import SCMPrior

        prior = SCMPrior(
            batch_size=1,
            min_features=config.min_features,
            max_features=config.max_features,
            max_classes=2,
            min_seq_len=config.sequence_length,
            max_seq_len=config.sequence_length + 1,
            min_train_size=config.initial_support_count,
            max_train_size=config.initial_support_count + 1,
            prior_type=config.prior_type,
            n_jobs=1,
            device=config.device,
        )

    examples = []
    while len(examples) < batch_size:
        source = (
            _base_task_from_dump(path, config, rng)
            if config.source == "h5"
            else _base_task_from_prior(prior, config, rng)
        )
        found = False
        for x, y_base, y_other in source:
            episode = _build_episode(x, y_base, y_other, config, rng)
            if episode is not None:
                examples.append(episode)
                found = True
                break
        if not found:
            raise RuntimeError(
                "Could not build a region-split episode within max_record_attempts; "
                "shrink the block sizes, widen split_quantile_range, or raise the attempt budget."
            )

    device = torch.device(config.device)
    feature_width = max(item[0].shape[1] for item in examples)
    if any(item[0].shape[1] != feature_width for item in examples):
        raise RuntimeError("A batch must share one feature width; narrow min_features/max_features.")

    x = torch.from_numpy(np.stack([item[0] for item in examples])).float().to(device)
    candidates = torch.from_numpy(np.stack([item[1] for item in examples])).long().to(device)
    task_indices = torch.tensor([item[2] for item in examples], device=device)
    labels = candidates[torch.arange(len(examples), device=device), task_indices]

    support_end = config.initial_support_count
    stream_end = support_end + config.stream_count
    zeros = torch.zeros(len(examples), device=device)
    ones = torch.ones(len(examples), device=device)

    structural_z = None
    feature_mask = None
    if config.compute_structural_latents:
        schema = StructuralLatentSchema(max_features=config.max_features)
        structural_z = torch.stack(
            [
                torch.stack(
                    tuple(
                        structural_latent_vector(
                            x[index], candidates[index, candidate], schema=schema, family=None, class_count=2
                        )
                        for candidate in range(2)
                    )
                )
                for index in range(len(examples))
            ]
        ).to(device)
        feature_mask = structural_feature_mask(feature_width, schema=schema).to(device).expand(len(examples), -1)

    return SequentialEpisodeBatch(
        initial_support_x=x[:, :support_end],
        initial_support_y=labels[:, :support_end].float(),
        stream_x=x[:, support_end:stream_end],
        stream_y=labels[:, support_end:stream_end].float(),
        query_x=x[:, stream_end:],
        query_y=labels[:, stream_end:],
        candidate_task=task_indices,
        candidate_support_y=candidates[:, :, :support_end],
        candidate_stream_y=candidates[:, :, support_end:stream_end],
        candidate_query_y=candidates[:, :, stream_end:],
        # Exact by construction rather than thresholded: the prior block is drawn
        # from outside the region and the later blocks from inside it.
        support_disagreement=zeros,
        stream_disagreement=ones,
        query_disagreement=ones,
        candidate_structural_z=structural_z,
        structural_feature_mask=feature_mask,
    )
