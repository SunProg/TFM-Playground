"""Paired-task episodes sampled from the TabICL SCM prior.

The generator deliberately keeps the feature matrix fixed while evaluating two
independently sampled SCMs on it.  The initial support is required to be nearly
compatible with both candidates; later rows must separate them.
"""

from __future__ import annotations

import copy
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from tfmplayground.experiments.train_sequential_latent_filter import (
    SequentialEpisodeBatch,
)


_H5_INDEX_CACHE: dict[str, tuple[int, dict[tuple[int, int], np.ndarray], np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class PriorBimodalConfig:
    initial_support_count: int = 32
    stream_count: int = 32
    query_count: int = 4
    min_features: int = 1
    max_features: int = 16
    support_disagreement_max: float = 0.20
    stream_disagreement_min: float = 0.25
    query_disagreement_min: float = 0.25
    max_pair_attempts: int = 64
    device: str = "cpu"
    prior_type: str = "mix_scm"


def _rng_state() -> tuple[object, tuple, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _set_rng_state(state: tuple[object, tuple, torch.Tensor]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])


def _sample_params(
    prior, config: PriorBimodalConfig, total_rows: int, features: int, *, num_classes: int = 2
) -> dict:
    sampled = prior.hp_sampling()
    sampled = {key: value() if callable(value) else value for key, value in sampled.items()}
    prior_type = prior.get_prior()
    params = {
        **prior.fixed_hp,
        **sampled,
        "seq_len": total_rows,
        "train_size": config.initial_support_count,
        "max_features": features,
        "num_features": features,
        "num_classes": num_classes,
        "prior_type": prior_type,
        "device": config.device,
        # Make the paired feature transformation deterministic and shared.
        "cat_prob": 0.0,
        "permute_features": False,
        "permute_labels": False,
        "balanced": True,
        "multiclass_type": "value",
        "multiclass_ordered_prob": 1.0,
        "pre_sample_cause_stats": False,
        # With direct predictive SCMs, X is the shared cause matrix while the
        # independently initialized task networks produce different y rules.
        "is_causal": False,
    }
    return params


def _candidate_dataset(params: dict, *, seed_state: tuple[object, tuple, torch.Tensor]):
    from tabicl.prior._mlp_scm import MLPSCM
    from tabicl.prior._reg2cls import Reg2Cls
    from tabicl.prior._tree_scm import TreeSCM

    prior_cls = MLPSCM if params["prior_type"] == "mlp_scm" else TreeSCM
    # TabICL's prior initialization performs in-place initialization on
    # parameters; no gradients are needed for a data generator.
    with torch.no_grad():
        model = prior_cls(**params)
    _set_rng_state(seed_state)
    with torch.no_grad():
        x, y = model()
        x, y = Reg2Cls(params)(x, y)
    return x.float(), y.long()


def _sample_pair(prior, config: PriorBimodalConfig, rng: np.random.Generator, *, features: int):
    total_rows = config.initial_support_count + config.stream_count + config.query_count
    params_a = _sample_params(prior, config, total_rows, features)
    params_b = copy.deepcopy(params_a)

    # Construct both task networks before resetting the sampling state.  The
    # same state then produces the same X rows for both candidates.
    state = _rng_state()
    x_a, y_a = _candidate_dataset(params_a, seed_state=state)
    x_b, y_b = _candidate_dataset(params_b, seed_state=state)
    if not torch.equal(x_a, x_b):
        # Some prior implementations keep additional process-local sampling
        # state. Treat a mismatch as a rejected pair rather than constructing
        # an invalid pair with different covariates.
        return None

    support_slice = slice(0, config.initial_support_count)
    stream_slice = slice(config.initial_support_count, config.initial_support_count + config.stream_count)
    query_slice = slice(config.initial_support_count + config.stream_count, total_rows)
    support_disagreement = (y_a[support_slice] != y_b[support_slice]).float().mean()
    stream_disagreement = (y_a[stream_slice] != y_b[stream_slice]).float().mean()
    query_disagreement = (y_a[query_slice] != y_b[query_slice]).float().mean()
    if support_disagreement > config.support_disagreement_max:
        return None
    if stream_disagreement < config.stream_disagreement_min:
        return None
    if query_disagreement < config.query_disagreement_min:
        return None
    true_task = int(rng.integers(0, 2))
    labels = (y_a, y_b)[true_task]
    return x_a, y_a, y_b, labels, true_task, support_disagreement, stream_disagreement, query_disagreement


def generate_prior_bimodal_episodes(
    config: PriorBimodalConfig,
    rng: np.random.Generator,
    *,
    batch_size: int,
) -> SequentialEpisodeBatch:
    """Generate a batch of paired-task, bimodal prior episodes."""
    from tabicl.prior._dataset import SCMPrior

    if config.initial_support_count < 4 or config.stream_count < 1 or config.query_count < 1:
        raise ValueError("Episode counts must be positive and support must contain at least four rows.")
    if config.max_pair_attempts < 1:
        raise ValueError("max_pair_attempts must be positive.")
    total_rows = config.initial_support_count + config.stream_count + config.query_count
    prior = SCMPrior(
        batch_size=1,
        min_features=config.min_features,
        max_features=config.max_features,
        max_classes=2,
        min_seq_len=total_rows,
        max_seq_len=total_rows + 1,
        min_train_size=config.initial_support_count,
        max_train_size=config.initial_support_count + 1,
        prior_type=config.prior_type,
        n_jobs=1,
        device=config.device,
    )
    examples = []
    # A batch must have one feature width so it can be represented as a dense
    # tensor. Feature width still varies from batch to batch.
    features = int(rng.integers(config.min_features, config.max_features + 1))
    for _ in range(batch_size):
        for _attempt in range(config.max_pair_attempts):
            pair = _sample_pair(prior, config, rng, features=features)
            if pair is not None:
                examples.append(pair)
                break
        else:
            raise RuntimeError(
                "Could not find a valid ambiguous task pair within max_pair_attempts; "
                "relax disagreement thresholds or increase the attempt budget."
            )

    device = torch.device(config.device)
    x = torch.stack([item[0] for item in examples]).to(device)
    candidate_a = torch.stack([item[1] for item in examples]).to(device)
    candidate_b = torch.stack([item[2] for item in examples]).to(device)
    labels = torch.stack([item[3] for item in examples]).to(device)
    support_end = config.initial_support_count
    stream_end = support_end + config.stream_count
    return SequentialEpisodeBatch(
        initial_support_x=x[:, :support_end],
        initial_support_y=labels[:, :support_end].float(),
        stream_x=x[:, support_end:stream_end],
        stream_y=labels[:, support_end:stream_end].float(),
        query_x=x[:, stream_end:],
        query_y=labels[:, stream_end:],
        candidate_task=torch.tensor([item[4] for item in examples], device=device),
        candidate_support_y=torch.stack(
            [torch.stack((item[1][:support_end], item[2][:support_end])) for item in examples]
        ).to(device),
        candidate_stream_y=torch.stack(
            [torch.stack((item[1][support_end:stream_end], item[2][support_end:stream_end])) for item in examples]
        ).to(device),
        candidate_query_y=torch.stack(
            [torch.stack((item[1][stream_end:], item[2][stream_end:])) for item in examples]
        ).to(device),
        support_disagreement=torch.tensor([float(item[5]) for item in examples], device=device),
        stream_disagreement=torch.tensor([float(item[6]) for item in examples], device=device),
        query_disagreement=torch.tensor([float(item[7]) for item in examples], device=device),
    )


def generate_h5_prior_bimodal_episodes(
    path: str,
    config: PriorBimodalConfig,
    rng: np.random.Generator,
    *,
    batch_size: int,
) -> SequentialEpisodeBatch:
    """Generate paired empirical-task episodes from a prior HDF5 dump.

    The dump contains observations rather than generator parameters.  We use
    one record's X matrix and pair it with labels from another record with the
    same active feature count and evaluation position.  Candidate metadata is
    retained only for evaluation.
    """
    import h5py

    if config.initial_support_count != 32 or config.stream_count != 32 or config.query_count != 4:
        raise ValueError("The HDF5 prior episode adapter requires a 32:32:4 layout.")
    cached = _H5_INDEX_CACHE.get(path)
    if cached is None:
        with h5py.File(path, "r") as handle:
            required = {"X", "y", "num_features", "single_eval_pos"}
            missing = required.difference(handle.keys())
            if missing:
                raise ValueError(f"Prior dump is missing required datasets: {sorted(missing)}")
            num_records = int(handle["X"].shape[0])
            if handle["X"].shape[1] < 68:
                raise ValueError("Prior dump must contain at least 68 rows per episode.")
            feature_counts = np.asarray(handle["num_features"][...], dtype=np.int64)
            eval_positions = np.asarray(handle["single_eval_pos"][...], dtype=np.int64)
        groups: dict[tuple[int, int], np.ndarray] = {}
        for feature_count, eval_position in zip(feature_counts.tolist(), eval_positions.tolist()):
            if eval_position >= config.initial_support_count + config.stream_count:
                groups.setdefault((feature_count, eval_position), [])
        # Some shipped dumps contain a few records with non-finite features (300k_150x5_2.h5
        # has 5 of 300000, each with ~300 of 750 NaN/inf cells). Drawing one makes the whole
        # forward/backward NaN, which gradient clipping cannot recover, so a long run dies
        # permanently and deterministically the first time the sampler hits one. Exclude them
        # from the candidate pools up front; the scan is done once per path and cached.
        with h5py.File(path, "r") as handle:
            corrupted = []
            chunk = 20_000
            for start in range(0, num_records, chunk):
                block = handle["X"][start : start + chunk]
                mask = ~np.isfinite(block)
                if mask.any():
                    corrupted.extend((np.unique(np.where(mask)[0]) + start).tolist())
        corrupted_set = set(corrupted)
        if corrupted_set:
            warnings.warn(
                f"Excluding {len(corrupted_set)} record(s) with non-finite features from "
                f"{path}: {sorted(corrupted_set)}",
                RuntimeWarning,
                stacklevel=2,
            )
        for key in list(groups):
            candidates = np.flatnonzero((feature_counts == key[0]) & (eval_positions == key[1]))
            if corrupted_set:
                candidates = np.array(
                    [index for index in candidates.tolist() if index not in corrupted_set],
                    dtype=candidates.dtype,
                )
            groups[key] = candidates
        groups = {key: values for key, values in groups.items() if len(values) >= 2}
        cached = (num_records, groups, feature_counts, eval_positions)
        _H5_INDEX_CACHE[path] = cached

    num_records, groups, feature_counts, eval_positions = cached
    with h5py.File(path, "r") as handle:
        if not groups:
            raise ValueError("Prior dump has no compatible pairs for the requested episode layout.")

        examples = []
        attempts_by_example = []
        group_keys = list(groups)
        # Keep one active feature width per batch so the returned tensors stay
        # dense, matching the existing prior loaders. Sampling the width from
        # records preserves the dump's natural imbalance.
        batch_feature_count = int(feature_counts[int(rng.integers(num_records))])
        batch_group_keys = [key for key in group_keys if key[0] == batch_feature_count]
        if not batch_group_keys:
            batch_group_keys = group_keys
        for _ in range(batch_size):
            for attempts in range(1, config.max_pair_attempts + 1):
                key = batch_group_keys[int(rng.integers(len(batch_group_keys)))]
                indices = groups[key]
                left = int(indices[int(rng.integers(len(indices)))])
                right = int(indices[int(rng.integers(len(indices) - 1))])
                if right >= left:
                    right += 1
                x = np.asarray(handle["X"][left, :, : key[0]], dtype=np.float32)
                y_left = np.asarray(handle["y"][left], dtype=np.int64)
                y_right = np.asarray(handle["y"][right], dtype=np.int64)
                split = key[1]
                support = slice(0, config.initial_support_count)
                stream = slice(config.initial_support_count, config.initial_support_count + config.stream_count)
                query = slice(split, split + config.query_count)
                support_disagreement = float((y_left[support] != y_right[support]).mean())
                stream_disagreement = float((y_left[stream] != y_right[stream]).mean())
                query_disagreement = float((y_left[query] != y_right[query]).mean())
                if support_disagreement > config.support_disagreement_max:
                    continue
                if stream_disagreement < config.stream_disagreement_min:
                    continue
                if query_disagreement < config.query_disagreement_min:
                    continue
                true_task = int(rng.integers(2))
                examples.append(
                    (
                        x,
                        y_left,
                        y_right,
                        true_task,
                        split,
                        support_disagreement,
                        stream_disagreement,
                        query_disagreement,
                    )
                )
                attempts_by_example.append(attempts)
                break
            else:
                raise RuntimeError(
                    "Could not find a valid paired HDF5 episode within max_pair_attempts; "
                    "relax thresholds or increase the attempt budget."
                )

    device = torch.device(config.device)
    x = torch.from_numpy(np.stack([item[0] for item in examples])).to(device)
    candidate_a = torch.from_numpy(np.stack([item[1] for item in examples])).to(device)
    candidate_b = torch.from_numpy(np.stack([item[2] for item in examples])).to(device)
    candidate_labels = torch.stack((candidate_a, candidate_b), dim=1)
    task_indices = torch.tensor([item[3] for item in examples], device=device)
    labels = candidate_labels[torch.arange(batch_size, device=device), task_indices]
    support_end = config.initial_support_count
    stream_end = support_end + config.stream_count
    query_x = torch.stack([x[index, item[4] : item[4] + config.query_count] for index, item in enumerate(examples)])
    return SequentialEpisodeBatch(
        initial_support_x=x[:, :support_end],
        initial_support_y=labels[:, :support_end].float(),
        stream_x=x[:, support_end:stream_end],
        stream_y=labels[:, support_end:stream_end].float(),
        query_x=query_x,
        query_y=labels[torch.arange(batch_size, device=device), :]
        .gather(
            1,
            torch.tensor(
                [list(range(item[4], item[4] + config.query_count)) for item in examples],
                device=device,
            ),
        )
        .long(),
        candidate_task=task_indices,
        candidate_support_y=candidate_labels[:, :, :support_end],
        candidate_stream_y=candidate_labels[:, :, support_end:stream_end],
        candidate_query_y=torch.stack(
            [candidate_labels[index, :, item[4] : item[4] + config.query_count] for index, item in enumerate(examples)]
        ),
        support_disagreement=torch.tensor([item[5] for item in examples], device=device),
        stream_disagreement=torch.tensor([item[6] for item in examples], device=device),
        query_disagreement=torch.tensor([item[7] for item in examples], device=device),
        pair_attempts=torch.tensor(attempts_by_example, device=device),
    )
