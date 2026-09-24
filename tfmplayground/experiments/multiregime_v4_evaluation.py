"""Fixed, factorial validation and test banks for multiregime-v4.

Every bank contains the full Cartesian product of support-set size, feature
count, regime count, requested positive-class ratio, and task family. Each
cell contains ``episodes_per_cell`` independently seeded episodes. Validation
and test use the same cells but disjoint seed roots, making both aggregate and
slice-level comparisons reproducible.

The HDF5 file stores padded tensors and one metadata row per episode. Use
``MultiregimeV4EvaluationBank`` to iterate unpadded episodes, or
``evaluate_multiregime_v4_bank`` to obtain per-episode, per-cell, and
per-factor metrics for a NanoTabPFN-compatible model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v4 import (
    FAMILIES,
    HP_PROFILES,
    MECHANISM_MODES,
    RULE_MODES,
    sample_episode_v4,
)

FACTORS = (
    "support_size",
    "num_features",
    "num_regimes",
    "num_classes",
    "class_ratio",
    "task_family",
    "mechanism_mode",
    "rule_mode",
)


@dataclass(frozen=True)
class V4EvaluationBankConfig:
    """Configuration for matched validation and test factorial episode banks."""

    validation_output: str
    test_output: str
    validation_seed: int = 900_001
    test_seed: int = 1_900_001
    episodes_per_cell: int = 8
    support_sizes: tuple[int, ...] = (32, 64, 128)
    num_features: tuple[int, ...] = (2, 4, 8, 12)
    #: K=1 is the single-rule control; K>=2 are multiregime tasks.
    num_regimes: tuple[int, ...] = (1, 2, 3, 4)
    #: Binary cells additionally vary the positive-class ratio (calibrated
    #: labels). Multiclass cells use the profile's label rule: native Reg2Cls
    #: boundaries under ``native`` (every requested class present in both
    #: halves), a uniform calibrated vector under the deprecated ``production``.
    num_classes: tuple[int, ...] = (2,)
    hp_profile: str = "native"
    #: 0.9 is the label-complement of 0.1, so retain only distinct binary
    #: imbalance conditions.
    class_ratios: tuple[float, ...] = (0.1, 0.3, 0.5)
    task_families: tuple[str, ...] = FAMILIES
    #: The paired shared-rule control and the actual multiregime task use the
    #: same predictive TabICL base prior and differ only at final label choice.
    rule_modes: tuple[str, ...] = RULE_MODES
    #: r_z changes SCM mechanisms; g_z keeps r(X) and changes label rules.
    mechanism_modes: tuple[str, ...] = MECHANISM_MODES
    query_size: int = 256
    #: Retained for compatibility with earlier dump commands. Production label
    #: calibration now uses the complete generated episode, like native
    #: rank-based Reg2Cls, rather than adding hidden rows.
    calibration_size: int = 256
    min_samples_per_regime: int = 32
    pad_features: int | None = None
    mlp_probability: float = 0.7
    #: Append the routing score as an extra (randomly placed) feature column,
    #: so the query row's regime is directly observable. A control for whether
    #: multiregime failures are about *inferring* the regime or about applying
    #: a regime-conditional rule once it is known. ``num_features`` keeps its
    #: factor meaning; the stored ``input_width`` is ``num_features + 1``.
    expose_z: bool = False
    overwrite: bool = False

    @property
    def input_width_extra(self) -> int:
        return int(self.expose_z)

    @property
    def episodes(self) -> int:
        class_conditions = sum(len(self.class_ratios) if count == 2 else 1 for count in self.num_classes)
        return (
            len(self.valid_support_regime_pairs)
            * len(self.num_features)
            * class_conditions
            * len(self.task_families)
            * len(self.mechanism_modes)
            * len(self.rule_modes)
            * self.episodes_per_cell
        )

    @property
    def valid_support_regime_pairs(self) -> tuple[tuple[int, int], ...]:
        """Support/K cells with enough support evidence for the requested K.

        K=1 is always a valid single-rule control. K>=2 is included only
        when the support set can allocate ``min_samples_per_regime`` rows to
        every regime.
        """
        return tuple(
            (support_size, num_regimes)
            for support_size, num_regimes in product(self.support_sizes, self.num_regimes)
            if num_regimes == 1 or num_regimes <= support_size // self.min_samples_per_regime
        )


def _unique(values: Sequence[Any], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _validate(config: V4EvaluationBankConfig) -> None:
    validation_path = Path(config.validation_output)
    test_path = Path(config.test_output)
    if validation_path == test_path:
        raise ValueError("validation_output and test_output must be different paths.")
    if config.validation_seed == config.test_seed:
        raise ValueError("validation_seed and test_seed must be different.")
    if config.episodes_per_cell < 1:
        raise ValueError("episodes_per_cell must be positive.")
    if config.query_size < 1 or config.calibration_size < 1:
        raise ValueError("query_size and calibration_size must be positive.")
    if config.min_samples_per_regime < 1:
        raise ValueError("min_samples_per_regime must be positive.")
    if not 0 <= config.mlp_probability <= 1:
        raise ValueError("mlp_probability must lie in [0, 1].")
    _unique(config.support_sizes, "support_sizes")
    _unique(config.num_features, "num_features")
    _unique(config.num_regimes, "num_regimes")
    _unique(config.num_classes, "num_classes")
    _unique(config.class_ratios, "class_ratios")
    _unique(config.task_families, "task_families")
    _unique(config.mechanism_modes, "mechanism_modes")
    _unique(config.rule_modes, "rule_modes")
    if config.hp_profile not in HP_PROFILES:
        raise ValueError(f"hp_profile must be one of {HP_PROFILES}.")
    if min(config.support_sizes) < 1 or min(config.num_features) < 1:
        raise ValueError("support_sizes and num_features must be positive.")
    if min(config.num_regimes) < 1:
        raise ValueError("num_regimes must be positive.")
    if min(config.num_classes) < 2:
        raise ValueError("num_classes must be at least two.")
    if any(not 0 < ratio < 1 for ratio in config.class_ratios):
        raise ValueError("class_ratios must lie strictly between 0 and 1.")
    if any(mode not in RULE_MODES for mode in config.rule_modes):
        raise ValueError(f"rule_modes must be drawn from {RULE_MODES}.")
    if any(mode not in MECHANISM_MODES for mode in config.mechanism_modes):
        raise ValueError(f"mechanism_modes must be drawn from {MECHANISM_MODES}.")
    unknown_families = set(config.task_families) - set(FAMILIES)
    if unknown_families:
        raise ValueError(f"Unknown task families: {sorted(unknown_families)}.")
    if not config.valid_support_regime_pairs:
        raise ValueError("No support/regime cell has enough support evidence.")
    if config.pad_features is not None and config.pad_features < max(config.num_features) + config.input_width_extra:
        raise ValueError("pad_features must accommodate every requested feature count (plus the expose_z column).")
    if not config.overwrite:
        existing = [str(path) for path in (validation_path, test_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite evaluation bank(s): {existing}.")


def _episode_seed(seed_root: int, cell_id: int, episode_in_cell: int) -> int:
    """Derive reproducible, disjoint episode seeds without order dependence."""
    sequence = np.random.SeedSequence([seed_root, cell_id, episode_in_cell])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _write_split(
    path: Path,
    *,
    split: str,
    seed_root: int,
    config: V4EvaluationBankConfig,
    pad_features: int,
) -> Path:
    max_rows = max(config.support_sizes) + config.query_size
    class_conditions = [
        (num_classes, class_ratio if num_classes == 2 else None)
        for num_classes in config.num_classes
        for class_ratio in (config.class_ratios if num_classes == 2 else (None,))
    ]
    base_cells = list(
        product(
            config.valid_support_regime_pairs,
            config.num_features,
            class_conditions,
            config.task_families,
            config.mechanism_modes,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "X",
            shape=(config.episodes, max_rows, pad_features),
            chunks=(1, max_rows, pad_features),
            compression="lzf",
            dtype="f4",
        )
        handle.create_dataset(
            "y",
            shape=(config.episodes, max_rows),
            chunks=(1, max_rows),
            compression="lzf",
            dtype="f4",
        )
        handle.create_dataset(
            "regime",
            shape=(config.episodes, max_rows),
            chunks=(1, max_rows),
            compression="lzf",
            dtype="i1",
        )
        metadata = {
            "cell_id": handle.create_dataset("cell_id", shape=(config.episodes,), dtype="i4"),
            "episode_in_cell": handle.create_dataset("episode_in_cell", shape=(config.episodes,), dtype="i4"),
            "episode_seed": handle.create_dataset("episode_seed", shape=(config.episodes,), dtype="u4"),
            "support_size": handle.create_dataset("support_size", shape=(config.episodes,), dtype="i4"),
            "query_size": handle.create_dataset("query_size", shape=(config.episodes,), dtype="i4"),
            "num_features": handle.create_dataset("num_features", shape=(config.episodes,), dtype="i4"),
            # Actual stored X width: num_features, plus one when expose_z appends the routing score.
            "input_width": handle.create_dataset("input_width", shape=(config.episodes,), dtype="i4"),
            # Column holding the routing score when expose_z, else -1.
            "z_column_index": handle.create_dataset("z_column_index", shape=(config.episodes,), dtype="i4"),
            "num_regimes": handle.create_dataset("num_regimes", shape=(config.episodes,), dtype="i1"),
            "num_classes": handle.create_dataset("num_classes", shape=(config.episodes,), dtype="i1"),
            "class_ratio": handle.create_dataset("class_ratio", shape=(config.episodes,), dtype="f4"),
            "task_family_index": handle.create_dataset("task_family_index", shape=(config.episodes,), dtype="i1"),
            "mechanism_mode_index": handle.create_dataset(
                "mechanism_mode_index", shape=(config.episodes,), dtype="i1"
            ),
            "rule_mode_index": handle.create_dataset("rule_mode_index", shape=(config.episodes,), dtype="i1"),
            "prior_type_index": handle.create_dataset("prior_type_index", shape=(config.episodes,), dtype="i1"),
            "sampled_is_causal": handle.create_dataset("sampled_is_causal", shape=(config.episodes,), dtype="i1"),
            "effective_is_causal": handle.create_dataset("effective_is_causal", shape=(config.episodes,), dtype="i1"),
            "scm_num_layers": handle.create_dataset("scm_num_layers", shape=(config.episodes,), dtype="i1"),
            "scm_hidden_dim": handle.create_dataset("scm_hidden_dim", shape=(config.episodes,), dtype="i4"),
            "scm_num_causes": handle.create_dataset("scm_num_causes", shape=(config.episodes,), dtype="i4"),
            "scm_metadata_json": handle.create_dataset(
                "scm_metadata_json", shape=(config.episodes,), dtype=h5py.string_dtype(encoding="utf-8")
            ),
            "support_positive_rate": handle.create_dataset(
                "support_positive_rate", shape=(config.episodes,), dtype="f4"
            ),
            "query_positive_rate": handle.create_dataset(
                "query_positive_rate", shape=(config.episodes,), dtype="f4"
            ),
        }
        handle.create_dataset("families", data=list(FAMILIES), dtype=h5py.string_dtype())
        handle.create_dataset("mechanism_modes", data=list(MECHANISM_MODES), dtype=h5py.string_dtype())
        handle.create_dataset("rule_modes", data=list(RULE_MODES), dtype=h5py.string_dtype())
        handle.create_dataset("prior_types", data=("mlp_scm", "tree_scm"), dtype=h5py.string_dtype())
        handle.create_dataset("evaluation_split", data=split, dtype=h5py.string_dtype())
        handle.create_dataset("expose_z", data=int(config.expose_z), dtype="i1")
        handle.create_dataset("hp_profile", data=config.hp_profile, dtype=h5py.string_dtype())
        handle.create_dataset("schema_version", data=5, dtype="i4")
        handle.create_dataset("episodes_per_cell", data=config.episodes_per_cell, dtype="i4")

        row = 0
        for base_cell_id, (
            support_regime,
            features,
            class_condition,
            family,
            mechanism_mode,
        ) in enumerate(base_cells):
            support_size, regimes = support_regime
            num_classes, class_ratio = class_condition
            family_index = FAMILIES.index(family)
            mechanism_mode_index = MECHANISM_MODES.index(mechanism_mode)
            for rule_mode_index, rule_mode in enumerate(config.rule_modes):
                cell_id = base_cell_id * len(config.rule_modes) + rule_mode_index
                for episode_in_cell in range(config.episodes_per_cell):
                    # Modes intentionally share this seed: their X and candidate
                    # rules are paired, and only Y_0 vs Y_z differs.
                    episode_seed = _episode_seed(seed_root, base_cell_id, episode_in_cell)
                    episode = sample_episode_v4(
                        episode_seed,
                        family=family,
                        min_features=features,
                        max_features=features,
                        num_regimes=regimes,
                        support_size=support_size,
                        query_size=config.query_size,
                        calibration_size=config.calibration_size,
                        pad_features=pad_features,
                        num_classes=num_classes,
                        max_classes=max(config.num_classes),
                        min_samples_per_regime=config.min_samples_per_regime,
                        mix_probs=(config.mlp_probability, 1.0 - config.mlp_probability),
                        class_ratio=class_ratio,
                        rule_mode=rule_mode,
                        mechanism_mode=mechanism_mode,
                        expose_z=config.expose_z,
                        hp_profile=config.hp_profile,
                        require_num_classes=True,
                    )
                    x = torch.cat((episode.support_x, episode.query_x), dim=1).numpy()[0]
                    y = torch.cat((episode.support_y, episode.query_y), dim=1).numpy()[0]
                    regime = np.concatenate((episode.support_z, episode.query_z))
                    rows = support_size + config.query_size
                    handle["X"][row, :rows] = x
                    handle["y"][row, :rows] = y
                    handle["regime"][row, :rows] = regime
                    metadata["cell_id"][row] = cell_id
                    metadata["episode_in_cell"][row] = episode_in_cell
                    metadata["episode_seed"][row] = episode_seed
                    metadata["support_size"][row] = support_size
                    metadata["query_size"][row] = config.query_size
                    metadata["num_features"][row] = features
                    metadata["input_width"][row] = episode.d
                    metadata["z_column_index"][row] = -1 if episode.z_column_index is None else episode.z_column_index
                    metadata["num_regimes"][row] = regimes
                    metadata["num_classes"][row] = num_classes
                    metadata["class_ratio"][row] = -1.0 if class_ratio is None else class_ratio
                    metadata["task_family_index"][row] = family_index
                    metadata["mechanism_mode_index"][row] = mechanism_mode_index
                    metadata["rule_mode_index"][row] = RULE_MODES.index(rule_mode)
                    metadata["prior_type_index"][row] = 0 if episode.prior_type == "mlp_scm" else 1
                    metadata["sampled_is_causal"][row] = episode.scm_metadata["sampled_is_causal"]
                    metadata["effective_is_causal"][row] = episode.scm_metadata["effective_is_causal"]
                    metadata["scm_num_layers"][row] = episode.scm_metadata["effective_num_layers"]
                    metadata["scm_hidden_dim"][row] = episode.scm_metadata["effective_hidden_dim"]
                    metadata["scm_num_causes"][row] = episode.scm_metadata["effective_num_causes"]
                    metadata["scm_metadata_json"][row] = json.dumps(episode.scm_metadata, sort_keys=True)
                    metadata["support_positive_rate"][row] = float(episode.support_y.mean())
                    metadata["query_positive_rate"][row] = float(episode.query_y.mean())
                    row += 1

    return path


def build_multiregime_v4_evaluation_banks(config: V4EvaluationBankConfig) -> tuple[Path, Path]:
    """Write matched factorial validation and test banks with disjoint seeds."""
    _validate(config)
    pad_features = (
        max(config.num_features) + config.input_width_extra if config.pad_features is None else config.pad_features
    )
    validation = _write_split(
        Path(config.validation_output),
        split="validation",
        seed_root=config.validation_seed,
        config=config,
        pad_features=pad_features,
    )
    test = _write_split(
        Path(config.test_output),
        split="test",
        seed_root=config.test_seed,
        config=config,
        pad_features=pad_features,
    )
    return validation, test


class MultiregimeV4EvaluationBank:
    """Read a fixed v4 evaluation bank one unpadded episode at a time."""

    def __init__(self, path: str | Path, *, device: str | torch.device = "cpu"):
        self.path = Path(path)
        self.device = device
        self._handle = h5py.File(self.path, "r")
        self._count = self._handle["X"].shape[0]
        self.episodes_per_cell = int(self._handle["episodes_per_cell"][()])
        self.families = tuple(value.decode("utf-8") for value in self._handle["families"][:])
        self.mechanism_modes = tuple(value.decode("utf-8") for value in self._handle["mechanism_modes"][:])
        self.rule_modes = tuple(value.decode("utf-8") for value in self._handle["rule_modes"][:])
        self.prior_types = tuple(value.decode("utf-8") for value in self._handle["prior_types"][:])
        self.split = self._handle["evaluation_split"][()].decode("utf-8")

    def __len__(self) -> int:
        return self._count

    def _metadata(self, index: int) -> dict[str, int | float | str]:
        handle = self._handle
        family_index = int(handle["task_family_index"][index])
        mechanism_mode_index = int(handle["mechanism_mode_index"][index])
        rule_mode_index = int(handle["rule_mode_index"][index])
        prior_type_index = int(handle["prior_type_index"][index])
        return {
            "episode_id": index,
            "cell_id": int(handle["cell_id"][index]),
            "episode_in_cell": int(handle["episode_in_cell"][index]),
            "episode_seed": int(handle["episode_seed"][index]),
            "support_size": int(handle["support_size"][index]),
            "query_size": int(handle["query_size"][index]),
            "num_features": int(handle["num_features"][index]),
            # Banks written before expose_z support have no input_width; width == num_features there.
            "input_width": (
                int(handle["input_width"][index]) if "input_width" in handle else int(handle["num_features"][index])
            ),
            "z_column_index": int(handle["z_column_index"][index]) if "z_column_index" in handle else -1,
            "num_regimes": int(handle["num_regimes"][index]),
            "num_classes": int(handle["num_classes"][index]) if "num_classes" in handle else 2,
            "class_ratio": (
                None if float(handle["class_ratio"][index]) < 0 else float(handle["class_ratio"][index])
            ),
            "task_family": self.families[family_index],
            "mechanism_mode": self.mechanism_modes[mechanism_mode_index],
            "rule_mode": self.rule_modes[rule_mode_index],
            "prior_type": self.prior_types[prior_type_index],
            "sampled_is_causal": bool(handle["sampled_is_causal"][index]),
            "effective_is_causal": bool(handle["effective_is_causal"][index]),
            "scm_num_layers": int(handle["scm_num_layers"][index]),
            "scm_hidden_dim": int(handle["scm_hidden_dim"][index]),
            "scm_num_causes": int(handle["scm_num_causes"][index]),
            "support_positive_rate": float(handle["support_positive_rate"][index]),
            "query_positive_rate": float(handle["query_positive_rate"][index]),
        }

    def scm_metadata(self, index: int) -> dict[str, Any]:
        """Return the full requested and effective SCM audit record for one episode."""
        if not 0 <= index < len(self):
            raise IndexError(f"Episode index {index} is outside [0, {len(self)}).")
        value = self._handle["scm_metadata_json"][index]
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return json.loads(text)

    def episode(self, index: int) -> dict[str, torch.Tensor | dict[str, int | float | str]]:
        if not 0 <= index < len(self):
            raise IndexError(f"Episode index {index} is outside [0, {len(self)}).")
        metadata = self._metadata(index)
        rows = metadata["support_size"] + metadata["query_size"]
        features = metadata["input_width"]  # == num_features unless the bank exposes z
        split = metadata["support_size"]
        x = torch.from_numpy(self._handle["X"][index, :rows, :features])[None].to(self.device)
        y = torch.from_numpy(self._handle["y"][index, :rows])[None].to(self.device)
        regime = torch.from_numpy(self._handle["regime"][index, :rows])[None].long().to(self.device)
        return {
            "support_x": x[:, :split],
            "support_y": y[:, :split],
            "query_x": x[:, split:],
            "query_y": y[:, split:].long(),
            "support_regime": regime[:, :split],
            "query_regime": regime[:, split:],
            "metadata": metadata,
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | dict[str, int | float | str]]]:
        for index in range(len(self)):
            yield self.episode(index)

    def cell_batch(
        self, start: int, *, max_episodes: int | None = None
    ) -> dict[str, torch.Tensor | list[dict[str, int | float | str]]]:
        """Return a contiguous rectangular sub-batch from one factorial cell."""
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be positive when supplied.")
        cell_start = (start // self.episodes_per_cell) * self.episodes_per_cell
        cell_stop = min(cell_start + self.episodes_per_cell, len(self))
        stop = cell_stop if max_episodes is None else min(start + max_episodes, cell_stop)
        episodes = [self.episode(index) for index in range(start, stop)]
        if not episodes:
            raise IndexError(f"Cell batch start {start} is outside [0, {len(self)}).")
        cell_ids = {episode["metadata"]["cell_id"] for episode in episodes}
        if len(cell_ids) != 1:
            raise RuntimeError("Evaluation-bank cells must be contiguous and have a fixed episode count.")
        return {
            "support_x": torch.cat([episode["support_x"] for episode in episodes]),
            "support_y": torch.cat([episode["support_y"] for episode in episodes]),
            "query_x": torch.cat([episode["query_x"] for episode in episodes]),
            "query_y": torch.cat([episode["query_y"] for episode in episodes]),
            "support_regime": torch.cat([episode["support_regime"] for episode in episodes]),
            "query_regime": torch.cat([episode["query_regime"] for episode in episodes]),
            "metadata": [episode["metadata"] for episode in episodes],
        }

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> MultiregimeV4EvaluationBank:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _episode_auc(probabilities: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """One-vs-rest macro AUC for one episode; NaN if the query labels are degenerate.

    A single episode's query set can easily miss a class entirely (small query
    sizes, skewed ``class_ratio``), which makes ``roc_auc_score`` undefined.
    Such episodes are reported as NaN and excluded by the ``nanmean`` calls in
    ``_summary``/``overall`` rather than treated as errors.
    """
    labels_present = torch.unique(target)
    if labels_present.numel() < 2:
        return float("nan")
    # The model's output head is padded to a fixed max_classes; slice to this
    # episode's actual class count, then renormalize so each row sums to 1
    # again (conditioning on "it's one of the valid classes" for this
    # episode). sklearn's multiclass roc_auc_score requires normalized rows.
    valid_probs = probabilities[:, :num_classes].cpu().double()  # on CPU: MPS has no float64
    valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs_np = valid_probs.cpu().numpy()
    target_np = target.cpu().numpy()
    try:
        if num_classes == 2:
            return float(roc_auc_score(target_np, probs_np[:, 1]))
        return float(
            roc_auc_score(target_np, probs_np, multi_class="ovr", average="macro", labels=list(range(num_classes)))
        )
    except ValueError:
        return float("nan")


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _regime_position_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Within-episode distribution of the loss over routing regimes, for K>=2 episodes.

    ``regime_position`` 0 is the first rule of the episode.  ``spread`` is the within-episode
    max-min cross entropy: large values mean the model fits some regimes and not others.
    """
    by_position: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    spreads: dict[int, list[float]] = defaultdict(list)
    variances: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        counts = row.get("regime_query_counts") or []
        if len(counts) < 2:
            continue
        k = int(row["num_regimes"])
        spreads[k].append(float(row["regime_cross_entropy_spread"]))
        variances[k].append((float(row.get("regime_cross_entropy_var", float("nan"))),
                             float(row.get("regime_cross_entropy_sampling_var", float("nan"))),
                             float(row.get("regime_cross_entropy_excess_var", float("nan")))))
        for position, (ce, accuracy, positive, auc) in enumerate(
            zip(row["regime_cross_entropy"], row["regime_accuracy"], row["regime_positive_rate"],
                row.get("regime_auc") or [float("nan")] * len(row["regime_cross_entropy"]), strict=True)
        ):
            by_position[(k, position)].append((ce, accuracy, positive, auc))
    result = []
    for (k, position), values in sorted(by_position.items()):
        result.append(
            {
                "num_regimes": k,
                "regime_position": position,
                "episodes": len(values),
                "query_cross_entropy": float(np.mean([value[0] for value in values])),
                "query_accuracy": float(np.mean([value[1] for value in values])),
                "query_positive_rate": float(np.mean([value[2] for value in values])),
                "query_auc": _nanmean(value[3] for value in values),
                "auc_defined_episodes": int(sum(1 for value in values if value[3] == value[3])),
                "within_episode_spread": float(np.mean(spreads[k])),
                "within_episode_variance": _nanmean(value[0] for value in variances[k]),
                "within_episode_sampling_variance": _nanmean(value[1] for value in variances[k]),
                "within_episode_excess_variance": _nanmean(value[2] for value in variances[k]),
            }
        )
    return result


def _summary(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "episodes": len(group),
                "query_cross_entropy": float(np.mean([row["query_cross_entropy"] for row in group])),
                "query_accuracy": float(np.mean([row["query_accuracy"] for row in group])),
                "query_auc": _nanmean(row["query_auc"] for row in group),
            }
        )
    return result


def evaluate_multiregime_v4_bank(
    model: torch.nn.Module,
    bank: MultiregimeV4EvaluationBank | str | Path,
    *,
    device: str | torch.device | None = None,
    max_episodes_per_forward: int | None = None,
) -> dict[str, Any]:
    """Score every bank episode and summarize the full grid and each factor."""
    if max_episodes_per_forward is not None and max_episodes_per_forward < 1:
        raise ValueError("max_episodes_per_forward must be positive when supplied.")
    owned_bank = not isinstance(bank, MultiregimeV4EvaluationBank)
    evaluation_bank = (
        MultiregimeV4EvaluationBank(bank, device="cpu" if device is None else device) if owned_bank else bank
    )
    was_training = model.training
    model.eval()
    per_episode: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for cell_start in range(0, len(evaluation_bank), evaluation_bank.episodes_per_cell):
                cell_stop = min(cell_start + evaluation_bank.episodes_per_cell, len(evaluation_bank))
                batch_size = cell_stop - cell_start if max_episodes_per_forward is None else max_episodes_per_forward
                for start in range(cell_start, cell_stop, batch_size):
                    batch = evaluation_bank.cell_batch(start, max_episodes=max_episodes_per_forward)
                    logits = model(batch["support_x"], batch["support_y"], batch["query_x"])
                    target = batch["query_y"]
                    per_query_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
                    ).reshape_as(target)
                    per_episode_loss = per_query_loss.mean(dim=-1)
                    per_episode_accuracy = (logits.argmax(dim=-1) == target).float().mean(dim=-1)
                    probabilities = F.softmax(logits, dim=-1)
                    per_query_correct = (logits.argmax(dim=-1) == target).float()
                    for metadata, loss, accuracy, probs, episode_target, episode_loss, episode_correct, episode_regime in zip(
                        batch["metadata"], per_episode_loss, per_episode_accuracy, probabilities, target,
                        per_query_loss, per_query_correct, batch["query_regime"], strict=True
                    ):
                        metrics = dict(metadata)
                        metrics["query_cross_entropy"] = float(loss.item())
                        metrics["query_accuracy"] = float(accuracy.item())
                        metrics["query_auc"] = _episode_auc(probs, episode_target, int(metadata["num_classes"]))
                        # Per-regime breakdown within the episode: how the loss is distributed over the K
                        # routing regimes (rule 0 first).  Empty regimes are omitted from the lists.
                        regimes = episode_regime.tolist()
                        by_regime: dict[int, list[int]] = {}
                        for position, regime_id in enumerate(regimes):
                            by_regime.setdefault(int(regime_id), []).append(position)
                        ordered = sorted(by_regime)
                        metrics["regime_ids"] = ordered
                        metrics["regime_query_counts"] = [len(by_regime[r]) for r in ordered]
                        metrics["regime_cross_entropy"] = [
                            float(episode_loss[by_regime[r]].mean().item()) for r in ordered
                        ]
                        metrics["regime_accuracy"] = [
                            float(episode_correct[by_regime[r]].mean().item()) for r in ordered
                        ]
                        metrics["regime_positive_rate"] = [
                            float((episode_target[by_regime[r]] > 0).float().mean().item()) for r in ordered
                        ]
                        # Macro OvR AUC per regime; NaN when that regime's query rows carry a single class
                        # (common at class_ratio 0.1 with 256/K rows), so callers must nanmean over it.
                        metrics["regime_auc"] = [
                            _episode_auc(probs[by_regime[r]], episode_target[by_regime[r]], int(metadata["num_classes"]))
                            for r in ordered
                        ]
                        if len(ordered) > 1:
                            values = metrics["regime_cross_entropy"]
                            metrics["regime_cross_entropy_spread"] = max(values) - min(values)
                            # Variance of the per-regime CEs (the range above grows with K by construction),
                            # with the part explained by finite regime sizes subtracted: each regime's mean is
                            # estimated from 256/K rows, so it carries sampling variance var(row losses)/n.
                            mean_ce = sum(values) / len(values)
                            between = sum((value - mean_ce) ** 2 for value in values) / (len(values) - 1)
                            sampling = sum(
                                float(episode_loss[by_regime[r]].var(unbiased=True).item()) / len(by_regime[r])
                                for r in ordered if len(by_regime[r]) > 1
                            ) / len(ordered)
                            metrics["regime_cross_entropy_var"] = between
                            metrics["regime_cross_entropy_sampling_var"] = sampling
                            metrics["regime_cross_entropy_excess_var"] = between - sampling
                            metrics["regime_cross_entropy_best"] = min(values)
                            metrics["regime_cross_entropy_worst"] = max(values)
                            metrics["regime_cross_entropy_rule0_minus_rest"] = values[0] - (
                                sum(values[1:]) / len(values[1:])
                            )
                        per_episode.append(metrics)
    finally:
        model.train(was_training)
        if owned_bank:
            evaluation_bank.close()

    return {
        "split": evaluation_bank.split,
        "episodes": len(per_episode),
        "overall": {
            "query_cross_entropy": float(np.mean([row["query_cross_entropy"] for row in per_episode])),
            "query_accuracy": float(np.mean([row["query_accuracy"] for row in per_episode])),
            "query_auc": _nanmean(row["query_auc"] for row in per_episode),
        },
        "per_episode": per_episode,
        "by_cell": _summary(per_episode, FACTORS),
        "by_support_size": _summary(per_episode, ("support_size",)),
        "by_num_features": _summary(per_episode, ("num_features",)),
        "by_num_regimes": _summary(per_episode, ("num_regimes",)),
        "by_num_classes": _summary(per_episode, ("num_classes",)),
        "by_class_ratio": _summary(per_episode, ("class_ratio",)),
        "by_task_family": _summary(per_episode, ("task_family",)),
        "by_mechanism_mode": _summary(per_episode, ("mechanism_mode",)),
        "by_rule_mode": _summary(per_episode, ("rule_mode",)),
        "by_prior_type": _summary(per_episode, ("prior_type",)),
        "by_sampled_is_causal": _summary(per_episode, ("sampled_is_causal",)),
        "by_effective_is_causal": _summary(per_episode, ("effective_is_causal",)),
        "by_scm_num_layers": _summary(per_episode, ("scm_num_layers",)),
        "by_scm_hidden_dim": _summary(per_episode, ("scm_hidden_dim",)),
        "by_scm_num_causes": _summary(per_episode, ("scm_num_causes",)),
        "by_regime_position": _regime_position_summary(per_episode),
    }


def _parse_csv(text: str, converter, name: str) -> tuple[Any, ...]:
    values = tuple(converter(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must be a non-empty comma-separated list.")
    return values


def build_parser() -> argparse.ArgumentParser:
    defaults = V4EvaluationBankConfig(validation_output="", test_output="")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--validation-seed", type=int, default=defaults.validation_seed)
    parser.add_argument("--test-seed", type=int, default=defaults.test_seed)
    parser.add_argument("--episodes-per-cell", type=int, default=defaults.episodes_per_cell)
    parser.add_argument(
        "--support-sizes",
        type=lambda value: _parse_csv(value, int, "support-sizes"),
        default=defaults.support_sizes,
    )
    parser.add_argument(
        "--num-features",
        type=lambda value: _parse_csv(value, int, "num-features"),
        default=defaults.num_features,
    )
    parser.add_argument(
        "--num-regimes",
        type=lambda value: _parse_csv(value, int, "num-regimes"),
        default=defaults.num_regimes,
    )
    parser.add_argument(
        "--num-classes",
        type=lambda value: _parse_csv(value, int, "num-classes"),
        default=defaults.num_classes,
    )
    parser.add_argument(
        "--class-ratios",
        type=lambda value: _parse_csv(value, float, "class-ratios"),
        default=defaults.class_ratios,
    )
    parser.add_argument(
        "--task-families",
        type=lambda value: _parse_csv(value, str, "task-families"),
        default=defaults.task_families,
    )
    parser.add_argument(
        "--mechanism-modes",
        type=lambda value: _parse_csv(value, str, "mechanism-modes"),
        default=defaults.mechanism_modes,
    )
    parser.add_argument(
        "--rule-modes",
        type=lambda value: _parse_csv(value, str, "rule-modes"),
        default=defaults.rule_modes,
    )
    parser.add_argument("--query-size", type=int, default=defaults.query_size)
    parser.add_argument("--calibration-size", type=int, default=defaults.calibration_size)
    parser.add_argument("--min-samples-per-regime", type=int, default=defaults.min_samples_per_regime)
    parser.add_argument("--pad-features", type=int, default=defaults.pad_features)
    parser.add_argument(
        "--expose-z",
        action="store_true",
        default=defaults.expose_z,
        help="append the routing score as an extra feature column (regime-observable control)",
    )
    parser.add_argument("--mlp-probability", type=float, default=defaults.mlp_probability)
    parser.add_argument("--hp-profile", choices=HP_PROFILES, default=defaults.hp_profile)
    parser.add_argument("--overwrite", action="store_true", default=defaults.overwrite)
    return parser


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    return build_multiregime_v4_evaluation_banks(V4EvaluationBankConfig(**vars(build_parser().parse_args(argv))))


if __name__ == "__main__":
    validation_path, test_path = main()
    print(f"wrote validation bank to {validation_path}")
    print(f"wrote test bank to {test_path}")
