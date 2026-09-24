"""Rank multiregime-v4 NanoTabPFN checkpoints on BeyondArena.

This evaluator is intentionally independent of the official TabArena runner.  It
uses Data Foundry's immutable containers and official outer folds, then exposes
the complete training fold to the existing NanoTabPFN call interface.  In
particular, it never turns an oversized fold into a smaller benchmark: a fold
that cannot fit on the selected device is written as ``unsupported``.

The command line entry point is::

    python -m tfmplayground.experiments.evaluate_multiregime_v4_beyondarena \
        --run-roots runs/multiregime-v4 --output-dir results/beyondarena-v4

The optional ``beyondarena`` project extra installs Data Foundry.  The module
can still be imported without that extra so its filtering, metrics, checkpoint
selection, and aggregation helpers remain unit-testable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from tfmplayground.interface import get_feature_preprocessor, init_model_from_state_dict_file

try:
    from tfmplayground.models.slot_regime import load_checkpoint_for_inference
except ModuleNotFoundError:  # Older v4 training checkout: standard NanoTabPFN checkpoints.
    def load_checkpoint_for_inference(path: str | Path, device: str | torch.device = "cpu") -> Any:
        return init_model_from_state_dict_file(str(path)).to(device).eval()

ROWS_MAX = 10_000
FEATURES_MAX = 30
CLASSES_MIN = 2
CLASSES_MAX = 5
ROW_BUCKETS = ((1, 1_000, "small_rows"), (1_001, 5_000, "medium_rows"), (5_001, 10_000, "large_rows"))
FEATURE_BUCKETS = ((1, 10, "small_feat"), (11, 20, "medium_feat"), (21, 30, "large_feat"))
REGIMES = ("IID", "Temporal", "Grouped")


class UnsupportedFoldError(RuntimeError):
    """Raised when a complete official fold is not supported by the hardware/model."""


@dataclass(frozen=True)
class BeyondArenaConfig:
    """Evaluation and output configuration."""

    run_roots: tuple[str, ...] = ()
    output_dir: str = "results/beyondarena_v4"
    device: str = "cpu"
    data_cache: str | None = None
    query_chunk_size: int = 128
    num_mem_chunks: int = 8
    rows_max: int = ROWS_MAX
    features_max: int = FEATURES_MAX
    classes_min: int = CLASSES_MIN
    classes_max: int = CLASSES_MAX
    smoke_task: str | None = None
    task_names_file: str | None = None
    baseline_models: tuple[str, ...] = ()
    fail_on_task_error: bool = False


@dataclass(frozen=True)
class BeyondArenaTask:
    """A filtered task plus the container needed for fold evaluation."""

    name: str
    uuid: str
    checksum: str
    rows: int
    raw_features: int
    classes: int
    problem_type: str
    regime: str
    target_column: str
    group_columns: tuple[str, ...]
    text_features: tuple[str, ...]
    high_cardinality_features: tuple[str, ...]
    container: Any = field(repr=False, compare=False)
    time_column: str | None = None

    @property
    def problem_bucket(self) -> str:
        return "binary" if self.classes == 2 else "multi"

    @property
    def row_bucket(self) -> str | None:
        return bucket_value(self.rows, ROW_BUCKETS)

    @property
    def feature_bucket(self) -> str | None:
        return bucket_value(self.raw_features, FEATURE_BUCKETS)

    def conditions(self) -> tuple[str, ...]:
        """Return all requested ranking slices containing this task."""
        conditions = ["all", self.problem_bucket, self.regime]
        if self.row_bucket is not None:
            conditions.append(self.row_bucket)
            conditions.append(f"{self.row_bucket}__{self.regime}")
        if self.feature_bucket is not None:
            conditions.append(self.feature_bucket)
        conditions.append(f"{self.problem_bucket}__{self.regime}")
        return tuple(dict.fromkeys(conditions))


@dataclass(frozen=True)
class TaskInspection:
    """The complete manifest decision for one official container."""

    task: BeyondArenaTask | None
    row: dict[str, Any]


@dataclass(frozen=True)
class CheckpointCandidate:
    """A metadata-verified checkpoint found under one run directory."""

    path: Path
    run_dir: Path
    model_identity: str
    family: str
    size: str
    validation_ce: float | None
    step: int | None
    state: Mapping[str, Any] = field(repr=False, compare=False)
    z_expose: bool = False


@dataclass(frozen=True)
class ModelSelection:
    """One requested checkpoint policy for a v4 run."""

    model_identity: str
    family: str
    size: str
    checkpoint_policy: str
    checkpoint_path: Path | None
    selection_status: str
    selection_reason: str | None
    validation_ce: float | None
    run_dir: Path
    candidate_count: int
    model_kind: str = "nanotabpfn"
    z_expose: bool = False


def bucket_value(value: int, buckets: Sequence[tuple[int, int, str]]) -> str | None:
    """Return the inclusive bucket containing ``value`` or ``None``."""

    value = int(value)
    for lower, upper, name in buckets:
        if lower <= value <= upper:
            return name
    return None


def row_bucket(rows: int) -> str | None:
    """Public row-bucket helper used by the filtering tests."""

    return bucket_value(rows, ROW_BUCKETS)


def feature_bucket(features: int) -> str | None:
    """Public raw-feature bucket helper used by the filtering tests."""

    return bucket_value(features, FEATURE_BUCKETS)


def _as_columns(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _attribute_or_mapping(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read an attribute or mapping key without depending on pydantic internals."""

    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


def _metadata_number(objects: Sequence[Any], names: Sequence[str]) -> int | None:
    for obj in objects:
        value = _attribute_or_mapping(obj, names)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def _problem_type(task_metadata: Any) -> str:
    value = str(_attribute_or_mapping(task_metadata, ("problem_type", "task_type"), "unknown"))
    return value.strip().lower()


def _regime(task_metadata: Any) -> str:
    raw = _attribute_or_mapping(task_metadata, ("split_regime",), None)
    if raw is None:
        if _attribute_or_mapping(task_metadata, ("time_on",), None) is not None:
            raw = "temporal_non_iid"
        elif _attribute_or_mapping(task_metadata, ("group_on",), None) is not None:
            raw = "grouped_non_iid"
        else:
            raw = "iid"
    normalized = str(raw).strip().lower().replace("-", "_")
    if normalized in {"iid", "independent", "independent_and_identically_distributed"}:
        return "IID"
    if "temporal" in normalized or "time" in normalized:
        return "Temporal"
    if "group" in normalized:
        return "Grouped"
    raise ValueError(f"Unknown BeyondArena split regime: {raw!r}")


def _is_text_column(series: pd.Series) -> bool:
    dtype = series.dtype
    dtype_name = str(dtype).lower()
    if dtype_name in {"string", "string[python]", "string[pyarrow]"}:
        return True
    if not (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
        return False
    values = series.dropna()
    if values.empty or not values.map(lambda value: isinstance(value, str)).all():
        return False
    unique = int(values.nunique(dropna=True))
    if unique > 100 or unique / max(len(values), 1) >= 0.5:
        return True
    # Long free-form strings are text even when a small fixture happens to repeat
    # values. Short object columns remain ordinary categorical features.
    return float(values.map(len).mean()) > 64


def _is_high_cardinality_column(series: pd.Series) -> bool:
    dtype = series.dtype
    if not (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        return False
    values = series.dropna()
    unique = int(values.nunique(dropna=True))
    return unique > 100 or unique / max(len(values), 1) >= 0.5


def _explicit_feature_flags(container: Any) -> tuple[bool, bool]:
    """Read optional future Data Foundry feature flags when present.

    Data Foundry 0.0.5 exposes dtypes through the curated DataFrame, while a
    few local warehouse versions attach summary fields to the container.  Honor
    those fields when available without making the benchmark depend on a
    private schema extension.
    """

    objects = [container, getattr(container, "dataset_metadata", None), getattr(container, "task_metadata", None)]
    text = False
    high_cardinality = False
    for obj in objects:
        for key in ("has_text", "contains_text", "text_features", "has_text_features"):
            value = _attribute_or_mapping(obj, (key,), None)
            if isinstance(value, bool):
                text |= value
            elif isinstance(value, (list, tuple, set, dict)):
                text |= bool(value)
        for key in (
            "has_high_cardinality",
            "high_cardinality",
            "has_high_cardinality_categoricals",
            "high_cardinality_categorical",
        ):
            value = _attribute_or_mapping(obj, (key,), None)
            if isinstance(value, bool):
                high_cardinality |= value
            elif isinstance(value, (list, tuple, set, dict)):
                high_cardinality |= bool(value)
    return text, high_cardinality


def inspect_beyondarena_container(
    container: Any,
    *,
    rows_max: int = ROWS_MAX,
    features_max: int = FEATURES_MAX,
    classes_min: int = CLASSES_MIN,
    classes_max: int = CLASSES_MAX,
) -> TaskInspection:
    """Inspect and filter one Data Foundry container.

    The raw feature count is measured before removing the grouped identifier;
    the model-input feature count is recorded separately in fold metrics.  This
    keeps the selection rule faithful to the published raw-table limit while
    still guaranteeing that identifiers cannot enter the model.
    """

    dataset = getattr(container, "dataset", None)
    dataset_metadata = getattr(container, "dataset_metadata", None)
    task_metadata = getattr(container, "task_metadata", None)
    name = str(_attribute_or_mapping(dataset_metadata, ("unique_name", "name"), "unknown"))
    uuid = str(getattr(container, "uuid", ""))
    checksum = str(getattr(container, "checksum", ""))
    row: dict[str, Any] = {
        "task_name": name,
        "uuid": uuid,
        "checksum": checksum,
        "status": "skipped",
        "reason": None,
    }
    if dataset is None or task_metadata is None:
        row["reason"] = "missing_container_data_or_task_metadata"
        return TaskInspection(None, row)
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)
    target = str(_attribute_or_mapping(task_metadata, ("target_column_name", "target"), ""))
    group_columns = _as_columns(_attribute_or_mapping(task_metadata, ("group_on",), None))
    time_value = _attribute_or_mapping(task_metadata, ("time_on",), None)
    time_column = None if time_value in (None, "") else str(time_value)
    problem_type = _problem_type(task_metadata)
    metadata_objects = (dataset_metadata, task_metadata)
    rows = _metadata_number(metadata_objects, ("n_rows", "num_rows", "rows", "num_instances")) or len(dataset)
    raw_features = _metadata_number(metadata_objects, ("n_features", "num_features", "features"))
    if raw_features is None:
        raw_features = max(0, len(dataset.columns) - (1 if target in dataset.columns else 0))
    if target not in dataset.columns:
        row.update({"rows": rows, "raw_features": raw_features, "problem_type": problem_type})
        row["reason"] = "target_column_missing"
        return TaskInspection(None, row)
    target_values = dataset[target].dropna()
    classes = _metadata_number(metadata_objects, ("n_classes", "num_classes", "classes"))
    if classes is None:
        classes = int(target_values.nunique(dropna=True))
    feature_columns = [column for column in dataset.columns if column != target and column not in group_columns]
    text_features = tuple(column for column in feature_columns if _is_text_column(dataset[column]))
    high_cardinality_features = tuple(
        column
        for column in feature_columns
        if column not in text_features and _is_high_cardinality_column(dataset[column])
    )
    explicit_text, explicit_high_cardinality = _explicit_feature_flags(container)
    row.update(
        {
            "rows": rows,
            "raw_features": raw_features,
            "classes": classes,
            "problem_type": problem_type,
            "regime": _regime(task_metadata),
            "target_column": target,
            "group_columns": ";".join(group_columns),
            "time_column": time_column,
            "text_features": ";".join(text_features),
            "high_cardinality_features": ";".join(high_cardinality_features),
        }
    )
    if "classification" not in problem_type:
        row["reason"] = "not_classification"
    elif rows < 1 or rows > rows_max:
        row["reason"] = "rows_limit"
    elif raw_features < 1 or raw_features > features_max:
        row["reason"] = "raw_features_limit"
    elif classes < classes_min or classes > classes_max:
        row["reason"] = "class_count_limit"
    elif explicit_text or text_features:
        row["reason"] = "text_features_excluded"
    elif explicit_high_cardinality or high_cardinality_features:
        row["reason"] = "high_cardinality_features_excluded"
    else:
        task = BeyondArenaTask(
            name=name,
            uuid=uuid,
            checksum=checksum,
            rows=int(rows),
            raw_features=int(raw_features),
            classes=int(classes),
            problem_type=problem_type,
            regime=_regime(task_metadata),
            target_column=target,
            group_columns=group_columns,
            text_features=text_features,
            high_cardinality_features=high_cardinality_features,
            container=container,
            time_column=time_column,
        )
        row["status"] = "eligible"
        return TaskInspection(task, row)
    return TaskInspection(None, row)


def load_beyondarena_collection():
    """Import the optional official collection lazily."""

    try:
        from data_foundry.collections import BEYOND_ARENA
    except ImportError as error:  # pragma: no cover - exercised only without optional extra
        raise ImportError(
            "BeyondArena evaluation requires the optional dependency: "
            "uv sync --extra beyondarena"
        ) from error
    return BEYOND_ARENA


def _task_names_from_manifest(path: str | Path, *, rows_max: int, features_max: int) -> tuple[str, ...]:
    """Read prior eligible tasks and add tasks admitted by the current limits."""

    frame = pd.read_csv(path)
    required = {"task_name", "status", "rows", "raw_features", "classes", "problem_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Task manifest {path} is missing columns: {sorted(missing)}")
    names = frame.loc[frame["status"].eq("eligible"), "task_name"].astype(str).tolist()
    extension = (
        frame["problem_type"].astype(str).str.contains("classification", case=False, na=False)
        & frame["rows"].le(rows_max)
        & frame["raw_features"].le(features_max)
        & frame["classes"].between(CLASSES_MIN, CLASSES_MAX)
    )
    for column in ("text_features", "high_cardinality_features"):
        if column in frame:
            extension &= frame[column].isna() | frame[column].astype(str).isin(("", "nan"))
    names.extend(frame.loc[extension, "task_name"].astype(str).tolist())
    return tuple(dict.fromkeys(names))


def discover_beyondarena_tasks(
    *,
    collection: Any | None = None,
    data_cache: str | None = None,
    task_names: Sequence[str] | None = None,
    rows_max: int = ROWS_MAX,
    features_max: int = FEATURES_MAX,
    classes_min: int = CLASSES_MIN,
    classes_max: int = CLASSES_MAX,
) -> tuple[list[BeyondArenaTask], list[dict[str, Any]]]:
    """Load every official container and return eligible tasks plus manifest rows."""

    collection = load_beyondarena_collection() if collection is None else collection
    tasks: list[BeyondArenaTask] = []
    manifest: list[dict[str, Any]] = []
    entries = getattr(collection, "entries", None)
    requested_names = set(task_names or ())
    if entries is None:
        containers: Iterable[Any] = collection.iter_containers(cache_dir=data_cache, load_dataset=True)
    else:
        # Loading entry-by-entry preserves an audit row when one container is
        # unavailable instead of losing all later containers to one exception.
        containers = (
            collection.get_dataset(entry.uuid, cache_dir=data_cache, load_dataset=True)
            for entry in entries
            if not requested_names or str(getattr(entry, "unique_name", "")) in requested_names
        )
    if entries is None:
        iterator = iter(containers)
        while True:
            try:
                container = next(iterator)
            except StopIteration:
                break
            except Exception as error:  # pragma: no cover - depends on remote/cache failures
                manifest.append({"status": "error", "reason": f"{type(error).__name__}: {error}"})
                break
            inspection = inspect_beyondarena_container(
                container,
                rows_max=rows_max,
                features_max=features_max,
                classes_min=classes_min,
                classes_max=classes_max,
            )
            manifest.append(inspection.row)
            if inspection.task is not None:
                tasks.append(inspection.task)
        return tasks, manifest
    for entry in entries:
        if requested_names and str(getattr(entry, "unique_name", "")) not in requested_names:
            continue
        try:
            container = collection.get_dataset(entry.uuid, cache_dir=data_cache, load_dataset=True)
            inspection = inspect_beyondarena_container(
                container,
                rows_max=rows_max,
                features_max=features_max,
                classes_min=classes_min,
                classes_max=classes_max,
            )
        except Exception as error:  # pragma: no cover - depends on remote/cache failures
            inspection = TaskInspection(
                None,
                {
                    "task_name": str(getattr(entry, "unique_name", "unknown")),
                    "uuid": str(getattr(entry, "uuid", "")),
                    "status": "error",
                    "reason": f"{type(error).__name__}: {error}",
                },
            )
        manifest.append(inspection.row)
        if inspection.task is not None:
            tasks.append(inspection.task)
    return tasks, manifest


def _json_metadata(run_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("config.json", "metadata.json", "run_metadata.json", "run_config.json", "args.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            result.update(payload)
    return result


def _v4_verified(state: Mapping[str, Any], external: Mapping[str, Any]) -> bool:
    """Require explicit checkpoint/run metadata for multiregime-v4 provenance."""

    sources: list[Mapping[str, Any]] = [state, external]
    for source in (
        state.get("training_config"),
        state.get("config"),
        state.get("metadata"),
        external.get("training_config"),
        external.get("metadata"),
    ):
        if isinstance(source, Mapping):
            sources.append(source)
    for source in sources:
        source_value = str(source.get("multiregime_source", "")).strip().lower()
        if source_value == "v4":
            return True
        for key in ("v4_provenance", "multiregime_v4", "is_multiregime_v4"):
            if source.get(key) is True:
                return True
        for key in ("v4_dump_path", "multiregime_version"):
            value = source.get(key)
            if value not in (None, "") and "v4" in str(value).lower():
                return True
        for key in ("protocol", "prior_version", "training_prior", "experiment", "model_type"):
            value = str(source.get(key, "")).lower()
            if re.search(r"(?:multiregime|multi_regime)[-_ ]?v4", value):
                return True
    return False


def _z_expose(state: Mapping[str, Any], external: Mapping[str, Any], run_dir: Path) -> bool:
    """Detect checkpoints trained with the synthetic routing score exposed."""

    if any(part.lower().replace("-", "_") == "expose_z" for part in run_dir.parts):
        return True
    sources: list[Mapping[str, Any]] = [state, external]
    for source in (
        state.get("training_config"),
        state.get("config"),
        state.get("metadata"),
        external.get("training_config"),
        external.get("metadata"),
    ):
        if isinstance(source, Mapping):
            sources.append(source)
    for source in sources:
        for key in ("z_expose", "expose_z"):
            value = source.get(key)
            if isinstance(value, bool) and value:
                return True
        for key in ("expose_z_probability", "z_expose_probability"):
            try:
                if float(source.get(key, 0.0)) > 0.0:
                    return True
            except (TypeError, ValueError):
                pass
        for key in ("v4_dump_path", "v4_shared_dump_path", "v4_validation_bank_path", "v4_test_bank_path"):
            if "expose_z" in str(source.get(key, "")).lower():
                return True
    return False


def _validation_ce(state: Mapping[str, Any]) -> float | None:
    for source in (state.get("validation"), state.get("ordinary_validation")):
        if isinstance(source, Mapping):
            for key in ("query_cross_entropy", "cross_entropy", "ce", "loss"):
                value = source.get(key)
                if value is not None:
                    try:
                        result = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(result):
                        return result
    for key in ("validation_query_cross_entropy", "ordinary_validation_ce", "validation_ce"):
        value = state.get(key)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return None


def _step(state: Mapping[str, Any], path: Path) -> int | None:
    value = state.get("step")
    if value is None:
        match = re.search(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _family(metadata: Mapping[str, Any], run_dir: Path) -> str:
    for key in ("family", "model_family", "run_family", "arm"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for component in reversed(run_dir.parts):
        lowered = component.lower()
        if "mr-only" in lowered or "mr_only" in lowered or "multiregime_only" in lowered:
            return "mr-only"
        if "canonical" in lowered or lowered == "original" or lowered.startswith("original-"):
            return "canonical"
        if "r_z" in lowered:
            return "r_z"
        if "g_z" in lowered:
            return "g_z"
    return "v4"


def _size(metadata: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    for key in ("size", "model_size", "width"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    architecture = state.get("architecture")
    if isinstance(architecture, Mapping):
        keys = ("embedding_size", "num_layers", "num_attention_heads", "mlp_hidden_size")
        values = [architecture.get(key) for key in keys]
        if any(value is not None for value in values):
            return "e{}-l{}-h{}-m{}".format(*("?" if value is None else value for value in values))
    return "unknown"


def _model_identity(metadata: Mapping[str, Any], run_dir: Path) -> str:
    for key in ("model_identity", "model_name", "run_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return run_dir.name


def _checkpoint_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".pth" else []
    # Inspect every PyTorch artifact and let the explicit metadata gate below
    # decide whether it is a v4 checkpoint.  Restricting discovery by filename
    # would silently miss best/step checkpoints from other v4 run writers.
    return sorted(root.rglob("*.pth"))


def discover_v4_checkpoints(run_roots: Sequence[str | Path]) -> tuple[list[ModelSelection], list[dict[str, Any]]]:
    """Discover verified runs and return final/best-own-validation selections."""

    by_run: dict[Path, list[CheckpointCandidate]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root_value in run_roots:
        root = Path(root_value).expanduser()
        for path in _checkpoint_files(root):
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(state, Mapping):
                    raise ValueError("checkpoint payload is not a mapping")
                external = _json_metadata(path.parent)
                if not _v4_verified(state, external):
                    audit.append({"path": str(path), "status": "rejected", "reason": "missing_explicit_v4_metadata"})
                    continue
                metadata = dict(external)
                for key in ("training_config", "config"):
                    if isinstance(state.get(key), Mapping):
                        metadata.update(state[key])
                identity = _model_identity(metadata, path.parent)
                by_run[path.parent].append(
                    CheckpointCandidate(
                        path=path,
                        run_dir=path.parent,
                        model_identity=identity,
                        family=_family(metadata, path.parent),
                        size=_size(metadata, state),
                        validation_ce=_validation_ce(state),
                        step=_step(state, path),
                        state=state,
                        z_expose=_z_expose(state, external, path.parent),
                    )
                )
                audit.append({
                    "path": str(path),
                    "status": "verified",
                    "model_identity": identity,
                    "family": _family(metadata, path.parent),
                    "size": _size(metadata, state),
                    "validation_ce": _validation_ce(state),
                    "z_expose": _z_expose(state, external, path.parent),
                })
            except Exception as error:
                audit.append({"path": str(path), "status": "error", "reason": f"{type(error).__name__}: {error}"})

    selections: list[ModelSelection] = []
    identity_counts: dict[str, int] = defaultdict(int)
    for candidates in by_run.values():
        identity_counts[candidates[0].model_identity] += 1
    for run_dir, candidates in sorted(by_run.items(), key=lambda item: str(item[0])):
        first = candidates[0]
        model_identity = first.model_identity
        if identity_counts[model_identity] > 1:
            model_identity = f"{model_identity}@{run_dir}"
        final = next((candidate for candidate in candidates if candidate.path.name == "final_checkpoint.pth"), None)
        best = min(
            (candidate for candidate in candidates if candidate.validation_ce is not None),
            key=lambda candidate: (
                candidate.validation_ce,
                candidate.step if candidate.step is not None else math.inf,
                str(candidate.path),
            ),
            default=None,
        )
        for policy, candidate, reason in (
            ("final", final, None if final is not None else "final_checkpoint_missing_or_unverified"),
            ("best_own_val", best, None if best is not None else "no_stored_ordinary_validation_ce"),
        ):
            selections.append(
                ModelSelection(
                    model_identity=model_identity,
                    family=first.family,
                    size=first.size,
                    checkpoint_policy=policy,
                    checkpoint_path=None if candidate is None else candidate.path,
                    selection_status="available" if candidate is not None else "unavailable",
                    selection_reason=reason,
                    validation_ce=None if candidate is None else candidate.validation_ce,
                    run_dir=run_dir,
                    candidate_count=len(candidates),
                    z_expose=first.z_expose,
                )
            )
    return selections, audit


# Conventional (non-in-context) baselines, default hyper-parameters, fitted per
# fold on the same preprocessed features as the published models. Kept
# deliberately untuned: they are the "what does a standard classifier get with
# no effort" reference, not a tuned upper bound.
CONVENTIONAL_MODELS = ("logreg", "rf", "hgb", "xgboost", "lightgbm", "catboost")
BASELINE_MODELS = ("tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3", "tabicl-v1", "tabicl-v2", *CONVENTIONAL_MODELS)
TABICL_CHECKPOINTS = {
    "tabicl-v1": "tabicl-classifier-v1-20250208.ckpt",
    "tabicl-v2": "tabicl-classifier-v2-20260212.ckpt",
}
TABPFN_CHECKPOINTS = {
    # v2.2's automatic resolver is not reliable on compute nodes without
    # outbound access; the Slurm wrapper supplies this downloaded official
    # artifact.  Leaving it unset preserves package auto-resolution elsewhere.
    "tabpfn-v2.2": os.environ.get("TABPFN_V22_CHECKPOINT"),
    "tabpfn-v2.6": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "tabpfn-v3": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
}
_BASELINE_ALIASES = {
    "v2.2": "tabpfn-v2.2",
    "v2.6": "tabpfn-v2.6",
    "v3": "tabpfn-v3",
    "tabicl1": "tabicl-v1",
    "tabicl2": "tabicl-v2",
}


def _canonical_baseline_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    normalized = _BASELINE_ALIASES.get(normalized, normalized)
    if normalized not in BASELINE_MODELS:
        choices = ", ".join(BASELINE_MODELS)
        raise ValueError(f"Unknown baseline {name!r}; choose from {choices}.")
    return normalized


def baseline_selections(names: Sequence[str]) -> tuple[list[ModelSelection], list[dict[str, Any]]]:
    """Build one published-default selection for each requested external model."""

    selections: list[ModelSelection] = []
    audit: list[dict[str, Any]] = []
    for raw_name in names:
        name = _canonical_baseline_name(raw_name)
        checkpoint = TABPFN_CHECKPOINTS.get(name)
        checkpoint_path = Path(checkpoint) if checkpoint is not None else Path(f"external://{name}")
        if name in CONVENTIONAL_MODELS:
            family, version = "sklearn", name
        else:
            family = "tabicl" if name.startswith("tabicl-") else "tabpfn"
            version = name.removeprefix(f"{family}-")
        selections.append(
            ModelSelection(
                model_identity=name,
                family=family,
                size=version,
                checkpoint_policy="published_default",
                checkpoint_path=checkpoint_path,
                selection_status="available",
                selection_reason=None,
                validation_ce=None,
                run_dir=Path(f"external://{name}"),
                candidate_count=1,
                model_kind=family,
            )
        )
        audit.append(
            {
                "path": str(checkpoint_path),
                "status": "external_published_default",
                "model_identity": name,
                "family": family,
                "size": version,
                "validation_ce": None,
            }
        )
    return selections, audit


def build_conventional_model(name: str) -> Any:
    """Default-hyper-parameter conventional classifiers (sklearn API); n_jobs from SLURM when present."""

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0) or -1
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, random_state=0)
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=0)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=n_jobs, random_state=0, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=n_jobs, random_state=0, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, thread_count=n_jobs, random_seed=0, verbose=0, allow_writing_files=False)
    raise ValueError(f"Unknown conventional model {name!r}; choose from {', '.join(CONVENTIONAL_MODELS)}.")


def build_external_model(selection: ModelSelection, device: str) -> Any:
    """Construct a published TabPFN/TabICL classifier lazily on the worker."""

    if selection.model_kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = selection.model_identity.removeprefix("tabpfn-")
        model_version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[version]
        kwargs = {
            "device": device,
            "random_state": 0,
            "n_estimators": 8,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "fit_mode": "fit_preprocessors",
            "show_progress_bar": False,
        }
        path = str(selection.checkpoint_path) if selection.checkpoint_path is not None else ""
        if path.startswith("external://"):
            return TabPFNClassifier.create_default_for_version(model_version, **kwargs)
        return TabPFNClassifier(model_path=path, **kwargs)
    if selection.model_kind == "sklearn":
        return build_conventional_model(selection.model_identity)
    if selection.model_kind == "tabicl":
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            checkpoint_version=TABICL_CHECKPOINTS[selection.model_identity],
            n_estimators=8,
            softmax_temperature=0.9,
            device=device,
            random_state=0,
            verbose=False,
        )
    raise ValueError(f"Unsupported external model kind: {selection.model_kind!r}")


def select_checkpoint(candidates: Sequence[CheckpointCandidate], policy: str) -> CheckpointCandidate | None:
    """Select a final or lowest ordinary-validation-CE checkpoint deterministically."""

    if policy == "final":
        return next((candidate for candidate in candidates if candidate.path.name == "final_checkpoint.pth"), None)
    if policy != "best_own_val":
        raise ValueError(f"Unknown checkpoint policy: {policy}")
    return min(
        (candidate for candidate in candidates if candidate.validation_ce is not None),
        key=lambda candidate: (
            candidate.validation_ce,
            candidate.step if candidate.step is not None else math.inf,
            str(candidate.path),
        ),
        default=None,
    )


def prepare_fold_data(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    target_column: str,
    group_columns: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the existing feature preprocessor on train only and transform both folds."""

    excluded = {target_column, *group_columns}
    train_features = train_frame.drop(columns=[column for column in excluded if column in train_frame.columns])
    test_features = test_frame.drop(columns=[column for column in excluded if column in test_frame.columns])
    if train_features.shape[1] == 0:
        raise ValueError("No model input features remain after target/group exclusion.")
    preprocessor = get_feature_preprocessor(train_features)
    train_x = np.asarray(preprocessor.fit_transform(train_features), dtype=np.float32)
    test_x = np.asarray(preprocessor.transform(test_features), dtype=np.float32)
    return train_x, test_x, train_features.to_numpy(copy=True), test_features.to_numpy(copy=True)


def _encode_fold_labels(
    train_values: Sequence[Any], test_values: Sequence[Any], classes: int
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([pd.Series(train_values), pd.Series(test_values)], ignore_index=True)
    uniques = list(pd.unique(combined.dropna()))
    if len(uniques) != classes:
        raise ValueError(f"Expected {classes} classes from task metadata, observed {len(uniques)} in fold.")
    mapping = {value: index for index, value in enumerate(uniques)}
    try:
        train_y = np.asarray([mapping[value] for value in train_values], dtype=np.int64)
        test_y = np.asarray([mapping[value] for value in test_values], dtype=np.int64)
    except KeyError as error:
        raise ValueError("The official test fold contains a target value absent from the training fold.") from error
    return train_y, test_y


def _normalise_probabilities(probabilities: np.ndarray, classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        if classes != 2:
            raise UnsupportedFoldError("A one-dimensional prediction cannot represent multiclass probabilities.")
        positive = np.clip(probabilities, 0.0, 1.0)
        probabilities = np.column_stack((1.0 - positive, positive))
    if probabilities.ndim != 2 or probabilities.shape[1] != classes:
        raise UnsupportedFoldError(
            f"Model emitted shape {probabilities.shape}; expected (queries, {classes}) probabilities."
        )
    if not np.isfinite(probabilities).all():
        raise UnsupportedFoldError("Model emitted non-finite probabilities.")
    probabilities = np.clip(probabilities, 1e-7, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def predict_external_full_fold(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    classes: int,
    query_chunk_size: int = 128,
) -> np.ndarray:
    """Fit a published sklearn-style baseline on the complete support fold."""

    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive.")
    if len(train_x) < 1 or len(test_x) < 1:
        raise ValueError("Both official train and test folds must be non-empty.")
    if len(np.unique(train_y)) < 2:
        raise UnsupportedFoldError("Published classifiers require at least two support classes.")
    try:
        model.fit(train_x, train_y)
        model_classes = np.asarray(model.classes_)
        predictions: list[np.ndarray] = []
        for start in range(0, len(test_x), query_chunk_size):
            raw = np.asarray(model.predict_proba(test_x[start : start + query_chunk_size]))
            if raw.ndim != 2 or raw.shape[0] != min(query_chunk_size, len(test_x) - start):
                raise UnsupportedFoldError(f"Published model emitted unexpected shape {raw.shape}.")
            probabilities = np.full((raw.shape[0], classes), 1e-7, dtype=np.float64)
            for column, label in enumerate(model_classes):
                label_index = int(label)
                if 0 <= label_index < classes:
                    probabilities[:, label_index] = raw[:, column]
            predictions.append(_normalise_probabilities(probabilities, classes))
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        raise UnsupportedFoldError(f"complete fold exceeded device memory: {error}") from error
    except RuntimeError as error:
        message = str(error).lower()
        if any(token in message for token in ("out of memory", "memory allocation", "too many resources")):
            raise UnsupportedFoldError(f"complete fold exceeded device limits: {error}") from error
        raise
    except (ValueError, IndexError) as error:
        # Published baselines can reject a complete fold for unsupported
        # cardinality/shape constraints.  Preserve the no-fallback contract.
        raise UnsupportedFoldError(f"published model cannot score complete fold: {error}") from error
    return np.concatenate(predictions, axis=0)


@torch.no_grad()
def predict_full_fold(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    classes: int,
    device: str,
    query_chunk_size: int = 128,
    num_mem_chunks: int = 8,
) -> np.ndarray:
    """Score every query row using the complete support fold.

    Chunking changes only the query batch size; ``train_x`` and ``train_y`` are
    reconstructed in every call without subsampling or truncation.
    """

    if query_chunk_size < 1 or num_mem_chunks < 1:
        raise ValueError("query_chunk_size and num_mem_chunks must be positive.")
    if len(train_x) < 1 or len(test_x) < 1:
        raise ValueError("Both official train and test folds must be non-empty.")
    model.to(device).eval()
    support_x = torch.as_tensor(train_x, dtype=torch.float32, device=device).unsqueeze(0)
    support_y = torch.as_tensor(train_y, dtype=torch.float32, device=device).unsqueeze(0)
    predictions: list[np.ndarray] = []
    for start in range(0, len(test_x), query_chunk_size):
        query = torch.as_tensor(
            test_x[start : start + query_chunk_size], dtype=torch.float32, device=device
        ).unsqueeze(0)
        table_x = torch.cat((support_x, query), dim=1)
        try:
            logits = model(
                (table_x, support_y),
                train_test_split_index=len(train_x),
                num_mem_chunks=num_mem_chunks,
            )
        except (torch.cuda.OutOfMemoryError, MemoryError) as error:
            raise UnsupportedFoldError(f"complete fold exceeded device memory: {error}") from error
        except RuntimeError as error:
            message = str(error).lower()
            if any(token in message for token in ("out of memory", "mps", "memory allocation", "too many resources")):
                raise UnsupportedFoldError(f"complete fold exceeded device limits: {error}") from error
            raise
        logits = torch.as_tensor(logits)
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != query.shape[1]:
            raise UnsupportedFoldError(f"Model emitted unexpected logits shape {tuple(logits.shape)}")
        if logits.shape[-1] < classes:
            raise UnsupportedFoldError(
                f"Model has {logits.shape[-1]} output classes but the task requires {classes}."
            )
        probabilities = F.softmax(logits[..., :classes], dim=-1)[0].detach().cpu().numpy()
        predictions.append(_normalise_probabilities(probabilities, classes))
    return np.concatenate(predictions, axis=0)


def fold_metrics(train_y: np.ndarray, test_y: np.ndarray, probabilities: np.ndarray, classes: int) -> dict[str, float]:
    """Compute the three requested metrics and their training-fold baselines."""

    probabilities = _normalise_probabilities(probabilities, classes)
    counts = np.bincount(train_y, minlength=classes).astype(np.float64)
    prior = counts / counts.sum()
    prior = np.clip(prior, 1e-7, 1.0)
    prior /= prior.sum()
    prior_ce = float(log_loss(test_y, np.broadcast_to(prior, (len(test_y), classes)), labels=np.arange(classes)))
    model_ce = float(log_loss(test_y, probabilities, labels=np.arange(classes)))
    if len(np.unique(test_y)) < classes:
        auc = float("nan")
    else:
        try:
            if classes == 2:
                auc = float(roc_auc_score(test_y, probabilities[:, 1]))
            else:
                auc = float(roc_auc_score(test_y, probabilities, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == test_y))
    majority_accuracy = float(np.mean(test_y == int(np.argmax(counts))))
    return {
        "model_cross_entropy": model_ce,
        "prior_cross_entropy": prior_ce,
        "excess_cross_entropy": model_ce - prior_ce,
        "macro_ovr_auc": auc,
        "accuracy": accuracy,
        "majority_accuracy": majority_accuracy,
        "accuracy_gain": accuracy - majority_accuracy,
    }


def _is_unsupported(error: BaseException) -> bool:
    if isinstance(error, (UnsupportedFoldError, MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return any(token in message for token in ("out of memory", "memory allocation", "too many resources"))
    return False


def _task_metric_conditions(tasks: Sequence[BeyondArenaTask]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for condition in task.conditions():
            result[condition].add(task.name)
    return result


def _selection_row(selection: ModelSelection) -> dict[str, Any]:
    return {
        "model_identity": selection.model_identity,
        "family": selection.family,
        "size": selection.size,
        "checkpoint_policy": selection.checkpoint_policy,
        "checkpoint_path": None if selection.checkpoint_path is None else str(selection.checkpoint_path),
        "selection_status": selection.selection_status,
        "selection_reason": selection.selection_reason,
        "validation_ce": selection.validation_ce,
        "run_dir": str(selection.run_dir),
        "candidate_count": selection.candidate_count,
        "model_kind": selection.model_kind,
        "z_expose": selection.z_expose,
    }


def evaluate_selected_models(
    tasks: Sequence[BeyondArenaTask],
    selections: Sequence[ModelSelection],
    *,
    device: str,
    query_chunk_size: int,
    num_mem_chunks: int,
    model_loader: Callable[[str | Path, str], Any] = load_checkpoint_for_inference,
    fail_on_task_error: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate selected checkpoints on every official fold."""

    rows: list[dict[str, Any]] = []
    for selection in selections:
        base = _selection_row(selection)
        if selection.checkpoint_path is None:
            for task in tasks:
                for condition in task.conditions():
                    rows.append({
                        **base,
                        "task_name": task.name,
                        "task_uuid": task.uuid,
                        "repeat": None,
                        "fold": None,
                        "condition": condition,
                        "regime": task.regime,
                        "problem_bucket": task.problem_bucket,
                        "row_bucket": task.row_bucket,
                        "feature_bucket": task.feature_bucket,
                        "status": "unavailable",
                        "reason": selection.selection_reason,
                    })
            continue
        try:
            if selection.model_kind in {"tabpfn", "tabicl", "sklearn"}:
                model = build_external_model(selection, device)
            else:
                model = model_loader(selection.checkpoint_path, device)
        except Exception as error:
            for task in tasks:
                for condition in task.conditions():
                    rows.append({
                        **base,
                        "task_name": task.name,
                        "task_uuid": task.uuid,
                        "repeat": None,
                        "fold": None,
                        "condition": condition,
                        "regime": task.regime,
                        "problem_bucket": task.problem_bucket,
                        "row_bucket": task.row_bucket,
                        "feature_bucket": task.feature_bucket,
                        "status": "unsupported" if _is_unsupported(error) else "error",
                        "reason": f"{type(error).__name__}: {error}",
                    })
            if fail_on_task_error:
                raise
            continue
        for task in tasks:
            dataset = task.container.dataset
            splits = task.container.experiment_metadata.splits
            for repeat, folds in sorted(splits.items(), key=lambda item: int(item[0])):
                for fold, split in sorted(folds.items(), key=lambda item: int(item[0])):
                    train_indices, test_indices = split
                    fold_base = {
                        **base,
                        "task_name": task.name,
                        "task_uuid": task.uuid,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "regime": task.regime,
                        "problem_bucket": task.problem_bucket,
                        "row_bucket": task.row_bucket,
                        "feature_bucket": task.feature_bucket,
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                    }
                    try:
                        train_frame = dataset.iloc[list(train_indices)]
                        test_frame = dataset.iloc[list(test_indices)]
                        train_x, test_x, raw_train_x, raw_test_x = prepare_fold_data(
                            train_frame,
                            test_frame,
                            target_column=task.target_column,
                            group_columns=task.group_columns,
                        )
                        train_y, test_y = _encode_fold_labels(
                            train_frame[task.target_column].tolist(),
                            test_frame[task.target_column].tolist(),
                            task.classes,
                        )
                        if selection.model_kind in {"tabpfn", "tabicl", "sklearn"}:
                            # A fresh estimator per fold: XGBClassifier.fit rewrites self.objective to
                            # multi:softprob after a multiclass task, which then breaks every later binary fit.
                            fold_model = build_conventional_model(selection.model_identity) if selection.model_kind == "sklearn" else model
                            probabilities = predict_external_full_fold(
                                fold_model,
                                train_x,
                                train_y,
                                test_x,
                                classes=task.classes,
                                query_chunk_size=query_chunk_size,
                            )
                        else:
                            probabilities = predict_full_fold(
                                model,
                                train_x,
                                train_y,
                                test_x,
                                classes=task.classes,
                                device=device,
                                query_chunk_size=query_chunk_size,
                                num_mem_chunks=num_mem_chunks,
                            )
                        metrics = fold_metrics(train_y, test_y, probabilities, task.classes)
                        for condition in task.conditions():
                            rows.append({
                                **fold_base,
                                "condition": condition,
                                "status": "evaluated",
                                "reason": None,
                                "raw_train_features": raw_train_x.shape[1],
                                "processed_features": train_x.shape[1],
                                **metrics,
                            })
                    except Exception as error:
                        if fail_on_task_error:
                            raise
                        for condition in task.conditions():
                            rows.append({
                                **fold_base,
                                "condition": condition,
                                "status": "unsupported" if _is_unsupported(error) else "error",
                                "reason": f"{type(error).__name__}: {error}",
                            })
    return rows


METRIC_DIRECTIONS = {
    "excess_cross_entropy": False,
    "macro_ovr_auc": True,
    "accuracy_gain": True,
}


def aggregate_rankings(
    tasks: Sequence[BeyondArenaTask],
    selections: Sequence[ModelSelection],
    fold_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Rank models per task after fold averaging, then average task ranks.

    The reported ``score`` remains the equal-task-weighted metric score.  The
    reported ``rank`` is the mean of ranks assigned independently within each
    task, so a task cannot dominate because it has more folds or rows.
    """

    condition_tasks = _task_metric_conditions(tasks)
    frame = pd.DataFrame(fold_rows)
    ranking_rows: list[dict[str, Any]] = []
    task_score_groups: dict[tuple[str, str, str], dict[tuple[str, str], float]] = defaultdict(dict)
    metric_names = tuple(METRIC_DIRECTIONS)
    for selection in selections:
        selection_base = _selection_row(selection)
        selection_key = (selection.model_identity, selection.checkpoint_policy)
        model_rows = (
            frame[
                (frame.model_identity == selection.model_identity)
                & (frame.checkpoint_policy == selection.checkpoint_policy)
            ]
            if not frame.empty
            else pd.DataFrame()
        )
        for condition, condition_task_names in sorted(condition_tasks.items()):
            for metric in metric_names:
                valid = (
                    model_rows[
                        (model_rows.condition == condition)
                        & (model_rows.status == "evaluated")
                        & model_rows[metric].notna()
                    ]
                    if not model_rows.empty
                    else pd.DataFrame()
                )
                if valid.empty:
                    score = float("nan")
                    task_count = 0
                    fold_count = 0
                else:
                    # First average all folds for each task.  Only then is each
                    # task given one equal-weight contribution to the score.
                    task_scores = valid.groupby("task_name", sort=False)[metric].mean()
                    score = float(task_scores.mean())
                    task_count = int(task_scores.size)
                    fold_count = int(valid["task_name"].count())
                    for task_name, task_score in task_scores.items():
                        task_score_groups[(condition, metric, str(task_name))][selection_key] = float(task_score)
                unsupported_tasks = 0
                failed_tasks = 0
                for task_name in condition_task_names:
                    task_rows = (
                        model_rows[model_rows.task_name == task_name]
                        if not model_rows.empty
                        else pd.DataFrame()
                    )
                    if task_rows.empty or not (task_rows.status == "evaluated").any():
                        statuses = set(task_rows.status.tolist()) if not task_rows.empty else {"unavailable"}
                        if "unsupported" in statuses or "unavailable" in statuses:
                            unsupported_tasks += 1
                        elif statuses:
                            failed_tasks += 1
                ranking_rows.append({
                    **selection_base,
                    "condition": condition,
                    "metric": metric,
                    "rank": None,
                    "score": score,
                    "task_count": task_count,
                    "fold_count": fold_count,
                    "rank_task_count": 0,
                    "eligible_task_count": len(condition_task_names),
                    "unsupported_task_count": unsupported_tasks,
                    "unsupported_task_coverage": unsupported_tasks / len(condition_task_names),
                    "supported_task_coverage": task_count / len(condition_task_names),
                    "failed_task_count": failed_tasks,
                })
    rankings = pd.DataFrame(ranking_rows)
    if rankings.empty:
        return rankings

    rank_sums: defaultdict[tuple[str, str, tuple[str, str]], float] = defaultdict(float)
    rank_counts: defaultdict[tuple[str, str, tuple[str, str]], int] = defaultdict(int)
    for (condition, metric, _task_name), model_scores in task_score_groups.items():
        task_ranks = pd.Series(model_scores, dtype=float).rank(
            method="min", ascending=not METRIC_DIRECTIONS[metric], na_option="keep"
        )
        for selection_key, task_rank in task_ranks.items():
            if pd.notna(task_rank):
                rank_key = (condition, metric, selection_key)
                rank_sums[rank_key] += float(task_rank)
                rank_counts[rank_key] += 1

    for index, row in rankings.iterrows():
        rank_key = (row["condition"], row["metric"], (row["model_identity"], row["checkpoint_policy"]))
        count = rank_counts.get(rank_key, 0)
        rankings.at[index, "rank_task_count"] = count
        if count:
            rankings.at[index, "rank"] = rank_sums[rank_key] / count
    rankings["rank"] = rankings["rank"].astype(float)
    return rankings


def _markdown_summary(
    config: BeyondArenaConfig,
    tasks: Sequence[BeyondArenaTask],
    manifest: Sequence[Mapping[str, Any]],
    selections: Sequence[ModelSelection],
    rankings: pd.DataFrame,
) -> str:
    lines = [
        "# Multiregime-v4 NanoTabPFN on BeyondArena",
        "",
        "Protocol: official Data Foundry containers and every official outer fold; "
        "complete training folds are used as context.",
        "",
        "- Eligible tasks: classes 2–5, rows ≤ 10,000, raw features ≤ 30, classification only.",
                    "- Excluded: text and high-cardinality categorical tasks; target and grouped identifiers "
        "are removed from model inputs.",
        "- Model selection: `final_checkpoint.pth` and lowest stored ordinary-prior validation CE (`best_own_val`).",
        "- Aggregation: folds are averaged within task, then tasks receive equal weight for scores; "
        "ranks are computed within each task and averaged across tasks.",
        "- Metrics: excess CE (lower), macro OVR AUC (higher), and accuracy gain (higher).",
        "",
        f"Eligible tasks: {len(tasks)} / manifest rows: {len(manifest)}",
        f"Selections: {len(selections)} "
        f"({sum(selection.selection_status == 'available' for selection in selections)} available)",
        "",
    ]
    if rankings.empty:
        lines.append("No ranking scores were produced. Check task_manifest.csv and checkpoints.csv.")
        return "\n".join(lines) + "\n"
    available = rankings[(rankings.selection_status == "available") & rankings.score.notna()]
    if available.empty:
        lines.append("No available model selection produced a score.")
        return "\n".join(lines) + "\n"
    lines.extend(["## Best available selection by condition and metric", ""])
    for (condition, metric), group in available.groupby(["condition", "metric"], sort=True):
        ascending = not METRIC_DIRECTIONS[metric]
        best = group.sort_values("score", ascending=ascending).iloc[0]
        lines.append(
            f"- `{condition}` / `{metric}`: `{best.model_identity}` "
            f"({best.checkpoint_policy}), score `{best.score:.6g}`, "
            f"tasks `{int(best.task_count)}`, unsupported coverage `{best.unsupported_task_coverage:.1%}`"
        )
    lines.append("")
    lines.append(
        "Unsupported full-fold evaluations have no metric score and are never replaced "
        "with a subsampled result."
    )
    return "\n".join(lines) + "\n"


def run(
    config: BeyondArenaConfig,
    *,
    collection: Any | None = None,
    model_loader: Callable[[str | Path, str], Any] = load_checkpoint_for_inference,
) -> Path:
    """Run the full sweep and write the four requested artifacts."""

    if config.query_chunk_size < 1 or config.num_mem_chunks < 1:
        raise ValueError("query_chunk_size and num_mem_chunks must be positive.")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    tasks, manifest = discover_beyondarena_tasks(
        collection=collection,
        data_cache=config.data_cache,
        rows_max=config.rows_max,
        features_max=config.features_max,
        classes_min=config.classes_min,
        classes_max=config.classes_max,
        task_names=(config.smoke_task,)
        if config.smoke_task is not None
        else (
            _task_names_from_manifest(
                config.task_names_file,
                rows_max=config.rows_max,
                features_max=config.features_max,
            )
            if config.task_names_file is not None
            else None
        ),
    )
    if config.smoke_task is not None:
        tasks = [task for task in tasks if task.name == config.smoke_task]
        if not tasks:
            raise ValueError(f"Smoke task {config.smoke_task!r} was not eligible or was not found.")
    pd.DataFrame(manifest).to_csv(output / "task_manifest.csv", index=False)
    selections, checkpoint_audit = discover_v4_checkpoints(config.run_roots)
    baselines, baseline_audit = baseline_selections(config.baseline_models)
    selections.extend(baselines)
    checkpoint_audit.extend(baseline_audit)
    pd.DataFrame(checkpoint_audit).to_csv(output / "checkpoints.csv", index=False)
    fold_rows = evaluate_selected_models(
        tasks,
        selections,
        device=config.device,
        query_chunk_size=config.query_chunk_size,
        num_mem_chunks=config.num_mem_chunks,
        model_loader=model_loader,
        fail_on_task_error=config.fail_on_task_error,
    )
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    rankings = aggregate_rankings(tasks, selections, fold_rows)
    rankings.to_csv(output / "rankings.csv", index=False)
    (output / "ranking_summary.md").write_text(_markdown_summary(config, tasks, manifest, selections, rankings))
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    defaults = BeyondArenaConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", required=True, help="Comma-separated run roots containing v4 .pth checkpoints.")
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--data-cache", default=defaults.data_cache)
    parser.add_argument("--rows-max", type=int, default=defaults.rows_max)
    parser.add_argument("--features-max", type=int, default=defaults.features_max)
    parser.add_argument("--query-chunk-size", type=int, default=defaults.query_chunk_size)
    parser.add_argument("--num-mem-chunks", type=int, default=defaults.num_mem_chunks)
    parser.add_argument("--smoke-task", default=None, help="Evaluate exactly one eligible task before the full sweep.")
    parser.add_argument(
        "--task-names-file",
        default=defaults.task_names_file,
        help="Prior task_manifest.csv; reuse eligible tasks and add tasks within current limits.",
    )
    parser.add_argument(
        "--baselines",
        default="",
        help="Comma-separated published baselines: tabpfn-v2.2,tabpfn-v2.6,tabpfn-v3,tabicl-v1,tabicl-v2,logreg,rf,hgb,xgboost,lightgbm,catboost.",
    )
    parser.add_argument("--fail-on-task-error", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BeyondArenaConfig(
        run_roots=tuple(filter(None, (item.strip() for item in args.run_roots.split(",")))),
        output_dir=args.output_dir,
        device=args.device,
        data_cache=args.data_cache,
        rows_max=args.rows_max,
        features_max=args.features_max,
        query_chunk_size=args.query_chunk_size,
        num_mem_chunks=args.num_mem_chunks,
        smoke_task=args.smoke_task,
        task_names_file=args.task_names_file,
        baseline_models=tuple(filter(None, (item.strip() for item in args.baselines.split(",")))),
        fail_on_task_error=args.fail_on_task_error,
    )
    try:
        print(run(config), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BeyondArenaConfig",
    "BeyondArenaTask",
    "CheckpointCandidate",
    "ModelSelection",
    "TaskInspection",
    "UnsupportedFoldError",
    "aggregate_rankings",
    "baseline_selections",
    "bucket_value",
    "discover_beyondarena_tasks",
    "discover_v4_checkpoints",
    "evaluate_selected_models",
    "feature_bucket",
    "fold_metrics",
    "inspect_beyondarena_container",
    "load_beyondarena_collection",
    "predict_full_fold",
    "predict_external_full_fold",
    "prepare_fold_data",
    "row_bucket",
    "run",
    "select_checkpoint",
]
