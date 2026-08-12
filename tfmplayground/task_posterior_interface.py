"""Scikit-learn interface for the task-posterior adapter."""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from tfmplayground.interface import NanoTabPFNClassifier, get_feature_preprocessor
from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    load_task_posterior_checkpoint,
)
from tfmplayground.utils import get_default_device


def _dense_float32(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


class TaskPosteriorClassifier(ClassifierMixin, BaseEstimator):
    """TabArena-facing classifier with bounded, deterministic context ensembles.

    Fitting stores the in-context table; it does not optimize on the benchmark
    split.  A trained adapter checkpoint can be supplied with ``model``.  With
    ``model=None``, the zero-initialized adapter is an exact vanilla baseline.
    """

    def __init__(
        self,
        model: NanoTabPFNTaskPosteriorAdapter | str | None = None,
        *,
        particle_count: int = 4,
        context_size: int = 1024,
        context_ensembles: int = 4,
        random_state: int | None = 0,
        context_mode: str = "iid_set",
        device: str | torch.device | None = None,
        query_chunk_size: int = 128,
        num_mem_chunks: int = 8,
    ):
        self.model = model
        self.particle_count = particle_count
        self.context_size = context_size
        self.context_ensembles = context_ensembles
        self.random_state = random_state
        self.context_mode = context_mode
        self.device = device
        self.query_chunk_size = query_chunk_size
        self.num_mem_chunks = num_mem_chunks

    def _validate_hyperparameters(self) -> None:
        if self.particle_count < 1:
            raise ValueError("particle_count must be positive.")
        if self.context_size < 2 or self.context_size > 10_000:
            raise ValueError("context_size must be between 2 and 10,000.")
        if self.context_ensembles < 1 or self.query_chunk_size < 1 or self.num_mem_chunks < 1:
            raise ValueError("Ensemble and chunk counts must be positive.")
        if self.context_mode not in {"iid_set", "sequential"}:
            raise ValueError("context_mode must be 'iid_set' or 'sequential'.")

    def _initialize_model(self) -> NanoTabPFNTaskPosteriorAdapter:
        if isinstance(self.model, NanoTabPFNTaskPosteriorAdapter):
            adapter = self.model
        elif isinstance(self.model, (str, os.PathLike)):
            adapter, _ = load_task_posterior_checkpoint(self.model)
        elif self.model is None:
            # Reuse the existing official-checkpoint cache/download behavior.
            backbone = NanoTabPFNClassifier(device="cpu").model
            adapter = NanoTabPFNTaskPosteriorAdapter(
                backbone,
                particle_count=self.particle_count,
                max_classes=min(10, backbone.num_outputs),
                context_mode=self.context_mode,
            )
        else:
            raise TypeError("model must be an adapter, checkpoint path, or None.")
        if adapter.particle_count != self.particle_count:
            raise ValueError(
                f"Configured particle_count={self.particle_count}, checkpoint has {adapter.particle_count}."
            )
        if adapter.context_mode != self.context_mode:
            raise ValueError(f"Configured context_mode={self.context_mode!r}, checkpoint has {adapter.context_mode!r}.")
        return adapter

    def fit(self, X, y):
        self._validate_hyperparameters()
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(y) < 2:
            raise ValueError("At least two training rows are required.")
        if getattr(X, "shape", (0, 0))[1] > 500:
            raise ValueError("The adapter supports at most 500 input features.")
        self.label_encoder_ = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        if not 2 <= len(self.classes_) <= 10:
            raise ValueError("The adapter supports classification with 2 to 10 classes.")
        self.feature_preprocessor_ = get_feature_preprocessor(X)
        self.X_train_ = _dense_float32(self.feature_preprocessor_.fit_transform(X))
        self.y_train_ = self.label_encoder_.transform(np.asarray(y)).astype(np.int64)
        self.model_ = self._initialize_model()
        if len(self.classes_) > self.model_.max_classes:
            raise ValueError(f"Checkpoint supports {self.model_.max_classes} classes, found {len(self.classes_)}.")
        self.device_ = torch.device(self.device if self.device is not None else get_default_device())
        self.model_.to(self.device_).eval()
        return self

    def _context_indices(self, ensemble: int) -> np.ndarray:
        seed = (0 if self.random_state is None else int(self.random_state)) + ensemble * 104_729
        rng = np.random.default_rng(seed)
        count = min(self.context_size, len(self.y_train_))
        if count == len(self.y_train_):
            return rng.permutation(count)
        indices, _ = train_test_split(
            np.arange(len(self.y_train_)),
            train_size=count,
            stratify=self.y_train_,
            random_state=seed,
        )
        return rng.permutation(indices)

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query_x = _dense_float32(self.feature_preprocessor_.transform(X))
        ensemble_predictions = []
        for ensemble in range(self.context_ensembles):
            indices = self._context_indices(ensemble)
            context_x = torch.as_tensor(self.X_train_[indices], dtype=torch.float32, device=self.device_).unsqueeze(0)
            context_y = torch.as_tensor(self.y_train_[indices], dtype=torch.long, device=self.device_).unsqueeze(0)
            chunks = []
            for start in range(0, len(query_x), self.query_chunk_size):
                query = torch.as_tensor(
                    query_x[start : start + self.query_chunk_size],
                    dtype=torch.float32,
                    device=self.device_,
                ).unsqueeze(0)
                if self.context_mode == "iid_set":
                    prediction = self.model_(
                        context_x,
                        context_y,
                        query,
                        class_count=len(self.classes_),
                        num_mem_chunks=self.num_mem_chunks,
                    )
                else:
                    split = max(2, len(indices) // 2)
                    split = min(split, len(indices) - 1)
                    prediction = self.model_.forward_sequential(
                        context_x[:, :split],
                        context_y[:, :split],
                        context_x[:, split:],
                        context_y[:, split:],
                        query,
                        class_count=len(self.classes_),
                        num_mem_chunks=self.num_mem_chunks,
                    )
                chunks.append(prediction.marginal_probabilities()[0].cpu().numpy())
            ensemble_predictions.append(np.concatenate(chunks, axis=0))
        return np.mean(ensemble_predictions, axis=0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}
