"""Scikit-learn interfaces for slot-free continuous uncertainty models.

``fit`` only stores the labelled context.  ``predict_proba`` returns the frozen
vanilla probabilities and records uncertainty diagnostics as a side channel, so
query labels can never influence model construction or the posterior.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from tfmplayground.interface import NanoTabPFNClassifier, get_feature_preprocessor, init_model_from_state_dict_file
from tfmplayground.models.continuous_posterior import (
    ContextResamplingUncertainty,
    NanoTabPFNContinuousPosteriorModel,
    load_continuous_checkpoint,
)
from tfmplayground.utils import get_default_device


def _dense(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def deterministic_context_indices(row_count: int, context_size: int, random_state: int) -> np.ndarray:
    """The context protocol shared by every arm of this comparison.

    It reproduces ``VanillaNanoTabPFNClassifier._context_indices`` exactly so
    that all methods see identical labelled context rows.
    """
    if context_size < 2:
        raise ValueError("context_size must be at least two.")
    count = min(context_size, row_count)
    rng = np.random.default_rng(random_state)
    if count == row_count:
        return rng.permutation(row_count)
    return rng.choice(row_count, size=count, replace=False)


class _ContextClassifierBase(ClassifierMixin, BaseEstimator):
    """Shared binary fit/context handling for uncertainty arms."""

    def __init__(
        self,
        model=None,
        *,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        query_chunk_size: int = 512,
    ):
        self.model = model
        self.context_size = context_size
        self.random_state = random_state
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        self.query_chunk_size = query_chunk_size

    def _fit_common(self, X, y) -> None:
        if len(X) != len(y) or len(y) < 2:
            raise ValueError("X and y must contain the same at least two rows.")
        self.label_encoder_ = LabelEncoder().fit(np.asarray(y))
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("The continuous uncertainty track supports binary classification only.")
        self.classes_ = self.label_encoder_.classes_
        self.feature_preprocessor_ = get_feature_preprocessor(X)
        self.X_train_ = _dense(self.feature_preprocessor_.fit_transform(X))
        self.y_train_ = self.label_encoder_.transform(np.asarray(y)).astype(np.float32)
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive.")

    def _context_indices(self) -> np.ndarray:
        return deterministic_context_indices(len(self.y_train_), self.context_size, self.random_state)

    def _context_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._context_indices()
        support_x = torch.as_tensor(self.X_train_[indices], device=self.device_).unsqueeze(0)
        support_y = torch.as_tensor(self.y_train_[indices], device=self.device_).unsqueeze(0)
        return support_x, support_y

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _diagnostics_from_prediction(prediction) -> dict[str, np.ndarray]:
    return {
        "predictive_entropy": prediction.predictive_entropy()[0].cpu().numpy(),
        "expected_conditional_entropy": prediction.expected_conditional_entropy()[0].cpu().numpy(),
        "mutual_information": prediction.mutual_information()[0].cpu().numpy(),
        "epistemic_variance": prediction.epistemic_variance()[0].cpu().numpy(),
        "sample_mean_preservation_error": np.full(
            prediction.sample_positive.shape[-1],
            float(prediction.mean_preservation_error().max()),
            dtype=float,
        ),
    }


class ContinuousUncertaintyClassifier(_ContextClassifierBase):
    """Deployed classifier for a continuous or Beta uncertainty checkpoint.

    The returned probabilities are the frozen vanilla ones.  Posterior samples
    are used only to populate ``last_diagnostics_``.
    """

    def __init__(
        self,
        model=None,
        *,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        query_chunk_size: int = 512,
        num_samples: int = 32,
        inference_seed: int = 0,
    ):
        super().__init__(
            model,
            context_size=context_size,
            random_state=random_state,
            device=device,
            num_mem_chunks=num_mem_chunks,
            query_chunk_size=query_chunk_size,
        )
        self.num_samples = num_samples
        self.inference_seed = inference_seed

    def fit(self, X, y):
        self._fit_common(X, y)
        if isinstance(self.model, torch.nn.Module):
            model = self.model
        elif isinstance(self.model, (str, os.PathLike)):
            model, _checkpoint = load_continuous_checkpoint(self.model)
        elif self.model is None:
            backbone = NanoTabPFNClassifier(device="cpu").model
            model = NanoTabPFNContinuousPosteriorModel(backbone)
        else:
            raise TypeError("model must be an uncertainty module, checkpoint path, or None.")
        target_device = self.device if self.device is not None else get_default_device()
        self.model_ = model.to(target_device).requires_grad_(False).eval()
        self.device_ = next(self.model_.parameters()).device
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = _dense(self.feature_preprocessor_.transform(X))
        support_x, support_y = self._context_tensors()
        probabilities: list[np.ndarray] = []
        diagnostics: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(query), self.query_chunk_size):
            chunk = torch.as_tensor(query[start : start + self.query_chunk_size], device=self.device_).unsqueeze(0)
            # Queries attend only to the support rows, and the latent draws come
            # from the fixed inference seed, so chunking is exact.
            prediction = self.model_(
                support_x,
                support_y,
                chunk,
                num_mem_chunks=self.num_mem_chunks,
                num_samples=self.num_samples,
                sample_seed=self.inference_seed,
            )
            probabilities.append(prediction.marginal_probabilities()[0].cpu().numpy())
            for key, value in _diagnostics_from_prediction(prediction).items():
                diagnostics.setdefault(key, []).append(value)
        self.last_diagnostics_ = {key: np.concatenate(values) for key, values in diagnostics.items()}
        return np.concatenate(probabilities)


class ContextResamplingClassifier(_ContextClassifierBase):
    """Non-learned uncertainty from deterministic stratified support subsets."""

    def __init__(
        self,
        model=None,
        *,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        query_chunk_size: int = 512,
        num_subsets: int = 16,
        fractions: tuple[float, ...] = (0.50, 0.75, 0.90),
        subset_seed: int = 0,
    ):
        super().__init__(
            model,
            context_size=context_size,
            random_state=random_state,
            device=device,
            num_mem_chunks=num_mem_chunks,
            query_chunk_size=query_chunk_size,
        )
        self.num_subsets = num_subsets
        self.fractions = fractions
        self.subset_seed = subset_seed

    def fit(self, X, y):
        self._fit_common(X, y)
        if isinstance(self.model, torch.nn.Module):
            backbone = self.model
        elif isinstance(self.model, (str, os.PathLike)):
            backbone = init_model_from_state_dict_file(str(self.model))
        elif self.model is None:
            backbone = NanoTabPFNClassifier(device="cpu").model
        else:
            raise TypeError("model must be a nanoTabPFN module, checkpoint path, or None.")
        target_device = self.device if self.device is not None else get_default_device()
        backbone = backbone.to(target_device).requires_grad_(False).eval()
        self.device_ = next(backbone.parameters()).device
        self.model_ = ContextResamplingUncertainty(
            backbone,
            num_subsets=self.num_subsets,
            fractions=tuple(self.fractions),
            seed=self.subset_seed,
        )
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = _dense(self.feature_preprocessor_.transform(X))
        support_x, support_y = self._context_tensors()
        probabilities: list[np.ndarray] = []
        diagnostics: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(query), self.query_chunk_size):
            chunk = torch.as_tensor(query[start : start + self.query_chunk_size], device=self.device_).unsqueeze(0)
            prediction = self.model_(support_x, support_y, chunk, num_mem_chunks=self.num_mem_chunks)
            probabilities.append(prediction.marginal_probabilities()[0].cpu().numpy())
            for key, value in _diagnostics_from_prediction(prediction).items():
                diagnostics.setdefault(key, []).append(value)
        self.last_diagnostics_ = {key: np.concatenate(values) for key, values in diagnostics.items()}
        return np.concatenate(probabilities)
