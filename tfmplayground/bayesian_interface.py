"""Scikit-learn interface for the static Bayesian nanoTabPFN model."""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from tfmplayground.interface import NanoTabPFNClassifier, get_feature_preprocessor, init_model_from_state_dict_file
from tfmplayground.models.hypothesis import (
    NanoTabPFNBayesianModel,
    NanoTabPFNHypothesisModel,
    load_bayesian_checkpoint,
)
from tfmplayground.utils import get_default_device


def _dense(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


class BayesianNanoTabPFNClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic batch classifier backed by a Bayesian hypothesis head.

    ``fit`` only stores the labelled context.  ``predict_proba`` passes the
    untouched query features with labels omitted, so query labels cannot
    influence model construction or posterior weights.
    """

    def __init__(
        self,
        model: NanoTabPFNBayesianModel | str | os.PathLike | None = None,
        *,
        num_hypotheses: int = 2,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
    ):
        self.model = model
        self.num_hypotheses = num_hypotheses
        self.context_size = context_size
        self.random_state = random_state
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X, y):
        if self.num_hypotheses < 2:
            raise ValueError("num_hypotheses must be at least two.")
        if len(X) != len(y) or len(y) < 2:
            raise ValueError("X and y must contain the same at least two rows.")
        self.label_encoder_ = LabelEncoder().fit(np.asarray(y))
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("BayesianNanoTabPFNClassifier currently supports binary classification only.")
        self.classes_ = self.label_encoder_.classes_
        self.feature_preprocessor_ = get_feature_preprocessor(X)
        self.X_train_ = _dense(self.feature_preprocessor_.fit_transform(X))
        self.y_train_ = self.label_encoder_.transform(np.asarray(y)).astype(np.float32)
        if self.context_size < 2:
            raise ValueError("context_size must be at least two.")
        if isinstance(self.model, NanoTabPFNHypothesisModel):
            model = self.model
        elif isinstance(self.model, (str, os.PathLike)):
            model, _ = load_bayesian_checkpoint(self.model)
        elif self.model is None:
            backbone = NanoTabPFNClassifier(device="cpu").model
            model = NanoTabPFNBayesianModel(backbone, num_hypotheses=self.num_hypotheses)
        else:
            raise TypeError("model must be a Bayesian model, checkpoint path, or None.")
        if model.num_hypotheses != self.num_hypotheses:
            raise ValueError("num_hypotheses does not match the supplied Bayesian checkpoint.")
        self.model_ = model.to(self.device if self.device is not None else get_default_device()).eval()
        self.device_ = next(self.model_.parameters()).device
        return self

    def _context_indices(self) -> np.ndarray:
        count = min(self.context_size, len(self.y_train_))
        rng = np.random.default_rng(self.random_state)
        if count == len(self.y_train_):
            return rng.permutation(len(self.y_train_))
        # Deterministic random subset; stratification is unnecessary for the
        # model and can fail on tiny binary contexts.
        return rng.choice(len(self.y_train_), size=count, replace=False)

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = _dense(self.feature_preprocessor_.transform(X))
        indices = self._context_indices()
        support_x = torch.as_tensor(self.X_train_[indices], device=self.device_).unsqueeze(0)
        support_y = torch.as_tensor(self.y_train_[indices], device=self.device_).unsqueeze(0)
        query_x = torch.as_tensor(query, device=self.device_).unsqueeze(0)
        prediction = self.model_(support_x, support_y, query_x, num_mem_chunks=self.num_mem_chunks)
        probabilities = prediction.marginal_probabilities()[0].cpu().numpy()
        self.last_diagnostics_ = {
            "predictive_entropy": prediction.predictive_entropy()[0].cpu().numpy(),
            "mutual_information": prediction.mutual_information()[0].cpu().numpy(),
            "hypothesis_disagreement": prediction.hypothesis_disagreement()[0].cpu().numpy(),
            "epistemic_variance": prediction.epistemic_variance()[0].cpu().numpy(),
            "posterior_entropy": prediction.posterior_entropy()[0].cpu().numpy(),
            "effective_hypothesis_count": prediction.effective_hypothesis_count()[0].cpu().numpy(),
            "effective_sample_size": prediction.effective_sample_size()[0].cpu().numpy(),
            "posterior_weights": prediction.posterior_weights[0].cpu().numpy(),
        }
        if hasattr(prediction, "base_probabilities"):
            self.last_diagnostics_["mean_preservation_error"] = prediction.mean_preservation_error()[0].cpu().numpy()
        return probabilities

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class VanillaNanoTabPFNClassifier(ClassifierMixin, BaseEstimator):
    """Vanilla nanoTabPFN baseline with the same preprocessing/context protocol."""

    def __init__(
        self,
        model=None,
        *,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
    ):
        self.model = model
        self.context_size = context_size
        self.random_state = random_state
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X, y):
        if len(X) != len(y) or len(y) < 2:
            raise ValueError("X and y must contain the same at least two rows.")
        self.label_encoder_ = LabelEncoder().fit(np.asarray(y))
        if len(self.label_encoder_.classes_) != 2:
            raise ValueError("Vanilla comparison currently supports binary classification only.")
        self.classes_ = self.label_encoder_.classes_
        self.feature_preprocessor_ = get_feature_preprocessor(X)
        self.X_train_ = _dense(self.feature_preprocessor_.fit_transform(X))
        self.y_train_ = self.label_encoder_.transform(np.asarray(y)).astype(np.float32)
        if isinstance(self.model, torch.nn.Module):
            model = self.model
        elif isinstance(self.model, (str, os.PathLike)):
            model = init_model_from_state_dict_file(str(self.model))
        elif self.model is None:
            model = NanoTabPFNClassifier(device="cpu").model
        else:
            raise TypeError("model must be a nanoTabPFN module, checkpoint path, or None.")
        # Match the mean-preserving model's frozen-backbone execution path.
        # PyTorch may otherwise select a slightly different attention kernel
        # based on requires_grad and differ by a few floating-point ulps.
        target_device = self.device if self.device is not None else get_default_device()
        self.model_ = model.to(target_device).requires_grad_(False).eval()
        self.device_ = next(self.model_.parameters()).device
        return self

    def _context_indices(self) -> np.ndarray:
        count = min(self.context_size, len(self.y_train_))
        rng = np.random.default_rng(self.random_state)
        if count == len(self.y_train_):
            return rng.permutation(len(self.y_train_))
        return rng.choice(len(self.y_train_), size=count, replace=False)

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = _dense(self.feature_preprocessor_.transform(X))
        indices = self._context_indices()
        support_x = torch.as_tensor(self.X_train_[indices], device=self.device_).unsqueeze(0)
        support_y = torch.as_tensor(self.y_train_[indices], device=self.device_).unsqueeze(0)
        query_x = torch.as_tensor(query, device=self.device_).unsqueeze(0)
        logits = self.model_(support_x, support_y, query_x, num_mem_chunks=self.num_mem_chunks)[..., :2]
        return logits.softmax(-1)[0].cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
