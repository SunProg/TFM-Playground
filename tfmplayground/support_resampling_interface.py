"""Scikit-learn interfaces for the support-resampling and intrinsic-posterior arms.

Mirrors ``continuous_interface.py``: ``fit`` only stores the labelled context,
and ``predict_proba`` returns the frozen vanilla probabilities with uncertainty
recorded as a side channel in ``last_diagnostics_``, so query labels can never
influence model construction or the reported dispersion.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from tfmplayground.continuous_interface import _ContextClassifierBase
from tfmplayground.experiments.support_resampling import build_ensemble, compute_intrinsic_posterior
from tfmplayground.interface import NanoTabPFNClassifier, init_model_from_state_dict_file
from tfmplayground.utils import get_default_device


class SupportResamplingClassifier(_ContextClassifierBase):
    """Deployed classifier whose uncertainty comes from resampling the labelled context.

    The returned probabilities are the frozen vanilla ones (the full context,
    unresampled); resampled members are used only to populate
    ``last_diagnostics_``.
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
        scheme: str = "bootstrap",
        members: int = 32,
        fraction: float = 0.8,
        ensemble_seed: int = 0,
    ):
        super().__init__(
            model,
            context_size=context_size,
            random_state=random_state,
            device=device,
            num_mem_chunks=num_mem_chunks,
            query_chunk_size=query_chunk_size,
        )
        self.scheme = scheme
        self.members = members
        self.fraction = fraction
        self.ensemble_seed = ensemble_seed

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
        self.model_ = backbone.to(target_device).requires_grad_(False).eval()
        self.device_ = next(self.model_.parameters()).device
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = np.asarray(self.feature_preprocessor_.transform(X), dtype=np.float32)
        if hasattr(query, "toarray"):
            query = query.toarray()
        support_x, support_y = self._context_tensors()
        support_x, support_y = support_x[0], support_y[0]
        # The full labelled context is resampled directly here -- unlike the
        # synthetic stage-1 evaluation, deployment has no separate held-out pool
        # to score members against, and the base-point pass must see the same
        # full context the vanilla arm does, so the deployed probability stays
        # exactly the vanilla one (see build_ensemble's base-point pass).
        probabilities: list[np.ndarray] = []
        diagnostics: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(query), self.query_chunk_size):
            chunk = torch.as_tensor(query[start : start + self.query_chunk_size], device=self.device_)
            ensemble = build_ensemble(
                self.model_,
                support_x,
                support_y,
                chunk,
                scheme=self.scheme,
                members=self.members,
                fraction=self.fraction,
                seed=self.ensemble_seed,
                num_mem_chunks=self.num_mem_chunks,
                compute_gradient=False,
            )
            positive = ensemble.base_positive.cpu().numpy()
            probabilities.append(np.stack((1.0 - positive, positive), axis=-1))
            diagnostics.setdefault("mutual_information", []).append(
                ensemble.probability_mutual_information().cpu().numpy()
            )
            diagnostics.setdefault("epistemic_variance", []).append(
                ensemble.probability_dispersion().cpu().numpy()
            )
        self.last_diagnostics_ = {key: np.concatenate(values) for key, values in diagnostics.items()}
        return np.concatenate(probabilities)


class IntrinsicPosteriorClassifier(_ContextClassifierBase):
    """Deployed classifier whose uncertainty is the model's own joint posterior; no resampling.

    :func:`compute_intrinsic_posterior` batches ``2 * query_chunk_size`` members
    of ``context_size + 1`` rows each into one forward pass, so
    ``query_chunk_size`` needs to be much smaller here than for the resampling
    or vanilla arms at TabArena's default ``context_size=1024`` -- the default
    below is deliberately conservative.
    """

    def __init__(
        self,
        model=None,
        *,
        context_size: int = 1024,
        random_state: int = 0,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        query_chunk_size: int = 64,
        score: str = "joint_mi",
    ):
        super().__init__(
            model,
            context_size=context_size,
            random_state=random_state,
            device=device,
            num_mem_chunks=num_mem_chunks,
            query_chunk_size=query_chunk_size,
        )
        if score not in ("joint_mi", "self_conditioning"):
            raise ValueError("score must be 'joint_mi' or 'self_conditioning'.")
        self.score = score

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
        self.model_ = backbone.to(target_device).requires_grad_(False).eval()
        self.device_ = next(self.model_.parameters()).device
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ("model_", "X_train_", "classes_"))
        query = np.asarray(self.feature_preprocessor_.transform(X), dtype=np.float32)
        if hasattr(query, "toarray"):
            query = query.toarray()
        support_x, support_y = self._context_tensors()
        support_x, support_y = support_x[0], support_y[0]
        probabilities: list[np.ndarray] = []
        diagnostics: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(query), self.query_chunk_size):
            chunk = torch.as_tensor(query[start : start + self.query_chunk_size], device=self.device_)
            posterior = compute_intrinsic_posterior(
                self.model_, support_x, support_y, chunk, num_mem_chunks=self.num_mem_chunks
            )
            positive = posterior.base_positive.cpu().numpy()
            probabilities.append(np.stack((1.0 - positive, positive), axis=-1))
            if self.score == "joint_mi":
                mutual_information = posterior.joint_mutual_information()
                count = mutual_information.shape[0]
                mask = ~torch.eye(count, dtype=torch.bool)
                per_query = torch.nan_to_num(mutual_information, nan=0.0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                per_query = posterior.self_conditioning()
            diagnostics.setdefault("mutual_information", []).append(per_query.cpu().numpy())
        self.last_diagnostics_ = {key: np.concatenate(values) for key, values in diagnostics.items()}
        return np.concatenate(probabilities)
