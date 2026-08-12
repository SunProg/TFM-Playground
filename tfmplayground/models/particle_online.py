"""Causal online adapter for :class:`AdaptiveKParticleFilter`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from tfmplayground.models.adaptive_particle_filter import AdaptiveKParticleFilter
from tfmplayground.models.batch_particle_filter import BatchCausalParticleFilter, PendingBatchUpdate
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class AdaptiveParticleOnlineClassifier:
    """Expose the adaptive particle model through predict-then-update batches.

    The first revealed rows become the fixed initial support. Later labels are
    retained only as filter evidence. Thus a batch's labels cannot enter the
    model invocation that predicted that batch.
    """

    def __init__(
        self,
        model: AdaptiveKParticleFilter,
        *,
        device: str | torch.device = "cpu",
        initial_support_limit: int = 128,
        particle_to_regime: Mapping[int, int] | None = None,
        num_mem_chunks: int = 1,
    ):
        if initial_support_limit < 2:
            raise ValueError("initial_support_limit must be at least two.")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.initial_support_limit = initial_support_limit
        self.particle_to_regime = dict(particle_to_regime or {})
        particle_count = model.particle_model.num_hypotheses
        if any(index < 0 or index >= particle_count for index in self.particle_to_regime):
            raise ValueError("particle_to_regime contains an invalid particle index.")
        self.num_mem_chunks = num_mem_chunks
        self._support_x: np.ndarray | None = None
        self._support_y = np.empty(0, dtype=np.int64)
        self._support_initialized = False
        self._stream_x: np.ndarray | None = None
        self._stream_y = np.empty(0, dtype=np.int64)
        self._diagnostics: dict[str, Any] = {}

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, x: np.ndarray, *, regime_hint: int | None = None) -> np.ndarray:
        del regime_hint
        query = self._clean(x)
        if self._support_x is None or self._support_y.size < 2 or np.unique(self._support_y).size < 2:
            positive = float(self._support_y.mean()) if self._support_y.size else 0.5
            self._diagnostics = {}
            return np.column_stack((np.full(len(query), 1 - positive), np.full(len(query), positive)))

        stream_x = np.empty((0, query.shape[1]), dtype=np.float32) if self._stream_x is None else self._stream_x
        with torch.no_grad():
            prediction = self.model(
                torch.as_tensor(self._support_x, device=self.device)[None],
                torch.as_tensor(self._support_y, device=self.device)[None].float(),
                torch.as_tensor(stream_x, device=self.device)[None],
                torch.as_tensor(self._stream_y, device=self.device)[None],
                torch.as_tensor(query, device=self.device)[None],
                num_mem_chunks=self.num_mem_chunks,
            )
            if self.particle_to_regime:
                matched = torch.tensor(
                    [sorted(self.particle_to_regime)],
                    device=self.device,
                    dtype=torch.long,
                )
                probability = prediction.matched_marginal_probabilities(matched)[0, -1].cpu().numpy()
            else:
                probability = prediction.marginal_probabilities()[0, -1].cpu().numpy()
            weights = prediction.log_weights[0, -1].exp().cpu().numpy()
        particle = int(weights.argmax())
        self._diagnostics = {
            "particle_weights": weights,
            "active_particle": particle,
            "predicted_regime": self.particle_to_regime.get(particle),
        }
        return probability

    def update(self, x: np.ndarray, y: np.ndarray, *, regime: int | None = None) -> None:
        del regime
        x = self._clean(x)
        y = np.asarray(y, dtype=np.int64)
        if y.shape != (len(x),):
            raise ValueError("y must align with x rows.")
        support_count = min(self.initial_support_limit, len(y)) if not self._support_initialized else 0
        if support_count:
            values = x[:support_count]
            self._support_x = values.copy() if self._support_x is None else np.concatenate((self._support_x, values))
            self._support_y = np.concatenate((self._support_y, y[:support_count]))
            self._support_initialized = True
        if support_count < len(y):
            values = x[support_count:]
            self._stream_x = values.copy() if self._stream_x is None else np.concatenate((self._stream_x, values))
            self._stream_y = np.concatenate((self._stream_y, y[support_count:]))

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


class BatchParticleOnlineClassifier:
    """NumPy adapter preserving the model's explicit predict/reveal boundary."""

    def __init__(
        self,
        model: BatchCausalParticleFilter,
        *,
        device: str | torch.device = "cpu",
        temporal: bool = True,
        particle_to_regime: Mapping[int, int] | None = None,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.temporal = temporal
        self.particle_to_regime = dict(particle_to_regime or {})
        self._state = None
        self._pending: PendingBatchUpdate | None = None
        self._diagnostics: dict[str, Any] = {}

    def predict_proba(self, x: np.ndarray, *, regime_hint: int | None = None) -> np.ndarray:
        del regime_hint
        if self._pending is not None:
            raise RuntimeError("The preceding batch must be revealed before another prediction.")
        clean = torch.as_tensor(
            np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            device=self.device,
        )[None]
        if self._state is None:
            self._state = self.model.initial_state(clean.shape[-1], device=self.device)
        with torch.no_grad():
            self._pending = self.model.predict_batch(self._state, clean)
        weights = self._pending.transitioned_log_weights[0].exp().cpu().numpy()
        particle = int(weights.argmax())
        self._diagnostics = {
            "particle_weights": weights,
            "active_particle": particle,
            "predicted_regime": self.particle_to_regime.get(particle),
            "ambiguity_probability": float(self._pending.ambiguity_probability[0]),
            "previous_surprise": float(self._state.previous_surprise[0]),
        }
        return self._pending.probabilities[0].cpu().numpy()

    def update(self, x: np.ndarray, y: np.ndarray, *, regime: int | None = None) -> None:
        del regime
        clean = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.as_tensor(np.asarray(y, dtype=np.int64), device=self.device)[None]
        if self._pending is None:
            values = torch.as_tensor(clean, device=self.device)[None]
            if self._state is None:
                self._state = self.model.initial_state(values.shape[-1], device=self.device)
            context_x = torch.cat((self._state.context_x, values), dim=1)
            context_y = torch.cat((self._state.context_y, labels), dim=1)
            context_x, context_y = self.model._capped_context(context_x, context_y, temporal=self.temporal)
            self._state = replace(self._state, context_x=context_x, context_y=context_y)
            return
        if clean.shape != tuple(self._pending.x.shape[1:]):
            raise ValueError("update x must be the batch committed by predict_proba.")
        self._state = self.model.reveal_batch(self._pending, labels, temporal=self.temporal)
        self._pending = None

    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics


class NanoTabPFNContextOnlineClassifier:
    """Frozen nanoTabPFN with a bounded cumulative or recent labelled context."""

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        device: str | torch.device = "cpu",
        context_limit: int = 1024,
        temporal: bool = True,
    ):
        self.backbone = backbone.to(device).requires_grad_(False).eval()
        self.device = torch.device(device)
        self.context_limit = context_limit
        self.temporal = temporal
        self.x: np.ndarray | None = None
        self.y = np.empty(0, dtype=np.int64)

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, x: np.ndarray, *, regime_hint: int | None = None) -> np.ndarray:
        del regime_hint
        query = self._clean(x)
        if self.x is None or self.y.size == 0:
            return np.full((len(query), 2), 0.5)
        with torch.no_grad():
            support = torch.as_tensor(self.x, device=self.device)[None]
            target = torch.as_tensor(query, device=self.device)[None]
            labels = torch.as_tensor(self.y, device=self.device)[None].float()
            logits = self.backbone(
                (torch.cat((support, target), dim=1), labels), train_test_split_index=len(self.y)
            )[..., :2]
        return logits[0].softmax(-1).cpu().numpy()

    def update(self, x: np.ndarray, y: np.ndarray, *, regime: int | None = None) -> None:
        del regime
        values = self._clean(x)
        labels = np.asarray(y, dtype=np.int64)
        self.x = values.copy() if self.x is None else np.concatenate((self.x, values))
        self.y = np.concatenate((self.y, labels))
        if self.y.size > self.context_limit:
            if self.temporal:
                indices = np.arange(self.y.size - self.context_limit, self.y.size)
            else:
                pieces = []
                for label in (0, 1):
                    candidates = np.flatnonzero(self.y == label)
                    count = min(self.context_limit // 2, len(candidates))
                    if count:
                        pieces.append(candidates[np.linspace(0, len(candidates) - 1, count).round().astype(int)])
                indices = np.sort(np.concatenate(pieces))
                if len(indices) < self.context_limit:
                    remaining = np.setdiff1d(np.arange(self.y.size), indices, assume_unique=True)
                    indices = np.sort(np.concatenate((indices, remaining[: self.context_limit - len(indices)])))
            self.x, self.y = self.x[indices], self.y[indices]

    def diagnostics(self) -> Mapping[str, Any]:
        return {}
