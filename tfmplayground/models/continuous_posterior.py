"""Slot-free continuous Bayesian uncertainty for a frozen nanoTabPFN mean.

Two independent paths are used.  A frozen vanilla ``NanoTabPFNModel`` supplies
the predictive mean ``mu_q``; a second copy of the same checkpoint supplies
representations for uncertainty only.  The uncertainty path never touches the
mean path, so class probabilities are bit-for-bit the vanilla ones.

The posterior is an anonymous, continuous function-space posterior.  There are
no persistent hypothesis slots, no candidate identities, no posterior slot
weights, no Hungarian matching, and no configurable hypothesis count ``K``.
``S`` is a Monte-Carlo sampling count only: sample ``s`` has no meaning that
persists between episodes.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer

CONTINUOUS_CHECKPOINT_TYPE = "nanotabpfn_continuous_posterior"
BETA_CHECKPOINT_TYPE = "nanotabpfn_beta_concentration"
CHECKPOINT_FORMAT_VERSION = 1
UNCERTAINTY_MODES = ("frozen", "adapters", "full")
#: Probability margin kept away from zero and one so that entropies stay finite.
PROBABILITY_MARGIN = 1e-6


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter ``h + W_up(GELU(W_down(LayerNorm(h))))``.

    ``W_up`` is zero-initialized, so a freshly created adapter is exactly the
    identity and an adapter-equipped encoder starts from the pretrained
    representation.
    """

    def __init__(self, embedding_size: int, bottleneck: int):
        super().__init__()
        if bottleneck < 1:
            raise ValueError("adapter bottleneck must be positive.")
        self.norm = nn.LayerNorm(embedding_size)
        self.down = nn.Linear(embedding_size, bottleneck)
        self.up = nn.Linear(bottleneck, embedding_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(F.gelu(self.down(self.norm(hidden))))


class AdaptedTransformerEncoderLayer(TransformerEncoderLayer):
    """Pretrained transformer layer with three residual adapters.

    One adapter follows each of the feature-attention, datapoint-attention, and
    feed-forward stages.  The pretrained parameters keep their original state
    dictionary names, so the vanilla checkpoint loads unchanged.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        adapter_bottleneck: int = 32,
        **kwargs: Any,
    ):
        super().__init__(embedding_size, nhead, mlp_hidden_size, **kwargs)
        self.adapter_bottleneck = adapter_bottleneck
        self.feature_adapter = BottleneckAdapter(embedding_size, adapter_bottleneck)
        self.datapoint_adapter = BottleneckAdapter(embedding_size, adapter_bottleneck)
        self.mlp_adapter = BottleneckAdapter(embedding_size, adapter_bottleneck)

    @classmethod
    def from_pretrained(cls, layer: TransformerEncoderLayer, adapter_bottleneck: int) -> AdaptedTransformerEncoderLayer:
        embedding_size = layer.norm1.normalized_shape[0]
        adapted = cls(
            embedding_size,
            layer.self_attention_between_features.num_heads,
            layer.linear1.out_features,
            adapter_bottleneck=adapter_bottleneck,
        )
        missing, unexpected = adapted.load_state_dict(layer.state_dict(), strict=False)
        if unexpected:
            raise ValueError(f"Unexpected pretrained parameters for an adapted layer: {sorted(unexpected)}")
        if any(not name.startswith(("feature_adapter", "datapoint_adapter", "mlp_adapter")) for name in missing):
            raise ValueError("The adapted layer did not receive every pretrained parameter.")
        return adapted

    def adapt_after_feature_attention(self, src: torch.Tensor) -> torch.Tensor:
        return self.feature_adapter(src)

    def adapt_after_datapoint_attention(self, src: torch.Tensor) -> torch.Tensor:
        return self.datapoint_adapter(src)

    def adapt_after_mlp(self, src: torch.Tensor) -> torch.Tensor:
        return self.mlp_adapter(src)

    def adapter_parameters(self):
        for module in (self.feature_adapter, self.datapoint_adapter, self.mlp_adapter):
            yield from module.parameters()


def install_adapters(backbone: NanoTabPFNModel, adapter_bottleneck: int) -> NanoTabPFNModel:
    """Replace every transformer layer in place with an adapter-equipped copy."""
    for index, layer in enumerate(backbone.transformer_blocks):
        if isinstance(layer, AdaptedTransformerEncoderLayer):
            continue
        backbone.transformer_blocks[index] = AdaptedTransformerEncoderLayer.from_pretrained(layer, adapter_bottleneck)
    return backbone


def adapter_parameters(backbone: NanoTabPFNModel):
    for layer in backbone.transformer_blocks:
        if isinstance(layer, AdaptedTransformerEncoderLayer):
            yield from layer.adapter_parameters()


# --------------------------------------------------------------------------- #
# Deterministic quasi-random noise
# --------------------------------------------------------------------------- #
def sobol_standard_normal(
    num_samples: int,
    dimension: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Antithetic scrambled-Sobol draws mapped to standard-normal values.

    ``num_samples`` must be even: the second half of the returned tensor is the
    negation of the first half, so the sample set is exactly symmetric and its
    empirical mean is zero up to floating point.
    """
    if num_samples < 2 or num_samples % 2 != 0:
        raise ValueError("num_samples must be an even number of at least two.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=int(seed))
    uniform = engine.draw(num_samples // 2).to(dtype=torch.float64)
    uniform = uniform.clamp(1e-12, 1.0 - 1e-12)
    normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    return torch.cat((normal, -normal), dim=0).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    """``H(p) = -p log p - (1-p) log(1-p)`` in nats."""
    probability = probability.clamp(PROBABILITY_MARGIN, 1.0 - PROBABILITY_MARGIN)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())


@dataclass(frozen=True)
class ContinuousPosteriorPrediction:
    """Anonymous equal-weight posterior samples around the frozen vanilla mean.

    Attributes:
        base_probabilities: ``(batch, query, 2)`` vanilla nanoTabPFN class
            probabilities.  This is what is returned to users.
        sample_positive: ``(batch, sample, query)`` class-1 probability
            ``p_qs`` of anonymous posterior sample ``s``.
        dispersion_gate: ``(batch, query)`` learned gate ``g_q`` in ``(0, 1)``.
        dispersion_bound: ``(batch, query)`` largest safe scale ``b_q``.
    """

    base_probabilities: torch.Tensor
    sample_positive: torch.Tensor
    dispersion_gate: torch.Tensor
    dispersion_bound: torch.Tensor

    @property
    def base_positive(self) -> torch.Tensor:
        return self.base_probabilities[..., 1]

    @property
    def num_samples(self) -> int:
        return self.sample_positive.shape[1]

    def marginal_probabilities(self) -> torch.Tensor:
        """The deployed prediction: always the vanilla probabilities."""
        return self.base_probabilities

    def sample_probabilities(self) -> torch.Tensor:
        """``(batch, sample, query, 2)`` two-column posterior sample probabilities."""
        positive = self.sample_positive
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predictive_entropy(self) -> torch.Tensor:
        return binary_entropy(self.base_positive)

    def expected_conditional_entropy(self) -> torch.Tensor:
        return binary_entropy(self.sample_positive).mean(dim=1)

    def mutual_information(self) -> torch.Tensor:
        return (self.predictive_entropy() - self.expected_conditional_entropy()).clamp_min(0.0)

    def epistemic_variance(self) -> torch.Tensor:
        return (self.sample_positive - self.base_positive[:, None, :]).square().mean(dim=1)

    def epistemic_covariance(self) -> torch.Tensor:
        """``(batch, query, query)`` covariance of ``p_qs`` across samples."""
        deviation = self.sample_positive - self.base_positive[:, None, :]
        return torch.einsum("bsq,bsr->bqr", deviation, deviation) / deviation.shape[1]

    def mean_preservation_error(self) -> torch.Tensor:
        """``(batch,)`` maximum ``|average_s p_qs - mu_q|`` over queries."""
        return (self.sample_positive.mean(dim=1) - self.base_positive).abs().amax(dim=-1)

    def joint_log_probabilities(self, outcomes: torch.Tensor | None = None) -> torch.Tensor:
        """Coherent joint probability of complete binary query vectors."""
        num_samples, query_count = self.sample_positive.shape[1], self.sample_positive.shape[2]
        if outcomes is None:
            outcomes = all_binary_outcomes(query_count, device=self.sample_positive.device)
        outcomes = outcomes.to(device=self.sample_positive.device, dtype=self.sample_positive.dtype)
        if outcomes.ndim != 2 or outcomes.shape[1] != query_count:
            raise ValueError(f"outcomes must have shape (n, {query_count}), found {tuple(outcomes.shape)}.")
        positive = self.sample_positive.clamp(PROBABILITY_MARGIN, 1.0 - PROBABILITY_MARGIN)
        # (batch, outcome, sample, query)
        selected = outcomes[None, :, None, :] * positive[:, None] + (1.0 - outcomes[None, :, None, :]) * (
            1.0 - positive[:, None]
        )
        per_sample = selected.log().sum(dim=-1)
        return torch.logsumexp(per_sample, dim=-1) - math.log(num_samples)

    def joint_probabilities(self, outcomes: torch.Tensor | None = None) -> torch.Tensor:
        return self.joint_log_probabilities(outcomes).exp()

    def summary(self) -> dict[str, torch.Tensor]:
        return {
            "vanilla_probabilities": self.base_probabilities,
            "sample_probabilities": self.sample_probabilities(),
            "predictive_entropy": self.predictive_entropy(),
            "expected_conditional_entropy": self.expected_conditional_entropy(),
            "mutual_information": self.mutual_information(),
            "epistemic_variance": self.epistemic_variance(),
            "epistemic_covariance": self.epistemic_covariance(),
            "max_mean_preservation_error": self.mean_preservation_error(),
        }


@dataclass(frozen=True)
class BetaConcentrationPrediction(ContinuousPosteriorPrediction):
    """Mean-constrained Beta ablation with one concentration per query.

    ``concentration`` is ``kappa_q``; the Beta shape parameters are
    ``alpha_q = mu_q * kappa_q`` and ``beta_q = (1 - mu_q) * kappa_q``, so the
    distribution mean is exactly ``mu_q``.  Unlike the continuous posterior the
    sample mean is only correct in expectation, so ``mean_preservation_error``
    is a Monte-Carlo quantity here; the deployed probability is still exactly
    ``mu_q``.
    """

    concentration: torch.Tensor

    def analytic_epistemic_variance(self) -> torch.Tensor:
        mean = self.base_positive
        return mean * (1.0 - mean) / (self.concentration + 1.0)


def all_binary_outcomes(query_count: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Every binary label vector of length ``query_count`` in canonical order."""
    if query_count < 1 or query_count > 20:
        raise ValueError("query_count must lie between one and twenty for exact enumeration.")
    indices = torch.arange(2**query_count, device=device)
    shifts = torch.arange(query_count - 1, -1, -1, device=device)
    return ((indices[:, None] >> shifts[None, :]) & 1).to(torch.long)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class DeepSetsSupportEncoder(nn.Module):
    """Permutation-invariant support pooling using the mean and variance."""

    def __init__(self, embedding_size: int, hidden_size: int, context_size: int):
        super().__init__()
        self.element = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_size),
        )
        self.pool = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, context_size),
        )

    def forward(self, support_embeddings: torch.Tensor) -> torch.Tensor:
        features = self.element(support_embeddings)
        mean = features.mean(dim=1)
        variance = features.var(dim=1, unbiased=False)
        return self.pool(torch.cat((mean, variance), dim=-1))


class LatentGenerator(nn.Module):
    """Conditional residual MLP mapping ``(c_D, epsilon_s)`` to ``z_s``."""

    def __init__(self, context_size: int, latent_dim: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(context_size + latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, context: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # context: (batch, context); noise: (sample, latent)
        batch_size, num_samples = context.shape[0], noise.shape[0]
        expanded_context = context[:, None].expand(-1, num_samples, -1)
        expanded_noise = noise[None].expand(batch_size, -1, -1)
        return expanded_noise + self.body(torch.cat((expanded_context, expanded_noise), dim=-1))


class QueryDeviationDecoder(nn.Module):
    """Raw deviation ``r_qs`` for query ``q`` under anonymous latent draw ``z_s``."""

    def __init__(self, embedding_size: int, context_size: int, latent_dim: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(embedding_size + context_size + latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, query_embeddings: torch.Tensor, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # query_embeddings: (batch, query, embedding); latents: (batch, sample, latent)
        query_count = query_embeddings.shape[1]
        num_samples = latents.shape[1]
        query = query_embeddings[:, None].expand(-1, num_samples, -1, -1)
        context_expanded = context[:, None, None].expand(-1, num_samples, query_count, -1)
        latent = latents[:, :, None].expand(-1, -1, query_count, -1)
        return self.body(torch.cat((query, context_expanded, latent), dim=-1)).squeeze(-1)


class DispersionGate(nn.Module):
    """Learned per-query gate ``g_q`` in ``(0, 1)``."""

    def __init__(self, embedding_size: int, context_size: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(embedding_size + context_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, query_embeddings: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_expanded = context[:, None].expand(-1, query_embeddings.shape[1], -1)
        return torch.sigmoid(self.body(torch.cat((query_embeddings, context_expanded), dim=-1)).squeeze(-1))


# --------------------------------------------------------------------------- #
# Mean preservation
# --------------------------------------------------------------------------- #
def safe_dispersion_bound(base_positive: torch.Tensor, deviations: torch.Tensor, *, sample_dim: int) -> torch.Tensor:
    """Largest scale keeping ``mu + scale * deviation`` inside ``(0, 1)``.

    Args:
        base_positive: ``(batch, query)`` vanilla class-1 probability ``mu_q``.
        deviations: centred deviations with a sample axis at ``sample_dim``.
    """
    positive_extent = deviations.clamp_min(0.0).amax(dim=sample_dim)
    negative_extent = (-deviations).clamp_min(0.0).amax(dim=sample_dim)
    headroom_up = (1.0 - PROBABILITY_MARGIN - base_positive).clamp_min(0.0)
    headroom_down = (base_positive - PROBABILITY_MARGIN).clamp_min(0.0)
    up = headroom_up / positive_extent.clamp_min(PROBABILITY_MARGIN)
    down = headroom_down / negative_extent.clamp_min(PROBABILITY_MARGIN)
    return torch.minimum(up, down)


def centre_and_scale(
    base_positive: torch.Tensor,
    raw: torch.Tensor,
    gate: torch.Tensor,
    *,
    max_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre raw deviations across samples and rescale them safely.

    Returns the sample probabilities ``p_qs`` and the bound ``b_q``.  The
    sample average of ``p_qs`` equals ``mu_q`` by construction because the
    centred deviations sum to zero and the scale does not depend on ``s``.
    """
    centred = raw - raw.mean(dim=1, keepdim=True)
    bound = safe_dispersion_bound(base_positive, centred, sample_dim=1)
    if max_scale is not None:
        bound = bound.clamp(max=max_scale)
    scale = gate * bound
    return base_positive[:, None, :] + scale[:, None, :] * centred, bound


def project_candidate_posterior(
    base_positive: torch.Tensor,
    candidate_positive: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project an exact candidate posterior onto the fixed vanilla mean.

    Args:
        base_positive: ``(batch, query)`` frozen vanilla mean ``mu_q``.
        candidate_positive: ``(batch, candidate, query)`` candidate class-1
            probabilities ``theta_qh``.
        candidate_weights: ``(batch, candidate)`` exact posterior weights
            ``rho_h``.

    Returns:
        The projected probabilities and the per-query scale ``safe_scale_q``.
        Candidate weights are preserved exactly and the weighted mean of the
        projected probabilities equals ``mu_q``.
    """
    teacher_mean = (candidate_weights[:, :, None] * candidate_positive).sum(dim=1)
    deviation = candidate_positive - teacher_mean[:, None, :]
    scale = safe_dispersion_bound(base_positive, deviation, sample_dim=1).clamp(max=1.0)
    projected = base_positive[:, None, :] + scale[:, None, :] * deviation
    return projected, scale


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class _MeanPreservingUncertaintyModel(nn.Module):
    """Shared frozen-mean / trainable-uncertainty two-path scaffolding."""

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        uncertainty_mode: str = "adapters",
        adapter_bottleneck: int = 32,
        hidden_size: int | None = None,
        context_size: int | None = None,
        uncertainty_backbone: NanoTabPFNModel | None = None,
    ):
        super().__init__()
        if uncertainty_mode not in UNCERTAINTY_MODES:
            raise ValueError(f"uncertainty_mode must be one of {UNCERTAINTY_MODES}.")
        if backbone.num_outputs < 2:
            raise ValueError("The binary experiment requires a backbone with at least two outputs.")
        self.uncertainty_mode = uncertainty_mode
        self.adapter_bottleneck = adapter_bottleneck if uncertainty_mode == "adapters" else None
        self.mean_backbone = backbone
        uncertainty = copy.deepcopy(backbone) if uncertainty_backbone is None else uncertainty_backbone
        if uncertainty_mode == "adapters":
            install_adapters(uncertainty, adapter_bottleneck)
        self.uncertainty_backbone = uncertainty
        self.embedding_size = backbone.embedding_size
        self.hidden_size = backbone.mlp_hidden_size if hidden_size is None else hidden_size
        self.context_size = backbone.embedding_size if context_size is None else context_size
        self.support_encoder = DeepSetsSupportEncoder(self.embedding_size, self.hidden_size, self.context_size)
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        # The mean path is frozen for the lifetime of the model.
        self.mean_backbone.requires_grad_(False)
        if self.uncertainty_mode == "frozen":
            self.uncertainty_backbone.requires_grad_(False)
        elif self.uncertainty_mode == "adapters":
            self.uncertainty_backbone.requires_grad_(False)
            for parameter in adapter_parameters(self.uncertainty_backbone):
                parameter.requires_grad_(True)
        else:
            self.uncertainty_backbone.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _split_arguments(self, args: tuple, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if len(args) == 3:
            support_x, support_y, query_x = args
        elif len(args) == 1 and isinstance(args[0], tuple):
            full_x, support_y = args[0]
            split = kwargs.pop("train_test_split_index", None)
            if split is None:
                raise TypeError("train_test_split_index is required for the concatenated-table interface.")
            support_x, query_x = full_x[:, :split], full_x[:, split:]
        else:
            raise TypeError("Expected (x_train, y_train, x_query) or ((x, y), train_test_split_index=...).")
        return support_x, support_y, query_x, kwargs

    @torch.no_grad()
    def _frozen_mean(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int,
    ) -> torch.Tensor:
        logits = self.mean_backbone(support_x, support_y, query_x, num_mem_chunks=num_mem_chunks)[..., :2]
        return logits.softmax(dim=-1)

    def _uncertainty_representations(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        split = support_x.shape[1]
        source = (torch.cat((support_x, query_x), dim=1), support_y)
        if self.uncertainty_mode == "frozen":
            with torch.no_grad():
                encoded = self.uncertainty_backbone.encode_table(source, split, num_mem_chunks=num_mem_chunks)
            encoded = encoded.detach()
        else:
            encoded = self.uncertainty_backbone.encode_table(source, split, num_mem_chunks=num_mem_chunks)
        target_embeddings = encoded[:, :, -1, :]
        return target_embeddings[:, :split], target_embeddings[:, split:]

    def architecture(self) -> dict[str, Any]:
        backbone = self.mean_backbone
        return {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "uncertainty_mode": self.uncertainty_mode,
            "adapter_bottleneck": self.adapter_bottleneck,
            "hidden_size": self.hidden_size,
            "context_size": self.context_size,
        }


class NanoTabPFNContinuousPosteriorModel(_MeanPreservingUncertaintyModel):
    """Continuous anonymous function-space posterior with an exact vanilla mean.

    Latent draws ``z_s`` are shared by every query in an episode, so one draw
    represents one coherent possible prediction function over the whole query
    set.  Increasing ``num_samples`` reduces Monte-Carlo error only; it does
    not add learned hypotheses.
    """

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        uncertainty_mode: str = "adapters",
        adapter_bottleneck: int = 32,
        latent_dim: int = 32,
        num_samples: int = 32,
        inference_seed: int = 0,
        hidden_size: int | None = None,
        context_size: int | None = None,
        uncertainty_backbone: NanoTabPFNModel | None = None,
    ):
        super().__init__(
            backbone,
            uncertainty_mode=uncertainty_mode,
            adapter_bottleneck=adapter_bottleneck,
            hidden_size=hidden_size,
            context_size=context_size,
            uncertainty_backbone=uncertainty_backbone,
        )
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if num_samples < 2 or num_samples % 2 != 0:
            raise ValueError("num_samples must be an even number of at least two (antithetic pairs).")
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.inference_seed = inference_seed
        self.latent_generator = LatentGenerator(self.context_size, latent_dim, self.hidden_size)
        self.query_decoder = QueryDeviationDecoder(
            self.embedding_size, self.context_size, latent_dim, self.hidden_size
        )
        self.dispersion_gate = DispersionGate(self.embedding_size, self.context_size, self.hidden_size)
        self._apply_freezing()

    def forward(self, *args, **kwargs) -> ContinuousPosteriorPrediction:
        support_x, support_y, query_x, kwargs = self._split_arguments(args, kwargs)
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        num_samples = kwargs.pop("num_samples", None) or self.num_samples
        sample_seed = kwargs.pop("sample_seed", None)
        sample_seed = self.inference_seed if sample_seed is None else int(sample_seed)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        base_probabilities = self._frozen_mean(support_x, support_y, query_x, num_mem_chunks=num_mem_chunks)
        base_positive = base_probabilities[..., 1]
        support_embeddings, query_embeddings = self._uncertainty_representations(
            support_x, support_y, query_x, num_mem_chunks=num_mem_chunks
        )
        context = self.support_encoder(support_embeddings)
        noise = sobol_standard_normal(
            num_samples,
            self.latent_dim,
            seed=sample_seed,
            device=context.device,
            dtype=context.dtype,
        )
        latents = self.latent_generator(context, noise)
        raw = self.query_decoder(query_embeddings, context, latents)
        gate = self.dispersion_gate(query_embeddings, context)
        sample_positive, bound = centre_and_scale(base_positive, raw, gate)
        return ContinuousPosteriorPrediction(base_probabilities, sample_positive, gate, bound)


class NanoTabPFNBetaConcentrationModel(_MeanPreservingUncertaintyModel):
    """Deliberately low-capacity ablation: one Beta concentration per query.

    The Beta mean is exactly ``mu_q``, but a per-query Beta cannot represent
    multimodal disagreement or coherent cross-query structure.
    """

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        uncertainty_mode: str = "adapters",
        adapter_bottleneck: int = 32,
        num_samples: int = 32,
        inference_seed: int = 0,
        min_concentration: float = 1e-2,
        hidden_size: int | None = None,
        context_size: int | None = None,
        uncertainty_backbone: NanoTabPFNModel | None = None,
    ):
        super().__init__(
            backbone,
            uncertainty_mode=uncertainty_mode,
            adapter_bottleneck=adapter_bottleneck,
            hidden_size=hidden_size,
            context_size=context_size,
            uncertainty_backbone=uncertainty_backbone,
        )
        if num_samples < 2 or num_samples % 2 != 0:
            raise ValueError("num_samples must be an even number of at least two.")
        if min_concentration <= 0:
            raise ValueError("min_concentration must be positive.")
        self.num_samples = num_samples
        self.inference_seed = inference_seed
        self.min_concentration = min_concentration
        self.concentration_head = nn.Sequential(
            nn.Linear(self.embedding_size + self.context_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self._apply_freezing()

    def forward(self, *args, **kwargs) -> BetaConcentrationPrediction:
        support_x, support_y, query_x, kwargs = self._split_arguments(args, kwargs)
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        num_samples = kwargs.pop("num_samples", None) or self.num_samples
        sample_seed = kwargs.pop("sample_seed", None)
        sample_seed = self.inference_seed if sample_seed is None else int(sample_seed)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        base_probabilities = self._frozen_mean(support_x, support_y, query_x, num_mem_chunks=num_mem_chunks)
        base_positive = base_probabilities[..., 1].clamp(PROBABILITY_MARGIN, 1.0 - PROBABILITY_MARGIN)
        support_embeddings, query_embeddings = self._uncertainty_representations(
            support_x, support_y, query_x, num_mem_chunks=num_mem_chunks
        )
        context = self.support_encoder(support_embeddings)
        context_expanded = context[:, None].expand(-1, query_embeddings.shape[1], -1)
        raw = self.concentration_head(torch.cat((query_embeddings, context_expanded), dim=-1)).squeeze(-1)
        concentration = F.softplus(raw) + self.min_concentration
        alpha = base_positive * concentration
        beta = (1.0 - base_positive) * concentration
        distribution = torch.distributions.Beta(alpha.clamp_min(1e-4), beta.clamp_min(1e-4))
        # Reparameterized Beta draws are not Sobol points; a forked, seeded
        # generator keeps them reproducible for a given inference seed.
        with torch.random.fork_rng(devices=[] if base_positive.device.type == "cpu" else [base_positive.device]):
            torch.manual_seed(sample_seed)
            samples = distribution.rsample((num_samples,))
        sample_positive = samples.permute(1, 0, 2).clamp(PROBABILITY_MARGIN, 1.0 - PROBABILITY_MARGIN)
        gate = torch.zeros_like(base_positive)
        bound = torch.zeros_like(base_positive)
        return BetaConcentrationPrediction(base_probabilities, sample_positive, gate, bound, concentration)


class ContextResamplingUncertainty:
    """Non-learned uncertainty from deterministic stratified support subsets.

    The final prediction uses the complete labelled context, so the mean is the
    vanilla mean.  Subset predictions are re-centred around that mean before
    posterior moments are computed.
    """

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        num_subsets: int = 16,
        fractions: tuple[float, ...] = (0.50, 0.75, 0.90),
        seed: int = 0,
    ):
        if num_subsets < 2:
            raise ValueError("num_subsets must be at least two.")
        if not fractions or any(not 0 < fraction < 1 for fraction in fractions):
            raise ValueError("fractions must lie strictly between zero and one.")
        self.backbone = backbone.requires_grad_(False).eval()
        self.num_subsets = num_subsets
        self.fractions = tuple(fractions)
        self.seed = seed

    def _subset_indices(self, support_y: torch.Tensor, draw: int) -> torch.Tensor:
        """Deterministic label-stratified subset for one batch item."""
        fraction = self.fractions[draw % len(self.fractions)]
        generator = torch.Generator(device="cpu").manual_seed(self.seed * 100_003 + draw)
        labels = support_y.detach().cpu()
        chosen: list[torch.Tensor] = []
        for value in torch.unique(labels):
            positions = torch.nonzero(labels == value, as_tuple=False).squeeze(-1)
            count = max(1, int(round(fraction * positions.numel())))
            permutation = torch.randperm(positions.numel(), generator=generator)
            chosen.append(positions[permutation[:count]])
        return torch.cat(chosen).sort().values

    @torch.no_grad()
    def __call__(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int = 1,
    ) -> ContinuousPosteriorPrediction:
        if support_x.shape[0] != 1:
            raise ValueError("The context-resampling baseline evaluates one episode at a time.")
        base_probabilities = self.backbone(support_x, support_y, query_x, num_mem_chunks=num_mem_chunks)[
            ..., :2
        ].softmax(dim=-1)
        base_positive = base_probabilities[..., 1]
        draws = []
        for draw in range(self.num_subsets):
            indices = self._subset_indices(support_y[0], draw).to(support_x.device)
            logits = self.backbone(
                support_x[:, indices], support_y[:, indices], query_x, num_mem_chunks=num_mem_chunks
            )[..., :2]
            draws.append(logits.softmax(dim=-1)[..., 1])
        raw = torch.stack(draws, dim=1)
        gate = torch.ones_like(base_positive)
        sample_positive, bound = centre_and_scale(base_positive, raw, gate, max_scale=1.0)
        return ContinuousPosteriorPrediction(base_probabilities, sample_positive, gate, bound)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def continuous_checkpoint(
    model: _MeanPreservingUncertaintyModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_path: str,
    source_checkpoint_sha256: str,
    stage: str,
    step: int | None = None,
    optimizer_state: dict[str, Any] | None = None,
    validation_metrics: dict[str, Any] | None = None,
    random_seeds: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    architecture = model.architecture()
    if isinstance(model, NanoTabPFNContinuousPosteriorModel):
        model_type = CONTINUOUS_CHECKPOINT_TYPE
        architecture["latent_dim"] = model.latent_dim
    elif isinstance(model, NanoTabPFNBetaConcentrationModel):
        model_type = BETA_CHECKPOINT_TYPE
        architecture["min_concentration"] = model.min_concentration
    else:
        raise TypeError("Unsupported model for the continuous checkpoint format.")
    architecture["num_samples"] = model.num_samples
    architecture["inference_seed"] = model.inference_seed
    return {
        "model_type": model_type,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "sample_generation": {
            "scheme": "scrambled_sobol_antithetic" if model_type == CONTINUOUS_CHECKPOINT_TYPE else "beta_rsample",
            "num_samples": model.num_samples,
            "inference_seed": model.inference_seed,
        },
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "training_config": training_config,
        "stage": stage,
        "step": step,
        "validation_metrics": validation_metrics,
        "random_seeds": random_seeds,
        "selection": selection,
        "source_checkpoint_path": source_checkpoint_path,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }


def save_continuous_checkpoint(path: str | Path, model: _MeanPreservingUncertaintyModel, **kwargs: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(continuous_checkpoint(model, **kwargs), Path(path))


def load_continuous_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[_MeanPreservingUncertaintyModel, dict[str, Any]]:
    """Rebuild a continuous or Beta uncertainty model from a checkpoint."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_type = checkpoint.get("model_type")
    if model_type not in {CONTINUOUS_CHECKPOINT_TYPE, BETA_CHECKPOINT_TYPE}:
        raise ValueError(
            "Checkpoint is not a continuous or Beta uncertainty model; slot checkpoints load with "
            "tfmplayground.models.hypothesis.load_bayesian_checkpoint."
        )
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    shared = {
        "uncertainty_mode": architecture["uncertainty_mode"],
        "adapter_bottleneck": architecture.get("adapter_bottleneck") or 32,
        "num_samples": architecture["num_samples"],
        "inference_seed": architecture.get("inference_seed", 0),
        "hidden_size": architecture.get("hidden_size"),
        "context_size": architecture.get("context_size"),
    }
    if model_type == CONTINUOUS_CHECKPOINT_TYPE:
        model = NanoTabPFNContinuousPosteriorModel(backbone, latent_dim=architecture["latent_dim"], **shared)
    else:
        model = NanoTabPFNBetaConcentrationModel(
            backbone, min_concentration=architecture.get("min_concentration", 1e-2), **shared
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
