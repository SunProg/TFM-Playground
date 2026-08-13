"""Coherent latent-hypothesis head for a pretrained nanoTabPFN backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


@dataclass(frozen=True)
class HypothesisPrediction:
    """Predictions conditional on shared task hypotheses."""

    slot_logits: torch.Tensor
    slot_log_weights: torch.Tensor
    row_log_evidence: torch.Tensor

    @property
    def query_probabilities(self) -> torch.Tensor:
        """Probability of every query label conditional on every hypothesis.

        The shape is ``(batch, query, hypothesis, class)``.  Keeping the
        hypothesis axis explicit is important: averaging logits would destroy
        the epistemic information represented by this head.
        """
        return self.slot_logits.softmax(dim=-1)

    @property
    def slot_probabilities(self) -> torch.Tensor:
        return self.query_probabilities

    @property
    def posterior_weights(self) -> torch.Tensor:
        return self.slot_log_weights.exp()

    def hypothesis_probabilities(self) -> torch.Tensor:
        """Alias used by the static Bayesian API."""
        return self.query_probabilities

    def marginal_probabilities(self) -> torch.Tensor:
        slot_probabilities = self.query_probabilities
        weights = self.slot_log_weights.exp()[:, None, :, None]
        return (weights * slot_probabilities).sum(dim=2)

    def mixture_probabilities(self) -> torch.Tensor:
        return self.marginal_probabilities()

    def predictive_entropy(self) -> torch.Tensor:
        probabilities = self.marginal_probabilities().clamp_min(torch.finfo(self.slot_logits.dtype).tiny)
        return -(probabilities * probabilities.log()).sum(dim=-1)

    def expected_hypothesis_entropy(self) -> torch.Tensor:
        probabilities = self.query_probabilities.clamp_min(torch.finfo(self.slot_logits.dtype).tiny)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return (self.posterior_weights[:, None, :] * entropy).sum(dim=-1)

    def mutual_information(self) -> torch.Tensor:
        """Per-query predictive minus expected aleatoric entropy."""
        return (self.predictive_entropy() - self.expected_hypothesis_entropy()).clamp_min(0.0)

    def epistemic_uncertainty(self) -> torch.Tensor:
        return self.mutual_information()

    def posterior_entropy(self) -> torch.Tensor:
        weights = self.posterior_weights.clamp_min(torch.finfo(self.slot_logits.dtype).tiny)
        return -(weights * weights.log()).sum(dim=-1)

    def effective_hypothesis_count(self) -> torch.Tensor:
        return self.posterior_entropy().exp()

    def effective_sample_size(self) -> torch.Tensor:
        """Importance-sampling ESS of the normalized hypothesis weights."""
        return self.posterior_weights.square().sum(dim=-1).clamp_min(
            torch.finfo(self.slot_logits.dtype).tiny
        ).reciprocal()

    def should_resample(self, threshold_fraction: float = 0.5) -> torch.Tensor:
        if not 0 < threshold_fraction <= 1:
            raise ValueError("threshold_fraction must lie in (0, 1].")
        return self.effective_sample_size() < threshold_fraction * self.slot_logits.shape[2]

    @torch.no_grad()
    def systematic_resample_indices(
        self,
        num_samples: int | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw fixed-cost hypothesis ancestors without changing prediction.

        The default API intentionally keeps the lower-variance weighted
        mixture. Callers may use these indices only when downstream evaluation
        of every retained hypothesis is too expensive.
        """
        if num_samples is None:
            num_samples = self.slot_logits.shape[2]
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        weights = self.posterior_weights
        offset = torch.rand(
            weights.shape[0],
            1,
            device=weights.device,
            dtype=weights.dtype,
            generator=generator,
        )
        positions = (torch.arange(num_samples, device=weights.device, dtype=weights.dtype)[None] + offset)
        positions = positions / num_samples
        cumulative = weights.cumsum(dim=-1)
        cumulative[:, -1] = 1.0
        return torch.searchsorted(cumulative.contiguous(), positions.contiguous(), right=False)

    def hypothesis_disagreement(self) -> torch.Tensor:
        """Weighted variance of the positive-class probabilities.

        This is zero for identical hypotheses and positive whenever weighted
        hypotheses disagree, while remaining bounded and differentiable.
        """
        positive = self.query_probabilities[..., 1]
        mean = (positive * self.posterior_weights[:, None, :]).sum(dim=-1, keepdim=True)
        return ((positive - mean).square() * self.posterior_weights[:, None, :]).sum(dim=-1)

    def epistemic_variance(self) -> torch.Tensor:
        return self.hypothesis_disagreement()

    def joint_log_probabilities(self, outcomes: torch.Tensor | None = None) -> torch.Tensor:
        """Return the coherent mixture probability for complete binary query vectors."""
        batch_size, query_count, num_hypotheses, num_classes = self.slot_logits.shape
        if num_classes != 2:
            raise ValueError(f"Binary joint prediction requires two outputs, found {num_classes}.")
        if outcomes is None:
            indices = torch.arange(2**query_count, device=self.slot_logits.device)
            shifts = torch.arange(query_count - 1, -1, -1, device=self.slot_logits.device)
            outcomes = ((indices[:, None] >> shifts[None, :]) & 1).long()
        outcomes = outcomes.to(device=self.slot_logits.device, dtype=torch.long)
        if outcomes.ndim != 2 or outcomes.shape[1] != query_count:
            raise ValueError(f"outcomes must have shape (n, {query_count}), found {tuple(outcomes.shape)}.")

        log_probabilities = F.log_softmax(self.slot_logits, dim=-1)
        expanded = log_probabilities[:, None].expand(-1, outcomes.shape[0], -1, -1, -1)
        gather_index = outcomes[None, :, :, None, None].expand(batch_size, -1, -1, num_hypotheses, 1)
        trajectory_log_probabilities = expanded.gather(-1, gather_index).squeeze(-1).sum(dim=2)
        return torch.logsumexp(trajectory_log_probabilities + self.slot_log_weights[:, None, :], dim=-1)

    def joint_probabilities(self, outcomes: torch.Tensor | None = None) -> torch.Tensor:
        return self.joint_log_probabilities(outcomes).exp()


@dataclass(frozen=True)
class MeanPreservingPrediction(HypothesisPrediction):
    """Static hypothesis posterior whose marginal equals frozen nanoTabPFN.

    ``slot_logits`` contains logarithms of the corrected, mean-preserving
    probabilities. ``raw_slot_logits`` remains available for candidate-slot
    supervision during training.
    """

    raw_slot_logits: torch.Tensor
    base_probabilities: torch.Tensor
    fold_assignments: torch.Tensor

    @property
    def raw_query_probabilities(self) -> torch.Tensor:
        return self.raw_slot_logits.softmax(dim=-1)

    def mean_preservation_error(self) -> torch.Tensor:
        return (self.marginal_probabilities() - self.base_probabilities).abs().amax(dim=(-1, -2))


class NanoTabPFNHypothesisModel(nn.Module):
    """A two-or-more particle task posterior built on nanoTabPFN embeddings."""

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        num_hypotheses: int = 2,
        num_outputs: int = 2,
        query_count: int | None = None,
    ):
        super().__init__()
        if num_hypotheses < 2:
            raise ValueError("num_hypotheses must be at least two.")
        if num_outputs != 2:
            raise ValueError("The research hypothesis model currently supports binary outputs only.")
        if query_count is not None and (query_count < 1 or num_hypotheses > 2**query_count):
            raise ValueError("num_hypotheses must satisfy num_hypotheses <= 2**query_count.")
        self.backbone = backbone
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
        self.query_count = query_count
        embedding_size = backbone.embedding_size
        hidden_size = backbone.mlp_hidden_size

        self.hypothesis_queries = nn.Parameter(torch.empty(num_hypotheses, embedding_size))
        nn.init.normal_(self.hypothesis_queries, std=embedding_size**-0.5)
        self.slot_attention = nn.MultiheadAttention(
            embedding_size,
            backbone.num_attention_heads,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(embedding_size)
        self.evidence_head = nn.Sequential(
            nn.Linear(embedding_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.slot_prior_logits = nn.Parameter(torch.zeros(num_hypotheses))
        self.slot_decoder = nn.Sequential(
            nn.Linear(embedding_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> HypothesisPrediction:
        if len(args) == 3:
            x_train, y_train, x_query = args
            src = (torch.cat((x_train, x_query), dim=1), y_train)
            train_test_split_index = x_train.shape[1]
        elif len(args) == 1 and isinstance(args[0], tuple):
            src = args[0]
            train_test_split_index = kwargs.pop("train_test_split_index", None)
        else:
            raise TypeError("Expected (x_train, y_train, x_query) or ((x, y), train_test_split_index=...).")
        if train_test_split_index is None:
            raise TypeError("train_test_split_index is required for the concatenated-table interface.")
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        encoded = self.backbone.encode_table(src, train_test_split_index, num_mem_chunks=num_mem_chunks)
        target_embeddings = encoded[:, :, -1, :]
        support_embeddings = target_embeddings[:, :train_test_split_index]
        query_embeddings = target_embeddings[:, train_test_split_index:]
        batch_size, support_count, embedding_size = support_embeddings.shape

        slot_seeds = self.hypothesis_queries[None].expand(batch_size, -1, -1)
        attended_slots = self.slot_attention(slot_seeds, support_embeddings, support_embeddings, need_weights=False)[0]
        slots = self.slot_norm(slot_seeds + attended_slots)

        support_expanded = support_embeddings[:, :, None, :].expand(-1, -1, self.num_hypotheses, -1)
        slot_for_support = slots[:, None, :, :].expand(-1, support_count, -1, -1)
        evidence_features = torch.cat((support_expanded, slot_for_support, support_expanded * slot_for_support), dim=-1)
        row_log_evidence = self.evidence_head(evidence_features).squeeze(-1)
        slot_scores = row_log_evidence.sum(dim=1) + self.slot_prior_logits
        slot_log_weights = F.log_softmax(slot_scores, dim=-1)

        query_count = query_embeddings.shape[1]
        if self.num_hypotheses > 2**query_count:
            raise ValueError(
                f"num_hypotheses={self.num_hypotheses} cannot represent distinct binary hypotheses for "
                f"query_count={query_count}; require num_hypotheses <= 2**query_count."
            )
        query_expanded = query_embeddings[:, :, None, :].expand(-1, -1, self.num_hypotheses, -1)
        slot_for_queries = slots[:, None, :, :].expand(-1, query_count, -1, -1)
        decoder_features = torch.cat((query_expanded, slot_for_queries, query_expanded * slot_for_queries), dim=-1)
        slot_logits = self.slot_decoder(decoder_features)
        return HypothesisPrediction(slot_logits, slot_log_weights, row_log_evidence)

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def unfreeze_final_backbone_blocks(self, count: int = 2) -> None:
        self.backbone.requires_grad_(False)
        for block in self.backbone.transformer_blocks[-count:]:
            block.requires_grad_(True)


def _permutation_invariant_folds(
    x: torch.Tensor,
    y: torch.Tensor,
    num_partitions: int,
) -> torch.Tensor:
    """Create balanced, row-permutation-equivariant two-fold assignments."""
    feature_index = torch.arange(1, x.shape[2] + 1, device=x.device, dtype=x.dtype)
    assignments = []
    for partition in range(num_partitions):
        salt = float(partition + 1)
        scores = torch.sin(x * feature_index[None, None, :] * (12.9898 + salt * 7.233)).sum(-1)
        scores = scores + torch.cos(x.square() * feature_index[None, None, :] * (4.1414 + salt * 3.117)).sum(-1)
        scores = scores + y * (0.137 + salt * 0.019)
        order = scores.argsort(dim=1)
        assignments.append(order.argsort(dim=1).remainder(2).bool())
    return torch.stack(assignments)


def _gather_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("Every batch item must contribute the same number of rows to a fold.")
    indices = mask.to(torch.int64).argsort(dim=1, descending=True)[:, : int(counts[0])]
    expansion = indices[(...,) + (None,) * (values.ndim - 2)].expand(-1, -1, *values.shape[2:])
    return values.gather(1, expansion)


class NanoTabPFNMeanPreservingBayesianModel(NanoTabPFNHypothesisModel):
    """Frozen nanoTabPFN mean with cross-fitted latent-task uncertainty.

    The backbone supplies the predictive mean and never receives gradients.
    Hypothesis weights are support-label likelihoods computed out of fold, and
    weighted slot deviations are centered exactly around the vanilla mean.
    """

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        num_hypotheses: int = 2,
        num_outputs: int = 2,
        query_count: int | None = None,
        *,
        num_partitions: int = 2,
        likelihood_temperature: float = 0.1,
    ):
        super().__init__(backbone, num_hypotheses, num_outputs, query_count)
        if num_partitions < 1:
            raise ValueError("num_partitions must be positive.")
        if likelihood_temperature <= 0:
            raise ValueError("likelihood_temperature must be positive.")
        self.num_partitions = num_partitions
        self.likelihood_temperature = likelihood_temperature
        hidden_size = backbone.mlp_hidden_size
        self.dispersion_head = nn.Sequential(
            nn.Linear(backbone.embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.constant_(self.dispersion_head[-1].bias, -2.0)
        # The inherited direct evidence head is retained in state dictionaries
        # for compatibility, but mean-preserving models use cross-fit evidence.
        self.evidence_head.requires_grad_(False)
        self.backbone.requires_grad_(False)

    def _encode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        *,
        num_mem_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode_table(
            (torch.cat((support_x, query_x), dim=1), support_y),
            support_x.shape[1],
            num_mem_chunks=num_mem_chunks,
        )[:, :, -1, :]
        return encoded[:, : support_x.shape[1]].detach(), encoded[:, support_x.shape[1] :].detach()

    def _make_slots(self, support_embeddings: torch.Tensor) -> torch.Tensor:
        seeds = self.hypothesis_queries[None].expand(support_embeddings.shape[0], -1, -1)
        attended = self.slot_attention(seeds, support_embeddings, support_embeddings, need_weights=False)[0]
        return self.slot_norm(seeds + attended)

    def _decode(self, embeddings: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        rows = embeddings[:, :, None].expand(-1, -1, self.num_hypotheses, -1)
        expanded_slots = slots[:, None].expand(-1, embeddings.shape[1], -1, -1)
        return self.slot_decoder(torch.cat((rows, expanded_slots, rows * expanded_slots), dim=-1))

    def _crossfit_log_weights(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *,
        num_mem_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, support_count = support_y.shape
        if support_count < 4:
            assignments = torch.empty(0, batch_size, support_count, dtype=torch.bool, device=support_x.device)
            row_log_evidence = torch.zeros(
                batch_size,
                support_count,
                self.num_hypotheses,
                device=support_x.device,
                dtype=support_x.dtype,
            )
            prior = F.log_softmax(self.slot_prior_logits[None].expand(batch_size, -1), dim=-1)
            return prior, row_log_evidence, assignments

        assignments = _permutation_invariant_folds(support_x, support_y, self.num_partitions)
        row_log_evidence = torch.zeros(
            self.num_partitions,
            batch_size,
            support_count,
            self.num_hypotheses,
            device=support_x.device,
            dtype=support_x.dtype,
        )
        for partition in range(self.num_partitions):
            for heldout_fold in (False, True):
                heldout_mask = assignments[partition] == heldout_fold
                context_mask = ~heldout_mask
                context_x = _gather_rows(support_x, context_mask)
                context_y = _gather_rows(support_y, context_mask)
                heldout_x = _gather_rows(support_x, heldout_mask)
                heldout_y = _gather_rows(support_y, heldout_mask).long()
                context_embeddings, heldout_embeddings = self._encode(
                    context_x,
                    context_y,
                    heldout_x,
                    num_mem_chunks=num_mem_chunks,
                )
                logits = self._decode(heldout_embeddings, self._make_slots(context_embeddings))
                observed = F.log_softmax(logits, dim=-1).gather(
                    -1,
                    heldout_y[:, :, None, None].expand(-1, -1, self.num_hypotheses, 1),
                ).squeeze(-1)
                row_log_evidence[partition][heldout_mask] = observed.reshape(-1, self.num_hypotheses)
        mean_evidence = row_log_evidence.mean(dim=0)
        scores = self.slot_prior_logits[None] + self.likelihood_temperature * mean_evidence.sum(dim=1)
        return F.log_softmax(scores, dim=-1), mean_evidence, assignments

    def forward(self, *args, **kwargs) -> MeanPreservingPrediction:
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
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if self.num_hypotheses > 2 ** query_x.shape[1]:
            raise ValueError(
                f"num_hypotheses={self.num_hypotheses} cannot represent distinct binary hypotheses for "
                f"query_count={query_x.shape[1]}; require num_hypotheses <= 2**query_count."
            )

        support_embeddings, query_embeddings = self._encode(
            support_x,
            support_y,
            query_x,
            num_mem_chunks=num_mem_chunks,
        )
        slots = self._make_slots(support_embeddings)
        raw_slot_logits = self._decode(query_embeddings, slots)
        slot_log_weights, row_log_evidence, assignments = self._crossfit_log_weights(
            support_x,
            support_y,
            num_mem_chunks=num_mem_chunks,
        )

        base_logits = self.backbone.decoder(query_embeddings)[..., :2]
        base_probabilities = base_logits.softmax(dim=-1)
        base_positive = base_probabilities[..., 1]
        weights = slot_log_weights.exp()[:, None, :]
        raw_positive = raw_slot_logits.softmax(dim=-1)[..., 1]
        centered = raw_positive - (weights * raw_positive).sum(dim=-1, keepdim=True)

        eps = torch.finfo(centered.dtype).eps
        positive_extent = centered.clamp_min(0).amax(dim=-1)
        negative_extent = (-centered).clamp_min(0).amax(dim=-1)
        positive_limit = (1.0 - eps - base_positive).clamp_min(0) / positive_extent.clamp_min(eps)
        negative_limit = (base_positive - eps).clamp_min(0) / negative_extent.clamp_min(eps)
        bound = torch.minimum(torch.ones_like(base_positive), torch.minimum(positive_limit, negative_limit))
        learned_scale = torch.sigmoid(self.dispersion_head(query_embeddings).squeeze(-1))
        corrected_positive = base_positive[:, :, None] + (bound * learned_scale)[:, :, None] * centered
        corrected_positive = corrected_positive.clamp(eps, 1.0 - eps)
        corrected = torch.stack((1.0 - corrected_positive, corrected_positive), dim=-1)
        # Log probabilities are valid logits and make the inherited API exact.
        corrected_logits = corrected.log()
        return MeanPreservingPrediction(
            corrected_logits,
            slot_log_weights,
            row_log_evidence,
            raw_slot_logits,
            base_probabilities,
            assignments,
        )

    def unfreeze_final_backbone_blocks(self, count: int = 2) -> None:
        del count
        raise RuntimeError("The mean-preserving Bayesian model requires a frozen nanoTabPFN backbone.")


# The original experiment exposed this class under a slot-oriented name.  The
# model is now the static Bayesian track; retain the old name as an alias so
# old research scripts and checkpoints continue to work.
NanoTabPFNBayesianModel = NanoTabPFNMeanPreservingBayesianModel
NanoTabPFNBayesianHypothesisModel = NanoTabPFNMeanPreservingBayesianModel
NanoTabPFNStaticBayesianModel = NanoTabPFNMeanPreservingBayesianModel
BayesianPrediction = MeanPreservingPrediction


def hypothesis_checkpoint(
    model: NanoTabPFNHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backbone = model.backbone
    checkpoint = {
        "model_type": "nanotabpfn_hypothesis",
        "architecture": {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "num_hypotheses": model.num_hypotheses,
            "num_outputs": model.num_outputs,
            "mean_preserving": isinstance(model, NanoTabPFNMeanPreservingBayesianModel),
            "num_partitions": getattr(model, "num_partitions", None),
            "likelihood_temperature": getattr(model, "likelihood_temperature", None),
        },
        "model": model.state_dict(),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "training_config": training_config,
        "stage": stage,
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    return checkpoint


def save_hypothesis_checkpoint(
    path: str | Path,
    model: NanoTabPFNHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    torch.save(
        hypothesis_checkpoint(
            model,
            training_config=training_config,
            source_checkpoint_sha256=source_checkpoint_sha256,
            stage=stage,
            optimizer_state=optimizer_state,
        ),
        Path(path),
    )


def load_hypothesis_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNHypothesisModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if checkpoint.get("model_type") != "nanotabpfn_hypothesis":
        raise ValueError("Checkpoint is not a nanoTabPFN hypothesis model.")
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    model = NanoTabPFNHypothesisModel(
        backbone,
        num_hypotheses=architecture["num_hypotheses"],
        num_outputs=architecture["num_outputs"],
    )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def bayesian_checkpoint(
    model: NanoTabPFNHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    optimizer_state: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the static Bayesian model with explicit provenance."""
    checkpoint = hypothesis_checkpoint(
        model,
        training_config=training_config,
        source_checkpoint_sha256=source_checkpoint_sha256,
        stage=stage,
        optimizer_state=optimizer_state,
    )
    checkpoint["model_type"] = "nanotabpfn_bayesian"
    checkpoint["selection"] = selection
    return checkpoint


def save_bayesian_checkpoint(
    path: str | Path,
    model: NanoTabPFNHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
    optimizer_state: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        bayesian_checkpoint(
            model,
            training_config=training_config,
            source_checkpoint_sha256=source_checkpoint_sha256,
            stage=stage,
            optimizer_state=optimizer_state,
            selection=selection,
        ),
        Path(path),
    )


def load_bayesian_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNHypothesisModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    if checkpoint.get("model_type") not in {"nanotabpfn_bayesian", "nanotabpfn_hypothesis"}:
        raise ValueError("Checkpoint is not a static Bayesian nanoTabPFN model.")
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    if architecture.get("mean_preserving", False):
        model = NanoTabPFNBayesianModel(
            backbone,
            num_hypotheses=architecture["num_hypotheses"],
            num_outputs=architecture["num_outputs"],
            num_partitions=architecture.get("num_partitions") or 2,
            likelihood_temperature=architecture.get("likelihood_temperature") or 0.1,
        )
    else:
        # Legacy static checkpoints retain their original non-mean-preserving
        # behavior and can still be inspected or compared.
        model = NanoTabPFNHypothesisModel(
            backbone,
            num_hypotheses=architecture["num_hypotheses"],
            num_outputs=architecture["num_outputs"],
        )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
