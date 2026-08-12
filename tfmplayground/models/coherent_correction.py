"""Non-leaking task-hypothesis weighting for nanoTabPFN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.hypothesis import HypothesisPrediction
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def permutation_invariant_folds(
    x: torch.Tensor,
    *,
    num_partitions: int = 2,
) -> torch.Tensor:
    """Assign rows to balanced folds using deterministic hashes of x only.

    Returns a boolean tensor shaped ``(partitions, batch, rows)``. True marks
    fold one and False marks fold zero. Sorting by content makes the assignment
    equivariant to support-row permutation for non-identical rows.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, rows, features).")
    if x.shape[1] < 2:
        raise ValueError("Cross-fitting requires at least two support rows.")
    feature_index = torch.arange(1, x.shape[2] + 1, device=x.device, dtype=x.dtype)
    assignments = []
    for partition in range(num_partitions):
        salt = float(partition + 1)
        scores = torch.sin(x * feature_index[None, None, :] * (12.9898 + salt * 7.233)).sum(-1) + torch.cos(
            x.square() * feature_index[None, None, :] * (4.1414 + salt * 3.117)
        ).sum(-1)
        order = scores.argsort(dim=1)
        ranks = order.argsort(dim=1)
        assignments.append(ranks.remainder(2).bool())
    return torch.stack(assignments)


def gather_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("Every batch item must contribute the same number of rows to a fold.")
    indices = mask.to(torch.int64).argsort(dim=1, descending=True)[:, : int(counts[0])]
    expansion = indices[(...,) + (None,) * (values.ndim - 2)].expand(-1, -1, *values.shape[2:])
    return values.gather(1, expansion)


@dataclass(frozen=True)
class CrossFitPrediction(HypothesisPrediction):
    full_slots: torch.Tensor
    context_slots: torch.Tensor
    heldout_logits: torch.Tensor
    fold_assignments: torch.Tensor

    def alignment_loss(self) -> torch.Tensor:
        reference = self.full_slots[:, None, None]
        return (1.0 - F.cosine_similarity(reference, self.context_slots, dim=-1)).mean()


@dataclass(frozen=True)
class VariationalPrediction(HypothesisPrediction):
    full_slots: torch.Tensor
    context_slots: torch.Tensor

    def alignment_loss(self) -> torch.Tensor:
        reference = self.full_slots[:, None]
        return (1.0 - F.cosine_similarity(reference, self.context_slots, dim=-1)).mean()


class _SharedHypothesisHead(nn.Module):
    def __init__(self, backbone: NanoTabPFNModel, num_hypotheses: int = 2, num_outputs: int = 2):
        super().__init__()
        if num_hypotheses != 2 or num_outputs != 2:
            raise ValueError("The correction models currently require two binary hypotheses.")
        self.backbone = backbone
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
        embedding_size = backbone.embedding_size
        hidden_size = backbone.mlp_hidden_size
        self.hypothesis_queries = nn.Parameter(torch.empty(num_hypotheses, embedding_size))
        nn.init.normal_(self.hypothesis_queries, std=embedding_size**-0.5)
        self.slot_attention = nn.MultiheadAttention(embedding_size, backbone.num_attention_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(embedding_size)
        self.slot_decoder = nn.Sequential(
            nn.Linear(embedding_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.slot_prior_logits = nn.Parameter(torch.zeros(num_hypotheses))

    def _encode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode_table(
            (torch.cat((support_x, query_x), dim=1), support_y),
            support_x.shape[1],
        )[:, :, -1, :]
        return encoded[:, : support_x.shape[1]], encoded[:, support_x.shape[1] :]

    def _make_slots(self, support_embeddings: torch.Tensor) -> torch.Tensor:
        seeds = self.hypothesis_queries[None].expand(support_embeddings.shape[0], -1, -1)
        attended = self.slot_attention(seeds, support_embeddings, support_embeddings, need_weights=False)[0]
        return self.slot_norm(seeds + attended)

    def _decode(self, query_embeddings: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        query_count = query_embeddings.shape[1]
        query = query_embeddings[:, :, None].expand(-1, -1, self.num_hypotheses, -1)
        slot = slots[:, None].expand(-1, query_count, -1, -1)
        return self.slot_decoder(torch.cat((query, slot, query * slot), dim=-1))

    def context_slots(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        support_embeddings, _ = self._encode(support_x, support_y, query_x)
        return self._make_slots(support_embeddings)

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def unfreeze_final_backbone_blocks(self, count: int = 2) -> None:
        self.backbone.requires_grad_(False)
        for block in self.backbone.transformer_blocks[-count:]:
            block.requires_grad_(True)


class NanoTabPFNCrossFitHypothesisModel(_SharedHypothesisHead):
    """Weight hypotheses by out-of-fold predictive support likelihood."""

    model_type = "nanotabpfn_crossfit_hypothesis"

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        num_hypotheses: int = 2,
        num_outputs: int = 2,
        num_partitions: int = 2,
    ):
        super().__init__(backbone, num_hypotheses, num_outputs)
        if num_partitions < 1:
            raise ValueError("num_partitions must be positive.")
        self.num_partitions = num_partitions

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        num_mem_chunks: int = 1,
    ) -> CrossFitPrediction:
        del num_mem_chunks
        full_x, support_y = src
        support_x = full_x[:, :train_test_split_index]
        query_x = full_x[:, train_test_split_index:]
        support_embeddings, query_embeddings = self._encode(support_x, support_y, query_x)
        full_slots = self._make_slots(support_embeddings)
        query_logits = self._decode(query_embeddings, full_slots)

        assignments = permutation_invariant_folds(support_x, num_partitions=self.num_partitions)
        batch_size, support_count = support_y.shape
        row_log_likelihoods = torch.zeros(
            self.num_partitions,
            batch_size,
            support_count,
            self.num_hypotheses,
            device=full_x.device,
            dtype=full_x.dtype,
        )
        heldout_logits = torch.zeros(
            self.num_partitions,
            batch_size,
            support_count,
            self.num_hypotheses,
            self.num_outputs,
            device=full_x.device,
            dtype=full_x.dtype,
        )
        context_slots = []
        for partition in range(self.num_partitions):
            partition_slots = []
            for heldout_fold in (False, True):
                heldout_mask = assignments[partition] == heldout_fold
                context_mask = ~heldout_mask
                context_x = gather_rows(support_x, context_mask)
                context_y = gather_rows(support_y, context_mask)
                heldout_x = gather_rows(support_x, heldout_mask)
                heldout_y = gather_rows(support_y, heldout_mask).long()
                context_embeddings, heldout_embeddings = self._encode(context_x, context_y, heldout_x)
                slots = self._make_slots(context_embeddings)
                partition_slots.append(slots)
                logits = self._decode(heldout_embeddings, slots)
                log_probabilities = F.log_softmax(logits, dim=-1)
                gathered = log_probabilities.gather(
                    -1,
                    heldout_y[:, :, None, None].expand(-1, -1, self.num_hypotheses, 1),
                ).squeeze(-1)
                row_log_likelihoods[partition][heldout_mask] = gathered.reshape(-1, self.num_hypotheses)
                heldout_logits[partition][heldout_mask] = logits.reshape(-1, self.num_hypotheses, self.num_outputs)
            context_slots.append(torch.stack(partition_slots))
        context_slots_tensor = torch.stack(context_slots, dim=1).permute(2, 1, 0, 3, 4)
        mean_row_log_likelihood = row_log_likelihoods.mean(dim=0)
        slot_scores = mean_row_log_likelihood.sum(dim=1) + self.slot_prior_logits
        return CrossFitPrediction(
            query_logits,
            F.log_softmax(slot_scores, dim=-1),
            mean_row_log_likelihood,
            full_slots,
            context_slots_tensor,
            heldout_logits,
            assignments,
        )


class NanoTabPFNVariationalHypothesisModel(_SharedHypothesisHead):
    """Conditional two-state latent model used when cross-fitting fails."""

    model_type = "nanotabpfn_variational_hypothesis"

    def __init__(self, backbone: NanoTabPFNModel, num_hypotheses: int = 2, num_outputs: int = 2):
        super().__init__(backbone, num_hypotheses, num_outputs)
        embedding_size = backbone.embedding_size
        hidden_size = backbone.mlp_hidden_size
        self.posterior_row_encoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_size),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(embedding_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_hypotheses),
        )

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        num_mem_chunks: int = 1,
    ) -> VariationalPrediction:
        del num_mem_chunks
        full_x, support_y = src
        support_x = full_x[:, :train_test_split_index]
        query_x = full_x[:, train_test_split_index:]
        support_embeddings, query_embeddings = self._encode(support_x, support_y, query_x)
        slots = self._make_slots(support_embeddings)
        slot_logits = self._decode(query_embeddings, slots)
        pooled = self.posterior_row_encoder(support_embeddings).mean(dim=1)
        log_count = pooled.new_full((pooled.shape[0], 1), math.log1p(train_test_split_index))
        posterior_logits = self.posterior_head(torch.cat((pooled, log_count), dim=-1)) + self.slot_prior_logits
        row_diagnostics = pooled[:, None, : self.num_hypotheses]
        fold = permutation_invariant_folds(support_x, num_partitions=1)[0]
        context_slots = []
        for heldout_fold in (False, True):
            context_mask = fold != heldout_fold
            context_slots.append(
                self.context_slots(
                    gather_rows(support_x, context_mask),
                    gather_rows(support_y, context_mask),
                    query_x,
                )
            )
        return VariationalPrediction(
            slot_logits,
            F.log_softmax(posterior_logits, dim=-1),
            row_diagnostics,
            slots,
            torch.stack(context_slots, dim=1),
        )


def correction_checkpoint(
    model: NanoTabPFNCrossFitHypothesisModel | NanoTabPFNVariationalHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
) -> dict[str, Any]:
    backbone = model.backbone
    architecture = {
        "num_layers": backbone.num_layers,
        "embedding_size": backbone.embedding_size,
        "num_attention_heads": backbone.num_attention_heads,
        "mlp_hidden_size": backbone.mlp_hidden_size,
        "backbone_num_outputs": backbone.num_outputs,
        "num_hypotheses": model.num_hypotheses,
        "num_outputs": model.num_outputs,
    }
    if isinstance(model, NanoTabPFNCrossFitHypothesisModel):
        architecture["num_partitions"] = model.num_partitions
    return {
        "model_type": model.model_type,
        "architecture": architecture,
        "model": model.state_dict(),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "training_config": training_config,
        "stage": stage,
    }


def save_correction_checkpoint(
    path: str | Path,
    model: NanoTabPFNCrossFitHypothesisModel | NanoTabPFNVariationalHypothesisModel,
    *,
    training_config: dict[str, Any],
    source_checkpoint_sha256: str,
    stage: str,
) -> None:
    torch.save(
        correction_checkpoint(
            model,
            training_config=training_config,
            source_checkpoint_sha256=source_checkpoint_sha256,
            stage=stage,
        ),
        Path(path),
    )


def load_correction_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNCrossFitHypothesisModel | NanoTabPFNVariationalHypothesisModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    architecture = checkpoint["architecture"]
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    common = {
        "backbone": backbone,
        "num_hypotheses": architecture["num_hypotheses"],
        "num_outputs": architecture["num_outputs"],
    }
    if checkpoint["model_type"] == NanoTabPFNCrossFitHypothesisModel.model_type:
        model = NanoTabPFNCrossFitHypothesisModel(**common, num_partitions=architecture["num_partitions"])
    elif checkpoint["model_type"] == NanoTabPFNVariationalHypothesisModel.model_type:
        model = NanoTabPFNVariationalHypothesisModel(**common)
    else:
        raise ValueError(f"Unknown correction checkpoint type: {checkpoint.get('model_type')}")
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
