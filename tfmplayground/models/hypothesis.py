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

    def marginal_probabilities(self) -> torch.Tensor:
        slot_probabilities = F.softmax(self.slot_logits, dim=-1)
        weights = self.slot_log_weights.exp()[:, None, :, None]
        return (weights * slot_probabilities).sum(dim=2)

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


class NanoTabPFNHypothesisModel(nn.Module):
    """A two-or-more particle task posterior built on nanoTabPFN embeddings."""

    def __init__(self, backbone: NanoTabPFNModel, num_hypotheses: int = 2, num_outputs: int = 2):
        super().__init__()
        if num_hypotheses < 2:
            raise ValueError("num_hypotheses must be at least two.")
        if num_outputs != 2:
            raise ValueError("The research hypothesis model currently supports binary outputs only.")
        self.backbone = backbone
        self.num_hypotheses = num_hypotheses
        self.num_outputs = num_outputs
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
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        num_mem_chunks: int = 1,
    ) -> HypothesisPrediction:
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
