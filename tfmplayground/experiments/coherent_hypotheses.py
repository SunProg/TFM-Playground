"""Differentiable utilities for coherent binary query prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def binary_vectors(query_count: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    if not 1 <= query_count <= 4:
        raise ValueError("Differentiable binary enumeration supports between one and four queries.")
    indices = torch.arange(2**query_count, device=device)
    shifts = torch.arange(query_count - 1, -1, -1, device=device)
    return ((indices[:, None] >> shifts[None, :]) & 1).long()


def sequence_to_canonical_indices(order: Sequence[int], *, device=None) -> torch.Tensor:
    query_count = len(order)
    if sorted(order) != list(range(query_count)):
        raise ValueError(f"order must be a permutation of 0..{query_count - 1}, got {tuple(order)}.")
    sequence = binary_vectors(query_count, device=device)
    canonical = torch.zeros_like(sequence)
    canonical[:, torch.as_tensor(order, device=sequence.device)] = sequence
    powers = 2 ** torch.arange(query_count - 1, -1, -1, device=sequence.device)
    return (canonical * powers).sum(dim=1)


def enumerate_chain_joint_torch(
    model,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    order: Sequence[int],
) -> torch.Tensor:
    """Enumerate a differentiable binary chain-rule joint in canonical outcome order."""
    trials, query_count, feature_count = query_x.shape
    order = tuple(order)
    if sorted(order) != list(range(query_count)):
        raise ValueError(f"order must be a permutation of 0..{query_count - 1}, got {order}.")
    branch_probabilities = torch.ones(trials, 1, device=query_x.device, dtype=query_x.dtype)

    for depth, query_index in enumerate(order):
        branch_count = 2**depth
        prefixes = (
            binary_vectors(depth, device=query_x.device)
            if depth
            else torch.zeros(1, 0, device=query_x.device, dtype=torch.long)
        )
        repeated_x = support_x[:, None].expand(-1, branch_count, -1, -1)
        repeated_y = support_y[:, None].expand(-1, branch_count, -1)
        if depth:
            conditioned_x = query_x[:, order[:depth], :][:, None].expand(-1, branch_count, -1, -1)
            conditioned_y = prefixes[None].expand(trials, -1, -1).to(support_y.dtype)
            context_x = torch.cat((repeated_x, conditioned_x), dim=2)
            context_y = torch.cat((repeated_y, conditioned_y), dim=2)
        else:
            context_x, context_y = repeated_x, repeated_y
        next_query = query_x[:, query_index : query_index + 1, :][:, None].expand(-1, branch_count, -1, -1)
        flat_context_x = context_x.reshape(trials * branch_count, -1, feature_count)
        flat_context_y = context_y.reshape(trials * branch_count, -1)
        flat_query = next_query.reshape(trials * branch_count, 1, feature_count)
        full_x = torch.cat((flat_context_x, flat_query), dim=1)
        logits = model((full_x, flat_context_y), train_test_split_index=flat_context_x.shape[1])
        conditionals = F.softmax(logits[:, 0, :2], dim=-1).reshape(trials, branch_count, 2)
        branch_probabilities = (branch_probabilities[:, :, None] * conditionals).reshape(trials, -1)

    canonical_indices = sequence_to_canonical_indices(order, device=query_x.device)
    return branch_probabilities.index_select(1, torch.argsort(canonical_indices))


def exact_binary_joint(posterior_b: torch.Tensor, query_count: int) -> torch.Tensor:
    joint = torch.zeros(posterior_b.shape[0], 2**query_count, device=posterior_b.device, dtype=posterior_b.dtype)
    joint[:, 0] = 1.0 - posterior_b
    joint[:, -1] = posterior_b
    return joint


def jensen_shannon_divergence(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    midpoint = 0.5 * (left + right)

    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.where(p > 0, p * (p.clamp_min(epsilon).log() - q.clamp_min(epsilon).log()), 0).sum(-1)

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def outcome_indices(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    powers = 2 ** torch.arange(labels.shape[1] - 1, -1, -1, device=labels.device)
    return (labels * powers).sum(dim=1)
