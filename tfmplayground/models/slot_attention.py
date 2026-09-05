"""Slot Attention (Locatello et al., 2020) for tabular support sets.

The hypothesis heads in this package used to call a single
``nn.MultiheadAttention(seeds, support, support)`` pass "slot attention".  It is
not: that softmax runs over the evidence keys, so every slot independently
computes a weighted average of the *same* support rows and nothing makes two
slots describe different things.  Slot attention normalizes the attention over
the **slot** axis instead, so slots compete for each input row, and iterates that
competition with a GRU.  Competition is the entire mechanism -- it is what lets a
minority subset of rows claim a slot of its own rather than being averaged into
the majority.

The vision analogy is exact.  Swap pixels for support rows and objects for
label-generating regimes: a small object still wins a slot precisely because the
normalization is over slots.

``competitive=False`` normalizes over inputs instead, at identical parameter
count and iteration count.  That is the controlled ablation for "did competition
do the work, or did the extra GRU and MLP parameters?" -- it is a research
switch, not a compatibility path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class SlotAttention(nn.Module):
    """Iterative competitive attention that maps a set of inputs onto ``K`` slots.

    Args:
        num_slots: Number of slots ``K``.
        slot_size: Slot and projection width ``E``; must match the input width.
        mlp_hidden_size: Hidden width of the per-slot residual MLP.
        num_iterations: Number of competition rounds ``T``.
        epsilon: Offset added to the attention before the weighted-mean
            normalization, so an unclaimed slot cannot divide by zero.
        competitive: Normalize the attention over slots (the real mechanism).
            ``False`` normalizes over inputs, which is ordinary cross-attention.
        eval_seed: Seed used for the slot draw in ``eval()`` mode when no
            generator is supplied, which makes inference reproducible by default.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        mlp_hidden_size: int,
        num_iterations: int = 3,
        epsilon: float = 1e-8,
        competitive: bool = True,
        eval_seed: int = 0,
        max_log_sigma: float | None = None,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")
        if slot_size < 1:
            raise ValueError("slot_size must be positive.")
        if mlp_hidden_size < 1:
            raise ValueError("mlp_hidden_size must be positive.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be at least one.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.competitive = competitive
        self.eval_seed = eval_seed
        self.max_log_sigma = max_log_sigma

        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # One (mu, sigma) shared by every slot, rather than one learned vector
        # per slot.  Slots are therefore anonymous and exchangeable, which is the
        # direct answer to MEAN_PRESERVING_BAYESIAN_TRIAL.md's finding that
        # "globally persistent learned slots have no stable semantics" when the
        # candidate label functions are redrawn every episode.
        self.slots_mu = nn.Parameter(torch.empty(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )

    def initial_slots(self, batch_size: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``mu + exp(log_sigma) * noise``, shape ``(batch, num_slots, slot_size)``.

        The slots are *sampled* rather than set to ``mu`` on purpose.  Identical
        slots are a symmetric fixed point of the competition: equal slots produce
        equal attention columns, hence equal updates, forever.  The noise is what
        breaks that tie, so it is load-bearing rather than regularization.

        Resampling every training step matters for a second reason: a slot that
        always starts from the same point drifts back toward a persistent
        identity, which is the failure MEAN_PRESERVING_BAYESIAN_TRIAL.md recorded
        ("globally persistent learned slots have no stable semantics").

        In ``eval()`` mode with no explicit generator the draw is seeded from
        ``eval_seed`` instead, so inference is reproducible by default -- nothing
        is being learned there, and every evaluator in this package compares
        repeated predictions.  An explicit ``generator`` always wins.
        """
        shape = (batch_size, self.num_slots, self.slot_size)
        if generator is None and not self.training:
            generator = torch.Generator(device=self.slots_mu.device).manual_seed(self.eval_seed)
        if generator is None:
            noise = torch.randn(shape, device=self.slots_mu.device, dtype=self.slots_mu.dtype)
        else:
            # torch.randn rejects a generator whose device differs from the
            # requested one, so draw on the generator's device and move after.
            noise = torch.randn(shape, generator=generator, device=generator.device, dtype=self.slots_mu.dtype)
            noise = noise.to(self.slots_mu.device)
        log_sigma = self.slots_log_sigma
        if self.max_log_sigma is not None:
            log_sigma = log_sigma.clamp(max=self.max_log_sigma)
        return self.slots_mu + log_sigma.exp() * noise

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        slots: torch.Tensor | None = None,
        compatibility: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind ``inputs`` to slots.

        Args:
            inputs: ``(batch, num_inputs, slot_size)`` set to explain.
            generator: Seeded generator for the slot initialization.
            slots: Optional explicit initialization, ``(batch, num_slots,
                slot_size)``, bypassing the Gaussian draw.  Used by the tests to
                check permutation equivariance over the slot axis.
            compatibility: Optional replacement for the dot-product score,
                called with the layer-normalized slots and returning
                ``(batch, num_inputs, num_slots)`` logits.  Everything else in
                the loop is unchanged -- softmax over slots, weighted mean, GRU,
                the same number of iterations -- so this swaps *what "belongs
                together" means* and nothing else.

                The default asks "does this row resemble this slot", which is
                the right question for pixels and the wrong one here: what marks
                a row as belonging to the minority regime is that its *label*
                disagrees with what the majority hypothesis predicts at its
                features, which is a residual against a hypothesis rather than a
                similarity to a prototype.  A caller that scores
                ``log p(y | x, slot)`` instead is still computing a dot product
                against the slot -- with the row's label choosing the key, and a
                log-partition term subtracted.  That subtraction is the part
                that matters: the bare dot product lets a slot win rows by
                growing its norm, without ever having to predict them
                correctly, and one-slot-takes-all is reachable that way.

        Returns:
            ``slots`` of shape ``(batch, num_slots, slot_size)`` and the final
            ``attention`` of shape ``(batch, num_inputs, num_slots)``.  With
            ``competitive=True`` the attention rows sum to one over slots, so it
            reads directly as a per-input-row soft assignment.
        """
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, num_inputs, slot_size).")
        if inputs.shape[-1] != self.slot_size:
            raise ValueError(f"inputs must have width {self.slot_size}, got {inputs.shape[-1]}.")
        if inputs.shape[1] < 1:
            raise ValueError("inputs must contain at least one element.")

        normalized = self.norm_inputs(inputs)
        k = self.project_k(normalized)
        v = self.project_v(normalized)

        if slots is None:
            slots = self.initial_slots(inputs.shape[0], generator=generator)
        elif slots.shape != (inputs.shape[0], self.num_slots, self.slot_size):
            raise ValueError("slots must have shape (batch, num_slots, slot_size).")

        attention = None
        for _ in range(self.num_iterations):
            slots_previous = slots
            normalized_slots = self.norm_slots(slots)
            if compatibility is None:
                q = self.project_q(normalized_slots) * self.slot_size**-0.5
                # (batch, num_inputs, num_slots)
                logits = torch.matmul(k, q.transpose(-1, -2))
            else:
                logits = compatibility(normalized_slots)
                if logits.shape != (inputs.shape[0], inputs.shape[1], self.num_slots):
                    raise ValueError(
                        f"compatibility must return (batch, num_inputs, num_slots), got {tuple(logits.shape)}."
                    )
            # dim=-1 makes the slots compete for each input row; dim=-2 is the
            # ordinary cross-attention this module exists to replace.
            attention = logits.softmax(dim=-1 if self.competitive else -2)
            weights = attention + self.epsilon
            weights = weights / weights.sum(dim=-2, keepdim=True)
            updates = torch.matmul(weights.transpose(-1, -2), v)

            batch_size = inputs.shape[0]
            slots = self.gru(
                updates.reshape(batch_size * self.num_slots, self.slot_size),
                slots_previous.reshape(batch_size * self.num_slots, self.slot_size),
            ).reshape(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attention

    def extra_repr(self) -> str:
        return (
            f"num_slots={self.num_slots}, slot_size={self.slot_size}, "
            f"num_iterations={self.num_iterations}, competitive={self.competitive}"
        )


class SlotBindingMixin:
    """Slot construction shared by every head that reasons over task hypotheses.

    Builds a :class:`SlotAttention` under the attribute ``slot_binding`` and
    exposes :meth:`make_slots` plus the checkpoint keys describing it.
    """

    def _init_slot_binding(
        self,
        *,
        num_slots: int,
        embedding_size: int,
        mlp_hidden_size: int,
        num_slot_iterations: int = 3,
        competitive_slots: bool = True,
    ) -> None:
        self.num_slot_iterations = num_slot_iterations
        self.competitive_slots = competitive_slots
        self.slot_binding = SlotAttention(
            num_slots,
            embedding_size,
            mlp_hidden_size,
            num_iterations=num_slot_iterations,
            competitive=competitive_slots,
        )

    def make_slots(
        self, support: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(slots, attention)`` for one support set."""
        return self.slot_binding(support, generator=generator)

    def slot_binding_architecture(self) -> dict[str, Any]:
        """Checkpoint keys describing how this head builds its slots."""
        return {
            "num_slot_iterations": self.num_slot_iterations,
            "competitive_slots": self.competitive_slots,
        }


def slot_binding_kwargs(architecture: dict[str, Any]) -> dict[str, Any]:
    """Read slot-binding settings out of a checkpoint's ``architecture`` dict."""
    return {
        "num_slot_iterations": architecture.get("num_slot_iterations", 3),
        "competitive_slots": architecture.get("competitive_slots", True),
    }


def slot_assignment_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Normalized entropy of each input row's slot assignment, ``(batch, num_inputs)``.

    Zero means a row is claimed outright by one slot; one means it is spread
    evenly and no binding happened.  Only meaningful for competitive attention,
    whose rows sum to one over slots.
    """
    if attention.ndim != 3:
        raise ValueError("attention must have shape (batch, num_inputs, num_slots).")
    num_slots = attention.shape[-1]
    if num_slots < 2:
        return torch.zeros(attention.shape[:2], device=attention.device, dtype=attention.dtype)
    normalized = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy / torch.log(torch.tensor(float(num_slots), device=attention.device, dtype=attention.dtype))


__all__ = [
    "SlotAttention",
    "SlotBindingMixin",
    "slot_assignment_entropy",
    "slot_binding_kwargs",
]
