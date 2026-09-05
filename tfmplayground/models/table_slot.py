"""Feature-aware table slots for the multi-regime v2 experiments.

Unlike the historical slot heads, these adapters never take the target column
as their input.  They first compete over every encoded cell in each row, then
compete over feature-pooled support rows.  The two paths are deliberately kept
separate so feature routing and regime routing can be inspected independently.

``scope`` selects which of those two competitions actually runs, which is the
ablation the paired paths were kept separate for:

``"cell_and_data"``
    Both, cells first: the historical behaviour and the default.
``"cell"``
    Cells only.  The decoder still needs one slot set per episode, so the
    per-row cell slots are averaged over the support rows and each row's
    assignment is the mean of its cells' assignments.  Competitive attention
    sums to one over slots for every cell, so that mean is still a
    distribution and needs no renormalization.
``"data"``
    Feature-pooled rows only, with the cell path removed entirely rather than
    gated off -- a zero ``feature_mix`` would still train its parameters.

The three scopes are separate models, not one model with a switch: each drops
the parameters of the path it does not run, so a scope comparison answers "does
this competition help" rather than "did the extra parameters help".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel, TransformerEncoderLayer
from tfmplayground.models.slot_attention import SlotAttention
from tfmplayground.models.slot_regime import SlotRegimePrediction, _SlotDecoder

#: Which slot competitions an adapter runs.  See the module docstring.
SLOT_SCOPES = ("cell_and_data", "cell", "data")

SlotScope = Literal["cell_and_data", "cell", "data"]

#: How a query row's routing to slots is computed, head mode only.
#:
#: ``"decoder"`` -- the historical design: the *labelled*-pass query
#: embedding (which has already attended over support rows carrying real
#: labels, through ordinary full self-attention) is decoded against the
#: slots by the learned MLP gate.  A query row can therefore get a good
#: prediction without the slot competition or its gate contributing anything
#: -- the backbone's own attention may already be carrying the signal.
#:
#: ``"blind_decoder"`` -- the query embedding is taken from the *label-blind*
#: pass instead (`_blind_pass`, same one `reconstruct_support` uses), so no
#: label information can reach it by attending over labelled support states.
#: Whatever the slots contribute now has to come through the slots
#: themselves.  Still uses the learned MLP gate.
#:
#: ``"blind_similarity"`` -- blind query embedding as above, and the gate
#: itself is replaced: instead of the learned mask channel, each slot's
#: routing weight is the cosine similarity between the query's blind
#: embedding and that slot's blind-support centroid (support rows' blind
#: embeddings, weighted by their own assignment `a[i,k]`).  Both sides of the
#: comparison are computed identically and never see a label, so the routing
#: key is guaranteed recoverable from `x` alone -- the thing `"decoder"` and
#: `"blind_decoder"` never guarantee.
QueryRoutingMode = Literal["decoder", "blind_decoder", "blind_similarity"]
QUERY_ROUTING_MODES = ("decoder", "blind_decoder", "blind_similarity")


@dataclass
class TableSlotState:
    table: torch.Tensor
    pooled_rows: torch.Tensor
    slots: torch.Tensor
    feature_attention: torch.Tensor
    support_attention: torch.Tensor


class TableSlotAdapter(nn.Module):
    """One or both competitive slot paths over a complete ``(B,R,C,E)`` table."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        *,
        num_slots: int = 4,
        num_iterations: int = 3,
        scope: SlotScope = "cell_and_data",
    ):
        super().__init__()
        if scope not in SLOT_SCOPES:
            raise ValueError(f"scope must be one of {SLOT_SCOPES}, got {scope!r}.")
        self.num_slots = num_slots
        self.scope = scope
        self.runs_cells = scope in ("cell_and_data", "cell")
        self.runs_data = scope in ("cell_and_data", "data")
        if self.runs_cells:
            self.feature_slots = SlotAttention(
                num_slots,
                embedding_size,
                hidden_size,
                num_iterations=num_iterations,
                competitive=True,
                # SCM episodes can have high-variance feature scales.  Keep sampled
                # slot seeds bounded while retaining learned, non-degenerate seeds.
                max_log_sigma=2.0,
            )
            self.feature_write = nn.Sequential(
                nn.Linear(embedding_size, embedding_size), nn.GELU(), nn.Linear(embedding_size, embedding_size)
            )
            self.feature_norm = nn.LayerNorm(embedding_size)
            # Convex blends start active: a zero residual gate would make this
            # pilot silently reduce to the ordinary backbone.
            self.feature_mix = nn.Parameter(torch.zeros(()))
        if self.runs_data:
            self.datapoint_slots = SlotAttention(
                num_slots,
                embedding_size,
                hidden_size,
                num_iterations=num_iterations,
                competitive=True,
                max_log_sigma=2.0,
            )
            self.row_write = nn.MultiheadAttention(embedding_size, 1, batch_first=True)
            self.row_norm = nn.LayerNorm(embedding_size)
            self.row_mix = nn.Parameter(torch.zeros(()))
        self.last_state: TableSlotState | None = None

    def feature_path(self, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the rewritten table, ``(B,R,C,K)`` attention and ``(B,R,K,E)`` slots."""
        if not self.runs_cells:
            raise RuntimeError(f"scope={self.scope!r} has no cell path.")
        b, r, c, e = table.shape
        slots, attention = self.feature_slots(table.reshape(b * r, c, e))
        attention = attention.reshape(b, r, c, self.num_slots)
        slots = slots.reshape(b, r, self.num_slots, e)
        reconstruction = torch.einsum("brcs,brse->brce", attention, slots)
        reconstruction = self.feature_norm(self.feature_write(reconstruction))
        mixed = (1 - self.feature_mix.sigmoid()) * table + self.feature_mix.sigmoid() * reconstruction
        return mixed, attention, slots

    def cell_state(
        self, table: torch.Tensor, split: int, attention: torch.Tensor, slots: torch.Tensor
    ) -> TableSlotState:
        """Read a cell-scope episode off the per-row cell competition.

        The decoder needs one slot set per episode and one distribution per
        support row; the cell path produces one of each per *row* and per
        *cell*.  Averaging the support rows' slots gives the former, and the
        mean of a row's cell assignments gives the latter -- still a
        distribution, because competitive attention sums to one over slots for
        every cell independently.
        """
        if not 1 <= split < table.shape[1]:
            raise ValueError("split must leave support and query rows.")
        state = TableSlotState(
            table,
            table.mean(dim=2),
            slots[:, :split].mean(dim=1),
            attention,
            attention[:, :split].mean(dim=2),
        )
        self.last_state = state
        return state

    def datapoint_path(self, table: torch.Tensor, split: int) -> TableSlotState:
        if not self.runs_data:
            raise RuntimeError(f"scope={self.scope!r} has no data path.")
        if not 1 <= split < table.shape[1]:
            raise ValueError("split must leave support and query rows.")
        # Mean is invariant to a permutation of columns and includes the padded
        # target token for query rows without ever revealing query labels.
        pooled = table.mean(dim=2)
        slots, attention = self.datapoint_slots(pooled[:, :split])
        reconstructed_rows = self.row_norm(self.row_write(pooled, slots, slots, need_weights=False)[0])
        mixed_rows = (1 - self.row_mix.sigmoid()) * pooled + self.row_mix.sigmoid() * reconstructed_rows
        mixed_table = (1 - self.row_mix.sigmoid()) * table + self.row_mix.sigmoid() * mixed_rows[:, :, None, :]
        state = TableSlotState(mixed_table, mixed_rows, slots, torch.empty(0, device=table.device), attention)
        self.last_state = state
        return state

    def forward(self, table: torch.Tensor, split: int) -> TableSlotState:
        if not self.runs_cells:
            state = self.datapoint_path(table, split)
            self.last_state = state
            return state
        adjusted, feature_attention, feature_slots = self.feature_path(table)
        if self.runs_data:
            state = self.datapoint_path(adjusted, split)
            state.feature_attention = feature_attention
        else:
            state = self.cell_state(adjusted, split, feature_attention, feature_slots)
        self.last_state = state
        return state


class TableSlotTransformerEncoderLayer(TransformerEncoderLayer):
    """Ordinary attention block with table slots at its two stage boundaries.

    Each stage boundary belongs to one scope, so a single-scope layer adapts at
    one boundary and passes the other through untouched.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        *,
        num_slots: int = 4,
        num_iterations: int = 3,
        scope: SlotScope = "cell_and_data",
    ):
        super().__init__(embedding_size, nhead, mlp_hidden_size)
        self.table_slots = TableSlotAdapter(
            embedding_size, mlp_hidden_size, num_slots=num_slots, num_iterations=num_iterations, scope=scope
        )
        self._split: int | None = None
        self._feature_attention: torch.Tensor | None = None
        self._feature_slots: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        layer: TransformerEncoderLayer,
        *,
        num_slots: int,
        num_slot_iterations: int,
        scope: SlotScope = "cell_and_data",
    ):
        adapted = cls(
            layer.norm1.normalized_shape[0],
            layer.self_attention_between_features.num_heads,
            layer.linear1.out_features,
            num_slots=num_slots,
            num_iterations=num_slot_iterations,
            scope=scope,
        )
        missing, unexpected = adapted.load_state_dict(layer.state_dict(), strict=False)
        if unexpected or any(not key.startswith("table_slots.") for key in missing):
            raise ValueError("Could not transfer ordinary transformer parameters to a table-slot layer.")
        return adapted

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        self._split = train_test_split_index
        return super().forward(src, train_test_split_index, num_mem_chunks)

    def adapt_after_feature_attention(self, src: torch.Tensor) -> torch.Tensor:
        if not self.table_slots.runs_cells:
            return src
        adjusted, attention, slots = self.table_slots.feature_path(src)
        self._feature_attention, self._feature_slots = attention, slots
        return adjusted

    def adapt_after_datapoint_attention(self, src: torch.Tensor) -> torch.Tensor:
        assert self._split is not None
        if self.table_slots.runs_data:
            state = self.table_slots.datapoint_path(src, self._split)
            if self._feature_attention is not None:
                state.feature_attention = self._feature_attention
        else:
            # The cell path already rewrote ``src`` at the earlier boundary, so
            # this reads the episode off it rather than adapting a second time.
            assert self._feature_attention is not None and self._feature_slots is not None
            state = self.table_slots.cell_state(src, self._split, self._feature_attention, self._feature_slots)
        self.table_slots.last_state = state
        return state.table


def install_table_slot_layers(
    backbone: NanoTabPFNModel,
    *,
    layer_indices: Sequence[int] = (3, 4, 5),
    num_slots: int = 4,
    num_slot_iterations: int = 3,
    scope: SlotScope = "cell_and_data",
) -> NanoTabPFNModel:
    selected = tuple(layer_indices)
    if not selected or any(index < 0 or index >= backbone.num_layers for index in selected):
        raise ValueError("table slot layer indices must identify backbone blocks.")
    for index in selected:
        if not isinstance(backbone.transformer_blocks[index], TableSlotTransformerEncoderLayer):
            backbone.transformer_blocks[index] = TableSlotTransformerEncoderLayer.from_pretrained(
                backbone.transformer_blocks[index],
                num_slots=num_slots,
                num_slot_iterations=num_slot_iterations,
                scope=scope,
            )
    return backbone


class TableSlotModel(nn.Module):
    """Common mixture decoder for head, in-backbone, and multi-tap table slots."""

    model_type = "multiregime_v2_table_slot"

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        *,
        mode: str,
        num_slots: int = 4,
        layer_indices: Sequence[int] = (3, 4, 5),
        num_slot_iterations: int = 3,
        max_classes: int = 2,
        scope: SlotScope = "cell_and_data",
        query_routing_mode: QueryRoutingMode = "decoder",
    ):
        super().__init__()
        if mode not in ("head", "backbone", "mufasa"):
            raise ValueError("mode must be head, backbone, or mufasa.")
        if scope not in SLOT_SCOPES:
            raise ValueError(f"scope must be one of {SLOT_SCOPES}, got {scope!r}.")
        if query_routing_mode not in QUERY_ROUTING_MODES:
            raise ValueError(f"query_routing_mode must be one of {QUERY_ROUTING_MODES}, got {query_routing_mode!r}.")
        if query_routing_mode != "decoder" and mode != "head":
            # Ignoring this would report a run nobody configured; the blind
            # pass this needs is only wired up for the head placement.
            raise ValueError(f"query_routing_mode={query_routing_mode!r} needs mode='head', not mode={mode!r}.")
        self.backbone, self.mode, self.num_slots, self.layer_indices = backbone, mode, num_slots, tuple(layer_indices)
        self.scope = scope
        self.query_routing_mode = query_routing_mode
        if mode == "head":
            self.adapters = nn.ModuleList(
                [
                    TableSlotAdapter(
                        backbone.embedding_size,
                        backbone.mlp_hidden_size,
                        num_slots=num_slots,
                        num_iterations=num_slot_iterations,
                        scope=scope,
                    )
                ]
            )
        elif mode == "backbone":
            install_table_slot_layers(
                backbone,
                layer_indices=self.layer_indices,
                num_slots=num_slots,
                num_slot_iterations=num_slot_iterations,
                scope=scope,
            )
            self.adapters = nn.ModuleList()
        else:
            self.adapters = nn.ModuleList(
                TableSlotAdapter(
                    backbone.embedding_size,
                    backbone.mlp_hidden_size,
                    num_slots=num_slots,
                    num_iterations=num_slot_iterations,
                    scope=scope,
                )
                for _ in self.layer_indices
            )
            self.slot_fusion = nn.Sequential(
                nn.Linear(2 * backbone.embedding_size, backbone.embedding_size),
                nn.GELU(),
                nn.Linear(backbone.embedding_size, backbone.embedding_size),
            )
            self.query_fusion = nn.Sequential(
                nn.Linear(2 * backbone.embedding_size, backbone.embedding_size),
                nn.GELU(),
                nn.Linear(backbone.embedding_size, backbone.embedding_size),
            )
            self.layer_logits = nn.Parameter(torch.zeros(len(self.layer_indices)))
        self.decoder = _SlotDecoder(backbone.embedding_size, backbone.mlp_hidden_size, max_classes)
        self.last_feature_attention: torch.Tensor | None = None
        self.last_support_attention: torch.Tensor | None = None
        self.last_support_attention_for_loss: torch.Tensor | None = None
        self.last_slots: torch.Tensor | None = None
        self.last_query_gates: torch.Tensor | None = None
        self.last_slot_utilization: torch.Tensor | None = None
        self.last_assignment_entropy: torch.Tensor | None = None
        #: How often the Hungarian matching fell back to the identity, and what
        #: the first such episode looked like.  Both are diagnostics: a run that
        #: leans on the fallback is not the run that was intended.
        self.alignment_fallbacks = 0
        self.first_alignment_fallback: dict[str, object] | None = None

    @staticmethod
    def _args(args, kwargs):
        if len(args) == 3:
            x = torch.cat((args[0], args[2]), 1)
            y = args[1]
            split = args[0].shape[1]
        elif len(args) == 1 and isinstance(args[0], tuple):
            x, y = args[0]
            split = kwargs.pop("train_test_split_index")
        else:
            raise TypeError("Expected support_x, support_y, query_x or concatenated source.")
        chunks = kwargs.pop("num_mem_chunks", 1)
        reconstruct = bool(kwargs.pop("reconstruct_support", False))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return x, y, int(split), chunks, reconstruct

    def _record_alignment_fallback(self, cost: np.ndarray, reference: TableSlotState, candidate: TableSlotState):
        """Characterize the first non-finite cost instead of only counting it.

        ``linear_sum_assignment`` rejects a non-finite matrix outright, which
        killed four runs 50-80% of the way through training with no record of
        what had gone wrong.  The details are captured once -- which of the two
        cost terms is bad, and whether the slots or the attention carried the
        NaN in -- so a recurrence is characterized rather than guessed at.
        """
        self.alignment_fallbacks += 1
        if self.first_alignment_fallback is not None:
            return
        self.first_alignment_fallback = {
            "fallbacks": self.alignment_fallbacks,
            "cost_nan": int(np.isnan(cost).sum()),
            "cost_inf": int(np.isinf(cost).sum()),
            "reference_slots_finite": bool(torch.isfinite(reference.slots).all()),
            "candidate_slots_finite": bool(torch.isfinite(candidate.slots).all()),
            "reference_attention_finite": bool(torch.isfinite(reference.support_attention).all()),
            "candidate_attention_finite": bool(torch.isfinite(candidate.support_attention).all()),
            "reference_slot_absmax": float(reference.slots.detach().abs().max()),
            "candidate_slot_absmax": float(candidate.slots.detach().abs().max()),
        }

    def _align(self, reference: TableSlotState, candidate: TableSlotState) -> torch.Tensor:
        indices = []
        for b in range(reference.slots.shape[0]):
            slot_cost = (
                1
                - F.normalize(reference.slots[b].detach(), dim=-1) @ F.normalize(candidate.slots[b].detach(), dim=-1).T
            )
            assignment_cost = (
                torch.cdist(
                    reference.support_attention[b].detach().T,
                    candidate.support_attention[b].detach().T,
                    p=2,
                )
                / max(1, reference.support_attention.shape[1]) ** 0.5
            )
            cost = (slot_cost + 0.5 * assignment_cost).cpu().numpy()
            if not np.isfinite(cost).all():
                # Leaving this episode's slots in the order they were produced
                # is the identity assignment, which is what the matching would
                # return for an already-aligned pair.  One bad episode must not
                # end a run that is otherwise training normally.
                self._record_alignment_fallback(cost, reference, candidate)
                indices.append(torch.arange(cost.shape[0], device=candidate.slots.device))
                continue
            rows, cols = linear_sum_assignment(cost)
            perm = np.empty(len(rows), dtype=np.int64)
            perm[rows] = cols
            indices.append(torch.as_tensor(perm, device=candidate.slots.device))
        return torch.stack(indices)

    def _record(self, state: TableSlotState, log_gate: torch.Tensor):
        self.last_feature_attention = state.feature_attention
        self.last_support_attention = state.support_attention.detach()
        self.last_support_attention_for_loss = state.support_attention
        self.last_slots = state.slots
        self.last_query_gates = log_gate.exp()
        self.last_slot_utilization = state.support_attention.mean((0, 1))
        self.last_assignment_entropy = (
            -(state.support_attention * state.support_attention.clamp_min(1e-12).log()).sum(-1).mean()
        )

    def _blind_pass(
        self, x: torch.Tensor, y: torch.Tensor, split: int, chunks: int, state: TableSlotState
    ) -> TableSlotState:
        """Re-encode the table with every target cell holding the mean support label.

        Slot Attention's decoder rebuilds a pixel from a slot plus that pixel's
        *position* -- never from the pixel's own colour, or reconstruction would
        be trivial.  The tabular analogue of position is the row's features with
        its label withheld, so this runs a second backbone pass in which every
        target cell holds the episode's mean support label.

        ``TargetEncoder`` already pads non-support target cells with exactly
        that mean, so passing a constant ``y`` makes the whole target column one
        value and the pass is invariant to any relabelling preserving the mean.
        No masking machinery is needed and none is added.  Query rows get the
        same treatment as support rows here, unlike the labelled pass, where a
        query row's own target cell is already the mean but its *attended*
        state still picks up real label information from the labelled support
        rows around it -- this pass has no label anywhere in the table for
        anything to pick up.

        Shared by support reconstruction and the two blind query-routing
        modes, so a run using both pays for this pass once, not twice.
        """
        blind_y = y.float()
        blind_y = blind_y.mean(dim=1, keepdim=True).expand_as(blind_y)
        blind_state = self.adapters[0](self.backbone.encode_table((x, blind_y), split, chunks), split)
        # The blind pass must not own the diagnostics ``_record`` reads back.
        self.adapters[0].last_state = state
        return blind_state

    def _reconstruct_support(self, state: TableSlotState, blind_state: TableSlotState, split: int) -> torch.Tensor:
        """``(B,S,C)`` log probabilities of every support label under its own assignment.

        The slots and the assignment come from the *labelled* pass: only the row
        representation is blinded.  ``a[i,k]`` is the mixture weight, so the
        decoder's own mask channel is deliberately discarded here.
        """
        support_logits, _ = self.decoder(blind_state.pooled_rows[:, :split], state.slots)
        log_assignment = state.support_attention.clamp_min(1e-12).log()
        return torch.logsumexp(log_assignment[..., None] + F.log_softmax(support_logits, -1), dim=2)

    def _similarity_gate(self, state: TableSlotState, blind_state: TableSlotState, split: int) -> torch.Tensor:
        """``(B,Q,K)`` log routing weights from blind-embedding cosine similarity.

        Each slot's centroid is the assignment-weighted mean of the *blind*
        support embeddings it claimed -- computed the same way a support row's
        own reconstruction target is, so a query is compared against exactly
        the representation its own class-mates would have produced.  Both
        sides of the comparison are blind, so nothing here can key off a label
        that was never available to the query in the first place.
        """
        support_blind = blind_state.pooled_rows[:, :split]  # (B,S,E)
        weights = state.support_attention  # (B,S,K)
        centroids = torch.einsum("bsk,bse->bke", weights, support_blind)
        centroids = centroids / weights.sum(dim=1).clamp_min(1e-6)[..., None]  # (B,K,E)
        query_blind = F.normalize(blind_state.pooled_rows[:, split:], dim=-1)  # (B,Q,E)
        centroids = F.normalize(centroids, dim=-1)
        similarity = torch.einsum("bqe,bke->bqk", query_blind, centroids)
        return F.log_softmax(similarity, dim=-1)

    def forward(self, *args, **kwargs) -> SlotRegimePrediction:
        x, y, split, chunks, reconstruct = self._args(args, kwargs)
        if reconstruct and self.mode != "head":
            # Ignoring the flag would report a run nobody configured; the
            # reconstruction pilot is scoped to the head placement.
            raise ValueError(f"reconstruct_support is a head-mode setting, not mode={self.mode!r}.")
        if self.mode == "head":
            state = self.adapters[0](self.backbone.encode_table((x, y.float()), split, chunks), split)
        elif self.mode == "backbone":
            encoded = self.backbone.encode_table((x, y.float()), split, chunks)
            layer = next(
                layer
                for layer in reversed(self.backbone.transformer_blocks)
                if isinstance(layer, TableSlotTransformerEncoderLayer)
            )
            state = layer.table_slots.last_state
            assert state is not None
            # The final MLP operates after the adapter, so use its feature-pooled
            # representation for the decoder while retaining that layer's slots.
            state = TableSlotState(
                encoded, encoded.mean(2), state.slots, state.feature_attention, state.support_attention
            )
        else:
            xenc = self.backbone.feature_encoder(x, split)
            yenc = self.backbone.target_encoder(y.float().unsqueeze(-1) if y.ndim == 2 else y.float(), xenc.shape[1])
            table = torch.cat((xenc, yenc), 2)
            states = []
            for index, block in enumerate(self.backbone.transformer_blocks):
                table = block(table, split, chunks)
                if index in self.layer_indices:
                    states.append(self.adapters[len(states)](table, split))
            ref = states[-1]
            aligned = []
            for state in states:
                permutation = self._align(ref, state)
                aligned.append(
                    TableSlotState(
                        state.table,
                        state.pooled_rows,
                        torch.stack(
                            [state.slots[b].index_select(0, permutation[b]) for b in range(permutation.shape[0])]
                        ),
                        state.feature_attention,
                        torch.stack(
                            [
                                state.support_attention[b].index_select(-1, permutation[b])
                                for b in range(permutation.shape[0])
                            ]
                        ),
                    )
                )
            weights = self.layer_logits.softmax(0)
            slots = torch.stack([s.slots for s in aligned])
            queries = torch.stack([s.pooled_rows[:, split:] for s in aligned])
            attentions = torch.stack([s.support_attention for s in aligned])
            state = TableSlotState(
                ref.table,
                self.query_fusion(
                    torch.cat(((weights[:, None, None, None] * queries).sum(0), queries.max(0).values), -1)
                ),
                self.slot_fusion(torch.cat(((weights[:, None, None, None] * slots).sum(0), slots.max(0).values), -1)),
                ref.feature_attention,
                (weights[:, None, None, None] * attentions).sum(0),
            )
        # The blind pass is needed for support reconstruction and for either
        # blind query-routing mode; computed once and shared, so a run using
        # more than one of these still pays for it only once.
        blind_state = None
        if reconstruct or self.query_routing_mode != "decoder":
            blind_state = self._blind_pass(x, y, split, chunks, state)

        # ``logits`` always comes from the labelled pass: a query row's own
        # target cell is masked, but its attended state still carries real
        # label signal from the labelled support rows around it, and that
        # in-context signal is exactly what makes prediction work. Only the
        # gate -- the piece that must not be able to key off a leaked label
        # -- switches to the blind embedding.
        query = state.pooled_rows if self.mode == "mufasa" else state.pooled_rows[:, split:]
        logits, masks = self.decoder(query, state.slots)
        if self.query_routing_mode == "blind_decoder":
            _, blind_masks = self.decoder(blind_state.pooled_rows[:, split:], state.slots)
            log_gate = F.log_softmax(blind_masks, -1)
        elif self.query_routing_mode == "blind_similarity":
            log_gate = self._similarity_gate(state, blind_state, split)
        else:
            log_gate = F.log_softmax(masks, -1)
        # Runs only when asked, so ordinary inference costs exactly what it did.
        reconstruction = self._reconstruct_support(state, blind_state, split) if reconstruct else None
        self._record(state, log_gate)
        return SlotRegimePrediction(logits, log_gate, state.support_attention, reconstruction)


__all__ = [
    "QUERY_ROUTING_MODES",
    "QueryRoutingMode",
    "SLOT_SCOPES",
    "SlotScope",
    "TableSlotAdapter",
    "TableSlotModel",
    "TableSlotState",
    "TableSlotTransformerEncoderLayer",
    "install_table_slot_layers",
]
