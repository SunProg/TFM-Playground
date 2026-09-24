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

#: Which axial NanoTabPFN attention stages are replaced by Slot Attention.
#: ``"both"`` is the actual slot-only information route: query labels can be
#: predicted from labelled support rows only through the per-column data slots.
AttentionReplacement = Literal["none", "feature", "datapoint", "both"]
ATTENTION_REPLACEMENTS = ("none", "feature", "datapoint", "both")

#: How a query row's routing to slots is computed.
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
#:
#: ``"tabpfn_attention"`` -- reuse the backbone's actual query-to-support
#: attention map.  Each query-support weight is multiplied by the support
#: row's slot assignment, and the resulting slot-specific support value is
#: fused into that slot before decoding.  This keeps TabPFN's labelled query
#: pathway while making support retrieval visible to the slot experts.
#:
#: ``"posterior_attention"`` -- compute each support row's posterior
#: ``rho[i,c,k] ∝ a[i,c,k] p(y_i | h_i^blind, s_k)`` and use the real
#: per-feature TabPFN query-to-support attention to form
#: ``g[q,k] = mean_c sum_i A[q,i,c] rho[i,c,k]``.
#:
#: ``"direct_slot"`` -- in ``table_slot_backbone`` only, each adapted layer
#: lets every query cell address the support-derived slots directly with the
#: same Slot Attention key/query projections.  The resulting slot read is
#: written back to that query cell, and the deepest such assignment is the
#: prediction mixture gate.  It deliberately does not reuse a final-backbone
#: query-to-support attention map.
QueryRoutingMode = Literal[
    "decoder", "blind_decoder", "blind_similarity", "tabpfn_attention", "posterior_attention", "direct_slot"
]
QUERY_ROUTING_MODES = (
    "decoder", "blind_decoder", "blind_similarity", "tabpfn_attention", "posterior_attention", "direct_slot"
)

# How the feature-axis and datapoint-axis competitions are exposed to the
# prediction head.  ``shared`` is the historical row-pooled head.  The
# factorized path keeps the two axes separate until query decoding and is the
# direct analogue of NanoTabPFN's (B*R,C,E) and (B*C,R,E) attention views.
SlotComposition = Literal["shared", "factorized"]
SLOT_COMPOSITIONS = ("shared", "factorized")

#: What weights a slot's contribution when the support labels are reconstructed,
#: head mode only.
#:
#: ``"attention"`` -- the historical design: ``a[i,k]``, slot attention's own
#: assignment of support row ``i``, is the mixture weight, and the decoder's
#: mask channel is discarded.  The query side gates on that discarded channel
#: instead, so the two sides route by different quantities and only one of them
#: is trained.
#:
#: ``"alpha"`` -- Locatello's own compositing rule: the decoder emits content
#: *and* an unnormalized alpha channel from one pathway, the alphas are
#: softmaxed across slots, and that softmax is the mixture weight.  The paper
#: has exactly one routing quantity and the reconstruction trains it directly.
#: Here that makes ``L_rec`` and the query mixture the same expression --
#: ``decoder(blind row, slots)``, softmax the mask over slots, composite --
#: evaluated on rows that do and do not carry a target.  Pair it with
#: ``query_routing_mode="blind_decoder"`` so both sides read the same
#: label-blind embedding; with ``"decoder"`` the query alpha is still computed
#: from the labelled pass and only the weight is shared.
ReconstructionMixture = Literal["attention", "alpha"]
RECONSTRUCTION_MIXTURES = ("attention", "alpha")


@dataclass
class TableSlotState:
    table: torch.Tensor
    pooled_rows: torch.Tensor
    slots: torch.Tensor
    feature_attention: torch.Tensor
    support_attention: torch.Tensor
    # Per-row feature slots are retained when the cell competition runs.  The
    # old head did not need them after rewriting ``table``; the factorized head
    # uses them together with datapoint slots instead of pooling columns.
    feature_slots: torch.Tensor | None = None
    # Column-wise datapoint competition, before any column aggregation:
    # ``(B,C,K,E)`` slots and ``(B,S,C,K)`` support assignments.
    datapoint_slots_by_column: torch.Tensor | None = None
    support_attention_by_column: torch.Tensor | None = None
    # Pre-write-back cells consumed by datapoint Slot Attention.  Cross-fitted
    # held-out rows are assigned using this representation, whose target cell
    # contains only the context mean rather than the held-out label.
    datapoint_input_table: torch.Tensor | None = None
    # Query-cell-to-slot assignments from the data path after column alignment:
    # ``(B,Q,C,K)``.  Present only for direct in-backbone query slot reads.
    query_slot_attention_by_column: torch.Tensor | None = None


def row_positions(count: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    """Deterministic addresses; contains no row features or labels."""
    position = torch.arange(count, device=reference.device, dtype=torch.float32)[:, None]
    frequency = torch.exp(-torch.arange(0, width, 2, device=reference.device).float() * (9.210340372 / width))
    phase = position * frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[:, :width].to(reference.dtype)[None]


class TableSlotAdapter(nn.Module):
    """One or both competitive slot paths over a complete ``(B,R,C,E)`` table."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        *,
        num_slots: int = 4,
        num_iterations: int = 3,
        num_heads: int = 4,
        scope: SlotScope = "cell_and_data",
        direct_query_slot: bool = False,
        attention_replacement: AttentionReplacement = "none",
        preserve_cells: bool = False,
    ):
        super().__init__()
        if scope not in SLOT_SCOPES:
            raise ValueError(f"scope must be one of {SLOT_SCOPES}, got {scope!r}.")
        self.num_slots = num_slots
        self.scope = scope
        self.preserve_cells = preserve_cells
        self.direct_query_slot = direct_query_slot
        if attention_replacement not in ATTENTION_REPLACEMENTS:
            raise ValueError(
                f"attention_replacement must be one of {ATTENTION_REPLACEMENTS}, got {attention_replacement!r}."
            )
        self.attention_replacement = attention_replacement
        self.replaces_feature_attention = attention_replacement in ("feature", "both")
        self.replaces_datapoint_attention = attention_replacement in ("datapoint", "both")
        self.runs_cells = scope in ("cell_and_data", "cell")
        self.runs_data = scope in ("cell_and_data", "data")
        if direct_query_slot and not self.runs_data:
            raise ValueError("A direct query-slot read needs the datapoint slot path.")
        if direct_query_slot and self.replaces_datapoint_attention:
            raise ValueError("direct_query_slot and datapoint-attention replacement are alternative query paths.")
        if self.replaces_feature_attention and not self.runs_cells:
            raise ValueError("feature-attention replacement needs cell or cell_and_data scope.")
        if self.replaces_datapoint_attention and not self.runs_data:
            raise ValueError("datapoint-attention replacement needs data or cell_and_data scope.")
        if self.runs_cells:
            self.feature_slots = SlotAttention(
                num_slots,
                embedding_size,
                hidden_size,
                num_iterations=num_iterations,
                num_heads=num_heads,
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
                num_heads=num_heads,
                competitive=True,
                max_log_sigma=2.0,
            )
            if direct_query_slot:
                # A query cell reads the slots that were formed from support
                # cells in its own feature column.  This is deliberately a
                # cellwise residual, not the historical pooled row rewrite.
                self.direct_query_slot_write = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.GELU(),
                    nn.Linear(embedding_size, embedding_size),
                )
                self.direct_query_slot_norm = nn.LayerNorm(embedding_size)
                self.direct_query_slot_mix = nn.Parameter(torch.zeros(()))
            elif self.replaces_datapoint_attention:
                # This replaces NanoTabPFN's support self-attention and its
                # query-to-support cross-attention.  Both support and query
                # cells read the same support-derived slot set, so the latter
                # has no labelled support path that bypasses Slot Attention.
                self.datapoint_slot_write = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.GELU(),
                    nn.Linear(embedding_size, embedding_size),
                )
                self.datapoint_slot_mix = nn.Parameter(torch.zeros(()))
            elif not preserve_cells:
                self.row_write = nn.MultiheadAttention(embedding_size, 1, batch_first=True)
                self.row_norm = nn.LayerNorm(embedding_size)
                self.row_mix = nn.Parameter(torch.zeros(()))
        self.embedding_reconstruction = False
        self.last_embedding_target: torch.Tensor | None = None
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

    def replace_feature_attention(self, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace feature self-attention with a per-row slot read.

        Each row's ``C`` cells compete for ``K`` slots.  Broadcasting the
        slots back to the cells is the Slot Attention analogue of a feature
        attention value read; the residual is the same identity path the
        original Transformer block retains.
        """
        if not self.replaces_feature_attention:
            raise RuntimeError("feature attention is not configured for replacement.")
        batch, rows, columns, width = table.shape
        slots, attention = self.feature_slots(table.reshape(batch * rows, columns, width))
        attention = attention.reshape(batch, rows, columns, self.num_slots)
        slots = slots.reshape(batch, rows, self.num_slots, width)
        read = torch.einsum("brck,brke->brce", attention, slots)
        residual = self.feature_write(read)
        return table + self.feature_mix.sigmoid() * residual, attention, slots

    @staticmethod
    def _align_datapoint_columns(
        slots: torch.Tensor,
        attention: torch.Tensor,
        query_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align anonymous per-column slots before exposing a shared slot axis.

        Each feature-column sequence samples its own exchangeable slot seeds.
        Averaging raw index ``k`` across columns would consequently mix
        unrelated experts.  Match every column to column zero using both slot
        geometry and its support-row assignments.  The selected permutations
        are detached decisions; the selected slot tensors keep gradients.
        """
        batch, columns, count, _width = slots.shape
        if query_attention is not None and query_attention.shape != (
            batch,
            query_attention.shape[1],
            columns,
            count,
        ):
            raise ValueError("query attention must have shape (batch, query rows, columns, slots).")
        if columns == 1 or count == 1:
            if query_attention is None:
                return slots, attention
            return slots, attention, query_attention
        permutations: list[torch.Tensor] = []
        for batch_index in range(batch):
            reference_slots = F.normalize(slots[batch_index, 0].detach(), dim=-1)
            reference_attention = attention[batch_index, :, 0].detach().T
            per_column = [torch.arange(count, device=slots.device)]
            for column in range(1, columns):
                candidate_slots = F.normalize(slots[batch_index, column].detach(), dim=-1)
                slot_cost = 1 - reference_slots @ candidate_slots.T
                assignment_cost = torch.cdist(
                    reference_attention, attention[batch_index, :, column].detach().T, p=2
                ) / max(1, attention.shape[1]) ** 0.5
                cost = (slot_cost + 0.5 * assignment_cost).cpu().numpy()
                if not np.isfinite(cost).all():
                    per_column.append(torch.arange(count, device=slots.device))
                    continue
                rows, column_indices = linear_sum_assignment(cost)
                permutation = np.empty(len(rows), dtype=np.int64)
                permutation[rows] = column_indices
                per_column.append(torch.as_tensor(permutation, device=slots.device))
            permutations.append(torch.stack(per_column))
        permutation = torch.stack(permutations)
        aligned_slots = torch.gather(slots, 2, permutation[..., None].expand_as(slots))
        per_column_attention = attention.permute(0, 2, 1, 3)
        aligned_attention = torch.gather(
            per_column_attention, 3, permutation[:, :, None, :].expand_as(per_column_attention)
        ).permute(0, 2, 1, 3)
        if query_attention is None:
            return aligned_slots, aligned_attention
        per_column_query = query_attention.permute(0, 2, 1, 3)
        aligned_query = torch.gather(
            per_column_query, 3, permutation[:, :, None, :].expand_as(per_column_query)
        ).permute(0, 2, 1, 3)
        return aligned_slots, aligned_attention, aligned_query

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
            slots,
        )
        self.last_state = state
        return state

    def datapoint_path(self, table: torch.Tensor, split: int) -> TableSlotState:
        if not self.runs_data:
            raise RuntimeError(f"scope={self.scope!r} has no data path.")
        if not 1 <= split < table.shape[1]:
            raise ValueError("split must leave support and query rows.")
        batch, rows, columns, width = table.shape
        table_view = table.permute(0, 2, 1, 3).reshape(batch * columns, rows, width)
        support_view = table_view[:, :split]
        # Standard Slot Attention binds support cells to slots.  Its attention
        # query is therefore the slots themselves; query-to-support retrieval
        # remains the backbone's TabPFN attention and is composed with these
        # assignments by the prediction gate.
        self.last_embedding_target = table[:, :split, -1, :].detach() if self.embedding_reconstruction else None
        addressed = support_view
        if self.embedding_reconstruction:
            positions = row_positions(split, width, support_view).expand(batch, columns, split, width)
            addressed = addressed + positions.reshape(batch * columns, split, width)
        column_slots, column_attention = self.datapoint_slots(addressed)
        query_attention_by_column = None
        if self.direct_query_slot:
            query_view = table_view[:, split:]
            query_attention_by_column = self.datapoint_slots.assignment(query_view, column_slots)
            query_attention_by_column = query_attention_by_column.reshape(
                batch, columns, rows - split, self.num_slots
            ).permute(0, 2, 1, 3)
        # Match NanoTabPFN's datapoint-attention view: each feature column is a
        # separate row sequence, so data slots see ``(B*C,S,E)`` rather than a
        # column-mean row token.  Any aggregation happens only after this
        # competition, for the legacy row-rewrite state.
        # The reconstruction target is the support row's target-column token,
        # which is the representation the normal TabPFN decoder consumes.  It
        # must not be the unconditional mean over feature columns: that would
        # ask the slots to reconstruct a pooled table row rather than the
        # support label embedding.
        datapoint_slots_by_column = column_slots.reshape(batch, columns, self.num_slots, width)
        support_attention_by_column = column_attention.reshape(batch, columns, split, self.num_slots).permute(
            0, 2, 1, 3
        )
        if query_attention_by_column is None:
            datapoint_slots_by_column, support_attention_by_column = self._align_datapoint_columns(
                datapoint_slots_by_column, support_attention_by_column
            )
        else:
            (
                datapoint_slots_by_column,
                support_attention_by_column,
                query_attention_by_column,
            ) = self._align_datapoint_columns(
                datapoint_slots_by_column, support_attention_by_column, query_attention_by_column
            )
        # The shared/head interface still exposes one slot set and one support
        # assignment per row.  These are column means *after* competition;
        # factorized decoding reads the retained column-wise tensors above.
        slots = datapoint_slots_by_column.mean(dim=1)
        attention = support_attention_by_column.mean(dim=2)
        pooled = table.mean(dim=2)
        if self.direct_query_slot:
            query_read = torch.einsum(
                "bqck,bcke->bqce", query_attention_by_column, datapoint_slots_by_column
            )
            query_residual = self.direct_query_slot_norm(self.direct_query_slot_write(query_read))
            mixed_table = table.clone()
            mixed_table[:, split:] = table[:, split:] + self.direct_query_slot_mix.sigmoid() * query_residual
            mixed_rows = mixed_table.mean(dim=2)
        elif self.preserve_cells:
            # Factorized composition must not reintroduce the pooled row through
            # the historical slot write-back.  The per-column competition has
            # already happened; keep the cell table intact for target-token and
            # support-value reads.
            mixed_rows = pooled
            mixed_table = table
        else:
            reconstructed_rows = self.row_norm(self.row_write(pooled, slots, slots, need_weights=False)[0])
            mixed_rows = (1 - self.row_mix.sigmoid()) * pooled + self.row_mix.sigmoid() * reconstructed_rows
            mixed_table = (1 - self.row_mix.sigmoid()) * table + self.row_mix.sigmoid() * mixed_rows[
                :, :, None, :
            ]
        state = TableSlotState(
            mixed_table,
            mixed_rows,
            slots,
            torch.empty(0, device=table.device),
            attention,
            datapoint_slots_by_column=datapoint_slots_by_column,
            support_attention_by_column=support_attention_by_column,
            datapoint_input_table=table,
            query_slot_attention_by_column=query_attention_by_column,
        )
        self.last_state = state
        return state

    def replace_datapoint_attention(self, table: torch.Tensor, split: int) -> TableSlotState:
        """Replace the per-column data attention with support-derived slots.

        The original block first lets support rows attend to support rows, then
        lets every query row attend to those support states.  Here one Slot
        Attention pass constructs slots from the support cells in each column.
        Both support and query cells then read those slots with the same
        slot-competition compatibility.  Consequently a query can receive a
        support label only through these slots.
        """
        if not self.replaces_datapoint_attention:
            raise RuntimeError("datapoint attention is not configured for replacement.")
        if not 1 <= split < table.shape[1]:
            raise ValueError("split must leave support and query rows.")
        batch, rows, columns, width = table.shape
        table_view = table.permute(0, 2, 1, 3).reshape(batch * columns, rows, width)
        support_view, query_view = table_view[:, :split], table_view[:, split:]
        column_slots, support_assignment = self.datapoint_slots(support_view)
        query_assignment = self.datapoint_slots.assignment(query_view, column_slots)

        slots_by_column = column_slots.reshape(batch, columns, self.num_slots, width)
        support_by_column = support_assignment.reshape(batch, columns, split, self.num_slots).permute(0, 2, 1, 3)
        query_by_column = query_assignment.reshape(batch, columns, rows - split, self.num_slots).permute(0, 2, 1, 3)
        slots_by_column, support_by_column, query_by_column = self._align_datapoint_columns(
            slots_by_column, support_by_column, query_by_column
        )

        support_read = torch.einsum("bsck,bcke->bsce", support_by_column, slots_by_column)
        query_read = torch.einsum("bqck,bcke->bqce", query_by_column, slots_by_column)
        read = torch.cat((support_read, query_read), dim=1)
        residual = self.datapoint_slot_write(read)
        mixed_table = table + self.datapoint_slot_mix.sigmoid() * residual
        state = TableSlotState(
            mixed_table,
            mixed_table.mean(dim=2),
            slots_by_column.mean(dim=1),
            torch.empty(0, device=table.device),
            support_by_column.mean(dim=2),
            datapoint_slots_by_column=slots_by_column,
            support_attention_by_column=support_by_column,
            datapoint_input_table=table,
            query_slot_attention_by_column=query_by_column,
        )
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
            state.feature_slots = feature_slots
        else:
            state = self.cell_state(adjusted, split, feature_attention, feature_slots)
        self.last_state = state
        return state


class TableSlotTransformerEncoderLayer(TransformerEncoderLayer):
    """Axial attention block with optional table-slot replacements.

    A slot path can either adapt an ordinary attention output at a stage
    boundary (the historical experiment) or replace that attention stage.  The
    latter is used to test whether slots can carry the support-to-query path,
    rather than merely decorate an already-solved attention computation.
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        *,
        num_slots: int = 4,
        num_iterations: int = 3,
        num_heads: int = 4,
        scope: SlotScope = "cell_and_data",
        direct_query_slot: bool = False,
        attention_replacement: AttentionReplacement = "none",
    ):
        super().__init__(embedding_size, nhead, mlp_hidden_size)
        self.table_slots = TableSlotAdapter(
            embedding_size,
            mlp_hidden_size,
            num_slots=num_slots,
            num_iterations=num_iterations,
            num_heads=num_heads,
            scope=scope,
            direct_query_slot=direct_query_slot,
            attention_replacement=attention_replacement,
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
        num_heads: int = 4,
        scope: SlotScope = "cell_and_data",
        direct_query_slot: bool = False,
        attention_replacement: AttentionReplacement = "none",
    ):
        adapted = cls(
            layer.norm1.normalized_shape[0],
            layer.self_attention_between_features.num_heads,
            layer.linear1.out_features,
            num_slots=num_slots,
            num_iterations=num_slot_iterations,
            num_heads=num_heads,
            scope=scope,
            direct_query_slot=direct_query_slot,
            attention_replacement=attention_replacement,
        )
        missing, unexpected = adapted.load_state_dict(layer.state_dict(), strict=False)
        allowed_missing = ("table_slots.",)
        if unexpected or any(not key.startswith(allowed_missing) for key in missing):
            raise ValueError("Could not transfer ordinary transformer parameters to a table-slot layer.")
        return adapted

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        self._split = train_test_split_index
        self._feature_attention = None
        self._feature_slots = None
        return super().forward(src, train_test_split_index, num_mem_chunks)

    def feature_attention_stage(self, src: torch.Tensor, num_mem_chunks: int = 1) -> torch.Tensor:
        if not self.table_slots.replaces_feature_attention:
            return super().feature_attention_stage(src, num_mem_chunks)
        adjusted, attention, slots = self.table_slots.replace_feature_attention(src)
        self._feature_attention, self._feature_slots = attention, slots
        # Keep NanoTabPFN's feature-stage normalization; only its attention
        # value read has been replaced.
        return self.norm1(adjusted)

    def datapoint_attention_stage(
        self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1
    ) -> torch.Tensor:
        if not self.table_slots.replaces_datapoint_attention:
            return super().datapoint_attention_stage(src, train_test_split_index, num_mem_chunks)
        state = self.table_slots.replace_datapoint_attention(src, train_test_split_index)
        if self._feature_attention is not None:
            state.feature_attention = self._feature_attention
            state.feature_slots = self._feature_slots
        self.table_slots.last_state = state
        return state.table

    def adapt_after_feature_attention(self, src: torch.Tensor) -> torch.Tensor:
        if self.table_slots.replaces_feature_attention:
            return src
        if not self.table_slots.runs_cells:
            return src
        adjusted, attention, slots = self.table_slots.feature_path(src)
        self._feature_attention, self._feature_slots = attention, slots
        return adjusted

    def adapt_after_datapoint_attention(self, src: torch.Tensor) -> torch.Tensor:
        assert self._split is not None
        if self.table_slots.replaces_datapoint_attention:
            # `datapoint_attention_stage` has already made the state and
            # returned its table, so a second pass would duplicate the slot
            # read and reintroduce an unintended adapter.
            return src
        if self.table_slots.runs_data:
            state = self.table_slots.datapoint_path(src, self._split)
            if self._feature_attention is not None:
                state.feature_attention = self._feature_attention
                state.feature_slots = self._feature_slots
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
    num_heads: int = 4,
    scope: SlotScope = "cell_and_data",
    direct_query_slot: bool = False,
    attention_replacement: AttentionReplacement = "none",
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
                num_heads=num_heads,
                scope=scope,
                direct_query_slot=direct_query_slot,
                attention_replacement=attention_replacement,
            )
    return backbone


def collect_table_slot_state(backbone: NanoTabPFNModel) -> TableSlotState | None:
    """Return the state from the deepest table-slot layer that ran.

    Replacement models use NanoTabPFN's native decoder, so they do not emit a
    ``SlotRegimePrediction``.  Keeping this small read-only helper lets their
    support and query assignments be evaluated without adding a second decoder
    or a mixture gate to the prediction path.
    """
    state = None
    for layer in backbone.transformer_blocks:
        if isinstance(layer, TableSlotTransformerEncoderLayer) and layer.table_slots.last_state is not None:
            state = layer.table_slots.last_state
    return state


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
        num_slot_heads: int = 4,
        max_classes: int = 2,
        scope: SlotScope = "cell_and_data",
        query_routing_mode: QueryRoutingMode = "decoder",
        reconstruction_mixture: ReconstructionMixture = "attention",
        embedding_reconstruction: bool = False,
        query_content_mode: str = "labelled",
        decoder_interaction: str = "full",
        slot_composition: SlotComposition = "shared",
    ):
        super().__init__()
        if mode not in ("head", "backbone", "mufasa"):
            raise ValueError("mode must be head, backbone, or mufasa.")
        if scope not in SLOT_SCOPES:
            raise ValueError(f"scope must be one of {SLOT_SCOPES}, got {scope!r}.")
        if query_routing_mode not in QUERY_ROUTING_MODES:
            raise ValueError(f"query_routing_mode must be one of {QUERY_ROUTING_MODES}, got {query_routing_mode!r}.")
        if query_routing_mode == "direct_slot":
            if mode != "backbone":
                raise ValueError("query_routing_mode='direct_slot' needs mode='backbone'.")
            if scope == "cell":
                raise ValueError("query_routing_mode='direct_slot' needs data or cell_and_data scope.")
        elif query_routing_mode != "decoder" and mode != "head":
            # Ignoring this would report a run nobody configured; the blind
            # pass this needs is only wired up for the head placement.
            raise ValueError(f"query_routing_mode={query_routing_mode!r} needs mode='head', not mode={mode!r}.")
        if reconstruction_mixture not in RECONSTRUCTION_MIXTURES:
            raise ValueError(
                f"reconstruction_mixture must be one of {RECONSTRUCTION_MIXTURES}, got {reconstruction_mixture!r}."
            )
        if reconstruction_mixture != "attention" and mode != "head":
            # Same reasoning as the routing guard: the reconstruction this
            # weights is only wired up for the head placement, so accepting the
            # setting elsewhere would report a run nobody configured.
            raise ValueError(f"reconstruction_mixture={reconstruction_mixture!r} needs mode='head', not mode={mode!r}.")
        self.backbone, self.mode, self.num_slots, self.layer_indices = backbone, mode, num_slots, tuple(layer_indices)
        self.scope = scope
        self.query_routing_mode = query_routing_mode
        if query_content_mode not in ("labelled", "blind"):
            raise ValueError("query_content_mode must be labelled or blind.")
        if query_content_mode == "blind" and mode != "head":
            raise ValueError("Blind query content requires head mode.")
        self.query_content_mode = query_content_mode
        self.reconstruction_mixture = reconstruction_mixture
        if decoder_interaction not in ("full", "product"):
            raise ValueError("decoder_interaction must be 'full' or 'product'.")
        self.decoder_interaction = decoder_interaction
        if slot_composition not in SLOT_COMPOSITIONS:
            raise ValueError(f"slot_composition must be one of {SLOT_COMPOSITIONS}, got {slot_composition!r}.")
        if slot_composition == "factorized" and mode != "head":
            raise ValueError("slot_composition='factorized' currently requires mode='head'.")
        if slot_composition == "factorized" and scope != "cell_and_data":
            raise ValueError("slot_composition='factorized' requires scope='cell_and_data'.")
        if slot_composition == "factorized" and query_routing_mode != "tabpfn_attention":
            raise ValueError("slot_composition='factorized' requires query_routing_mode='tabpfn_attention'.")
        if query_routing_mode == "posterior_attention" and scope == "cell":
            raise ValueError("query_routing_mode='posterior_attention' requires data or cell_and_data scope.")
        self.slot_composition = slot_composition
        if embedding_reconstruction and (mode != "head" or scope == "cell"):
            raise ValueError("Embedding reconstruction requires head mode with data or cell_and_data scope.")
        self.embedding_reconstruction = embedding_reconstruction
        if mode == "head":
            self.adapters = nn.ModuleList(
                [
                    TableSlotAdapter(
                        backbone.embedding_size,
                        backbone.mlp_hidden_size,
                        num_slots=num_slots,
                        num_iterations=num_slot_iterations,
                        num_heads=num_slot_heads,
                        scope=scope,
                        preserve_cells=slot_composition == "factorized",
                    )
                ]
            )
        elif mode == "backbone":
            install_table_slot_layers(
                backbone,
                layer_indices=self.layer_indices,
                num_slots=num_slots,
                num_slot_iterations=num_slot_iterations,
                num_heads=num_slot_heads,
                scope=scope,
                direct_query_slot=query_routing_mode == "direct_slot",
            )
            self.adapters = nn.ModuleList()
        else:
            self.adapters = nn.ModuleList(
                TableSlotAdapter(
                    backbone.embedding_size,
                    backbone.mlp_hidden_size,
                    num_slots=num_slots,
                    num_iterations=num_slot_iterations,
                    num_heads=num_slot_heads,
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
        self.decoder = _SlotDecoder(
            backbone.embedding_size,
            backbone.mlp_hidden_size,
            max_classes,
            interaction=decoder_interaction,
        )
        if slot_composition == "factorized":
            # A pair slot carries one feature-axis slot and one datapoint-axis
            # slot.  The projection is deliberately before the shared decoder:
            # it lets the decoder see both axis identities while keeping the
            # number of output heads at K_feature*K_data.
            self.factorized_pair_projection = nn.Sequential(
                nn.Linear(3 * backbone.embedding_size, backbone.embedding_size),
                nn.GELU(),
                nn.Linear(backbone.embedding_size, backbone.embedding_size),
            )
        else:
            self.factorized_pair_projection = None
        if query_routing_mode in ("tabpfn_attention", "posterior_attention"):
            # Capture the final backbone block's real query-to-support map.
            # This is a non-parameter research hook and leaves ordinary
            # TableSlotModel modes on the original fast path.
            final_block = backbone.transformer_blocks[-1]
            if not hasattr(final_block, "capture_query_support_attention"):
                raise TypeError("backbone blocks do not expose TabPFN query-to-support attention.")
            final_block.capture_query_support_attention = True
            self.support_context_projection = nn.Sequential(
                nn.Linear(backbone.embedding_size, backbone.embedding_size),
                nn.GELU(),
                nn.Linear(backbone.embedding_size, backbone.embedding_size),
            )
        else:
            self.support_context_projection = None
        if embedding_reconstruction:
            self.adapters[0].embedding_reconstruction = True
            self.embedding_decoder = _SlotDecoder(
                backbone.embedding_size, backbone.mlp_hidden_size, backbone.embedding_size
            )
        self.last_feature_attention: torch.Tensor | None = None
        self.last_support_attention: torch.Tensor | None = None
        self.last_support_attention_for_loss: torch.Tensor | None = None
        self.last_slots: torch.Tensor | None = None
        self.last_query_gates: torch.Tensor | None = None
        self.last_query_support_attention: torch.Tensor | None = None
        self.last_query_support_attention_by_column: torch.Tensor | None = None
        self.last_query_slot_attention_by_column: torch.Tensor | None = None
        #: ``rho[i,k]`` from the full labelled support set for
        #: ``query_routing_mode='posterior_attention'``.
        self.last_support_posterior: torch.Tensor | None = None
        self.last_slot_utilization: torch.Tensor | None = None
        self.last_assignment_entropy: torch.Tensor | None = None
        #: ``KL(a || alpha)`` on the support rows, set by ``_reconstruct_support``.
        #: The two routings were never forced to agree and nothing measured
        #: whether they did; this is that number.  ``None`` until a
        #: reconstructing forward pass has run.
        self.last_gate_agreement: torch.Tensor | None = None
        #: ``(B,S,K)`` live log alpha on the support rows, for a consistency term.
        self.last_support_alpha_for_loss: torch.Tensor | None = None
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
        crossfit_posterior_gate = bool(kwargs.pop("crossfit_posterior_gate", False))
        crossfit_folds = int(kwargs.pop("crossfit_folds", 2))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return x, y, int(split), chunks, reconstruct, crossfit_posterior_gate, crossfit_folds

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

    def _decoder_rows(self, state: TableSlotState, split: int, *, support: bool) -> torch.Tensor:
        """Return the row representation consumed by the prediction decoder.

        NanoTabPFN decodes the final target-column token for each row, rather
        than a mean over all columns.  The slot adapter still keeps its pooled
        row representation for slot competition, but prediction and support
        reconstruction must preserve the backbone's target-token interface.
        Mufasa's explicit multi-tap query fusion remains its own representation.
        """
        if self.mode == "mufasa":
            # Mufasa's fused state already contains query rows only.
            return state.pooled_rows
        rows = state.table[..., -1, :]
        return rows[:, :split] if support else rows[:, split:]

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
        """``(B,S,C)`` log probabilities of every support label under its own routing.

        The slots come from the *labelled* pass; only the row representation is
        blinded.  What weights each slot's contribution is
        ``reconstruction_mixture``:

        ``"attention"``  ``a[i,k]``, slot attention's assignment, and the
        decoder's mask channel is discarded.

        ``"alpha"``  the softmax of that mask channel, which is what Locatello
        composites with.  The query side already gates on exactly this
        quantity, so under ``"alpha"`` the reconstruction and the query mixture
        stop being two mechanisms that merely share weights and become one
        expression evaluated on rows that do and do not carry a target.

        The mask is decoded either way -- it is a channel of the same output --
        so the agreement between the two routings is recorded for free.
        """
        support_logits, support_masks = self.decoder(self._decoder_rows(blind_state, split, support=True), state.slots)
        log_alpha = F.log_softmax(support_masks, dim=-1)
        # Diagnostic, never a loss unless `gate_consistency_weight` asks for it:
        # how far the decoder's alpha is from the competition's assignment on
        # the same rows.  Under "attention" these were never forced to agree and
        # nothing measured whether they did.
        self.last_gate_agreement = F.kl_div(
            log_alpha, state.support_attention.detach(), reduction="batchmean"
        ).detach()
        self.last_support_alpha_for_loss = log_alpha
        log_weight = (
            log_alpha
            if self.reconstruction_mixture == "alpha"
            else state.support_attention.clamp_min(1e-12).log()
        )
        return torch.logsumexp(log_weight[..., None] + F.log_softmax(support_logits, -1), dim=2)

    def _support_log_posterior(
        self, state: TableSlotState, blind_state: TableSlotState, support_y: torch.Tensor, split: int
    ) -> torch.Tensor:
        """Return ``log rho[i,c,k]`` from support assignment and label evidence."""
        if state.support_attention_by_column is None:
            raise RuntimeError("Posterior routing requires the per-column datapoint assignments.")
        support_rows = self._decoder_rows(blind_state, split, support=True)
        logits, _masks = self.decoder(support_rows, state.slots)
        log_probability = F.log_softmax(logits, dim=-1)
        labels = support_y.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_slots, 1)
        log_label_probability = log_probability.gather(-1, labels).squeeze(-1)
        log_assignment = state.support_attention_by_column.clamp_min(1e-12).log()
        return F.log_softmax(log_assignment + log_label_probability[:, :, None], dim=-1)

    @staticmethod
    def _posterior_query_gate(query_support_attention: torch.Tensor, log_posterior: torch.Tensor) -> torch.Tensor:
        """Transport support-label posterior evidence through TabPFN retrieval."""
        if query_support_attention.ndim != 4 or log_posterior.ndim != 4:
            raise ValueError("Posterior routing needs A[q,i,c] and rho[i,c,k].")
        if (
            query_support_attention.shape[0] != log_posterior.shape[0]
            or query_support_attention.shape[2] != log_posterior.shape[2]
            or query_support_attention.shape[-1] != log_posterior.shape[1]
        ):
            raise ValueError("Query-support attention and support posterior have incompatible cell or support axes.")
        # Each feature-specific attention map sums to one over support rows.
        # Divide by C so their aggregate remains a probability over slots.
        probability = torch.einsum("bqci,bick->bqk", query_support_attention, log_posterior.exp())
        probability = probability / query_support_attention.shape[2]
        return probability.clamp_min(1e-12).log()

    def _crossfit_log_posterior(
        self, x: torch.Tensor, y: torch.Tensor, split: int, chunks: int, folds: int, state: TableSlotState
    ) -> torch.Tensor:
        """Score each support label with slots made from the other folds only.

        The held-out rows are appended after the context, so NanoTabPFN pads
        their target tokens with the context-label mean.  Their labels are read
        only in the final likelihood lookup, never while making slots or
        representations.  This is intentionally a detached teacher branch.
        """
        if self.mode != "head" or self.scope == "cell":
            raise RuntimeError("Cross-fitted posterior routing requires head mode with a datapoint slot path.")
        if not 2 <= folds <= split:
            raise ValueError("crossfit_folds must lie in [2, support size].")
        adapter = self.adapters[0]
        previous_target = adapter.last_embedding_target
        columns = state.support_attention_by_column.shape[2] if state.support_attention_by_column is not None else 0
        posterior = x.new_empty(x.shape[0], split, columns, self.num_slots)
        fold_index = torch.arange(split, device=x.device).remainder(folds)
        try:
            with torch.no_grad():
                for fold in range(folds):
                    held = (fold_index == fold).nonzero(as_tuple=False).squeeze(-1)
                    context = (fold_index != fold).nonzero(as_tuple=False).squeeze(-1)
                    context_size, held_size = int(context.numel()), int(held.numel())
                    fold_x = torch.cat((x[:, context], x[:, held], x[:, split:]), dim=1)
                    # Passing only context labels makes every held target cell
                    # label-blind inside ``encode_table``.
                    encoded = self.backbone.encode_table((fold_x, y[:, context].float()), context_size, chunks)
                    fold_state = adapter(encoded, context_size)
                    if fold_state.datapoint_slots_by_column is None or fold_state.datapoint_input_table is None:
                        raise RuntimeError("Cross-fitted posterior routing needs retained datapoint slots and inputs.")
                    held_rows = fold_state.table[:, context_size : context_size + held_size, -1, :]
                    held_cells = fold_state.datapoint_input_table[:, context_size : context_size + held_size]
                    batch, _held, fold_columns, width = held_cells.shape
                    if fold_columns != columns:
                        raise RuntimeError("Cross-fitted posterior changed the table's feature count.")
                    cell_inputs = held_cells.permute(0, 2, 1, 3).reshape(batch * columns, held_size, width)
                    column_slots = fold_state.datapoint_slots_by_column.reshape(
                        batch * columns, self.num_slots, width
                    )
                    assignment = adapter.datapoint_slots.assignment(cell_inputs, column_slots)
                    assignment = assignment.reshape(batch, columns, held_size, self.num_slots).permute(0, 2, 1, 3)
                    logits, _masks = self.decoder(held_rows, fold_state.slots)
                    log_probability = F.log_softmax(logits, dim=-1)
                    labels = y[:, held].long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_slots, 1)
                    log_label_probability = log_probability.gather(-1, labels).squeeze(-1)
                    posterior[:, held] = F.log_softmax(
                        assignment.clamp_min(1e-12).log() + log_label_probability[:, :, None], dim=-1
                    )
        finally:
            adapter.last_state, adapter.last_embedding_target = state, previous_target
        return posterior

    def _similarity_gate(self, state: TableSlotState, blind_state: TableSlotState, split: int) -> torch.Tensor:
        """``(B,Q,K)`` log routing weights from blind-embedding cosine similarity.

        Each slot's centroid is the assignment-weighted mean of the *blind*
        support embeddings it claimed -- computed the same way a support row's
        own reconstruction target is, so a query is compared against exactly
        the representation its own class-mates would have produced.  Both
        sides of the comparison are blind, so nothing here can key off a label
        that was never available to the query in the first place.
        """
        support_blind = self._decoder_rows(blind_state, split, support=True)  # (B,S,E)
        weights = state.support_attention  # (B,S,K)
        centroids = torch.einsum("bsk,bse->bke", weights, support_blind)
        centroids = centroids / weights.sum(dim=1).clamp_min(1e-6)[..., None]  # (B,K,E)
        query_blind = F.normalize(self._decoder_rows(blind_state, split, support=False), dim=-1)  # (B,Q,E)
        centroids = F.normalize(centroids, dim=-1)
        similarity = torch.einsum("bqe,bke->bqk", query_blind, centroids)
        return F.log_softmax(similarity, dim=-1)

    def _support_conditioned_slots(
        self, state: TableSlotState, query_support_attention: torch.Tensor, split: int
    ) -> torch.Tensor:
        """Fuse TabPFN's query-to-support retrieval into every slot expert.

        ``query_support_attention`` is the actual final-backbone attention map,
        ``(B,Q,S)``.  Multiplying it by the support row's slot assignment keeps
        the support value path query-specific and slot-specific at once.
        """
        if self.support_context_projection is None:
            raise RuntimeError("support context projection is not enabled.")
        if query_support_attention.shape[:2] != (state.slots.shape[0], state.pooled_rows.shape[1] - split):
            raise ValueError("query-support attention has incompatible query dimensions.")
        support_values = self._decoder_rows(state, split, support=True)
        if query_support_attention.shape[-1] != support_values.shape[1]:
            raise ValueError("query-support attention has incompatible support dimensions.")
        joint = query_support_attention.clamp_min(0)[..., None] * state.support_attention[:, None]
        normalizer = joint.sum(dim=2).clamp_min(1e-6)
        context = torch.einsum("bqsk,bse->bqke", joint, support_values) / normalizer[..., None]
        shared_slots = state.slots[:, None].expand(-1, context.shape[1], -1, -1)
        return shared_slots + self.support_context_projection(context)

    def _factorized_pair_slots(
        self, feature_slots: torch.Tensor, datapoint_slots: torch.Tensor
    ) -> torch.Tensor:
        """Combine feature-axis and datapoint-axis slots without column pooling.

        ``feature_slots`` is ``(B,Q,K_f,E)`` and ``datapoint_slots`` is
        ``(B,Q,K_d,E)``.  The returned ``(B,Q,K_f*K_d,E)`` tensor is still
        consumed by the ordinary slot decoder, so the only new operation is
        the factorized slot composition itself.
        """
        if self.factorized_pair_projection is None:
            raise RuntimeError("factorized pair projection is not enabled.")
        if feature_slots.ndim != 4 or datapoint_slots.ndim != 4:
            raise ValueError("factorized slots must both have shape (B,Q,K,E).")
        if feature_slots.shape[:2] != datapoint_slots.shape[:2]:
            raise ValueError("feature and datapoint slots must have matching batch/query dimensions.")
        feature = feature_slots[:, :, :, None, :]
        datapoint = datapoint_slots[:, :, None, :, :]
        feature_count, data_count = feature_slots.shape[2], datapoint_slots.shape[2]
        pair = torch.cat(
            (
                feature.expand(-1, -1, -1, data_count, -1),
                datapoint.expand(-1, -1, feature_count, -1, -1),
                feature * datapoint,
            ),
            dim=-1,
        )
        batch, query, feature_count, data_count, width = pair.shape
        return self.factorized_pair_projection(pair.reshape(batch, query, feature_count * data_count, width))

    def _factorized_support_path(
        self, state: TableSlotState, query_support_attention: torch.Tensor, split: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query pair slots and the direct feature×row gate.

        The feature mass comes from each query row's cell assignments.  The
        datapoint mass is the actual TabPFN query-to-support attention pushed
        through support-row assignments.  Their outer product is a normalized
        gate, so support retrieval and slot allocation cannot be bypassed by a
        second learned mask channel.
        """
        if state.feature_slots is None or state.feature_attention.numel() == 0:
            raise RuntimeError("factorized composition needs retained feature slots and assignments.")
        if state.datapoint_slots_by_column is None or state.support_attention_by_column is None:
            raise RuntimeError("factorized composition needs column-wise datapoint slots and assignments.")
        if query_support_attention.ndim != 4:
            raise ValueError("factorized query-support attention must have shape (B,Q,C,S).")
        support_values = state.table[:, :split]
        column_slots = state.datapoint_slots_by_column.mean(dim=1)
        joint = torch.einsum("bqcs,bsck->bqsk", query_support_attention, state.support_attention_by_column)
        data_mass = joint.sum(dim=2).clamp_min(1e-8)
        context = torch.einsum(
            "bqcs,bsck,bsce->bqke",
            query_support_attention,
            state.support_attention_by_column,
            support_values,
        )
        context = context / data_mass[..., None]
        shared_slots = column_slots[:, None].expand(-1, context.shape[1], -1, -1)
        data_slots = shared_slots + self.support_context_projection(context)
        feature_slots = state.feature_slots[:, split:]
        feature_mass = state.feature_attention[:, split:].mean(dim=2).clamp_min(1e-8)
        pair_slots = self._factorized_pair_slots(feature_slots, data_slots)
        log_gate = (
            feature_mass.log()[..., :, None] + data_mass.log()[..., None, :]
        ).reshape(feature_mass.shape[0], feature_mass.shape[1], -1)
        return pair_slots, F.log_softmax(log_gate, dim=-1)

    def _factorized_support_slots(self, state: TableSlotState, split: int) -> torch.Tensor:
        """Build pair slots for support rows used by reconstruction.

        Query routing consumes feature×datapoint pair slots.  Reading pooled
        ``state.slots`` for the embedding auxiliary would train a shortcut that
        the query gate never uses, so support reconstruction must use the same
        pair representation.
        """
        if state.feature_slots is None or state.feature_attention.numel() == 0:
            raise RuntimeError("factorized support reconstruction needs retained feature slots and assignments.")
        if state.datapoint_slots_by_column is None or state.support_attention_by_column is None:
            raise RuntimeError("factorized support reconstruction needs column-wise datapoint slots and assignments.")
        feature_slots = state.feature_slots[:, :split]
        datapoint_slots = state.datapoint_slots_by_column.mean(dim=1)[:, None].expand(-1, split, -1, -1)
        return self._factorized_pair_slots(feature_slots, datapoint_slots)

    def _reconstruct_support_factorized(
        self, state: TableSlotState, blind_state: TableSlotState, split: int
    ) -> torch.Tensor:
        """Reconstruct support labels with the same factorized route as queries."""
        if state.feature_slots is None or state.feature_attention.numel() == 0:
            raise RuntimeError("factorized reconstruction needs retained feature slots and assignments.")
        batch, _rows = state.feature_slots.shape[:2]
        support = split
        feature_slots = state.feature_slots[:, :split]
        feature_mass = state.feature_attention[:, :split].mean(dim=2).clamp_min(1e-8)
        if state.support_attention_by_column is None or state.datapoint_slots_by_column is None:
            raise RuntimeError("factorized reconstruction needs column-wise datapoint slots and assignments.")
        data_mass = state.support_attention_by_column.mean(dim=2).clamp_min(1e-8)
        datapoint_slots = state.datapoint_slots_by_column.mean(dim=1)[:, None].expand(-1, split, -1, -1)
        pair_slots = self._factorized_pair_slots(feature_slots, datapoint_slots)
        log_weight = (feature_mass.log()[..., :, None] + data_mass.log()[..., None, :]).reshape(batch, support, -1)
        support_rows = self._decoder_rows(blind_state, split, support=True)
        support_logits, support_masks = self.decoder(support_rows, pair_slots)
        pair_log_alpha = F.log_softmax(support_masks.reshape(batch, support, -1), dim=-1)
        alpha = pair_log_alpha.exp().reshape(batch, support, feature_slots.shape[2], datapoint_slots.shape[2])
        aggregated_alpha = alpha.sum(dim=2)
        self.last_gate_agreement = F.kl_div(
            aggregated_alpha.clamp_min(1e-12).log(), state.support_attention.detach(), reduction="batchmean"
        ).detach()
        self.last_support_alpha_for_loss = pair_log_alpha
        mixture_log_weight = pair_log_alpha if self.reconstruction_mixture == "alpha" else log_weight
        support_log_probabilities = F.log_softmax(
            support_logits.reshape(batch, support, -1, support_logits.shape[-1]), -1
        )
        return torch.logsumexp(
            mixture_log_weight[..., None] + support_log_probabilities,
            dim=2,
        )

    def forward(self, *args, **kwargs) -> SlotRegimePrediction:
        gate_temperature = float(kwargs.pop("gate_temperature", 1.0))
        if not np.isfinite(gate_temperature) or gate_temperature <= 0:
            raise ValueError("gate_temperature must be finite and positive.")
        reconstruct_embeddings = bool(kwargs.pop("reconstruct_embeddings", False))
        if reconstruct_embeddings and not self.embedding_reconstruction:
            raise ValueError("Enable embedding_reconstruction when constructing the model.")
        x, y, split, chunks, reconstruct, crossfit_posterior_gate, crossfit_folds = self._args(args, kwargs)
        if crossfit_posterior_gate and self.query_routing_mode != "posterior_attention":
            raise ValueError("crossfit_posterior_gate requires query_routing_mode='posterior_attention'.")
        if reconstruct and self.mode != "head":
            # Ignoring the flag would report a run nobody configured; the
            # reconstruction pilot is scoped to the head placement.
            raise ValueError(f"reconstruct_support is a head-mode setting, not mode={self.mode!r}.")
        encode_chunks = 1 if self.query_routing_mode in ("tabpfn_attention", "posterior_attention") else chunks
        if self.mode == "head":
            encoded = self.backbone.encode_table((x, y.float()), split, encode_chunks)
            # Keep the target-token embedding from before the slot adapter can
            # rewrite cells.  This is the support label representation used by
            # the embedding reconstruction objective; it is detached so the
            # target branch cannot provide a copied-target shortcut.
            label_embedding_target = encoded[:, :split, -1, :].detach() if self.embedding_reconstruction else None
            state = self.adapters[0](encoded, split)
            if label_embedding_target is not None:
                self.adapters[0].last_embedding_target = label_embedding_target
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
                encoded,
                encoded.mean(2),
                state.slots,
                state.feature_attention,
                state.support_attention,
                state.feature_slots,
                state.datapoint_slots_by_column,
                state.support_attention_by_column,
                state.datapoint_input_table,
                state.query_slot_attention_by_column,
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
                        state.feature_slots,
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
        embedding_prediction = embedding_target = None
        if reconstruct_embeddings:
            embedding_target = self.adapters[0].last_embedding_target
            positions = row_positions(split, state.slots.shape[-1], state.slots).expand(x.shape[0], -1, -1)
            embedding_slots = (
                self._factorized_support_slots(state, split)
                if self.slot_composition == "factorized"
                else state.slots
            )
            values, masks = self.embedding_decoder(positions, embedding_slots)
            embedding_prediction = (masks.softmax(-1)[..., None] * values).sum(2)
        # Cleared rather than left standing: these are set only by a
        # reconstructing pass, and a stale value read off the model after an
        # ordinary one would report an agreement this call never measured.
        self.last_gate_agreement = None
        self.last_support_alpha_for_loss = None
        self.last_support_posterior = None
        self.last_query_slot_attention_by_column = None
        query_support_attention = None
        query_support_attention_by_column = None
        if self.query_routing_mode in ("tabpfn_attention", "posterior_attention"):
            final_block = self.backbone.transformer_blocks[-1]
            query_support_attention = final_block.last_query_support_attention
            query_support_attention_by_column = final_block.last_query_support_attention_by_feature
            if query_support_attention is None:
                raise RuntimeError("TabPFN query-to-support attention was not captured.")
            if self.slot_composition == "factorized" and query_support_attention_by_column is None:
                raise RuntimeError("Factorized routing requires per-column query-to-support attention.")
        # The blind pass is needed for support reconstruction and for either
        # blind query-routing mode; computed once and shared, so a run using
        # more than one of these still pays for it only once.
        blind_state = None
        if (
            reconstruct
            or self.query_routing_mode in ("blind_decoder", "blind_similarity", "posterior_attention")
            or self.query_content_mode == "blind"
        ):
            blind_state = self._blind_pass(x, y, split, encode_chunks, state)

        if self.query_routing_mode in ("tabpfn_attention", "posterior_attention"):
            self.last_query_support_attention = query_support_attention
            self.last_query_support_attention_by_column = query_support_attention_by_column
            if self.slot_composition == "factorized":
                decoder_slots, factorized_log_gate = self._factorized_support_path(
                    state, query_support_attention_by_column, split
                )
            else:
                decoder_slots = self._support_conditioned_slots(state, query_support_attention, split)
                factorized_log_gate = None
        else:
            self.last_query_support_attention = None
            self.last_query_support_attention_by_column = None
            decoder_slots = state.slots
            factorized_log_gate = None

        # Historically only routing could be blinded. The opt-in content
        # ablation also removes individual support labels from the decoder's
        # row input. Labelled slots are retained in both cases. Blind rows
        # still know the episode's mean support label.
        query = self._decoder_rows(state, split, support=False)
        if self.query_content_mode == "blind":
            query = self._decoder_rows(blind_state, split, support=False)
        logits, masks = self.decoder(query, decoder_slots)
        if self.slot_composition == "factorized":
            # The factorized gate is the outer product of query feature mass
            # and query-to-support row mass.  Decoder masks remain an output
            # diagnostic, but cannot replace this support-linked route.
            assert factorized_log_gate is not None
            log_gate = factorized_log_gate
        elif self.query_routing_mode == "blind_decoder":
            _, blind_masks = self.decoder(self._decoder_rows(blind_state, split, support=False), state.slots)
            log_gate = F.log_softmax(blind_masks, -1)
        elif self.query_routing_mode == "blind_similarity":
            log_gate = self._similarity_gate(state, blind_state, split)
        elif self.query_routing_mode == "posterior_attention":
            log_posterior = self._support_log_posterior(state, blind_state, y, split)
            self.last_support_posterior = log_posterior.detach().exp()
            if query_support_attention_by_column is None:
                raise RuntimeError("Posterior routing requires per-column TabPFN query-to-support attention.")
            log_gate = self._posterior_query_gate(query_support_attention_by_column, log_posterior)
        elif self.query_routing_mode == "direct_slot":
            if state.query_slot_attention_by_column is None:
                raise RuntimeError("Direct slot routing requires a query-slot assignment from a backbone adapter.")
            self.last_query_slot_attention_by_column = state.query_slot_attention_by_column
            log_gate = state.query_slot_attention_by_column.mean(dim=2).clamp_min(1e-12).log()
        else:
            log_gate = F.log_softmax(masks, -1)
        if gate_temperature != 1.0:
            log_gate = F.log_softmax(log_gate / gate_temperature, dim=-1)
        if crossfit_posterior_gate:
            # The cross-fitted route is a teacher.  In particular its
            # full-context retrieval map must not receive gradient and learn
            # to make the target easier to imitate.
            with torch.no_grad():
                log_crossfit_posterior = self._crossfit_log_posterior(
                    x, y, split, encode_chunks, crossfit_folds, state
                )
                crossfit_log_gate = self._posterior_query_gate(
                    query_support_attention_by_column, log_crossfit_posterior
                )
                if gate_temperature != 1.0:
                    crossfit_log_gate = F.log_softmax(crossfit_log_gate / gate_temperature, dim=-1)
        else:
            crossfit_log_gate = None
        # Runs only when asked, so ordinary inference costs exactly what it did.
        if reconstruct:
            reconstruction = (
                self._reconstruct_support_factorized(state, blind_state, split)
                if self.slot_composition == "factorized"
                else self._reconstruct_support(state, blind_state, split)
            )
        else:
            reconstruction = None
        self._record(state, log_gate)
        return SlotRegimePrediction(
            logits,
            log_gate,
            state.support_attention,
            reconstruction,
            embedding_prediction,
            embedding_target,
            crossfit_log_gate,
        )


__all__ = [
    "QUERY_ROUTING_MODES",
    "QueryRoutingMode",
    "SLOT_COMPOSITIONS",
    "SLOT_SCOPES",
    "SlotComposition",
    "SlotScope",
    "TableSlotAdapter",
    "TableSlotModel",
    "TableSlotState",
    "TableSlotTransformerEncoderLayer",
    "install_table_slot_layers",
]
