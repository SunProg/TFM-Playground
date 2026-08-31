"""Row-level regime mixture over slot-attention slots, for a nanoTabPFN backbone.

The multiregime prior is a *row-level* mixture: one shared feature matrix is
labelled by two independently drawn label functions and a contamination fraction
of support rows -- and, independently, of query rows -- is relabelled under the
second (``continuous_episodes._build_multiregime_item``).  Its own comment is
explicit: "the truth for an individual row is a per-row mixture, not a single
resolved candidate the way it is for every other condition."

Every existing head in this package answers with one mixture weight per
*episode*, which structurally cannot express that.  This head gates per *query
row* instead, which is the tabular reading of the vision decoder's alpha masks:
slots decode independently and are composited by a softmax over slots.

Nothing here supervises which slot owns which regime.  Slots specialize because
they compete for support rows and because the mixture is the only path to the
label, exactly as object slots emerge from reconstruction alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_attention import SlotBindingMixin, slot_binding_kwargs


@dataclass(frozen=True)
class SlotRegimePrediction:
    """Per-slot predictions plus the per-query mixture over them.

    Attributes:
        slot_logits: ``(batch, query, slot, class)`` logits of each slot read
            independently.
        log_gate: ``(batch, query, slot)`` log mixture weight of each slot *for
            that query row*.  Rows sum to one over slots.
        support_attention: ``(batch, support, slot)`` slot competition over the
            labelled rows.  Under competitive slot attention each support row's
            weights sum to one, so this is a soft per-row regime assignment.
    """

    slot_logits: torch.Tensor
    log_gate: torch.Tensor
    support_attention: torch.Tensor

    @property
    def num_slots(self) -> int:
        return self.slot_logits.shape[2]

    def slot_log_probabilities(self) -> torch.Tensor:
        """``(batch, query, slot, class)`` log probabilities per slot."""
        return F.log_softmax(self.slot_logits, dim=-1)

    def slot_probabilities(self) -> torch.Tensor:
        return self.slot_log_probabilities().exp()

    def marginal_log_probabilities(self) -> torch.Tensor:
        """``(batch, query, class)`` mixture, marginalizing the slot axis out.

        ``logsumexp_k(log_gate + log p(y | slot k))``.  This is a sum over slots,
        so it is invariant to their order -- which is why no slot-to-regime
        matching is needed anywhere.
        """
        return torch.logsumexp(self.log_gate[..., None] + self.slot_log_probabilities(), dim=2)

    def marginal_probabilities(self) -> torch.Tensor:
        """``(batch, query, class)``; the distribution every interface reads."""
        return self.marginal_log_probabilities().exp()

    def gate(self) -> torch.Tensor:
        return self.log_gate.exp()

    def predictive_entropy(self) -> torch.Tensor:
        """``(batch, query)`` entropy of the mixture."""
        log_probabilities = self.marginal_log_probabilities()
        return -(log_probabilities.exp() * log_probabilities).sum(-1)

    def expected_slot_entropy(self) -> torch.Tensor:
        """``(batch, query)`` gate-weighted mean entropy of the individual slots."""
        log_probabilities = self.slot_log_probabilities()
        entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
        return (self.gate() * entropy).sum(-1)

    def mutual_information(self) -> torch.Tensor:
        """``(batch, query)`` disagreement between slots; zero when they concur."""
        return (self.predictive_entropy() - self.expected_slot_entropy()).clamp_min(0)

    def gate_entropy(self) -> torch.Tensor:
        """``(batch, query)`` normalized entropy of the per-row slot gate.

        Zero means a query row is explained by one slot outright, one means the
        gate is uniform and no routing happened.
        """
        entropy = -(self.gate() * self.log_gate).sum(-1)
        return entropy / math.log(self.num_slots) if self.num_slots > 1 else entropy

    def slot_disagreement(self) -> torch.Tensor:
        """``(batch, query)`` gate-weighted variance of the class-1 probability."""
        positive = (
            self.slot_probabilities()[..., 1] if self.slot_logits.shape[-1] > 1 else self.slot_probabilities()[..., 0]
        )
        mean = (self.gate() * positive).sum(-1, keepdim=True)
        return (self.gate() * (positive - mean).square()).sum(-1)


class _SlotDecoder(nn.Module):
    """Decode one query row against one slot: class logits plus a mask logit.

    The extra channel is the tabular alpha mask.  In the vision model the
    spatial broadcast decoder emits RGB *and* alpha from one pathway, so what a
    slot predicts and where it applies are tied together and composited by a
    softmax over slots.  Emitting the mask as an extra output channel here keeps
    that property: the routing cannot drift away from the decoder's competence,
    because both come from the same weights.
    """

    def __init__(self, embedding_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            nn.Linear(embedding_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes + 1),
        )

    def forward(self, rows: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class logits, mask logits)`` of shapes ``(B,Q,K,C)`` and ``(B,Q,K)``."""
        row_count, slot_count = rows.shape[1], slots.shape[1]
        expanded_rows = rows[:, :, None].expand(-1, -1, slot_count, -1)
        expanded_slots = slots[:, None].expand(-1, row_count, -1, -1)
        features = torch.cat((expanded_rows, expanded_slots, expanded_rows * expanded_slots), dim=-1)
        decoded = self.body(features)
        return decoded[..., : self.num_classes], decoded[..., self.num_classes]


class NanoTabPFNSlotRegimeModel(SlotBindingMixin, nn.Module):
    """nanoTabPFN whose query predictions are a per-row mixture over slots.

    Trainable end to end.  Unlike the adapter heads in this package the backbone
    is *not* frozen in ``__init__``: this model is meant to be pretrained from
    scratch alongside its slots.  ``freeze_backbone()`` is available for the
    adapter-style use.
    """

    model_type = "nanotabpfn_slot_regime"

    def __init__(
        self,
        backbone: NanoTabPFNModel,
        num_slots: int = 2,
        max_classes: int = 2,
        *,
        num_slot_iterations: int = 3,
        competitive_slots: bool = True,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")
        if max_classes < 2:
            raise ValueError("max_classes must be at least two.")
        if max_classes > backbone.num_outputs:
            raise ValueError("max_classes cannot exceed the backbone output count.")
        self.backbone = backbone
        self.num_slots = num_slots
        self.max_classes = max_classes
        embedding_size = backbone.embedding_size
        hidden_size = backbone.mlp_hidden_size
        self._init_slot_binding(
            num_slots=num_slots,
            embedding_size=embedding_size,
            mlp_hidden_size=hidden_size,
            num_slot_iterations=num_slot_iterations,
            competitive_slots=competitive_slots,
        )
        # Query rows carry no label, so they cannot compete for slots the way
        # support rows do.  They are routed by the decoder's own mask channel,
        # which is the tabular reading of the vision decoder's alpha output.
        self.slot_decoder = _SlotDecoder(embedding_size, hidden_size, max_classes)

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def adapter_parameters(self):
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("backbone."))

    def _split_arguments(self, args: tuple, kwargs: dict):
        """Accept both repository calling conventions, as every head here does."""
        if len(args) == 3:
            support_x, support_y, query_x = args
            source = (torch.cat((support_x, query_x), dim=1), support_y)
            split = support_x.shape[1]
        elif len(args) == 1 and isinstance(args[0], tuple):
            source = args[0]
            split = kwargs.pop("train_test_split_index", None)
            if split is None:
                raise TypeError("train_test_split_index is required for the concatenated-table interface.")
        else:
            raise TypeError("Expected (support_x, support_y, query_x) or ((x, y), train_test_split_index=...).")
        num_mem_chunks = kwargs.pop("num_mem_chunks", 1)
        generator = kwargs.pop("generator", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return source, int(split), num_mem_chunks, generator

    def forward(self, *args, **kwargs) -> SlotRegimePrediction:
        source, split, num_mem_chunks, generator = self._split_arguments(args, kwargs)
        x_source, y_source = source
        if x_source.ndim != 3:
            raise ValueError("Features must have shape (batch, rows, features).")
        if split < 1 or split >= x_source.shape[1]:
            raise ValueError("train_test_split_index must leave at least one support and one query row.")
        encoded = self.backbone.encode_table(
            (x_source, y_source.float()),
            train_test_split_index=split,
            num_mem_chunks=num_mem_chunks,
        )
        # The target column carries the label embedding after repeated
        # feature/row attention, so a support state depends on both x_i and y_i.
        states = encoded[:, :, -1, :]
        support_states, query_states = states[:, :split], states[:, split:]

        slots, support_attention = self.make_slots(support_states, generator=generator)
        slot_logits, mask_logits = self.slot_decoder(query_states, slots)
        # Softmax over slots, exactly as the vision decoder normalizes its alpha
        # masks across slots before compositing.
        log_gate = F.log_softmax(mask_logits, dim=-1)
        return SlotRegimePrediction(
            slot_logits=slot_logits,
            log_gate=log_gate,
            support_attention=support_attention,
        )


class SlotLogitsAdapter(nn.Module):
    """Present a slot model through the plain ``NanoTabPFNModel`` call signature.

    Returns ``log`` mixture probabilities, which are valid logits -- softmax of a
    log-probability vector is the probability vector itself.  ``hypothesis.py``
    uses the same trick.  This lets every existing evaluator score a slot
    checkpoint unchanged instead of growing a second prediction path.
    """

    def __init__(self, model: NanoTabPFNSlotRegimeModel):
        super().__init__()
        self.model = model

    @property
    def backbone(self) -> NanoTabPFNModel:
        return self.model.backbone

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs).marginal_log_probabilities()


def slot_regime_loss(prediction: SlotRegimePrediction, target_y: torch.Tensor) -> torch.Tensor:
    """Mixture negative log likelihood on the realized query labels.

    The whole objective.  There is deliberately no specialization, coherence,
    diversity or posterior-supervision term: those hand the model the task
    assignment that slot competition is supposed to discover, and this repo has
    twice recorded them failing to buy specialization
    (``MEAN_PRESERVING_BAYESIAN_TRIAL.md``, ``H5_ALLTERMS_MODEL.md``).
    """
    log_probabilities = prediction.marginal_log_probabilities()
    if target_y.shape != log_probabilities.shape[:2]:
        raise ValueError("target_y must have shape (batch, query rows).")
    return F.nll_loss(log_probabilities.flatten(0, 1), target_y.reshape(-1).long())


def slot_regime_checkpoint(
    model: NanoTabPFNSlotRegimeModel,
    *,
    training_config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backbone = model.backbone
    checkpoint: dict[str, Any] = {
        "model_type": model.model_type,
        "architecture": {
            "num_layers": backbone.num_layers,
            "embedding_size": backbone.embedding_size,
            "num_attention_heads": backbone.num_attention_heads,
            "mlp_hidden_size": backbone.mlp_hidden_size,
            "backbone_num_outputs": backbone.num_outputs,
            "num_slots": model.num_slots,
            "max_classes": model.max_classes,
            **model.slot_binding_architecture(),
        },
        "model": model.state_dict(),
        "training_config": training_config,
    }
    if extra:
        checkpoint.update(extra)
    return checkpoint


def save_slot_regime_checkpoint(path: str | Path, model: NanoTabPFNSlotRegimeModel, **kwargs: Any) -> None:
    torch.save(slot_regime_checkpoint(model, **kwargs), Path(path))


def build_slot_regime_model(architecture: dict[str, Any]) -> NanoTabPFNSlotRegimeModel:
    """Reconstruct a model from a checkpoint's ``architecture`` dictionary."""
    backbone = NanoTabPFNModel(
        num_layers=architecture["num_layers"],
        embedding_size=architecture["embedding_size"],
        num_attention_heads=architecture["num_attention_heads"],
        mlp_hidden_size=architecture["mlp_hidden_size"],
        num_outputs=architecture["backbone_num_outputs"],
    )
    return NanoTabPFNSlotRegimeModel(
        backbone,
        num_slots=architecture["num_slots"],
        max_classes=architecture["max_classes"],
        **slot_binding_kwargs(architecture),
    )


def load_slot_regime_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[NanoTabPFNSlotRegimeModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not is_slot_regime_checkpoint(checkpoint):
        raise ValueError("Checkpoint is not a nanoTabPFN slot regime model.")
    model = build_slot_regime_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def load_checkpoint_for_inference(path: str | Path, device: str | torch.device = "cpu"):
    """Load either checkpoint kind behind the plain ``NanoTabPFNModel`` signature.

    A slot checkpoint comes back wrapped in :class:`SlotLogitsAdapter`, which
    emits log mixture probabilities -- valid logits, so callers that do
    ``model(...)[..., :2].softmax(-1)`` keep working untouched.  This is what
    lets the TabArena and locked-episode evaluators score slot runs without
    growing a second prediction path.
    """
    from tfmplayground.interface import init_model_from_state_dict_file  # local: avoids an import cycle

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    architecture = state.get("architecture") or {}
    if architecture.get("model_kind") == "slot_backbone":
        # Slots live inside the transformer layers, so the layers must be
        # installed before the state dict will match.
        from tfmplayground.models.slot_backbone import install_slot_layers

        backbone = NanoTabPFNModel(
            num_layers=architecture["num_layers"],
            embedding_size=architecture["embedding_size"],
            num_attention_heads=architecture["num_attention_heads"],
            mlp_hidden_size=architecture["mlp_hidden_size"],
            num_outputs=architecture["num_outputs"],
        )
        install_slot_layers(
            backbone,
            num_slots=architecture["num_slots"],
            num_slot_iterations=architecture.get("num_slot_iterations", 3),
            competitive_slots=architecture.get("competitive_slots", True),
            # Absent from checkpoints written before the compatibility became
            # configurable, and all of those scored by dot product.
            compatibility=architecture.get("slot_compatibility", "dot"),
            max_classes=architecture.get("max_classes", 2),
        )
        backbone.load_state_dict(state["model"])
        return backbone.to(device).eval()
    if is_slot_regime_checkpoint(state):
        model = build_slot_regime_model(state["architecture"])
        model.load_state_dict(state["model"])
        return SlotLogitsAdapter(model).to(device).eval()
    return init_model_from_state_dict_file(str(path)).to(device).eval()


def is_slot_regime_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Whether a loaded checkpoint holds a slot regime model.

    Pretraining runs stamp ``slot_tabpfn_<prior mode>_scm_pretraining`` so the
    prior composition stays visible in run metadata, so match on the architecture
    instead of on one exact string.
    """
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        return False
    # slot_backbone also carries num_slots but is a plain NanoTabPFNModel with
    # slot layers installed, not this head.
    return "num_slots" in architecture and architecture.get("model_kind") != "slot_backbone"


__all__ = [
    "NanoTabPFNSlotRegimeModel",
    "SlotLogitsAdapter",
    "SlotRegimePrediction",
    "build_slot_regime_model",
    "is_slot_regime_checkpoint",
    "load_checkpoint_for_inference",
    "load_slot_regime_checkpoint",
    "save_slot_regime_checkpoint",
    "slot_regime_checkpoint",
    "slot_regime_loss",
]
