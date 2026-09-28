"""Slot-routed classification head for a frozen TabICLv2 backbone.

The model keeps TabICLv2's column embedding, row interaction, and in-context
transformer intact.  Its pretrained MLP prediction head is replaced by a
competitive slot router that aggregates labeled support rows into class-valued
slots and routes query rows to those slots.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tabicl._model.learning import ICLearning
from tabicl._model.tabicl import TabICL
from tabicl._sklearn.classifier import TabICLClassifier
from torch import Tensor, nn

from tfmplayground.models.slot_attention import SlotAttention


class TabICLSlotRouter(nn.Module):
    """Two-hop row-to-slot-to-label decoder operating on ICL row states."""

    def __init__(
        self,
        d_model: int,
        max_classes: int,
        num_slots: int,
        *,
        seed: int = 0,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive.")
        if max_classes < 2:
            raise ValueError("max_classes must be at least 2.")
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")
        slot_heads = next(heads for heads in (8, 4, 2, 1) if d_model % heads == 0)
        self.max_classes = max_classes
        self.num_slots = num_slots
        self.d_model = d_model
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_size=d_model,
            mlp_hidden_size=2 * d_model,
            num_iterations=num_iterations,
            num_heads=slot_heads,
            competitive=True,
            eval_seed=seed,
        )
        self.row_query = nn.Linear(d_model, d_model, bias=False)
        self.slot_key = nn.Linear(d_model, d_model, bias=False)
        self.active_slots: tuple[int, ...] | None = None
        self.collect_assignment_details = False
        self.last_diagnostics: dict[str, Tensor] = {}

    def set_active_slots(self, slots: Sequence[int] | None) -> None:
        """Temporarily restrict query routing for slot-ablation evaluation."""
        if slots is None:
            self.active_slots = None
            return
        selected = tuple(sorted(set(int(slot) for slot in slots)))
        if not selected:
            raise ValueError("At least one active slot is required.")
        if selected[0] < 0 or selected[-1] >= self.num_slots:
            raise ValueError(f"Active slots must be in [0, {self.num_slots - 1}].")
        self.active_slots = selected

    def forward(self, row_states: Tensor, y_train: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if row_states.ndim != 3:
            raise ValueError("row_states must have shape (batch, rows, width).")
        if y_train.ndim != 2 or y_train.shape[0] != row_states.shape[0]:
            raise ValueError("y_train must have shape (batch, support_rows).")
        support_size = y_train.shape[1]
        if support_size < 1 or support_size >= row_states.shape[1]:
            raise ValueError("The table must contain at least one support and one query row.")
        if row_states.shape[-1] != self.d_model:
            raise ValueError(f"Expected row width {self.d_model}, got {row_states.shape[-1]}.")

        support = row_states[:, :support_size]
        query = row_states[:, support_size:]
        slots, assignments = self.slot_attention(support)
        active_mask = torch.ones(self.num_slots, dtype=torch.bool, device=row_states.device)
        if self.active_slots is not None:
            active_mask.zero_()
            active_mask[list(self.active_slots)] = True

        assignments = assignments * active_mask.view(1, 1, -1)
        assignments = assignments / assignments.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        support_weights = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-8)
        labels = y_train.long()
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.max_classes):
            raise ValueError(f"Labels must be encoded in [0, {self.max_classes - 1}].")
        label_values = F.one_hot(labels, num_classes=self.max_classes).to(row_states.dtype)
        slot_values = torch.einsum("bsk,bsc->bkc", support_weights, label_values)

        query_vectors = self.row_query(query)
        slot_keys = self.slot_key(slots)
        similarities = query_vectors @ slot_keys.transpose(-1, -2) / (self.d_model**0.5)
        similarities = similarities.masked_fill(~active_mask.view(1, 1, -1), torch.finfo(similarities.dtype).min)
        query_assignments = similarities.softmax(dim=-1)
        probabilities = torch.einsum("bqk,bkc->bqc", query_assignments, slot_values)
        probabilities = probabilities.clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        support_entropy = -(assignments * assignments.clamp_min(1e-8).log()).sum(dim=-1)
        query_entropy = -(query_assignments * query_assignments.clamp_min(1e-8).log()).sum(dim=-1)
        diagnostics = {
            "slot_occupancy": assignments.mean(dim=1).detach(),
            "slot_assignment_entropy": support_entropy.mean(dim=1).detach(),
            "query_slot_entropy": query_entropy.mean(dim=1).detach(),
        }
        if self.collect_assignment_details:
            diagnostics["support_assignments"] = assignments.detach()
            diagnostics["query_assignments"] = query_assignments.detach()
        self.last_diagnostics = diagnostics
        return probabilities, diagnostics


class SlotRoutedICLearning(ICLearning):
    """TabICL ICL predictor with the official encoder and a slot decoder."""

    def __init__(self, pretrained_predictor: ICLearning, num_slots: int, *, seed: int = 0) -> None:
        nn.Module.__init__(self)
        self.max_classes = pretrained_predictor.max_classes
        if self.max_classes <= 1:
            raise ValueError("SlotRoutedICLearning supports classification checkpoints only.")
        self.norm_first = pretrained_predictor.norm_first
        self.tf_icl = pretrained_predictor.tf_icl
        self.y_encoder = pretrained_predictor.y_encoder
        if self.norm_first:
            self.ln = pretrained_predictor.ln
        self.inference_mgr = pretrained_predictor.inference_mgr
        with torch.no_grad():
            d_model = pretrained_predictor.y_encoder(torch.zeros((1, 1), dtype=torch.float32)).shape[-1]
        self.slot_decoder = TabICLSlotRouter(
            d_model=d_model,
            max_classes=self.max_classes,
            num_slots=num_slots,
            seed=seed,
        )
        self.last_diagnostics: dict[str, Tensor] = {}

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Return all-row logits in the same layout expected by TabICLv2."""
        train_size = y_train.shape[1]
        representations = R.clone()
        label_repr = self.y_encoder(y_train.float())
        representations[:, :train_size] = representations[:, :train_size] + label_repr
        source = self.tf_icl(representations, train_size=train_size)
        if self.norm_first:
            source = self.ln(source)
        probabilities, diagnostics = self.slot_decoder(source, y_train)
        self.last_diagnostics = diagnostics
        log_probabilities = probabilities.clamp_min(1e-8).log()
        support_logits = log_probabilities.new_zeros((R.shape[0], train_size, self.max_classes))
        return torch.cat((support_logits, log_probabilities), dim=1)

    def forward_with_cache(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("Slot-routed TabICL checkpoints require kv_cache=False.")

    def forward_with_repr_cache(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("Slot-routed TabICL checkpoints require kv_cache=False.")


def install_slot_decoder(model: TabICL, num_slots: int, *, seed: int = 0) -> TabICLSlotRouter:
    """Replace only the pretrained MLP head while retaining every other module."""
    original = model.icl_predictor
    model.icl_predictor = SlotRoutedICLearning(original, num_slots, seed=seed)
    return model.icl_predictor.slot_decoder



class TabPFNv3AttentionICLearning(ICLearning):
    """TabICLv2 ICL encoder with TabPFNv3's official many-class decoder."""

    def __init__(self, pretrained_predictor: ICLearning) -> None:
        nn.Module.__init__(self)
        from tabpfn.architectures.tabpfn_v3 import ManyClassDecoder

        self.max_classes = pretrained_predictor.max_classes
        if self.max_classes <= 1:
            raise ValueError("TabPFNv3AttentionICLearning supports classification checkpoints only.")
        self.norm_first = pretrained_predictor.norm_first
        self.tf_icl = pretrained_predictor.tf_icl
        self.y_encoder = pretrained_predictor.y_encoder
        if self.norm_first:
            self.ln = pretrained_predictor.ln
        self.inference_mgr = pretrained_predictor.inference_mgr
        with torch.no_grad():
            d_model = self.y_encoder(torch.zeros((1, 1), dtype=torch.float32)).shape[-1]
        # These are TabPFNv3's default multiclass-decoder dimensions.
        self.tabpfnv3_decoder = ManyClassDecoder(
            max_num_classes=self.max_classes,
            input_size=d_model,
            head_dim=64,
            num_heads=6,
        )

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Return support placeholders and TabPFNv3 attention-decoder query logits."""
        train_size = y_train.shape[1]
        representations = R.clone()
        representations[:, :train_size] += self.y_encoder(y_train.float())
        source = self.tf_icl(representations, train_size=train_size)
        if self.norm_first:
            source = self.ln(source)
        train_embeddings = source[:, :train_size]
        test_embeddings = source[:, train_size:]
        train_keys = self.tabpfnv3_decoder.project_keys(train_embeddings)
        query_log_probabilities = self.tabpfnv3_decoder(
            train_keys,
            test_embeddings,
            y_train.long(),
        ).transpose(0, 1)
        support_logits = query_log_probabilities.new_zeros(
            (R.shape[0], train_size, self.max_classes)
        )
        return torch.cat((support_logits, query_log_probabilities), dim=1)

    def forward_with_cache(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("Decoder-only TabICL checkpoints require kv_cache=False.")

    def forward_with_repr_cache(self, *args: Any, **kwargs: Any) -> Tensor:
        raise RuntimeError("Decoder-only TabICL checkpoints require kv_cache=False.")


def install_tabpfnv3_decoder(model: TabICL) -> nn.Module:
    """Install TabPFNv3's ManyClassDecoder architecture over TabICLv2 ICL states."""
    model.icl_predictor = TabPFNv3AttentionICLearning(model.icl_predictor)
    return model.icl_predictor.tabpfnv3_decoder

def _load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Not a TabICL checkpoint: {path}")
    return checkpoint


class TabICLSlotDecoderClassifier(TabICLClassifier):
    """TabICL's sklearn preprocessing/inference with a trained slot checkpoint."""

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: str | list[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: int | None = 8,
        kv_cache: bool | str = False,
        model_path: str | Path | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        device: str | torch.device | None = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: str | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        inference_config: Any = None,
        expected_num_slots: int | None = None,
    ) -> None:
        if kv_cache:
            raise ValueError("Decoder-only TabICL inference currently requires kv_cache=False.")
        self.expected_num_slots = expected_num_slots
        super().__init__(
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold,
            softmax_temperature=softmax_temperature,
            average_logits=average_logits,
            support_many_classes=support_many_classes,
            batch_size=batch_size,
            kv_cache=False,
            model_path=model_path,
            allow_auto_download=allow_auto_download,
            checkpoint_version=checkpoint_version,
            device=device,
            use_amp=use_amp,
            use_fa3=use_fa3,
            offload_mode=offload_mode,
            disk_offload_dir=disk_offload_dir,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            inference_config=inference_config,
        )

    def _load_model(self) -> None:
        if self.model_path is None:
            raise ValueError("Provide model_path pointing to a trained TabICL slot-decoder checkpoint.")
        model_path = Path(self.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Slot-decoder checkpoint not found: {model_path}")
        checkpoint = _load_checkpoint_file(model_path)
        metadata = checkpoint.get("slot_decoder")
        if not isinstance(metadata, dict) or metadata.get("head_kind") not in {"slot", "mlp", "tabpfnv3"}:
            raise ValueError("Checkpoint does not contain a supported TabICL decoder.")
        head_kind = metadata["head_kind"]
        num_slots = int(metadata["num_slots"]) if head_kind == "slot" else None
        if self.expected_num_slots is not None and num_slots != self.expected_num_slots:
            raise ValueError(f"Checkpoint has K={num_slots} slots, expected K={self.expected_num_slots}.")

        self.model_path_ = model_path.resolve()
        state_dict_format = checkpoint.get("state_dict_format")
        if state_dict_format == "decoder_only_v1":
            base_checkpoint_path = metadata.get("base_checkpoint_path")
            if base_checkpoint_path and not Path(base_checkpoint_path).is_file():
                base_checkpoint_path = None
            self.model_, self.model_config_, loaded_base_path = load_released_tabicl_model(
                checkpoint_path=base_checkpoint_path,
                checkpoint_version=metadata.get("base_checkpoint_version", self.checkpoint_version),
                device="cpu",
            )
            if dict(checkpoint["config"]) != dict(self.model_config_):
                raise ValueError("Released TabICLv2 configuration differs from the decoder checkpoint.")
            expected_hash = metadata.get("base_checkpoint_sha256")
            if expected_hash:
                digest = hashlib.sha256()
                with loaded_base_path.open("rb") as base_file:
                    for chunk in iter(lambda: base_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError("Released TabICLv2 weights differ from the decoder checkpoint provenance.")
            if head_kind == "slot":
                install_slot_decoder(self.model_, num_slots, seed=int(metadata.get("seed", 0)))
                decoder = self.model_.icl_predictor.slot_decoder
            elif head_kind == "tabpfnv3":
                decoder = install_tabpfnv3_decoder(self.model_)
            else:
                decoder = self.model_.icl_predictor.decoder
            decoder.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            # Backward compatibility for full-model checkpoints.
            self.model_config_ = checkpoint["config"]
            self.model_ = TabICL(**self.model_config_)
            if head_kind == "slot":
                install_slot_decoder(self.model_, num_slots, seed=int(metadata.get("seed", 0)))
            elif head_kind == "tabpfnv3":
                install_tabpfnv3_decoder(self.model_)
            self.model_.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model_.eval()

    def predict_proba_with_slot_mask(self, X: Any, active_slots: Sequence[int] | None) -> Any:
        """Predict with selected slots enabled; useful for leave-one-slot-out checks."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before requesting slot-ablation predictions.")
        if not isinstance(self.model_.icl_predictor, SlotRoutedICLearning):
            raise RuntimeError("Slot ablation is available only for a slot-decoder checkpoint.")
        router = self.model_.icl_predictor.slot_decoder
        previous = router.active_slots
        router.set_active_slots(active_slots)
        try:
            return self.predict_proba(X)
        finally:
            router.set_active_slots(previous)

    def get_last_slot_diagnostics(self) -> dict[str, Tensor]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before requesting slot diagnostics.")
        if not isinstance(self.model_.icl_predictor, SlotRoutedICLearning):
            raise RuntimeError("Slot diagnostics are available only for a slot-decoder checkpoint.")
        return dict(self.model_.icl_predictor.last_diagnostics)

    def predict_proba_with_slot_diagnostics(self, X: Any) -> tuple[Any, dict[str, Tensor]]:
        """Return predictions and optional row-level assignments for a diagnostics pass."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before requesting slot diagnostics.")
        if not isinstance(self.model_.icl_predictor, SlotRoutedICLearning):
            raise RuntimeError("Slot diagnostics are available only for a slot-decoder checkpoint.")
        router = self.model_.icl_predictor.slot_decoder
        router.collect_assignment_details = True
        try:
            probabilities = self.predict_proba(X)
            diagnostics = self.get_last_slot_diagnostics()
        finally:
            router.collect_assignment_details = False
        return probabilities, diagnostics


def load_released_tabicl_model(
    *,
    checkpoint_path: str | None = None,
    checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
    device: str = "cpu",
) -> tuple[TabICL, dict[str, Any], Path]:
    """Load the official released checkpoint using TabICL's own loader."""
    loader = TabICLClassifier(
        model_path=checkpoint_path,
        allow_auto_download=checkpoint_path is None,
        checkpoint_version=checkpoint_version,
        device=device,
        kv_cache=False,
    )
    loader._load_model()
    return loader.model_, dict(loader.model_config_), Path(loader.model_path_).resolve()
