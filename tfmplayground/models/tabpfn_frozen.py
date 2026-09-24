"""Frozen real-TabPFN backbone exposing an ``encode_table``-compatible interface.

``NanoTabPFNModel.encode_table`` (nanotabpfn.py) gives research heads the full
per-cell embedding table ``(B, R, C, E)`` before the output decoder is
applied. Real TabPFN (the `tabpfn` PyPI package) has no equivalent method:
its own ``only_return_standard_out=False`` forward mode and
``TabPFNClassifier.get_embeddings()`` only ever return the *target-column*
embedding (``x_BRCD[:, :, -1]``), not the full per-column table. To get the
full table this module hooks an internal module directly -- which one
depends on the checkpoint's architecture generation:

- TabPFNV2 (``tabpfn.architectures.tabpfn_v2``): a per-cell ``(B,R,C,E)``
  table is threaded through a stack of ``TabPFNBlock`` entries in
  ``model.blocks``; hooking any one of them gives the full table at that
  depth, same shape at every layer.
- TabPFNV3 (``tabpfn.architectures.tabpfn_v3``): columns are aggregated into
  a per-row representation *before* its ``icl_blocks`` transformer ever
  runs (via ``column_aggregator``), so hooking ``icl_blocks`` would give the
  same target-column-only shape ``get_embeddings()`` already gives. The full
  per-column-per-row table exists only briefly, as the output of
  ``model.feature_distribution_embedder`` (tabpfn_v3.py's ``_process_row_chunk``)
  -- there is no layer-index choice for V3, only this one checkpoint.

Column-count caveat (architecture-dependent, verified empirically, not just
from source reading -- see tests/test_tabpfn_frozen.py):

- TabPFNV2: several (preprocessed) raw feature columns are packed into one
  embedding column (``features_per_group``, default 2), so the returned
  tensor's column axis is ``C ~= ceil(num_raw_features / features_per_group)``,
  not raw-feature-identity-preserving.
- TabPFNV3: despite ``feature_group_size`` sounding like the same kind of
  packing, it is not -- V3's "feature grouping" gives each column extra
  circular-shifted-neighbor context rather than merging columns, so no
  reduction happens: ``C == num_raw_features + 1`` (the ``+1`` is the target
  column, appended the same way nanotabpfn's own ``encode_table`` appends
  it). Raw feature identity is fully column-for-column preserved.

Either way, with ensembling, feature shuffling, and class shuffling all
disabled below, column count and order are deterministic and stable across
episodes for a fixed feature-count schema.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class FrozenTabPFNBackbone:
    """Wraps a frozen, single-member ``TabPFNClassifier`` for embedding extraction.

    Not an ``nn.Module``: it holds no ``nn.Parameter``, so any model that
    stores one of these as an attribute automatically excludes real TabPFN's
    weights from its own ``.parameters()`` -- no ``requires_grad_`` bookkeeping
    needed for gradient isolation, only for hygiene (see ``freeze_backbone``).
    """

    def __init__(
        self,
        *,
        layer_index: int = -1,
        device: str = "cpu",
        model_path: str | None = None,
        random_state: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        # tabpfn is an optional extra (`uv sync --extra tabpfn`); import lazily
        # so this module stays importable without it, and opt out of
        # tabpfn-common-utils' telemetry the same way evaluate_tabarena_small.py does.
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        # Opt-in build cache (tabpfn/model_loading.py): with fit_mode=
        # "fit_preprocessors" (cache_trainset_representation=False), setting
        # this lets tabpfn reuse the already-built Architecture instance
        # across .fit() calls instead of reinstantiating the network and
        # reloading its weights from the checkpoint on every single episode --
        # essential given the on-the-fly (no precomputed cache) design here.
        # Does not affect correctness either way: _current_model() always
        # re-fetches the live model reference after every .fit() below.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.preprocessing import PreprocessorConfig
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "FrozenTabPFNBackbone requires the optional 'tabpfn' extra: uv sync --extra tabpfn"
            ) from error

        # Normalized up front: _PerDeviceModelCache.get() looks the model up
        # by torch.device key, and torch.device("cpu") is neither == nor
        # hash-equal to the plain string "cpu", so a raw string here would
        # raise KeyError. For CUDA specifically, an index-less torch.device
        # ("cuda") is *also* not hash-equal to the indexed device
        # (device(type='cuda', index=0)) that _get_current_device(model)
        # actually returns for a GPU tensor -- CPU has no index concept, so
        # this half of the bug only shows up once actually run on a GPU.
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        kwargs = {"device": self.device}
        if model_path:
            kwargs["model_path"] = model_path
        self._classifier = TabPFNClassifier(
            n_estimators=1,
            categorical_features_indices=[],
            inference_precision=dtype,
            random_state=random_state,
            fit_mode="fit_preprocessors",
            inference_config={
                # The least column-disruptive built-in preset: no quantile
                # transform, no SVD-appended columns, no append_original --
                # see the module docstring's column-count caveat.
                "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none", categorical_name="numeric")],
                # Both default to "shuffle": without disabling them, TabPFN
                # rotates/shuffles feature columns and class labels on every
                # .fit() call even at n_estimators=1, breaking determinism
                # (decision: single deterministic member, no ensembling).
                "FEATURE_SHIFT_METHOD": None,
                "CLASS_SHIFT_METHOD": None,
            },
            **kwargs,
        )
        self.layer_index = layer_index
        self.dtype = dtype
        self._embedding_size = None
        self._num_layers = None

    def freeze_backbone(self) -> None:
        """No-op for API parity with this repo's frozen-backbone convention
        (sequential_latent_filter.py, hypothesis.py, coherent_correction.py,
        task_posterior_adapter.py). Real TabPFN's weights are never registered
        as parameters of any model holding this wrapper, so there is nothing
        to actually freeze -- this exists only so call sites don't need to
        special-case a backbone that happens to already be frozen by construction.
        """

    def _current_model(self):
        """The Architecture instance backing the classifier's most recent
        .fit() call. Must be re-fetched after every .fit(), never cached
        across calls: unless TABPFN_MODEL_CACHE_SIZE enables the opt-in build
        cache, each .fit() constructs a brand-new Architecture instance, so a
        cached reference from an earlier .fit() would silently point at a
        stale, orphaned object that the next predict_proba() never runs
        through -- and any forward hook registered on it would never fire.
        """
        return self._classifier.executor_.model_caches[0].get(self.device)

    def _warm_up(self):
        """Run one trivial .fit() purely to trigger checkpoint loading, so
        embedding_size/num_layers are available before any real episode is
        processed (e.g. at a model's __init__ time, which needs
        backbone.embedding_size before encode_table is ever called)."""
        if self._embedding_size is not None:
            return
        # Distinct rows: identical rows would make every "feature" constant,
        # which TabPFN's preprocessing rejects outright (RemoveConstantFeaturesStep
        # raises "All features are constant") rather than just no-op-ing.
        self._classifier.fit(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), np.array([0, 1]))
        model = self._current_model()
        if hasattr(model, "blocks"):  # TabPFNV2-style
            self._embedding_size = model.embedding_dim
            self._num_layers = len(model.blocks)
        else:  # TabPFNV3-style
            # model.embedding_dim (== icl_emsize) is the POST-aggregation,
            # per-row embedding width of icl_blocks -- not the width of the
            # per-column table this backbone actually hooks. That table's
            # width is model.emsize (== config.embed_dim), the
            # feature_distribution_embedder's own output width.
            self._embedding_size = model.emsize
            self._num_layers = None

    @property
    def embedding_size(self) -> int:
        self._warm_up()
        return self._embedding_size

    @property
    def num_layers(self) -> int | None:
        """Number of hookable layers, or ``None`` for architectures (e.g.
        TabPFNV3) with a single fixed hook point rather than a layer choice."""
        self._warm_up()
        return self._num_layers

    def _hook_target(self, model):
        """The submodule whose forward output is the full per-column-per-row
        (B, R, C, E) table for this checkpoint's architecture. See the module
        docstring for why this differs between TabPFNV2 and TabPFNV3.
        """
        if hasattr(model, "blocks"):  # TabPFNV2-style
            layer_index = self.layer_index if self.layer_index >= 0 else len(model.blocks) + self.layer_index
            return model.blocks[layer_index]
        if hasattr(model, "feature_distribution_embedder"):  # TabPFNV3-style
            return model.feature_distribution_embedder
        raise NotImplementedError(
            f"FrozenTabPFNBackbone does not know how to extract a full per-column "
            f"table from architecture {type(model).__name__}; only TabPFNV2-style "
            "(.blocks) and TabPFNV3-style (.feature_distribution_embedder) are supported."
        )

    def encode_table(
        self,
        src: tuple[torch.Tensor, torch.Tensor],
        train_test_split_index: int,
        num_mem_chunks: int = 1,  # noqa: ARG002 - accepted for interface parity with
        # NanoTabPFNModel.encode_table; real TabPFN manages its own memory via
        # PerformanceOptions, so this repo has no lever to plumb through here.
    ) -> torch.Tensor:
        """Encode a complete support/query table via a frozen real-TabPFN forward pass.

        Mirrors ``NanoTabPFNModel.encode_table``'s signature and row layout
        (support rows before ``train_test_split_index``, query rows after),
        including its contract that ``src[1]`` holds *only* the support
        labels (shape ``(B, split)`` or ``(B, split, 1)``, matching
        ``x_src[:, :split]`` -- nanotabpfn's own ``TargetEncoder`` pads this
        out to the full row count internally). Unlike nanotabpfn's continuous
        target encoder, ``src[1]`` here must be **raw integer class labels**:
        real ``TabPFNClassifier.fit`` is a classifier, not a continuous
        target encoder. Passing ``support_y.float()`` here (as
        ``AttentionSlotRouter.forward`` does for nanotabpfn) is a bug.
        """
        x_src, y_src = src
        split = train_test_split_index
        batch_size = x_src.shape[0]

        outputs = []
        num_columns = None
        for index in range(batch_size):
            support_x_np = x_src[index, :split].detach().cpu().numpy()
            support_y_np = y_src[index].reshape(-1).detach().cpu().numpy()
            query_x_np = x_src[index, split:].detach().cpu().numpy()

            self._classifier.fit(support_x_np, support_y_np)
            model = self._current_model()
            target_module = self._hook_target(model)

            captured: list[torch.Tensor] = []

            def _capture(_module, _inputs, output, captured=captured):
                # Append the raw tensor -- no .clone() here. predict_proba runs
                # inside torch.inference_mode(), and a tensor cloned *inside*
                # that context is still an inference tensor that cannot be
                # used in any later .backward(); the torch.cat below (which
                # creates a fresh tensor, same as .clone()) must happen after
                # the context has exited.
                captured.append(output[0])

            handle = target_module.register_forward_hook(_capture)
            try:
                self._classifier.predict_proba(query_x_np)
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError(
                    f"Forward hook on {type(target_module).__name__} never fired for "
                    f"episode {index}; predict_proba may not have reached it."
                )

            # Row-chunked inference (TabPFNV3's PerformanceOptions.use_chunkwise_inference,
            # for large episodes) fires the hook once per row chunk, in row order --
            # concatenating reconstructs the full table regardless of whether
            # chunking was active (a single-chunk list concatenates to itself).
            x_BRCD = torch.cat(captured, dim=1).to(dtype=self.dtype, device=x_src.device)
            if num_columns is None:
                num_columns = x_BRCD.shape[-2]
            elif x_BRCD.shape[-2] != num_columns:
                raise RuntimeError(
                    f"TabPFN produced {x_BRCD.shape[-2]} columns for episode {index} but "
                    f"{num_columns} for an earlier episode in the same batch -- likely "
                    "per-episode constant-feature removal; encode_table requires a "
                    "stable column count across the batch."
                )
            outputs.append(x_BRCD.squeeze(0))  # drop TabPFN's own n_estimators=1 axis

        return torch.stack(outputs, dim=0)
