"""Smoke tests for FrozenTabPFNBackbone / AttentionSlotRouterFrozenTabPFN.

Gated behind the optional 'tabpfn' extra AND an explicit opt-in env var:
TabPFNClassifier(model_path="auto") downloads a pretrained checkpoint on
first use, which is undesirable as an unconditional part of the default
test run (no network in most CI environments, and it's slow even with one).
The default ("auto") version also requires one-time browser/API-key license
acceptance (tabpfn.errors.TabPFNLicenseError) before it can download -- set
TFMPLAYGROUND_TABPFN_MODEL_PATH to an already-downloaded checkpoint file to
skip both the download and the license check entirely (tabpfn only checks
the license inside its download path; loading an existing file on disk never
goes through it).

Run explicitly with:
    TFMPLAYGROUND_RUN_REAL_TABPFN_TESTS=1 \\
    TFMPLAYGROUND_TABPFN_MODEL_PATH=/path/to/a/tabpfn-*.ckpt \\
    uv run pytest tests/test_tabpfn_frozen.py
"""

import math
import os

import pytest
import torch

pytest.importorskip("tabpfn")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TFMPLAYGROUND_RUN_REAL_TABPFN_TESTS"),
    reason="Downloads a real TabPFN checkpoint; set TFMPLAYGROUND_RUN_REAL_TABPFN_TESTS=1 to run.",
)

from tfmplayground.models.attention_slot_router import (  # noqa: E402
    AttentionSlotRouterFrozenTabPFN,
    attention_slot_router_loss,
)
from tfmplayground.models.tabpfn_frozen import FrozenTabPFNBackbone  # noqa: E402


def _model_path():
    return os.environ.get("TFMPLAYGROUND_TABPFN_MODEL_PATH")


def _synthetic_batch(batch_size=2, support_size=8, query_size=4, num_features=5, num_classes=2):
    generator = torch.Generator().manual_seed(0)
    support_x = torch.randn(batch_size, support_size, num_features, generator=generator)
    query_x = torch.randn(batch_size, query_size, num_features, generator=generator)
    support_y = torch.randint(0, num_classes, (batch_size, support_size), generator=generator)
    query_y = torch.randint(0, num_classes, (batch_size, query_size), generator=generator)
    return support_x, support_y, query_x, query_y


def test_encode_table_shape_and_grouping():
    support_x, support_y, query_x, _ = _synthetic_batch(num_features=5)
    backbone = FrozenTabPFNBackbone(device="cpu", model_path=_model_path())
    split = support_x.shape[1]
    x = torch.cat((support_x, query_x), 1)
    table = backbone.encode_table((x, support_y), split)
    batch_size, rows, columns, embedding = table.shape
    assert batch_size == support_x.shape[0]
    assert rows == split + query_x.shape[1]
    assert embedding == backbone.embedding_size
    # Column-count contract differs by architecture -- see tabpfn_frozen.py's
    # module docstring. V2 packs features_per_group raw columns into one
    # embedding column; V3 preserves every raw feature as its own column and
    # appends one more for the target (verified empirically, not assumed).
    model = backbone._current_model()
    if hasattr(model, "features_per_group"):
        assert columns == math.ceil(5 / model.features_per_group)
    else:
        assert columns == 5 + 1


def test_encode_table_is_deterministic():
    support_x, support_y, query_x, _ = _synthetic_batch()
    backbone = FrozenTabPFNBackbone(device="cpu", model_path=_model_path())
    split = support_x.shape[1]
    x = torch.cat((support_x, query_x), 1)
    first = backbone.encode_table((x, support_y), split)
    second = backbone.encode_table((x, support_y), split)
    assert torch.allclose(first, second, atol=1e-5)


def test_no_gradient_reaches_tabpfn_parameters():
    support_x, support_y, query_x, query_y = _synthetic_batch()
    model = AttentionSlotRouterFrozenTabPFN(scope="data", hidden=16, num_slots=4, tabpfn_model_path=_model_path())
    prediction, _ = model(support_x, support_y, query_x)
    loss = attention_slot_router_loss(prediction, query_y)
    loss.backward()
    tabpfn_model = model.backbone._current_model()
    assert all(parameter.grad is None for parameter in tabpfn_model.parameters())
    for name, parameter in model.named_parameters():
        assert not name.startswith("backbone.")


def test_mix_scm_episodes_have_a_stable_column_count():
    """Regression test for the constant-column risk documented in
    attention_slot_router_frozen_tabpfn_scale.py's build_prior(): OriginalPrior
    zero-pads each episode's actual feature count up to max_features, and real
    TabPFN's preprocessing drops all-zero (constant) columns per episode. With
    a real min_features/max_features range, different episodes in a batch can
    end up with different post-preprocessing column counts. Fixing
    min_features == max_features (as build_prior does) must keep the column
    count stable across a batch of several such episodes.
    """
    import argparse

    from tfmplayground.experiments.attention_slot_router_frozen_tabpfn_scale import build_prior

    args = argparse.Namespace(prior="mix_scm", support_size=32, query_size=8, features=8)
    sample_episode, _, _, _ = build_prior(args)
    episodes = [sample_episode(seed) for seed in range(4)]
    support_x = torch.cat([e.support_x for e in episodes])
    support_y = torch.cat([e.support_y for e in episodes])
    query_x = torch.cat([e.query_x for e in episodes])

    backbone = FrozenTabPFNBackbone(device="cpu", model_path=_model_path())
    split = support_x.shape[1]
    x = torch.cat((support_x, query_x), 1)
    table = backbone.encode_table((x, support_y), split)  # raises RuntimeError on mismatch
    assert table.shape[0] == len(episodes)
    assert table.shape[-2] == args.features + 1  # +1 for the target column (V3-style)


def test_head_trains_on_a_fixed_batch():
    support_x, support_y, query_x, query_y = _synthetic_batch()
    model = AttentionSlotRouterFrozenTabPFN(scope="data", hidden=16, num_slots=4, tabpfn_model_path=_model_path())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        optimizer.zero_grad()
        prediction, _ = model(support_x, support_y, query_x)
        loss = attention_slot_router_loss(prediction, query_y)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    losses = [step() for _ in range(5)]
    assert losses[-1] < losses[0]
