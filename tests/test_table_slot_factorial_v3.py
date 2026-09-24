import pytest
import torch
import torch.nn.functional as F

from tfmplayground.experiments.table_slot_factorial_v3 import (
    SLOT_USEFUL_LOSS_MODES,
    PilotConfig,
    architecture,
    build_model,
    configurations,
    early_schedule,
    heldout_partition,
    responsibility_losses,
    training_step,
    usage_loss,
)
from tfmplayground.models.slot_regime import SlotRegimePrediction, load_checkpoint_for_inference
from tfmplayground.models.table_slot import TableSlotModel


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return PilotConfig(
        seed=11,
        width=12,
        hidden=24,
        layers=2,
        heads=3,
        support_size=8,
        query_size=4,
        micro_batch_size=2,
        accumulate_gradients=1,
        steps=2,
        warmup_steps=1,
    )


def tensors():
    torch.manual_seed(12)
    return torch.randn(2, 8, 3), torch.tensor([[0, 1] * 4] * 2).float(), torch.randn(2, 4, 3), torch.randint(2, (2, 4))


@pytest.mark.parametrize("index", range(25))
def test_each_scope_switch_and_plain_trains(config, index):
    cell = configurations()[index]
    model, _ = build_model(config, cell)
    if index != 24:
        assert type(model) is TableSlotModel and model.mode == "head" and model.scope == cell["scope"]
    before = model.backbone.feature_encoder if index != 24 else model.feature_encoder
    parameter = next(before.parameters())
    saved = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = training_step(model, tensors(), cell, 0, config, optimizer)
    assert stats["query_nll"] > 0
    assert not torch.equal(saved, parameter)
    assert (stats["heldout_gate"] > 0) == (2 in cell["fixes"])
    assert (stats["usage_weight"] > 0) == (3 in cell["fixes"])


@pytest.mark.parametrize("scope_offset", (0, 8, 16))
def test_blind_content_uses_labelled_slots(config, scope_offset):
    model, _ = build_model(config, configurations()[scope_offset + 1])
    model.eval()
    sx, sy, qx, _ = tensors()
    captured = []
    hook = model.decoder.register_forward_pre_hook(
        lambda module, inputs: captured.append(tuple(v.detach().clone() for v in inputs))
    )
    model(sx, sy, qx)
    model(sx, 1 - sy, qx)  # preserves the mean label exactly
    hook.remove()
    assert len(captured) == 2
    torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)
    assert not torch.allclose(captured[0][1], captured[1][1], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("index", (2, 4, 7, 10, 12, 15, 18, 20, 23))
def test_heldout_labels_cannot_change_predictions_or_slots(config, index):
    sx, sy, _, _ = tensors()
    cx, cy, hx, hy = heldout_partition(sx, sy, seed=41)
    model, _ = build_model(config, configurations()[index])
    model.eval()
    first = model(cx, cy, hx)
    slots = model.last_slots.detach().clone()
    # Alter only held-out labels in the original support table and partition
    # again. The feature/label inputs to the second model pass must be equal.
    order = torch.randperm(sx.shape[1], generator=torch.Generator().manual_seed(41))
    changed = sy.clone()
    changed[:, order[:2]] = 1 - changed[:, order[:2]]
    cx2, cy2, hx2, hy2 = heldout_partition(sx, changed, seed=41)
    torch.testing.assert_close(cy, cy2)
    assert not torch.equal(hy, hy2)
    second = model(cx2, cy2, hx2)
    torch.testing.assert_close(slots, model.last_slots, rtol=0, atol=0)
    torch.testing.assert_close(first.slot_logits, second.slot_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.log_gate, second.log_gate, rtol=0, atol=0)


def test_responsibility_loss_matches_heldout_marginal_gradient():
    logits = torch.randn(2, 3, 4, 2, requires_grad=True)
    gate_logits = torch.randn(2, 3, 4, requires_grad=True)
    prediction = SlotRegimePrediction(logits, gate_logits.log_softmax(-1), torch.empty(0))
    labels = torch.randint(2, (2, 3))
    gate, expert = responsibility_losses(prediction, labels)
    em_grad = torch.autograd.grad(gate + expert, (logits, gate_logits), retain_graph=True)
    nll = F.nll_loss(prediction.marginal_log_probabilities().flatten(0, 1), labels.flatten())
    marginal_grad = torch.autograd.grad(nll, (logits, gate_logits))
    for first, second in zip(em_grad, marginal_grad, strict=True):
        torch.testing.assert_close(first, second)


def test_temporary_usage_loss_and_temperature():
    assert early_schedule(0, True, 2000) == (2.0, 0.01)
    assert early_schedule(2000, True, 2000) == (1.0, 0.0)
    assert early_schedule(5000, True, 2000) == (1.0, 0.0)
    assert early_schedule(0, False, 2000) == (1.0, 0.0)
    logits = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    usage_loss(logits.log_softmax(-1)).backward()
    assert logits.grad[0, 0, 0] > 0 and (logits.grad[0, 0, 1:] < 0).all()


@pytest.mark.parametrize("mode", ("reconstruction", "reconstruction_mi"))
def test_slot_useful_loss_supervises_support_and_uses_alpha(config, mode):
    cell = dict(configurations()[1], slot_useful_loss=mode)
    model, _ = build_model(config, cell)
    assert model.reconstruction_mixture == "alpha"
    stats = training_step(model, tensors(), cell, 0, config, torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert stats["slot_reconstruction_nll"] > 0
    assert (stats["slot_mi_loss"] > 0) == (mode == "reconstruction_mi")


def test_slot_useful_loss_modes_are_append_only():
    assert SLOT_USEFUL_LOSS_MODES == ("none", "reconstruction", "reconstruction_mi")


def test_tabpfn_attention_routing_is_an_append_only_training_mode(config):
    cell = dict(
        configurations()[1],
        query_routing_mode="tabpfn_attention",
        decoder_interaction="product",
    )
    model, _ = build_model(config, cell)
    stats = training_step(model, tensors(), cell, 0, config, torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert model.query_routing_mode == "tabpfn_attention"
    assert model.decoder_interaction == "product"
    assert model.decoder.body[0].in_features == config.width
    assert model.last_query_support_attention.shape == (2, config.query_size, config.support_size)
    assert stats["query_nll"] > 0


@pytest.mark.parametrize("index", (0, 1, 9, 17, 24))
def test_checkpoint_roundtrip_preserves_content_mode(config, tmp_path, index):
    cell = configurations()[index]
    model, _ = build_model(config, cell)
    model.eval()
    path = tmp_path / "checkpoint.pth"
    torch.save(dict(model=model.state_dict(), architecture=architecture(model, config)), path)
    restored = load_checkpoint_for_inference(path).eval()
    sx, sy, qx, _ = tensors()
    prediction = model(sx, sy, qx)
    expected = prediction.marginal_log_probabilities() if index != 24 else prediction
    torch.testing.assert_close(restored(sx, sy, qx), expected)


def test_factorial_and_identical_backbone_initialization(config):
    cells = configurations()
    assert len(cells) == 75
    assert len({(c["prior"], c["scope"], c["fixes"]) for c in cells}) == 75
    assert len({build_model(config, c)[1] for c in cells[:25]}) == 1
    for offset in (0, 8, 16):
        baseline, _ = build_model(config, cells[offset])
        changed, _ = build_model(config, cells[offset + 7])
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, changed.state_dict()[key], atol=0, rtol=0)


def test_default_forward_and_explicit_temperature_one_identical(config):
    model, _ = build_model(config, configurations()[0])
    model.eval()
    sx, sy, qx, _ = tensors()
    first = model(sx, sy, qx)
    second = model(sx, sy, qx, gate_temperature=1.0)
    torch.testing.assert_close(first.slot_logits, second.slot_logits, atol=0, rtol=0)
    torch.testing.assert_close(first.log_gate, second.log_gate, atol=0, rtol=0)
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            model(sx, sy, qx, gate_temperature=value)
