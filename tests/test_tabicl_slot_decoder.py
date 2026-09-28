import pytest
import torch
from tabicl._model.tabicl import TabICL

from tfmplayground.models.tabicl_slot_decoder import install_slot_decoder, install_tabpfnv3_decoder


def _tiny_tabicl() -> TabICL:
    return TabICL(
        max_classes=6,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        col_target_aware=False,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        col_ssmax=False,
        icl_ssmax=False,
        row_rope_interleaved=False,
        zero_init=False,
    )


@pytest.mark.parametrize("num_slots", [4, 8, 16, 32])
def test_slot_decoder_outputs_finite_multiclass_logits(num_slots: int) -> None:
    torch.manual_seed(123)
    model = _tiny_tabicl()
    router = install_slot_decoder(model, num_slots, seed=9)
    model.train()

    x = torch.randn(2, 11, 4)
    y_train = torch.tensor([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]])
    logits = model(x, y_train)

    assert logits.shape == (2, 5, 6)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.logsumexp(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert router.last_diagnostics["slot_occupancy"].shape == (2, num_slots)
    assert router.last_diagnostics["slot_assignment_entropy"].shape == (2,)


def test_backbone_is_frozen_and_only_slot_parameters_receive_gradients() -> None:
    model = _tiny_tabicl()
    router = install_slot_decoder(model, 4, seed=13)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    router.requires_grad_(True)

    x = torch.randn(1, 9, 3)
    y_train = torch.tensor([[0, 1, 0, 1, 0]])
    loss = torch.nn.functional.cross_entropy(model(x, y_train).reshape(-1, 6), torch.tensor([0, 1, 0, 1]))
    loss.backward()

    assert any(parameter.grad is not None for parameter in router.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("icl_predictor.slot_decoder.")
    )


def test_slot_checkpoint_state_roundtrips_and_ablation_is_usable() -> None:
    torch.manual_seed(321)
    model = _tiny_tabicl()
    router = install_slot_decoder(model, 4, seed=17)
    model.train()
    x = torch.randn(1, 10, 3)
    y_train = torch.tensor([[0, 1, 0, 1, 0, 1]])
    torch.manual_seed(17)
    expected = model(x, y_train)

    restored = _tiny_tabicl()
    install_slot_decoder(restored, 4, seed=17)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.train()
    torch.manual_seed(17)
    actual = restored(x, y_train)
    assert torch.allclose(expected, actual, atol=1e-6)

    restored.icl_predictor.slot_decoder.set_active_slots([0, 1])
    torch.manual_seed(17)
    ablated = restored(x, y_train)
    assert ablated.shape == expected.shape
    assert torch.isfinite(ablated).all()
    assert not torch.allclose(expected, ablated)
    with pytest.raises(ValueError, match="At least one active slot"):
        router.set_active_slots([])


def test_tabpfnv3_decoder_multiclass_gradients_and_roundtrip() -> None:
    pytest.importorskip("tabpfn.architectures.tabpfn_v3")
    torch.manual_seed(24)
    model = _tiny_tabicl()
    model.requires_grad_(False)
    decoder = install_tabpfnv3_decoder(model)
    model.train()
    x = torch.randn(2, 9, 3)
    y_train = torch.tensor([[0, 1, 2, 0, 1], [2, 1, 0, 2, 1]])
    expected = model(x, y_train)
    assert expected.shape == (2, 4, 6)
    assert torch.isfinite(expected).all()
    torch.nn.functional.cross_entropy(
        expected.reshape(-1, 6), torch.tensor([0, 1, 2, 0, 2, 1, 0, 2])
    ).backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("icl_predictor.tabpfnv3_decoder.")
    )

    restored = _tiny_tabicl()
    install_tabpfnv3_decoder(restored)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.train()
    assert torch.allclose(expected, restored(x, y_train), atol=1e-6)
