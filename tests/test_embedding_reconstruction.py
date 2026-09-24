"""Embedding reconstruction must learn through slots, not a copied target."""

import numpy as np
import pytest
import torch

from tfmplayground.experiments.pretrain_slot_tabpfn import (
    SlotPretrainingConfig,
    _checkpoint,
    build_model,
    build_parser,
    slot_training_loss,
    validate_config,
)
from tfmplayground.models.slot_regime import embedding_reconstruction_loss, load_checkpoint_for_inference


def config(**kwargs):
    defaults = dict(
        device="cpu",
        model_kind="table_slot_head",
        embedding_size=12,
        num_attention_heads=3,
        mlp_hidden_size=24,
        num_layers=2,
        num_slots=2,
        embedding_reconstruction_weight=0.3,
    )
    defaults.update(kwargs)
    return SlotPretrainingConfig(**defaults)


def batch():
    return {
        "x": torch.randn(1, 7, 2),
        "y": torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]),
        "train_test_split_index": 4,
    }


@pytest.mark.parametrize("scope", ["data", "cell_and_data"])
def test_target_decoder_and_gradients(scope):
    model = build_model(config(table_slot_scope=scope)).eval()
    data = batch()
    inputs, decoder_inputs, tables = [], [], []
    def record_slot_input(_, args):
        inputs.append(args[0])

    h1 = model.adapters[0].datapoint_slots.register_forward_pre_hook(record_slot_input)
    h2 = model.embedding_decoder.register_forward_pre_hook(lambda _, args: decoder_inputs.append(args))
    h0 = model.adapters[0].register_forward_pre_hook(lambda _, args: tables.append(args[0]))
    output = model((data["x"], data["y"][:, :4]), train_test_split_index=4, reconstruct_embeddings=True)
    h0.remove()
    h1.remove()
    h2.remove()
    target = output.support_embedding_target
    assert target.shape == output.support_embedding_reconstruction.shape == (1, 4, 12)
    assert not target.requires_grad
    positions, slots = decoder_inputs[0]
    # Datapoint slots compete independently in each feature column.  The
    # target itself is the support row's target-column token, while the slot
    # input is every support cell plus the shared row position.
    torch.testing.assert_close(target, tables[0][:, :4, -1, :].detach())
    table = tables[0]
    table_view = table.permute(0, 2, 1, 3).reshape(table.shape[0] * table.shape[2], table.shape[1], 12)
    support_cells = table_view[:, :4]
    if scope == "data":
        expected = support_cells + positions.expand(support_cells.shape[0], -1, -1)
        torch.testing.assert_close(inputs[0], expected)
    else:
        # The cell stage intentionally rewrites the table before the datapoint
        # competition; its exact values are covered by table-slot tests.
        assert inputs[0].shape == support_cells.shape
    assert not positions.requires_grad
    values, masks = model.embedding_decoder(positions, slots)
    torch.testing.assert_close(output.support_embedding_reconstruction, (values * masks.softmax(-1)[..., None]).sum(2))
    loss = embedding_reconstruction_loss(output)
    torch.testing.assert_close(loss, ((output.support_embedding_reconstruction - target) ** 2).mean())
    loss.backward()
    for module in (model.embedding_decoder, model.adapters[0].datapoint_slots):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # Position queries do not depend on the row content; fixed slots cannot copy a changed target.
    model.adapters[0].last_embedding_target = torch.randn_like(target)
    values2, masks2 = model.embedding_decoder(positions, slots.detach())
    torch.testing.assert_close(values, values2)
    torch.testing.assert_close(masks, masks2)


def test_training_objective_and_optional_label_reconstruction():
    cfg = config(support_reconstruction_weight=0.7, reconstruction_mixture="alpha")
    model = build_model(cfg).eval()
    loss, metrics = slot_training_loss(model, batch(), cfg)
    assert float(loss.detach()) == pytest.approx(
        metrics["target_loss"] + 0.7 * metrics["reconstruction_nll"] + 0.3 * metrics["embedding_reconstruction_mse"]
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_crossfit_posterior_gate_is_an_auxiliary_only_training_term():
    cfg = config(
        query_routing_mode="posterior_attention",
        table_slot_scope="data",
        posterior_crossfit_weight=0.4,
        embedding_reconstruction_weight=0.0,
    )
    validate_config(cfg)
    model = build_model(cfg).train()
    loss, metrics = slot_training_loss(model, batch(), cfg)
    assert "posterior_crossfit_gate_kl" in metrics
    # The ordinary prediction call carries no teacher branch, so inference
    # remains full-support and does not pay for the cross-fitted passes.
    prediction = model(batch()["x"][:, :4], batch()["y"][:, :4], batch()["x"][:, 4:])
    assert prediction.crossfit_log_gate is None
    assert float(loss.detach()) > metrics["target_loss"]
    loss.backward()


def test_support_reconstruction_can_target_label_embedding():
    cfg = config(
        support_reconstruction_weight=0.7,
        support_reconstruction_target="embedding",
        embedding_reconstruction_weight=0.0,
    )
    model = build_model(cfg).eval()
    loss, metrics = slot_training_loss(model, batch(), cfg)
    assert "reconstruction_embedding_mse" in metrics
    assert "reconstruction_nll" not in metrics
    assert float(loss.detach()) == pytest.approx(
        metrics["target_loss"] + 0.7 * metrics["reconstruction_embedding_mse"]
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_factorized_embedding_reconstruction_uses_pair_slots():
    cfg = config(
        table_slot_scope="cell_and_data",
        slot_composition="factorized",
        query_routing_mode="tabpfn_attention",
        support_reconstruction_weight=0.7,
        support_reconstruction_target="embedding",
        embedding_reconstruction_weight=0.0,
    )
    model = build_model(cfg).eval()
    decoder_slots = []
    handle = model.embedding_decoder.register_forward_pre_hook(lambda _, args: decoder_slots.append(args[1]))
    data = batch()
    model(data["x"][:, :4], data["y"][:, :4], data["x"][:, 4:], reconstruct_embeddings=True)
    handle.remove()
    assert decoder_slots[0].shape == (1, 4, cfg.num_slots * cfg.num_slots, cfg.embedding_size)
    loss, metrics = slot_training_loss(model, data, cfg)
    assert "reconstruction_embedding_mse" in metrics
    assert torch.isfinite(loss)
    loss.backward()


def test_checkpoint_roundtrip_and_no_inference_decoder(tmp_path):
    cfg = config()
    model = build_model(cfg).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = _checkpoint(model, optimizer, scheduler, cfg, 0, None, np.random.default_rng(1))
    path = tmp_path / "embedding.pth"
    torch.save(payload, path)
    restored = load_checkpoint_for_inference(path).model.eval()
    data = batch()
    args = (data["x"][:, :4], data["y"][:, :4], data["x"][:, 4:])
    with torch.no_grad():
        original = model(*args, reconstruct_embeddings=True)
        loaded = restored(*args, reconstruct_embeddings=True)
        torch.testing.assert_close(original.support_embedding_reconstruction, loaded.support_embedding_reconstruction)
        torch.testing.assert_close(original.slot_logits, loaded.slot_logits)
        assert restored(*args).support_embedding_reconstruction is None


def test_embedding_target_checkpoint_rebuilds_decoder(tmp_path):
    cfg = config(support_reconstruction_weight=0.7, support_reconstruction_target="embedding")
    model = build_model(cfg).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = _checkpoint(model, optimizer, scheduler, cfg, 0, None, np.random.default_rng(1))
    assert payload["architecture"]["embedding_reconstruction"] is True
    path = tmp_path / "support-label-embedding.pth"
    torch.save(payload, path)
    restored = load_checkpoint_for_inference(path).model.eval()
    data = batch()
    with torch.no_grad():
        output = restored(
            data["x"][:, :4], data["y"][:, :4], data["x"][:, 4:], reconstruct_embeddings=True
        )
    assert output.support_embedding_reconstruction is not None


def test_validation_and_cli():
    assert (
        build_parser()
        .parse_args(["--output-dir", "unused", "--embedding-reconstruction-weight", "0.5"])
        .embedding_reconstruction_weight
        == 0.5
    )
    assert (
        build_parser()
        .parse_args(["--output-dir", "unused", "--support-reconstruction-target", "embedding"])
        .support_reconstruction_target
        == "embedding"
    )
    with pytest.raises(ValueError):
        validate_config(config(table_slot_scope="cell"))
    with pytest.raises(ValueError, match="data or cell_and_data"):
        validate_config(
            config(
                table_slot_scope="cell",
                support_reconstruction_weight=0.5,
                support_reconstruction_target="embedding",
            )
        )
    disabled = SlotPretrainingConfig(
        device="cpu",
        model_kind="table_slot_head",
        embedding_size=12,
        num_attention_heads=3,
        mlp_hidden_size=24,
        num_layers=2,
    )
    model = build_model(disabled)
    assert not any("embedding_decoder" in key for key in model.state_dict())
    with pytest.raises(ValueError, match="Enable"):
        model(torch.randn(1, 4, 2), torch.zeros(1, 4), torch.randn(1, 3, 2), reconstruct_embeddings=True)
