from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tabicl._model.tabicl import TabICL
from tabicl.train._run import Trainer as OfficialTrainer

from tfmplayground.experiments.pretrain_tabicl_v2_slot_decoder import (
    TabICLSlotDecoderTrainer,
    build_parser,
)
from tfmplayground.models.tabicl_slot_decoder import TabICLSlotDecoderClassifier

TINY_CONFIG = {
    "max_classes": 6,
    "embed_dim": 8,
    "col_num_blocks": 1,
    "col_nhead": 2,
    "col_num_inds": 4,
    "col_target_aware": False,
    "col_feature_group": False,
    "row_num_blocks": 1,
    "row_nhead": 2,
    "row_num_cls": 2,
    "icl_num_blocks": 1,
    "icl_nhead": 2,
    "ff_factor": 2,
    "col_ssmax": False,
    "icl_ssmax": False,
    "row_rope_interleaved": False,
    "zero_init": False,
}


def test_slot_decoder_trainer_reuses_upstream_trainer() -> None:
    assert issubclass(TabICLSlotDecoderTrainer, OfficialTrainer)


def test_default_configuration_matches_tabicl_v2_classifier_stage3() -> None:
    config = build_parser().parse_args(["--device", "cpu", "--checkpoint_dir", "/tmp/tabicl-slot-test"])

    assert config.head_kind == "slot"
    assert config.slot_count == 4
    assert config.max_steps == 10_000
    assert config.batch_size == 64
    assert config.micro_batch_size == 1
    assert config.lr == 2e-5
    assert config.muon is True
    assert config.scheduler == "cosine_with_restarts"
    assert config.gradient_clipping == 1.0
    assert config.prior_type == "graph_scm"
    assert config.min_seq_len == 400
    assert config.max_seq_len == 60_000
    assert config.min_train_size == 0.79
    assert config.max_train_size == 0.81
    assert config.torch_seed == config.np_seed == 2402
    assert config.col_nhead == 8
    assert config.col_affine is False
    assert config.col_feature_group == "same"
    assert config.col_feature_group_size == 3
    assert config.col_target_aware is True
    assert config.col_ssmax is True
    assert config.icl_nhead == 8
    assert config.icl_ssmax is True
    assert config.ssmax_type == "qassmax-mlp-elementwise"
    assert config.ff_factor == 2
    assert config.recompute is False


def test_matched_mlp_head_is_an_explicit_training_mode() -> None:
    config = build_parser().parse_args(
        ["--device", "cpu", "--checkpoint_dir", "/tmp/tabicl-slot-test", "--head-kind", "mlp"]
    )
    assert config.head_kind == "mlp"


@pytest.mark.parametrize("mlp_init", ["pretrained", "random"])
def test_mlp_control_freezes_everything_except_the_original_decoder(tmp_path, monkeypatch, mlp_init) -> None:
    monkeypatch.setattr(TabICLSlotDecoderTrainer, "configure_prior", lambda self: None)
    base_model = TabICL(**TINY_CONFIG)
    base_checkpoint = tmp_path / "released-compatible.ckpt"
    torch.save({"config": TINY_CONFIG, "state_dict": base_model.state_dict()}, base_checkpoint)
    config = build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--amp",
            "False",
            "--use_flash_attn3",
            "False",
            "--muon",
            "False",
            "--head-kind",
            "mlp",
            "--mlp-init",
            mlp_init,
            "--max_classes",
            "6",
            "--checkpoint_dir",
            str(tmp_path / "mlp"),
            "--base-checkpoint",
            str(base_checkpoint),
        ]
    )
    trainer = TabICLSlotDecoderTrainer(config)
    trainable = [name for name, parameter in trainer.raw_model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("icl_predictor.decoder.") for name in trainable)

    matching_weights = all(
        torch.equal(parameter, base_model.state_dict()[name])
        for name, parameter in trainer.raw_model.named_parameters()
        if name.startswith("icl_predictor.decoder.")
    )
    assert matching_weights == (mlp_init == "pretrained")
    assert all(
        torch.equal(parameter, base_model.state_dict()[name])
        for name, parameter in trainer.raw_model.named_parameters()
        if not name.startswith("icl_predictor.decoder.")
    )

    control_checkpoint = tmp_path / "mlp-control.ckpt"
    torch.save(
        {
            "config": TINY_CONFIG,
            "state_dict": base_model.state_dict(),
            "slot_decoder": {"format": "tabicl-slot-decoder-v1", "head_kind": "mlp", "num_slots": None},
        },
        control_checkpoint,
    )
    classifier = TabICLSlotDecoderClassifier(
        model_path=control_checkpoint,
        device="cpu",
        n_estimators=1,
        use_amp=False,
        use_fa3=False,
    )
    X = np.random.default_rng(5).normal(size=(10, 3)).astype(np.float32)
    y = np.array(["a", "b"] * 5)
    classifier.fit(X[:6], y[:6])
    assert classifier.predict_proba(X[6:]).shape == (4, 2)


def test_official_training_smoke_loads_freezes_trains_saves_and_predicts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(TabICLSlotDecoderTrainer, "configure_prior", lambda self: None)
    torch.manual_seed(73)
    base_model = TabICL(**TINY_CONFIG)
    base_checkpoint = tmp_path / "released-compatible.ckpt"
    torch.save({"config": TINY_CONFIG, "state_dict": base_model.state_dict()}, base_checkpoint)

    config = build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--amp",
            "False",
            "--use_flash_attn3",
            "False",
            "--muon",
            "True",
            "--max_steps",
            "1",
            "--batch_size",
            "2",
            "--micro_batch_size",
            "2",
            "--max_classes",
            "6",
            "--max_features",
            "4",
            "--max_seq_len",
            "8",
            "--checkpoint_dir",
            str(tmp_path / "training"),
            "--base-checkpoint",
            str(base_checkpoint),
        ]
    )
    trainer = TabICLSlotDecoderTrainer(config)
    assert trainer.base_checkpoint_path == base_checkpoint
    assert all(
        not parameter.requires_grad
        for name, parameter in trainer.raw_model.named_parameters()
        if not name.startswith("icl_predictor.slot_decoder.")
    )
    assert trainer.raw_model.col_embedder.training is True
    assert trainer.raw_model.icl_predictor.tf_icl.training is True
    assert trainer.raw_model.icl_predictor.slot_decoder.training is True

    # Match the official PriorDataset batch schema and run the upstream loss,
    # gradient clipping, optimizer, and scheduler path for one small batch.
    batch = (
        torch.randn(2, 8, 4),
        torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [1, 0, 2, 1, 0, 2, 1, 0]]),
        torch.tensor([4, 4]),
        torch.tensor([8, 8]),
        torch.tensor([5, 5]),
    )
    before = {name: parameter.detach().clone() for name, parameter in trainer.raw_model.named_parameters()}
    metrics = trainer.run_batch(batch)
    assert np.isfinite(metrics["ce"])
    assert "slot_assignment_entropy" in metrics
    assert trainer.raw_model.col_embedder.training is True
    assert trainer.raw_model.icl_predictor.tf_icl.training is True
    assert trainer.raw_model.icl_predictor.slot_decoder.training is True
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in trainer.raw_model.named_parameters()
        if not name.startswith("icl_predictor.slot_decoder.")
    )
    slot_parameters = [
        parameter
        for name, parameter in trainer.raw_model.named_parameters()
        if name.startswith("icl_predictor.slot_decoder.")
    ]
    backbone_parameters = [
        parameter
        for name, parameter in trainer.raw_model.named_parameters()
        if not name.startswith("icl_predictor.slot_decoder.")
    ]
    assert any(parameter in trainer.optimizer.state for parameter in slot_parameters)
    assert all(parameter not in trainer.optimizer.state for parameter in backbone_parameters)

    saved = tmp_path / "training" / "step-1.ckpt"
    trainer.curr_step = 1
    trainer.save_checkpoint(saved.name)
    saved_checkpoint = torch.load(saved, map_location="cpu", weights_only=True)
    saved_metadata = saved_checkpoint["slot_decoder"]
    assert saved_checkpoint["state_dict_format"] == "decoder_only_v1"
    assert set(saved_checkpoint["state_dict"]) == set(trainer.raw_model.icl_predictor.slot_decoder.state_dict())
    expected_decoder_state = {
        key: value.detach().clone()
        for key, value in trainer.raw_model.icl_predictor.slot_decoder.state_dict().items()
    }
    with torch.no_grad():
        for parameter in trainer.raw_model.icl_predictor.slot_decoder.parameters():
            parameter.zero_()
    trainer.curr_step = 0
    trainer.load_checkpoint()
    assert trainer.curr_step == 1
    assert all(
        torch.equal(expected_decoder_state[key], value)
        for key, value in trainer.raw_model.icl_predictor.slot_decoder.state_dict().items()
    )
    assert saved_metadata["slot_configuration"]["slot_size"] == 16
    assert saved_metadata["slot_configuration"]["num_heads"] == 8
    assert saved_metadata["tabicl_version"] == "2.2.0"
    assert saved_metadata["base_checkpoint_version"] == "tabicl-classifier-v2-20260212.ckpt"
    assert saved_metadata["prior"]["graph_scm"]["filter_unpredictable_graphs"] is True
    metrics_rows = [json.loads(row) for row in (tmp_path / "training" / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics_rows) == 1 and metrics_rows[0]["step"] == 1

    # The adapter must reconstruct the custom head and preserve sklearn inference.
    classifier = TabICLSlotDecoderClassifier(
        model_path=saved,
        expected_num_slots=4,
        device="cpu",
        n_estimators=1,
        use_amp=False,
        use_fa3=False,
    )
    X = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
    y = np.array(["a", "b"] * 6)
    classifier.fit(X[:8], y[:8])
    probabilities = classifier.predict_proba(X[8:])
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    diagnostic_probabilities, diagnostics = classifier.predict_proba_with_slot_diagnostics(X[8:])
    assert np.allclose(probabilities, diagnostic_probabilities, atol=1e-6)
    assert diagnostics["support_assignments"].shape[-1] == 4
    assert "slot_occupancy" in diagnostics
    ablated = classifier.predict_proba_with_slot_mask(X[8:], [1, 2, 3])
    assert ablated.shape == probabilities.shape
    assert not np.allclose(probabilities, ablated)
