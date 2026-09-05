import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tfmplayground.experiments.multiregime_v2 import RegimeGeneratorConfig, sample_regime_episode
from tfmplayground.experiments.pretrain_multiregime_v2 import (
    V2TrainingConfig,
    _checkpoint,
    build_model,
    episode_loss,
    validate_config,
)
from tfmplayground.experiments.pretrain_slot_tabpfn import (
    SlotPretrainingConfig,
    slot_batch_loss,
    slot_training_loss,
)
from tfmplayground.experiments.pretrain_slot_tabpfn import build_model as build_slot_model
from tfmplayground.models.table_slot import TableSlotModel
from tfmplayground.models.slot_regime import (
    load_checkpoint_for_inference,
    slot_mi_loss,
    slot_utilization_scores,
    support_reconstruction_loss,
)


def _config(kind: str, scope: str = "cell_and_data") -> V2TrainingConfig:
    return V2TrainingConfig(
        device="cpu",
        model_type=kind,
        table_slot_scope=scope,
        embedding_size=12,
        num_attention_heads=3,
        mlp_hidden_size=24,
        num_layers=6,
        support_size=4,
        query_size=3,
        min_features=2,
        max_features=2,
        slot_layer_indices=(3, 4, 5),
        validation_episodes=1,
    )


def _episode():
    return sample_regime_episode(
        RegimeGeneratorConfig(support_size=4, query_size=3, min_features=2, max_features=2), num_regimes=2
    )


def test_table_slot_variants_are_feature_aware_and_trainable():
    for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa"):
        config, episode = _config(kind), _episode()
        model = build_model(config)
        loss, _ = episode_loss(model, episode, config)
        loss.backward()
        assert model.last_feature_attention.shape == (1, 7, 3, 4)
        assert model.last_support_attention.shape == (1, 4, 4)
        assert model.last_slots.shape == (1, 4, 12)
        assert model.last_query_gates.shape == (1, 3, 4)
        assert model.last_query_gates.sum(-1).allclose(torch.ones(1, 3))
        names = dict(model.named_parameters())
        assert any("feature_slots" in name and value.grad is not None for name, value in names.items())
        assert any("datapoint_slots" in name and value.grad is not None for name, value in names.items())


def test_single_scope_variants_run_only_their_own_competition():
    """A scope must drop the other path's parameters, not merely stop using it.

    A gated-off path still trains, so "does the cell competition help" could
    only be answered by a model that does not have one.
    """
    for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa"):
        for scope, present, absent in (
            ("cell", "feature_slots", "datapoint_slots"),
            ("data", "datapoint_slots", "feature_slots"),
        ):
            config, episode = _config(kind, scope), _episode()
            model = build_model(config)
            loss, _ = episode_loss(model, episode, config)
            loss.backward()
            names = dict(model.named_parameters())
            assert any(present in name and value.grad is not None for name, value in names.items())
            assert not any(absent in name for name in names)
            # Every scope still answers with the same mixture interface.
            assert model.last_slots.shape == (1, 4, 12)
            assert model.last_support_attention.shape == (1, 4, 4)
            assert model.last_query_gates.sum(-1).allclose(torch.ones(1, 3))
            # Competitive attention sums to one over slots per cell, so the
            # cell scope's column-mean is a distribution without rescaling.
            assert model.last_support_attention.sum(-1).allclose(torch.ones(1, 4))


def test_cell_scope_keeps_feature_attention_and_data_scope_has_none():
    cell = build_model(_config("table_slot_head", "cell"))
    data = build_model(_config("table_slot_head", "data"))
    episode = _episode()
    with torch.no_grad():
        cell(*episode.latent_inputs())
        data(*episode.latent_inputs())
    assert cell.last_feature_attention.shape == (1, 7, 3, 4)
    assert data.last_feature_attention.numel() == 0


def test_single_scope_checkpoints_round_trip():
    for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa"):
        for scope in ("cell", "data"):
            config, episode = _config(kind, scope), _episode()
            model = build_model(config).eval()
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            payload = _checkpoint(model, optimizer, scheduler, config, 0, np.random.default_rng(1), None)
            assert payload["architecture"]["table_slot_scope"] == scope
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pth"
                torch.save(payload, path)
                restored = load_checkpoint_for_inference(path).eval()
                torch.testing.assert_close(
                    model(*episode.latent_inputs()).marginal_log_probabilities(),
                    restored(*episode.latent_inputs()),
                )


def test_scope_is_a_table_slot_setting_only():
    for kind, scope in (("mufasa_slot_tabpfn", "cell"), ("table_slot_head", "rows")):
        try:
            validate_config(_config(kind, scope))
        except ValueError as error:
            assert "table_slot_scope" in str(error)
        else:
            raise AssertionError(f"{kind} with scope {scope} must not validate.")


def test_non_target_features_change_feature_assignments():
    model, episode = build_model(_config("table_slot_head")).eval(), _episode()
    with torch.no_grad():
        model(*episode.latent_inputs())
        before = model.last_feature_attention.clone()
        changed = episode.support_x.clone()
        changed[:, :, 0] += 3.0
        model(changed, episode.support_y, episode.query_x)
    assert not torch.allclose(before[:, :, :-1], model.last_feature_attention[:, :, :-1])


def test_table_slot_checkpoint_round_trip():
    for kind in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa"):
        config, episode = _config(kind), _episode()
        model = build_model(config).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        payload = _checkpoint(model, optimizer, scheduler, config, 0, np.random.default_rng(1), None)
        assert payload["architecture"]["target_inclusive_routing"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(payload, path)
            restored = load_checkpoint_for_inference(path).eval()
            torch.testing.assert_close(
                model(*episode.latent_inputs()).marginal_log_probabilities(), restored(*episode.latent_inputs())
            )


def _head_model_and_episode():
    torch.manual_seed(0)
    return build_model(_config("table_slot_head")).eval(), _episode()


def test_label_blind_pass_is_invariant_to_mean_preserving_relabelling():
    """A permutation of the support labels must not move the reconstruction.

    The label-blind pass exists so the decoder rebuilds a label from the row's
    features rather than from the label itself.  ``TargetEncoder`` already pads
    non-support target cells with ``mean(support_y)``, so passing that constant
    for every row makes the whole target column one value -- and any relabelling
    preserving the mean, permutations included, leaves the pass untouched.
    """
    model, episode = _head_model_and_episode()
    support_x, support_y, query_x = episode.latent_inputs()
    permuted = support_y.flip(1)
    assert not torch.equal(support_y, permuted)
    torch.testing.assert_close(support_y.mean(1), permuted.mean(1), atol=0, rtol=0)

    split = support_x.shape[1]
    table = torch.cat((support_x, query_x), 1)
    with torch.no_grad():
        # One shared state, so the labelled pass is held fixed and the only
        # thing varying is the labels the blind pass is handed.
        state = model.adapters[0](model.backbone.encode_table((table, support_y.float()), split, 1), split)
        blind = [
            model._reconstruct_support(state, model._blind_pass(table, y, split, 1, state), split)
            for y in (support_y, permuted)
        ]
    torch.testing.assert_close(blind[0], blind[1], atol=0, rtol=0)

    # The labelled pass must still see the relabelling, or this passes for the
    # wrong reason: an invariance that holds because nothing depends on y at all.
    with torch.no_grad():
        first = model(support_x, support_y, query_x, reconstruct_support=True)
        second = model(support_x, permuted, query_x, reconstruct_support=True)
    assert not torch.allclose(first.support_attention, second.support_attention)
    assert not torch.allclose(
        first.support_reconstruction_log_probabilities, second.support_reconstruction_log_probabilities
    )


def test_support_reconstruction_is_a_distribution_over_classes():
    model, episode = _head_model_and_episode()
    with torch.no_grad():
        prediction = model(*episode.latent_inputs(), reconstruct_support=True)
    reconstruction = prediction.support_reconstruction_log_probabilities
    assert reconstruction.shape == (1, 4, 2)
    torch.testing.assert_close(reconstruction.exp().sum(-1), torch.ones(1, 4))


def test_reconstruction_gradients_reach_attention_slots_decoder_and_backbone():
    model, episode = _head_model_and_episode()
    model.train()
    support_x, support_y, query_x = episode.latent_inputs()
    support_reconstruction_loss(
        model(support_x, support_y, query_x, reconstruct_support=True), support_y
    ).backward()
    names = dict(model.named_parameters())
    for name in (
        # The competition that produces a[i,k], and the slots it produces.
        "adapters.0.datapoint_slots.project_q.weight",
        "adapters.0.datapoint_slots.slots_mu",
        "adapters.0.datapoint_slots.gru.weight_ih",
        # The shared decoder, reused rather than duplicated.
        "decoder.body.0.weight",
        # And the backbone, through both passes.
        "backbone.target_encoder.linear_layer.weight",
    ):
        assert names[name].grad is not None, name
        assert torch.isfinite(names[name].grad).all(), name


def test_mi_loss_penalizes_uniform_and_collapse_and_bottoms_out_at_balanced_one_hot():
    uniform = torch.full((1, 8, 4), 0.25)
    collapsed = torch.zeros(1, 8, 4)
    collapsed[..., 0] = 1.0
    balanced = torch.zeros(1, 8, 4)
    balanced[0, torch.arange(8), torch.arange(8) % 4] = 1.0
    # Uniform is maximally unsharp; collapse uses one slot.  Each costs exactly
    # one of the two terms, so both score 1.
    torch.testing.assert_close(slot_mi_loss(uniform), torch.tensor(1.0))
    torch.testing.assert_close(slot_mi_loss(collapsed), torch.tensor(1.0))
    torch.testing.assert_close(slot_mi_loss(balanced), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert slot_mi_loss(balanced) < slot_mi_loss(uniform)
    assert slot_mi_loss(balanced) < slot_mi_loss(collapsed)
    # Sharp but unbalanced sits between: the row term is paid, the usage term
    # is not.
    skewed = torch.zeros(1, 8, 4)
    skewed[0, torch.arange(8), (torch.arange(8) < 6).long()] = 1.0
    assert slot_mi_loss(balanced) < slot_mi_loss(skewed) < slot_mi_loss(uniform)


def test_utilization_scores_read_sharpness_and_balance():
    balanced = torch.zeros(1, 8, 4)
    balanced[0, torch.arange(8), torch.arange(8) % 4] = 1.0
    scores = slot_utilization_scores(balanced)
    assert scores["support_row_entropy"] == 0.0
    assert scores["support_utilization_entropy"] == pytest.approx(1.0, abs=1e-6)
    assert scores["support_effective_slots"] == pytest.approx(4.0, abs=1e-5)
    assert scores["support_max_utilization"] == pytest.approx(0.25, abs=1e-6)
    assert scores["support_hard_max_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert [scores[f"support_hard_fraction_{k}"] for k in range(4)] == [pytest.approx(0.25)] * 4


def test_reconstruction_is_a_head_setting_only():
    for kind in ("table_slot_backbone", "table_slot_mufasa"):
        model, episode = build_model(_config(kind)).eval(), _episode()
        with pytest.raises(ValueError, match="head-mode setting"):
            model(*episode.latent_inputs(), reconstruct_support=True)


def test_zero_weights_preserve_the_existing_loss_and_inference_exactly():
    """Both weights at zero must reproduce every earlier run bit for bit.

    The second backbone pass is skipped outright rather than run and multiplied
    by zero, so there is no extra RNG draw and no extra arithmetic to perturb.
    """
    config = SlotPretrainingConfig(
        device="cpu",
        model_kind="table_slot_head",
        num_slots=4,
        embedding_size=12,
        num_attention_heads=3,
        mlp_hidden_size=24,
        num_layers=2,
        max_classes=2,
    )
    torch.manual_seed(0)
    model = build_slot_model(config).eval()
    episode = _episode()
    batch = SimpleNamespace(
        support_x=episode.support_x, support_y=episode.support_y, query_x=episode.query_x, query_y=episode.query_y
    )
    with torch.no_grad():
        total, components = slot_training_loss(model, batch, config)
        torch.testing.assert_close(total, slot_batch_loss(model, batch), atol=0, rtol=0)
    assert components == {}
    with torch.no_grad():
        plain = model(*episode.latent_inputs())
    assert plain.support_reconstruction_log_probabilities is None


def test_closure_checkpoint_round_trips_and_historical_ones_still_load():
    config, episode = _config("table_slot_head"), _episode()
    model = build_model(config).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = _checkpoint(model, optimizer, scheduler, config, 0, np.random.default_rng(1), None)
    payload["architecture"]["support_reconstruction_weight"] = 1.0
    payload["architecture"]["slot_mi_weight"] = 0.05
    expected = model(*episode.latent_inputs()).marginal_log_probabilities()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        torch.save(payload, path)
        torch.testing.assert_close(load_checkpoint_for_inference(path).eval()(*episode.latent_inputs()), expected)
        # A checkpoint written before the closure objective existed carries
        # neither key and must restore identically: they are metadata, not
        # constructor arguments.
        del payload["architecture"]["support_reconstruction_weight"]
        del payload["architecture"]["slot_mi_weight"]
        historical = Path(directory) / "historical.pth"
        torch.save(payload, historical)
        torch.testing.assert_close(
            load_checkpoint_for_inference(historical).eval()(*episode.latent_inputs()), expected
        )


def test_query_routing_modes_all_produce_a_valid_gate_and_leave_support_side_untouched():
    """The three modes are a superset of behavior, not a fork.

    Swapping ``query_routing_mode`` must not change support-side
    reconstruction or the labelled-pass slots/assignment at all -- only which
    embedding and which gate feed the *query* prediction.
    """
    config, episode = _config("table_slot_head"), _episode()
    support_x, support_y, query_x = episode.latent_inputs()
    reference_attention = None
    reference_reconstruction = None
    reference_logits = None
    for mode in ("decoder", "blind_decoder", "blind_similarity"):
        torch.manual_seed(0)
        model = build_model(config).eval()
        model.query_routing_mode = mode
        with torch.no_grad():
            prediction = model(support_x, support_y, query_x, reconstruct_support=True)
        torch.testing.assert_close(prediction.gate().sum(-1), torch.ones(1, 3), atol=1e-5, rtol=0)
        if reference_attention is None:
            reference_attention = prediction.support_attention
            reference_reconstruction = prediction.support_reconstruction_log_probabilities
            reference_logits = prediction.slot_logits
        else:
            torch.testing.assert_close(prediction.support_attention, reference_attention, atol=0, rtol=0)
            torch.testing.assert_close(
                prediction.support_reconstruction_log_probabilities, reference_reconstruction, atol=0, rtol=0
            )
            # ``logits`` must come from the labelled-pass query in every mode --
            # only the gate is allowed to switch to the blind embedding. A model
            # that instead blinds the prediction path too would starve it of the
            # in-context label signal that makes prediction work at all.
            torch.testing.assert_close(prediction.slot_logits, reference_logits, atol=0, rtol=0)


def test_blind_query_embedding_is_invariant_to_label_permutation_but_slots_are_not():
    """The blind *embedding* is invariant; the decoded output need not be.

    ``slot_logits`` also depends on ``state.slots``, which comes from the
    labelled pass and is not blind -- each support row's own true label (not
    just the support mean) shapes its own encoded state, so the slots the
    competition converges to genuinely differ under a permutation.  Checking
    the embedding directly, rather than the full decode, is what isolates the
    claim "blind_decoder" actually makes.
    """
    model, episode = build_model(_config("table_slot_head")).eval(), _episode()
    support_x, support_y, query_x = episode.latent_inputs()
    permuted = support_y.flip(1)
    split = support_x.shape[1]
    table = torch.cat((support_x, query_x), 1)
    with torch.no_grad():
        state = model.adapters[0](model.backbone.encode_table((table, support_y.float()), split, 1), split)
        blind_queries = [
            model._blind_pass(table, y, split, 1, state).pooled_rows[:, split:] for y in (support_y, permuted)
        ]
        # The labelled pass genuinely differs under permutation -- the
        # invariance is specific to the blind embedding, not the whole model.
        labelled_slots = [
            model.adapters[0](model.backbone.encode_table((table, y.float()), split, 1), split).slots
            for y in (support_y, permuted)
        ]
    torch.testing.assert_close(blind_queries[0], blind_queries[1], atol=0, rtol=0)
    assert not torch.allclose(labelled_slots[0], labelled_slots[1])


def test_similarity_gate_routes_a_query_toward_its_nearest_slot_centroid():
    model, episode = build_model(_config("table_slot_head")).eval(), _episode()
    model.query_routing_mode = "blind_similarity"
    support_x, support_y, query_x = episode.latent_inputs()
    with torch.no_grad():
        prediction = model(support_x, support_y, query_x)
    torch.testing.assert_close(prediction.gate().sum(-1), torch.ones(1, 3), atol=1e-5, rtol=0)
    # A query identical to a support row should route toward whatever that
    # support row's own slot competition favoured most.
    with torch.no_grad():
        support_as_query = model(support_x, support_y, support_x)
    top_support_slot = model.last_support_attention.argmax(-1)
    top_query_slot = support_as_query.gate().argmax(-1)
    assert (top_support_slot == top_query_slot).float().mean() > 0.5


def test_query_routing_mode_is_a_head_setting_only():
    for kind, mode in (("table_slot_backbone", "backbone"), ("table_slot_mufasa", "mufasa")):
        model = build_model(_config(kind))
        with pytest.raises(ValueError, match="mode='head'"):
            TableSlotModel(
                model.backbone, mode=mode, num_slots=4, max_classes=2, query_routing_mode="blind_decoder"
            )


def test_query_routing_mode_checkpoint_round_trips_and_defaults_to_decoder():
    config, episode = _config("table_slot_head"), _episode()
    model = build_model(config).eval()
    model.query_routing_mode = "blind_similarity"
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = _checkpoint(model, optimizer, scheduler, config, 0, np.random.default_rng(1), None)
    payload["architecture"]["query_routing_mode"] = "blind_similarity"
    expected = model(*episode.latent_inputs()).marginal_log_probabilities()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        torch.save(payload, path)
        torch.testing.assert_close(load_checkpoint_for_inference(path).eval()(*episode.latent_inputs()), expected)
        # A checkpoint written before this axis existed carries no key and
        # must restore as the historical "decoder" behaviour.
        del payload["architecture"]["query_routing_mode"]
        historical = Path(directory) / "historical.pth"
        torch.save(payload, historical)
        restored = load_checkpoint_for_inference(historical).eval()
        assert restored.model.query_routing_mode == "decoder"


def test_attention_mixture_reproduces_the_historical_reconstruction_exactly():
    """The default compositing rule must be bit-identical to what ran before.

    ``_reconstruct_support`` now always decodes the mask channel, because the
    agreement between the two routings is worth recording and the mask is a
    channel of an output the decoder already produced.  Selecting the historical
    weight afterwards must leave the returned tensor untouched: the grid indices
    are append-only and the closure cells at 180-185 have to stay comparable
    with anything scored against them.
    """
    model, episode = _head_model_and_episode()
    assert model.reconstruction_mixture == "attention"
    support_x, support_y, query_x = episode.latent_inputs()
    split = support_x.shape[1]
    table = torch.cat((support_x, query_x), 1)
    with torch.no_grad():
        state = model.adapters[0](model.backbone.encode_table((table, support_y.float()), split, 1), split)
        blind = model._blind_pass(table, support_y, split, 1, state)
        # The expression as it stood before the compositing axis existed.
        logits, _ = model.decoder(blind.pooled_rows[:, :split], state.slots)
        expected = torch.logsumexp(
            state.support_attention.clamp_min(1e-12).log()[..., None] + torch.log_softmax(logits, -1), dim=2
        )
        torch.testing.assert_close(model._reconstruct_support(state, blind, split), expected, atol=0, rtol=0)


def test_alpha_mixture_composites_by_the_decoder_mask_and_stays_a_distribution():
    """Locatello's rule: softmax the alpha across slots and composite with it.

    The point of the axis is that this is the *same* quantity the query side
    gates on, so the reconstruction now trains the query gate rather than a
    second routing nothing scores.
    """
    model, episode = _head_model_and_episode()
    model.reconstruction_mixture = "alpha"
    support_x, support_y, query_x = episode.latent_inputs()
    split = support_x.shape[1]
    table = torch.cat((support_x, query_x), 1)
    with torch.no_grad():
        state = model.adapters[0](model.backbone.encode_table((table, support_y.float()), split, 1), split)
        blind = model._blind_pass(table, support_y, split, 1, state)
        logits, masks = model.decoder(blind.pooled_rows[:, :split], state.slots)
        expected = torch.logsumexp(
            torch.log_softmax(masks, -1)[..., None] + torch.log_softmax(logits, -1), dim=2
        )
        actual = model._reconstruct_support(state, blind, split)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(1, split))
    # And it is genuinely a different objective, not a renamed one.
    model.reconstruction_mixture = "attention"
    with torch.no_grad():
        assert not torch.allclose(model._reconstruct_support(state, blind, split), actual)


def test_alpha_with_blind_decoder_routes_support_and_query_rows_by_one_function():
    """The property the whole change exists to create.

    Under ``reconstruction_mixture="alpha"`` and
    ``query_routing_mode="blind_decoder"`` the mixture weight is
    ``softmax_k(decoder_mask(blind row, slot k))`` on both sides, so moving a
    row from the support side to the query side must not change the routing it
    receives.  That identity is what makes ``L_rec`` train the query gate:
    before it, the reconstruction weighted by ``a[i,k]`` and the query gated on
    an alpha nothing scored.

    Only the *gate* is shared, deliberately.  The query's class logits still
    come from the labelled pass, because a query row's attended state carries
    the in-context label signal that makes prediction work at all; blinding
    that too would connect the two branches by destroying one of them.
    """
    model, episode = _head_model_and_episode()
    model.reconstruction_mixture = "alpha"
    model.query_routing_mode = "blind_decoder"
    support_x, support_y, query_x = episode.latent_inputs()
    split = support_x.shape[1]
    table = torch.cat((support_x, query_x), 1)
    with torch.no_grad():
        state = model.adapters[0](model.backbone.encode_table((table, support_y.float()), split, 1), split)
        blind = model._blind_pass(table, support_y, split, 1, state)
        # Every row's gate, computed by the one function, support and query alike.
        _, all_masks = model.decoder(blind.pooled_rows, state.slots)
        every_gate = torch.log_softmax(all_masks, -1)
        # The gate the query side actually applies.
        prediction = model(support_x, support_y, query_x, reconstruct_support=True)
    # Default float32 tolerances rather than exact: the model decodes the
    # support and query rows in two calls and this decodes all of them in one,
    # so the matmul reduction order differs by an ulp.  The claim is that the
    # two sides apply the same function, not that they were computed together.
    torch.testing.assert_close(prediction.log_gate, every_gate[:, split:])
    # And the weight the reconstruction actually composites with.
    torch.testing.assert_close(model.last_support_alpha_for_loss, every_gate[:, :split])


def test_alpha_reconstruction_gradients_reach_the_mask_head_and_the_slots():
    """Under "alpha" the loss trains the routing head, which "attention" never did.

    ``a[i,k]`` leaves the loss expression, and that is correct: Locatello's
    attention never appears in the reconstruction loss either.  Gradient still
    reaches the competition, because slots are built as ``weights.T @ v`` and go
    straight into the decoder.
    """
    model, episode = _head_model_and_episode()
    model.reconstruction_mixture = "alpha"
    model.train()
    support_x, support_y, query_x = episode.latent_inputs()
    support_reconstruction_loss(
        model(support_x, support_y, query_x, reconstruct_support=True), support_y
    ).backward()
    names = dict(model.named_parameters())
    for name in (
        # The mask channel is the last row of the decoder's output layer, and
        # it is the query gate: this is the connection the axis exists to make.
        "decoder.body.2.weight",
        # Still the competition and its slots, reached through slot construction
        # rather than through the mixture weight.
        "adapters.0.datapoint_slots.project_q.weight",
        "adapters.0.datapoint_slots.slots_mu",
        "adapters.0.datapoint_slots.gru.weight_ih",
        "backbone.target_encoder.linear_layer.weight",
    ):
        assert names[name].grad is not None, name
        assert torch.isfinite(names[name].grad).all(), name
    mask_row = names["decoder.body.2.weight"].grad[-1]
    assert mask_row.abs().sum() > 0, "the alpha channel took no gradient"


def test_reconstruction_mixture_is_a_head_setting_only():
    for kind, mode in (("table_slot_backbone", "backbone"), ("table_slot_mufasa", "mufasa")):
        model = build_model(_config(kind))
        with pytest.raises(ValueError, match="mode='head'"):
            TableSlotModel(
                model.backbone, mode=mode, num_slots=4, max_classes=2, reconstruction_mixture="alpha"
            )


def test_reconstruction_mixture_checkpoint_round_trips_and_defaults_to_attention():
    config, episode = _config("table_slot_head"), _episode()
    model = build_model(config).eval()
    model.reconstruction_mixture = "alpha"
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = _checkpoint(model, optimizer, scheduler, config, 0, np.random.default_rng(1), None)
    payload["architecture"]["reconstruction_mixture"] = "alpha"
    expected = model(*episode.latent_inputs()).marginal_log_probabilities()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pth"
        torch.save(payload, path)
        restored = load_checkpoint_for_inference(path).eval()
        assert restored.model.reconstruction_mixture == "alpha"
        torch.testing.assert_close(restored(*episode.latent_inputs()), expected)
        # A checkpoint written before this axis existed carries no key and must
        # restore as the historical `a[i,k]` weighting.
        del payload["architecture"]["reconstruction_mixture"]
        historical = Path(directory) / "historical.pth"
        torch.save(payload, historical)
        assert load_checkpoint_for_inference(historical).eval().model.reconstruction_mixture == "attention"


def test_alpha_mixture_requires_a_reconstruction_weight_to_apply_it():
    """Naming a compositing rule at weight zero would report an unapplied objective."""
    from tfmplayground.experiments.pretrain_slot_tabpfn import validate_config as validate_slot_config

    base = dict(device="cpu", model_kind="table_slot_head", num_slots=4, reconstruction_mixture="alpha")
    with pytest.raises(ValueError, match="nonzero"):
        validate_slot_config(SlotPretrainingConfig(**base))
    validate_slot_config(SlotPretrainingConfig(**base, support_reconstruction_weight=1.0))
    with pytest.raises(ValueError, match="table_slot_head"):
        validate_slot_config(
            SlotPretrainingConfig(
                device="cpu",
                model_kind="table_slot_backbone",
                num_slots=4,
                reconstruction_mixture="alpha",
                support_reconstruction_weight=1.0,
            )
        )
