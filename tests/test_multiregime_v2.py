import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.benchmark_multiregime_v2 import (
    BenchmarkConfig,
    analytic_grid,
    binary_metrics,
    oracle_hgb_predictions,
    recovery_score,
    run_benchmark,
)
from tfmplayground.experiments.multiregime_prior_dump import PriorDumpReader, write_prior_dump
from tfmplayground.experiments.multiregime_v2 import (
    RegimeGeneratorConfig,
    legacy_single_regime_batch,
    sample_regime_episode,
    tensor_hash,
)
from tfmplayground.experiments.pretrain_multiregime_v2 import (
    V2TrainingConfig,
    build_model,
    curriculum_cell,
    episode_loss,
    hungarian_auxiliary_loss,
    run_pretraining,
    select_auxiliary_weight,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_backbone import SlotTransformerEncoderLayer, install_slot_layers
from tfmplayground.models.slot_regime import SlotRegimePrediction, load_checkpoint_for_inference


class GeneratorTests(unittest.TestCase):
    def config(self, **kwargs):
        return RegimeGeneratorConfig(
            support_size=32,
            query_size=16,
            min_features=4,
            max_features=4,
            seed=17,
            **kwargs,
        )

    def test_samples_x_once_and_has_padded_reproducible_shapes(self):
        calls = []

        def sampler(rng, rows, features):
            calls.append((rows, features))
            return rng.normal(size=(rows, features))

        first = sample_regime_episode(self.config(max_regimes=4), num_regimes=3, x_sampler=sampler)
        second = sample_regime_episode(self.config(max_regimes=4), num_regimes=3)
        self.assertEqual(calls, [(48, 4)])
        self.assertEqual(first.support_x.shape, (1, 32, 4))
        self.assertEqual(first.counterfactual_query_probabilities.shape, (1, 4, 16))
        self.assertEqual(first.active_regime_mask.tolist(), [[True, True, True, False]])
        self.assertEqual(tensor_hash(first), tensor_hash(second))

    def test_alpha_zero_is_identical_and_distance_is_monotone(self):
        episodes = [
            sample_regime_episode(self.config(regime_separation=alpha), num_regimes=4)
            for alpha in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]
        zero = episodes[0].counterfactual_query_probabilities[0, :4]
        torch.testing.assert_close(zero, zero[:1].expand_as(zero), atol=0, rtol=0)
        distances = [episode.metadata["realized_score_distance"] for episode in episodes]
        self.assertEqual(distances, sorted(distances))
        np.testing.assert_allclose(distances, (0.0, 0.25, 0.5, 1.0, 2.0), atol=1e-12)

    def test_gate_calibration_and_x_independent_control(self):
        episode = sample_regime_episode(
            self.config(imbalance_ratio=0.05, gate_strength=0.0), num_regimes=4
        )
        requested = np.asarray(episode.metadata["requested_proportions"])
        realized = np.asarray(episode.metadata["realized_gate_proportions"])
        np.testing.assert_allclose(realized, requested, atol=1e-10)
        probabilities = torch.cat(
            (episode.support_gate_probabilities, episode.query_gate_probabilities), dim=1
        )[0, :, :4]
        torch.testing.assert_close(probabilities, probabilities[:1].expand_as(probabilities), atol=1e-7, rtol=0)
        self.assertAlmostEqual(float(requested.min() / requested.max()), 0.05)

    def test_noise_and_ratio_edges_are_validated(self):
        with self.assertRaises(ValueError):
            self.config(label_noise=0.51)
        with self.assertRaises(ValueError):
            self.config(imbalance_ratio=0.0)
        noisy = sample_regime_episode(self.config(label_noise=0.5), num_regimes=2)
        self.assertTrue(torch.all(noisy.counterfactual_query_probabilities[:, :2] == 0.5))

    def test_z_and_metadata_are_absent_from_latent_inputs(self):
        episode = sample_regime_episode(self.config(), num_regimes=2)
        inputs = tuple(tensor.clone() for tensor in episode.latent_inputs())
        altered = copy.deepcopy(episode)
        altered.metadata["num_regimes"] = 999
        altered.support_z.fill_(1)
        for expected, actual in zip(inputs, altered.latent_inputs(), strict=True):
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_unsupported_scm_analytic_flags_fail_validation(self):
        with self.assertRaises(ValueError):
            self.config(backend="tabicl_scm", difference_components=("nonlinear",))

    def test_legacy_k1_delegates_without_copying_or_transforming(self):
        batch = {"x": torch.randn(1, 4, 2), "y": torch.zeros(1, 4)}

        class Prior:
            def __iter__(self):
                return iter((batch,))

        self.assertIs(legacy_single_regime_batch(Prior()), batch)

    def test_prior_dump_replays_the_online_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / "dump"
            write_prior_dump(
                dump,
                seed=2402,
                max_steps=1,
                micro_batch_size=2,
                accumulate_gradients=1,
                support_size=8,
                query_size=4,
                min_features=2,
                max_features=2,
                shard_episodes=1,
            )
            reader = PriorDumpReader(dump)
            first, metadata = reader.next_episode()
            second, _ = reader.next_episode()
            self.assertEqual(reader.index, 2)
            self.assertEqual(first.metadata["episode_id"], metadata["episode_id"])
            self.assertNotEqual(tensor_hash(first), tensor_hash(second))


class BenchmarkTests(unittest.TestCase):
    def test_grid_has_305_unique_cells(self):
        grid = analytic_grid()
        self.assertEqual(len(grid), 305)
        serialized = {json.dumps(cell, sort_keys=True) for cell in grid}
        self.assertEqual(len(serialized), 305)

    def test_perfect_constant_and_undefined_metrics(self):
        perfect = binary_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(perfect["balanced_accuracy"], 1.0)
        self.assertLess(perfect["log_loss"], 1e-12)
        constant = binary_metrics(np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertEqual(constant["auroc"], 0.5)
        one_class = binary_metrics(np.zeros(4), np.full(4, 0.2))
        self.assertIsNone(one_class["auroc"])
        self.assertEqual(one_class["auroc_reason"], "single_observed_class")
        self.assertIsNone(one_class["auprc"])

    def test_recovery_is_unclipped_and_null_for_zero_denominator(self):
        self.assertAlmostEqual(recovery_score("log_loss", 0.2, 0.8, 0.4)["value"], 1.5)
        result = recovery_score("auroc", 0.7, 0.6, 0.6000001)
        self.assertIsNone(result["value"])
        self.assertEqual(result["reason"], "oracle_not_better_than_pooled")

    def test_oracle_missing_support_falls_back_without_crashing(self):
        episode = sample_regime_episode(
            RegimeGeneratorConfig(
                support_size=1,
                query_size=32,
                min_features=2,
                max_features=2,
                imbalance_ratio=0.05,
                seed=3,
            ),
            num_regimes=4,
        )
        pooled = np.linspace(0.1, 0.9, 32)
        prediction, info = oracle_hgb_predictions(episode, pooled)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(len(info["missing_support_regimes"]), 0)
        query_z = episode.query_z[0].numpy()
        for regime in info["missing_support_regimes"]:
            np.testing.assert_allclose(prediction[query_z == regime], pooled[query_z == regime])

    def test_tiny_locked_run_is_non_overwriting_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first"
            config = BenchmarkConfig(
                output_dir=str(first_path),
                generator=RegimeGeneratorConfig(min_features=2, max_features=2),
                evaluation_seeds=(7,),
                episodes_per_seed=1,
                query_size=8,
                bootstrap_samples=20,
                include_main_grid=False,
                include_mechanisms=True,
                include_scm_grid=False,
            )
            run_benchmark(config)
            first = json.loads((first_path / "episode_manifest.json").read_text())
            second_path = Path(directory) / "second"
            run_benchmark(replace(config, output_dir=str(second_path)))
            second = json.loads((second_path / "episode_manifest.json").read_text())
            self.assertEqual(first, second)
            with self.assertRaises(FileExistsError):
                run_benchmark(config)
            self.assertEqual(len((first_path / "episode_metrics.csv").read_text().splitlines()), 49)


class ModelAndTrainingTests(unittest.TestCase):
    @staticmethod
    def architecture():
        return {
            "embedding_size": 8,
            "num_attention_heads": 2,
            "mlp_hidden_size": 16,
            "num_layers": 3,
            "num_outputs": 2,
        }

    def test_layer_selection_defaults_all_but_v2_selects_exactly_one(self):
        all_layers = install_slot_layers(NanoTabPFNModel(**self.architecture()))
        all_count = sum(isinstance(layer, SlotTransformerEncoderLayer) for layer in all_layers.transformer_blocks)
        self.assertEqual(all_count, 3)
        one_layer = install_slot_layers(NanoTabPFNModel(**self.architecture()), layer_indices=(0,))
        slots = [
            index
            for index, layer in enumerate(one_layer.transformer_blocks)
            if isinstance(layer, SlotTransformerEncoderLayer)
        ]
        self.assertEqual(slots, [0])
        self.assertEqual(one_layer.transformer_blocks[0].slot_position, "after_datapoint")

    def test_curriculum_phase_boundaries_and_lambda_tie_break(self):
        rng = np.random.default_rng(1)
        self.assertEqual(curriculum_cell(20, 100, rng).num_regimes, 1)
        self.assertIn(curriculum_cell(21, 100, rng).num_regimes, (1, 2))
        self.assertEqual(curriculum_cell(40, 100, rng).alpha, 0.25)
        self.assertIn(curriculum_cell(41, 100, rng).alpha, (0.25, 0.5, 1.0))
        self.assertEqual(select_auxiliary_weight({0.01: 0.3, 0.1: 0.3, 1.0: 0.4}), 0.01)

    def test_hungarian_loss_is_slot_permutation_invariant_for_s_greater_k(self):
        support = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]], requires_grad=True)
        gate = torch.tensor([[[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]], requires_grad=True)
        prediction = SlotRegimePrediction(
            slot_logits=torch.randn(1, 2, 3, 2, requires_grad=True),
            log_gate=gate.log(),
            support_attention=support,
        )
        episode = sample_regime_episode(
            RegimeGeneratorConfig(support_size=2, query_size=2, min_features=2, max_features=2, seed=2),
            num_regimes=2,
        )
        episode.support_z[:] = torch.tensor([[0, 1]])
        episode.query_z[:] = torch.tensor([[0, 1]])
        loss, _ = hungarian_auxiliary_loss(prediction, episode)
        permutation = torch.tensor([2, 0, 1])
        permuted = SlotRegimePrediction(
            slot_logits=prediction.slot_logits[:, :, permutation],
            log_gate=prediction.log_gate[:, :, permutation],
            support_attention=prediction.support_attention[:, :, permutation],
        )
        other, _ = hungarian_auxiliary_loss(permuted, episode)
        torch.testing.assert_close(loss, other)

    def test_target_and_auxiliary_gradients_reach_every_slot_path(self):
        config = V2TrainingConfig(
            device="cpu",
            model_type="slot_tabpfn",
            aux_regime_weight=0.1,
            num_slots=4,
            support_size=8,
            query_size=4,
            min_features=2,
            max_features=2,
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=2,
        )
        model = build_model(config)
        episode = sample_regime_episode(config.generator(), num_regimes=3).to("cpu")
        loss, _ = episode_loss(model, episode, config)
        loss.backward()
        layer = model.backbone.transformer_blocks[0]
        self.assertIsNotNone(layer.slot_attention.slots_mu.grad)
        self.assertIsNotNone(layer.write_back.in_proj_weight.grad)
        self.assertIsNotNone(layer.slot_mix.grad)
        self.assertIsNotNone(model.decoder.body[0].weight.grad)

    def test_mufasa_fuses_selected_layers_and_backpropagates(self):
        config = V2TrainingConfig(
            device="cpu",
            model_type="mufasa_slot_tabpfn",
            slot_layer_indices=(0, 1),
            aux_regime_weight=0.1,
            num_slots=4,
            support_size=8,
            query_size=4,
            min_features=2,
            max_features=2,
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=2,
        )
        model = build_model(config)
        episode = sample_regime_episode(config.generator(), num_regimes=3).to("cpu")
        loss, _ = episode_loss(model, episode, config)
        prediction = model(episode.support_x, episode.support_y, episode.query_x)
        self.assertEqual(prediction.slot_logits.shape, (1, 4, 4, 2))
        self.assertEqual(prediction.support_attention.shape, (1, 8, 4))
        loss.backward()
        self.assertIsNotNone(model.slot_modules[0].project_q.weight.grad)
        self.assertIsNotNone(model.slot_modules[1].project_q.weight.grad)
        self.assertIsNotNone(model.decoder.body[0].weight.grad)

    def test_checkpoint_roundtrip_preserves_v2_schema_and_one_slot_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = V2TrainingConfig(
                device="cpu",
                max_steps=1,
                micro_batch_size=1,
                accumulate_gradients=1,
                warmup_steps=1,
                validation_interval=1,
                validation_episodes=1,
                checkpoint_interval=1,
                model_type="slot_tabpfn",
                num_slots=4,
                support_size=8,
                query_size=4,
                min_features=2,
                max_features=2,
                embedding_size=8,
                num_attention_heads=2,
                mlp_hidden_size=16,
                num_layers=2,
            )
            checkpoint = run_pretraining(config, Path(directory) / "run")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(state["architecture"]["slot_layer_index"], 0)
            self.assertEqual(state["generator_schema"]["max_regimes"], 4)
            restored = load_checkpoint_for_inference(checkpoint)
            layers = restored.model.backbone.transformer_blocks
            self.assertEqual(sum(isinstance(layer, SlotTransformerEncoderLayer) for layer in layers), 1)

    def test_tiny_cpu_runs_for_standard_supervised_slot_and_oracle(self):
        modes = (
            {"model_type": "tabpfn", "input_mode": "latent", "aux_regime_weight": 0.0},
            {"model_type": "slot_tabpfn", "input_mode": "latent", "aux_regime_weight": 0.0},
            {"model_type": "slot_tabpfn", "input_mode": "latent", "aux_regime_weight": 0.1},
            {"model_type": "tabpfn", "input_mode": "oracle_one_hot", "aux_regime_weight": 0.0},
            {
                "model_type": "mufasa_slot_tabpfn",
                "input_mode": "latent",
                "aux_regime_weight": 0.0,
                "slot_layer_indices": (0,),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mode in enumerate(modes):
                config = V2TrainingConfig(
                    device="cpu",
                    max_steps=1,
                    micro_batch_size=1,
                    accumulate_gradients=1,
                    warmup_steps=1,
                    validation_interval=1,
                    validation_episodes=1,
                    checkpoint_interval=0,
                    support_size=6,
                    query_size=3,
                    min_features=2,
                    max_features=2,
                    embedding_size=8,
                    num_attention_heads=2,
                    mlp_hidden_size=16,
                    num_layers=1,
                    num_slots=4,
                    **mode,
                )
                self.assertTrue(run_pretraining(config, Path(directory) / f"mode-{index}").is_file())

    def test_resume_reproduces_the_same_next_batch_loss_and_weights(self):
        config = V2TrainingConfig(
            device="cpu",
            max_steps=2,
            micro_batch_size=1,
            accumulate_gradients=1,
            warmup_steps=1,
            validation_interval=2,
            validation_episodes=1,
            checkpoint_interval=1,
            support_size=6,
            query_size=3,
            min_features=2,
            max_features=2,
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            uninterrupted_dir = Path(directory) / "uninterrupted"
            uninterrupted = run_pretraining(config, uninterrupted_dir)
            resumed_dir = Path(directory) / "resumed"
            resumed_dir.mkdir()
            resumed = run_pretraining(
                config,
                resumed_dir,
                resume_checkpoint=uninterrupted_dir / "step-000001-checkpoint.pth",
            )
            expected = torch.load(uninterrupted, map_location="cpu", weights_only=False)
            actual = torch.load(resumed, map_location="cpu", weights_only=False)
            self.assertEqual(expected["episode_rng_state"], actual["episode_rng_state"])
            for name in expected["model"]:
                torch.testing.assert_close(expected["model"][name], actual["model"][name], atol=0, rtol=0)
            expected_row = json.loads((uninterrupted_dir / "history.jsonl").read_text().splitlines()[-1])
            actual_row = json.loads((resumed_dir / "history.jsonl").read_text().splitlines()[-1])
            self.assertEqual(expected_row["loss"], actual_row["loss"])


if __name__ == "__main__":
    unittest.main()
