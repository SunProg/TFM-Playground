import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.evaluate_integrated_tabarena import (
    predict_integrated,
    select_train_indices,
    split_prior_update_counts,
    summarize,
)
from tfmplayground.experiments.train_integrated_latent_filter import (
    IntegratedTrainingConfig,
    integrated_loss,
    run,
    train_segment,
)
from tfmplayground.experiments.train_sequential_latent_filter import (
    SequentialFilterConfig,
    generate_controlled_episodes,
)
from tfmplayground.models.integrated_latent_filter import (
    NanoTabPFNIntegratedLatentFilter,
    backbone_sha256,
    load_integrated_checkpoint,
    save_integrated_checkpoint,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone(num_layers: int = 3) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=num_layers,
        num_outputs=3,
    )


def episode_config(**kwargs) -> SequentialFilterConfig:
    return SequentialFilterConfig(
        device="cpu",
        initial_support_count=8,
        stream_count=8,
        query_count=4,
        batch_size=2,
        controlled_steps=0,
        tabicl_steps=0,
        evaluation_trials=2,
        ordinary_evaluation_batches=0,
        plots=False,
        **kwargs,
    )


class IntegratedLatentFilterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.model = NanoTabPFNIntegratedLatentFilter(tiny_backbone())
        self.batch = generate_controlled_episodes(
            episode_config(), np.random.default_rng(93), condition="noisy", batch_size=2
        )

    def predict(self, model=None, batch=None):
        model = self.model if model is None else model
        batch = self.batch if batch is None else batch
        return model(
            batch.initial_support_x,
            batch.initial_support_y,
            batch.stream_x,
            batch.stream_y,
            batch.query_x,
        )

    def test_zero_adapter_gates_reproduce_vanilla_row_embeddings_exactly(self):
        self.model.eval()
        all_x = torch.cat((self.batch.initial_support_x, self.batch.stream_x, self.batch.query_x), dim=1)
        vanilla = self.model.backbone.encode_table(
            (all_x, self.batch.initial_support_y),
            self.batch.initial_support_x.shape[1],
        )[:, self.batch.initial_support_x.shape[1] :, -1]
        integrated = self.model.raw_logits(
            self.batch.initial_support_x,
            self.batch.initial_support_y,
            self.batch.stream_x,
            self.batch.query_x,
        )
        torch.testing.assert_close(integrated.adapter_gates, torch.zeros(3), rtol=0, atol=0)
        torch.testing.assert_close(integrated.final_row_states, vanilla, rtol=0, atol=0)

    def test_stream_label_cannot_change_own_logits_or_latents(self):
        original = self.predict()
        changed = copy.copy(self.batch)
        changed.stream_y = self.batch.stream_y.clone()
        changed.stream_y[:, 3] = 1 - changed.stream_y[:, 3]
        altered = self.predict(batch=changed)
        torch.testing.assert_close(original.stream_logits[:, 3], altered.stream_logits[:, 3], rtol=0, atol=0)
        torch.testing.assert_close(original.log_weights[:, 3], altered.log_weights[:, 3], rtol=0, atol=0)
        torch.testing.assert_close(original.latent_states, altered.latent_states, rtol=0, atol=0)
        self.assertFalse(torch.equal(original.log_weights[:, 4], altered.log_weights[:, 4]))

    def test_target_rows_do_not_contextualize_each_other(self):
        self.model.eval()
        original = self.predict()
        changed = copy.copy(self.batch)
        changed.query_x = self.batch.query_x.clone()
        changed.query_x[:, 0] += 100
        altered = self.predict(batch=changed)
        torch.testing.assert_close(original.query_logits[:, 1:], altered.query_logits[:, 1:], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original.stream_logits, altered.stream_logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original.latent_states, altered.latent_states, rtol=0, atol=0)

    def test_centered_residuals_and_equal_residual_evidence(self):
        prediction = self.predict()
        torch.testing.assert_close(
            prediction.stream_residuals.sum(2),
            torch.zeros_like(prediction.stream_residuals[..., 0]),
            atol=1e-6,
            rtol=0,
        )
        with torch.no_grad():
            for parameter in self.model.decoder.compatibility_head.parameters():
                parameter.zero_()
        equal = self.predict()
        torch.testing.assert_close(equal.stream_logits[:, :, 0], equal.stream_logits[:, :, 1], rtol=0, atol=0)
        torch.testing.assert_close(equal.log_weights.exp(), torch.full_like(equal.log_weights, 0.5), rtol=0, atol=0)

    def test_weights_and_permutation_invariances(self):
        original = self.predict(self.model.eval())
        self.assertTrue(torch.isfinite(original.log_weights).all())
        torch.testing.assert_close(original.log_weights.exp().sum(-1), torch.ones_like(original.log_weights[..., 0]))
        torch.testing.assert_close(original.log_weights[:, 0].exp(), torch.full((2, 2), 0.5), rtol=0, atol=0)

        order = torch.tensor([4, 1, 7, 0, 5, 2, 6, 3])
        reordered = copy.copy(self.batch)
        reordered.stream_x = self.batch.stream_x[:, order]
        reordered.stream_y = self.batch.stream_y[:, order]
        reordered_prediction = self.predict(batch=reordered)
        torch.testing.assert_close(
            original.log_weights[:, -1], reordered_prediction.log_weights[:, -1], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            original.joint_probabilities()[:, -1],
            reordered_prediction.joint_probabilities()[:, -1],
            atol=1e-6,
            rtol=1e-6,
        )

        query_order = torch.tensor([2, 0, 3, 1])
        permuted = copy.copy(self.batch)
        permuted.query_x = self.batch.query_x[:, query_order]
        permuted_joint = self.predict(batch=permuted).joint_probabilities()
        outcomes = torch.arange(16)
        bits = (outcomes[:, None] >> torch.arange(3, -1, -1)) & 1
        remap = (bits[:, query_order] * (2 ** torch.arange(3, -1, -1))).sum(-1)
        torch.testing.assert_close(original.joint_probabilities(), permuted_joint[:, :, remap], atol=1e-6, rtol=1e-6)

    def test_backbone_gradient_policies(self):
        names = [name for name, _ in self.model.backbone.named_parameters()]
        for stage in ("frozen", "partial", "full"):
            model = copy.deepcopy(self.model)
            model.set_trainability(stage)
            model.zero_grad(set_to_none=True)
            loss, _ = integrated_loss(
                model,
                self.batch,
                IntegratedTrainingConfig(prior_count=128, update_count=128, batch_size=2, plots=False),
            )
            loss.backward()
            gradients = {name: parameter.grad is not None for name, parameter in model.backbone.named_parameters()}
            if stage == "frozen":
                self.assertFalse(any(gradients.values()))
            elif stage == "partial":
                self.assertTrue(any(gradients[name] for name in names if name.startswith("transformer_blocks.2.")))
                self.assertTrue(any(gradients[name] for name in names if name.startswith("transformer_blocks.1.")))
                self.assertFalse(any(gradients[name] for name in names if name.startswith("transformer_blocks.0.")))
                self.assertFalse(any(gradients[name] for name in names if name.startswith("feature_encoder.")))
            else:
                self.assertTrue(any(gradients[name] for name in names if name.startswith("feature_encoder.")))
                self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))

    def test_checkpoint_roundtrip_preserves_predictions_and_metadata(self):
        expected = self.predict(self.model.eval())
        self.model.set_query_temperature(0.6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrated.pth"
            save_integrated_checkpoint(
                path,
                self.model,
                training_config={"seed": 93},
                source_checkpoint_sha256="official",
                stage="frozen",
                lineage={"parent": "initial"},
            )
            loaded, metadata = load_integrated_checkpoint(path)
            actual = self.predict(loaded.eval())
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(expected.stream_logits, actual.stream_logits, rtol=0, atol=0)
        self.assertEqual(loaded.query_temperature, 0.6)
        self.assertEqual(metadata["lineage"], {"parent": "initial"})
        self.assertEqual(metadata["backbone_sha256"], backbone_sha256(loaded))

    def test_curriculum_schedule_matches_fresh_prefix(self):
        config = IntegratedTrainingConfig(
            prior_count=128,
            update_count=128,
            batch_size=1,
            accumulate_gradients=1,
            validation_interval=10,
            patience=2,
            plots=False,
        )
        initial = copy.deepcopy(self.model)
        curriculum, _, _ = train_segment(
            copy.deepcopy(initial), config, candidate="curriculum", trainability="partial", start_step=1, end_step=2
        )
        fresh, history, _ = train_segment(
            copy.deepcopy(initial), config, candidate="fresh", trainability="partial", start_step=0, end_step=2
        )
        self.assertEqual(history[-1]["step"], 2)
        self.assertFalse(
            all(torch.equal(a, b) for a, b in zip(curriculum.parameters(), fresh.parameters(), strict=True))
        )

    def test_tabarena_context_selection_and_chunked_prediction(self):
        labels = np.tile(np.array([0, 1]), 20)
        first = select_train_indices(labels, 16, seed=11)
        second = select_train_indices(labels, 16, seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(labels[first]), {0, 1})
        features = np.random.default_rng(11).normal(size=(25, 2)).astype(np.float32)
        probability = predict_integrated(
            self.model.eval(),
            features[:16],
            labels[:16],
            features[16:],
            prior_count=8,
            update_count=8,
            device="cpu",
            query_chunk_size=4,
            num_mem_chunks=1,
        )
        self.assertEqual(probability.shape, (9,))
        self.assertTrue(np.isfinite(probability).all())
        with self.assertRaisesRegex(ValueError, "Updates cannot exceed"):
            predict_integrated(
                self.model,
                features[:16],
                labels[:16],
                features[16:],
                prior_count=7,
                update_count=8,
                device="cpu",
                query_chunk_size=4,
                num_mem_chunks=1,
            )

    def test_select_train_indices_uses_all_rows_when_count_covers_dataset(self):
        labels = np.tile(np.array([0, 1]), 20)
        selected = select_train_indices(labels, len(labels), seed=5)
        np.testing.assert_array_equal(np.sort(selected), np.arange(len(labels)))
        with self.assertRaisesRegex(ValueError, "Need at least"):
            select_train_indices(labels, len(labels) + 1, seed=5)

    def test_split_prior_update_counts_preserves_ratio(self):
        self.assertEqual(split_prior_update_counts(1000, 128, 128), (500, 500))
        self.assertEqual(split_prior_update_counts(9, 128, 128), (5, 4))
        prior, update = split_prior_update_counts(101, 3, 1)
        self.assertEqual(prior + update, 101)
        self.assertLessEqual(update, prior)
        self.assertGreaterEqual(update, 1)
        with self.assertRaisesRegex(ValueError, "Need at least"):
            split_prior_update_counts(1, 128, 128)

    def test_split_prior_update_counts_never_lets_update_exceed_prior_for_odd_totals(self):
        for total in range(2, 40):
            prior, update = split_prior_update_counts(total, 128, 128)
            self.assertEqual(prior + update, total)
            self.assertLessEqual(update, prior)

    def test_tabarena_summary_is_dataset_macro_average(self):
        metrics = pd.DataFrame(
            [
                {"dataset": "a", "model": "vanilla", "roc_auc": 0.6, "balanced_accuracy": 0.5},
                {"dataset": "a", "model": "controlled", "roc_auc": 0.7, "balanced_accuracy": 0.6},
                {"dataset": "b", "model": "vanilla", "roc_auc": 0.8, "balanced_accuracy": 0.7},
                {"dataset": "b", "model": "controlled", "roc_auc": 0.7, "balanced_accuracy": 0.6},
            ]
        )
        result = summarize(metrics)
        self.assertEqual(result["datasets_evaluated"], 2)
        self.assertAlmostEqual(result["models"]["vanilla"]["mean_roc_auc"], 0.7)
        self.assertAlmostEqual(result["models"]["controlled"]["mean_roc_auc_delta_vs_vanilla"], 0.0)


class IntegratedLatentFilterCLITests(unittest.TestCase):
    def test_tiny_escalation_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "nano.pth"
            backbone = tiny_backbone()
            torch.save(
                {
                    "architecture": {
                        "embedding_size": backbone.embedding_size,
                        "num_attention_heads": backbone.num_attention_heads,
                        "mlp_hidden_size": backbone.mlp_hidden_size,
                        "num_layers": backbone.num_layers,
                        "num_outputs": backbone.num_outputs,
                    },
                    "model": backbone.state_dict(),
                },
                checkpoint,
            )
            output = root / "run"
            config = IntegratedTrainingConfig(
                checkpoint=str(checkpoint),
                output_dir=str(output),
                prior_count=128,
                update_count=128,
                batch_size=1,
                accumulate_gradients=1,
                frozen_steps=1,
                partial_extra_steps=0,
                full_extra_steps=0,
                validation_interval=1,
                patience=1,
                calibration_trials=1,
                selection_trials=1,
                final_trials=1,
                ordinary_evaluation_batches=0,
                tabicl_steps=1,
                plots=False,
            )
            result = run(config)
            self.assertEqual(result, output.resolve())
            for name in (
                "initial_checkpoint.pth",
                "candidate_ranking.csv",
                "selected_checkpoint.pth",
                "final_trajectory_metrics.csv",
                "final_summary.csv",
                "final_gate.json",
                "ordinary_accuracy.json",
                "selection.json",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertFalse((output / "tabicl_checkpoint.pth").exists())


if __name__ == "__main__":
    unittest.main()
