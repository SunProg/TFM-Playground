import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import h5py

from tfmplayground.experiments.prior_bimodal_episodes import (
    PriorBimodalConfig,
    generate_h5_prior_bimodal_episodes,
    generate_prior_bimodal_episodes,
)
from tfmplayground.models.integrated_latent_filter import NanoTabPFNIntegratedLatentFilter
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=2,
        num_outputs=3,
    )


def episode_config(**kwargs) -> PriorBimodalConfig:
    values = dict(
        initial_support_count=8,
        stream_count=8,
        query_count=4,
        min_features=1,
        max_features=2,
        max_pair_attempts=500,
        device="cpu",
    )
    values.update(kwargs)
    return PriorBimodalConfig(**values)


class PriorBimodalEpisodeTests(unittest.TestCase):
    def test_h5_loader_pairs_empirical_labels_lazily(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.h5"
            x = np.arange(2 * 70 * 2, dtype=np.float32).reshape(2, 70, 2)
            y = np.zeros((2, 70), dtype=np.float32)
            y[1, 32:68] = 1.0
            with h5py.File(path, "w") as handle:
                handle.create_dataset("X", data=x)
                handle.create_dataset("y", data=y)
                handle.create_dataset("num_features", data=np.array([2, 2], dtype=np.int32))
                handle.create_dataset("single_eval_pos", data=np.array([64, 64], dtype=np.int32))
            batch = generate_h5_prior_bimodal_episodes(
                str(path),
                PriorBimodalConfig(
                    initial_support_count=32,
                    stream_count=32,
                    query_count=4,
                    min_features=1,
                    max_features=2,
                    max_pair_attempts=8,
                ),
                np.random.default_rng(4),
                batch_size=1,
            )
        self.assertEqual(batch.initial_support_x.shape, (1, 32, 2))
        self.assertEqual(batch.stream_x.shape, (1, 32, 2))
        self.assertEqual(batch.query_x.shape, (1, 4, 2))
        self.assertEqual(batch.candidate_query_y.shape, (1, 2, 4))
        self.assertEqual(int(batch.pair_attempts[0]), 1)
        self.assertEqual(float(batch.support_disagreement[0]), 0.0)
        self.assertEqual(float(batch.stream_disagreement[0]), 1.0)
        self.assertEqual(float(batch.query_disagreement[0]), 1.0)

    def test_generator_returns_shared_features_and_valid_ambiguity_metadata(self):
        batch = generate_prior_bimodal_episodes(
            episode_config(), np.random.default_rng(17), batch_size=2
        )
        self.assertEqual(batch.initial_support_x.shape, (2, 8, 2))
        self.assertEqual(batch.stream_x.shape, (2, 8, 2))
        self.assertEqual(batch.query_x.shape, (2, 4, 2))
        self.assertEqual(batch.candidate_support_y.shape, (2, 2, 8))
        selected_support = batch.candidate_support_y[
            torch.arange(2), batch.candidate_task
        ]
        self.assertTrue(torch.equal(selected_support.float(), batch.initial_support_y))
        self.assertTrue((batch.support_disagreement <= 0.20).all())
        self.assertTrue((batch.stream_disagreement >= 0.25).all())
        self.assertTrue((batch.query_disagreement >= 0.25).all())
        self.assertTrue(((batch.candidate_task == 0) | (batch.candidate_task == 1)).all())

    def test_generator_has_bounded_failure_path(self):
        config = episode_config(
            support_disagreement_max=-1.0,
            stream_disagreement_min=1.0,
            query_disagreement_min=1.0,
            max_pair_attempts=1,
        )
        with self.assertRaises(RuntimeError):
            generate_prior_bimodal_episodes(config, np.random.default_rng(3), batch_size=1)


class PriorBimodalParticleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = NanoTabPFNIntegratedLatentFilter(tiny_backbone(), num_hypotheses=2)
        self.batch = generate_prior_bimodal_episodes(
            episode_config(), np.random.default_rng(17), batch_size=2
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

    def test_particle_index_permutation_preserves_ensemble(self):
        original = self.predict()
        swapped = copy.deepcopy(self.model)
        with torch.no_grad():
            swapped.initial_latents.copy_(self.model.initial_latents.flip(0))
        permuted = self.predict(swapped)
        torch.testing.assert_close(
            original.joint_probabilities(), permuted.joint_probabilities(), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            original.marginal_probabilities(), permuted.marginal_probabilities(), atol=1e-6, rtol=1e-6
        )

    def test_stream_permutation_preserves_final_ensemble(self):
        original = self.predict()
        order = torch.as_tensor([5, 1, 7, 0, 3, 6, 2, 4])
        reordered = copy.copy(self.batch)
        reordered.stream_x = self.batch.stream_x[:, order]
        reordered.stream_y = self.batch.stream_y[:, order]
        permuted = self.predict(batch=reordered)
        torch.testing.assert_close(
            original.joint_probabilities()[:, -1],
            permuted.joint_probabilities()[:, -1],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_particles_can_be_measured_for_collapse(self):
        prediction = self.predict()
        slot_joint = prediction.slot_joint_log_probabilities().exp()
        difference = (slot_joint[:, 0] - slot_joint[:, 1]).abs().sum(-1)
        self.assertEqual(difference.shape, (2,))
        self.assertTrue(torch.isfinite(difference).all())


if __name__ == "__main__":
    unittest.main()
