import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.train_adaptive_particle_filter import adaptive_loss
from tfmplayground.experiments.train_sequential_latent_filter import (
    SequentialFilterConfig,
    generate_controlled_episodes,
)
from tfmplayground.models.adaptive_particle_filter import (
    expand_two_to_k_particles,
    load_adaptive_checkpoint,
    save_adaptive_checkpoint,
)
from tfmplayground.models.integrated_latent_filter import NanoTabPFNIntegratedLatentFilter
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(embedding_size=8, num_attention_heads=2, mlp_hidden_size=16, num_layers=2, num_outputs=3)


class AdaptiveParticleFilterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        source = NanoTabPFNIntegratedLatentFilter(backbone())
        self.model = expand_two_to_k_particles(source, backbone(), particle_count=4)
        config = SequentialFilterConfig(
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
        )
        self.batch = generate_controlled_episodes(config, np.random.default_rng(17), condition="neutral", batch_size=2)

    def prediction(self, **kwargs):
        return self.model(
            self.batch.initial_support_x,
            self.batch.initial_support_y,
            self.batch.stream_x,
            self.batch.stream_y,
            self.batch.query_x,
            **kwargs,
        )

    def test_k4_expansion_preserves_duplicate_particle_pairs(self):
        prediction = self.prediction()
        self.assertEqual(prediction.log_weights.shape, (2, 9, 4))
        torch.testing.assert_close(prediction.query_logits[:, :, 0], prediction.query_logits[:, :, 2])
        torch.testing.assert_close(prediction.query_logits[:, :, 1], prediction.query_logits[:, :, 3])

    def test_zero_ambiguity_is_exact_vanilla(self):
        prediction = self.prediction(ambiguity_override=0.0)
        expected = prediction.vanilla_query_logits.softmax(-1)
        actual = prediction.marginal_probabilities()
        torch.testing.assert_close(actual, expected[:, None].expand_as(actual), rtol=0, atol=0)
        base_observed = (
            prediction.vanilla_stream_logits.log_softmax(-1)
            .gather(-1, self.batch.stream_y.long().unsqueeze(-1))
            .squeeze(-1)
        )
        torch.testing.assert_close(
            prediction.prequential_log_likelihood_for(self.batch.stream_y),
            base_observed,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_unmatched_particles_can_be_suppressed(self):
        prediction = self.prediction(ambiguity_override=1.0)
        matched = torch.tensor([[0, 1], [1, 2]])
        probabilities = prediction.matched_marginal_probabilities(matched)
        torch.testing.assert_close(
            probabilities.sum(-1),
            torch.ones_like(probabilities[..., 0]),
        )

    def test_saturated_gate_keeps_finite_gradients(self):
        """A saturated gate must not poison the backward pass.

        sigmoid saturates to exactly 1.0 in float32 for logits above ~17, and the gradient
        of an unguarded log1p(-alpha) is -1/(1-alpha) = -inf there, which becomes NaN in the
        gate's gradients and cannot be recovered by clipping. Note this must drive alpha
        *through the gate* -- an `ambiguity_override` constant does not require grad, so it
        never exercises this path and would make the test vacuous.
        """
        with torch.no_grad():
            self.model.ambiguity_gate[-1].bias.fill_(40.0)
        self.model.particle_model.requires_grad_(False)
        self.model.ambiguity_gate.requires_grad_(True)

        prediction = self.prediction()
        self.assertEqual(
            float(prediction.ambiguity_probability.max()),
            1.0,
            "test setup failed to saturate the gate to exactly 1.0",
        )
        likelihood = prediction.prequential_log_likelihood_for(self.batch.stream_y)
        self.assertTrue(torch.isfinite(likelihood).all(), "saturated gate gave non-finite likelihood")

        (-likelihood.mean()).backward()
        gradients = [
            parameter.grad for parameter in self.model.ambiguity_gate.parameters() if parameter.grad is not None
        ]
        self.assertTrue(gradients, "expected gate gradients to be populated")
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients),
            "saturated gate produced non-finite gate gradients",
        )

    def test_gate_only_training_has_gate_gradients(self):
        self.model.particle_model.requires_grad_(False)
        prediction = self.prediction()
        loss, _ = adaptive_loss(prediction, self.batch, controlled=True)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.ambiguity_gate.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.particle_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.vanilla_backbone.parameters()))

    def test_checkpoint_roundtrip(self):
        self.model.particle_model.set_evidence_disagreement_js_threshold(0.1)
        expected = self.prediction()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.pth"
            save_adaptive_checkpoint(
                path,
                self.model,
                training_config={"seed": 17},
                source_checkpoint_sha256="source",
            )
            loaded, metadata = load_adaptive_checkpoint(path)
            actual = loaded(
                self.batch.initial_support_x,
                self.batch.initial_support_y,
                self.batch.stream_x,
                self.batch.stream_y,
                self.batch.query_x,
            )
        torch.testing.assert_close(expected.marginal_probabilities(), actual.marginal_probabilities(), rtol=0, atol=0)
        self.assertEqual(metadata["architecture"]["particle_count"], 4)
        self.assertEqual(metadata["architecture"]["evidence_disagreement_js_threshold"], 0.1)
        self.assertEqual(loaded.particle_model.evidence_disagreement_js_threshold, 0.1)


if __name__ == "__main__":
    unittest.main()
