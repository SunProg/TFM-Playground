import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.continuous_interface import (
    ContextResamplingClassifier,
    ContinuousUncertaintyClassifier,
    deterministic_context_indices,
)
from tfmplayground.experiments.continuous_episodes import (
    ALL_FAMILIES,
    ANALYTIC_FAMILIES,
    CANDIDATE_COUNTS,
    HELDOUT_FAMILIES,
    HELDOUT_REGIME,
    MULTIREGIME_HELDOUT_PAIRS,
    TRAIN_FAMILIES,
    TRAIN_REGIME,
    available_support_sizes,
    curriculum_condition,
    exact_candidate_posterior,
    random_label_episode,
    sample_episode,
    sample_multiregime_episode,
    sample_scm_multiregime_episode,
    sample_paired_episode,
)
from tfmplayground.experiments.continuous_sweep import (
    configuration_flags,
    configuration_label,
    screening_configurations,
    summarize_sweep,
)
from tfmplayground.experiments.evaluate_continuous_synthetic import (
    METRIC_DIRECTION,
    RISK_LAMBDAS,
    SyntheticEvaluationConfig,
    continuous_posterior_benefit,
    evaluate_arm,
    representation_benefit,
    select_risk_lambda,
)
from tfmplayground.experiments.evaluate_continuous_tabarena import (
    TABARENA_BINARY_TASK_NAMES,
    error_detection_scores,
    summarize,
    tabarena_gate,
)
from tfmplayground.experiments.train_continuous_bayesian import (
    ContinuousTrainingConfig,
    build_model,
    continuous_losses,
    energy_distance,
    evidence_monotonicity_loss,
    teacher_targets,
    train,
    validate_config,
)
from tfmplayground.models.continuous_posterior import (
    AdaptedTransformerEncoderLayer,
    BottleneckAdapter,
    ContextResamplingUncertainty,
    ContinuousPosteriorPrediction,
    NanoTabPFNBetaConcentrationModel,
    NanoTabPFNContinuousPosteriorModel,
    all_binary_outcomes,
    centre_and_scale,
    install_adapters,
    load_continuous_checkpoint,
    project_candidate_posterior,
    save_continuous_checkpoint,
    sobol_standard_normal,
)
from tfmplayground.models.hypothesis import NanoTabPFNBayesianModel, save_bayesian_checkpoint
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone(seed: int = 0) -> NanoTabPFNModel:
    torch.manual_seed(seed)
    return NanoTabPFNModel(16, 2, 32, 2, 3)


def tiny_continuous(seed: int = 0, **kwargs) -> NanoTabPFNContinuousPosteriorModel:
    defaults = {"latent_dim": 8, "num_samples": 8}
    defaults.update(kwargs)
    return NanoTabPFNContinuousPosteriorModel(tiny_backbone(seed), **defaults).eval()


def tiny_episode(batch: int = 2, support: int = 12, query: int = 4, features: int = 3):
    generator = torch.Generator().manual_seed(7)
    support_x = torch.randn(batch, support, features, generator=generator)
    support_y = torch.randint(0, 2, (batch, support), generator=generator).float()
    query_x = torch.randn(batch, query, features, generator=generator)
    return support_x, support_y, query_x


class AdapterTests(unittest.TestCase):
    def test_adapters_start_as_the_identity(self):
        backbone = tiny_backbone(1)
        source = (torch.randn(2, 9, 3), torch.randint(0, 2, (2, 5)).float())
        reference = backbone.encode_table(source, 5)
        adapted = install_adapters(tiny_backbone(1), 4)
        torch.testing.assert_close(adapted.encode_table(source, 5), reference, atol=0, rtol=0)

    def test_adapter_count_and_zero_initialization(self):
        adapted = install_adapters(tiny_backbone(2), 4)
        layers = [layer for layer in adapted.transformer_blocks if isinstance(layer, AdaptedTransformerEncoderLayer)]
        self.assertEqual(len(layers), adapted.num_layers)
        for layer in layers:
            for adapter in (layer.feature_adapter, layer.datapoint_adapter, layer.mlp_adapter):
                self.assertTrue(torch.equal(adapter.up.weight, torch.zeros_like(adapter.up.weight)))
                self.assertTrue(torch.equal(adapter.up.bias, torch.zeros_like(adapter.up.bias)))

    def test_six_layer_backbone_has_eighteen_adapters(self):
        torch.manual_seed(3)
        adapted = install_adapters(NanoTabPFNModel(16, 2, 32, 6, 3), 4)
        adapters = [
            module for module in adapted.modules() if isinstance(module, BottleneckAdapter)
        ]
        self.assertEqual(len(adapters), 18)

    def test_ordinary_model_numerics_are_unchanged(self):
        backbone = tiny_backbone(4)
        support_x, support_y, query_x = tiny_episode()
        first = backbone(support_x, support_y, query_x)
        second = backbone(support_x, support_y, query_x)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        # The stage hooks are the identity on the pretrained layer.
        layer = backbone.transformer_blocks[0]
        sample = torch.randn(2, 4, 3, 16)
        torch.testing.assert_close(layer.adapt_after_feature_attention(sample), sample, atol=0, rtol=0)
        torch.testing.assert_close(layer.adapt_after_datapoint_attention(sample), sample, atol=0, rtol=0)
        torch.testing.assert_close(layer.adapt_after_mlp(sample), sample, atol=0, rtol=0)


class SobolNoiseTests(unittest.TestCase):
    def test_antithetic_pairs_and_determinism(self):
        noise = sobol_standard_normal(8, 4, seed=11)
        self.assertEqual(noise.shape, (8, 4))
        torch.testing.assert_close(noise[:4], -noise[4:])
        torch.testing.assert_close(noise.mean(dim=0), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(noise, sobol_standard_normal(8, 4, seed=11), atol=0, rtol=0)
        self.assertFalse(torch.equal(noise, sobol_standard_normal(8, 4, seed=12)))

    def test_odd_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            sobol_standard_normal(7, 4, seed=0)


class ContinuousPosteriorModelTests(unittest.TestCase):
    def test_mean_is_exactly_vanilla(self):
        model = tiny_continuous(5)
        support_x, support_y, query_x = tiny_episode()
        prediction = model(support_x, support_y, query_x)
        vanilla = model.mean_backbone(support_x, support_y, query_x)[..., :2].softmax(-1)
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla, atol=0, rtol=0)
        self.assertLessEqual(float(prediction.mean_preservation_error().max()), 1e-6)
        self.assertTrue(((prediction.sample_positive > 0) & (prediction.sample_positive < 1)).all())

    def test_uncertainty_outputs_are_complete_and_finite(self):
        model = tiny_continuous(6)
        prediction = model(*tiny_episode())
        summary = prediction.summary()
        for key in (
            "vanilla_probabilities",
            "sample_probabilities",
            "predictive_entropy",
            "expected_conditional_entropy",
            "mutual_information",
            "epistemic_variance",
            "epistemic_covariance",
            "max_mean_preservation_error",
        ):
            self.assertIn(key, summary)
            self.assertTrue(torch.isfinite(summary[key]).all())
        self.assertEqual(summary["epistemic_covariance"].shape, (2, 4, 4))
        torch.testing.assert_close(
            summary["epistemic_covariance"].diagonal(dim1=-2, dim2=-1),
            prediction.epistemic_variance(),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            prediction.joint_probabilities().sum(-1), torch.ones(2), atol=1e-5, rtol=1e-5
        )

    def test_support_permutation_invariance_and_query_equivariance(self):
        model = tiny_continuous(7)
        support_x, support_y, query_x = tiny_episode()
        prediction = model(support_x, support_y, query_x)
        support_order = torch.randperm(support_x.shape[1], generator=torch.Generator().manual_seed(1))
        permuted = model(support_x[:, support_order], support_y[:, support_order], query_x)
        torch.testing.assert_close(
            permuted.mutual_information(), prediction.mutual_information(), atol=1e-5, rtol=1e-4
        )
        query_order = torch.tensor([2, 0, 3, 1])
        reordered = model(support_x, support_y, query_x[:, query_order])
        torch.testing.assert_close(
            reordered.sample_positive, prediction.sample_positive[:, :, query_order], atol=1e-5, rtol=1e-4
        )

    def test_sample_seed_controls_determinism(self):
        model = tiny_continuous(8)
        episode = tiny_episode()
        first = model(*episode, sample_seed=3).sample_positive
        second = model(*episode, sample_seed=3).sample_positive
        third = model(*episode, sample_seed=4).sample_positive
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, third))

    def test_more_samples_do_not_change_the_prediction(self):
        model = tiny_continuous(9)
        episode = tiny_episode()
        few = model(*episode, num_samples=8)
        many = model(*episode, num_samples=32)
        torch.testing.assert_close(many.marginal_probabilities(), few.marginal_probabilities(), atol=0, rtol=0)
        self.assertEqual(many.num_samples, 32)

    def test_uncertainty_modes_select_the_expected_parameters(self):
        frozen = NanoTabPFNContinuousPosteriorModel(tiny_backbone(10), uncertainty_mode="frozen", latent_dim=8)
        adapters = NanoTabPFNContinuousPosteriorModel(tiny_backbone(10), uncertainty_mode="adapters", latent_dim=8)
        full = NanoTabPFNContinuousPosteriorModel(tiny_backbone(10), uncertainty_mode="full", latent_dim=8)
        frozen_count = sum(p.numel() for p in frozen.trainable_parameters())
        adapter_count = sum(p.numel() for p in adapters.trainable_parameters())
        full_count = sum(p.numel() for p in full.trainable_parameters())
        self.assertLess(frozen_count, adapter_count)
        self.assertLess(adapter_count, full_count)
        for model in (frozen, adapters, full):
            self.assertFalse(any(p.requires_grad for p in model.mean_backbone.parameters()))

    def test_gradients_are_finite_and_leave_the_mean_path_untouched(self):
        model = NanoTabPFNContinuousPosteriorModel(tiny_backbone(11), latent_dim=8, num_samples=8)
        reference = {name: value.clone() for name, value in model.mean_backbone.state_dict().items()}
        rng = np.random.default_rng(0)
        episode = sample_episode(
            rng, condition="ambiguous", batch_size=1, num_candidates=4, support_size=16, query_count=4,
            noise=0.05, family="linear",
        )
        config = ContinuousTrainingConfig(num_samples=8, batch_size=1)
        prediction = model(episode.support_x, episode.support_y, episode.query_x)
        loss, metrics, _targets = continuous_losses(prediction, episode, config)
        loss.backward()
        gradients = [p.grad for p in model.trainable_parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        optimizer.step()
        for name, value in model.mean_backbone.state_dict().items():
            torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
        self.assertLessEqual(metrics["mean_preservation_error"], 1e-6)

    def test_query_labels_never_reach_the_model(self):
        model = tiny_continuous(12)
        support_x, support_y, query_x = tiny_episode()
        first = model(support_x, support_y, query_x).sample_positive
        # There is no query-label argument at all; the only labelled input is the
        # support table, so altering query labels cannot change the prediction.
        second = model((torch.cat((support_x, query_x), dim=1), support_y), train_test_split_index=support_x.shape[1])
        torch.testing.assert_close(first, second.sample_positive, atol=0, rtol=0)


class ContextModeTests(unittest.TestCase):
    """Per-query cross-attention context, selected by an encoder probe."""

    def _model(self, mode: str, seed: int = 50):
        return NanoTabPFNContinuousPosteriorModel(
            tiny_backbone(seed), latent_dim=8, num_samples=8, context_mode=mode
        ).eval()

    def test_cross_attention_preserves_the_model_symmetries(self):
        model = self._model("cross_attention")
        support_x, support_y, query_x = tiny_episode()
        prediction = model(support_x, support_y, query_x)
        vanilla = model.mean_backbone(support_x, support_y, query_x)[..., :2].softmax(-1)
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla, atol=0, rtol=0)
        self.assertLessEqual(float(prediction.mean_preservation_error().max()), 1e-6)
        order = torch.randperm(support_x.shape[1], generator=torch.Generator().manual_seed(2))
        permuted = model(support_x[:, order], support_y[:, order], query_x)
        torch.testing.assert_close(
            permuted.mutual_information(), prediction.mutual_information(), atol=1e-5, rtol=1e-4
        )
        query_order = torch.tensor([2, 0, 3, 1])
        reordered = model(support_x, support_y, query_x[:, query_order])
        torch.testing.assert_close(
            reordered.sample_positive, prediction.sample_positive[:, :, query_order], atol=1e-5, rtol=1e-4
        )

    def test_latent_draws_stay_query_independent(self):
        """One latent sample must remain one coherent function over all queries.

        Only the decoder and the gate may see a per-query context; the latent
        generator keeps consuming the global pool.
        """
        model = self._model("cross_attention", seed=51)
        support_x, support_y, query_x = tiny_episode()
        support, query = model._uncertainty_representations(
            support_x, support_y, query_x, num_mem_chunks=1
        )
        global_context, query_context = model._contexts(support, query)
        self.assertEqual(global_context.shape, (support_x.shape[0], model.context_size))
        self.assertEqual(query_context.shape[:2], query_x.shape[:2])
        order = torch.tensor([2, 0, 3, 1])
        _reordered_global, _ = model._contexts(support, query[:, order])
        torch.testing.assert_close(_reordered_global, global_context, atol=0, rtol=0)

    def test_deepsets_remains_the_default_and_broadcasts(self):
        model = self._model("deepsets", seed=52)
        self.assertEqual(model.context_mode, "deepsets")
        self.assertIsNone(model.query_context_encoder)
        support_x, support_y, query_x = tiny_episode()
        support, query = model._uncertainty_representations(
            support_x, support_y, query_x, num_mem_chunks=1
        )
        global_context, query_context = model._contexts(support, query)
        torch.testing.assert_close(
            query_context, global_context[:, None].expand(-1, query_x.shape[1], -1), atol=0, rtol=0
        )

    def test_checkpoint_without_context_mode_loads_as_deepsets(self):
        model = self._model("deepsets", seed=53)
        episode = tiny_episode()
        expected = model(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_context.pth"
            save_continuous_checkpoint(
                path,
                model,
                training_config={},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="5" * 64,
                stage="continuous_adapters",
            )
            checkpoint = torch.load(path, weights_only=False)
            del checkpoint["architecture"]["context_mode"]
            torch.save(checkpoint, path)
            restored, _checkpoint = load_continuous_checkpoint(path)
        self.assertEqual(restored.context_mode, "deepsets")
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)

    def test_cross_attention_round_trip(self):
        model = self._model("cross_attention", seed=54)
        episode = tiny_episode()
        expected = model(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross.pth"
            save_continuous_checkpoint(
                path,
                model,
                training_config={},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="6" * 64,
                stage="continuous_adapters",
            )
            restored, checkpoint = load_continuous_checkpoint(path)
        self.assertEqual(checkpoint["architecture"]["context_mode"], "cross_attention")
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)

    def test_unknown_context_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            NanoTabPFNContinuousPosteriorModel(tiny_backbone(55), latent_dim=8, context_mode="global")


class BetaAblationTests(unittest.TestCase):
    def test_beta_mean_is_exactly_vanilla_and_concentration_is_positive(self):
        model = NanoTabPFNBetaConcentrationModel(tiny_backbone(13), num_samples=8).eval()
        support_x, support_y, query_x = tiny_episode()
        prediction = model(support_x, support_y, query_x)
        vanilla = model.mean_backbone(support_x, support_y, query_x)[..., :2].softmax(-1)
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla, atol=0, rtol=0)
        self.assertTrue((prediction.concentration > 0).all())
        analytic = prediction.analytic_epistemic_variance()
        self.assertTrue(torch.isfinite(analytic).all())
        self.assertTrue((analytic <= 0.25 + 1e-6).all())

    def test_beta_samples_are_reproducible_for_a_seed(self):
        model = NanoTabPFNBetaConcentrationModel(tiny_backbone(14), num_samples=8).eval()
        episode = tiny_episode()
        first = model(*episode, sample_seed=5).sample_positive
        second = model(*episode, sample_seed=5).sample_positive
        torch.testing.assert_close(first, second, atol=0, rtol=0)


class ContextResamplingTests(unittest.TestCase):
    def test_full_context_mean_is_preserved_and_samples_are_recentred(self):
        backbone = tiny_backbone(15)
        baseline = ContextResamplingUncertainty(backbone, num_subsets=4)
        support_x, support_y, query_x = tiny_episode(batch=1, support=20)
        prediction = baseline(support_x, support_y, query_x)
        vanilla = backbone(support_x, support_y, query_x)[..., :2].softmax(-1)
        torch.testing.assert_close(prediction.marginal_probabilities(), vanilla, atol=0, rtol=0)
        self.assertLessEqual(float(prediction.mean_preservation_error().max()), 1e-6)
        self.assertEqual(prediction.sample_positive.shape[1], 4)

    def test_subsets_are_deterministic_and_stratified(self):
        backbone = tiny_backbone(16)
        baseline = ContextResamplingUncertainty(backbone, num_subsets=6)
        labels = torch.tensor([0.0] * 10 + [1.0] * 10)
        first = baseline._subset_indices(labels, 0)
        self.assertTrue(torch.equal(first, baseline._subset_indices(labels, 0)))
        self.assertGreater(int((labels[first] == 0).sum()), 0)
        self.assertGreater(int((labels[first] == 1).sum()), 0)


class DispersionShapeTests(unittest.TestCase):
    """The safe bound is set by the largest sample, so shape drives dispersion."""

    @staticmethod
    def _shape_ratio(raw, deviation_clip, mu=0.5):
        base = torch.full((1, 3), mu)
        gate = torch.ones(1, 3)
        samples, bound = centre_and_scale(base, raw, gate, deviation_clip=deviation_clip)
        prediction = ContinuousPosteriorPrediction(
            torch.stack((1.0 - base, base), dim=-1), samples, gate, bound
        )
        return float(prediction.deviation_shape_ratio().mean()), prediction

    def _heavy_tailed(self):
        generator = torch.Generator().manual_seed(4)
        raw = torch.randn(1, 32, 3, generator=generator) * 0.1
        raw[0, 0] += 3.0
        return raw

    def test_outlier_sample_throttles_the_bulk_without_the_clip(self):
        legacy, _ = self._shape_ratio(self._heavy_tailed(), None)
        self.assertLess(legacy, 0.30)

    def test_clip_restores_the_bulk_of_the_headroom(self):
        clipped, _ = self._shape_ratio(self._heavy_tailed(), 1.5)
        self.assertGreater(clipped, 0.30)
        legacy, _ = self._shape_ratio(self._heavy_tailed(), None)
        self.assertGreater(clipped, legacy)

    def test_clip_preserves_genuine_multimodality(self):
        generator = torch.Generator().manual_seed(5)
        bimodal = torch.cat(
            (torch.full((1, 16, 3), -1.0), torch.full((1, 16, 3), 1.0)), dim=1
        ) + 0.05 * torch.randn(1, 32, 3, generator=generator)
        legacy, _ = self._shape_ratio(bimodal, None)
        clipped, _ = self._shape_ratio(bimodal, 1.5)
        # A well-separated bimodal shape is already flat, so the clip must
        # leave it essentially untouched rather than compressing the modes.
        self.assertAlmostEqual(clipped, legacy, delta=0.1)
        self.assertGreater(clipped, 0.8)

    def test_mean_preservation_holds_under_the_clip_at_extreme_means(self):
        raw = self._heavy_tailed()
        for mean in (0.001, 0.5, 0.999):
            with self.subTest(mean=mean):
                _ratio, prediction = self._shape_ratio(raw, 1.5, mu=mean)
                self.assertLessEqual(float(prediction.mean_preservation_error().max()), 1e-6)
                self.assertTrue((prediction.sample_positive > 0).all())
                self.assertTrue((prediction.sample_positive < 1).all())

    def test_parameterization_can_express_the_teacher_scale(self):
        generator = torch.Generator().manual_seed(6)
        raw = torch.randn(1, 32, 3, generator=generator)
        _ratio, prediction = self._shape_ratio(raw, 1.5)
        # An open gate at mu=0.5 must be able to reach the mutual information
        # the projected teacher asks for on ambiguous episodes (about 0.18).
        self.assertGreater(float(prediction.mutual_information().mean()), 0.1)

    def test_clip_is_rejected_when_not_positive(self):
        with self.assertRaises(ValueError):
            centre_and_scale(torch.full((1, 2), 0.5), torch.randn(1, 4, 2), torch.ones(1, 2), deviation_clip=0.0)
        with self.assertRaises(ValueError):
            NanoTabPFNContinuousPosteriorModel(tiny_backbone(40), latent_dim=8, deviation_clip=-1.0)


class MeanPreservingProjectionTests(unittest.TestCase):
    def test_projection_preserves_weights_and_mean(self):
        base = torch.tensor([[0.02, 0.5, 0.97]])
        candidates = torch.tensor([[[0.9, 0.1, 0.2], [0.1, 0.9, 0.8], [0.5, 0.5, 0.5]]])
        weights = torch.tensor([[0.5, 0.3, 0.2]])
        projected, scale = project_candidate_posterior(base, candidates, weights)
        torch.testing.assert_close((weights[:, :, None] * projected).sum(1), base, atol=1e-6, rtol=0)
        self.assertTrue(((projected > 0) & (projected < 1)).all())
        self.assertTrue((scale <= 1.0 + 1e-6).all())
        self.assertTrue((scale > 0).all())

    def test_projection_keeps_the_direction_of_disagreement(self):
        base = torch.tensor([[0.5]])
        candidates = torch.tensor([[[0.9], [0.1]]])
        weights = torch.tensor([[0.5, 0.5]])
        projected, scale = project_candidate_posterior(base, candidates, weights)
        self.assertGreater(float(projected[0, 0, 0]), float(projected[0, 1, 0]))
        self.assertAlmostEqual(float(scale), 1.0, places=5)

    def test_teacher_targets_match_the_exact_posterior_when_feasible(self):
        rng = np.random.default_rng(3)
        episode = sample_episode(
            rng, condition="ambiguous", batch_size=1, num_candidates=4, support_size=16, query_count=4,
            noise=0.10, family="linear",
        )
        base = torch.full((1, 4), 0.5)
        targets = teacher_targets(episode, base)
        torch.testing.assert_close(
            (targets.weights[:, :, None] * targets.projected_positive).sum(1), base, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(targets.weights, episode.posterior, atol=1e-6, rtol=1e-5)
        self.assertTrue((targets.mutual_information >= 0).all())


class LossTests(unittest.TestCase):
    def test_energy_distance_is_zero_for_matching_distributions(self):
        samples = torch.tensor([[[0.2, 0.8], [0.6, 0.4]]])
        weights = torch.tensor([[0.5, 0.5]])
        self.assertAlmostEqual(float(energy_distance(samples, samples, weights)), 0.0, places=6)

    def test_energy_distance_grows_with_mismatch(self):
        samples = torch.tensor([[[0.2, 0.8], [0.2, 0.8]]])
        near = torch.tensor([[[0.25, 0.75], [0.15, 0.85]]])
        far = torch.tensor([[[0.9, 0.1], [0.8, 0.2]]])
        weights = torch.tensor([[0.5, 0.5]])
        self.assertLess(
            float(energy_distance(samples, near, weights)), float(energy_distance(samples, far, weights))
        )

    def test_evidence_monotonicity_only_penalizes_increases(self):
        class Stub:
            def __init__(self, value):
                self.value = value

            def mutual_information(self):
                return self.value

        rising = evidence_monotonicity_loss(Stub(torch.tensor([[0.1]])), Stub(torch.tensor([[0.4]])))
        falling = evidence_monotonicity_loss(Stub(torch.tensor([[0.4]])), Stub(torch.tensor([[0.1]])))
        self.assertAlmostEqual(float(rising), 0.3, places=6)
        self.assertAlmostEqual(float(falling), 0.0, places=6)

    def test_loss_components_are_logged_separately(self):
        model = tiny_continuous(17)
        rng = np.random.default_rng(1)
        episode = sample_episode(
            rng, condition="identifiable", batch_size=1, num_candidates=4, support_size=16, query_count=4,
            noise=0.05, family="tree",
        )
        config = ContinuousTrainingConfig(num_samples=8, batch_size=1)
        prediction = model(episode.support_x, episode.support_y, episode.query_x)
        _loss, metrics, _targets = continuous_losses(prediction, episode, config)
        for key in (
            "energy_distance",
            "mutual_information_loss",
            "variance_loss",
            "covariance_loss",
            "joint_query_loss",
            "selection_loss",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]))

    def test_moment_weight_is_neutral_at_one_and_scales_the_moment_terms(self):
        model = tiny_continuous(43)
        rng = np.random.default_rng(2)
        episode = sample_episode(
            rng, condition="ambiguous", batch_size=1, num_candidates=4, support_size=16, query_count=4,
            noise=0.05, family="linear",
        )
        prediction = model(episode.support_x, episode.support_y, episode.query_x)
        baseline, metrics, _targets = continuous_losses(
            prediction, episode, ContinuousTrainingConfig(num_samples=8, batch_size=1)
        )
        neutral, _metrics, _targets = continuous_losses(
            prediction, episode, ContinuousTrainingConfig(num_samples=8, batch_size=1, moment_weight=1.0)
        )
        torch.testing.assert_close(neutral, baseline, atol=0, rtol=0)
        scaled, _metrics, _targets = continuous_losses(
            prediction, episode, ContinuousTrainingConfig(num_samples=8, batch_size=1, moment_weight=4.0)
        )
        expected = baseline + 3.0 * (0.5 * metrics["mutual_information_loss"] + 0.5 * metrics["variance_loss"])
        torch.testing.assert_close(scaled, expected, atol=1e-6, rtol=1e-5)

    def test_dispersion_diagnostics_are_logged(self):
        model = tiny_continuous(44)
        rng = np.random.default_rng(3)
        episode = sample_episode(
            rng, condition="ambiguous", batch_size=1, num_candidates=4, support_size=16, query_count=4,
            noise=0.05, family="linear",
        )
        prediction = model(episode.support_x, episode.support_y, episode.query_x)
        _loss, metrics, _targets = continuous_losses(
            prediction, episode, ContinuousTrainingConfig(num_samples=8, batch_size=1)
        )
        for key in ("dispersion_gate", "dispersion_bound", "deviation_shape_ratio"):
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]))
        self.assertTrue(0.0 <= metrics["dispersion_gate"] <= 1.0)
        self.assertTrue(0.0 <= metrics["deviation_shape_ratio"] <= 1.0)

    def test_configuration_validation(self):
        validate_config(ContinuousTrainingConfig())
        with self.assertRaises(ValueError):
            validate_config(ContinuousTrainingConfig(num_samples=7))
        with self.assertRaises(ValueError):
            validate_config(ContinuousTrainingConfig(model_type="beta", uncertainty_mode="full"))
        with self.assertRaises(ValueError):
            validate_config(ContinuousTrainingConfig(ambiguous_probability=0.9))


class EpisodeTests(unittest.TestCase):
    def test_ambiguous_episodes_have_a_uniform_posterior(self):
        rng = np.random.default_rng(5)
        for candidates in CANDIDATE_COUNTS:
            episode = sample_episode(
                rng, condition="ambiguous", batch_size=1, num_candidates=candidates, support_size=32,
                query_count=4, noise=0.05, family="linear",
            )
            self.assertEqual(episode.num_candidates, candidates)
            torch.testing.assert_close(
                episode.posterior, torch.full((1, candidates), 1.0 / candidates), atol=1e-5, rtol=1e-4
            )
            self.assertGreater(float(episode.candidate_query_positive.var(dim=1).mean()), 0.01)

    def test_identifiable_episodes_concentrate_the_posterior(self):
        rng = np.random.default_rng(6)
        episode = sample_episode(
            rng, condition="identifiable", batch_size=2, num_candidates=8, support_size=64, query_count=4,
            noise=0.05, family="tree",
        )
        self.assertGreater(float(episode.posterior.max(dim=-1).values.min()), 0.5)
        self.assertGreater(episode.metadata["identifying_fraction"], 0.0)

    def test_conditions_differ_only_in_identifying_evidence(self):
        """The ambiguous/identifiable contrast must not be a support-difficulty cue.

        Selecting the support from a different region per condition let probe
        heads latch onto a family-specific surface cue and then invert on
        held-out families, so shape and region are now held fixed.
        """
        ambiguous = sample_episode(
            np.random.default_rng(21), condition="ambiguous", batch_size=1, num_candidates=8,
            support_size=64, query_count=6, noise=0.05, family="linear",
        )
        identifiable = sample_episode(
            np.random.default_rng(21), condition="identifiable", batch_size=1, num_candidates=8,
            support_size=64, query_count=6, noise=0.05, family="linear",
        )
        self.assertEqual(ambiguous.support_x.shape, identifiable.support_x.shape)
        self.assertEqual(ambiguous.query_x.shape, identifiable.query_x.shape)
        self.assertAlmostEqual(ambiguous.metadata["identifying_fraction"], 0.0)
        torch.testing.assert_close(
            ambiguous.posterior, torch.full((1, 8), 1.0 / 8), atol=1e-5, rtol=1e-4
        )
        self.assertGreater(float(identifiable.posterior.max()), 0.5)

    def test_noisy_episodes_have_no_candidate_disagreement(self):
        rng = np.random.default_rng(7)
        episode = sample_episode(
            rng, condition="noisy", batch_size=1, num_candidates=4, support_size=32, query_count=4,
            noise=0.20, family="smooth",
        )
        self.assertAlmostEqual(float(episode.candidate_query_positive.var(dim=1).max()), 0.0, places=6)

    def test_paired_episodes_add_identifying_evidence(self):
        rng = np.random.default_rng(8)
        short, long = sample_paired_episode(
            rng, batch_size=1, num_candidates=8, support_size=64, query_count=5, noise=0.05, family="linear"
        )
        # The arms differ only in how identifying the support is: same support
        # size, same queries, same candidates.
        self.assertEqual(short.support_x.shape, long.support_x.shape)
        torch.testing.assert_close(short.query_x, long.query_x)
        torch.testing.assert_close(short.candidate_query_positive, long.candidate_query_positive)
        self.assertAlmostEqual(float(short.posterior.max()), 1.0 / 8, places=4)
        self.assertGreater(float(long.posterior.max()), 0.5)

    def test_families_and_regimes_are_disjoint(self):
        self.assertEqual(set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES), set())
        self.assertEqual(set(TRAIN_FAMILIES) | set(HELDOUT_FAMILIES), set(ALL_FAMILIES))
        self.assertLess(TRAIN_REGIME.max_features, HELDOUT_REGIME.min_features)
        self.assertGreaterEqual(len(HELDOUT_FAMILIES), 1)

    def test_query_counts_stay_inside_the_enumerable_range(self):
        rng = np.random.default_rng(9)
        for _ in range(5):
            episode = sample_episode(
                rng, condition="ambiguous", batch_size=1, num_candidates=2, support_size=32, family="linear"
            )
            self.assertTrue(4 <= episode.query_count <= 8)

    def test_support_size_budget_caps_the_drawn_sizes(self):
        self.assertEqual(available_support_sizes(512), (32, 64, 128, 256, 512))
        self.assertEqual(available_support_sizes(128), (32, 64, 128))
        with self.assertRaises(ValueError):
            available_support_sizes(16)
        rng = np.random.default_rng(31)
        for _ in range(5):
            episode = sample_episode(
                rng, condition="ambiguous", batch_size=1, num_candidates=2, query_count=4,
                family="linear", max_support_size=64,
            )
            self.assertLessEqual(episode.support_x.shape[1], 64)

    def test_exact_posterior_matches_a_hand_computation(self):
        labels = np.asarray([1, 0])
        candidates = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        posterior = exact_candidate_posterior(labels, candidates)
        expected = np.asarray([0.81, 0.01]) / 0.82
        np.testing.assert_allclose(posterior, expected, atol=1e-6)

    def test_random_label_episodes_are_diagnostic_only(self):
        rng = np.random.default_rng(10)
        episode = random_label_episode(rng, batch_size=1, support_size=16, query_count=4)
        self.assertEqual(episode.condition, "random_label_diagnostic")
        self.assertEqual(episode.num_candidates, 1)

    def test_curriculum_weights_sum_to_one(self):
        weights = ContinuousTrainingConfig().curriculum()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["ambiguous"], 0.30)
        self.assertAlmostEqual(weights["identifiable"], 0.20)
        self.assertAlmostEqual(weights["noisy"], 0.15)
        self.assertAlmostEqual(weights["paired"], 0.15)
        self.assertAlmostEqual(weights["multiregime"], 0.20)
        rng = np.random.default_rng(11)
        drawn = {curriculum_condition(rng, weights) for _ in range(200)}
        self.assertEqual(drawn, {"ambiguous", "identifiable", "noisy", "paired", "multiregime"})


class MultiregimeEpisodeTests(unittest.TestCase):
    def test_shapes_and_metadata(self):
        rng = np.random.default_rng(3)
        episode = sample_multiregime_episode(
            rng, regime=TRAIN_REGIME, batch_size=2, support_size=64, query_count=6, contamination=0.3
        )
        self.assertEqual(episode.condition, "multiregime")
        self.assertEqual(tuple(episode.support_x.shape[:2]), (2, 64))
        self.assertEqual(tuple(episode.query_x.shape[:2]), (2, 6))
        self.assertEqual(tuple(episode.candidate_query_positive.shape), (2, 2, 6))
        self.assertEqual(tuple(episode.posterior.shape), (2, 2))
        self.assertIsNotNone(episode.query_regime_source)
        self.assertEqual(tuple(episode.query_regime_source.shape), (2, 6))
        self.assertTrue(set(episode.query_regime_source.flatten().tolist()) <= {0, 1})

    def test_heldout_regime_draws_only_reserved_pairs(self):
        seen = set()
        for seed in range(60):
            rng = np.random.default_rng(seed)
            episode = sample_multiregime_episode(rng, regime=HELDOUT_REGIME, batch_size=1, support_size=32, query_count=4)
            seen.add(frozenset({episode.metadata["base_family"], episode.metadata["other_family"]}))
        self.assertTrue(seen)
        self.assertTrue(seen.issubset(set(MULTIREGIME_HELDOUT_PAIRS)))

    def test_train_regime_never_draws_reserved_pairs(self):
        seen = set()
        for seed in range(200):
            rng = np.random.default_rng(1000 + seed)
            episode = sample_multiregime_episode(rng, regime=TRAIN_REGIME, batch_size=1, support_size=32, query_count=4)
            seen.add(frozenset({episode.metadata["base_family"], episode.metadata["other_family"]}))
        self.assertTrue(seen.isdisjoint(set(MULTIREGIME_HELDOUT_PAIRS)))
        self.assertTrue(seen.issubset({frozenset({a, b}) for a in ANALYTIC_FAMILIES for b in ANALYTIC_FAMILIES if a != b}))

    def test_deterministic_given_seed(self):
        first = sample_multiregime_episode(np.random.default_rng(7), support_size=32, query_count=4, contamination=0.2)
        second = sample_multiregime_episode(np.random.default_rng(7), support_size=32, query_count=4, contamination=0.2)
        torch.testing.assert_close(first.support_x, second.support_x, atol=0, rtol=0)
        torch.testing.assert_close(first.support_y, second.support_y, atol=0, rtol=0)
        torch.testing.assert_close(first.query_regime_source, second.query_regime_source, atol=0, rtol=0)

    def test_no_regime_label_reaches_the_model_inputs(self):
        """query_regime_source is diagnostic-only: never present in support_x/support_y/query_x."""
        episode = sample_multiregime_episode(np.random.default_rng(2), support_size=16, query_count=4)
        model_inputs = (episode.support_x, episode.support_y, episode.query_x)
        for tensor in model_inputs:
            self.assertNotEqual(tensor.shape, episode.query_regime_source.shape)

    def test_posterior_is_a_valid_distribution_over_two_candidates(self):
        episode = sample_multiregime_episode(np.random.default_rng(4), batch_size=3, support_size=48, query_count=5)
        self.assertEqual(episode.posterior.shape[-1], 2)
        torch.testing.assert_close(episode.posterior.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=0)
        self.assertTrue(bool((episode.posterior >= 0).all()))

    def test_contamination_zero_matches_single_regime_query_labels(self):
        """With contamination=0, every query row is base-regime and unlabelled as 'other'."""
        episode = sample_multiregime_episode(
            np.random.default_rng(9), batch_size=1, support_size=32, query_count=6, contamination=0.0
        )
        self.assertTrue(bool((episode.query_regime_source == 0).all()))


class ScmMultiregimeEpisodeTests(unittest.TestCase):
    """sample_scm_multiregime_episode: two draws from the official TabICL SCM prior."""

    def test_train_regime_uses_mlp_scm(self):
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(1), regime=TRAIN_REGIME, batch_size=1, support_size=32, query_count=4
        )
        self.assertEqual(episode.family, "mlp_scm")
        self.assertEqual(episode.condition, "multiregime")
        self.assertEqual(tuple(episode.support_x.shape[:2]), (1, 32))
        self.assertEqual(tuple(episode.query_regime_source.shape), (1, 4))

    def test_heldout_regime_uses_tree_scm(self):
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(2), regime=HELDOUT_REGIME, batch_size=1, support_size=32, query_count=4
        )
        self.assertEqual(episode.family, "tree_scm")

    def test_posterior_is_a_valid_distribution_over_two_candidates(self):
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(3), batch_size=2, support_size=24, query_count=4
        )
        self.assertEqual(episode.posterior.shape[-1], 2)
        torch.testing.assert_close(episode.posterior.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=0)
        self.assertTrue(bool((episode.posterior >= 0).all()))

    def test_contamination_zero_matches_single_regime_query_labels(self):
        episode = sample_scm_multiregime_episode(
            np.random.default_rng(4), batch_size=1, support_size=24, query_count=4, contamination=0.0
        )
        self.assertTrue(bool((episode.query_regime_source == 0).all()))

    def test_no_regime_label_reaches_the_model_inputs(self):
        episode = sample_scm_multiregime_episode(np.random.default_rng(5), batch_size=1, support_size=16, query_count=4)
        for tensor in (episode.support_x, episode.support_y, episode.query_x):
            self.assertNotEqual(tensor.shape, episode.query_regime_source.shape)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_reproduces_predictions_exactly(self):
        model = NanoTabPFNContinuousPosteriorModel(tiny_backbone(18), latent_dim=8, num_samples=8, inference_seed=5)
        model.eval()
        episode = tiny_episode()
        expected = model(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuous.pth"
            save_continuous_checkpoint(
                path,
                model,
                training_config={"seed": 1},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="0" * 64,
                stage="continuous_adapters",
                step=100,
                validation_metrics={"selection_loss": 0.5},
                random_seeds={"training_seed": 1, "inference_seed": 5},
            )
            restored, checkpoint = load_continuous_checkpoint(path)
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)
        self.assertEqual(checkpoint["model_type"], "nanotabpfn_continuous_posterior")
        self.assertEqual(checkpoint["format_version"], 1)
        self.assertEqual(checkpoint["architecture"]["uncertainty_mode"], "adapters")
        self.assertEqual(checkpoint["architecture"]["adapter_bottleneck"], 32)
        self.assertEqual(checkpoint["architecture"]["latent_dim"], 8)
        self.assertEqual(checkpoint["sample_generation"]["scheme"], "scrambled_sobol_antithetic")
        self.assertEqual(checkpoint["source_checkpoint_sha256"], "0" * 64)
        self.assertEqual(checkpoint["step"], 100)
        self.assertIn("validation_metrics", checkpoint)
        self.assertIn("random_seeds", checkpoint)

    def test_checkpoint_records_the_deviation_clip(self):
        model = NanoTabPFNContinuousPosteriorModel(
            tiny_backbone(41), latent_dim=8, num_samples=8, deviation_clip=2.0
        ).eval()
        episode = tiny_episode()
        expected = model(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clipped.pth"
            save_continuous_checkpoint(
                path,
                model,
                training_config={},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="3" * 64,
                stage="continuous_adapters",
            )
            restored, checkpoint = load_continuous_checkpoint(path)
        self.assertEqual(checkpoint["architecture"]["deviation_clip"], 2.0)
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)

    def test_checkpoint_without_a_clip_reloads_onto_the_legacy_path(self):
        legacy = NanoTabPFNContinuousPosteriorModel(
            tiny_backbone(42), latent_dim=8, num_samples=8, deviation_clip=None
        ).eval()
        episode = tiny_episode()
        expected = legacy(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            save_continuous_checkpoint(
                path,
                legacy,
                training_config={},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="4" * 64,
                stage="continuous_adapters",
            )
            # Checkpoints written before the clip existed omit the key entirely.
            checkpoint = torch.load(path, weights_only=False)
            del checkpoint["architecture"]["deviation_clip"]
            torch.save(checkpoint, path)
            restored, _checkpoint = load_continuous_checkpoint(path)
        self.assertIsNone(restored.deviation_clip)
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)

    def test_beta_round_trip(self):
        model = NanoTabPFNBetaConcentrationModel(tiny_backbone(19), num_samples=8, inference_seed=2).eval()
        episode = tiny_episode()
        expected = model(*episode).sample_positive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beta.pth"
            save_continuous_checkpoint(
                path,
                model,
                training_config={},
                source_checkpoint_path="checkpoints/nanotabpfn.pth",
                source_checkpoint_sha256="1" * 64,
                stage="beta_adapters",
            )
            restored, checkpoint = load_continuous_checkpoint(path)
        self.assertEqual(checkpoint["model_type"], "nanotabpfn_beta_concentration")
        torch.testing.assert_close(restored(*episode).sample_positive, expected, atol=0, rtol=0)

    def test_slot_checkpoints_still_load_and_are_rejected_by_the_new_loader(self):
        slot = NanoTabPFNBayesianModel(tiny_backbone(20), num_hypotheses=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slot.pth"
            save_bayesian_checkpoint(
                path, slot, training_config={}, source_checkpoint_sha256="2" * 64, stage="mean_preserving_frozen"
            )
            from tfmplayground.models.hypothesis import load_bayesian_checkpoint

            restored, _checkpoint = load_bayesian_checkpoint(path)
            self.assertEqual(restored.num_hypotheses, 2)
            with self.assertRaises(ValueError):
                load_continuous_checkpoint(path)


class InterfaceTests(unittest.TestCase):
    def test_context_protocol_matches_the_vanilla_baseline(self):
        from tfmplayground.bayesian_interface import VanillaNanoTabPFNClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] > 0).astype(int)
        vanilla = VanillaNanoTabPFNClassifier(tiny_backbone(21), context_size=32, device="cpu").fit(X, y)
        continuous = ContinuousUncertaintyClassifier(
            tiny_continuous(21), context_size=32, device="cpu", num_samples=8
        ).fit(X, y)
        np.testing.assert_array_equal(continuous._context_indices(), vanilla._context_indices())
        np.testing.assert_array_equal(
            deterministic_context_indices(60, 32, 0), vanilla._context_indices()
        )

    def test_predicted_probabilities_are_the_vanilla_ones(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 3))
        y = (X[:, 1] > 0).astype(int)
        backbone = tiny_backbone(22)
        model = NanoTabPFNContinuousPosteriorModel(backbone, latent_dim=8, num_samples=8)
        classifier = ContinuousUncertaintyClassifier(model, context_size=40, device="cpu", num_samples=8)
        classifier.fit(X[:60], y[:60])
        probabilities = classifier.predict_proba(X[60:])
        from tfmplayground.bayesian_interface import VanillaNanoTabPFNClassifier

        vanilla = VanillaNanoTabPFNClassifier(model.mean_backbone, context_size=40, device="cpu").fit(X[:60], y[:60])
        np.testing.assert_allclose(probabilities, vanilla.predict_proba(X[60:]), atol=1e-6)
        self.assertEqual(set(classifier.classes_), {0, 1})
        for key in ("predictive_entropy", "expected_conditional_entropy", "mutual_information", "epistemic_variance"):
            self.assertEqual(len(classifier.last_diagnostics_[key]), 20)

    def test_query_chunking_is_exact(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(70, 3))
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        model = tiny_continuous(23)
        whole = ContinuousUncertaintyClassifier(
            model, context_size=40, device="cpu", num_samples=8, query_chunk_size=512
        ).fit(X[:50], y[:50])
        chunked = ContinuousUncertaintyClassifier(
            model, context_size=40, device="cpu", num_samples=8, query_chunk_size=4
        ).fit(X[:50], y[:50])
        # Queries attend only to the support rows, so chunking is exact up to
        # the attention kernel that PyTorch selects for a given batch shape.
        np.testing.assert_allclose(whole.predict_proba(X[50:]), chunked.predict_proba(X[50:]), atol=1e-6)
        np.testing.assert_allclose(
            whole.last_diagnostics_["mutual_information"],
            chunked.last_diagnostics_["mutual_information"],
            atol=1e-6,
        )

    def test_context_resampling_classifier_matches_vanilla_probabilities(self):
        from tfmplayground.bayesian_interface import VanillaNanoTabPFNClassifier

        rng = np.random.default_rng(3)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        backbone = tiny_backbone(24)
        baseline = ContextResamplingClassifier(
            backbone, context_size=40, device="cpu", num_subsets=4
        ).fit(X[:60], y[:60])
        vanilla = VanillaNanoTabPFNClassifier(backbone, context_size=40, device="cpu").fit(X[:60], y[:60])
        np.testing.assert_allclose(baseline.predict_proba(X[60:]), vanilla.predict_proba(X[60:]), atol=1e-6)
        self.assertIn("mutual_information", baseline.last_diagnostics_)

    def test_multiclass_targets_are_rejected(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 3))
        y = rng.integers(0, 3, size=30)
        with self.assertRaises(ValueError):
            ContinuousUncertaintyClassifier(tiny_continuous(25), context_size=16, device="cpu").fit(X, y)


class TrainingTests(unittest.TestCase):
    def test_short_cpu_training_run_improves_and_selects_a_checkpoint(self):
        config = ContinuousTrainingConfig(
            num_samples=8,
            batch_size=1,
            max_steps=4,
            validation_interval=2,
            validation_episodes=2,
            validation_support_size=32,
            validation_query_count=4,
            include_scm_families=False,
            latent_dim=8,
        )
        model = build_model(config, tiny_backbone(26))
        trained, history, selection = train(model, config)
        self.assertEqual(len(history), 4)
        self.assertIn(selection["best_step"], (2, 4))
        self.assertTrue(math.isfinite(selection["best_validation_selection_loss"]))
        self.assertLessEqual(
            selection["validation_metrics"]["mean_preservation_error"], 1e-6
        )
        self.assertFalse(any(p.requires_grad for p in trained.mean_backbone.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA training test requires a GPU.")
    def test_cuda_training_step_matches_the_cpu_contract(self):
        config = ContinuousTrainingConfig(
            device="cuda",
            num_samples=8,
            batch_size=1,
            max_steps=2,
            validation_interval=2,
            validation_episodes=2,
            validation_support_size=32,
            validation_query_count=4,
            include_scm_families=False,
            latent_dim=8,
        )
        model = build_model(config, tiny_backbone(30)).to("cuda")
        trained, history, selection = train(model, config)
        self.assertEqual(len(history), 2)
        self.assertTrue(math.isfinite(selection["best_validation_selection_loss"]))
        self.assertLessEqual(selection["validation_metrics"]["mean_preservation_error"], 1e-6)
        self.assertFalse(any(p.requires_grad for p in trained.mean_backbone.parameters()))

    def test_require_cuda_is_asserted_before_training(self):
        from tfmplayground.experiments.train_continuous_bayesian import run_training

        config = ContinuousTrainingConfig(require_cuda=True, device="cpu")
        if torch.cuda.is_available():
            self.skipTest("The assertion only fires without CUDA.")
        with self.assertRaises(AssertionError):
            run_training(config)

    def test_beta_training_configuration_builds(self):
        config = ContinuousTrainingConfig(model_type="beta", num_samples=8, latent_dim=8)
        model = build_model(config, tiny_backbone(27))
        self.assertIsInstance(model, NanoTabPFNBetaConcentrationModel)


class SyntheticEvaluationTests(unittest.TestCase):
    def test_arm_evaluation_reports_every_required_metric(self):
        config = SyntheticEvaluationConfig(
            episodes_per_condition=2, support_size=32, query_count=4, num_samples=8
        )
        report = evaluate_arm(tiny_continuous(28), config)
        for key in (
            "energy_distance",
            "mutual_information_mae",
            "epistemic_variance_mae",
            "covariance_error",
            "joint_query_nll",
            "posterior_sample_coverage",
            "candidate_mass_total_variation",
            "sample_mean_preservation_error",
        ):
            self.assertIn(key, report["headline"])
            self.assertIn(key, METRIC_DIRECTION)
        self.assertLessEqual(report["headline"]["deployed_probability_difference"], 1e-6)
        self.assertIn("qualitative", report)
        self.assertEqual(set(report["evidence"]), {"32", "64", "128", "256", "512"})

    def test_risk_lambda_is_selected_from_the_declared_grid(self):
        config = SyntheticEvaluationConfig(episodes_per_condition=4, support_size=32, query_count=4, num_samples=8)
        selection = select_risk_lambda(tiny_continuous(29), config)
        self.assertIn(selection["selected_lambda"], RISK_LAMBDAS)
        self.assertEqual(len(selection["grid"]), len(RISK_LAMBDAS))
        self.assertIn("synthetic", selection["selection_data"])

    def test_representation_and_posterior_gates(self):
        frozen = {"headline": {"mutual_information_mae": 0.10, "epistemic_variance_mae": 0.10, "joint_query_nll": 2.0}}
        adapter = {
            "headline": {"mutual_information_mae": 0.08, "epistemic_variance_mae": 0.10, "joint_query_nll": 2.01}
        }
        gate = representation_benefit(adapter, frozen)
        self.assertTrue(gate["passed"])
        worse = {
            "headline": {"mutual_information_mae": 0.08, "epistemic_variance_mae": 0.10, "joint_query_nll": 2.5}
        }
        self.assertFalse(representation_benefit(worse, frozen)["passed"])

        beta = {
            "headline": {
                "energy_distance": 0.3,
                "mutual_information_mae": 0.2,
                "epistemic_variance_mae": 0.2,
                "covariance_error": 0.2,
                "joint_query_nll": 2.5,
            }
        }
        better = {
            "headline": {
                "energy_distance": 0.2,
                "mutual_information_mae": 0.1,
                "epistemic_variance_mae": 0.3,
                "covariance_error": 0.3,
                "joint_query_nll": 2.6,
            }
        }
        self.assertTrue(continuous_posterior_benefit(better, beta)["passed"])


class TabArenaEvaluationTests(unittest.TestCase):
    def test_task_list_matches_the_slot_trial(self):
        self.assertEqual(len(TABARENA_BINARY_TASK_NAMES), 20)
        self.assertEqual(len(set(TABARENA_BINARY_TASK_NAMES)), 20)
        self.assertIn("jm1", TABARENA_BINARY_TASK_NAMES)

    def test_error_detection_scores(self):
        scores = np.asarray([0.1, 0.2, 0.9, 0.8])
        errors = np.asarray([0, 0, 1, 1])
        result = error_detection_scores(scores, errors)
        self.assertAlmostEqual(result["error_auroc"], 1.0)
        self.assertLess(result["aurc"], 0.5)
        constant = error_detection_scores(scores, np.zeros(4, dtype=int))
        self.assertIsNone(constant["error_auroc"])

    def test_gate_requires_improvement_without_material_harm(self):
        vanilla = {"error_auroc": 0.75, "aurc": 0.10}
        useful = tabarena_gate(
            {"raw_epistemic": {"error_auroc": 0.77, "aurc": 0.09}, "combined_risk": None}, vanilla
        )
        self.assertTrue(useful["passed"])
        harmful = tabarena_gate(
            {"raw_epistemic": {"error_auroc": 0.60, "aurc": 0.30}, "combined_risk": None}, vanilla
        )
        self.assertFalse(harmful["passed"])
        neutral = tabarena_gate(
            {"raw_epistemic": {"error_auroc": 0.752, "aurc": 0.099}, "combined_risk": None}, vanilla
        )
        self.assertFalse(neutral["passed"])

    def test_evaluate_split_shares_the_context_and_preserves_the_mean(self):
        from tfmplayground.bayesian_interface import VanillaNanoTabPFNClassifier
        from tfmplayground.experiments.evaluate_continuous_tabarena import evaluate_split

        rng = np.random.default_rng(12)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] - X[:, 3] > 0).astype(int)
        backbone = tiny_backbone(31)
        arms = {
            "vanilla": VanillaNanoTabPFNClassifier(backbone, context_size=64, device="cpu"),
            "context_resampling": ContextResamplingClassifier(
                backbone, context_size=64, device="cpu", num_subsets=4
            ),
            "adapter_continuous": ContinuousUncertaintyClassifier(
                NanoTabPFNContinuousPosteriorModel(backbone, latent_dim=8, num_samples=8),
                context_size=64,
                device="cpu",
                num_samples=8,
            ),
        }
        report = evaluate_split(X[:90], y[:90], X[90:], y[90:], arms=arms, risk_lambda=0.5)
        self.assertTrue(report["context_indices_identical"])
        self.assertFalse(report["query_labels_used_for_construction"])
        for name, entry in report["arms"].items():
            self.assertLessEqual(entry["max_probability_difference_to_vanilla"], 1e-6, name)
            self.assertIn("nll", entry["predictive"])
        self.assertIn("raw_epistemic", report["arms"]["adapter_continuous"])
        self.assertIn("combined_risk", report["arms"]["adapter_continuous"])
        self.assertNotIn("raw_epistemic", report["arms"]["vanilla"])

    def test_summarize_reports_bootstrap_intervals_and_equality(self):
        def report(auroc: float) -> dict:
            return {
                "vanilla_predictive_entropy": {"error_auroc": 0.7, "aurc": 0.1},
                "arms": {
                    "adapter_continuous": {
                        "predictive": {"nll": 0.3, "brier": 0.1, "roc_auc": 0.8, "accuracy": 0.85, "ece": 0.03},
                        "max_probability_difference_to_vanilla": 1e-8,
                        "raw_epistemic": {"error_auroc": auroc, "aurc": 0.09},
                        "combined_risk": {"error_auroc": auroc + 0.01, "aurc": 0.088},
                    }
                },
            }

        summary = summarize([report(0.72), report(0.74), report(0.76)], arm_names=["adapter_continuous"])
        entry = summary["arms"]["adapter_continuous"]
        self.assertTrue(entry["matches_vanilla_probabilities_at_1e-6"])
        self.assertAlmostEqual(entry["raw_epistemic"]["error_auroc"], 0.74, places=6)
        self.assertIn("raw_epistemic.error_auroc", entry["bootstrap"])
        self.assertTrue(entry["gate"]["passed"])


class SweepTests(unittest.TestCase):
    def test_screening_grid_sizes(self):
        configurations = screening_configurations()
        self.assertEqual(len(configurations), 68)
        counts: dict[str, int] = {}
        for configuration in configurations:
            counts[configuration["architecture"]] = counts.get(configuration["architecture"], 0) + 1
        self.assertEqual(counts["adapter_continuous"], 36)
        self.assertEqual(counts["frozen_continuous"], 12)
        self.assertEqual(counts["full_continuous"], 8)
        self.assertEqual(counts["beta_adapter"], 12)
        self.assertEqual(len({configuration_label(item) for item in configurations}), 68)
        self.assertTrue(all(item["num_samples"] == 32 for item in configurations))
        # Every architecture is screened at both moment weights so that the
        # adapter-versus-frozen and adapter-versus-Beta gates stay like for like.
        for architecture in ("adapter_continuous", "frozen_continuous", "full_continuous", "beta_adapter"):
            weights = {
                item["moment_weight"] for item in configurations if item["architecture"] == architecture
            }
            self.assertEqual(weights, {1.0, 4.0})

    def test_flags_encode_the_screening_and_final_budgets(self):
        screening = configuration_flags(0)
        self.assertIn("--max-steps 1500", screening)
        self.assertIn("--patience 8", screening)
        self.assertIn("--validation-interval 100", screening)
        self.assertIn("--moment-weight 1", screening)
        final = configuration_flags(0, final=True, seed=2403)
        self.assertIn("--max-steps 5000", final)
        self.assertIn("--patience 10", final)
        self.assertIn("--seed 2403", final)
        self.assertIn("--moment-weight 4", configuration_flags(1))
        with self.assertRaises(IndexError):
            configuration_flags(68)

    def test_summarize_selects_two_configurations_per_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, loss in enumerate([0.5, 0.4, 0.6]):
                run = root / f"run-{index}"
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "continuous",
                            "uncertainty_mode": "adapters",
                            "adapter_bottleneck": 32,
                            "latent_dim": 32,
                            "learning_rate": 1e-4,
                            "seed": 2402,
                        }
                    )
                )
                (run / "selection.json").write_text(
                    json.dumps(
                        {
                            "best_step": 100 * (index + 1),
                            "best_validation_selection_loss": loss,
                            "selected_checkpoint": str(run / "best.pth"),
                        }
                    )
                )
            summary = summarize_sweep(root)
        selected = summary["selected"]["adapter_continuous"]
        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(selected[0]["validation_selection_loss"], 0.4)
        self.assertEqual(summary["final_seeds"], [2402, 2403, 2404])


class BinaryOutcomeTests(unittest.TestCase):
    def test_enumeration_is_canonical(self):
        outcomes = all_binary_outcomes(3)
        self.assertEqual(outcomes.shape, (8, 3))
        self.assertTrue(torch.equal(outcomes[0], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(outcomes[-1], torch.ones(3, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
