import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tfmplayground.experiments.run_task_posterior_local_evaluation import (
    LocalEvaluationConfig,
    _parameter_groups,
    _probe_r2,
)
from tfmplayground.experiments.structural_latents import (
    SCALAR_NAMES,
    StructuralLatentSchema,
    structural_feature_mask,
    structural_latent_vector,
)
from tfmplayground.experiments.train_task_posterior_adapter import (
    TaskPosteriorTrainingConfig,
    contrastive_episode_objective,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.task_posterior_adapter import (
    NanoTabPFNTaskPosteriorAdapter,
    RegimeParticleAssignment,
    TaskPosteriorPrediction,
    load_task_posterior_checkpoint,
    save_task_posterior_checkpoint,
    structural_latent_loss,
    task_posterior_loss,
)


def tiny_backbone(num_outputs: int = 3) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=2,
        num_outputs=num_outputs,
    )


class StructuralLatentSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = StructuralLatentSchema(max_features=6)

    def test_layout_places_the_family_block_last(self):
        self.assertEqual(self.schema.latent_dim, 6 + len(SCALAR_NAMES) + 2)
        self.assertEqual(self.schema.family_slice.stop, self.schema.latent_dim)
        self.assertEqual(self.schema.family_slice.start, self.schema.continuous_dim)
        self.assertEqual(len(self.schema.dimension_names()), self.schema.latent_dim)

    def test_rejects_degenerate_schemas(self):
        with self.assertRaises(ValueError):
            StructuralLatentSchema(max_features=0)
        with self.assertRaises(ValueError):
            StructuralLatentSchema(max_features=4, families=("mlp_scm",))

    def test_feature_mask_marks_only_real_features(self):
        mask = structural_feature_mask(2, schema=self.schema)
        torch.testing.assert_close(mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            structural_feature_mask(7, schema=self.schema)


class StructuralLatentVectorTests(unittest.TestCase):
    def setUp(self):
        self.schema = StructuralLatentSchema(max_features=4)
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randn(160, 2, generator=generator)
        self.y = (self.x[:, 0] > 0).long()

    def vector(self, x, y):
        return structural_latent_vector(x, y, schema=self.schema, family="mlp_scm", class_count=2)

    def test_relevance_separates_informative_from_irrelevant_features(self):
        relevance = self.vector(self.x, self.y)[self.schema.relevance_slice]
        self.assertGreater(float(relevance[0]), 0.9)
        self.assertLess(float(relevance[1]), 0.2)
        # Padding beyond the table's real width stays exactly zero.
        torch.testing.assert_close(relevance[2:], torch.zeros(2))

    def test_relevance_follows_a_column_permutation(self):
        baseline = self.vector(self.x, self.y)[self.schema.relevance_slice]
        permuted = self.vector(self.x[:, [1, 0]], self.y)[self.schema.relevance_slice]
        torch.testing.assert_close(permuted[:2], baseline[:2].flip(0))

    def test_label_noise_entry_rises_with_flipped_labels(self):
        offset = self.schema.scalar_slice.start
        clean = float(self.vector(self.x, self.y)[offset + SCALAR_NAMES.index("label_noise")])
        generator = torch.Generator().manual_seed(11)
        flips = torch.rand(self.y.shape, generator=generator) < 0.30
        noisy_y = torch.where(flips, 1 - self.y, self.y)
        noisy = float(self.vector(self.x, noisy_y)[offset + SCALAR_NAMES.index("label_noise")])
        self.assertGreater(noisy, clean + 0.10)

    def test_complexity_entry_separates_a_threshold_rule_from_xor(self):
        offset = self.schema.scalar_slice.start + SCALAR_NAMES.index("boundary_complexity")
        threshold = float(self.vector(self.x, self.y)[offset])
        xor_y = ((self.x[:, 0] > 0) ^ (self.x[:, 1] > 0)).long()
        xor = float(self.vector(self.x, xor_y)[offset])
        self.assertLess(threshold, 0.05)
        self.assertGreater(xor, 0.30)

    def test_class_balance_entry_is_normalized_entropy(self):
        offset = self.schema.scalar_slice.start + SCALAR_NAMES.index("class_balance")
        balanced = float(self.vector(self.x, self.y)[offset])
        skewed_y = torch.zeros_like(self.y)
        skewed_y[:8] = 1
        skewed = float(self.vector(self.x, skewed_y)[offset])
        self.assertGreater(balanced, 0.95)
        self.assertLess(skewed, 0.35)

    def test_family_block_is_a_one_hot_of_the_generating_family(self):
        mlp = self.vector(self.x, self.y)[self.schema.family_slice]
        tree = structural_latent_vector(self.x, self.y, schema=self.schema, family="tree_scm", class_count=2)[
            self.schema.family_slice
        ]
        torch.testing.assert_close(mlp, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(tree, torch.tensor([0.0, 1.0]))

    def test_an_unknown_family_leaves_the_block_all_zero(self):
        vector = structural_latent_vector(self.x, self.y, schema=self.schema, family=None, class_count=2)
        torch.testing.assert_close(vector[self.schema.family_slice], torch.zeros(2))
        # Every other block is unaffected by losing the family.
        known = self.vector(self.x, self.y)
        torch.testing.assert_close(
            vector[: self.schema.continuous_dim], known[: self.schema.continuous_dim], rtol=0, atol=0
        )

    def test_rejects_unknown_families_and_oversized_tables(self):
        with self.assertRaises(ValueError):
            structural_latent_vector(self.x, self.y, schema=self.schema, family="gp", class_count=2)
        with self.assertRaises(ValueError):
            structural_latent_vector(
                torch.randn(10, 9), torch.zeros(10, dtype=torch.long), schema=self.schema, family="mlp_scm"
            )

    def test_summaries_are_computed_on_cpu_regardless_of_input_device(self):
        # MPS has no float64, so the summary must not follow the batch device.
        self.assertEqual(self.vector(self.x, self.y).device.type, "cpu")

    def test_constant_column_and_single_class_labels_stay_finite(self):
        constant = torch.ones(20, 2)
        labels = torch.zeros(20, dtype=torch.long)
        vector = self.vector(constant, labels)
        self.assertTrue(bool(torch.isfinite(vector).all()))
        torch.testing.assert_close(vector[self.schema.relevance_slice], torch.zeros(4))


class StructuralLatentLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.family_count = 2
        self.latent_dim = 7
        self.continuous_dim = self.latent_dim - self.family_count
        self.batch, self.particles, self.candidates = 2, 4, 2
        self.structural_z = torch.rand(self.batch, self.candidates, self.latent_dim)
        self.structural_z[..., self.continuous_dim :] = 0.0
        self.structural_z[:, 0, self.continuous_dim] = 1.0
        self.structural_z[:, 1, self.continuous_dim + 1] = 1.0
        self.assignment = RegimeParticleAssignment(torch.tensor([[0, 1], [2, 3]]))
        self.probe_output = torch.zeros(self.batch, self.particles, self.latent_dim)
        for row in range(self.batch):
            for candidate in range(self.candidates):
                particle = int(self.assignment.particle_for_regime[row, candidate])
                target = self.structural_z[row, candidate]
                self.probe_output[row, particle, : self.continuous_dim] = target[: self.continuous_dim]
                # Confident family logits stand in for a perfect one-hot.
                self.probe_output[row, particle, self.continuous_dim :] = 50.0 * target[self.continuous_dim :]

    def prediction(self, structural: torch.Tensor | None) -> TaskPosteriorPrediction:
        return TaskPosteriorPrediction(
            vanilla_logits=torch.zeros(self.batch, 3, 2),
            particle_logits=torch.zeros(self.batch, 3, self.particles, 2),
            log_weights=torch.full((self.batch, self.particles), -torch.tensor(4.0).log()),
            slots=torch.zeros(self.batch, self.particles, 8),
            residuals=torch.zeros(self.batch, 3, self.particles, 2),
            structural=structural,
        )

    def test_perfect_probe_output_scores_near_zero(self):
        loss = structural_latent_loss(
            self.prediction(self.probe_output),
            self.structural_z,
            self.assignment,
            family_count=self.family_count,
        )
        self.assertLess(float(loss), 1e-6)

    def test_perturbing_a_matched_particle_increases_the_loss(self):
        perturbed = self.probe_output.clone()
        perturbed[0, 0, 0] += 0.5
        baseline = structural_latent_loss(
            self.prediction(self.probe_output), self.structural_z, self.assignment, family_count=self.family_count
        )
        worse = structural_latent_loss(
            self.prediction(perturbed), self.structural_z, self.assignment, family_count=self.family_count
        )
        self.assertGreater(float(worse), float(baseline))

    def test_an_all_zero_family_block_drops_the_family_term(self):
        unknown = self.structural_z.clone()
        unknown[..., self.continuous_dim :] = 0.0
        # Wrong family logits must not be penalized when the family is unknown.
        wrong = self.probe_output.clone()
        wrong[..., self.continuous_dim :] *= -1.0
        baseline = structural_latent_loss(
            self.prediction(self.probe_output), unknown, self.assignment, family_count=self.family_count
        )
        same = structural_latent_loss(self.prediction(wrong), unknown, self.assignment, family_count=self.family_count)
        torch.testing.assert_close(same, baseline)
        self.assertLess(float(baseline), 1e-6)

    def test_a_known_family_still_penalizes_wrong_logits(self):
        wrong = self.probe_output.clone()
        wrong[..., self.continuous_dim :] *= -1.0
        penalized = structural_latent_loss(
            self.prediction(wrong), self.structural_z, self.assignment, family_count=self.family_count
        )
        self.assertGreater(float(penalized), 1.0)

    def test_unmatched_particles_are_ignored(self):
        untouched = self.probe_output.clone()
        untouched[0, 2] += 5.0
        untouched[0, 3] += 5.0
        baseline = structural_latent_loss(
            self.prediction(self.probe_output), self.structural_z, self.assignment, family_count=self.family_count
        )
        same = structural_latent_loss(
            self.prediction(untouched), self.structural_z, self.assignment, family_count=self.family_count
        )
        torch.testing.assert_close(same, baseline)

    def test_masked_relevance_entries_do_not_contribute(self):
        mask = torch.zeros(self.batch, 3)
        mask[:, 0] = 1.0
        perturbed = self.probe_output.clone()
        perturbed[0, 0, 2] += 5.0
        baseline = structural_latent_loss(
            self.prediction(self.probe_output),
            self.structural_z,
            self.assignment,
            family_count=self.family_count,
            feature_mask=mask,
        )
        same = structural_latent_loss(
            self.prediction(perturbed),
            self.structural_z,
            self.assignment,
            family_count=self.family_count,
            feature_mask=mask,
        )
        torch.testing.assert_close(same, baseline)

    def test_requires_a_probe_and_a_consistent_assignment(self):
        with self.assertRaises(ValueError):
            structural_latent_loss(
                self.prediction(None), self.structural_z, self.assignment, family_count=self.family_count
            )
        wrong = RegimeParticleAssignment(torch.tensor([[0, 1]]))
        with self.assertRaises(ValueError):
            structural_latent_loss(
                self.prediction(self.probe_output), self.structural_z, wrong, family_count=self.family_count
            )


class StructuralProbeAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.schema = StructuralLatentSchema(max_features=5)
        generator = torch.Generator().manual_seed(42)
        self.context_x = torch.randn(2, 12, 5, generator=generator)
        self.context_y = torch.randint(0, 3, (2, 12), generator=generator)
        self.query_x = torch.randn(2, 6, 5, generator=generator)

    def adapter(self, **kwargs) -> NanoTabPFNTaskPosteriorAdapter:
        return NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(),
            particle_count=4,
            max_classes=3,
            structural_latent_dim=self.schema.latent_dim,
            structural_family_count=self.schema.family_count,
            **kwargs,
        ).eval()

    def test_a_probe_does_not_disturb_vanilla_identity(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        vanilla = model.backbone(self.context_x, self.context_y.float(), self.query_x)[..., :3]
        torch.testing.assert_close(
            prediction.particle_logits, vanilla[:, :, None].expand(-1, -1, 4, -1), rtol=0, atol=0
        )
        self.assertIsNotNone(prediction.structural)
        assert prediction.structural is not None
        self.assertEqual(tuple(prediction.structural.shape), (2, 4, self.schema.latent_dim))

    def test_probe_output_is_bounded_on_the_continuous_block(self):
        model = self.adapter()
        with torch.no_grad():
            model.structural_probe.output.weight.mul_(50.0)
            model.structural_probe.output.bias.add_(30.0)
        structural = model(self.context_x, self.context_y, self.query_x, class_count=3).structural
        assert structural is not None
        continuous = structural[..., : self.schema.continuous_dim]
        self.assertTrue(bool(((continuous >= 0.0) & (continuous <= 1.0)).all()))

    def test_detached_probe_leaves_slots_free_of_structural_gradient(self):
        detached = self.adapter(structural_detach=True).train()
        shaping = self.adapter(structural_detach=False).train()
        for model, expect_gradient in ((detached, False), (shaping, True)):
            model.zero_grad()
            structural = model(self.context_x, self.context_y, self.query_x, class_count=3).structural
            assert structural is not None
            structural.square().mean().backward()
            gradient = model.slot_queries.grad
            has_gradient = gradient is not None and bool(gradient.abs().sum() > 0)
            self.assertEqual(has_gradient, expect_gradient)

    def test_zero_weight_keeps_the_total_loss_unchanged(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        target = torch.randint(0, 3, (2, 6))
        candidates = torch.randint(0, 3, (2, 2, 6))
        structural_z = torch.rand(2, 2, self.schema.latent_dim)
        baseline = task_posterior_loss(prediction, target, candidate_y=candidates)
        with_structure = task_posterior_loss(
            prediction,
            target,
            candidate_y=candidates,
            structural_z=structural_z,
            structural_family_count=self.schema.family_count,
            structural_weight=0.0,
            assignment=baseline.assignment,
        )
        torch.testing.assert_close(with_structure.total, baseline.total, rtol=0, atol=0)
        torch.testing.assert_close(with_structure.structural, torch.zeros(()), rtol=0, atol=0)

    def test_missing_structural_truth_skips_the_term_without_nan(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        target = torch.randint(0, 3, (2, 6))
        candidates = torch.randint(0, 3, (2, 2, 6))
        losses = task_posterior_loss(
            prediction,
            target,
            candidate_y=candidates,
            structural_z=None,
            structural_weight=1.0,
        )
        self.assertTrue(bool(torch.isfinite(losses.total)))
        torch.testing.assert_close(losses.structural, torch.zeros(()), rtol=0, atol=0)

    def test_structural_supervision_without_candidates_is_refused(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        with self.assertRaises(ValueError):
            task_posterior_loss(
                prediction,
                torch.randint(0, 3, (2, 6)),
                structural_z=torch.rand(2, 2, self.schema.latent_dim),
                structural_family_count=self.schema.family_count,
                structural_weight=1.0,
            )


class GaussianSlotModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        generator = torch.Generator().manual_seed(42)
        self.context_x = torch.randn(2, 12, 5, generator=generator)
        self.context_y = torch.randint(0, 3, (2, 12), generator=generator)
        self.query_x = torch.randn(2, 6, 5, generator=generator)

    def adapter(self, **kwargs) -> NanoTabPFNTaskPosteriorAdapter:
        return NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(), particle_count=4, max_classes=3, slot_mode="gaussian", **kwargs
        ).eval()

    def test_rejects_unknown_slot_modes(self):
        with self.assertRaises(ValueError):
            NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), max_classes=3, slot_mode="variational")

    def test_sampled_particles_are_still_exactly_vanilla(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        vanilla = model.backbone(self.context_x, self.context_y.float(), self.query_x)[..., :3]
        torch.testing.assert_close(prediction.vanilla_logits, vanilla, rtol=0, atol=0)
        torch.testing.assert_close(
            prediction.particle_logits, vanilla[:, :, None].expand(-1, -1, 4, -1), rtol=0, atol=0
        )

    def test_a_single_query_bank_still_produces_distinct_slots(self):
        model = self.adapter()
        self.assertEqual(tuple(model.slot_queries.shape), (1, 8))
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        self.assertEqual(tuple(prediction.slots.shape), (2, 4, 8))
        self.assertGreater(float(prediction.slot_dispersion().min()), 0.0)

    def test_evaluation_is_reproducible_and_training_resamples(self):
        model = self.adapter()
        first = model(self.context_x, self.context_y, self.query_x, class_count=3).slots
        second = model(self.context_x, self.context_y, self.query_x, class_count=3).slots
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        model.train()
        torch.manual_seed(0)
        third = model(self.context_x, self.context_y, self.query_x, class_count=3).slots
        fourth = model(self.context_x, self.context_y, self.query_x, class_count=3).slots
        self.assertGreater(float((third - fourth).abs().max()), 0.0)

    def test_a_different_sample_seed_gives_a_different_evaluation_draw(self):
        baseline = self.adapter(slot_sample_seed=0)
        shifted = self.adapter(slot_sample_seed=1)
        shifted.load_state_dict(baseline.state_dict())
        first = baseline(self.context_x, self.context_y, self.query_x, class_count=3).slots
        second = shifted(self.context_x, self.context_y, self.query_x, class_count=3).slots
        self.assertGreater(float((first - second).abs().max().detach()), 0.0)

    def test_particle_count_is_a_runtime_knob(self):
        model = self.adapter()
        model.particle_count = 8
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        self.assertEqual(tuple(prediction.log_weights.shape), (2, 8))
        self.assertEqual(tuple(prediction.particle_logits.shape), (2, 6, 8, 3))

    def test_kl_starts_finite_and_is_reported_by_the_loss(self):
        model = self.adapter()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        assert prediction.kl is not None
        self.assertTrue(bool(torch.isfinite(prediction.kl).all()))
        losses = task_posterior_loss(prediction, torch.randint(0, 3, (2, 6)), kl_weight=0.5)
        torch.testing.assert_close(losses.kl, prediction.kl.mean())

    def test_deterministic_mode_reports_no_kl(self):
        model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3).eval()
        prediction = model(self.context_x, self.context_y, self.query_x, class_count=3)
        self.assertIsNone(prediction.kl)
        losses = task_posterior_loss(prediction, torch.randint(0, 3, (2, 6)), kl_weight=1.0)
        torch.testing.assert_close(losses.kl, torch.zeros(()), rtol=0, atol=0)

    def test_extreme_features_do_not_produce_non_finite_slots(self):
        model = self.adapter()
        prediction = model(self.context_x * 1e6, self.context_y, self.query_x * 1e6, class_count=3)
        assert prediction.kl is not None
        self.assertTrue(bool(torch.isfinite(prediction.slots).all()))
        self.assertTrue(bool(torch.isfinite(prediction.particle_logits).all()))
        self.assertTrue(bool(torch.isfinite(prediction.kl).all()))


class StructuralTrainingObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.schema = StructuralLatentSchema(max_features=4)

    def batch(self, *, structure: bool):
        generator = torch.Generator().manual_seed(1)
        fields = {
            "initial_support_x": torch.randn(2, 10, 4, generator=generator),
            "initial_support_y": torch.randint(0, 2, (2, 10), generator=generator),
            "stream_x": torch.randn(2, 6, 4, generator=generator),
            "stream_y": torch.randint(0, 2, (2, 6), generator=generator),
            "query_x": torch.randn(2, 4, 4, generator=generator),
            "query_y": torch.randint(0, 2, (2, 4), generator=generator),
            "candidate_stream_y": torch.randint(0, 2, (2, 2, 6), generator=generator),
            "candidate_query_y": torch.randint(0, 2, (2, 2, 4), generator=generator),
            "candidate_structural_z": None,
            "structural_feature_mask": None,
        }
        if structure:
            fields["candidate_structural_z"] = torch.rand(2, 2, self.schema.latent_dim, generator=generator)
            fields["structural_feature_mask"] = torch.ones(2, 4)
        return SimpleNamespace(**fields)

    def model(self, slot_mode: str) -> NanoTabPFNTaskPosteriorAdapter:
        return NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(),
            particle_count=4,
            max_classes=3,
            slot_mode=slot_mode,
            structural_latent_dim=self.schema.latent_dim,
            structural_family_count=self.schema.family_count,
        ).train()

    def config(self, **kwargs) -> TaskPosteriorTrainingConfig:
        return TaskPosteriorTrainingConfig(structural_family_count=self.schema.family_count, **kwargs)

    def test_both_slot_modes_train_end_to_end(self):
        for slot_mode in ("deterministic", "gaussian"):
            with self.subTest(slot_mode=slot_mode):
                model = self.model(slot_mode)
                objective = contrastive_episode_objective(
                    model, self.batch(structure=True), self.config(structural_weight=0.5, kl_weight=0.1)
                )
                self.assertGreater(float(objective.prior_only.structural.detach()), 0.0)
                objective.total.backward()
                gradients = {
                    name: parameter.grad
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not name.startswith("backbone.")
                }
                self.assertGreater(float(gradients["structural_probe.output.weight"].abs().sum()), 0.0)
                self.assertGreater(float(gradients["slot_queries"].abs().sum()), 0.0)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients.values()))

    def test_kl_is_reported_only_by_the_gaussian_arm(self):
        batch = self.batch(structure=True)
        config = self.config(structural_weight=0.5, kl_weight=0.1)
        deterministic = contrastive_episode_objective(self.model("deterministic"), batch, config)
        gaussian = contrastive_episode_objective(self.model("gaussian"), batch, config)
        torch.testing.assert_close(deterministic.prior_only.kl, torch.zeros(()), rtol=0, atol=0)
        self.assertGreater(float(gaussian.prior_only.kl.detach()), 0.0)

    def test_ordinary_episodes_without_structure_still_train(self):
        model = self.model("deterministic")
        objective = contrastive_episode_objective(
            model, self.batch(structure=False), self.config(structural_weight=0.5)
        )
        torch.testing.assert_close(objective.prior_only.structural, torch.zeros(()), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(objective.total)))

    def test_structure_without_candidate_labels_is_dropped_not_raised(self):
        # An ordinary episode stripped of its candidate labels can still carry a
        # structural tensor; without candidates there is no matching, so the
        # objective must skip the term rather than fail.
        batch = self.batch(structure=True)
        batch.candidate_stream_y = None
        batch.candidate_query_y = None
        objective = contrastive_episode_objective(
            self.model("deterministic"), batch, self.config(structural_weight=0.5)
        )
        torch.testing.assert_close(objective.prior_only.structural, torch.zeros(()), rtol=0, atol=0)
        torch.testing.assert_close(objective.updated.structural, torch.zeros(()), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(objective.total)))

    def test_updated_pass_drops_structure_when_only_query_labels_are_missing(self):
        batch = self.batch(structure=True)
        batch.candidate_query_y = None
        objective = contrastive_episode_objective(
            self.model("deterministic"), batch, self.config(structural_weight=0.5)
        )
        torch.testing.assert_close(objective.updated.structural, torch.zeros(()), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(objective.total)))

    def test_a_probe_free_model_ignores_structural_truth(self):
        model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3).train()
        objective = contrastive_episode_objective(model, self.batch(structure=True), self.config(structural_weight=0.5))
        torch.testing.assert_close(objective.prior_only.structural, torch.zeros(()), rtol=0, atol=0)

    def test_negative_weights_are_refused(self):
        with self.assertRaises(ValueError):
            self.config(structural_weight=-0.1).validate()
        with self.assertRaises(ValueError):
            self.config(kl_weight=-1.0).validate()


class LocalEvaluationWiringTests(unittest.TestCase):
    def test_both_episode_sources_accept_a_structural_probe(self):
        for source in ("h5", "tabicl"):
            LocalEvaluationConfig(
                output_dir="x", structural_probe=True, structural_weight=1.0, episode_source=source
            ).validate()

    def test_an_untrained_probe_is_refused(self):
        # Detach stops the gradient at the slots, not the probe; zero weight
        # leaves the probe random and its R2 meaningless.
        with self.assertRaises(ValueError):
            LocalEvaluationConfig(output_dir="x", structural_probe=True, structural_detach=True).validate()

    def test_weights_without_a_probe_are_refused(self):
        with self.assertRaises(ValueError):
            LocalEvaluationConfig(output_dir="x", structural_weight=0.5).validate()
        with self.assertRaises(ValueError):
            LocalEvaluationConfig(output_dir="x", structural_detach=True).validate()

    def test_unknown_episode_sources_are_refused(self):
        with self.assertRaises(ValueError):
            LocalEvaluationConfig(output_dir="x", episode_source="openml").validate()

    def test_probe_r2_is_one_for_a_perfect_probe_and_zero_for_the_mean(self):
        schema = StructuralLatentSchema(max_features=3)
        generator = torch.Generator().manual_seed(2)
        targets = torch.rand(8, 2, schema.latent_dim, generator=generator)
        targets[..., schema.family_slice] = 0.0
        targets[:, 0, schema.family_slice.start] = 1.0
        targets[:, 1, schema.family_slice.start + 1] = 1.0

        perfect = _probe_r2([targets.clone()], [targets], schema)
        self.assertAlmostEqual(perfect["mean_continuous_r2"], 1.0, places=6)
        self.assertAlmostEqual(perfect["family_accuracy"], 1.0, places=6)

        mean_only = targets.flatten(0, 1).mean(0).expand_as(targets.flatten(0, 1)).reshape(targets.shape)
        baseline = _probe_r2([mean_only.contiguous()], [targets], schema)
        self.assertAlmostEqual(baseline["mean_continuous_r2"], 0.0, places=6)

    def test_probe_r2_reports_no_family_accuracy_when_the_family_is_unknown(self):
        schema = StructuralLatentSchema(max_features=3)
        generator = torch.Generator().manual_seed(4)
        targets = torch.rand(8, 2, schema.latent_dim, generator=generator)
        targets[..., schema.family_slice] = 0.0
        report = _probe_r2([targets.clone()], [targets], schema)
        self.assertIsNone(report["family_accuracy"])
        self.assertEqual(report["family_known_fraction"], 0.0)

    def test_parameter_groups_separate_the_probe_from_the_adapter(self):
        schema = StructuralLatentSchema(max_features=4)
        model = NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(),
            particle_count=4,
            max_classes=3,
            structural_latent_dim=schema.latent_dim,
            structural_family_count=schema.family_count,
        )
        adapter, probe = _parameter_groups(model)
        probe_ids = {id(parameter) for parameter in model.structural_probe.parameters()}
        self.assertEqual({id(parameter) for parameter in probe}, probe_ids)
        self.assertTrue(probe_ids.isdisjoint({id(parameter) for parameter in adapter}))
        backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
        self.assertTrue(backbone_ids.isdisjoint({id(parameter) for parameter in adapter + probe}))
        # Together they must still cover everything model.adapter_parameters() trains.
        self.assertEqual(
            {id(parameter) for parameter in model.adapter_parameters()},
            {id(parameter) for parameter in adapter + probe},
        )

    def test_a_probe_free_model_has_no_probe_group(self):
        model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3)
        adapter, probe = _parameter_groups(model)
        self.assertEqual(probe, [])
        self.assertTrue(adapter)

    def test_probe_r2_is_empty_without_predictions(self):
        self.assertEqual(_probe_r2([], [], StructuralLatentSchema(max_features=3)), {})


class StructuralCheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_slot_mode_and_probe(self):
        torch.manual_seed(5)
        schema = StructuralLatentSchema(max_features=5)
        model = NanoTabPFNTaskPosteriorAdapter(
            tiny_backbone(),
            particle_count=4,
            max_classes=3,
            slot_mode="gaussian",
            slot_sample_seed=13,
            structural_latent_dim=schema.latent_dim,
            structural_family_count=schema.family_count,
            structural_detach=True,
        ).eval()
        generator = torch.Generator().manual_seed(6)
        context_x = torch.randn(2, 12, 5, generator=generator)
        context_y = torch.randint(0, 3, (2, 12), generator=generator)
        query_x = torch.randn(2, 6, 5, generator=generator)
        with torch.no_grad():
            model.structural_probe.output.weight.normal_()
            model.residual_decoder.output.weight.normal_(std=0.01)
        expected = model(context_x, context_y, query_x, class_count=3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            save_task_posterior_checkpoint(path, model, training_config={}, lineage={}, data_provenance={})
            restored, _ = load_task_posterior_checkpoint(path)
        restored.eval()

        self.assertEqual(restored.slot_mode, "gaussian")
        self.assertEqual(restored.slot_sample_seed, 13)
        self.assertTrue(restored.structural_detach)
        assert restored.structural_probe is not None
        self.assertEqual(restored.structural_probe.latent_dim, schema.latent_dim)
        actual = restored(context_x, context_y, query_x, class_count=3)
        torch.testing.assert_close(actual.particle_logits, expected.particle_logits, rtol=0, atol=0)
        assert actual.structural is not None and expected.structural is not None
        torch.testing.assert_close(actual.structural, expected.structural, rtol=0, atol=0)

    def test_legacy_checkpoints_load_as_deterministic_and_probe_free(self):
        torch.manual_seed(5)
        model = NanoTabPFNTaskPosteriorAdapter(tiny_backbone(), particle_count=4, max_classes=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            save_task_posterior_checkpoint(path, model, training_config={}, lineage={}, data_provenance={})
            checkpoint = torch.load(path)
            for key in ("slot_mode", "slot_sample_seed", "structural_latent_dim", "structural_detach"):
                checkpoint["architecture"].pop(key)
            torch.save(checkpoint, path)
            restored, _ = load_task_posterior_checkpoint(path)
        self.assertEqual(restored.slot_mode, "deterministic")
        self.assertIsNone(restored.structural_probe)


if __name__ == "__main__":
    unittest.main()
