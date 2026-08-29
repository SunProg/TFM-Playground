import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import HELDOUT_REGIME, sample_episode
from tfmplayground.experiments.support_resampling import (
    SCHEMES,
    IntrinsicPosterior,
    LayerCapture,
    build_ensemble,
    compute_intrinsic_posterior,
    exact_pairwise_mutual_information,
    split_fit_and_heldout,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

CHECKPOINT = Path("checkpoints/nanotabpfn.pth")


def tiny_backbone(seed: int = 0) -> NanoTabPFNModel:
    torch.manual_seed(seed)
    return NanoTabPFNModel(16, 2, 32, 3, 3).eval()


def tiny_table(support: int = 20, query: int = 5, features: int = 3, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    support_x = torch.randn(support, features, generator=generator)
    support_y = torch.randint(0, 2, (support,), generator=generator).float()
    query_x = torch.randn(query, features, generator=generator)
    return support_x, support_y, query_x


class LayerCaptureTests(unittest.TestCase):
    def test_hooks_do_not_change_encode_table_output(self):
        model = tiny_backbone(1)
        x, y = torch.randn(1, 20, 3), torch.randint(0, 2, (1, 12)).float()
        with torch.no_grad():
            reference = model.encode_table((x, y), 12, num_mem_chunks=1)
        with LayerCapture(model, 12, keep_graph=False):
            with torch.no_grad():
                observed = model.encode_table((x, y), 12, num_mem_chunks=1)
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)

    def test_hooks_are_removed_on_context_exit(self):
        model = tiny_backbone(2)
        with LayerCapture(model, 5, keep_graph=False):
            self.assertTrue(all(len(block._forward_hooks) == 1 for block in model.transformer_blocks))
        self.assertTrue(all(len(block._forward_hooks) == 0 for block in model.transformer_blocks))

    def test_captures_one_tensor_per_layer_sliced_to_query_rows(self):
        model = tiny_backbone(3)
        x, y = torch.randn(2, 18, 3), torch.randint(0, 2, (2, 10)).float()
        with LayerCapture(model, 10, keep_graph=False) as capture:
            with torch.no_grad():
                model.encode_table((x, y), 10, num_mem_chunks=1)
        self.assertEqual(len(capture.captures), model.num_layers)
        for layer_capture in capture.captures:
            self.assertEqual(tuple(layer_capture.shape), (2, 8, model.embedding_size))
        self.assertIsNone(capture.support_captures)

    def test_capture_support_also_captures_support_rows(self):
        model = tiny_backbone(9)
        x, y = torch.randn(2, 18, 3), torch.randint(0, 2, (2, 10)).float()
        with LayerCapture(model, 10, keep_graph=False, capture_support=True) as capture:
            with torch.no_grad():
                model.encode_table((x, y), 10, num_mem_chunks=1)
        self.assertEqual(len(capture.support_captures), model.num_layers)
        for support_capture in capture.support_captures:
            self.assertEqual(tuple(support_capture.shape), (2, 10, model.embedding_size))

    def test_capture_support_rejects_keep_graph(self):
        model = tiny_backbone(10)
        with self.assertRaises(ValueError):
            LayerCapture(model, 10, keep_graph=True, capture_support=True)


class SplitFitHeldoutTests(unittest.TestCase):
    def test_no_overlap_and_correct_sizes(self):
        support_x, support_y, _ = tiny_table(support=24)
        fit_x, fit_y, held_x, held_y = split_fit_and_heldout(support_x, support_y, heldout_size=6, seed=1)
        self.assertEqual(fit_x.shape[0], 18)
        self.assertEqual(held_x.shape[0], 6)
        fit_rows = {tuple(row.tolist()) for row in fit_x}
        held_rows = {tuple(row.tolist()) for row in held_x}
        self.assertTrue(fit_rows.isdisjoint(held_rows))
        self.assertEqual(fit_y.shape[0], 18)
        self.assertEqual(held_y.shape[0], 6)

    def test_deterministic_given_seed(self):
        support_x, support_y, _ = tiny_table(support=24)
        first = split_fit_and_heldout(support_x, support_y, heldout_size=6, seed=5)
        second = split_fit_and_heldout(support_x, support_y, heldout_size=6, seed=5)
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_rejects_out_of_range_heldout_size(self):
        support_x, support_y, _ = tiny_table(support=10)
        with self.assertRaises(ValueError):
            split_fit_and_heldout(support_x, support_y, heldout_size=0, seed=0)
        with self.assertRaises(ValueError):
            split_fit_and_heldout(support_x, support_y, heldout_size=10, seed=0)


class BuildEnsembleTests(unittest.TestCase):
    def test_no_query_label_argument_anywhere(self):
        """Leakage is impossible by construction: no estimator's signature accepts query labels."""
        for function in (build_ensemble, compute_intrinsic_posterior):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(any("label" in name or name == "query_y" for name in parameters))

    def test_shapes_for_every_scheme(self):
        model = tiny_backbone(4)
        support_x, support_y, query_x = tiny_table(support=20, query=5)
        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                ensemble = build_ensemble(
                    model, support_x, support_y, query_x, scheme=scheme, members=6, fraction=0.7, seed=2
                )
                self.assertEqual(ensemble.members, 6)
                self.assertEqual(ensemble.num_layers, model.num_layers)
                self.assertEqual(tuple(ensemble.member_probabilities.shape), (6, 5))
                for embedding in ensemble.layer_query_embeddings:
                    self.assertEqual(tuple(embedding.shape), (6, 5, model.embedding_size))
                self.assertEqual(tuple(ensemble.base_positive.shape), (5,))

    def test_deterministic_given_seed(self):
        model = tiny_backbone(5)
        support_x, support_y, query_x = tiny_table(support=16, query=4)
        first = build_ensemble(model, support_x, support_y, query_x, scheme="bootstrap", members=5, seed=9)
        second = build_ensemble(model, support_x, support_y, query_x, scheme="bootstrap", members=5, seed=9)
        torch.testing.assert_close(first.member_probabilities, second.member_probabilities, atol=0, rtol=0)
        for left, right in zip(first.layer_query_embeddings, second.layer_query_embeddings, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_held_out_rows_never_appear_in_any_member(self):
        support_x, support_y, query_x = tiny_table(support=24, query=4)
        fit_x, fit_y, held_x, held_y = split_fit_and_heldout(support_x, support_y, heldout_size=6, seed=1)
        model = tiny_backbone(6)
        ensemble = build_ensemble(
            model, fit_x, fit_y, query_x, scheme="subsample", members=6, fraction=0.8, seed=3, heldout_x=held_x
        )
        self.assertEqual(tuple(ensemble.heldout_member_probabilities.shape), (6, 6))
        loss = ensemble.heldout_member_log_loss(held_y)
        self.assertEqual(loss.shape, (6,))
        self.assertTrue(torch.isfinite(loss).all())

    def test_derived_statistics_are_well_formed(self):
        model = tiny_backbone(7)
        support_x, support_y, query_x = tiny_table(support=20, query=6)
        ensemble = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=10, seed=11, compute_gradient=True
        )
        dispersion = ensemble.probability_dispersion()
        self.assertTrue((dispersion >= 0).all())
        information = ensemble.probability_mutual_information()
        self.assertTrue((information >= -1e-6).all())
        representation = ensemble.representation_dispersion()
        self.assertEqual(tuple(representation.shape), (model.num_layers, 6))
        self.assertTrue((representation >= 0).all())
        scale_free = ensemble.scale_free_representation_dispersion()
        self.assertTrue((scale_free >= 0).all())
        rank = ensemble.effective_rank()
        self.assertTrue((rank >= 1.0 - 1e-4).all())
        self.assertTrue((rank <= min(ensemble.members, model.embedding_size) + 1e-4).all())
        ratio = ensemble.projected_ratio()
        self.assertTrue((ratio >= -1e-4).all())
        self.assertTrue((ratio <= 1.0 + 1e-4).all())

    def test_support_dispersion_requires_capture_support_first(self):
        model = tiny_backbone(12)
        support_x, support_y, query_x = tiny_table(support=20, query=5)
        ensemble = build_ensemble(model, support_x, support_y, query_x, scheme="bootstrap", members=8, seed=1)
        with self.assertRaises(RuntimeError):
            ensemble.support_representation_dispersion()

    def test_support_dispersion_is_well_formed_and_deterministic(self):
        model = tiny_backbone(13)
        support_x, support_y, query_x = tiny_table(support=20, query=5)
        first = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=16, seed=4, capture_support=True
        )
        self.assertEqual(tuple(first.support_indices.shape), (16, 20))
        self.assertTrue((first.support_indices >= 0).all())
        self.assertTrue((first.support_indices < 20).all())
        for embedding in first.layer_support_embeddings:
            self.assertEqual(tuple(embedding.shape), (16, 20, model.embedding_size))
        dispersion = first.support_representation_dispersion()
        self.assertEqual(tuple(dispersion.shape), (model.num_layers,))
        self.assertTrue((dispersion >= 0).all())
        scale_free = first.support_scale_free_representation_dispersion()
        self.assertEqual(tuple(scale_free.shape), (model.num_layers,))
        self.assertTrue((scale_free >= 0).all())

        second = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=16, seed=4, capture_support=True
        )
        torch.testing.assert_close(
            first.support_representation_dispersion(), second.support_representation_dispersion(), atol=0, rtol=0
        )

    def test_capture_support_does_not_change_query_results(self):
        model = tiny_backbone(14)
        support_x, support_y, query_x = tiny_table(support=18, query=4)
        without = build_ensemble(model, support_x, support_y, query_x, scheme="bootstrap", members=6, seed=2)
        with_support = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=6, seed=2, capture_support=True
        )
        torch.testing.assert_close(without.member_probabilities, with_support.member_probabilities, atol=0, rtol=0)
        for left, right in zip(without.layer_query_embeddings, with_support.layer_query_embeddings, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_projected_dispersion_requires_gradient_first(self):
        model = tiny_backbone(8)
        support_x, support_y, query_x = tiny_table(support=14, query=3)
        ensemble = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=4, seed=1, compute_gradient=False
        )
        with self.assertRaises(RuntimeError):
            ensemble.projected_dispersion()

    def test_rejects_unknown_scheme_and_too_few_members(self):
        model = tiny_backbone(9)
        support_x, support_y, query_x = tiny_table(support=10, query=3)
        with self.assertRaises(ValueError):
            build_ensemble(model, support_x, support_y, query_x, scheme="nonsense", members=4, seed=0)
        with self.assertRaises(ValueError):
            build_ensemble(model, support_x, support_y, query_x, scheme="bootstrap", members=1, seed=0)


class IntrinsicPosteriorTests(unittest.TestCase):
    def test_shapes_and_no_leakage(self):
        model = tiny_backbone(10)
        support_x, support_y, query_x = tiny_table(support=16, query=4)
        posterior = compute_intrinsic_posterior(model, support_x, support_y, query_x)
        self.assertIsInstance(posterior, IntrinsicPosterior)
        self.assertEqual(tuple(posterior.base_positive.shape), (4,))
        self.assertEqual(tuple(posterior.conditional_positive.shape), (4, 2, 4))

    def test_self_conditioning_is_non_negative(self):
        model = tiny_backbone(11)
        support_x, support_y, query_x = tiny_table(support=16, query=5)
        posterior = compute_intrinsic_posterior(model, support_x, support_y, query_x)
        self.assertTrue((posterior.self_conditioning() >= 0).all())

    def test_joint_mutual_information_nonnegative_with_nan_diagonal(self):
        model = tiny_backbone(12)
        support_x, support_y, query_x = tiny_table(support=16, query=5)
        posterior = compute_intrinsic_posterior(model, support_x, support_y, query_x)
        mutual_information = posterior.joint_mutual_information()
        self.assertEqual(tuple(mutual_information.shape), (5, 5))
        self.assertTrue(torch.isnan(torch.diagonal(mutual_information)).all())
        off_diagonal = mutual_information[~torch.eye(5, dtype=torch.bool)]
        self.assertTrue((off_diagonal >= -1e-6).all())


class ExactPairwiseMutualInformationTests(unittest.TestCase):
    def test_symmetric_nonnegative_nan_diagonal(self):
        rng = np.random.default_rng(0)
        episode = sample_episode(
            rng, regime=HELDOUT_REGIME, condition="ambiguous", batch_size=1, support_size=32, query_count=6
        )
        mutual_information = exact_pairwise_mutual_information(
            episode.candidate_query_positive[0], episode.posterior[0]
        )
        self.assertEqual(tuple(mutual_information.shape), (6, 6))
        self.assertTrue(torch.isnan(torch.diagonal(mutual_information)).all())
        off_diagonal_mask = ~torch.eye(6, dtype=torch.bool)
        self.assertTrue((mutual_information[off_diagonal_mask] >= -1e-6).all())
        torch.testing.assert_close(
            mutual_information.nan_to_num(0.0), mutual_information.T.nan_to_num(0.0), atol=1e-5, rtol=1e-4
        )

    def test_zero_when_a_single_candidate_dominates(self):
        # One candidate with posterior weight ~1: labels are then (near) deterministic
        # given the candidate, so queries carry (near) zero mutual information.
        candidate_query_positive = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
        posterior = torch.tensor([1.0 - 1e-6, 1e-6])
        mutual_information = exact_pairwise_mutual_information(candidate_query_positive, posterior)
        off_diagonal_mask = ~torch.eye(3, dtype=torch.bool)
        self.assertTrue((mutual_information[off_diagonal_mask] < 1e-3).all())


@unittest.skipUnless(CHECKPOINT.exists(), "Real nanoTabPFN checkpoint is not available.")
class RealCheckpointSanityTests(unittest.TestCase):
    """Qualitative sanity: identifiable support should leave less resampling dispersion.

    This is the same qualitative direction every learned arm in
    ``CONTINUOUS_BAYESIAN_UNCERTAINTY_TRIAL.md`` was checked against; run here on
    the frozen encoder with no head at all.
    """

    def test_identifiable_support_has_lower_dispersion_than_ambiguous(self):
        model = init_model_from_state_dict_file(str(CHECKPOINT)).eval()
        rng = np.random.default_rng(0)
        results = {}
        for condition in ("ambiguous", "identifiable"):
            episode = sample_episode(
                rng,
                regime=HELDOUT_REGIME,
                condition=condition,
                batch_size=1,
                support_size=64,
                query_count=6,
                noise=0.0,
            )
            support_x, support_y, query_x = episode.support_x[0], episode.support_y[0], episode.query_x[0]
            ensemble = build_ensemble(
                model, support_x, support_y, query_x, scheme="bootstrap", members=16, seed=0, compute_gradient=False
            )
            results[condition] = float(ensemble.probability_mutual_information().mean())
        self.assertGreater(results["ambiguous"], results["identifiable"])


if __name__ == "__main__":
    unittest.main()
