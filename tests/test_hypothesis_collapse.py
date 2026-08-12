import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tfmplayground.experiments.hypothesis_collapse import (
    ExperimentConfig,
    NanoTabPFNBinaryPredictor,
    _posterior_b_from_evidence,
    all_binary_vectors,
    compute_trial_metrics,
    enumerate_chain_joint,
    exact_joint_distribution,
    generate_global_ambiguity_tasks,
    independent_joint_distribution,
    main,
    validate_config,
)
from tfmplayground.experiments.hypothesis_collapse_large_scale import build_parser as build_large_scale_parser
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class QueryValuePredictor:
    """A deterministic stub whose class is encoded by the query feature."""

    def predict_binary_proba(self, support_x, support_y, query_x):
        del support_x, support_y
        class_one = (query_x[..., 0] > 0.5).astype(float)
        return np.stack((1.0 - class_one, class_one), axis=-1)


class HypothesisCollapseTests(unittest.TestCase):
    def test_large_scale_defaults_match_every_support_size_with_evidence(self):
        args = build_large_scale_parser().parse_args([])
        self.assertEqual(args.common_support_sizes, [16, 64, 128])
        self.assertTrue(set(args.common_support_sizes) <= set(args.evidence_counts))

    def test_generator_has_balanced_common_support_and_valid_query_modes(self):
        batch = generate_global_ambiguity_tasks(
            trials=20,
            query_count=4,
            evidence_count=0,
            common_support_size=16,
            support_noise=0.1,
            rng=np.random.default_rng(12),
        )
        self.assertEqual(batch.support_x.shape, (20, 16, 1))
        np.testing.assert_array_equal(batch.support_y.sum(axis=1), np.full(20, 8.0))
        self.assertTrue((batch.support_x < 0).all())
        expected_common_y = (batch.support_x[..., 0] < -1.0).astype(float)
        np.testing.assert_array_equal(batch.support_y, expected_common_y)
        np.testing.assert_allclose(batch.posterior_b, 0.5)
        query_patterns = np.repeat(batch.latent_task[:, None], 4, axis=1)
        valid_patterns = {tuple(pattern) for pattern in query_patterns}
        self.assertTrue(valid_patterns <= {(0, 0, 0, 0), (1, 1, 1, 1)})

    def test_exact_posterior_matches_noisy_evidence_likelihood(self):
        evidence = np.asarray([[1, 1], [0, 1], [0, 0]], dtype=np.int8)
        posterior = _posterior_b_from_evidence(evidence, support_noise=0.1)
        expected = np.asarray([0.81 / 0.82, 0.5, 0.01 / 0.82])
        np.testing.assert_allclose(posterior, expected)
        empty_posterior = _posterior_b_from_evidence(np.zeros((3, 0), dtype=np.int8), support_noise=0.1)
        np.testing.assert_allclose(empty_posterior, 0.5)
        noiseless = _posterior_b_from_evidence(np.asarray([[0, 0], [1, 1]]), support_noise=0.0)
        np.testing.assert_allclose(noiseless, [0.0, 1.0])

    def test_chain_enumeration_normalizes_and_remaps_reverse_order(self):
        support_x = np.asarray([[[-2.0], [-0.5]]], dtype=np.float32)
        support_y = np.asarray([[1.0, 0.0]], dtype=np.float32)
        query_x = np.asarray([[[0.1], [0.8], [1.0]]], dtype=np.float32)
        predictor = QueryValuePredictor()
        canonical = enumerate_chain_joint(predictor, support_x, support_y, query_x, (0, 1, 2))
        reverse = enumerate_chain_joint(predictor, support_x, support_y, query_x, (2, 1, 0))
        expected = np.zeros((1, 8))
        expected[0, 3] = 1.0  # canonical vector 011
        np.testing.assert_allclose(canonical, expected)
        np.testing.assert_allclose(reverse, expected)
        np.testing.assert_allclose(canonical.sum(axis=1), 1.0)

    def test_exact_and_independent_baseline_metrics(self):
        for query_count in (2, 3, 4):
            posterior_b = np.asarray([0.5])
            exact = exact_joint_distribution(posterior_b, query_count)
            independent = independent_joint_distribution(posterior_b, query_count)
            marginals = np.full((1, query_count), 0.5)

            exact_metrics = compute_trial_metrics(
                bayes_joint=exact,
                predicted_joint=exact,
                reverse_joint=exact,
                bayes_marginals=marginals,
                predicted_marginals=marginals,
            )
            self.assertAlmostEqual(float(exact_metrics["joint_js"][0]), 0.0)
            self.assertAlmostEqual(float(exact_metrics["incoherent_mass"][0]), 0.0)
            self.assertAlmostEqual(float(exact_metrics["order_inconsistency"][0]), 0.0)

            independent_metrics = compute_trial_metrics(
                bayes_joint=exact,
                predicted_joint=independent,
                reverse_joint=independent,
                bayes_marginals=marginals,
                predicted_marginals=marginals,
            )
            expected_incoherent = 1.0 - 2 ** (1 - query_count)
            self.assertAlmostEqual(float(independent_metrics["incoherent_mass"][0]), expected_incoherent)
            self.assertAlmostEqual(float(independent_metrics["marginal_js"][0]), 0.0)
            self.assertAlmostEqual(float(independent_metrics["order_inconsistency"][0]), 0.0)

    def test_tiny_nanotabpfn_model_enumerates_finite_joint(self):
        model = NanoTabPFNModel(
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=1,
            num_outputs=3,
        )
        predictor = NanoTabPFNBinaryPredictor(model, device="cpu", num_mem_chunks=2)
        batch = generate_global_ambiguity_tasks(
            trials=2,
            query_count=2,
            evidence_count=1,
            common_support_size=4,
            support_noise=0.1,
            rng=np.random.default_rng(7),
        )
        joint = enumerate_chain_joint(
            predictor,
            batch.support_x,
            batch.support_y,
            batch.query_x,
            (0, 1),
        )
        self.assertEqual(joint.shape, (2, 4))
        self.assertTrue(np.isfinite(joint).all())
        np.testing.assert_allclose(joint.sum(axis=1), 1.0, atol=1e-6)

    def test_config_rejects_duplicates_and_unsupported_queries(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_config(ExperimentConfig(query_counts=(2, 2)))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_config(ExperimentConfig(query_counts=(5,)))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_config(ExperimentConfig(evidence_counts=(0, 0)))
        with self.assertRaisesRegex(ValueError, "support-noise"):
            validate_config(ExperimentConfig(support_noise=0.5))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_config(ExperimentConfig(common_support_sizes=(16, 16)))
        with self.assertRaisesRegex(ValueError, "positive even"):
            validate_config(ExperimentConfig(common_support_sizes=(16, 63)))
        with tempfile.TemporaryDirectory() as existing_directory, self.assertRaises(FileExistsError):
            validate_config(ExperimentConfig(output_dir=existing_directory))

    def test_binary_predictor_rejects_single_output_model(self):
        model = NanoTabPFNModel(
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=1,
            num_outputs=1,
        )
        with self.assertRaisesRegex(ValueError, "at least two outputs"):
            NanoTabPFNBinaryPredictor(model, device="cpu")

    def test_checkpoint_free_cli_writes_consistent_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "baseline-run"
            return_code = main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--trials",
                    "2",
                    "--query-counts",
                    "2",
                    "--evidence-counts",
                    "0",
                    "2",
                    "--models",
                    "exact",
                    "independent",
                    "--no-plots",
                ]
            )
            self.assertEqual(return_code, 0)
            expected_files = {
                "config.json",
                "trial_metrics.csv",
                "joint_probabilities.csv",
                "summary.csv",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)
            metrics = pd.read_csv(output_dir / "trial_metrics.csv")
            joints = pd.read_csv(output_dir / "joint_probabilities.csv")
            self.assertEqual(len(metrics), 2 * 2 * 2)
            grouped_sums = joints.groupby(
                ["trial", "query_count", "evidence_count", "model", "order"]
            ).probability.sum()
            np.testing.assert_allclose(grouped_sums.to_numpy(), 1.0)
            self.assertEqual(all_binary_vectors(2).shape, (4, 2))

    def test_support_size_sweep_is_recorded_in_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "scale-run"
            return_code = main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--trials",
                    "2",
                    "--query-counts",
                    "2",
                    "--evidence-counts",
                    "0",
                    "--common-support-sizes",
                    "4",
                    "8",
                    "--models",
                    "exact",
                    "independent",
                    "--no-plots",
                ]
            )
            self.assertEqual(return_code, 0)
            config = json.loads((output_dir / "config.json").read_text())
            self.assertEqual(config["common_support_sizes"], [4, 8])
            for filename in ("trial_metrics.csv", "joint_probabilities.csv", "summary.csv"):
                data = pd.read_csv(output_dir / filename)
                self.assertEqual(set(data["common_support_size"]), {4, 8})


if __name__ == "__main__":
    unittest.main()
