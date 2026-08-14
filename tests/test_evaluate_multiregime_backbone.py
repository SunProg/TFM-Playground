import unittest

import numpy as np
import torch

from tfmplayground.experiments.continuous_episodes import HELDOUT_REGIME, TRAIN_REGIME
from tfmplayground.experiments.evaluate_multiregime_backbone import (
    SOURCE_NAMES,
    EvaluationConfig,
    _make_source,
    bootstrap_auc_acc,
    evaluate_cell,
    load_model,
    predict_positive,
)
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def tiny_backbone(seed: int = 0) -> NanoTabPFNModel:
    torch.manual_seed(seed)
    return NanoTabPFNModel(16, 2, 32, 2, 3)


class SourceDispatchTests(unittest.TestCase):
    def test_every_declared_source_is_constructible(self):
        for name in SOURCE_NAMES:
            source = _make_source(name)
            episode = source(
                np.random.default_rng(0),
                regime=TRAIN_REGIME,
                batch_size=1,
                support_size=16,
                query_count=4,
                contamination=0.3,
                device="cpu",
            )
            self.assertEqual(episode.condition, "multiregime")
            self.assertEqual(tuple(episode.query_regime_source.shape), (1, 4))

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            _make_source("not+a+real+source")

    def test_mlp_scm_source_ignores_caller_regime_and_uses_train(self):
        source = _make_source("mlp_scm")
        episode = source(
            np.random.default_rng(1),
            regime=HELDOUT_REGIME,
            batch_size=1,
            support_size=16,
            query_count=4,
            contamination=0.2,
            device="cpu",
        )
        self.assertEqual(episode.family, "mlp_scm")

    def test_tree_scm_source_ignores_caller_regime_and_uses_heldout(self):
        source = _make_source("tree_scm")
        episode = source(
            np.random.default_rng(1),
            regime=TRAIN_REGIME,
            batch_size=1,
            support_size=16,
            query_count=4,
            contamination=0.2,
            device="cpu",
        )
        self.assertEqual(episode.family, "tree_scm")


class BootstrapTests(unittest.TestCase):
    def test_returns_none_for_degenerate_or_tiny_inputs(self):
        result = bootstrap_auc_acc(np.array([0.6]), np.array([1]))
        self.assertIsNone(result["auc"])
        result = bootstrap_auc_acc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 0]))
        self.assertIsNone(result["auc"])

    def test_returns_finite_ci_for_a_real_split(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1] * 20)
        p = np.clip(y + rng.normal(scale=0.3, size=40), 0, 1)
        result = bootstrap_auc_acc(p, y, iterations=200)
        self.assertIsNotNone(result["auc"])
        self.assertTrue(result["auc"]["lower"] <= result["auc"]["mean"] <= result["auc"]["upper"])


class PredictAndEvaluateCellTests(unittest.TestCase):
    def test_predict_positive_shape_and_range(self):
        model = tiny_backbone(2)
        source = _make_source("mlp_scm")
        episode = source(
            np.random.default_rng(3),
            regime=TRAIN_REGIME,
            batch_size=1,
            support_size=16,
            query_count=6,
            contamination=0.3,
            device="cpu",
        )
        positive = predict_positive(model, episode)
        self.assertEqual(positive.shape, (1, 6))
        self.assertTrue(bool((positive >= 0).all() and (positive <= 1).all()))

    def test_evaluate_cell_reports_base_and_other_with_correct_counts(self):
        model = tiny_backbone(4)
        source = _make_source("mlp_scm")
        config = EvaluationConfig(query_count=8, support_size=16, episodes_per_cell=3, device="cpu")
        cell = evaluate_cell(model, source, TRAIN_REGIME, 0.3, config, seed=0)
        self.assertEqual(cell["n_base"] + cell["n_other"], config.query_count * config.episodes_per_cell)
        self.assertIn("base", cell)
        self.assertIn("other", cell)

    def test_evaluate_cell_zero_contamination_has_no_other_rows(self):
        model = tiny_backbone(5)
        source = _make_source("mlp_scm")
        config = EvaluationConfig(query_count=8, support_size=16, episodes_per_cell=2, device="cpu")
        cell = evaluate_cell(model, source, TRAIN_REGIME, 0.0, config, seed=0)
        self.assertEqual(cell["n_other"], 0)
        self.assertIsNone(cell["other"]["auc"])


class LoadModelTests(unittest.TestCase):
    def test_loads_plain_checkpoint_and_finetuned_checkpoint_identically(self):
        reference = init_model_from_state_dict_file("checkpoints/nanotabpfn.pth")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            finetuned_path = Path(tmp) / "backbone.pth"
            torch.save({"model": reference.state_dict(), "config": {}}, finetuned_path)

            baseline = load_model("checkpoints/nanotabpfn.pth", "cpu")
            finetuned = load_model(str(finetuned_path), "cpu")

            support_x, support_y = torch.randn(1, 10, 3), torch.randint(0, 2, (1, 10)).float()
            query_x = torch.randn(1, 4, 3)
            with torch.no_grad():
                a = baseline(support_x, support_y, query_x)
                b = finetuned(support_x, support_y, query_x)
            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
