import unittest
from pathlib import Path

import torch

from tfmplayground.experiments.layer_embedding_variance import (
    LayerVarianceConfig,
    _layer_rows,
    _write_report,
    run_layer_variance_report,
    synthetic_layer_variance,
)
from tfmplayground.experiments.support_resampling import build_ensemble
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

CHECKPOINT = Path("checkpoints/nanotabpfn.pth")


def tiny_backbone(seed: int = 0) -> NanoTabPFNModel:
    torch.manual_seed(seed)
    return NanoTabPFNModel(16, 2, 32, 3, 3).eval()


def tiny_ensemble(seed: int = 0, members: int = 8):
    model = tiny_backbone(seed)
    support_x = torch.randn(20, 3)
    support_y = torch.randint(0, 2, (20,)).float()
    query_x = torch.randn(5, 3)
    ensemble = build_ensemble(
        model,
        support_x,
        support_y,
        query_x,
        scheme="bootstrap",
        members=members,
        seed=seed,
        compute_gradient=False,
        capture_support=True,
    )
    return model, ensemble


class LayerRowsTests(unittest.TestCase):
    def test_one_row_per_layer_non_negative(self):
        model, ensemble = tiny_ensemble(1)
        rows = _layer_rows(ensemble, source="unit", dataset="toy")
        self.assertEqual(len(rows), model.num_layers)
        self.assertEqual([row["layer_index"] for row in rows], list(range(model.num_layers)))
        for row in rows:
            self.assertEqual(row["source"], "unit")
            self.assertEqual(row["dataset"], "toy")
            self.assertGreaterEqual(row["raw_variance"], 0.0)
            self.assertGreaterEqual(row["scale_free_variance"], 0.0)
            self.assertGreaterEqual(row["effective_rank"], 0.0)
            self.assertGreaterEqual(row["support_raw_variance"], 0.0)
            self.assertGreaterEqual(row["support_scale_free_variance"], 0.0)

    def test_support_columns_are_none_without_capture_support(self):
        model = tiny_backbone(5)
        support_x = torch.randn(20, 3)
        support_y = torch.randint(0, 2, (20,)).float()
        query_x = torch.randn(5, 3)
        ensemble = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=8, seed=5, compute_gradient=False
        )
        rows = _layer_rows(ensemble, source="unit", dataset="toy")
        for row in rows:
            self.assertIsNone(row["support_raw_variance"])
            self.assertIsNone(row["support_scale_free_variance"])

    def test_layer_fraction_spans_zero_to_one(self):
        _, ensemble = tiny_ensemble(2)
        rows = _layer_rows(ensemble, source="unit", dataset="toy")
        self.assertEqual(rows[0]["layer_fraction"], 0.0)
        self.assertEqual(rows[-1]["layer_fraction"], 1.0)

    def test_single_layer_model_gets_fraction_zero(self):
        model = NanoTabPFNModel(16, 2, 32, 1, 3).eval()
        support_x = torch.randn(20, 3)
        support_y = torch.randint(0, 2, (20,)).float()
        query_x = torch.randn(5, 3)
        ensemble = build_ensemble(
            model, support_x, support_y, query_x, scheme="bootstrap", members=4, seed=0, compute_gradient=False
        )
        rows = _layer_rows(ensemble, source="unit", dataset="toy")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["layer_fraction"], 0.0)


class SyntheticLayerVarianceTests(unittest.TestCase):
    def test_covers_every_condition_and_episode(self):
        model = tiny_backbone(3)
        config = LayerVarianceConfig(members=4, episodes_per_condition=2, support_size=16, query_count=4, device="cpu")
        rows = synthetic_layer_variance(model, config)
        self.assertEqual(len(rows), 3 * config.episodes_per_condition * model.num_layers)
        conditions = {row["dataset"].split("[")[0] for row in rows}
        self.assertEqual(conditions, {"ambiguous", "identifiable", "noisy"})
        self.assertTrue(all(row["source"] == "synthetic" for row in rows))


class WriteReportTests(unittest.TestCase):
    def test_writes_csv_and_json_with_expected_columns(self):
        import tempfile

        rows = [
            {
                "source": "synthetic",
                "dataset": "ambiguous[0]",
                "layer_index": layer,
                "layer_fraction": layer / 2,
                "raw_variance": 0.1 * (layer + 1),
                "scale_free_variance": 0.01 * (layer + 1),
                "effective_rank": float(layer + 1),
                "support_raw_variance": 0.2 * (layer + 1),
                "support_scale_free_variance": 0.02 * (layer + 1),
            }
            for layer in range(3)
        ]
        config = LayerVarianceConfig(episodes_per_condition=1)
        with tempfile.TemporaryDirectory() as tmp:
            destination = _write_report(rows, config, task_ids=(), output_dir=tmp)
            csv_path = destination / "layer_variance.csv"
            json_path = destination / "metrics.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            import json

            import pandas as pd

            frame = pd.read_csv(csv_path)
            self.assertEqual(
                set(frame.columns),
                {
                    "source",
                    "dataset",
                    "layer_index",
                    "layer_fraction",
                    "raw_variance",
                    "scale_free_variance",
                    "effective_rank",
                    "support_raw_variance",
                    "support_scale_free_variance",
                },
            )
            self.assertEqual(len(frame), 3)

            metrics = json.loads(json_path.read_text())
            self.assertEqual(metrics["row_count"], 3)
            self.assertEqual(len(metrics["summary"]), 3)


@unittest.skipUnless(CHECKPOINT.exists(), "Real nanoTabPFN checkpoint is not available.")
class RealCheckpointReportTests(unittest.TestCase):
    def test_run_layer_variance_report_end_to_end_synthetic_only(self):
        import tempfile

        model = init_model_from_state_dict_file(str(CHECKPOINT)).eval()
        config = LayerVarianceConfig(
            checkpoint=str(CHECKPOINT), members=4, episodes_per_condition=1, support_size=32, query_count=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = run_layer_variance_report(config, tmp, task_ids=(), skip_tabarena=True)
            self.assertTrue((destination / "layer_variance.csv").exists())
            self.assertTrue((destination / "metrics.json").exists())

            import pandas as pd

            frame = pd.read_csv(destination / "layer_variance.csv")
            self.assertEqual(len(frame), 3 * config.episodes_per_condition * model.num_layers)
            self.assertTrue((frame["raw_variance"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
