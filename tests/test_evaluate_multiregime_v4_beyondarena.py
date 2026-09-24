import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import (
    BeyondArenaConfig,
    BeyondArenaTask,
    CheckpointCandidate,
    ModelSelection,
    aggregate_rankings,
    discover_v4_checkpoints,
    feature_bucket,
    fold_metrics,
    inspect_beyondarena_container,
    predict_full_fold,
    prepare_fold_data,
    row_bucket,
    run,
    select_checkpoint,
)


@dataclass
class _DatasetMetadata:
    unique_name: str = "fixture"


@dataclass
class _TaskMetadata:
    target_column_name: str = "target"
    problem_type: str = "binary_classification"
    group_on: str | None = None
    time_on: str | None = None

    @property
    def split_regime(self):
        if self.time_on:
            return "temporal_non_iid"
        if self.group_on:
            return "grouped_non_iid"
        return "iid"


@dataclass
class _ExperimentMetadata:
    splits: dict


@dataclass
class _Container:
    dataset: pd.DataFrame
    dataset_metadata: _DatasetMetadata
    task_metadata: _TaskMetadata
    experiment_metadata: _ExperimentMetadata
    uuid: str = "fixture-uuid"
    checksum: str = "fixture-checksum"


class _RecordingModel(torch.nn.Module):
    def __init__(self, *, output_classes=2, fail_if_context_short=False):
        super().__init__()
        self.output_classes = output_classes
        self.fail_if_context_short = fail_if_context_short
        self.context_lengths = []

    def forward(self, source, train_test_split_index, num_mem_chunks=1):
        del num_mem_chunks
        table_x, support_y = source
        self.context_lengths.append((train_test_split_index, int(support_y.shape[1]), int(table_x.shape[1])))
        if self.fail_if_context_short and train_test_split_index < 3:
            raise RuntimeError("subsampled context was used")
        query_count = table_x.shape[1] - train_test_split_index
        logits = torch.zeros((1, query_count, self.output_classes), dtype=table_x.dtype)
        logits[..., 1] = table_x[:, train_test_split_index:, 0]
        return logits


class EvaluateMultiregimeV4BeyondArenaTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        self.assertEqual(row_bucket(1), "small_rows")
        self.assertEqual(row_bucket(1_000), "small_rows")
        self.assertEqual(row_bucket(1_001), "medium_rows")
        self.assertEqual(row_bucket(5_000), "medium_rows")
        self.assertEqual(row_bucket(5_001), "large_rows")
        self.assertEqual(row_bucket(10_000), "large_rows")
        self.assertIsNone(row_bucket(0))
        self.assertIsNone(row_bucket(10_001))
        self.assertEqual(feature_bucket(10), "small_feat")
        self.assertEqual(feature_bucket(11), "medium_feat")
        self.assertEqual(feature_bucket(20), "medium_feat")
        self.assertEqual(feature_bucket(21), "large_feat")
        self.assertEqual(feature_bucket(30), "large_feat")
        self.assertIsNone(feature_bucket(31))

    def test_filtering_uses_raw_features_and_excludes_text(self):
        frame = pd.DataFrame(
            {
                "numeric": [0.0, 1.0, 2.0, 3.0],
                "group_id": ["a", "a", "b", "b"],
                "target": [0, 1, 0, 1],
            }
        )
        container = _Container(
            frame,
            _DatasetMetadata(),
            _TaskMetadata(group_on="group_id"),
            _ExperimentMetadata({0: {0: ([0, 1], [2, 3])}}),
        )
        inspection = inspect_beyondarena_container(container)
        self.assertIsNotNone(inspection.task)
        self.assertEqual(inspection.task.raw_features, 2)
        self.assertEqual(inspection.task.group_columns, ("group_id",))
        self.assertEqual(inspection.task.regime, "Grouped")

        text_frame = frame.copy()
        text_frame["text"] = ["a long sentence " * 10] * 4
        text_container = _Container(
            text_frame,
            _DatasetMetadata(),
            _TaskMetadata(),
            _ExperimentMetadata({0: {0: ([0, 1], [2, 3])}}),
        )
        self.assertEqual(inspect_beyondarena_container(text_container).row["reason"], "text_features_excluded")

    def test_preprocessing_removes_target_and_group_and_fits_on_train(self):
        train = pd.DataFrame(
            {"feature": [0.0, 1.0], "group_id": ["train-a", "train-b"], "target": [0, 1]}
        )
        test = pd.DataFrame(
            {"feature": [10.0], "group_id": ["unseen-test-group"], "target": [1]}
        )
        train_x, test_x, raw_train, raw_test = prepare_fold_data(
            train, test, target_column="target", group_columns=("group_id",)
        )
        self.assertEqual(raw_train.shape[1], 1)
        self.assertEqual(raw_test.shape[1], 1)
        self.assertEqual(train_x.shape[1], test_x.shape[1])

    def test_binary_and_multiclass_metrics_use_prior_baselines(self):
        binary = fold_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]),
            2,
        )
        self.assertAlmostEqual(binary["accuracy"], 1.0)
        self.assertAlmostEqual(binary["majority_accuracy"], 0.5)
        self.assertAlmostEqual(binary["accuracy_gain"], 0.5)

        multiclass = fold_metrics(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 2]),
            np.eye(3),
            3,
        )
        self.assertAlmostEqual(multiclass["accuracy"], 1.0)
        self.assertAlmostEqual(multiclass["macro_ovr_auc"], 1.0)

    def test_chunking_scores_every_query_with_complete_context(self):
        model = _RecordingModel()
        probabilities = predict_full_fold(
            model,
            np.ones((5, 2), dtype=np.float32),
            np.array([0, 1, 0, 1, 0]),
            np.ones((7, 2), dtype=np.float32),
            classes=2,
            device="cpu",
            query_chunk_size=3,
            num_mem_chunks=2,
        )
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertEqual(model.context_lengths, [(5, 5, 8), (5, 5, 8), (5, 5, 6)])

    def test_checkpoint_selection_is_lowest_stored_validation_ce(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                CheckpointCandidate(root / "checkpoint-000010.pth", root, "m", "canonical", "small", 0.8, 10, {}),
                CheckpointCandidate(root / "checkpoint-000020.pth", root, "m", "canonical", "small", 0.4, 20, {}),
                CheckpointCandidate(root / "final_checkpoint.pth", root, "m", "canonical", "small", 0.6, 30, {}),
            ]
            self.assertEqual(select_checkpoint(candidates, "final").path.name, "final_checkpoint.pth")
            self.assertEqual(select_checkpoint(candidates, "best_own_val").path.name, "checkpoint-000020.pth")

    def test_checkpoint_discovery_requires_explicit_v4_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "canonical"
            run.mkdir()
            common = {
                "model_type": "nanotabpfn_plain_scm_pretraining",
                "architecture": {"embedding_size": 8, "num_layers": 1},
                "training_config": {
                    "multiregime_source": "v4",
                    "model_family": "canonical",
                    "model_size": "small",
                },
            }
            for filename, step, loss in (
                ("checkpoint-000010.pth", 10, 0.8),
                ("checkpoint-000020.pth", 20, 0.4),
                ("final_checkpoint.pth", 30, 0.6),
            ):
                torch.save({**common, "step": step, "validation": {"query_cross_entropy": loss}}, run / filename)
            torch.save(
                {"model_type": "nanotabpfn_plain_scm_pretraining", "architecture": common["architecture"]},
                root / "not-v4.pth",
            )
            selections, audit = discover_v4_checkpoints([root])
            self.assertEqual(len(selections), 2)
            best = next(selection for selection in selections if selection.checkpoint_policy == "best_own_val")
            self.assertEqual(best.checkpoint_path.name, "checkpoint-000020.pth")
            self.assertTrue(any(row["status"] == "rejected" for row in audit))

    def test_checkpoint_discovery_marks_z_expose_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "expose_z" / "r_z-fixed-small" / "seed-2402"
            run.mkdir(parents=True)
            torch.save(
                {
                    "model_type": "nanotabpfn_plain_scm_pretraining",
                    "architecture": {"embedding_size": 8, "num_layers": 1},
                    "training_config": {
                        "multiregime_source": "v4",
                        "v4_dump_path": "/scratch/expose_z/r_z-multiregime.h5",
                    },
                },
                run / "final_checkpoint.pth",
            )
            selections, _ = discover_v4_checkpoints([directory])
            self.assertEqual(len(selections), 2)
            self.assertTrue(all(selection.z_expose for selection in selections))

    def test_unsupported_fold_has_no_subsampled_score(self):
        frame = pd.DataFrame({"feature": range(6), "target": [0, 1, 0, 1, 0, 1]})
        task_container = _Container(
            frame,
            _DatasetMetadata(),
            _TaskMetadata(),
            _ExperimentMetadata({0: {0: ([0, 1, 2, 3], [4, 5])}}),
        )
        task = inspect_beyondarena_container(task_container).task
        selection = ModelSelection(
            "m",
            "canonical",
            "small",
            "final",
            Path("model.pth"),
            "available",
            None,
            None,
            Path("."),
            1,
        )

        class OOMModel(_RecordingModel):
            def forward(self, source, train_test_split_index, num_mem_chunks=1):
                del source, train_test_split_index, num_mem_chunks
                raise MemoryError("no room for complete fold")

        rows = []
        from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import evaluate_selected_models

        rows.extend(
            evaluate_selected_models(
                [task],
                [selection],
                device="cpu",
                query_chunk_size=2,
                num_mem_chunks=1,
                model_loader=lambda path, device: OOMModel(),
            )
        )
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "unsupported" for row in rows))
        self.assertTrue(all("excess_cross_entropy" not in row for row in rows))

    def test_rankings_average_folds_then_tasks(self):
        tasks = []
        for name in ("a", "b"):
            tasks.append(
                BeyondArenaTask(
                    name,
                    name,
                    "checksum",
                    100,
                    5,
                    2,
                    "binary_classification",
                    "IID",
                    "target",
                    (),
                    (),
                    (),
                    object(),
                )
            )
        selections = [
            ModelSelection("m", "canonical", "small", "final", Path("m.pth"), "available", None, None, Path("."), 1),
            ModelSelection("n", "canonical", "small", "final", Path("n.pth"), "available", None, None, Path("."), 1),
        ]
        rows = []
        for task in tasks:
            for fold, score in enumerate((1.0, 3.0)):
                rows.append(
                    {
                        "model_identity": "m",
                        "checkpoint_policy": "final",
                        "task_name": task.name,
                        "condition": "all",
                        "status": "evaluated",
                        "excess_cross_entropy": score if task.name == "a" else 5.0,
                        "macro_ovr_auc": 0.5,
                        "accuracy_gain": 0.0,
                        "fold": fold,
                    }
                )
            for fold, score in enumerate((4.0, 4.0) if task.name == "a" else (1.0, 1.0)):
                rows.append(
                    {
                        "model_identity": "n",
                        "checkpoint_policy": "final",
                        "task_name": task.name,
                        "condition": "all",
                        "status": "evaluated",
                        "excess_cross_entropy": score,
                        "macro_ovr_auc": 0.5,
                        "accuracy_gain": 0.0,
                        "fold": fold,
                    }
                )
        rankings = aggregate_rankings(tasks, selections, rows)
        all_excess = rankings[(rankings.condition == "all") & (rankings.metric == "excess_cross_entropy")]
        model_m = all_excess[all_excess.model_identity == "m"].iloc[0]
        model_n = all_excess[all_excess.model_identity == "n"].iloc[0]
        self.assertAlmostEqual(float(model_m.score), 3.5)
        self.assertEqual(int(model_m.task_count), 2)
        self.assertAlmostEqual(float(model_m["rank"]), 1.5)
        self.assertAlmostEqual(float(model_n["rank"]), 1.5)
        self.assertEqual(int(model_m["rank_task_count"]), 2)

    def test_one_task_smoke_writes_requested_outputs(self):
        frame = pd.DataFrame({"feature": range(6), "target": [0, 1, 0, 1, 0, 1]})
        container = _Container(
            frame,
            _DatasetMetadata(),
            _TaskMetadata(),
            _ExperimentMetadata({0: {0: ([0, 1, 2, 3], [4, 5])}}),
        )

        class _Entry:
            uuid = "fixture-uuid"
            unique_name = "fixture"

        class _Collection:
            entries = (_Entry(),)

            @staticmethod
            def get_dataset(uuid, **kwargs):
                del uuid, kwargs
                return container

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final_checkpoint.pth"
            torch.save(
                {
                    "model_type": "nanotabpfn_plain_scm_pretraining",
                    "training_config": {"multiregime_source": "v4"},
                    "architecture": {"num_outputs": 2},
                },
                checkpoint,
            )
            output = Path(directory) / "out"
            result = run(
                BeyondArenaConfig(
                    run_roots=(directory,),
                    output_dir=str(output),
                    smoke_task="fixture",
                    query_chunk_size=1,
                    num_mem_chunks=1,
                ),
                collection=_Collection(),
                model_loader=lambda path, device: _RecordingModel(),
            )
            self.assertEqual(result, output.resolve())
            for filename in ("task_manifest.csv", "fold_metrics.csv", "rankings.csv", "ranking_summary.md"):
                self.assertTrue((output / filename).is_file(), filename)
            fold = pd.read_csv(output / "fold_metrics.csv")
            self.assertEqual(set(fold[fold.checkpoint_policy == "final"].status), {"evaluated"})
            self.assertEqual(set(fold[fold.checkpoint_policy == "best_own_val"].status), {"unavailable"})


if __name__ == "__main__":
    unittest.main()
