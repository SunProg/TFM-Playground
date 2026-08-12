import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tfmplayground.experiments.environment_adaptation import (
    GroupedClassificationData,
    TemporalClassificationData,
    beyondarena_from_container,
    evaluate_grouped_few_shot,
    evaluate_temporal_delayed,
    evaluate_temporal_few_shot,
    tune_on_chronological_training_windows,
)
from tfmplayground.experiments.particle_benchmark import (
    RefitContextClassifier,
    RegimeStreamSpec,
    assert_exact_causality,
    evaluate_delayed_stream,
    generate_regime_stream,
    paired_bootstrap_improvement,
    real_data_promotion,
)


class RecordingClassifier(RefitContextClassifier):
    def __init__(self):
        super().__init__(window=32)
        self.events = []

    def predict_proba(self, x, *, regime_hint=None):
        self.events.append(("predict", len(x), self.y.size))
        return super().predict_proba(x, regime_hint=regime_hint)

    def update(self, x, y, *, regime=None):
        self.events.append(("update", len(y), self.y.size))
        return super().update(x, y, regime=regime)


class ParticleBenchmarkTests(unittest.TestCase):
    def test_generator_evaluates_regimes_on_same_rows(self):
        spec = RegimeStreamSpec(pattern=(0, 1, 0), dwell_lengths=(7, 11, 5), n_features=4, missing_rate=0.1)
        stream = generate_regime_stream(spec, 10)
        expected = stream.candidate_y[stream.regime, np.arange(len(stream.y))]
        np.testing.assert_array_equal(stream.y, expected)
        self.assertEqual(stream.segment_starts, (0, 7, 18))

    def test_delayed_loop_predicts_before_revealing_each_batch(self):
        stream = generate_regime_stream(RegimeStreamSpec(pattern=(0, 1), dwell_lengths=(9, 9), n_features=2), 4)
        predictor = RecordingClassifier()
        result = evaluate_delayed_stream(stream, {"model": predictor}, batch_size=4)
        event_names = [event[0] for event in predictor.events]
        self.assertEqual(event_names, ["predict", "update"] * 6)
        for predict, update in zip(predictor.events[::2], predictor.events[1::2], strict=True):
            self.assertEqual(predict[2], update[2])
        self.assertEqual(len(result.probabilities["model"]), 18)
        assert_exact_causality(RecordingClassifier, stream, batch_size=4)

    def test_batches_never_cross_switches(self):
        stream = generate_regime_stream(RegimeStreamSpec(pattern=(0, 1, 0), dwell_lengths=(3, 5, 2), n_features=1), 8)
        result = evaluate_delayed_stream(stream, {"model": RecordingClassifier()}, batch_size=4)
        self.assertEqual([(row.start, row.stop) for row in result.batches], [(0, 3), (3, 7), (7, 8), (8, 10)])

    def test_promotion_gates(self):
        synthetic = paired_bootstrap_improvement([0.4, 0.5, 0.45], [0.7, 0.8, 0.75], samples=1000)
        self.assertTrue(synthetic["significant"])
        real = real_data_promotion({"temporal": [0.01, 0.02], "grouped": [0.03, 0.01]})
        self.assertTrue(real["passed"])
        failed = real_data_promotion({"temporal": [0.01], "grouped": [-0.021]})
        self.assertEqual(failed["classification"], "mechanistic_result_only")


class EnvironmentAdaptationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(100, 3)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        self.temporal = TemporalClassificationData("time", x[:60], y[:60], x[60:], y[60:])
        groups = np.repeat(np.arange(5), 20)
        self.grouped = GroupedClassificationData("groups", x, y, groups, (3, 4))

    def test_tuning_and_temporal_protocols(self):
        name, scores = tune_on_chronological_training_windows(
            self.temporal, {"a": RecordingClassifier, "b": RecordingClassifier}, windows=2
        )
        self.assertIn(name, scores)
        delayed = evaluate_temporal_delayed(self.temporal, {"model": RecordingClassifier()}, batch_size=7)
        self.assertEqual(len(delayed.probabilities["model"]), 40)
        few_shot = evaluate_temporal_few_shot(self.temporal, RecordingClassifier, shots=(0, 8, 32, 128))
        self.assertEqual([row["shots"] for row in few_shot], [0, 8, 32])

    def test_group_order_is_seeded_and_reproducible(self):
        first = evaluate_grouped_few_shot(self.grouped, RecordingClassifier, shots=(0, 8), seed=17)
        second = evaluate_grouped_few_shot(self.grouped, RecordingClassifier, shots=(0, 8), seed=17)
        self.assertEqual(first, second)

    def test_data_foundry_identity_and_canonical_split_are_preserved(self):
        frame = pd.DataFrame(
            {
                "value": np.arange(120, dtype=float),
                "category": ["a", "b"] * 60,
                "group": np.repeat(["train", "g1", "g2"], 40),
                "target": [0, 1] * 60,
            }
        )
        container = SimpleNamespace(
            dataset=frame,
            task_metadata=SimpleNamespace(target_column_name="target", group_on="group"),
            experiment_metadata=SimpleNamespace(splits={0: {0: (np.arange(40), np.arange(40, 120))}}),
            uuid="fixed-uuid",
            checksum="fixed-checksum",
        )
        sliced = beyondarena_from_container(container, "musk")
        self.assertEqual((sliced.dataset_uuid, sliced.checksum), ("fixed-uuid", "fixed-checksum"))
        self.assertEqual(sliced.repeat, 0)
        self.assertEqual(sliced.fold, 0)
        self.assertEqual(set(sliced.data.held_out_groups), {"g1", "g2"})


if __name__ == "__main__":
    unittest.main()
