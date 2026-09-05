import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from tfmplayground.experiments import evaluate_plain_nanotabpfn as evaluation
from tfmplayground.experiments.evaluate_tabarena_small import SmallTabArenaConfig, _evaluation_rows
from tfmplayground.experiments.pretrain_plain_nanotabpfn import (
    PlainPretrainingConfig,
    make_prior,
    multiregime_probability,
    run_pretraining,
)
from tfmplayground.external_priors.tabicl import TabICLPriorDataLoader
from tfmplayground.interface import init_model_from_state_dict_file
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


class _FakeTabICLPrior:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __next__(self):
        x = torch.zeros((2, 160, 3), dtype=torch.float32)
        y = torch.arange(160).repeat(2, 1).remainder(2).float()
        return x, y, torch.tensor([3, 3]), torch.tensor([160, 160]), torch.tensor([128, 128])


class _TinyPrior:
    def __init__(self, batches: int, support_size: int, query_size: int):
        self.batches = batches
        self.support_size = support_size
        self.query_size = query_size

    def __iter__(self):
        for _ in range(self.batches):
            rows = self.support_size + self.query_size
            x = torch.randn((1, rows, 2))
            y = torch.arange(rows).remainder(2).unsqueeze(0).float()
            yield {"x": x, "y": y, "target_y": y, "train_test_split_index": self.support_size}


class TabICLPriorLoaderTests(unittest.TestCase):
    def test_support_size_derives_exact_five_fold_dataset_sizes(self):
        self.assertEqual(_evaluation_rows(SmallTabArenaConfig(folds=5, support_size=512)), 640)
        self.assertEqual(_evaluation_rows(SmallTabArenaConfig(folds=5, support_size=1024)), 1280)
        self.assertEqual(_evaluation_rows(SmallTabArenaConfig(folds=5, support_size=2048)), 2560)

    @patch("tfmplayground.external_priors.tabicl.TabICLPriorDataset", _FakeTabICLPrior)
    def test_exact_integer_support_split_is_forwarded_and_preserved(self):
        loader = TabICLPriorDataLoader(
            num_steps=1,
            batch_size=2,
            num_datapoints_min=160,
            num_datapoints_max=161,
            min_features=2,
            max_features=12,
            max_num_classes=2,
            device=torch.device("cpu"),
            min_train_size=128,
            max_train_size=129,
        )
        self.assertEqual(loader.pd.kwargs["min_train_size"], 128)
        self.assertEqual(loader.pd.kwargs["max_train_size"], 129)
        batch = next(iter(loader))
        self.assertEqual(batch["x"].shape, (2, 160, 3))
        self.assertEqual(batch["train_test_split_index"], 128)
        self.assertEqual(set(batch["y"].unique().tolist()), {0.0, 1.0})

    @patch("tfmplayground.experiments.pretrain_plain_nanotabpfn.TabICLPriorDataLoader")
    def test_runner_prior_uses_matched_table_geometry(self, loader_class):
        config = PlainPretrainingConfig(device="cpu")
        make_prior(config, batches=1)
        kwargs = loader_class.call_args.kwargs
        self.assertEqual((kwargs["num_datapoints_min"], kwargs["num_datapoints_max"]), (160, 161))
        self.assertEqual((kwargs["min_train_size"], kwargs["max_train_size"]), (128, 129))
        self.assertEqual((kwargs["min_features"], kwargs["max_features"]), (2, 12))


class _FlakyPrior:
    """Shared draw source: yields a non-finite batch every ``bad_every`` draws."""

    def __init__(self, support_size: int, query_size: int, *, bad_every: int = 0):
        self.support_size = support_size
        self.query_size = query_size
        self.bad_every = bad_every
        self.draws = 0

    def draw(self):
        self.draws += 1
        rows = self.support_size + self.query_size
        x = torch.randn((1, rows, 2))
        if self.bad_every and self.draws % self.bad_every == 0:
            x[0, 0, 0] = float("inf")
        y = torch.arange(rows).remainder(2).unsqueeze(0).float()
        return {"x": x, "y": y, "target_y": y, "train_test_split_index": self.support_size}


class _BoundedFlakyDraws:
    """A single-use, ``batches``-bounded view over a shared ``_FlakyPrior`` source.

    Mirrors ``_TinyPrior``'s contract: ``make_prior`` must return something that
    stops after the requested number of batches, since ``validate()`` consumes it
    with a plain ``for`` loop.
    """

    def __init__(self, source: _FlakyPrior, batches: int):
        self.source = source
        self.batches = batches

    def __iter__(self):
        for _ in range(self.batches):
            yield self.source.draw()


class PretrainingSmokeTests(unittest.TestCase):
    def test_curriculum_probability_has_plain_ramp_and_multiregime_phases(self):
        config = PlainPretrainingConfig(
            prior_mode="curriculum",
            max_steps=100,
        )
        self.assertEqual(multiregime_probability(config, 1), 0.0)
        self.assertEqual(multiregime_probability(config, 10), 0.0)
        self.assertAlmostEqual(multiregime_probability(config, 30), 0.25)
        self.assertEqual(multiregime_probability(config, 50), 0.5)
        self.assertEqual(multiregime_probability(config, 99), 0.5)

    def test_tiny_cpu_run_writes_resumable_inference_checkpoint(self):
        config = PlainPretrainingConfig(
            device="cpu",
            max_steps=2,
            micro_batch_size=1,
            accumulate_gradients=1,
            warmup_steps=1,
            validation_interval=2,
            validation_batches=1,
            checkpoint_interval=2,
            support_size=4,
            query_size=2,
            min_features=2,
            max_features=2,
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=1,
            tensorboard=False,
        )

        def fake_prior(config, *, batches, device=None):
            del device
            return _TinyPrior(batches, config.support_size, config.query_size)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "tfmplayground.experiments.pretrain_plain_nanotabpfn.make_prior", fake_prior
        ):
            output = run_pretraining(config, Path(temporary) / "run")
            self.assertTrue((output / "checkpoint-000002.pth").exists())
            resumed = replace(config, max_steps=3)
            output = run_pretraining(
                resumed,
                output,
                resume_checkpoint=output / "checkpoint-000002.pth",
            )
            final = output / "final_checkpoint.pth"
            self.assertTrue(final.exists())
            self.assertEqual(init_model_from_state_dict_file(final).num_layers, 1)
            history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in history], [1, 2, 3])

    def test_tabarena_epoch_checkpoints_are_pruned_except_resumable_milestones(self):
        config = self._tiny_config(
            max_steps=2,
            epoch_steps=1,
            checkpoint_interval=2,
            tabarena_every_epoch=True,
        )

        def fake_prior(config, *, batches, device=None):
            del device
            return _TinyPrior(batches, config.support_size, config.query_size)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "tfmplayground.experiments.pretrain_plain_nanotabpfn.make_prior", fake_prior
        ), patch(
            "tfmplayground.experiments.pretrain_plain_nanotabpfn.evaluate_tabarena_epoch",
            return_value={"tabarena_mean_roc_auc": 0.5, "tabarena_mean_accuracy": 0.5},
        ):
            output = run_pretraining(config, Path(temporary) / "run")
            self.assertFalse((output / "epoch-001-checkpoint.pth").exists())
            self.assertTrue((output / "epoch-002-checkpoint.pth").exists())
            self.assertTrue((output / "checkpoint-000002.pth").exists())

    def _tiny_config(self, **overrides) -> PlainPretrainingConfig:
        defaults = dict(
            device="cpu",
            max_steps=2,
            micro_batch_size=1,
            accumulate_gradients=1,
            warmup_steps=1,
            validation_interval=2,
            validation_batches=1,
            checkpoint_interval=2,
            support_size=4,
            query_size=2,
            min_features=2,
            max_features=2,
            embedding_size=8,
            num_attention_heads=2,
            mlp_hidden_size=16,
            num_layers=1,
            tensorboard=False,
        )
        defaults.update(overrides)
        return PlainPretrainingConfig(**defaults)

    def test_occasional_non_finite_batch_is_retried_not_fatal(self):
        config = self._tiny_config(max_steps=3)
        flaky = _FlakyPrior(config.support_size, config.query_size, bad_every=2)

        def fake_prior(config, *, batches, device=None):
            del config, device
            return _BoundedFlakyDraws(flaky, batches)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "tfmplayground.experiments.pretrain_plain_nanotabpfn.make_prior", fake_prior
        ):
            output = run_pretraining(config, Path(temporary) / "run")
            history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in history], [1, 2, 3])
            self.assertTrue(all(math.isfinite(row["query_cross_entropy"]) for row in history))
            validation_losses = [
                row["validation_query_cross_entropy"]
                for row in history
                if "validation_query_cross_entropy" in row
            ]
            self.assertTrue(all(math.isfinite(value) for value in validation_losses))

    def test_persistently_non_finite_batches_still_raise_after_retry_budget(self):
        config = self._tiny_config(max_steps=1)
        flaky = _FlakyPrior(config.support_size, config.query_size, bad_every=1)

        def fake_prior(config, *, batches, device=None):
            del config, device
            return _BoundedFlakyDraws(flaky, batches)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("tfmplayground.experiments.pretrain_plain_nanotabpfn.make_prior", fake_prior),
            self.assertRaisesRegex(RuntimeError, "Could not draw a finite training batch"),
        ):
            run_pretraining(config, Path(temporary) / "run")


class LockedEvaluationTests(unittest.TestCase):
    def test_locked_evaluator_reuses_one_episode_set_for_every_model(self):
        torch.manual_seed(7)
        model = NanoTabPFNModel(8, 2, 16, 1, 2)
        episode = evaluation.EvaluationEpisode(
            "locked-0000",
            "ordinary_mix_scm",
            None,
            torch.randn(4, 2),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            torch.randn(2, 2),
            torch.tensor([0, 1]),
            None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            checkpoint = temporary_path / "model.pth"
            torch.save(
                {
                    "architecture": {
                        "embedding_size": 8,
                        "num_attention_heads": 2,
                        "mlp_hidden_size": 16,
                        "num_layers": 1,
                        "num_outputs": 2,
                    },
                    "model": model.state_dict(),
                },
                checkpoint,
            )
            with patch(
                "tfmplayground.experiments.evaluate_plain_nanotabpfn.generate_evaluation_episodes",
                return_value=[episode],
            ):
                output = evaluation.evaluate_models(
                    {"first": checkpoint, "second": checkpoint},
                    evaluation.EvaluationConfig(device="cpu", ordinary_episodes=1, multiregime_episodes=1),
                    temporary_path / "evaluation",
                )
            manifest = [json.loads(line) for line in (output / "episode_manifest.jsonl").read_text().splitlines()]
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([entry["episode_id"] for entry in manifest], ["locked-0000"])
            self.assertEqual({row["episode_id"] for row in predictions}, {"locked-0000"})
            self.assertNotIn("other", {row["group"] for row in json.loads((output / "summary.json").read_text())})
            self.assertEqual(predictions[0]["probability"], predictions[2]["probability"])


if __name__ == "__main__":
    unittest.main()


def test_rng_state_restores_from_a_checkpoint_loaded_onto_another_device():
    """Resume must survive ``map_location='cuda'`` moving the RNG state.

    ``torch.set_rng_state`` takes only a CPU ByteTensor, and resume loads the
    whole checkpoint onto the training device, so the saved state arrives as a
    GPU tensor and the run died on its first line.  A dtype change stands in
    for the device change here, since CI has no GPU.
    """
    import torch

    from tfmplayground.experiments.pretrain_plain_nanotabpfn import _restore_rng_state, _serializable_rng_state

    torch.manual_seed(0)
    saved = _serializable_rng_state()
    expected = torch.rand(4)
    # Whatever the checkpoint round trip did to it, restoring must reproduce
    # exactly the draw the original state would have produced.
    saved["torch"] = saved["torch"].to(torch.int64)
    torch.manual_seed(12345)
    _restore_rng_state(saved)
    torch.testing.assert_close(torch.rand(4), expected)
