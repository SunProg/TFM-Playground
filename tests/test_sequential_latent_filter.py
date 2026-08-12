import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.train_sequential_latent_filter import (
    CONDITIONS,
    SequentialFilterConfig,
    controlled_gate_report,
    generate_controlled_episodes,
    next_tabicl_episode,
    predict_episode,
    run,
    sequential_filter_loss,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.sequential_latent_filter import (
    NanoTabPFNSequentialLatentFilter,
    SequentialFilterLogits,
    filter_sequential_logits,
    load_sequential_filter_checkpoint,
    save_sequential_filter_checkpoint,
)


def tiny_backbone() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=3,
    )


def tiny_config(**kwargs) -> SequentialFilterConfig:
    return SequentialFilterConfig(
        device="cpu",
        initial_support_count=4,
        stream_count=8,
        query_count=4,
        batch_size=5,
        controlled_steps=0,
        tabicl_steps=0,
        validation_interval=1,
        patience=1,
        evaluation_trials=2,
        ordinary_evaluation_batches=0,
        plots=False,
        **kwargs,
    )


class SequentialLatentFilterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.config = tiny_config()
        self.batch = generate_controlled_episodes(
            self.config, np.random.default_rng(71), condition="noisy", batch_size=2
        )
        self.model = NanoTabPFNSequentialLatentFilter(tiny_backbone())

    def test_label_cannot_change_its_own_pre_update_prediction(self):
        original = predict_episode(self.model.eval(), self.batch)
        changed_batch = copy.copy(self.batch)
        changed_batch.stream_y = self.batch.stream_y.clone()
        changed_batch.stream_y[:, 3] = 1.0 - changed_batch.stream_y[:, 3]
        changed = predict_episode(self.model, changed_batch)

        torch.testing.assert_close(original.stream_logits[:, 3], changed.stream_logits[:, 3], rtol=0, atol=0)
        torch.testing.assert_close(original.log_weights[:, 3], changed.log_weights[:, 3], rtol=0, atol=0)
        self.assertFalse(torch.equal(original.log_weights[:, 4], changed.log_weights[:, 4]))

    def test_weights_are_finite_normalized_and_initially_uniform(self):
        prediction = predict_episode(self.model, self.batch)
        self.assertTrue(torch.isfinite(prediction.log_weights).all())
        torch.testing.assert_close(
            prediction.log_weights.exp().sum(-1),
            torch.ones_like(prediction.log_weights[..., 0]),
        )
        torch.testing.assert_close(
            prediction.log_weights[:, 0].exp(),
            torch.full((2, 2), 0.5),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            prediction.joint_probabilities().sum(-1),
            torch.ones_like(prediction.log_weights[..., 0]),
        )

    def test_disagreement_threshold_skips_shared_rows_without_changing_prequential_score(self):
        stream_logits = torch.tensor([[[[2.00, -2.00], [1.95, -1.95]], [[-3.0, 3.0], [3.0, -3.0]]]])
        raw = SequentialFilterLogits(
            stream_logits=stream_logits,
            query_logits=torch.zeros(1, 4, 2, 2),
            slots=torch.zeros(1, 2, 1),
        )
        labels = torch.tensor([[0, 1]])
        identity = filter_sequential_logits(raw, labels)
        gated = filter_sequential_logits(raw, labels, evidence_disagreement_threshold=0.25)
        torch.testing.assert_close(
            gated.prequential_log_likelihood[:, 0],
            identity.prequential_log_likelihood[:, 0],
        )
        torch.testing.assert_close(gated.log_weights[:, 1].exp(), torch.full((1, 2), 0.5), rtol=0, atol=0)
        self.assertFalse(torch.equal(gated.log_weights[:, 2], gated.log_weights[:, 1]))

    def test_js_disagreement_gate_is_invariant_to_particle_duplication(self):
        # Two genuinely different Bernoulli hypotheses, then the same pair
        # duplicated eight times. A K-stable gate must make the same decision
        # and yield the same aggregate posterior over the two modes.
        pair = torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]])
        duplicated = pair.repeat(1, 1, 8, 1)
        query_pair = torch.zeros(1, 4, 2, 2)
        query_duplicated = torch.zeros(1, 4, 16, 2)
        labels = torch.tensor([[1]])
        k2 = filter_sequential_logits(
            SequentialFilterLogits(pair, query_pair, torch.zeros(1, 2, 1)),
            labels,
            evidence_disagreement_js_threshold=0.1,
        )
        k16 = filter_sequential_logits(
            SequentialFilterLogits(duplicated, query_duplicated, torch.zeros(1, 16, 1)),
            labels,
            evidence_disagreement_js_threshold=0.1,
        )
        k16_mode_weights = torch.stack(
            (
                k16.log_weights[:, -1, 0::2].exp().sum(-1),
                k16.log_weights[:, -1, 1::2].exp().sum(-1),
            ),
            dim=-1,
        )
        torch.testing.assert_close(k16_mode_weights, k2.log_weights[:, -1].exp())

    def test_js_disagreement_gate_blocks_small_shared_variation(self):
        logits = torch.tensor([[[[0.1, -0.1], [-0.1, 0.1]]]])
        raw = SequentialFilterLogits(logits, torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 1))
        prediction = filter_sequential_logits(
            raw,
            torch.tensor([[1]]),
            evidence_disagreement_js_threshold=0.1,
        )
        torch.testing.assert_close(
            prediction.log_weights[:, -1].exp(),
            torch.full((1, 2), 0.5),
            rtol=0,
            atol=0,
        )

    def test_transition_prior_recovers_mass_after_particle_collapse(self):
        # The first ten rows overwhelmingly support particle zero; the final
        # row supports particle one. With no transition its predictive mass is
        # effectively gone, while the transition prior restores it causally.
        first = torch.tensor([4.0, -4.0])
        second = torch.tensor([-4.0, 4.0])
        logits = torch.stack([torch.stack((first, second))] * 10 + [torch.stack((second, first))])[None]
        raw = SequentialFilterLogits(
            logits,
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 2, 1),
        )
        labels = torch.zeros(1, 11, dtype=torch.long)
        static = filter_sequential_logits(raw, labels)
        switching = filter_sequential_logits(raw, labels, transition_probability=0.05)
        self.assertGreater(
            switching.prequential_log_likelihood[0, -1],
            static.prequential_log_likelihood[0, -1],
        )

    def test_generator_covers_all_required_conditions(self):
        for condition in CONDITIONS:
            batch = generate_controlled_episodes(
                self.config, np.random.default_rng(4), condition=condition, batch_size=2
            )
            self.assertEqual(batch.conditions, (condition, condition))
            self.assertEqual(batch.stream_x.shape, (2, 8, 1))
            self.assertEqual(batch.query_y.shape, (2, 4))
            self.assertTrue(torch.isfinite(batch.exact_p1).all())
        neutral = generate_controlled_episodes(self.config, np.random.default_rng(5), condition="neutral", batch_size=2)
        torch.testing.assert_close(neutral.exact_p1, torch.full_like(neutral.exact_p1, 0.5))
        contradiction = generate_controlled_episodes(
            self.config, np.random.default_rng(6), condition="contradictory", batch_size=2
        )
        self.assertTrue(torch.all(contradiction.initial_evidence_class >= 0))

    def test_tabicl_updates_never_exceed_prior_size(self):
        rows = 16
        split = 12
        raw = {
            "x": torch.randn(2, rows, 1),
            "y": torch.randint(0, 2, (2, rows)).float(),
            "target_y": torch.randint(0, 2, (2, rows)).long(),
            "train_test_split_index": split,
        }
        episode = next_tabicl_episode(iter((raw,)), self.config, np.random.default_rng(9))
        self.assertEqual(episode.initial_support_x.shape[1], 4)
        self.assertEqual(episode.stream_x.shape[1], 4)

    def test_evidence_reordering_preserves_final_filter(self):
        original = predict_episode(self.model.eval(), self.batch)
        order = torch.tensor([4, 1, 7, 0, 5, 2, 6, 3])
        reordered_batch = copy.copy(self.batch)
        reordered_batch.stream_x = self.batch.stream_x[:, order]
        reordered_batch.stream_y = self.batch.stream_y[:, order]
        reordered = predict_episode(self.model, reordered_batch)
        torch.testing.assert_close(
            original.log_weights[:, -1].exp(), reordered.log_weights[:, -1].exp(), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            original.joint_probabilities()[:, -1],
            reordered.joint_probabilities()[:, -1],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_query_permutation_preserves_canonical_joint(self):
        original = predict_episode(self.model.eval(), self.batch).joint_probabilities()
        order = torch.tensor([2, 0, 3, 1])
        permuted_batch = copy.copy(self.batch)
        permuted_batch.query_x = self.batch.query_x[:, order]
        permuted = predict_episode(self.model, permuted_batch).joint_probabilities()
        outcomes = torch.arange(16)
        bits = (outcomes[:, None] >> torch.arange(3, -1, -1)) & 1
        permuted_indices = (bits[:, order] * (2 ** torch.arange(3, -1, -1))).sum(-1)
        torch.testing.assert_close(original, permuted[:, :, permuted_indices], atol=1e-6, rtol=1e-6)

    def test_one_filter_state_is_shared_by_every_query(self):
        prediction = predict_episode(self.model, self.batch)
        state = 5
        slot_probabilities = prediction.query_logits.softmax(-1)
        expected = (prediction.log_weights[:, state, None, :, None].exp() * slot_probabilities).sum(dim=2)
        torch.testing.assert_close(prediction.marginal_probabilities()[:, state], expected)

    def test_backbone_is_frozen_and_unchanged_by_optimizer(self):
        before = {name: value.detach().clone() for name, value in self.model.backbone.state_dict().items()}
        optimizer_parameters = list(self.model.head_parameters())
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=0.01)
        optimizer.zero_grad()
        loss, _ = sequential_filter_loss(self.model, self.batch)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.backbone.parameters()))
        optimizer.step()
        for name, value in self.model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_checkpoint_round_trip_is_exact(self):
        expected = predict_episode(self.model.eval(), self.batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filter.pth"
            gate = {"passed": True, "checks": {"test": True}}
            save_sequential_filter_checkpoint(
                path,
                self.model,
                training_config={"seed": 71},
                source_checkpoint_sha256="source",
                stage="controlled",
                controlled_gate=gate,
            )
            loaded, metadata = load_sequential_filter_checkpoint(path)
            actual = predict_episode(loaded.eval(), self.batch)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(actual.log_weights, expected.log_weights, rtol=0, atol=0)
        torch.testing.assert_close(actual.joint_probabilities(), expected.joint_probabilities(), rtol=0, atol=0)
        self.assertEqual(metadata["source_checkpoint_sha256"], "source")
        self.assertEqual(metadata["controlled_gate"], gate)

    def test_tiny_training_loss_decreases_and_is_finite(self):
        optimizer = torch.optim.Adam(self.model.head_parameters(), lr=0.01)
        initial = float(sequential_filter_loss(self.model, self.batch)[0].detach())
        for _ in range(8):
            optimizer.zero_grad()
            loss, _ = sequential_filter_loss(self.model, self.batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        final = float(sequential_filter_loss(self.model, self.batch)[0].detach())
        self.assertLess(final, initial)


class SequentialLatentFilterCLITests(unittest.TestCase):
    @staticmethod
    def _write_tiny_checkpoint(path: Path) -> None:
        backbone = tiny_backbone()
        torch.save(
            {
                "architecture": {
                    "embedding_size": backbone.embedding_size,
                    "num_attention_heads": backbone.num_attention_heads,
                    "mlp_hidden_size": backbone.mlp_hidden_size,
                    "num_layers": backbone.num_layers,
                    "num_outputs": backbone.num_outputs,
                },
                "model": backbone.state_dict(),
            },
            path,
        )

    def test_all_stage_smoke_writes_artifacts_and_skips_tabicl_on_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "nano.pth"
            output = root / "run"
            self._write_tiny_checkpoint(checkpoint)
            result = run(tiny_config(stage="all", checkpoint=str(checkpoint), output_dir=str(output)))
            self.assertEqual(result, output.resolve())
            required = (
                "config.json",
                "controlled_checkpoint.pth",
                "controlled_trajectory_metrics.csv",
                "controlled_summary.csv",
                "controlled_gate.json",
                "learning_curves.csv",
                "ordinary_accuracy.json",
                "selection.json",
            )
            self.assertTrue(all((output / name).is_file() for name in required))
            selection = json.loads((output / "selection.json").read_text())
            self.assertFalse(selection["tabicl_ran"])
            self.assertEqual(selection["tabicl_skipped_reason"], "controlled_gate_failed")
            self.assertFalse((output / "tabicl_checkpoint.pth").exists())

    def test_tabicl_rejects_missing_and_failed_controlled_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nano = root / "nano.pth"
            self._write_tiny_checkpoint(nano)
            missing_config = tiny_config(
                stage="tabicl",
                checkpoint=str(nano),
                controlled_checkpoint=str(root / "missing.pth"),
                output_dir=str(root / "missing-run"),
            )
            with self.assertRaises(FileNotFoundError):
                run(missing_config)

            failed = root / "failed.pth"
            model = NanoTabPFNSequentialLatentFilter(tiny_backbone())
            save_sequential_filter_checkpoint(
                failed,
                model,
                training_config={},
                source_checkpoint_sha256="source",
                stage="controlled",
                controlled_gate={"passed": False},
            )
            failed_config = replace(
                missing_config,
                controlled_checkpoint=str(failed),
                output_dir=str(root / "failed-run"),
            )
            with self.assertRaises(ValueError):
                run(failed_config)

    def test_gate_thresholds_are_executable(self):
        # The smoke path exercises the real report; this focused fixture locks
        # that every declared condition participates in the gate.
        rows = []
        for condition in CONDITIONS:
            for milestone in (0, 8):
                rows.append(
                    {
                        "condition": condition,
                        "trial": 0,
                        "milestone": milestone,
                        "weight_0": 0.5,
                        "supporting_weight": 0.5,
                        "initial_supporting_weight": 0.5,
                        "opposite_weight": 0.5,
                        "incoherent_mass": 0.5,
                        "slot_joint_js": 0.0,
                        "different_supporting_slots": 0.0,
                    }
                )
        import pandas as pd

        report = controlled_gate_report(
            pd.DataFrame(rows),
            {
                "query_permutation_max_error": 0.0,
                "evidence_order_weight_max_error": 0.0,
                "evidence_order_joint_max_error": 0.0,
            },
            stream_count=8,
        )
        self.assertFalse(report["passed"])
        self.assertIn("contradiction_reversal", report["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
