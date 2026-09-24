import json
import random
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from tabicl.prior import PriorDataset

from tfmplayground.experiments.dump_multiregime_v4_episodes import (
    DumpConfig,
    MultiregimeV4DumpLoader,
    dump_multiregime_v4_episodes,
)
from tfmplayground.experiments.multiregime_v4 import (
    sample_episode_v4,
    sample_generation_group_v4,
)
from tfmplayground.experiments.multiregime_v4_evaluation import (
    MultiregimeV4EvaluationBank,
    V4EvaluationBankConfig,
    build_multiregime_v4_evaluation_banks,
    evaluate_multiregime_v4_bank,
)


class _ConstantClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, support_x, support_y, query_x):
        del support_x, support_y
        self.calls += 1
        return torch.zeros((*query_x.shape[:2], 2), dtype=query_x.dtype, device=query_x.device)


class MultiregimeV4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def group(rng):
        return sample_generation_group_v4(
            rng,
            min_features=2,
            max_features=5,
            min_instances=48,
            max_instances=64,
            min_train_fraction=0.5,
            max_train_fraction=0.7,
            num_regimes=3,
            min_samples_per_regime=8,
            mix_probs=(1.0, 0.0),
        )

    def test_generation_groups_sample_width_and_length_and_native_mlp_causality(self):
        groups = [self.group(np.random.default_rng(seed)) for seed in range(20)]
        self.assertGreater(len({group.num_features for group in groups}), 1)
        self.assertGreater(len({group.rows for group in groups}), 1)
        for group in groups:
            self.assertTrue(2 <= group.num_features <= 5)
            self.assertTrue(48 <= group.rows <= 64)
            self.assertGreaterEqual(group.support_size, 24)
            self.assertGreaterEqual(group.query_size, 1)
            self.assertEqual(group.num_regimes, 3)
            self.assertEqual(group.prior_type, "mlp_scm")
            self.assertIsInstance(group.hparams["is_causal"], bool)
        self.assertEqual({group.hparams["is_causal"] for group in groups}, {False, True})

    def test_sampled_regime_count_is_limited_by_support_evidence(self):
        groups = [
            sample_generation_group_v4(
                np.random.default_rng(seed),
                min_features=2,
                max_features=5,
                min_instances=64,
                max_instances=192,
                min_train_fraction=0.2,
                max_train_fraction=0.9,
                min_regimes=2,
                max_regimes=4,
                min_samples_per_regime=8,
                mix_probs=(1.0, 0.0),
            )
            for seed in range(50)
        ]
        self.assertGreater(len({group.num_regimes for group in groups}), 1)
        for group in groups:
            self.assertGreaterEqual(group.support_size // 8, group.num_regimes)

    def test_fixed_geometry_reproduces_the_native_training_split(self):
        groups = [
            sample_generation_group_v4(
                np.random.default_rng(seed),
                min_features=2,
                max_features=12,
                min_instances=160,
                max_instances=160,
                min_train_fraction=0.1,
                max_train_fraction=0.9,
                min_regimes=2,
                max_regimes=4,
                min_samples_per_regime=8,
                fixed_support_size=128,
                fixed_query_size=32,
            )
            for seed in range(20)
        ]
        self.assertEqual({group.support_size for group in groups}, {128})
        self.assertEqual({group.query_size for group in groups}, {32})
        self.assertEqual({group.rows for group in groups}, {160})
        self.assertGreater(len({group.num_features for group in groups}), 1)

    def test_member_episodes_share_group_shape_but_have_independent_rows(self):
        group = self.group(np.random.default_rng(9))
        first = sample_episode_v4(
            101,
            family="soft_gate",
            min_features=2,
            max_features=5,
            num_regimes=3,
            calibration_size=16,
            pad_features=6,
            group=group,
        )
        second = sample_episode_v4(
            102,
            family="soft_gate",
            min_features=2,
            max_features=5,
            num_regimes=3,
            calibration_size=16,
            pad_features=6,
            group=group,
        )
        self.assertEqual(first.d, group.num_features)
        self.assertEqual(second.d, group.num_features)
        self.assertEqual(first.support_x.shape, (1, group.support_size, 6))
        self.assertEqual(second.query_x.shape, (1, group.query_size, 6))
        self.assertFalse(torch.equal(first.support_x, second.support_x))
        self.assertTrue(set(first.support_z).issubset({0, 1, 2}))

    def test_shared_and_multiregime_modes_are_paired_and_k1_is_identical(self):
        group = self.group(np.random.default_rng(31))
        common = dict(
            family="soft_gate",
            min_features=2,
            max_features=5,
            num_regimes=3,
            calibration_size=32,
            pad_features=6,
            group=group,
        )
        shared = sample_episode_v4(401, rule_mode="shared", **common)
        multiregime = sample_episode_v4(401, rule_mode="multiregime", **common)
        self.assertTrue(torch.equal(shared.support_x, multiregime.support_x))
        self.assertTrue(torch.equal(shared.query_x, multiregime.query_x))
        self.assertTrue(set(shared.support_z).issubset({0}))
        self.assertTrue(set(multiregime.support_z).issubset({0, 1, 2}))
        self.assertTrue(torch.any(shared.support_y != multiregime.support_y))

        g_common = {**common, "mechanism_mode": "g_z"}
        g_shared = sample_episode_v4(1, rule_mode="shared", **g_common)
        g_multiregime = sample_episode_v4(1, rule_mode="multiregime", **g_common)
        self.assertTrue(torch.equal(g_shared.support_x, g_multiregime.support_x))
        self.assertTrue(torch.equal(g_shared.query_x, g_multiregime.query_x))
        self.assertTrue(torch.any(g_shared.support_y != g_multiregime.support_y))

        k1 = sample_episode_v4(
            402,
            family="soft_gate",
            min_features=3,
            max_features=3,
            num_regimes=1,
            support_size=24,
            query_size=8,
            calibration_size=32,
            mix_probs=(1.0, 0.0),
            rule_mode="shared",
        )
        k1_multiregime = sample_episode_v4(
            402,
            family="soft_gate",
            min_features=3,
            max_features=3,
            num_regimes=1,
            support_size=24,
            query_size=8,
            calibration_size=32,
            mix_probs=(1.0, 0.0),
            rule_mode="multiregime",
        )
        self.assertTrue(torch.equal(k1.support_x, k1_multiregime.support_x))
        self.assertTrue(torch.equal(k1.support_y, k1_multiregime.support_y))
        self.assertTrue(torch.equal(k1.query_y, k1_multiregime.query_y))

    def test_multiclass_rules_are_paired_and_record_their_cardinality(self):
        group = sample_generation_group_v4(
            np.random.default_rng(44),
            min_features=2,
            max_features=2,
            min_instances=160,
            max_instances=160,
            min_train_fraction=0.8,
            max_train_fraction=0.8,
            num_regimes=3,
            min_samples_per_regime=32,
            mix_probs=(1.0, 0.0),
        )
        common = dict(
            family="soft_gate",
            min_features=2,
            max_features=2,
            num_regimes=3,
            group=group,
            num_classes=5,
            max_classes=5,
        )
        for mechanism_mode in ("r_z", "g_z"):
            # Native labels (default profile): paired X, native Reg2Cls rules, no calibrated class vector.
            shared = sample_episode_v4(55, rule_mode="shared", mechanism_mode=mechanism_mode, **common)
            multiregime = sample_episode_v4(55, rule_mode="multiregime", mechanism_mode=mechanism_mode, **common)
            for episode in (shared, multiregime):
                self.assertEqual(episode.num_classes, 5)
                self.assertLess(int(episode.support_y.max()), 5)
                self.assertLess(int(episode.query_y.max()), 5)
                self.assertEqual(episode.scm_metadata["num_classes"], 5)
                self.assertIsNone(episode.scm_metadata["target_class_probabilities"])
                self.assertEqual(episode.scm_metadata["class_probability_scope"], "native_reg2cls")
                self.assertFalse(episode.scm_metadata["label_prior_controlled"])
                self.assertEqual(len(episode.scm_metadata["candidate_num_classes"]), 3)
                support_classes = set(np.unique(episode.support_y.numpy()).tolist())
                self.assertEqual(support_classes, set(np.unique(episode.query_y.numpy()).tolist()))
                self.assertGreaterEqual(len(support_classes), 2)
            self.assertTrue(torch.equal(shared.support_x, multiregime.support_x))
            self.assertTrue(torch.equal(shared.query_x, multiregime.query_x))
            self.assertFalse(torch.equal(shared.support_y, multiregime.support_y))

            # Deprecated production profile keeps the calibrated uniform class vector.
            production = dict(common, group=sample_generation_group_v4(
                np.random.default_rng(44), min_features=2, max_features=2, min_instances=160, max_instances=160,
                min_train_fraction=0.8, max_train_fraction=0.8, num_regimes=3, min_samples_per_regime=32,
                mix_probs=(1.0, 0.0), hp_profile="production"))
            shared = sample_episode_v4(55, rule_mode="shared", mechanism_mode=mechanism_mode, **production)
            multiregime = sample_episode_v4(55, rule_mode="multiregime", mechanism_mode=mechanism_mode, **production)
            for episode in (shared, multiregime):
                self.assertEqual(episode.scm_metadata["target_class_probabilities"], [0.2] * 5)
                total_counts = np.asarray(
                    episode.scm_metadata["support_class_counts_by_routing_regime"]
                ) + np.asarray(episode.scm_metadata["query_class_counts_by_routing_regime"])
                # Exact duplicated observed-X rows cannot be split by a
                # deterministic score rule; otherwise rank rounding is one.
                self.assertTrue(np.all(np.ptp(total_counts, axis=1) <= 2))
            self.assertTrue(torch.equal(shared.support_x, multiregime.support_x))

    def test_binary_positive_rate_is_shared_by_routing_regime(self):
        for family in ("soft_gate", "persistent"):
            episode = sample_episode_v4(
                97,
                family=family,
                min_features=2,
                max_features=2,
                num_regimes=3,
                support_size=96,
                query_size=96,
                num_classes=2,
                max_classes=2,
                min_samples_per_regime=32,
                class_ratio=0.25,
                mechanism_mode="r_z",
            )
            self.assertEqual(episode.scm_metadata["target_class_probabilities"], [0.75, 0.25])
            total_counts = np.asarray(episode.scm_metadata["support_class_counts_by_routing_regime"])
            total_counts += np.asarray(episode.scm_metadata["query_class_counts_by_routing_regime"])
            for counts in total_counts:
                self.assertLessEqual(abs(counts[1] / counts.sum() - 0.25), 1.0 / counts.sum())

    def test_test_profile_k1_matches_native_tabicl_values(self):
        """K=1 must be a tensor-level native control, not merely same-shaped."""
        for seed in range(801, 809):
            episode = sample_episode_v4(
                seed,
                family="soft_gate",
                min_features=2,
                max_features=2,
                num_regimes=1,
                support_size=12,
                query_size=4,
                calibration_size=16,
                pad_features=2,
                mechanism_mode="g_z",
                hp_profile="tabicl_test",
            )
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            native = PriorDataset(
                batch_size=1,
                batch_size_per_gp=1,
                batch_size_per_subgp=1,
                min_features=2,
                max_features=2,
                max_classes=2,
                min_seq_len=16,
                max_seq_len=17,
                min_train_size=12,
                max_train_size=13,
                prior_type="mix_scm",
                n_jobs=1,
                device="cpu",
            )
            native_x, native_y, _, _, _ = next(native)
            observed_x = torch.cat((episode.support_x, episode.query_x), dim=1)
            observed_y = torch.cat((episode.support_y, episode.query_y), dim=1)
            self.assertTrue(torch.equal(observed_x, native_x), seed)
            self.assertTrue(torch.equal(observed_y, native_y), seed)
            self.assertTrue(episode.scm_metadata["native_k1_equivalence"])

    def test_default_profile_k1_matches_native_tabicl_for_both_mechanisms(self):
        """The default (native) profile must reproduce TabICL's dataset at K=1, not only tabicl_test."""
        for mechanism_mode in ("r_z", "g_z"):
            for seed in range(811, 817):
                episode = sample_episode_v4(
                    seed,
                    family="soft_gate",
                    min_features=2,
                    max_features=2,
                    num_regimes=1,
                    support_size=12,
                    query_size=4,
                    pad_features=2,
                    mechanism_mode=mechanism_mode,
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                native = PriorDataset(
                    batch_size=1,
                    batch_size_per_gp=1,
                    batch_size_per_subgp=1,
                    min_features=2,
                    max_features=2,
                    max_classes=2,
                    min_seq_len=16,
                    max_seq_len=17,
                    min_train_size=12,
                    max_train_size=13,
                    prior_type="mix_scm",
                    n_jobs=1,
                    device="cpu",
                )
                native_x, native_y, _, _, _ = next(native)
                observed_x = torch.cat((episode.support_x, episode.query_x), dim=1)
                observed_y = torch.cat((episode.support_y, episode.query_y), dim=1)
                self.assertTrue(torch.equal(observed_x, native_x), (mechanism_mode, seed))
                self.assertTrue(torch.equal(observed_y, native_y), (mechanism_mode, seed))
                self.assertTrue(episode.scm_metadata["native_k1_equivalence"])
                self.assertEqual(episode.scm_metadata["hp_profile"], "native")

    def test_native_labels_spread_binary_class_ratios_like_tabicl(self):
        """Native Reg2Cls boundaries give imbalanced binary tasks; the old calibrated path gave 50/50 only."""

        def majority_fractions(hp_profile: str, num_regimes: int, count: int = 120) -> np.ndarray:
            fractions = []
            for seed in range(3000, 3000 + count):
                episode = sample_episode_v4(
                    seed,
                    family="soft_gate",
                    min_features=3,
                    max_features=3,
                    num_regimes=num_regimes,
                    support_size=96,
                    query_size=32,
                    num_classes=2,
                    max_classes=2,
                    min_samples_per_regime=32,
                    mechanism_mode="r_z",
                    rule_mode="multiregime",
                    hp_profile=hp_profile,
                )
                y = torch.cat((episode.support_y, episode.query_y), dim=1).numpy().astype(int).ravel()
                z = np.concatenate((episode.support_z, episode.query_z))
                # per routing regime: each regime's rule has its own native class boundary
                for regime in range(num_regimes):
                    counts = np.bincount(y[z == regime], minlength=2)
                    fractions.append(counts.max() / counts.sum())
            return np.asarray(fractions)

        for num_regimes in (1, 3):
            native = majority_fractions("native", num_regimes)
            self.assertGreater(np.mean(native >= 0.7), 0.25, f"K={num_regimes}: native ratios are not spread")
            self.assertLess(np.mean(native < 0.55), 0.6, f"K={num_regimes}: native ratios pile up at 50/50")
        production = majority_fractions("production", 1, count=40)
        self.assertTrue(np.all(production < 0.55), "production profile should still calibrate to 50/50")

    def test_expose_z_appends_the_routing_score_under_native_labels(self):
        for family in ("soft_gate", "persistent"):
            for num_regimes in (1, 3):
                blind = sample_episode_v4(4242, family=family, min_features=3, max_features=3, num_regimes=num_regimes,
                                          support_size=96, query_size=32, max_classes=3, min_samples_per_regime=32, pad_features=4)
                exposed = sample_episode_v4(4242, family=family, min_features=3, max_features=3, num_regimes=num_regimes,
                                            support_size=96, query_size=32, max_classes=3, min_samples_per_regime=32, pad_features=4, expose_z=True)
                self.assertEqual(exposed.d, blind.d + 1)
                self.assertIsNotNone(exposed.z_column_index)
                z = exposed.support_x[0, :, exposed.z_column_index].numpy()
                self.assertTrue(np.allclose(z, blind.support_x[0, :, 0].numpy()))
                # same labels (rules are a deterministic function of the seed) and the same
                # feature columns, only reordered by the exposed column permutation
                self.assertTrue(torch.equal(exposed.support_y, blind.support_y))
                remaining = np.delete(exposed.support_x[0, :, : exposed.d].numpy(), exposed.z_column_index, axis=1)
                self.assertEqual(
                    sorted(map(tuple, remaining.T.round(5))), sorted(map(tuple, blind.support_x[0, :, : blind.d].numpy().T.round(5)))
                )
                again = sample_episode_v4(4242, family=family, min_features=3, max_features=3, num_regimes=num_regimes,
                                          support_size=96, query_size=32, max_classes=3, min_samples_per_regime=32, pad_features=4)
                self.assertTrue(torch.equal(again.support_y, blind.support_y))

    def test_paired_dumps_have_identical_base_tasks_and_different_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            shared_path = Path(directory) / "shared.h5"
            multiregime_path = Path(directory) / "multiregime.h5"
            common = dict(
                episodes=8,
                generation_group_size=4,
                min_features=2,
                max_features=4,
                min_instances=48,
                max_instances=60,
                min_train_fraction=0.5,
                max_train_fraction=0.7,
                min_regimes=2,
                max_regimes=3,
                min_samples_per_regime=8,
                calibration_size=16,
                mlp_probability=1.0,
                seed=29,
            )
            dump_multiregime_v4_episodes(DumpConfig(output=str(shared_path), rule_mode="shared", **common))
            dump_multiregime_v4_episodes(
                DumpConfig(output=str(multiregime_path), rule_mode="multiregime", **common)
            )
            with h5py.File(shared_path, "r") as shared, h5py.File(multiregime_path, "r") as multiregime:
                self.assertTrue(np.array_equal(shared["X"][:], multiregime["X"][:]))
                self.assertTrue(np.array_equal(shared["num_features"][:], multiregime["num_features"][:]))
                self.assertTrue(np.array_equal(shared["num_datapoints"][:], multiregime["num_datapoints"][:]))
                self.assertTrue(np.any(shared["y"][:] != multiregime["y"][:]))
                self.assertEqual(shared["rule_mode"][()].decode(), "shared")
                self.assertEqual(multiregime["rule_mode"][()].decode(), "multiregime")

    def test_dump_loader_never_crosses_a_generation_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v4.h5"
            dump_multiregime_v4_episodes(
                DumpConfig(
                    output=str(path),
                    episodes=8,
                    generation_group_size=4,
                    min_features=2,
                    max_features=4,
                    min_instances=48,
                    max_instances=60,
                    min_train_fraction=0.5,
                    max_train_fraction=0.7,
                    min_regimes=2,
                    max_regimes=3,
                    min_samples_per_regime=8,
                    calibration_size=16,
                    mlp_probability=1.0,
                    seed=17,
                )
            )
            loader = MultiregimeV4DumpLoader(path, batch_size=4)
            try:
                first, second = loader.sample(), loader.sample()
            finally:
                loader.close()

        for batch in (first, second):
            self.assertEqual(len(torch.unique(batch["generation_group"])), 1)
            self.assertEqual(len(torch.unique(batch["num_datapoints"])), 1)
            self.assertEqual(len(torch.unique(batch["train_test_split_index"])), 1)
            rows = int(batch["num_datapoints"][0])
            split = int(batch["train_test_split_index"][0])
            self.assertEqual(batch["support_x"].shape[1], split)
            self.assertEqual(batch["query_x"].shape[1], rows - split)
            self.assertEqual(batch["support_x"].shape[0], 4)

    def test_factorial_validation_and_test_banks_have_disjoint_episode_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / "validation.h5"
            test_path = Path(directory) / "test.h5"
            config = V4EvaluationBankConfig(
                validation_output=str(validation_path),
                test_output=str(test_path),
                validation_seed=101,
                test_seed=202,
                episodes_per_cell=2,
                support_sizes=(16, 32),
                num_features=(2, 3),
                num_regimes=(1, 2),
                class_ratios=(0.25, 0.75),
                query_size=4,
                calibration_size=16,
                min_samples_per_regime=8,
                mlp_probability=1.0,
            )
            validation, test = build_multiregime_v4_evaluation_banks(config)
            self.assertEqual((validation, test), (validation_path, test_path))

            with h5py.File(validation_path, "r") as validation_handle, h5py.File(test_path, "r") as test_handle:
                self.assertEqual(validation_handle["X"].shape[0], 256)
                self.assertEqual(test_handle["X"].shape[0], 256)
                self.assertEqual(len(np.unique(validation_handle["cell_id"][:])), 128)
                self.assertTrue(np.all(np.bincount(validation_handle["cell_id"][:]) == 2))
                self.assertTrue(
                    set(validation_handle["episode_seed"][:]).isdisjoint(set(test_handle["episode_seed"][:]))
                )
                # Adjacent cells differ only by rule mode, so their base task
                # is paired for a direct single-rule vs multiregime contrast.
                self.assertTrue(np.array_equal(validation_handle["X"][0], validation_handle["X"][2]))
                self.assertEqual(validation_handle["episode_seed"][0], validation_handle["episode_seed"][2])
                self.assertIn("scm_metadata_json", validation_handle)
                metadata = json.loads(validation_handle["scm_metadata_json"][0].decode())
                self.assertIn("sampled_is_causal", metadata)
                self.assertIn("effective_num_layers", metadata)
                self.assertIn("tabicl_hparams", metadata)

            with MultiregimeV4EvaluationBank(validation_path) as bank:
                first = bank.episode(0)
                self.assertEqual(len(bank), 256)
                self.assertEqual(first["support_x"].shape, (1, 16, 2))
                self.assertEqual(first["query_x"].shape, (1, 4, 2))
                self.assertEqual(first["metadata"]["num_regimes"], 1)
                self.assertEqual(set(first["support_regime"].flatten().tolist()), {0})
                self.assertEqual(first["metadata"]["task_family"], "soft_gate")
                self.assertEqual(first["metadata"]["mechanism_mode"], "r_z")
                self.assertIn("sampled_is_causal", first["metadata"])
                self.assertEqual(bank.scm_metadata(0)["num_regimes"], 1)
                model = _ConstantClassifier()
                report = evaluate_multiregime_v4_bank(model, bank)

            self.assertEqual(report["episodes"], 256)
            self.assertEqual(model.calls, 128)
            self.assertEqual(len(report["per_episode"]), 256)
            self.assertEqual(len(report["by_cell"]), 128)
            self.assertEqual(len(report["by_support_size"]), 2)
            self.assertEqual(len(report["by_num_features"]), 2)
            self.assertEqual(len(report["by_num_regimes"]), 2)
            self.assertEqual(len(report["by_class_ratio"]), 2)
            self.assertEqual(len(report["by_task_family"]), 2)
            self.assertEqual(len(report["by_mechanism_mode"]), 2)
            self.assertEqual(len(report["by_rule_mode"]), 2)
            self.assertGreaterEqual(len(report["by_sampled_is_causal"]), 1)
            self.assertGreaterEqual(len(report["by_scm_num_layers"]), 1)

    def test_evaluation_omits_unsupported_support_regime_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            validation_path = Path(directory) / "validation.h5"
            test_path = Path(directory) / "test.h5"
            config = V4EvaluationBankConfig(
                validation_output=str(validation_path),
                test_output=str(test_path),
                validation_seed=301,
                test_seed=302,
                episodes_per_cell=1,
                support_sizes=(64, 128),
                num_features=(2,),
                num_regimes=(1, 2, 3, 4),
                num_classes=(2, 5),
                class_ratios=(0.5,),
                task_families=("soft_gate",),
                mechanism_modes=("r_z",),
                rule_modes=("shared",),
                query_size=32,
                calibration_size=16,
                min_samples_per_regime=32,
                mlp_probability=1.0,
            )
            self.assertEqual(
                config.valid_support_regime_pairs,
                ((64, 1), (64, 2), (128, 1), (128, 2), (128, 3), (128, 4)),
            )
            build_multiregime_v4_evaluation_banks(config)
            with h5py.File(validation_path, "r") as handle:
                self.assertEqual(handle["X"].shape[0], 12)
                self.assertEqual(set(handle["num_classes"][:].tolist()), {2, 5})
                # Native multiclass cells must realise every requested class in both halves.
                for i in range(handle["X"].shape[0]):
                    n = int(handle["num_classes"][i]); split = int(handle["support_size"][i]); rows = split + int(handle["query_size"][i])
                    y = handle["y"][i, :rows]
                    self.assertEqual(np.unique(y[:split]).size, n)
                    self.assertEqual(np.unique(y[split:]).size, n)
                pairs = set(zip(handle["support_size"][:].tolist(), handle["num_regimes"][:].tolist(), strict=True))
                self.assertNotIn((64, 3), pairs)
                self.assertNotIn((64, 4), pairs)


if __name__ == "__main__":
    unittest.main()
