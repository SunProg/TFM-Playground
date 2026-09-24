import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from tfmplayground.experiments.dump_tabicl_mix_scm_matched_geometry import (
    MatchedMixSCMDumpConfig,
    dump_tabicl_mix_scm_matched_geometry,
    read_geometry_groups,
)


class _FakeTabICLLoader:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pd = SimpleNamespace(prior=SimpleNamespace(n_jobs=None))
        self.calls.append(kwargs)

    def __iter__(self):
        rows = self.kwargs["num_datapoints_min"]
        width = self.kwargs["min_features"]
        batch_size = self.kwargs["batch_size"]
        split = self.kwargs["min_train_size"]
        x = torch.zeros((batch_size, rows, width), dtype=torch.float32)
        y = torch.arange(rows).repeat(batch_size, 1).remainder(2).float()
        yield {"x": x, "y": y, "target_y": y, "train_test_split_index": split}


def _write_geometry_source(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.zeros((4, 12, 5), dtype="f4"))
        handle.create_dataset("generation_group", data=np.array([4, 4, 9, 9], dtype="i4"))
        handle.create_dataset("num_datapoints", data=np.array([10, 10, 12, 12], dtype="i4"))
        handle.create_dataset("train_test_split_index", data=np.array([6, 6, 7, 7], dtype="i4"))
        handle.create_dataset("num_features", data=np.array([3, 3, 5, 5], dtype="i4"))
        handle.create_dataset("num_regimes", data=np.array([2, 2, 4, 4], dtype="i1"))


class MatchedMixSCMDumpTests(unittest.TestCase):
    def test_reads_contiguous_homogeneous_geometry_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "v4.h5"
            _write_geometry_source(source)
            groups, max_rows, max_features = read_geometry_groups(source, batch_size=2)

        self.assertEqual((max_rows, max_features), (12, 5))
        actual = [(g.group_id, g.rows, g.support_size, g.requested_num_features, g.num_regimes) for g in groups]
        self.assertEqual(actual, [(4, 10, 6, 3, 2), (9, 12, 7, 5, 4)])

    def test_writes_native_mix_scm_with_the_source_geometry(self):
        _FakeTabICLLoader.calls.clear()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "v4.h5"
            output = Path(temporary) / "matched.h5"
            _write_geometry_source(source)
            with patch(
                "tfmplayground.experiments.dump_tabicl_mix_scm_matched_geometry.TabICLPriorDataLoader",
                _FakeTabICLLoader,
            ):
                result = dump_tabicl_mix_scm_matched_geometry(
                    MatchedMixSCMDumpConfig(output=str(output), geometry_source=str(source), batch_size=2)
                )
            self.assertEqual(result, output)
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle["X"].shape, (4, 12, 5))
                np.testing.assert_array_equal(handle["num_datapoints"][:], [10, 10, 12, 12])
                np.testing.assert_array_equal(handle["train_test_split_index"][:], [6, 6, 7, 7])
                np.testing.assert_array_equal(handle["generation_group"][:], [4, 4, 9, 9])
                np.testing.assert_array_equal(handle["geometry_num_regimes"][:], [2, 2, 4, 4])
                self.assertEqual(handle["prior_family"][()].decode(), "tabicl_mix_scm_matched_geometry")

        self.assertEqual(len(_FakeTabICLLoader.calls), 2)
        actual_calls = [
            (call["prior_type"], call["num_datapoints_min"], call["min_features"], call["min_train_size"])
            for call in _FakeTabICLLoader.calls
        ]
        self.assertEqual(actual_calls, [("mix_scm", 10, 3, 6), ("mix_scm", 12, 5, 7)])


if __name__ == "__main__":
    unittest.main()
