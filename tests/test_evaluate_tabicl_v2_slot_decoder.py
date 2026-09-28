from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tfmplayground.experiments.evaluate_tabicl_v2_slot_decoder import _add_subgroup_summaries


@pytest.mark.parametrize("model_column", ["framework", "method"])
def test_tabarena_results_are_stratified_by_size_classes_and_imbalance(tmp_path, model_column) -> None:
    context = SimpleNamespace(
        task_metadata_collection=SimpleNamespace(
            task_grid=lambda: pd.DataFrame(
                {
                    "dataset": ["small_3class", "large_7class"],
                    "split": [0, 0],
                    "max_train_rows": [1_000, 5_000],
                    "n_classes": [3, 7],
                }
            )
        )
    )
    task_results = pd.DataFrame(
        {
            model_column: ["slot_k4", "mlp", "slot_k4", "mlp"],
            "dataset": ["small_3class", "small_3class", "large_7class", "large_7class"],
            "fold": [0, 0, 0, 0],
            "problem_type": ["multiclass"] * 4,
            "metric_error": [0.2, 0.3, 0.4, 0.5],
            "aux_metric_error": [0.1, 0.2, 0.3, 0.4],
        }
    )
    imbalance_csv = tmp_path / "imbalance.csv"
    pd.DataFrame({"dataset": ["small_3class", "large_7class"], "imbalance_ratio": [1.5, 7.0]}).to_csv(
        imbalance_csv, index=False
    )

    _add_subgroup_summaries(context, task_results, tmp_path, str(imbalance_csv))

    summary = pd.read_csv(tmp_path / "stratified_metrics.csv")
    strata = set(summary["stratum"])
    assert {"multiclass_train_le_2k", "multiclass_train_gt_2k"}.issubset(strata)
    assert {"multiclass_classes_3_5", "multiclass_classes_6_10"}.issubset(strata)
    assert {"multiclass_ratio_le_2", "multiclass_ratio_gt_5"}.issubset(strata)
    assert len(pd.read_csv(tmp_path / "task_metrics_with_metadata.csv")) == 4


def test_expanded_limits_are_specific_to_tabicl_adapters() -> None:
    from tfmplayground.tabarena_model import (
        APPLICABILITY_CONSTRAINTS,
        BAYESIAN_APPLICABILITY_CONSTRAINTS,
        TabArenaTabICLDecoderModel,
        TabArenaTabICLZeroShotModel,
    )

    assert APPLICABILITY_CONSTRAINTS["max_train_rows"] == 10_000
    assert APPLICABILITY_CONSTRAINTS["max_features"] == 500
    assert BAYESIAN_APPLICABILITY_CONSTRAINTS["max_train_rows"] == 10_000
    for adapter in (TabArenaTabICLDecoderModel, TabArenaTabICLZeroShotModel):
        limits = adapter.get_applicability_constraints()
        assert limits["max_train_rows"] == 100_000
        assert limits["max_features"] == 2_000
        assert limits["max_classes"] == 10
        limits["max_train_rows"] = 1
        assert adapter.get_applicability_constraints()["max_train_rows"] == 100_000
