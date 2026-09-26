"""Checks that the revised heart-site split keeps the query site observable."""

import numpy as np
import pandas as pd

from tfmplayground.experiments.evaluate_heart_sites_adaptation import (
    adaptation_folds,
    condition_frames,
)
from tfmplayground.experiments.evaluate_multiregime_v4_beyondarena import prepare_fold_data


def test_target_site_is_labelled_in_support_and_tags_have_matched_width():
    frame = pd.DataFrame({
        "x": np.arange(90),
        "heart_disease_diagnosis": [0, 1] * 45,
        "site": np.repeat(["cleveland", "hungary", "va_long_beach"], 30),
    })
    folds = adaptation_folds(frame, shots=6, repeats=2, seed=7)
    assert len(folds) == 6
    for support, query, _, site, _ in folds:
        train, test = frame.iloc[support], frame.iloc[query]
        assert not set(support) & set(query)
        assert len(support) + len(query) == len(frame)
        assert (train.site == site).sum() == 6
        assert test.site.eq(site).all()
        assert set(train.site) == set(frame.site)
        true_train, true_test, true_excluded = condition_frames(
            train, test, condition="site_true", shuffle_seed=9
        )
        shuffled_train, shuffled_test, shuffled_excluded = condition_frames(
            train, test, condition="site_shuffled", shuffle_seed=9
        )
        assert shuffled_train.site.value_counts().to_dict() == train.site.value_counts().to_dict()
        assert shuffled_test.site.equals(test.site)
        true_x, true_q, _, _ = prepare_fold_data(
            true_train, true_test,
            target_column="heart_disease_diagnosis", group_columns=true_excluded,
        )
        shuffled_x, shuffled_q, _, _ = prepare_fold_data(
            shuffled_train, shuffled_test,
            target_column="heart_disease_diagnosis", group_columns=shuffled_excluded,
        )
        assert true_x.shape == shuffled_x.shape
        assert true_q.shape == shuffled_q.shape
        # The target site's category is fitted on labelled support rows, so
        # every query receives its observed category instead of an unknown code.
        assert len(np.unique(true_x[:, -1])) == 3
        target_code = np.unique(true_x[train.site.eq(site).to_numpy(), -1])
        assert len(target_code) == 1
        assert np.all(true_q[:, -1] == target_code[0])
