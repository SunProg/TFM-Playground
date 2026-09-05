import numpy as np

from tfmplayground.experiments.tabarena_residual_probe import ResidualProbeConfig, probe_table


def _design(seed: int = 0, rows: int = 900, features: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features))
    first, second = rng.normal(size=features), rng.normal(size=features)
    single = (x @ first > 0).astype(int)
    gated = np.where(x[:, 0] > 0, x @ second > 0, x @ first > 0).astype(int)
    noisy = np.where(rng.random(rows) < 0.3, 1 - single, single)
    return x, single, gated, noisy


def test_routed_gain_separates_a_planted_regime_from_noise():
    """The statistic must be positive only when a second rule actually exists.

    A residual gap alone cannot say that: label noise produces one too.  These
    three tables are the reason the probe reports ``routed_gain`` rather than
    the gap, and the reason its experts are linear -- a boosted pooled model
    absorbs the gated rule and every case reads zero.
    """
    config = ResidualProbeConfig(outer_folds=3, inner_folds=3)
    x, single, gated, noisy = _design()
    planted = probe_table(x, gated, config)
    homogeneous = probe_table(x, single, config)
    noise = probe_table(x, noisy, config)

    assert planted["routed_gain"] > 0.0
    assert homogeneous["routed_gain"] < planted["routed_gain"]
    assert noise["routed_gain"] < planted["routed_gain"]
    # Noise is not covariate-dependent, so no gate can find it.
    assert noise["gate_auc"] < 0.6
    # The flexible reference absorbs the gated rule the linear experts cannot.
    assert planted["pooled_hgb_log_loss"] < planted["pooled_log_loss"]


def test_every_row_is_scored_out_of_fold():
    """An in-fold gate reports a perfect AUC on random labels; this must not."""
    config = ResidualProbeConfig(outer_folds=3, inner_folds=3)
    x, _, _, noisy = _design(seed=3)
    assert probe_table(x, noisy, config)["gate_auc"] < 0.65
