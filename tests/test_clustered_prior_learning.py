"""Checks of the prior's intended information structure."""

import unittest

import numpy as np

from tfmplayground.experiments.clustered_prior_learning import (
    model_inputs,
    routed_probability,
    sample_batch,
)


class ClusteredPriorTests(unittest.TestCase):
    def test_cluster_visibility_and_label_balance(self):
        x, y, z = sample_batch(9, batch_size=128)
        self.assertGreater(((x[:, :, 0] > 0) == z).mean(), 0.995)
        for k in (0, 1):
            self.assertLess(abs(y[z == k].mean() - 0.5), 0.02)

    def test_reproducible_and_fresh_rules(self):
        first = sample_batch(1)
        second = sample_batch(1)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[1], sample_batch(2)[1]))

    def test_latent_inputs_do_not_expose_regime_or_query_labels(self):
        x, y, z = sample_batch(3)
        before = model_inputs(x, y, z, 128, False)
        changed_y = y.copy()
        changed_y[:, 128:] = 1 - changed_y[:, 128:]
        after = model_inputs(x, changed_y, 1 - z, 128, False)
        for a, b in zip(before, after, strict=True):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertEqual(before[0].shape[-1], 3)
        self.assertEqual(model_inputs(x, y, z, 128, True)[0].shape[-1], 5)

    def test_oracle_learns_fresh_rules_from_support(self):
        accuracies = []
        for seed in range(20):
            x, y, z = [a[0] for a in sample_batch(seed)]
            p = routed_probability(x[:128], y[:128], x[128:], z[:128], z[128:])
            accuracies.append(((p >= 0.5) == y[128:]).mean())
        self.assertGreater(np.mean(accuracies), 0.96)


if __name__ == "__main__":
    unittest.main()
