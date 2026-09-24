import unittest

import torch

from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig, build_model, episode_loss
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig
from tfmplayground.experiments.scm_slot_learning import episode_batch
from tfmplayground.models.slot_regime import SlotRegimePrediction


class SCMSlotLearningTests(unittest.TestCase):
    def config(self, model_type):
        return V2TrainingConfig(
            seed=11,
            device="cpu",
            model_type=model_type,
            num_slots=2,
            slot_layer_index=0,
            slot_layer_indices=(0, 1),
            support_size=16,
            query_size=8,
            min_features=3,
            max_features=3,
            embedding_size=32,
            num_attention_heads=4,
            mlp_hidden_size=64,
            num_layers=2,
        )

    def test_slot_models_forward_and_backprop_on_scm_episode(self):
        episode = episode_batch(SCMRegimeConfig(support_size=16, query_size=8), 101, 2, query_size=8)
        for model_type in ("table_slot_head", "table_slot_backbone", "table_slot_mufasa"):
            model = build_model(self.config(model_type))
            loss, stats = episode_loss(model, episode, self.config(model_type))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("target_loss", stats)
            prediction = model(*episode.latent_inputs())
            self.assertIsInstance(prediction, SlotRegimePrediction)
            self.assertEqual(prediction.marginal_log_probabilities().shape, (2, 8, 2))


if __name__ == "__main__":
    unittest.main()
