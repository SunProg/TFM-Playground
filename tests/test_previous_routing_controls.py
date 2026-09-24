import pytest
import torch

from tfmplayground.experiments.reconstruction_routing_pilot import PreviousControl
from tfmplayground.models.slot_regime import embedding_reconstruction_loss, slot_mi_loss, support_reconstruction_loss


@pytest.mark.parametrize(
    "scope,objective",
    [
        (s, o)
        for s in ("data", "cell", "cell_and_data")
        for o in ("query_only", "label_alpha", "embedding_mse")
        if not (s == "cell" and o == "embedding_mse")
    ],
)
def test_previous_control_is_original_model_with_explicit_loss(scope, objective):
    torch.manual_seed(11)
    wrapper = PreviousControl(scope, objective).eval()
    x, y, q = torch.randn(2, 6, 3), torch.randint(2, (2, 6)), torch.randn(2, 5, 3)
    prediction, auxiliary = wrapper(x, y, q)
    reference = wrapper.model(
        x, y, q, reconstruct_support=objective == "label_alpha", reconstruct_embeddings=objective == "embedding_mse"
    )
    torch.testing.assert_close(prediction.marginal_log_probabilities(), reference.marginal_log_probabilities())
    assert wrapper.model.query_routing_mode == "decoder"
    if objective == "query_only":
        assert auxiliary == 0
        assert not wrapper.model.embedding_reconstruction
    elif objective == "label_alpha":
        expected = support_reconstruction_loss(reference, y) + 0.05 * slot_mi_loss(reference.support_attention)
        torch.testing.assert_close(auxiliary, expected)
    else:
        torch.testing.assert_close(auxiliary, embedding_reconstruction_loss(reference))
