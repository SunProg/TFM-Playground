import pytest
import torch

from tfmplayground.models.reconstruction_routing import ReconstructionRouter, align_cell_slots, bounded_attention
from tfmplayground.models.slot_regime import slot_regime_loss


@pytest.mark.parametrize("scope", ["data", "cell_and_data", "cell"])
@pytest.mark.parametrize("variant", ["masks", "values"])
def test_routing_gradients_and_roundtrip(scope, variant):
    torch.manual_seed(5)
    model = ReconstructionRouter(scope=scope, variant=variant).eval()
    x, y, q = torch.randn(2, 6, 3), torch.randint(2, (2, 6)), torch.randn(2, 4, 3)
    target = torch.randint(2, (2, 4))
    prediction, mse = model(x, y, q)
    torch.testing.assert_close(prediction.gate().sum(-1), torch.ones(2, 4))
    ce = slot_regime_loss(prediction, target)
    parameters = list(model.embedding_decoder.parameters())
    assert all(g is None for g in torch.autograd.grad(ce, parameters, allow_unused=True, retain_graph=True))
    assert torch.autograd.grad(ce, model.query_key.weight, retain_graph=True)[0].norm() > 0
    (ce + mse).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in parameters)
    clone = ReconstructionRouter(scope=scope, variant=variant).eval()
    clone.load_state_dict(model.state_dict())
    actual, loss = clone(x, y, q)
    torch.testing.assert_close(actual.marginal_log_probabilities(), prediction.marginal_log_probabilities())
    torch.testing.assert_close(loss, mse)


def test_alignment_remaps_independently_permuted_support_slots():
    slots = torch.randn(1, 1, 4, 12).expand(1, 3, 4, 12).clone()
    slots[:, 1] = slots[:, 1, [2, 0, 3, 1]]
    slots[:, 2] = slots[:, 2, [1, 3, 0, 2]]
    aligned, indices = align_cell_slots(slots)
    torch.testing.assert_close(aligned, aligned[:, :1].expand_as(aligned))
    torch.testing.assert_close(aligned, slots.gather(2, indices[..., None].expand_as(slots)))


def test_bounded_reconstruction_attention():
    scores = torch.randn(2, 3, 6, requires_grad=True)
    errors = torch.rand(2, 6, requires_grad=True)
    base = scores.softmax(-1)
    weighted = bounded_attention(scores, errors)
    assert (weighted >= 0.75 * base).all()
    torch.testing.assert_close(weighted.sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(bounded_attention(scores, errors, eta=0), base, atol=0, rtol=0)
    assert torch.autograd.grad(weighted[..., 0].sum(), errors, allow_unused=True)[0] is None
