"""Numerical checks of proposed routing algebra, not production integration."""
import json

import torch


def route(similarity, errors, masks, *, beta=0.5, eta=0.25):
    # Keep normalization in float32 in the intended mixed-precision implementation.
    similarity = similarity.float()
    errors = errors.float()
    u = (errors / (errors + errors.mean(-1, keepdim=True) + 1e-8)).detach()
    base = similarity.softmax(-1)
    attention = base if eta == 0 else (1 - eta) * base + eta * (similarity - beta * u[:, None]).softmax(-1)
    return attention, attention @ masks.detach().float()


def main():
    torch.manual_seed(19)
    similarity = torch.randn(2, 3, 7, requires_grad=True)
    errors = torch.rand(2, 7, requires_grad=True)
    logits = torch.randn(2, 7, 4, requires_grad=True)
    masks = logits.softmax(-1)
    attention, weights = route(similarity, errors, masks)
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert (attention >= 0.75 * similarity.softmax(-1)).all()
    (-weights[..., 0].log().mean()).backward()
    assert similarity.grad.abs().sum() > 0
    assert errors.grad is None and logits.grad is None

    base, _ = route(similarity, errors, masks, eta=0)
    expected = similarity.softmax(-1)
    torch.testing.assert_close(base, expected, atol=0, rtol=0)
    g1 = torch.autograd.grad(base.square().sum(), similarity, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected.square().sum(), similarity)[0]
    torch.testing.assert_close(g1, g2, atol=0, rtol=0)
    for value in (0., 2., 1e20):
        equal, _ = route(similarity, torch.full_like(errors, value), masks)
        torch.testing.assert_close(equal, expected)

    # Aggregation is invariant to joint support reordering and equivariant to slot reordering.
    p = torch.randperm(7)
    _, reordered = route(similarity[..., p], errors[:, p], masks[:, p])
    torch.testing.assert_close(reordered, weights)
    k = torch.randperm(4)
    _, reordered_slots = route(similarity, errors, masks[..., k])
    torch.testing.assert_close(reordered_slots, weights[..., k])

    # Query specificity is possible when masks differ across support rows.
    _, distinct = route(torch.tensor([[[4., 0.], [0., 4.]]]), torch.zeros(1, 2), torch.eye(2)[None])
    assert distinct[0, 0, 0] > 0.9 and distinct[0, 1, 1] > 0.9

    # But identical responsibilities make the route independent of every query.
    sim = torch.randn(1, 2, 3, requires_grad=True)
    constant = torch.tensor([[[0.5, 0.5]]]).expand(1, 3, 2)
    _, uniform = route(sim, torch.tensor([[0., 2., 4.]]), constant)
    torch.testing.assert_close(uniform, torch.full((1, 2, 2), 0.5))
    grad = torch.autograd.grad(-uniform[..., 0].log().sum(), sim)[0]
    assert grad.abs().max() < 1e-6

    # Cell masks can encode different partitions but have identical mean slot usage.
    cells = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    torch.testing.assert_close(cells.mean(1)[0], cells.mean(1)[1])
    print(json.dumps({
        "status": "algebra_checks_passed",
        "checks": ["normalization", "finite_nonnegative", "75_percent_floor", "detached_score_gradients",
                   "eta_zero_values_and_gradients", "equal_error_invariance", "support_permutation",
                   "slot_permutation", "query_specificity_possible"],
        "limitations_reproduced": ["identical_masks_remove_query_specificity_and_router_gradient",
                                   "mean_cell_masks_discard_cell_partition_information"],
        "uniform_mask_max_router_gradient": float(grad.abs().max()),
        "distinct_query_weights": distinct.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
