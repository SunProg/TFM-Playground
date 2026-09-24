import json
from argparse import Namespace

import pytest
import torch

from tfmplayground.experiments.reconstruction_routing_scale import configurations, run


def test_grid_is_unique_and_covers_scopes_seeds_and_weights():
    cells = configurations()
    assert len(cells) == 28
    assert len({tuple(c.items()) for c in cells}) == 28
    assert {c["scope"] for c in cells} == {"data", "cell", "cell_and_data", "vanilla"}
    assert {c["seed"] for c in cells} == {11, 12}
    assert not any(c["scope"] == "cell" and c["arm"] == "embedding_mse" for c in cells)
    assert [c["seed"] for c in cells if c["arm"] == "vanilla"] == [11, 12]


@pytest.mark.parametrize("index", (*range(13), 26, 27))
def test_train_validate_checkpoint_and_resume(index, tmp_path):
    torch.set_num_threads(1)
    args = Namespace(
        index=index,
        output=str(tmp_path),
        device="cpu",
        width=12,
        hidden=24,
        layers=1,
        heads=3,
        support_size=6,
        query_size=4,
        batch_size=1,
        accumulate=1,
        steps=1,
        warmup_steps=1,
        validation_interval=1,
        validation_episodes=1,
        learning_rate=1e-4,
    )
    run(args)
    state = torch.load(tmp_path / "latest.pth", weights_only=False)
    assert state["step"] == 1
    assert torch.isfinite(torch.tensor(state["best_nll"]))
    assert (tmp_path / "best.pth").exists()
    metrics = (tmp_path / "metrics.jsonl").read_text()
    assert "query_backbone_gradient_norm" in metrics
    assert "auxiliary_backbone_gradient_norm" in metrics
    run(args)
    assert (tmp_path / "metrics.jsonl").read_text() == metrics
    assert json.loads((tmp_path / "complete.json").read_text())["steps"] == 1
