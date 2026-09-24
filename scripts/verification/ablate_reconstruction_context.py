"""Evaluate saved value-routing pilots with the retrieved context removed."""

import argparse
import json
from pathlib import Path

import torch

from tfmplayground.experiments.reconstruction_routing_pilot import evaluate
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.reconstruction_routing import ReconstructionRouter

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
manifest = json.loads((args.directory / "manifest.json").read_text())
prior = SCMRegimeConfig(**{**manifest["prior"], "query_size": 64})
episodes = [sample_scm_regime_episode(prior, seed=900000 + i) for i in range(32)]
rows = []
for scope in ("data", "cell_and_data", "cell"):
    for seed in (11, 12):
        checkpoint = torch.load(args.directory / f"{scope}-values-{seed}.pth", weights_only=True)
        model = ReconstructionRouter(scope=scope, variant="values").eval()
        model.load_state_dict(checkpoint["model"])
        full, _ = evaluate(model, episodes)
        removed, _ = evaluate(model, episodes, drop_context=True)
        rows.append(
            dict(
                scope=scope,
                seed=seed,
                full_nll=full["nll"],
                full_accuracy=full["accuracy"],
                removed_nll=removed["nll"],
                removed_accuracy=removed["accuracy"],
            )
        )
(args.directory / "context_ablation.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
