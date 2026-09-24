#!/usr/bin/env python3
"""Create an explicitly versioned v3 evaluation-bank manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from tfmplayground.experiments.multiregime_v3 import OriginalPrior
from tfmplayground.experiments.pretrain_multiregime_v3 import PilotConfig, evaluation_bank, source_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot_config = json.loads(args.pilot_config.read_text())
    pilot_result = json.loads(args.pilot_result.read_text())
    config_data = dict(pilot_config["config"])
    config_data["device"] = "cpu"
    config = PilotConfig(**config_data)
    bank = evaluation_bank(config, OriginalPrior(config.generator()))
    hashes = {family: [episode.tensor_hash() for episode in episodes] for family, episodes in bank.items()}
    tabicl_version = importlib.metadata.version("tabicl")
    if tabicl_version != "2.1.1":
        raise RuntimeError(f"expected tabicl==2.1.1, found {tabicl_version}")

    manifest = {
        "bank_name": "multiregime_v3_evaluation_bank_tabicl211",
        "bank_tabicl_version": tabicl_version,
        "bank_source_hash": source_hash(),
        "pilot_source_hash": pilot_result.get("source_hash"),
        "pilot_config": str(args.pilot_config),
        "pilot_result": str(args.pilot_result),
        "config": pilot_config["config"],
        "evaluation_bank_hashes": hashes,
        "episodes_per_family": config.validation_episodes,
        "families": list(hashes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tabicl": tabicl_version, "hashes": hashes}))


if __name__ == "__main__":
    main()
