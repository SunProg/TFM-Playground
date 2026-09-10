#!/usr/bin/env python3
"""Retrofit v3 pilot checkpoints (written before checkpoints carried an
``architecture`` block) with the inference-ready metadata
``load_checkpoint_for_inference`` needs, so they can be scored by
``evaluate_tabarena_small``.

Runs written after the checkpoint format gained ``architecture`` natively
(``pretrain_multiregime_v3.inference_architecture``) do not need this; it
exists only to backfill an already-completed run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CELL_KIND = {0: "plain", 1: "slot", 2: "plain", 3: "slot", 4: "plain", 5: "slot"}
CELL_MODE = {0: "original", 1: "original", 2: "fixed", 3: "fixed", 4: "curriculum", 5: "curriculum"}


def architecture_for(kind: str, config: dict) -> dict:
    base = {
        "num_attention_heads": config["heads"],
        "embedding_size": config["width"],
        "mlp_hidden_size": config["hidden"],
        "num_layers": config["layers"],
        "num_outputs": 2,
    }
    if kind == "plain":
        return base
    return {
        "num_layers": base["num_layers"],
        "embedding_size": base["embedding_size"],
        "num_attention_heads": base["num_attention_heads"],
        "mlp_hidden_size": base["mlp_hidden_size"],
        "backbone_num_outputs": 2,
        "num_slots": config["num_slots"],
        "max_classes": 2,
        "num_slot_iterations": 3,
        "competitive_slots": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="e.g. runs/37103332")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = []
    for index in range(6):
        source = args.run_root / f"cell-{index}" / "checkpoint.pth"
        state = torch.load(source, map_location="cpu", weights_only=False)
        config = state["metadata"]["config"]
        kind, mode = CELL_KIND[index], CELL_MODE[index]
        name = f"multiregime_v3-{kind}-{mode}"
        payload = {
            "model": state["model"],
            "architecture": architecture_for(kind, config),
            "source_checkpoint": str(source),
            "step": state.get("step"),
            "cell_index": index,
            "kind": kind,
            "mode": mode,
        }
        output_path = args.output_root / f"{name}.pth"
        args.output_root.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        manifest.append({"name": name, "path": str(output_path), "kind": kind, "mode": mode, "cell_index": index})
        print(f"wrapped cell-{index} ({kind}, {mode}) -> {output_path}")

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrapped {len(manifest)} checkpoints in {args.output_root}")


if __name__ == "__main__":
    main()
