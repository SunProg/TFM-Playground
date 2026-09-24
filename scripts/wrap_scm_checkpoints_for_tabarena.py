#!/usr/bin/env python3
"""Wrap SCM sweep checkpoints with architecture metadata for TabArena inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HEAD_CONDITIONS = {
    "decoder_baseline": {"query_routing_mode": "decoder", "reconstruction_mixture": "attention"},
    "decoder_alpha": {"query_routing_mode": "decoder", "reconstruction_mixture": "alpha"},
    "blind_decoder": {"query_routing_mode": "blind_decoder", "reconstruction_mixture": "attention"},
    "blind_similarity": {"query_routing_mode": "blind_similarity", "reconstruction_mixture": "attention"},
}

def wrap(raw_path: Path, output_path: Path, architecture: dict[str, object]) -> None:
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    state_dict = raw["state_dict"]
    payload = {
        "model_type": architecture.get("model_kind", "tabpfn"),
        "architecture": architecture,
        "model": state_dict,
        "source_checkpoint": str(raw_path),
        "selected_step": int(raw.get("step", -1)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--nano-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--skip-variants",
        action="store_true",
        help="Do not wrap table_slot_backbone and table_slot_mufasa checkpoints.",
    )
    args = parser.parse_args()

    if not args.skip_variants and args.variant_root is None:
        parser.error("--variant-root is required unless --skip-variants is provided")

    base_architecture = {
        "num_layers": args.num_layers,
        "embedding_size": args.embedding_size,
        "num_attention_heads": args.num_attention_heads,
        "mlp_hidden_size": args.mlp_hidden_size,
        "num_outputs": 2,
    }

    manifest = []
    for condition, routing in HEAD_CONDITIONS.items():
        architecture = {
            **base_architecture,
            "model_kind": "table_slot_head",
            "num_slots": 2,
            "max_classes": 2,
            "slot_layer_indices": (0, 1),
            "num_slot_iterations": 3,
            "table_slot_scope": "cell_and_data",
            **routing,
        }
        for seed in (11, 12):
            raw_path = args.head_root / condition / f"seed-{seed}" / f"{condition}-seed-{seed}.pth"
            output_path = args.output_root / f"{condition}-seed-{seed}.pth"
            wrap(raw_path, output_path, architecture)
            manifest.append({"name": f"{condition}-seed-{seed}", "path": str(output_path), "model": condition})

    nano_architecture = {**base_architecture}
    for seed in (11, 12):
        raw_path = args.nano_root / f"seed-{seed}" / f"nanotabpfn-seed-{seed}.pth"
        output_path = args.output_root / f"nanotabpfn-seed-{seed}.pth"
        wrap(raw_path, output_path, nano_architecture)
        manifest.append({"name": f"nanotabpfn-seed-{seed}", "path": str(output_path), "model": "nanotabpfn"})

    if not args.skip_variants:
        assert args.variant_root is not None
        for model_type in ("table_slot_backbone", "table_slot_mufasa"):
            architecture = {
                **base_architecture,
                "model_kind": model_type,
                "num_slots": 2,
                "max_classes": 2,
                "slot_layer_indices": (0, 1),
                "num_slot_iterations": 3,
                "table_slot_scope": "cell_and_data",
            }
            for seed in (11, 12):
                raw_path = args.variant_root / model_type / f"seed-{seed}" / f"{model_type}-seed-{seed}.pth"
                output_path = args.output_root / f"{model_type}-seed-{seed}.pth"
                wrap(raw_path, output_path, architecture)
                manifest.append({"name": f"{model_type}-seed-{seed}", "path": str(output_path), "model": model_type})

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrapped {len(manifest)} checkpoints in {args.output_root}")


if __name__ == "__main__":
    main()
