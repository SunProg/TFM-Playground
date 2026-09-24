#!/usr/bin/env python3
"""Wrap reconstruction-routing checkpoints with TabArena architecture metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def wrap(task_dir: Path, output_path: Path) -> dict[str, object]:
    config = json.loads((task_dir / "config.json").read_text())
    cell = config["cell"]
    architecture = config["architecture"]
    checkpoint = task_dir / "best.pth"
    if not checkpoint.exists():
        checkpoint = task_dir / "latest.pth"
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_architecture = {
        "model_kind": "reconstruction_inference",
        **architecture,
        "scope": cell.get("scope"),
        "arm": cell["arm"],
        "selected_step": int(raw["step"]),
    }
    payload = {
        "model_type": "reconstruction_inference",
        "architecture": model_architecture,
        "model": raw["model"],
        "source_checkpoint": str(checkpoint),
        "selected_step": int(raw["step"]),
        "cell": cell,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "name": f"task-{task_dir.name.split('-')[-1]}-{cell['arm']}",
        "path": str(output_path),
        "task": int(task_dir.name.split("-")[-1]),
        "cell": cell,
        "selected_step": int(raw["step"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = []
    for task_dir in sorted(args.run_root.glob("task-*"), key=lambda p: int(p.name.split("-")[-1])):
        if not (task_dir / "config.json").exists():
            continue
        if not (task_dir / "best.pth").exists() and not (task_dir / "latest.pth").exists():
            continue
        task = int(task_dir.name.split("-")[-1])
        output = args.output_root / f"task-{task}.pth"
        manifest.append(wrap(task_dir, output))
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "standalone_checkpoints.txt").write_text(
        ",".join(f"{row['name']}={row['path']}" for row in manifest) + "\n"
    )
    print(json.dumps({"wrapped": len(manifest), "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
