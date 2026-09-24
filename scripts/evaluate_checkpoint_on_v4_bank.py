"""Evaluate one saved pretrain_plain_nanotabpfn checkpoint against a v4 bank.

Standalone (no training loop): loads a ``checkpoint-XXXXXX.pth`` (or
``final_checkpoint.pth``), rebuilds the model from its stored architecture,
and runs ``evaluate_multiregime_v4_bank`` against the requested bank (e.g. the
test bank), writing the full report as JSON. Intended for evaluating a
checkpoint other than the final one — e.g. the checkpoint with the best
own-prior validation loss — since the training loop itself only evaluates the
test bank once, at the end.

    python scripts/evaluate_checkpoint_on_v4_bank.py \\
        --checkpoint runs/.../checkpoint-004000.pth \\
        --bank data/.../test.h5 \\
        --output runs/.../v4_test/best_val_step004000.json \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tfmplayground.experiments.multiregime_v4_evaluation import evaluate_multiregime_v4_bank
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True, help="Path to the evaluation bank .h5 (e.g. test.h5)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-episodes-per-forward", type=int, default=None)
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available.")

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = NanoTabPFNModel(**state["architecture"]).to(args.device)
    model.load_state_dict(state["model"])

    max_episodes = args.max_episodes_per_forward
    if max_episodes is None:
        max_episodes = state.get("training_config", {}).get("v4_evaluation_batch_size")

    report = evaluate_multiregime_v4_bank(model, args.bank, device=args.device, max_episodes_per_forward=max_episodes)
    report["source_checkpoint"] = str(args.checkpoint)
    report["source_step"] = state["step"]
    report["source_validation_query_cross_entropy"] = (state.get("validation") or {}).get("query_cross_entropy")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"step={state['step']} own_val_loss={report['source_validation_query_cross_entropy']} -> {output_path}")


if __name__ == "__main__":
    main()
