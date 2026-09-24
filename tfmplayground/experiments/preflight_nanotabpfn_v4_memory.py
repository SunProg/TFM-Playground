"""Run one worst-geometry NanoTabPFN training update and report CUDA peak memory."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, required=True)
    parser.add_argument("--mlp-hidden-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=512)
    parser.add_argument("--query-size", type=int, default=512)
    parser.add_argument("--num-features", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--timed-updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2402)
    return parser


def main(argv: list[str] | None = None) -> dict[str, float | int]:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This memory preflight requires CUDA.")
    if min(
        args.embedding_size,
        args.num_attention_heads,
        args.mlp_hidden_size,
        args.num_layers,
        args.micro_batch_size,
        args.support_size,
        args.query_size,
        args.num_features,
        args.num_classes,
        args.timed_updates,
    ) < 1:
        raise ValueError("All dimensions must be positive.")
    if args.embedding_size % args.num_attention_heads:
        raise ValueError("embedding_size must be divisible by num_attention_heads.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = NanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_outputs=args.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    support_x = torch.randn(args.micro_batch_size, args.support_size, args.num_features, device=device)
    support_y = torch.randint(
        args.num_classes, (args.micro_batch_size, args.support_size), device=device
    ).float()
    query_x = torch.randn(args.micro_batch_size, args.query_size, args.num_features, device=device)
    query_y = torch.randint(
        args.num_classes, (args.micro_batch_size, args.query_size), device=device
    )
    def update() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(support_x, support_y, query_x)
        loss = F.cross_entropy(logits.reshape(-1, args.num_classes), query_y.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss

    # Materialize kernels and optimizer state before measuring steady-state
    # update time. This update also proves the full backward path fits.
    update()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.timed_updates):
        loss = update()
    torch.cuda.synchronize(device)
    report: dict[str, float | int] = {
        "embedding_size": args.embedding_size,
        "num_attention_heads": args.num_attention_heads,
        "mlp_hidden_size": args.mlp_hidden_size,
        "num_layers": args.num_layers,
        "micro_batch_size": args.micro_batch_size,
        "support_size": args.support_size,
        "query_size": args.query_size,
        "num_features": args.num_features,
        "num_classes": args.num_classes,
        "loss": float(loss.item()),
        "mean_update_seconds": (time.perf_counter() - started) / args.timed_updates,
        "timed_updates": args.timed_updates,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
