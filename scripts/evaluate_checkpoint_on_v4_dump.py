"""Evaluate saved checkpoints on a multiregime-v4 *training dump* (not a bank).

The training dumps written by ``dump_multiregime_v4_episodes.py`` have the
geometry the model actually trained on (e.g. 1024 rows, 2-12 features), which
the factorial evaluation bank does not. Scoring checkpoints here separates
"never learns regime-conditional prediction at all" from "learns it
in-distribution but it does not transfer to the bank's geometry".

Episodes are cropped and split exactly as ``MultiregimeV4DumpLoader`` does
(``[:num_datapoints, :num_features]``, split at ``train_test_split_index``).
Several checkpoints can be scored in one process so the dump is read once.

    python scripts/evaluate_checkpoint_on_v4_dump.py \\
        --dump data/.../r_z-multiregime.h5 --max-episodes 4096 \\
        --checkpoint runs/.../checkpoint-002000.pth --checkpoint runs/.../final_checkpoint.pth \\
        --outdir runs/.../v4_dump_eval --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _episode_auc(probabilities: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """OvR macro AUC for one episode; NaN when the query labels are degenerate.

    Local copy of the helper in multiregime_v4_evaluation so this script also
    runs against checkouts that predate the AUC addition there.
    """
    if torch.unique(target).numel() < 2:
        return float("nan")
    valid = probabilities[:, :num_classes].double()
    valid = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs_np, target_np = valid.cpu().numpy(), target.cpu().numpy()
    try:
        if num_classes == 2:
            return float(roc_auc_score(target_np, probs_np[:, 1]))
        return float(roc_auc_score(target_np, probs_np, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except ValueError:
        return float("nan")


def _nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float("nan") if np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _load_episodes(dump: h5py.File, max_episodes: int | None) -> list[dict]:
    count = dump["X"].shape[0]
    if max_episodes is not None:
        count = min(count, max_episodes)
    families = [f.decode("utf-8") for f in dump["families"][:]]
    rows = np.asarray(dump["num_datapoints"][:count], dtype=np.int64)
    splits = np.asarray(dump["train_test_split_index"][:count], dtype=np.int64)
    feats = np.asarray(dump["num_features"][:count], dtype=np.int64)
    regimes = np.asarray(dump["num_regimes"][:count], dtype=np.int64)
    classes = np.asarray(dump["num_classes"][:count], dtype=np.int64)
    family_idx = np.asarray(dump["family_index"][:count], dtype=np.int64)
    expose_z = np.asarray(dump["expose_z"][:count], dtype=bool)
    return [
        {
            "index": i,
            "num_datapoints": int(rows[i]),
            "split": int(splits[i]),
            "num_features": int(feats[i]),
            "num_regimes": int(regimes[i]),
            "num_classes": int(classes[i]),
            "family": families[family_idx[i]],
            "expose_z": bool(expose_z[i]),
        }
        for i in range(count)
    ]


def _batches(episodes: list[dict], max_batch: int):
    """Yield runs of consecutive episodes sharing geometry (like the trainer's same-group batches)."""
    i = 0
    while i < len(episodes):
        key = (episodes[i]["num_datapoints"], episodes[i]["split"], episodes[i]["num_features"])
        j = i + 1
        while j < len(episodes) and j - i < max_batch and (
            episodes[j]["num_datapoints"], episodes[j]["split"], episodes[j]["num_features"]
        ) == key:
            j += 1
        yield episodes[i:j]
        i = j


@torch.no_grad()
def score(model: torch.nn.Module, dump: h5py.File, episodes: list[dict], device: str, max_batch: int) -> list[dict]:
    model.eval()
    results: list[dict] = []
    for batch in _batches(episodes, max_batch):
        idx = [e["index"] for e in batch]
        rows, split, feats = batch[0]["num_datapoints"], batch[0]["split"], batch[0]["num_features"]
        x = torch.from_numpy(dump["X"][idx, :rows, :feats]).to(device)
        y = torch.from_numpy(dump["y"][idx, :rows]).to(device)
        logits = model(x[:, :split], y[:, :split], x[:, split:])
        target = y[:, split:].long()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").reshape_as(target)
        probs = F.softmax(logits, dim=-1)
        for k, meta in enumerate(batch):
            results.append(
                {
                    **meta,
                    "query_cross_entropy": float(ce[k].mean()),
                    "query_accuracy": float((logits[k].argmax(-1) == target[k]).float().mean()),
                    "query_auc": _episode_auc(probs[k], target[k], meta["num_classes"]),
                }
            )
    return results


def _summary(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[tuple(r[f] for f in fields)].append(r)
    out = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        out.append(
            {
                **dict(zip(fields, key)),
                "episodes": len(group),
                "query_cross_entropy": float(np.mean([r["query_cross_entropy"] for r in group])),
                "query_accuracy": float(np.mean([r["query_accuracy"] for r in group])),
                "query_auc": _nanmean(r["query_auc"] for r in group),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-episodes", type=int, default=4096)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump_name = Path(args.dump).stem

    with h5py.File(args.dump, "r") as dump:
        episodes = _load_episodes(dump, args.max_episodes)
        for ckpt_path in args.checkpoint:
            state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model = NanoTabPFNModel(**state["architecture"]).to(args.device)
            model.load_state_dict(state["model"])
            per_episode = score(model, dump, episodes, args.device, args.max_batch)
            report = {
                "dump": str(args.dump),
                "checkpoint": str(ckpt_path),
                "step": int(state["step"]),
                "episodes": len(per_episode),
                "overall": _summary(per_episode, ())[0],
                "by_num_regimes": _summary(per_episode, ("num_regimes",)),
                "by_family": _summary(per_episode, ("family",)),
                "by_num_regimes_family": _summary(per_episode, ("num_regimes", "family")),
                "by_num_classes": _summary(per_episode, ("num_classes",)),
                "per_episode": per_episode,
            }
            out = outdir / f"{dump_name}__step{int(state['step']):06d}.json"
            out.write_text(json.dumps(report, indent=1) + "\n")
            o = report["overall"]
            print(f"{dump_name} step={state['step']}: loss={o['query_cross_entropy']:.4f} acc={o['query_accuracy']:.4f} -> {out}")


if __name__ == "__main__":
    main()
