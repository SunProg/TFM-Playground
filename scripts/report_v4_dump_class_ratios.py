"""Class-count and binary class-ratio distribution of v4 dumps / banks, or of freshly generated episodes.

Used to verify that a prior contains imbalanced binary tasks (native TabICL: majority fraction ~uniform on
(0.5, 1)) rather than only 50/50 ones (the deprecated ``production`` profile). Stdlib + numpy + h5py; torch
only for ``--generate``.

    python scripts/report_v4_dump_class_ratios.py --h5 original.h5 r_z-multiregime.h5 --episodes 20000
    python scripts/report_v4_dump_class_ratios.py --generate 400 --num-regimes 1 --hp-profile native
    python scripts/report_v4_dump_class_ratios.py --native-tabicl 400          # TabICL's own PriorDataset
"""

from __future__ import annotations

import argparse

import numpy as np

BINS = (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.01)


def summarize(name: str, majority: np.ndarray, classes: np.ndarray) -> None:
    binary = majority[classes == 2]
    hist = np.histogram(binary, BINS)[0] / max(1, len(binary))
    counts = " ".join(f"{k}:{np.mean(classes == k):.2f}" for k in range(2, int(classes.max()) + 1))
    print(f"{name:34s} n={len(majority)} classes {counts}")
    if len(binary):
        print(
            f"   binary majority frac: mean {binary.mean():.3f} median {np.median(binary):.3f} "
            f">=0.7: {np.mean(binary >= 0.7):.2f} >=0.8: {np.mean(binary >= 0.8):.2f} >=0.9: {np.mean(binary >= 0.9):.2f}"
        )
        print("   hist [.50-.55 .55-.60 .60-.70 .70-.80 .80-.90 .90-1]: " + " ".join(f"{v:.2f}" for v in hist))


def stats_from_labels(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    majority, classes = [], []
    for y in rows:
        y = y[y >= 0].astype(int)
        c = np.bincount(y)
        c = c[c > 0]
        classes.append(len(c))
        majority.append(c.max() / c.sum())
    return np.asarray(majority), np.asarray(classes)


def report_h5(path: str, episodes: int) -> None:
    import h5py

    with h5py.File(path, "r") as h:
        n = min(episodes, h["X"].shape[0])
        y = h["y"][:n]
        if "num_datapoints" in h:
            rows = h["num_datapoints"][:]
            rows = np.repeat(rows, n) if len(rows) == 1 else rows[:n]
        elif "support_size" in h:  # evaluation bank
            rows = h["support_size"][:n] + h["query_size"][:n]
        else:
            rows = np.full(n, y.shape[1])
        profile = h["hp_profile"][()].decode() if "hp_profile" in h else "unknown (pre-2026-09-20 file)"
    majority, classes = stats_from_labels([y[i, : int(rows[i])] for i in range(n)])
    summarize(f"{path.split('/')[-1]} [{profile}]", majority, classes)


def report_generated(count: int, num_regimes: int, hp_profile: str, mechanism_mode: str, max_classes: int) -> None:
    import torch

    from tfmplayground.experiments.multiregime_v4 import sample_episode_v4, sample_generation_group_v4

    rng = np.random.default_rng(7)
    rows_list: list[np.ndarray] = []
    while len(rows_list) < count:
        group = sample_generation_group_v4(
            rng, min_features=2, max_features=12, min_instances=1024, max_instances=1024,
            min_train_fraction=0.1, max_train_fraction=0.9, num_regimes=num_regimes, min_samples_per_regime=32,
            hp_profile=hp_profile,
        )
        for _ in range(4):
            episode = sample_episode_v4(
                int(rng.integers(2**31)), family="soft_gate", min_features=2, max_features=12, num_regimes=num_regimes,
                group=group, max_classes=max_classes, mechanism_mode=mechanism_mode,
                rule_mode="multiregime" if num_regimes > 1 else "shared",
            )
            rows_list.append(torch.cat((episode.support_y, episode.query_y), 1).numpy().ravel())
    majority, classes = stats_from_labels(rows_list)
    summarize(f"generated v4 {hp_profile} K={num_regimes} {mechanism_mode}", majority, classes)


def report_native_tabicl(count: int, max_classes: int) -> None:
    import random

    import torch
    from tabicl.prior import PriorDataset

    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    ds = PriorDataset(
        batch_size=64, batch_size_per_gp=4, batch_size_per_subgp=4, min_features=2, max_features=12,
        max_classes=max_classes, min_seq_len=1024, max_seq_len=1025, min_train_size=0.1, max_train_size=0.9,
        prior_type="mix_scm", n_jobs=1, device="cpu",
    )
    rows_list: list[np.ndarray] = []
    while len(rows_list) < count:
        _, y, _, _, _ = next(ds)
        rows_list.extend(yi.numpy() for yi in y)
    majority, classes = stats_from_labels(rows_list)
    summarize("native TabICL PriorDataset", majority, classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", nargs="*", default=[], help="dump or bank files")
    parser.add_argument("--episodes", type=int, default=20000, help="episodes to read per file")
    parser.add_argument("--generate", type=int, default=0, help="generate this many episodes with sample_episode_v4")
    parser.add_argument("--num-regimes", type=int, default=1)
    parser.add_argument("--hp-profile", default="native")
    parser.add_argument("--mechanism-mode", default="r_z")
    parser.add_argument("--max-classes", type=int, default=5)
    parser.add_argument("--native-tabicl", type=int, default=0, help="sample this many datasets from TabICL's PriorDataset")
    args = parser.parse_args()
    for path in args.h5:
        report_h5(path, args.episodes)
    if args.generate:
        report_generated(args.generate, args.num_regimes, args.hp_profile, args.mechanism_mode, args.max_classes)
    if args.native_tabicl:
        report_native_tabicl(args.native_tabicl, args.max_classes)


if __name__ == "__main__":
    main()
