"""Regenerate evaluation-bank episodes whose features contain non-finite values.

Native TabICL feature processing occasionally emits NaN on degenerate columns; ``sample_episode_v4`` now
redraws such bases, so regenerating an affected episode from its stored seed and cell parameters gives a
finite replacement drawn by the same rule. Edits the HDF5 in place (a ``.pre-nonfinite-repair`` copy of
the affected rows is written next to it as JSON).

    python scripts/repair_v4_bank_nonfinite_episodes.py --bank validation.h5 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.experiments.multiregime_v4 import FAMILIES, MECHANISM_MODES, RULE_MODES, sample_episode_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", required=True, nargs="+")
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--min-samples-per-regime", type=int, default=32)
    parser.add_argument("--mlp-probability", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import os
    import shutil

    for bank in args.bank:
        # Training jobs may hold the bank open for reading (HDF5 lock), so patch a copy and
        # atomically swap it in; readers that reopen the path per validation pick up the new file.
        work = bank + ".repair-tmp"
        shutil.copyfile(bank, work)
        with h5py.File(work, "r+") as h:
            n = h["X"].shape[0]
            width = h["input_width"][:] if "input_width" in h else h["num_features"][:]
            rows = h["support_size"][:] + h["query_size"][:]
            bad = []
            for start in range(0, n, 2000):
                x = h["X"][start : start + 2000]
                for j in range(x.shape[0]):
                    i = start + j
                    if not np.isfinite(x[j, : int(rows[i]), : int(width[i])]).all():
                        bad.append(i)
            print(f"{bank}: {len(bad)} / {n} episodes with non-finite X: {bad[:20]}")
            if args.dry_run or not bad:
                os.remove(work)
                continue
            expose_z = bool(int(h["expose_z"][()]))
            hp_profile = h["hp_profile"][()].decode() if "hp_profile" in h else "native"
            pad = h["X"].shape[2]
            backup = {}
            for i in bad:
                meta = dict(
                    seed=int(h["episode_seed"][i]), support_size=int(h["support_size"][i]), num_features=int(h["num_features"][i]),
                    num_regimes=int(h["num_regimes"][i]), num_classes=int(h["num_classes"][i]),
                    class_ratio=None if float(h["class_ratio"][i]) < 0 else float(h["class_ratio"][i]),
                    family=FAMILIES[int(h["task_family_index"][i])], mechanism_mode=MECHANISM_MODES[int(h["mechanism_mode_index"][i])],
                    rule_mode=RULE_MODES[int(h["rule_mode_index"][i])],
                )
                backup[i] = {"meta": meta, "y": h["y"][i].tolist()}
                episode = sample_episode_v4(
                    meta["seed"], family=meta["family"], min_features=meta["num_features"], max_features=meta["num_features"],
                    num_regimes=meta["num_regimes"], support_size=meta["support_size"], query_size=args.query_size,
                    pad_features=pad, num_classes=meta["num_classes"], max_classes=max(meta["num_classes"], int(h["num_classes"][:].max())),
                    min_samples_per_regime=args.min_samples_per_regime, mix_probs=(args.mlp_probability, 1 - args.mlp_probability),
                    class_ratio=meta["class_ratio"], rule_mode=meta["rule_mode"], mechanism_mode=meta["mechanism_mode"],
                    expose_z=expose_z, hp_profile=hp_profile, require_num_classes=True,
                )
                x = torch.cat((episode.support_x, episode.query_x), dim=1).numpy()[0]
                y = torch.cat((episode.support_y, episode.query_y), dim=1).numpy()[0]
                assert np.isfinite(x).all(), f"replacement for episode {i} still non-finite"
                total = meta["support_size"] + args.query_size
                h["X"][i, :total, :] = x
                h["y"][i, :total] = y
                if "regime" in h:
                    h["regime"][i, :total] = np.concatenate((episode.support_z, episode.query_z))
                if "z_column_index" in h:
                    h["z_column_index"][i] = -1 if episode.z_column_index is None else episode.z_column_index
                print(f"  repaired episode {i}: {meta['family']} {meta['mechanism_mode']} K={meta['num_regimes']} C={meta['num_classes']} "
                      f"support={meta['support_size']} features={meta['num_features']} base_attempt={episode.scm_metadata.get('native_base_attempt')}")
            Path(bank + ".pre-nonfinite-repair.json").write_text(json.dumps(backup))
        os.replace(work, bank)
        print(f"  {bank}: repaired copy swapped in")


if __name__ == "__main__":
    main()
