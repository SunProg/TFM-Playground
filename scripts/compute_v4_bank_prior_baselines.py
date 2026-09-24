"""Per-episode class-prior baselines for a multiregime-v4 evaluation bank.

Binary cells vary ``class_ratio`` (0.1/0.3/0.5) and multiclass cells vary
``num_classes``, so raw cross entropy / accuracy have different floors per
cell: predicting the class prior already scores CE 0.33 vs 0.69 and accuracy
0.9 vs 0.5 at ratio 0.1 vs 0.5. Averaging raw metrics across cells therefore
mixes task difficulty with model skill.

This script scores, for every episode, the trivial predictor that outputs the
support set's empirical class frequencies for each query row:

* ``prior_cross_entropy``  - CE of that predictor on the query labels
* ``majority_accuracy``    - accuracy of always predicting the support majority
* ``uniform_cross_entropy`` - log(num_classes), for reference

Model metrics can then be reported as ``excess_cross_entropy = CE -
prior_cross_entropy`` (>= 0 means no better than predicting the prior; lower
is better) and ``accuracy_gain = accuracy - majority_accuracy`` (<= 0 means no
better than the majority class; higher is better). Depends only on the bank's
labels, so it applies retroactively to every existing report.

    python scripts/compute_v4_bank_prior_baselines.py --bank validation.h5 --output validation_prior.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tfmplayground.experiments.multiregime_v4_evaluation import MultiregimeV4EvaluationBank

EPS = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    with MultiregimeV4EvaluationBank(args.bank, device="cpu") as bank:
        for index in range(len(bank)):
            meta = bank._metadata(index)
            num_classes = int(meta["num_classes"])
            episode = bank.episode(index)
            sy = episode["support_y"][0].numpy().reshape(-1).astype(np.int64)
            qy = episode["query_y"][0].numpy().reshape(-1).astype(np.int64)
            counts = np.bincount(sy, minlength=num_classes).astype(np.float64)
            prior = np.clip(counts / counts.sum(), EPS, 1.0)
            prior = prior / prior.sum()
            rows[str(index)] = {
                "episode_id": index,
                "num_classes": num_classes,
                "class_ratio": meta["class_ratio"],
                "prior_cross_entropy": float(-np.log(prior[qy]).mean()),
                "majority_accuracy": float((qy == int(counts.argmax())).mean()),
                "uniform_cross_entropy": float(np.log(num_classes)),
            }
            if (index + 1) % 5000 == 0:
                print(f"{index + 1}/{len(bank)}", flush=True)
        split = bank.split

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bank": str(args.bank), "split": split, "episodes": rows}))
    ce = np.mean([r["prior_cross_entropy"] for r in rows.values()])
    acc = np.mean([r["majority_accuracy"] for r in rows.values()])
    print(f"{split}: {len(rows)} episodes, mean prior CE={ce:.4f}, mean majority acc={acc:.4f} -> {out}")


if __name__ == "__main__":
    main()
