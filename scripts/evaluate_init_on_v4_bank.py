"""Evaluate the step-0 (freshly initialized, untrained) model on a v4 bank.

``pretrain_plain_nanotabpfn.run_pretraining`` seeds with ``config.seed`` and
then constructs the model, so the initial weights are exactly reproducible
from a run's ``config.json``. This script rebuilds that step-0 model and scores
it against a bank, giving the trajectory a true starting point (the training
loop's first logged validation is at ``validation_interval``, not 0).

Because the init depends only on seed + architecture, every prior family with
the same seed and model size shares one step-0 model; pass several
``--output`` paths to write the same report to each of their run dirs (as
``v4_validation/step-000000.json`` so ``extract_v4_validation_cells.py`` picks
it up as step 0).

    python scripts/evaluate_init_on_v4_bank.py \\
        --config runs/.../original-small/seed-2402/config.json \\
        --bank data/.../validation.h5 \\
        --output runs/.../original-small/seed-2402/v4_validation/step-000000.json \\
        --output runs/.../r_z-fixed-small/seed-2402/v4_validation/step-000000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfmplayground.experiments.multiregime_v4_evaluation import evaluate_multiregime_v4_bank
from tfmplayground.experiments.pretrain_plain_nanotabpfn import PlainPretrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config.json from a run dir (seed + architecture)")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", action="append", required=True, help="Repeatable; report is written to each")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes-per-forward", type=int, default=None)
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text())
    config = PlainPretrainingConfig(**{k: v for k, v in raw.items() if k in PlainPretrainingConfig.__dataclass_fields__})

    # Mirror run_pretraining exactly: seed, then construct (init happens on CPU), then move.
    set_randomness_seed(config.seed)
    model = NanoTabPFNModel(**config.architecture()).to(args.device)

    max_episodes = args.max_episodes_per_forward or config.v4_evaluation_batch_size
    report = evaluate_multiregime_v4_bank(model, args.bank, device=args.device, max_episodes_per_forward=max_episodes)
    report["source_step"] = 0
    report["source_note"] = "untrained model at initialization, reproduced from config seed + architecture"

    text = json.dumps(report, indent=2) + "\n"
    for out in args.output:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"step=0 overall_loss={report['overall']['query_cross_entropy']:.4f} -> {path}")


if __name__ == "__main__":
    main()
