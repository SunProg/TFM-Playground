"""Entry point for K=2 particle training on the downloaded prior dump."""

from __future__ import annotations

import sys

from tfmplayground.experiments.train_prior_bimodal_filter import main as _main


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not any(value == "--prior-dump" or value.startswith("--prior-dump=") for value in values):
        values.extend(["--prior-dump", "300k_150x5_2.h5"])
    return _main(values)


if __name__ == "__main__":
    raise SystemExit(main())
