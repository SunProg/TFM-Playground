"""Compatibility entry point for static Bayesian nanoTabPFN training."""

from tfmplayground.experiments.train_bayesian_nanotabpfn import (
    StaticBayesianTrainingConfig,
    StaticEpisodeBatch,
    build_parser,
    controlled_static_batch,
    hypothesis_codebook,
    ordinary_static_batch,
    random_label_diagnostic_batch,
    run_training,
    static_bayesian_loss,
    structured_static_batch,
    train_frozen_stage,
    train_full_stage,
    validate_static_config,
)

__all__ = [
    "StaticBayesianTrainingConfig",
    "StaticEpisodeBatch",
    "build_parser",
    "controlled_static_batch",
    "hypothesis_codebook",
    "ordinary_static_batch",
    "random_label_diagnostic_batch",
    "run_training",
    "static_bayesian_loss",
    "structured_static_batch",
    "train_frozen_stage",
    "train_full_stage",
    "validate_static_config",
]

if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Wrote static Bayesian training artifacts to {run_training(StaticBayesianTrainingConfig(**vars(args)))}")
