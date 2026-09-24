#!/usr/bin/env python3
"""Re-score finetuned TabPFN rows whose support is too small to stratify."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v3 import OriginalPrior
from tfmplayground.experiments.pretrain_multiregime_v3 import PilotConfig, evaluation_bank, metrics


CHECKPOINTS = {
    "v2.2": None,
    "v2.6": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "v3": "/users/k23139234/repo/TFM-Playground/checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
}


def positive_probability(model, query_x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(query_x))
    classes = list(getattr(model, "classes_", [0, 1]))
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(query_x))


def build(version: str):
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    model_version = {"v2.2": ModelVersion.V2, "v2.6": ModelVersion.V2_6, "v3": ModelVersion.V3}[version]
    checkpoint = CHECKPOINTS[version]
    extra = {"model_path": checkpoint} if checkpoint else {}
    return FinetunedTabPFNClassifier(
        device="cuda",
        epochs=5,
        learning_rate=1e-5,
        weight_decay=0.01,
        validation_split_ratio=0.0,
        n_finetune_ctx_plus_query_samples=128,
        finetune_ctx_query_split_ratio=0.2,
        random_state=0,
        early_stopping=True,
        early_stopping_patience=2,
        validation_frequency=1,
        min_delta=1e-4,
        grad_clip_value=1.0,
        use_lr_scheduler=True,
        lr_warmup_only=False,
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_final_inference=8,
        use_activation_checkpointing=True,
        save_checkpoint_interval=None,
        use_fixed_preprocessing_seed=True,
        extra_classifier_kwargs={
            **extra,
            "n_estimators": 8,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "show_progress_bar": False,
        },
        eval_metric="log_loss",
        model_version=model_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=4)
    args = parser.parse_args()
    if importlib.metadata.version("tabicl") != "2.1.1":
        raise RuntimeError("This edge-row repair requires tabicl==2.1.1.")

    metadata = json.loads(args.pilot_config.read_text())
    config = PilotConfig(**{**metadata["config"], "device": "cpu"})
    episode = evaluation_bank(config, OriginalPrior(config.generator()))["original"][args.episode]
    support_x = episode.support_x[0].numpy().astype(np.float32)
    support_y = episode.support_y[0].numpy().reshape(-1).astype(np.int64)
    query_x = episode.query_x[0].numpy().astype(np.float32)
    query_y = episode.query_y[0].numpy().reshape(-1).astype(np.int64)
    rows = []
    for version in ("v2.2", "v2.6", "v3"):
        classifier = build(version)
        classifier.fit(support_x, support_y)
        probability = positive_probability(classifier, query_x)
        row = {
            "model": f"tabpfn-{version}-finetuned",
            "family": "original",
            "episode": args.episode,
            "episode_hash": episode.tensor_hash(),
            "query_rows": len(query_y),
            **metrics(query_y, probability),
            "auc": float(roc_auc_score(query_y, probability)),
            "validation_split_ratio_override": 0.0,
        }
        rows.append(row)
        del classifier
        torch.cuda.empty_cache()
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
