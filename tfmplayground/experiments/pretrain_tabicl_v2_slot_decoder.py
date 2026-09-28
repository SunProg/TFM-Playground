"""Train a slot decoder with TabICLv2's released pretraining machinery.

This entry point subclasses the upstream ``tabicl.train.Trainer`` only to
replace the model head, select the trainable parameters, and add checkpoint
metadata. Prior generation, training, optimization implementations, schedule,
gradient accumulation, and distributed execution stay upstream.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tabicl._model.attention import set_flash_attn3_enabled
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._run import Trainer as TabICLTrainer
from tabicl.train._train_config import build_parser as build_tabicl_parser
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP

from tfmplayground.models.tabicl_slot_decoder import (
    install_slot_decoder,
    install_tabpfnv3_decoder,
    load_released_tabicl_model,
)


def _stage3_defaults(parser: argparse.ArgumentParser) -> None:
    """Use the official TabICLv2 classifier Stage 3 recipe as the run default."""
    parser.set_defaults(
        wandb_log=False,
        dtype="float32",
        np_seed=2402,
        torch_seed=2402,
        max_steps=10_000,
        batch_size=64,
        micro_batch_size=1,
        lr=2e-5,
        muon=True,
        beta1=0.9,
        weight_decay=0.01,
        use_cautious_wd=False,
        scheduler="cosine_with_restarts",
        warmup_proportion=0.01,
        cosine_num_cycles=1,
        cosine_amplitude_decay=1.0,
        cosine_lr_end=1e-7,
        gradient_clipping=1.0,
        prior_type="graph_scm",
        prior_device="cpu",
        n_jobs=16,
        batch_size_per_gp=1,
        min_features=1,
        max_features=100,
        max_classes=10,
        min_seq_len=400,
        max_seq_len=60_000,
        log_seq_len=True,
        seq_len_per_gp=True,
        min_train_size=0.79,
        max_train_size=0.81,
        graph_noise=False,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
        min_n_nodes=2,
        max_n_nodes=32,
        cauchy_dag_offset=0.0,
        embed_dim=128,
        col_num_blocks=3,
        col_nhead=8,
        col_num_inds=128,
        col_affine=False,
        col_feature_group="same",
        col_feature_group_size=3,
        col_target_aware=True,
        col_ssmax=True,
        row_num_blocks=3,
        row_nhead=8,
        row_num_cls=4,
        row_rope_interleaved=False,
        icl_num_blocks=12,
        icl_nhead=8,
        icl_ssmax=True,
        ssmax_type="qassmax-mlp-elementwise",
        ff_factor=2,
        norm_first=True,
        zero_init=False,
        use_flash_attn3=True,
        recompute=False,
        save_temp_every=50,
        save_perm_every=1_000,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_tabicl_parser()
    _stage3_defaults(parser)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Optional local released TabICLv2 classifier checkpoint; defaults to TabICL's official download/cache.",
    )
    parser.add_argument(
        "--checkpoint-version",
        default="tabicl-classifier-v2-20260212.ckpt",
        help="Released TabICL classifier checkpoint version to load when --base-checkpoint is omitted.",
    )
    parser.add_argument("--head-kind", choices=("slot", "mlp", "tabpfnv3"), default="slot")
    parser.add_argument(
        "--mlp-init",
        choices=("pretrained", "random"),
        default="pretrained",
        help="Initialize the MLP decoder from released TabICLv2 weights or reset it before training.",
    )
    parser.add_argument("--slot-count", type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument("--fixed-seq-len", type=int, default=None, help="Use one fixed sequence length, as in classifier Stage 1.")
    return parser


class TabICLSlotDecoderTrainer(TabICLTrainer):
    """Upstream trainer with a checkpoint-compatible, decoder-only model build."""

    def build_model(self) -> None:
        model, model_config, checkpoint_path = load_released_tabicl_model(
            checkpoint_path=self.config.base_checkpoint,
            checkpoint_version=self.config.checkpoint_version,
            device="cpu",
        )
        if self.config.regression_method is not None:
            raise ValueError("The slot decoder experiment supports classification checkpoints only.")
        self.regression = False
        set_flash_attn3_enabled(self.config.use_flash_attn3)
        self._disable_cudnn_sdp = hasattr(torch.backends.cuda, "enable_cudnn_sdp") and "cuda" in self.config.device
        if model.max_classes != self.config.max_classes:
            raise ValueError(
                f"Synthetic prior max_classes={self.config.max_classes} does not match the released model's "
                f"max_classes={model.max_classes}."
            )

        self.model_config = dict(model_config)
        self.base_checkpoint_path = checkpoint_path
        self.base_checkpoint_version = self.config.checkpoint_version
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.base_checkpoint_sha256 = digest.hexdigest()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if self.config.head_kind == "slot":
            slot_decoder = install_slot_decoder(model, self.config.slot_count, seed=self.config.torch_seed)
            for parameter in slot_decoder.parameters():
                parameter.requires_grad_(True)
            self.trainable_module_name = "icl_predictor.slot_decoder"
            self.decoder_initialization = "random_slot"
        elif self.config.head_kind == "mlp":
            if self.config.mlp_init == "random":
                # Reset only the prediction MLP while preserving the released
                # TabICLv2 backbone and the caller's global RNG state.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(self.config.torch_seed)
                    for module in model.icl_predictor.decoder.modules():
                        if isinstance(module, torch.nn.Linear):
                            module.reset_parameters()
                self.decoder_initialization = "random_mlp"
            else:
                self.decoder_initialization = "pretrained_tabicl_v2_mlp"
            for parameter in model.icl_predictor.decoder.parameters():
                parameter.requires_grad_(True)
            self.trainable_module_name = "icl_predictor.decoder"
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.config.torch_seed)
                tabpfnv3_decoder = install_tabpfnv3_decoder(model)
            for parameter in tabpfnv3_decoder.parameters():
                parameter.requires_grad_(True)
            self.trainable_module_name = "icl_predictor.tabpfnv3_decoder"
            self.decoder_initialization = "random_tabpfnv3_many_class_decoder"

        model.to(device=self.config.device)
        model.train()

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable decoder parameters were created.")
        self.trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        if self.master_process:
            print(
                f"Loaded TabICLv2 from {checkpoint_path}; training {self.trainable_parameter_count:,} "
                f"parameters in {self.trainable_module_name}."
            )

        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def _decoder_module(self):
        predictor = self.raw_model.icl_predictor
        if self.config.head_kind == "slot":
            return predictor.slot_decoder
        if self.config.head_kind == "tabpfnv3":
            return predictor.tabpfnv3_decoder
        return predictor.decoder

    def load_checkpoint(self) -> None:
        """Restore decoder-only snapshots, while accepting older full-model snapshots."""
        checkpoint_path = self.config.checkpoint_path or self.get_latest_checkpoint()
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        metadata = checkpoint.get("slot_decoder")
        if not isinstance(metadata, dict):
            raise ValueError(f"Resume checkpoint lacks slot-decoder metadata: {checkpoint_path}")
        if metadata.get("head_kind") != self.config.head_kind:
            raise ValueError("Resume checkpoint head_kind differs from the requested training run.")
        saved_initialization = metadata.get(
            "decoder_initialization",
            "pretrained_tabicl_v2_mlp" if self.config.head_kind == "mlp" else "random_slot",
        )
        if saved_initialization != self.decoder_initialization:
            raise ValueError("Resume checkpoint decoder initialization differs from the requested training run.")
        if self.config.head_kind == "slot" and int(metadata.get("num_slots", -1)) != self.config.slot_count:
            raise ValueError("Resume checkpoint slot count differs from --slot-count.")
        if int(metadata.get("seed", -1)) != self.config.torch_seed:
            raise ValueError("Resume checkpoint seed differs from --torch_seed.")
        if metadata.get("base_checkpoint_sha256") != self.base_checkpoint_sha256:
            raise ValueError("Resume checkpoint was initialized from a different TabICLv2 checkpoint.")

        if checkpoint.get("state_dict_format") == "decoder_only_v1":
            self._decoder_module().load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            # Backward compatibility for full-model checkpoints produced before decoder-only saving.
            self.raw_model.load_state_dict(checkpoint["state_dict"], strict=True)

        if self.config.only_load_model:
            print("Only loading model weights.")
            return
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.curr_step = int(checkpoint["curr_step"])
        print(f"Resuming training at step {self.curr_step}")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "format": "tabicl-slot-decoder-v1",
            "head_kind": self.config.head_kind,
            "decoder_initialization": self.decoder_initialization,
            "mlp_init": self.config.mlp_init if self.config.head_kind == "mlp" else None,
            "tabpfn_version": (
                importlib.metadata.version("tabpfn") if self.config.head_kind == "tabpfnv3" else None
            ),
            "num_slots": self.config.slot_count if self.config.head_kind == "slot" else None,
            "tabpfnv3_decoder_configuration": (
                {
                    "implementation": "tabpfn.architectures.tabpfn_v3.ManyClassDecoder",
                    "head_dim": 64,
                    "num_heads": 6,
                    "initialization": "random",
                }
                if self.config.head_kind == "tabpfnv3"
                else None
            ),
            "seed": self.config.torch_seed,
            "tabicl_version": importlib.metadata.version("tabicl"),
            "base_checkpoint_version": self.base_checkpoint_version,
            "base_checkpoint_path": str(self.base_checkpoint_path),
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "trainable_module": self.trainable_module_name,
            "slot_configuration": {
                "num_slots": self.config.slot_count if self.config.head_kind == "slot" else None,
                "num_iterations": 3 if self.config.head_kind == "slot" else None,
                "competitive": True if self.config.head_kind == "slot" else None,
                "slot_size": self._decoder_module().d_model if self.config.head_kind == "slot" else None,
                "num_heads": (
                    self._decoder_module().slot_attention.num_heads
                    if self.config.head_kind == "slot"
                    else None
                ),
            },
            "prior": {
                "type": self.config.prior_type,
                "min_features": self.config.min_features,
                "max_features": self.config.max_features,
                "max_classes": self.config.max_classes,
                "min_seq_len": self.config.min_seq_len,
                "max_seq_len": self.config.max_seq_len,
                "min_train_size": self.config.min_train_size,
                "max_train_size": self.config.max_train_size,
                "graph_scm": asdict(PriorConfig.from_args(self.config)),
            },
            "schedule": {
                "max_steps": self.config.max_steps,
                "batch_size": self.config.batch_size,
                "micro_batch_size": self.config.micro_batch_size,
                "optimizer": "Muon" if self.config.muon else "AdamW",
                "lr": self.config.lr,
                "scheduler": self.config.scheduler,
                "warmup_proportion": self.config.warmup_proportion,
                "cosine_num_cycles": self.config.cosine_num_cycles,
                "cosine_amplitude_decay": self.config.cosine_amplitude_decay,
                "cosine_lr_end": self.config.cosine_lr_end,
                "gradient_clipping": self.config.gradient_clipping,
            },
        }

    def save_checkpoint(self, name: str) -> None:
        """Save only the trainable decoder, with state needed to resume training."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(self.config.checkpoint_dir) / name
        decoder_state = {
            key: value.detach().cpu()
            for key, value in self._decoder_module().state_dict().items()
        }
        checkpoint = {
            "config": self.model_config,
            "state_dict_format": "decoder_only_v1",
            "state_dict": decoder_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
            "slot_decoder": self._checkpoint_metadata(),
        }
        torch.save(checkpoint, checkpoint_path)

    def run_batch(self, batch: Any) -> dict[str, float]:
        """Keep upstream batch logic and attach the router's diagnostic metrics."""
        results = super().run_batch(batch)
        if self.config.head_kind == "slot":
            diagnostics = self.raw_model.icl_predictor.last_diagnostics
            if diagnostics:
                occupancy = diagnostics["slot_occupancy"]
                results["slot_occupancy_std"] = float(occupancy.std(dim=-1, unbiased=False).mean().item())
                results["slot_assignment_entropy"] = float(diagnostics["slot_assignment_entropy"].mean().item())
                results["query_slot_entropy"] = float(diagnostics["query_slot_entropy"].mean().item())
        if self.master_process and self.config.checkpoint_dir:
            metrics_path = Path(self.config.checkpoint_dir) / "metrics.jsonl"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps({"step": self.curr_step + 1, **results}) + "\n")
        return results


def run_training(config: argparse.Namespace) -> TabICLSlotDecoderTrainer:
    fixed_seq_len = getattr(config, "fixed_seq_len", None)
    if fixed_seq_len is not None:
        if fixed_seq_len < 1:
            raise ValueError("--fixed-seq-len must be positive.")
        config.min_seq_len = None
        config.max_seq_len = fixed_seq_len
        config.log_seq_len = False
        delattr(config, "fixed_seq_len")
    if config.head_kind == "mlp" and config.slot_count != 4:
        raise ValueError("The matched MLP control does not use --slot-count; leave it at the default 4.")
    if config.head_kind != "mlp" and config.mlp_init != "pretrained":
        raise ValueError("--mlp-init applies only to --head-kind mlp.")
    trainer = TabICLSlotDecoderTrainer(config)
    trainer.train()
    return trainer


def main(argv: list[str] | None = None) -> None:
    with contextlib.suppress(RuntimeError):
        set_start_method("spawn")
    run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
