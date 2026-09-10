"""Matched, bounded v3 prior pilot and implementation preflight.

Six cells: original/fixed/curriculum crossed with plain/slot. No Z supervision.
Run --preflight before --index; the gate is tied to source and architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from tfmplayground.experiments.multiregime_v3 import (
    FAMILIES,
    OriginalPrior,
    V3Config,
    group_posterior,
    oracle_probabilities,
    preserved_cpu_rng,
    sample_episode,
    support_responsibilities,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import NanoTabPFNSlotRegimeModel, SlotRegimePrediction, slot_regime_checkpoint


@dataclass(frozen=True)
class PilotConfig:
    # Mirrors ``SlotPretrainingConfig`` in ``pretrain_slot_tabpfn.py`` (the
    # grad-accumulating, matched plain/slot pretraining harness) wherever a
    # field means the same thing, so the v3 pilot trains at the same scale
    # instead of the earlier, much smaller smoke-test sizing.
    seed: int = 2402
    steps: int = 10_000
    micro_batch_size: int = 8
    accumulate_gradients: int = 4
    support_size: int = 128
    query_size: int = 32
    max_features: int = 12
    # Fixed one-hot code width; the realized per-episode group count is
    # sampled uniformly from 1..num_groups by ``task_config`` so batches stay
    # a consistent feature width while identifiability difficulty varies.
    num_groups: int = 5
    width: int = 192
    hidden: int = 768
    layers: int = 6
    heads: int = 6
    num_slots: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 2_000
    validation_interval: int = 5_000
    validation_episodes: int = 8
    device: str = "cpu"

    def generator(self):
        return V3Config(
            support_size=self.support_size,
            query_size=self.query_size,
            max_features=self.max_features,
            num_groups=self.num_groups,
        )

    def architecture(self):
        return dict(
            embedding_size=self.width,
            num_attention_heads=self.heads,
            mlp_hidden_size=self.hidden,
            num_layers=self.layers,
            num_outputs=2,
        )


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(config, kind):
    with preserved_cpu_rng(config.seed):
        backbone = NanoTabPFNModel(**config.architecture())
        initial_hash = state_hash(backbone.state_dict())
        model = (
            backbone
            if kind == "plain"
            else NanoTabPFNSlotRegimeModel(
                backbone, num_slots=config.num_slots, max_classes=2, num_slot_iterations=3, competitive_slots=True
            )
        )
    return model.to(config.device), initial_hash


def inference_architecture(model, kind, config):
    """The ``architecture`` block ``load_checkpoint_for_inference`` needs.

    v3 episodes pad inputs to ``max_features + num_groups`` columns for the
    nuisance group codes, but ``NanoTabPFNModel``'s feature encoder embeds
    each column independently and attends across whatever count is present,
    so it is not tied to that width -- real TabArena tables, with none of
    those nuisance columns, are a different but valid input.
    """
    if kind == "slot":
        return slot_regime_checkpoint(model)["architecture"]
    return {
        "num_attention_heads": config.heads,
        "embedding_size": config.width,
        "mlp_hidden_size": config.hidden,
        "num_layers": config.layers,
        "num_outputs": 2,
    }


def log_predictions(model, episodes, device):
    sx, sy, qx = (torch.cat([e.latent_inputs()[i] for e in episodes]).to(device) for i in range(3))
    prediction = model(sx, sy, qx)
    logs = (
        prediction.marginal_log_probabilities()
        if isinstance(prediction, SlotRegimePrediction)
        else F.log_softmax(prediction, dim=-1)
    )
    return logs, prediction


def mixture_probability(mode, step, steps):
    if mode == "original":
        return 0.0
    if mode == "fixed":
        return 0.5
    if mode != "curriculum":
        raise ValueError("Unknown prior mode.")
    fraction = step / steps
    return 0.5 * float(np.clip((fraction - 0.1) / 0.3, 0, 1))


def task_config(config, seed):
    rng = np.random.default_rng(seed)
    generator = replace(
        config.generator(),
        num_regimes=int(rng.choice((2, 3, 4), p=(0.5, 0.3, 0.2))),
        separation=float(rng.choice((0.0, 0.5, 1.0, 2.0, 3.0), p=(0.05, 0.15, 0.3, 0.3, 0.2))),
        gate_strength=float(rng.choice((0.25, 1.0, 2.5))),
        imbalance_ratio=float(rng.choice((0.15, 0.5, 1.0))),
    )
    active_groups = int(rng.integers(1, config.num_groups + 1))
    return generator, active_groups


def training_episode(config, original, mode, step, micro, within_batch):
    episodes_per_step = config.accumulate_gradients * config.micro_batch_size
    offset = micro * config.micro_batch_size + within_batch
    seed = int(config.seed * 1_000_000 + step * episodes_per_step + offset)
    rng = np.random.default_rng(seed)
    if rng.random() >= mixture_probability(mode, step, config.steps):
        return original.sample(seed)
    family = str(rng.choice(FAMILIES[1:], p=(0.2, 0.3, 0.2, 0.3)))
    generator, active_groups = task_config(config, seed)
    return sample_episode(generator, family=family, seed=seed, active_groups=active_groups)


def metrics(y, probability):
    y = np.asarray(y).reshape(-1)
    p = np.clip(np.asarray(probability).reshape(-1), 1e-7, 1 - 1e-7)
    nll = -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()
    # Equal-width probability calibration, with every observation assigned.
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean()) for b in range(10) if np.any(bins == b)
    )
    return {
        "log_loss": float(nll),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "ece_10": float(ece),
    }


def evaluation_bank(config, original):
    bank = {}
    for family_index, family in enumerate(FAMILIES):
        bank[family] = []
        for index in range(config.validation_episodes):
            seed = 10_000_000_000 + config.seed * 10_000 + family_index * 1000 + index
            if family == "original":
                # NumPy's legacy seed used by TabICL is a uint32.
                episode = original.sample(seed % (2**32 - 1))
            else:
                generator, active_groups = task_config(config, seed)
                episode = sample_episode(generator, family=family, seed=seed, active_groups=active_groups)
            bank[family].append(episode)
    return bank


def recovery_metrics(prediction, episode):
    """Chance-adjusted partitions; oracle assignments are references, not ARI ceilings."""
    output = {}
    rng = np.random.default_rng(episode.metadata["seed"])
    oracle = oracle_probabilities(episode)
    for split, assignments, reference, z in (
        (
            "support",
            prediction.support_attention[0].detach().cpu().numpy(),
            support_responsibilities(episode),
            episode.support_z,
        ),
        ("query", prediction.gate()[0].detach().cpu().numpy(), oracle["query_responsibilities"], episode.query_z),
    ):
        hard = assignments.argmax(-1)
        output[f"{split}_ari"] = float(adjusted_rand_score(z, hard))
        output[f"oracle_{split}_ari"] = float(adjusted_rand_score(z, reference.argmax(-1)))
        output[f"{split}_ari_permutation_null"] = float(
            np.mean([adjusted_rand_score(rng.permutation(z), hard) for _ in range(8)])
        )
        output[f"{split}_occupied_regimes"] = int(len(np.unique(z)))
        output[f"{split}_occupied_slots"] = int(len(np.unique(hard)))
    return output


@torch.no_grad()
def evaluate(model, bank, config, *, step, output):
    was_training = model.training
    model.eval()
    records = []
    summary = {}
    try:
        for family, episodes in bank.items():
            labels, probabilities = [], []
            family_rows = []
            for episode in episodes:
                logs, prediction = log_predictions(model, [episode], config.device)
                p = logs[0, :, 1].exp().cpu().numpy()
                y = episode.query_y.numpy().reshape(-1)
                result = metrics(y, p)
                row = {
                    "step": step,
                    "family": family,
                    "seed": episode.metadata["seed"],
                    "episode_hash": episode.metadata["tensor_hash"],
                    **result,
                }
                for name, values in oracle_probabilities(episode).items():
                    if name != "query_responsibilities":
                        row[f"oracle_{name}_log_loss"] = metrics(y, values)["log_loss"]
                if family == "persistent":
                    scrambled_x = episode.query_x.clone()
                    # Reassign query codes to different group identities while
                    # holding all covariates, support labels, and Y fixed.
                    scrambled_x[:, :, config.max_features :] = torch.roll(
                        scrambled_x[:, :, config.max_features :], 1, dims=-1
                    )
                    scrambled = replace(episode, query_x=scrambled_x)
                    scrambled_logs, _ = log_predictions(model, [scrambled], config.device)
                    row["scrambled_group_log_loss"] = metrics(y, scrambled_logs[0, :, 1].exp().cpu().numpy())[
                        "log_loss"
                    ]
                    row["group_information_gain"] = row["scrambled_group_log_loss"] - row["log_loss"]
                row["per_regime"] = {
                    str(k): metrics(y[episode.query_z == k], p[episode.query_z == k])
                    for k in np.unique(episode.query_z)
                }
                if isinstance(prediction, SlotRegimePrediction):
                    row["slot_gate_entropy"] = float(prediction.gate_entropy().mean())
                    row["slot_disagreement"] = float(prediction.slot_disagreement().mean())
                    if family not in ("original", "shared_rule"):
                        row.update(recovery_metrics(prediction, episode))
                labels.append(y)
                probabilities.append(p)
                records.append(row)
                family_rows.append(row)
            aggregate = metrics(np.concatenate(labels), np.concatenate(probabilities))
            aggregate["episode_log_loss_se"] = (
                float(np.std([r["log_loss"] for r in family_rows], ddof=1) / math.sqrt(len(family_rows)))
                if len(family_rows) > 1
                else None
            )
            for key in (
                "oracle_marginal_log_loss",
                "oracle_without_group_evidence_log_loss",
                "oracle_privileged_regime_log_loss",
                "group_information_gain",
                "slot_gate_entropy",
                "slot_disagreement",
                "support_ari",
                "query_ari",
                "oracle_support_ari",
                "oracle_query_ari",
                "support_ari_permutation_null",
                "query_ari_permutation_null",
            ):
                if key in family_rows[0]:
                    aggregate[key] = float(np.mean([r[key] for r in family_rows]))
            summary[family] = aggregate
    finally:
        model.train(was_training)
    with (output / "evaluation.jsonl").open("a") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return summary


def preflight(config, output):
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    generator = config.generator()
    for family in FAMILIES[1:]:
        episode = sample_episode(generator, family=family, seed=67)
        repeated = sample_episode(generator, family=family, seed=67)
        extended = sample_episode(replace(generator, query_size=generator.query_size + 3), family=family, seed=67)
        checks[f"{family}_reproducible"] = episode.tensor_hash() == repeated.tensor_hash()
        checks[f"{family}_query_count_independent"] = (
            torch.equal(episode.support_x, extended.support_x)
            and torch.equal(episode.support_y, extended.support_y)
            and episode.metadata["mechanism_hash"] == extended.metadata["mechanism_hash"]
        )
        oracle = oracle_probabilities(episode)
        relabeled = replace(episode, query_y=1 - episode.query_y, query_z=(episode.query_z + 1) % generator.num_regimes)
        checks[f"{family}_oracle_no_query_labels_or_z"] = np.array_equal(
            oracle["marginal"], oracle_probabilities(relabeled)["marginal"]
        )
        checks[f"{family}_finite"] = bool(np.isfinite(oracle["marginal"]).all())
        if family == "persistent":
            groups = np.concatenate((episode.support_groups, episode.query_groups))
            z = np.concatenate((episode.support_z, episode.query_z))
            checks["persistent_assignments"] = all(len(np.unique(z[groups == g])) == 1 for g in np.unique(groups))
    opposite = torch.linspace(-50, 50, 100)
    checks["opposite_logits_cancel"] = bool(
        torch.allclose(0.5 * (opposite.sigmoid() + (-opposite).sigmoid()), torch.full_like(opposite, 0.5))
    )
    posterior = group_posterior(
        np.array([0.5, 0.5]), np.tile([0.9, 0.1], (8, 1)), np.ones(8), np.zeros(8, dtype=int), 2
    )
    checks["group_evidence_accumulates"] = bool(posterior[0, 0] > 0.999 and posterior[1, 0] == 0.5)
    original = OriginalPrior(generator)
    first, second = original.sample(109), original.sample(109)
    checks["original_reproducible"] = first.tensor_hash() == second.tensor_hash()
    numerical = {}
    for kind in ("plain", "slot"):
        model, backbone_hash = build_model(config, kind)
        examples = [first, sample_episode(generator, family="persistent", seed=11)]
        model.train()
        logs, _ = log_predictions(model, examples, config.device)
        targets = torch.cat([e.query_y for e in examples]).to(config.device)
        loss = F.nll_loss(logs.reshape(-1, 2), targets.reshape(-1))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        checks[f"{kind}_finite_forward_backward"] = bool(
            torch.isfinite(loss)
            and grads
            and all(torch.isfinite(g).all() for g in grads)
            and any(g.abs().sum() > 0 for g in grads)
        )
        numerical[kind] = {
            "loss": float(loss.detach()),
            "backbone_hash": backbone_hash,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model
    checks["matched_initial_backbone"] = numerical["plain"]["backbone_hash"] == numerical["slot"]["backbone_hash"]
    bank = evaluation_bank(config, original)
    oracle_headroom = {}
    for family, episodes in bank.items():
        if family != "original":
            oracle_headroom[family] = {
                name: float(
                    np.mean([metrics(e.query_y.numpy(), oracle_probabilities(e)[name])["log_loss"] for e in episodes])
                )
                for name in ("marginal", "without_group_evidence", "privileged_regime")
            }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "source_hash": source_hash(),
        "architecture": config.architecture(),
        "generator": asdict(generator),
        "num_slots": config.num_slots,
        "device": config.device,
        "numerical": numerical,
        "oracle_headroom": oracle_headroom,
    }
    (output / "execution_gate.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)
    if not payload["passed"]:
        raise RuntimeError("V3 implementation gate failed.")
    return payload


def run(config, *, index, output, gate_path):
    if index not in range(6):
        raise ValueError("Pilot index must be 0-5.")
    if (
        config.steps < 1
        or config.micro_batch_size < 1
        or config.accumulate_gradients < 1
        or config.validation_episodes < 1
    ):
        raise ValueError("Step, batch, and validation counts must be positive.")
    gate = json.loads(gate_path.read_text())
    if not (
        gate.get("passed")
        and gate["source_hash"] == source_hash()
        and gate["architecture"] == config.architecture()
        and gate["generator"] == asdict(config.generator())
        and gate["num_slots"] == config.num_slots
    ):
        raise ValueError("A passing v3 gate matching source, architecture, and generator is required.")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback.")
    output.mkdir(parents=True, exist_ok=False)
    kind = ("plain", "slot")[index % 2]
    mode = ("original", "fixed", "curriculum")[index // 2]
    model, initial_hash = build_model(config, kind)
    torch.manual_seed(config.seed + 31)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    floor = config.min_learning_rate / config.learning_rate

    def lr_factor(step):
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        fraction = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    metadata = {
        "config": asdict(config),
        "index": index,
        "kind": kind,
        "mode": mode,
        "source_hash": source_hash(),
        "initial_backbone_hash": initial_hash,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gate": gate,
        "status": "running",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    original = OriginalPrior(config.generator())
    bank = evaluation_bank(config, original)
    metadata["evaluation_bank_hashes"] = {family: [e.tensor_hash() for e in eps] for family, eps in bank.items()}
    started = time.monotonic()
    initial = evaluate(model, bank, config, step=0, output=output)
    print(json.dumps({"step": 0, "kind": kind, "mode": mode, "validation": initial}), flush=True)
    (output / "initial_validation.json").write_text(json.dumps(initial, indent=2) + "\n")
    composition = Counter()
    stream = hashlib.sha256()
    with (output / "training.jsonl").open("w") as history, (output / "episodes.jsonl").open("w") as provenance:
        for step in range(1, config.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for micro in range(config.accumulate_gradients):
                episodes = [
                    training_episode(config, original, mode, step, micro, i) for i in range(config.micro_batch_size)
                ]
                for episode in episodes:
                    composition[episode.metadata["family"]] += 1
                    stream.update(episode.tensor_hash().encode())
                    provenance.write(json.dumps({"step": step, **episode.metadata}) + "\n")
                logs, _ = log_predictions(model, episodes, config.device)
                targets = torch.cat([e.query_y for e in episodes]).to(config.device)
                loss = F.nll_loss(logs.reshape(-1, 2), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}.")
                (loss / config.accumulate_gradients).backward()
                loss_total += float(loss.detach()) / config.accumulate_gradients
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            row = {
                "step": step,
                "loss": loss_total,
                "gradient_norm": float(norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
            history.write(json.dumps(row) + "\n")
            if step % 25 == 0 or step == 1:
                print(json.dumps(row), flush=True)
                history.flush()
                provenance.flush()
            if step % config.validation_interval == 0 or step == config.steps:
                validation = evaluate(model, bank, config, step=step, output=output)
                checkpoint = {
                    "model": model.state_dict(),
                    "architecture": inference_architecture(model, kind, config),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "metadata": metadata,
                    "validation": validation,
                    "training_stream_hash": stream.hexdigest(),
                    "composition": dict(composition),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
                temporary = output / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "checkpoint.pth")
                print(json.dumps({"step": step, "validation": validation}), flush=True)
    result = {
        **metadata,
        "status": "complete",
        "initial_validation": initial,
        "validation": validation,
        "composition": dict(composition),
        "training_stream_hash": stream.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int)
    defaults = PilotConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    config = PilotConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    torch.set_num_threads(1)
    if args.preflight:
        preflight(config, args.output)
    elif args.gate is None or args.index is None:
        parser.error("Training requires --index and --gate.")
    else:
        run(config, index=args.index, output=args.output, gate_path=args.gate)


if __name__ == "__main__":
    main()
