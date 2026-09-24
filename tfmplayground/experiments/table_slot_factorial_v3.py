"""Three-switch factorial inside TableSlotModel; never builds causal_routing.

Switches: (1) blind query content, (2) held-out support prediction, (3) early
soft routing/usage regularization. Baseline is the unchanged query-NLL head.
Run directly to avoid eagerly loading unrelated package registries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    for name in ("tfmplayground", "tfmplayground.experiments", "tfmplayground.models"):
        module = ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules.setdefault(name, module)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tfmplayground.experiments.multiregime_v3 import FAMILIES, OriginalPrior, sample_episode
from tfmplayground.experiments.pretrain_multiregime_v3 import (
    PilotConfig,
    source_hash,
    state_hash,
    task_config,
    training_episode,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import SlotRegimePrediction, slot_mi_loss, support_reconstruction_loss
from tfmplayground.models.table_slot import QUERY_ROUTING_MODES, TableSlotModel

COMBINATIONS = ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
SCOPES = ("data", "cell_and_data", "cell")
FIXES = {
    "1": "blind query decoder content; labelled support-derived slots retained",
    "2": "held-out support posterior-responsibility gate and expert supervision",
    "3": "temporary temperature and episode-local anti-starvation loss",
}

# Opt-in pilot axis.  Keep it outside ``configurations()`` so the submitted
# 75-cell array retains its historical index mapping.
SLOT_USEFUL_LOSS_MODES = ("none", "reconstruction", "reconstruction_mi")
SLOT_USEFUL_RECONSTRUCTION_WEIGHT = 1.0
SLOT_USEFUL_MI_WEIGHT = 0.05


def configurations():
    cells = []
    for prior in ("original", "fixed", "curriculum"):
        for scope in SCOPES:
            for fixes in COMBINATIONS:
                cells.append(dict(prior=prior, scope=scope, fixes=fixes, kind="table_slot_head"))
        cells.append(dict(prior=prior, scope=None, fixes=(), kind="plain"))
    return [dict(index=i, **cell) for i, cell in enumerate(cells)]


def build_model(config, cell):
    # Identical backbone initialization and slot initialization within each
    # scope, regardless of switches or prior. No additional learned modules.
    useful_mode = cell.get("slot_useful_loss", "none")
    if useful_mode not in SLOT_USEFUL_LOSS_MODES:
        raise ValueError(f"slot_useful_loss must be one of {SLOT_USEFUL_LOSS_MODES}, got {useful_mode!r}.")
    if useful_mode != "none" and cell["kind"] != "table_slot_head":
        raise ValueError("slot-useful reconstruction is implemented only for TableSlotModel head mode.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        backbone = NanoTabPFNModel(**config.architecture())
        initial_hash = state_hash(backbone.state_dict())
        model = (
            backbone
            if cell["kind"] == "plain"
            else TableSlotModel(
                backbone,
                mode="head",
                scope=cell["scope"],
                num_slots=config.num_slots,
                layer_indices=tuple(range(max(0, config.layers - 3), config.layers)),
                query_content_mode="blind" if 1 in cell["fixes"] else "labelled",
                query_routing_mode=cell.get("query_routing_mode", "decoder"),
                decoder_interaction=cell.get("decoder_interaction", "full"),
                slot_composition=cell.get("slot_composition", "shared"),
                # The support loss must train the same decoder alpha that
                # routes query rows; attention weighting would supervise a
                # second, unrelated routing quantity.
                reconstruction_mixture="alpha" if useful_mode != "none" else "attention",
            )
        )
    if cell["kind"] != "plain":
        assert type(model) is TableSlotModel and model.scope == cell["scope"] and model.mode == "head"
    else:
        assert type(model) is NanoTabPFNModel
    return model.to(config.device), initial_hash


def architecture(model, config):
    result = config.architecture()
    if type(model) is TableSlotModel:
        result.update(
            model_kind="table_slot_head",
            table_slot_scope=model.scope,
            num_slots=model.num_slots,
            max_classes=2,
            num_slot_iterations=3,
            slot_layer_indices=model.layer_indices,
            query_content_mode=model.query_content_mode,
            query_routing_mode=model.query_routing_mode,
            decoder_interaction=model.decoder_interaction,
            reconstruction_mixture=model.reconstruction_mixture,
            slot_composition=model.slot_composition,
        )
    return result


def stack(episodes, device):
    return tuple(
        torch.cat([getattr(e, field) for e in episodes]).to(device)
        for field in ("support_x", "support_y", "query_x", "query_y")
    )


def log_probabilities(prediction):
    return (
        prediction.marginal_log_probabilities()
        if isinstance(prediction, SlotRegimePrediction)
        else prediction.log_softmax(-1)
    )


def early_schedule(step, enabled, duration):
    fraction = max(0.0, 1.0 - step / max(1, duration)) if enabled else 0.0
    return 1.0 + fraction, 0.01 * fraction


def usage_loss(log_gate):
    # KL(uniform || episode mean query gates), averaged over episodes. Slot
    # identities are never combined across different episodes or forwards.
    usage = log_gate.exp().mean(1).clamp_min(1e-12)
    return (-math.log(usage.shape[-1]) - usage.log()).mean()


def heldout_partition(sx, sy, *, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(sx.shape[1], generator=generator).to(sx.device)
    heldout = max(1, sx.shape[1] // 4)
    context, targets = order[heldout:], order[:heldout]
    return sx[:, context], sy[:, context], sx[:, targets], sy[:, targets]


def responsibility_losses(prediction, labels):
    # This is the EM gradient of held-out marginal NLL, NOT an extra
    # identifiability guarantee. Only the targets here see held-out labels.
    labels = labels.reshape(*prediction.log_gate.shape[:2]).long()
    expert_logp = prediction.slot_logits.log_softmax(-1)
    target_logp = expert_logp.gather(-1, labels[..., None, None].expand(-1, -1, expert_logp.shape[2], 1)).squeeze(-1)
    posterior = (prediction.log_gate + target_logp).softmax(-1).detach()
    gate = -(posterior * prediction.log_gate).sum(-1).mean()
    expert = -(posterior * target_logp).sum(-1).mean()
    return gate, expert


def useful_slot_loss(prediction, support_y, mode):
    """Give each support assignment predictive responsibility.

    Query mixture NLL alone permits redundant slot experts and a uniform gate.
    Support reconstruction makes the slot assignment accountable for the
    labels it claims.  ``reconstruction_mi`` additionally uses the existing
    balanced-sharpness closure; it is kept separate so reconstruction-only can
    be evaluated without forcing artificial balance.
    """
    if mode not in SLOT_USEFUL_LOSS_MODES:
        raise ValueError(f"slot_useful_loss must be one of {SLOT_USEFUL_LOSS_MODES}, got {mode!r}.")
    zero = prediction.new_zeros(()) if torch.is_tensor(prediction) else prediction.slot_logits.new_zeros(())
    if mode == "none":
        return zero, zero
    if torch.is_tensor(prediction):
        raise ValueError("slot-useful reconstruction requires a TableSlotModel prediction.")
    reconstruction = support_reconstruction_loss(prediction, support_y)
    mi = slot_mi_loss(prediction.support_attention) if mode == "reconstruction_mi" else zero
    return reconstruction, mi


def training_step(model, tensors, cell, step, config, optimizer):
    sx, sy, qx, qy = tensors
    sy = sy.float()
    model.train()
    # Extra blind/held-out passes cannot advance the next step's RNG stream.
    torch.manual_seed(config.seed * 1_000_000 + step)
    optimizer.zero_grad(set_to_none=True)
    temperature, usage_weight = early_schedule(step, 3 in cell["fixes"], config.warmup_steps)
    useful_mode = cell.get("slot_useful_loss", "none")
    model_kwargs = {"gate_temperature": temperature} if cell["kind"] != "plain" else {}
    if useful_mode != "none":
        model_kwargs["reconstruct_support"] = True
    prediction = model(sx, sy, qx, **model_kwargs)
    logs = log_probabilities(prediction)
    query_nll = F.nll_loss(logs.flatten(0, 1), qy.reshape(-1).long())
    usage = usage_loss(prediction.log_gate) if usage_weight else query_nll.new_zeros(())
    reconstruction, slot_mi = useful_slot_loss(prediction, sy, useful_mode)
    primary = (
        query_nll
        + usage_weight * usage
        + SLOT_USEFUL_RECONSTRUCTION_WEIGHT * reconstruction
        + SLOT_USEFUL_MI_WEIGHT * slot_mi
    )
    if not torch.isfinite(primary):
        raise RuntimeError(f"non-finite primary loss at step {step}")
    primary.backward()
    values = dict(
        query_nll=float(query_nll.detach()),
        usage_loss=float(usage.detach()),
        usage_weight=usage_weight,
        temperature=temperature,
        heldout_gate=0.0,
        heldout_expert=0.0,
        heldout_nll=None,
        slot_reconstruction_nll=float(reconstruction.detach()),
        slot_mi_loss=float(slot_mi.detach()),
    )
    del primary, prediction, logs, query_nll, usage, reconstruction, slot_mi
    if 2 in cell["fixes"]:
        # A separate forward builds slots from the 75% context only. Neither
        # true query labels nor the held-out labels reach any model input.
        cx, cy, hx, hy = heldout_partition(sx, sy, seed=config.seed * 10_000_000 + step)
        torch.manual_seed(config.seed * 1_000_000 + step + 100_000_000)
        heldout_prediction = model(cx, cy, hx, gate_temperature=temperature)
        gate, expert = responsibility_losses(heldout_prediction, hy)
        with torch.no_grad():
            values["heldout_nll"] = float(
                F.nll_loss(heldout_prediction.marginal_log_probabilities().flatten(0, 1), hy.reshape(-1).long())
            )
        auxiliary = gate + expert  # unit held-out weight; no duplicate held-out marginal term
        if not torch.isfinite(auxiliary):
            raise RuntimeError(f"non-finite held-out loss at step {step}")
        auxiliary.backward()
        values.update(heldout_gate=float(gate.detach()), heldout_expert=float(expert.detach()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    values["gradient_norm"] = float(norm)
    return values


def make_bank(config, phase, count, eval_query_size):
    evaluation_config = replace(config, query_size=eval_query_size)
    original = OriginalPrior(evaluation_config.generator())
    base = {"validation": 2_000_000, "test": 7_000_000}[phase] + config.seed * 10_000
    bank = {}
    for fi, family in enumerate(FAMILIES):
        bank[family] = []
        for i in range(count):
            seed = base + fi * 1000 + i
            if family == "original":
                episode = original.sample(seed)
            else:
                generator, groups = task_config(evaluation_config, seed)
                episode = sample_episode(generator, family=family, seed=seed, active_groups=groups)
            bank[family].append(episode)
    return bank


def binary_metrics(logs, labels, mask=None):
    if mask is not None:
        logs, labels = logs[mask], labels[mask]
    if not len(labels):
        return dict(nll=None, accuracy=None, auc=None, n=0)
    array_y = labels.cpu().numpy()
    probabilities = logs[:, 1].exp().cpu().numpy()
    return dict(
        nll=float(F.nll_loss(logs, labels)),
        accuracy=float((logs.argmax(-1) == labels).float().mean()),
        auc=float(roc_auc_score(array_y, probabilities)) if len(np.unique(array_y)) == 2 else None,
        n=len(labels),
    )


def aggregate(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    output = {}
    for key in keys:
        if key in ("episode_index", "tensor_hash"):
            continue
        valid = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] is not None]
        if valid:
            output[key] = sum(valid) if key.endswith("_n") or key == "n" else float(np.mean(valid))
        if key.endswith("auc"):
            output[key + "_episodes"] = len(valid)
    return output


@torch.no_grad()
def evaluate(model, bank, device):
    model.eval()
    rows_by_family = {}
    for family, episodes in bank.items():
        rows = []
        for ei, episode in enumerate(episodes):
            sx, sy, qx, qy = stack([episode], device)
            prediction = model(sx, sy.float(), qx)  # evaluation always uses temperature 1
            logs = log_probabilities(prediction)[0]
            labels = qy.reshape(-1).long()
            row = dict(episode_index=ei, tensor_hash=episode.tensor_hash(), **binary_metrics(logs, labels))
            if family != "original":
                counts = np.bincount(episode.support_z, minlength=int(episode.metadata["num_regimes"]))
                # All tied largest regimes are major; every other regime is minor.
                major_z = np.flatnonzero(counts == counts.max())
                major = torch.as_tensor(np.isin(episode.query_z, major_z), device=device)
                for label, mask in (("major", major), ("minor", ~major)):
                    row.update({label + "_" + k: v for k, v in binary_metrics(logs, labels, mask).items()})
            if isinstance(prediction, SlotRegimePrediction):
                g = prediction.log_gate.exp()[0]
                k = g.shape[-1]
                usage = g.mean(0)
                hard = F.one_hot(g.argmax(-1), k).float().mean(0)
                expert = prediction.slot_logits.log_softmax(-1)[0]
                mean_logs = torch.logsumexp(usage.clamp_min(1e-12).log()[None, :, None] + expert, dim=1)
                uniform_logs = torch.logsumexp(expert - math.log(k), dim=1)
                row.update(
                    query_gate_entropy=float(-(g * g.clamp_min(1e-12).log()).sum(-1).mean() / math.log(k)),
                    query_gate_std=float(g.std(0, unbiased=False).mean()),
                    max_hard_slot_fraction=float(hard.max()),
                    used_hard_slots=int((hard > 0).sum()),
                    effective_soft_slots=float((-(usage * usage.clamp_min(1e-12).log()).sum()).exp()),
                    expert_probability_std=float(expert.exp()[..., 1].std(-1, unbiased=False).mean()),
                    mean_gate_nll_delta=float(F.nll_loss(mean_logs, labels)) - row["nll"],
                    uniform_gate_nll_delta=float(F.nll_loss(uniform_logs, labels)) - row["nll"],
                    support_assignment_entropy=float(model.last_assignment_entropy / math.log(k)),
                )
            rows.append(row)
        rows_by_family[family] = rows
    summary = {family: aggregate(rows) for family, rows in rows_by_family.items()}
    summary["overall"] = aggregate([r for rows in rows_by_family.values() for r in rows])
    return dict(summary=summary, episodes=rows_by_family)


def save_checkpoint(path, model, optimizer, config, cell, step, digest, **extra):
    payload = dict(
        model_type="table_slot_factorial_v3" if cell["kind"] != "plain" else "nanotabpfn",
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        architecture=architecture(model, config),
        config=asdict(config),
        cell=cell,
        step=step,
        source_hash=digest,
        **extra,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def learning_rate(config, step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def run(config, args):
    cell = dict(configurations()[args.index])
    if args.slot_useful_loss != "none":
        cell["slot_useful_loss"] = args.slot_useful_loss
    if args.query_routing_mode != "decoder":
        cell["query_routing_mode"] = args.query_routing_mode
    if args.decoder_interaction != "full":
        cell["decoder_interaction"] = args.decoder_interaction
    digest = source_hash()
    gate = json.loads(Path(args.preflight_record).read_text())
    expected = dict(source_hash=digest, config=asdict(config), eval_query_size=args.eval_query_size)
    if any(gate.get(key) != value for key, value in expected.items()) or not gate.get("passed"):
        raise RuntimeError("Preflight does not match this source/configuration.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError("Refusing to overwrite an existing run; use a fresh output directory.")
    model, initial_hash = build_model(config, cell)
    manifest = dict(
        cell=cell,
        switches=FIXES,
        config=asdict(config),
        architecture=architecture(model, config),
        model_class=type(model).__module__ + "." + type(model).__name__,
        source_hash=digest,
        backbone_initial_hash=initial_hash,
        parameters=sum(p.numel() for p in model.parameters()),
        objective="query NLL + enabled held-out gate/expert loss + enabled early usage loss",
        support_reconstruction_weight=(
            SLOT_USEFUL_RECONSTRUCTION_WEIGHT if cell.get("slot_useful_loss", "none") != "none" else 0
        ),
        embedding_reconstruction_weight=0,
        evaluation_query_size=args.eval_query_size,
        validation_episodes_per_family=config.validation_episodes,
        metric_aggregation="equal episode means; AUC excludes one-class subsets and reports valid counts",
        test_episodes_per_family=args.test_episodes,
        heldout_fraction=0.25,
        heldout_weight=1.0,
        usage_weight_peak=0.01,
        gate_temperature_peak=2.0,
        regularization_duration=config.warmup_steps,
        cell_scope_alignment="unchanged historical per-row slot averaging; no added alignment",
    )
    (out / "config.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    original = OriginalPrior(config.generator())
    validation = make_bank(config, "validation", config.validation_episodes, args.eval_query_size)
    validation_hashes = {e.tensor_hash() for es in validation.values() for e in es}
    (out / "validation_bank.json").write_text(
        json.dumps({f: [e.tensor_hash() for e in es] for f, es in validation.items()}, indent=2)
    )
    best = math.inf
    family_counts = Counter()
    stream = hashlib.sha256()
    started = time.monotonic()
    for step in range(config.steps):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        episodes = [
            training_episode(config, original, cell["prior"], step, 0, j) for j in range(config.micro_batch_size)
        ]
        for episode in episodes:
            family_counts[episode.metadata["family"]] += 1
            stream.update(episode.tensor_hash().encode())
        values = training_step(model, stack(episodes, config.device), cell, step, config, optimizer)
        if step == 0 or (step + 1) % 50 == 0:
            row = dict(
                step=step + 1,
                **values,
                lr=lr,
                elapsed_seconds=time.monotonic() - started,
                family_counts=dict(family_counts),
                train_stream_hash=stream.hexdigest(),
            )
            with (out / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (step + 1) % config.validation_interval == 0 or step + 1 == config.steps:
            result = evaluate(model, validation, config.device)
            result.update(step=step + 1, elapsed_seconds=time.monotonic() - started)
            (out / f"validation-{step + 1}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            score = result["summary"]["overall"]["nll"]
            improved = score < best
            best = min(best, score)
            extra = dict(
                best_validation_nll=best, train_stream_hash=stream.hexdigest(), family_counts=dict(family_counts)
            )
            save_checkpoint(out / "latest.pth", model, optimizer, config, cell, step + 1, digest, **extra)
            if improved:
                save_checkpoint(out / "best.pth", model, optimizer, config, cell, step + 1, digest, **extra)
            print(json.dumps(dict(step=step + 1, validation=result["summary"])), flush=True)
    test = make_bank(config, "test", args.test_episodes, args.eval_query_size)
    assert not validation_hashes.intersection(e.tensor_hash() for es in test.values() for e in es)
    for checkpoint_name in ("latest", "best"):
        checkpoint = torch.load(out / f"{checkpoint_name}.pth", map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, test, config.device)
        result.update(
            step=checkpoint["step"],
            selection=checkpoint_name,
            description="Independent held-out episode bank; major/minor defined by support regime counts",
        )
        (out / f"test-{checkpoint_name}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (out / "complete.json").write_text(json.dumps(dict(steps=config.steps, elapsed_seconds=time.monotonic() - started)))


def preflight(config, args):
    # Full architecture and batch-size forward/backward for every unique
    # scope/switch combination plus vanilla, on every available prior family.
    original = OriginalPrior(config.generator())
    episodes = [original.sample(991)]
    for i, family in enumerate(FAMILIES[1:]):
        generator, groups = task_config(config, 992 + i)
        episodes.append(sample_episode(generator, family=family, seed=992 + i, active_groups=groups))
    tensors = stack([episodes[i % len(episodes)] for i in range(config.micro_batch_size)], config.device)
    initial_hashes, checked = set(), []
    for cell in configurations()[:25]:
        model, initial_hash = build_model(config, cell)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        stats = training_step(model, tensors, cell, 0, config, optimizer)
        checked.append(dict(cell=cell, parameters=sum(p.numel() for p in model.parameters()), stats=stats))
        initial_hashes.add(initial_hash)
        print(json.dumps(checked[-1]), flush=True)
        del model, optimizer
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
    assert len(initial_hashes) == 1
    # Include actual 128-query validation shape and every family.
    eval_bank = make_bank(config, "validation", 1, args.eval_query_size)
    for i in (7, 15, 23, 24):
        model, _ = build_model(config, configurations()[i])
        evaluate(model, eval_bank, config.device)
        del model
    record = dict(
        passed=True,
        source_hash=source_hash(),
        config=asdict(config),
        eval_query_size=args.eval_query_size,
        checked=checked,
        matched_backbone_initial_hash=next(iter(initial_hashes)),
    )
    path = Path(args.preflight_record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False))
    print("PREFLIGHT PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(75))
    parser.add_argument("--output")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--preflight-record", default="preflight.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slot-useful-loss", choices=SLOT_USEFUL_LOSS_MODES, default="none")
    parser.add_argument("--query-routing-mode", choices=QUERY_ROUTING_MODES, default="decoder")
    parser.add_argument("--decoder-interaction", choices=("full", "product"), default="full")
    for key, default in dict(
        seed=11,
        steps=5000,
        width=192,
        hidden=768,
        layers=6,
        heads=6,
        batch_size=8,
        support_size=128,
        query_size=32,
        eval_query_size=128,
        max_features=12,
        num_groups=5,
        warmup_steps=2000,
        validation_interval=500,
        validation_episodes=8,
        test_episodes=32,
    ).items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    if args.matrix:
        print(json.dumps(configurations(), indent=2))
        return
    config = PilotConfig(
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        micro_batch_size=args.batch_size,
        accumulate_gradients=1,
        support_size=args.support_size,
        query_size=args.query_size,
        max_features=args.max_features,
        num_groups=args.num_groups,
        device=args.device,
        warmup_steps=args.warmup_steps,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
    )
    torch.set_num_threads(1)
    if args.preflight:
        preflight(config, args)
    else:
        if args.index is None or args.output is None:
            parser.error("Training requires --index and --output.")
        run(config, args)


if __name__ == "__main__":
    main()
