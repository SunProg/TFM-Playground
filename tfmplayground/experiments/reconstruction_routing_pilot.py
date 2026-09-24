"""Matched learning pilots; isolated from the submitted pretraining jobs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from tfmplayground.experiments.multiregime_v2 import stack_regime_episodes
from tfmplayground.experiments.scm_regime_prior import SCMRegimeConfig, sample_scm_regime_episode
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.reconstruction_routing import ReconstructionRouter
from tfmplayground.models.slot_regime import (
    SlotRegimePrediction,
    embedding_reconstruction_loss,
    slot_mi_loss,
    slot_regime_loss,
    support_reconstruction_loss,
)
from tfmplayground.models.table_slot import TableSlotModel


class PlainControl(torch.nn.Module):
    def __init__(self, *, width=24, hidden=48, layers=2, heads=3):
        super().__init__()
        self.backbone = NanoTabPFNModel(
            embedding_size=width,
            num_attention_heads=heads,
            mlp_hidden_size=hidden,
            num_layers=layers,
            num_outputs=2,
        )
        self.last = {}

    def forward(self, support_x, support_y, query_x, **kwargs):
        x = torch.cat((support_x, query_x), 1)
        table = self.backbone.encode_table((x, support_y.float()), support_x.shape[1])
        logits = self.backbone.decoder(table[:, support_x.shape[1] :, -1])
        zeros = logits.new_zeros(())
        self.last = {"mse": zeros}
        return SlotRegimePrediction(
            logits[:, :, None],
            logits.new_zeros(*logits.shape[:2], 1),
            logits.new_ones(x.shape[0], support_x.shape[1], 1),
        ), zeros


class PreviousControl(torch.nn.Module):
    """Earlier TableSlotModel, with its original decoder-alpha query gate."""

    def __init__(self, scope, objective, *, width=24, hidden=48, layers=2, heads=3, num_slots=4):
        super().__init__()
        self.objective = objective
        backbone = NanoTabPFNModel(
            embedding_size=width, num_attention_heads=heads, mlp_hidden_size=hidden, num_layers=layers, num_outputs=2
        )
        self.model = TableSlotModel(
            backbone,
            mode="head",
            scope=scope,
            num_slots=num_slots,
            query_routing_mode="decoder",
            reconstruction_mixture="alpha",
            embedding_reconstruction=objective == "embedding_mse",
        )
        self.last = {}

    def forward(self, support_x, support_y, query_x, **kwargs):
        prediction = self.model(
            support_x,
            support_y,
            query_x,
            reconstruct_support=self.objective == "label_alpha",
            reconstruct_embeddings=self.objective == "embedding_mse",
        )
        auxiliary = prediction.slot_logits.new_zeros(())
        mse = auxiliary
        label_nll = auxiliary
        mi = auxiliary
        if self.objective == "embedding_mse":
            mse = embedding_reconstruction_loss(prediction)
            auxiliary = mse
        elif self.objective == "label_alpha":
            label_nll = support_reconstruction_loss(prediction, support_y)
            mi = slot_mi_loss(prediction.support_attention)
            auxiliary = label_nll + 0.05 * mi
        self.last = {
            "mse": mse,
            "support_label_nll": label_nll,
            "mi": mi,
            "slot_std": self.model.last_slots.std(1, unbiased=False).mean().detach(),
            "query_gate_std": prediction.gate().std(1, unbiased=False).mean().detach(),
        }
        return prediction, auxiliary


def evaluate(model, episodes, **kwargs):
    model.eval()
    measurements = []
    with torch.no_grad():
        for episode in episodes:
            p, mse = model(*episode.latent_inputs(), **kwargs)
            logp = p.marginal_log_probabilities()
            row = {k: float(v) for k, v in model.last.items()}
            row.update(
                nll=float(slot_regime_loss(p, episode.query_y)),
                accuracy=float((logp.argmax(-1) == episode.query_y).float().mean()),
            )
            measurements.append(row)
    return {k: float(np.mean([m[k] for m in measurements])) for k in measurements[0]}, measurements


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    prior = SCMRegimeConfig(
        support_size=24,
        query_size=32,
        task_features=2,
        cue_separation=1.5,
        regime_probability=0.65,
        label_noise=0.05,
        calibration_size=128,
    )
    manifest = {
        "prior": asdict(prior),
        "steps": args.steps,
        "seeds": [11, 12],
        "batch_size": 2,
        "learning_rate": 0.001,
        "embedding_weight": 0.0 if args.control or args.previous_control in ("query_only", "label_alpha") else 1.0,
        "support_label_weight": 1.0 if args.previous_control == "label_alpha" else 0.0,
        "mi_weight": 0.05 if args.previous_control == "label_alpha" else 0.0,
        "width": 24,
        "layers": 2,
        "num_slots": 4,
        "beta": 0.5,
        "eta": 0.25,
        "cell_alignment": "Hungarian to support row zero",
        "cell_values_arm": "joint support-value retrieval and query-conditioned cell weighting",
        "device": "cpu",
        "plain_control": args.control,
        "previous_control": args.previous_control,
        "previous_objectives": {
            "query_only": "query NLL",
            "label_alpha": "query NLL + support-label NLL + 0.05 MI",
            "embedding_mse": "query NLL + support embedding MSE",
        },
        "note": "Synthetic learning pilot, not a four-prior sweep.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    heldout = [sample_scm_regime_episode(replace(prior, query_size=64), seed=900000 + i) for i in range(32)]
    results = []
    started = time.monotonic()
    for seed in (11, 12):
        training = [
            stack_regime_episodes([sample_scm_regime_episode(prior, seed=seed * 100000 + 2 * i + j) for j in range(2)])
            for i in range(args.steps)
        ]
        scopes = ("none",) if args.control else ("data", "cell_and_data", "cell")
        if args.previous_control == "embedding_mse":
            scopes = ("data", "cell_and_data")  # The earlier model did not support cell MSE.
        variants = ("vanilla",) if args.control else ("masks", "values")
        if args.previous_control:
            variants = (args.previous_control,)
        for scope in scopes:
            for variant in variants:
                torch.manual_seed(seed)
                if args.previous_control:
                    model = PreviousControl(scope, args.previous_control)
                else:
                    model = PlainControl() if args.control else ReconstructionRouter(scope=scope, variant=variant)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                initial, _ = evaluate(model, heldout[:8])
                gradients = []
                for step, episode in enumerate(training):
                    model.train()
                    torch.manual_seed(seed * 100000 + step)
                    optimizer.zero_grad()
                    prediction, mse = model(*episode.latent_inputs())
                    query_loss = slot_regime_loss(prediction, episode.query_y)
                    if step % 50 == 0 and not args.control and not args.previous_control:
                        grad = torch.autograd.grad(query_loss, model.query_key.weight, retain_graph=True)[0]
                        gradients.append(float(grad.norm()))
                    total = query_loss + mse
                    if not torch.isfinite(total):
                        raise RuntimeError(f"Nonfinite loss: {scope} {variant} {seed} {step}")
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if (step + 1) % 100 == 0:
                        print(
                            f"seed={seed} scope={scope} arm={variant} step={step + 1} loss={float(total.detach()):.4f}",
                            flush=True,
                        )
                final, episodes = evaluate(model, heldout)
                unweighted, _ = evaluate(model, heldout, eta=0)
                shuffled_errors, _ = evaluate(model, heldout, shuffle_errors=True)
                shuffled_labels = []
                for i, ep in enumerate(heldout):
                    gen = torch.Generator().manual_seed(4000 + i)
                    shuffled_labels.append(
                        replace(ep, support_y=ep.support_y[:, torch.randperm(prior.support_size, generator=gen)])
                    )
                shuffled, _ = evaluate(model, shuffled_labels)
                row = dict(
                    seed=seed,
                    scope=scope,
                    variant=variant,
                    initial=initial,
                    final=final,
                    unweighted=unweighted,
                    shuffled_errors=shuffled_errors,
                    shuffled_labels=shuffled,
                    query_key_gradient_norms=gradients,
                    heldout_episodes=episodes,
                )
                results.append(row)
                path = out / f"{scope}-{variant}-{seed}.pth"
                torch.save({"model": model.state_dict(), "scope": scope, "variant": variant, "seed": seed}, path)
                (out / "results.json").write_text(json.dumps(results, indent=2))
                print(
                    json.dumps(
                        {
                            "completed": f"{scope}-{variant}-{seed}",
                            "final": final,
                            "elapsed_s": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
    print(f"RESULTS {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--control", action="store_true")
    controls.add_argument("--previous-control", choices=("query_only", "label_alpha", "embedding_mse"))
    run(parser.parse_args())
