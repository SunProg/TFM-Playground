"""Where the support competition actually puts each row, on real tables.

Every screening metric in this project scores the slot assignment against a
*synthetic* regime tag.  Real tables carry no such tag, so the only honest
question there is the one this script answers: given the exact representation
``u_i`` that the competition reads, how is each support row assigned, and does
that assignment have any structure at all?

``u_i`` is captured rather than recomputed.  A forward pre-hook on
``adapters.0.datapoint_slots`` records the tensor the competition is actually
handed, so the projections below cannot drift from the model's own input the way
a reimplementation would.  ``pretrain_slot_tabpfn`` reports the same assignment
as summary statistics; this reports it per row, which is what distinguishes "the
entropy fell" from "the entropy fell because one slot took everything".

Outputs, per the layout the earlier artifacts already use:

``support-ui-summary.csv``      one row per (task, prior, arm), support *and*
                                 query metrics side by side
``per-support-slot-assignment-distributions.csv``  one row per support row
``per-query-slot-gate-distributions.csv``          one row per query row --
                                 the decoder's routing gate, since query rows
                                 never enter the competition (see
                                 ``capture_assignment``)
``plot-index.csv``              task to plot and representation paths
``representations/``            the captured ``u_i``, attention and query
                                 gate, as ``npz``
``plots/``                      PCA and t-SNE of the *support* ``u_i``,
                                 coloured by hard slot (query rows have no
                                 captured representation to project; see the
                                 query CSV/summary columns for their routing)

Tables are read from the cached frames named in ``--tasks-csv`` rather than
fetched, so a regeneration is reproducible and needs no network.

**On comparability with the artifacts written before this script existed.**
Those were produced by a pipeline that was never committed.  The schema here
matches theirs column for column, and ``captured_module`` resolves to the same
module for all three placements, but the *row sample* does not: this uses the
repository's own ``select_train_indices``, which stratifies and then permutes,
while the earlier files carry an ascending, differently drawn sample (about 80%
of rows overlap on the tables checked).  Numbers are therefore comparable across
arms **within one run of this script** and not row-for-row against the older
files.  Regenerate the baseline arms alongside the new ones rather than reading
the old summary as their control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import log_loss, normalized_mutual_info_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tfmplayground.experiments.evaluate_integrated_tabarena import select_train_indices
from tfmplayground.interface import get_feature_preprocessor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.slot_regime import SlotLogitsAdapter, load_checkpoint_for_inference
from tfmplayground.models.table_slot import TableSlotModel

#: Suffix of the module whose input is the exact ``u_i``.  The *last* match in
#: module order is captured, which is ``adapters.0`` for the head placement,
#: the deepest tap for MuFASA and the deepest transformer block for the
#: in-backbone one -- the same convention ``slot_backbone.deepest_slot_layer``
#: uses.  Its resolved name is recorded per row, so which representation was
#: captured is never a matter of inference.
CAPTURED_SUFFIX = "datapoint_slots"

#: What a plain nanoTabPFN checkpoint (no adapter, no competition) records
#: instead: the target-column row state after the full backbone, the same
#: quantity ``slot_regime.py`` and ``representation_probe.py`` read.  There is
#: no assignment to capture, so it is recorded verbatim as the "module" name.
VANILLA_SOURCE = "encode_table[target_column]"


@dataclass(frozen=True)
class Arm:
    """One checkpoint, labelled by the two axes the plots lay out."""

    prior: str
    model: str
    path: Path


@dataclass
class DistributionConfig:
    checkpoints: tuple[Arm, ...]
    tasks_csv: Path
    output_dir: Path
    #: Rows sampled per table before the support split.  768 reproduces the
    #: earlier artifacts, whose support counts are four fifths of this.
    #: ``None`` uncaps it -- every row in the cached frame is used, so the
    #: largest tables (Bank_Customer_Churn, coil2000) run close to 10,000
    #: support rows.  Full self-attention between datapoints is O(rows^2), so
    #: this is a GPU-and-``num_mem_chunks`` setting, not a laptop one.
    sample_rows: int | None = 768
    #: Passed straight through to ``encode_table``; chunks the O(rows^2)
    #: datapoint attention under ``torch.no_grad()`` (see
    #: ``nanotabpfn.memory_chunking``) so an uncapped table does not need its
    #: full attention matrix materialized at once.  Only correct without
    #: gradients, which is why this script never needs it disabled.
    num_mem_chunks: int = 1
    folds: int = 5
    seed: int = 2402
    device: str = "cpu"
    tsne_perplexity: float = 30.0
    priors: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    _slots: int = field(default=0, init=False)


def parse_arm(text: str) -> Arm:
    """``prior=arm=path`` -- three fields, because a path may contain ``=``."""
    prior, _, rest = text.partition("=")
    model, _, path = rest.partition("=")
    if not prior or not model or not path:
        raise ValueError(f"Expected prior=arm=path, got {text!r}.")
    return Arm(prior, model, Path(path))


def load_task_frame(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and encoded labels from a cached task frame."""
    frame = pd.read_pickle(path)
    if "__target__" not in frame:
        raise ValueError(f"{path} has no __target__ column.")
    labels = LabelEncoder().fit_transform(frame["__target__"].to_numpy())
    return frame.drop(columns=["__target__"]), labels


def prepare_table(
    features: pd.DataFrame, labels: np.ndarray, config: DistributionConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Subsample, preprocess, and split into support and query.

    Returns the encoded features, the labels, the original row indices, and the
    support count.  The support fraction matches the training side of a
    ``folds``-way split, so these rows are the context an evaluation would give
    the model rather than an arbitrary slice.
    """
    count = len(labels) if config.sample_rows is None else min(config.sample_rows, len(labels))
    selected = select_train_indices(labels, count, seed=config.seed)
    sampled, sampled_labels = features.iloc[selected], labels[selected]
    encoded = get_feature_preprocessor(sampled).fit_transform(sampled)
    encoded = np.asarray(encoded, dtype=np.float32)
    support = int(count * (config.folds - 1) / config.folds)
    if not 1 <= support < count:
        raise ValueError(f"A {count}-row sample leaves no support/query split at {config.folds} folds.")
    return encoded, sampled_labels, selected, support


def captured_module_name(model: TableSlotModel) -> str:
    """The last ``datapoint_slots`` module in module order."""
    names = [name for name, _ in model.named_modules() if name.endswith(CAPTURED_SUFFIX)]
    if not names:
        raise ValueError("This checkpoint runs no data-path competition, so there is no u_i to capture.")
    return names[-1]


@torch.no_grad()
def capture_vanilla(
    model: NanoTabPFNModel,
    encoded: np.ndarray,
    labels: np.ndarray,
    support: int,
    device: str,
    num_mem_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(u, query_probability)``: ``(support, E)`` row states, and each query
    row's predicted positive-class probability, ``(query,)``.

    ``u`` is the same target-column read as ``NanoTabPFNSlotRegimeModel.forward``
    and ``representation_probe.probe_checkpoint``: the label embedding after
    repeated feature/row attention, so a support row's state depends on both
    ``x_i`` and ``y_i``.  No competition means no ``a[i,k]`` to plot by, so the
    caller colours these rows by the one thing that *is* available: the true
    class.

    ``query_probability`` reuses the same ``encode_table`` call rather than a
    second forward pass through ``model(...)``: it is exactly what
    ``NanoTabPFNModel._forward`` computes -- ``decoder`` applied to the query
    rows' target-column state -- read off manually so both outputs share one
    pass over the backbone.
    """
    x = torch.as_tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.as_tensor(labels[:support], dtype=torch.float32, device=device).unsqueeze(0)
    model.to(device).eval()
    encoded_table = model.encode_table((x, y), train_test_split_index=support, num_mem_chunks=num_mem_chunks)
    u = encoded_table[0, :support, -1, :].cpu().numpy()
    query_state = encoded_table[:, support:, -1, :]
    query_logits = model.decoder(query_state)
    query_probability = query_logits.softmax(-1)[0, :, 1].cpu().numpy()
    return u, query_probability


@torch.no_grad()
def capture_assignment(
    model: TableSlotModel,
    name: str,
    encoded: np.ndarray,
    labels: np.ndarray,
    support: int,
    device: str,
    num_mem_chunks: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, attention, query_gate, query_probability)``.

    ``u`` and ``attention`` are ``(support, ...)``, taken off one forward hook
    rather than recomputed or read back off the model's ``last_*`` attributes.
    The point of the projection is to show what a single competition saw and
    decided, and MuFASA's recorded attention is a fusion across taps -- pairing
    that with one tap's ``u`` would plot two different things against each
    other.

    ``query_gate`` is ``(query, num_slots)`` and *is* read off ``last_query_gates``
    -- there is no hook target for it, since query rows never enter the
    competition module at all (``table_slot.py``: "Query rows carry no label,
    so they cannot compete for slots").  They are routed separately by the
    shared decoder's own softmax gate, which the support hook cannot see.

    ``query_probability`` is ``(query,)``, the positive-class probability of
    the model's own mixture prediction -- ``SlotRegimePrediction.marginal_probabilities()``,
    the exact quantity ``slot_regime_loss`` is trained against.  This is the
    ground truth answer to "did the sharpened support assignment help": a hard
    or soft statistic on the gate is a proxy, this is the outcome itself.

    All four come from the one forward pass already run for the support side,
    so none of this costs anything extra.
    """
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    module = dict(model.named_modules())[name]
    handle = module.register_forward_hook(
        lambda _module, inputs, output: captured.append((inputs[0].detach(), output[1].detach()))
    )
    try:
        x = torch.as_tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
        y = torch.as_tensor(labels[:support], dtype=torch.float32, device=device).unsqueeze(0)
        model.to(device).eval()
        # Chunking splits the O(rows^2) datapoint attention over columns
        # rather than rows, so it lowers peak memory without approximating
        # anything; see the config field comment for why no_grad makes this
        # safe.  Real cost only for the largest, uncapped tables.
        prediction = model((x, y), train_test_split_index=support, num_mem_chunks=num_mem_chunks)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"{name} never ran during the forward pass.")
    u, attention = captured[-1]
    query_gate = model.last_query_gates[0].detach().cpu().numpy()
    query_probability = prediction.marginal_probabilities()[0, :, 1].cpu().numpy()
    # The cell placement competes over every row's cells; only the support
    # rows' pooled states are wanted here.
    return (
        u[0, :support].cpu().numpy(),
        attention[0, :support].cpu().numpy(),
        query_gate,
        query_probability,
    )


def prediction_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Cross entropy and ROC AUC of the model's own query prediction.

    Every other query statistic in this module is a proxy for "did the
    sharpened support assignment help" -- gate entropy, gate NMI.  This is the
    outcome itself: ``probability`` is ``SlotRegimePrediction.marginal_probabilities()``
    for the slot models (the exact quantity ``slot_regime_loss`` trains
    against) or the plain softmax output for vanilla, so log loss here is
    cross entropy on the true class, computed the same way for every arm.

    ``NaN`` when a task's query slice is single-class -- both metrics are
    undefined without both classes present, and a real evaluation should not
    silently read a placeholder as a good or bad score.
    """
    if len(set(labels.tolist())) < 2:
        return {"query_cross_entropy": float("nan"), "query_roc_auc": float("nan")}
    return {
        "query_cross_entropy": float(log_loss(labels, probability, labels=[0, 1])),
        "query_roc_auc": float(roc_auc_score(labels, probability)),
    }


def gate_metrics(gate: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """The query-side analogue of ``assignment_metrics``, minus PCA variance.

    There is no captured representation for query rows to project (see
    ``capture_assignment``'s docstring), so this scores only the routing gate
    itself: how sharp it is, and whether the slot it hard-routes to tracks the
    row's true label -- the same question ``assignment_metrics`` answers on the
    support side, applied to the decoder's gate instead of the competition.
    """
    slots = gate.shape[1]
    normalized = gate / np.clip(gate.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(normalized * np.log(np.clip(normalized, 1e-12, None))).sum(axis=1)
    entropy = entropy / math.log(slots) if slots > 1 else np.zeros_like(entropy)
    hard = gate.argmax(axis=1)
    return {
        "mean_normalized_entropy": float(entropy.mean()),
        "mean_confidence": float(gate.max(axis=1).mean()),
        "hard_slot_nmi_with_label": float(normalized_mutual_info_score(labels, hard)),
        "hard_counts": np.bincount(hard, minlength=slots).tolist(),
        "normalized_entropy": entropy,
        "hard": hard,
    }


def assignment_metrics(attention: np.ndarray | None, u: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """The five numbers the summary carries, plus the hard slot counts.

    ``attention is None`` is the vanilla path: there is no assignment, so
    every slot-shaped statistic (entropy, confidence, NMI, hard counts) is left
    out rather than faked, and ``hard`` -- what ``_plot`` colours by -- is the
    true class label itself.  ``hard_slot_nmi_with_label`` would be a tautology
    against that stand-in, which is why it is omitted rather than reported as 1.0.
    """
    variance = PCA(n_components=2).fit(u).explained_variance_ratio_ if len(u) > 2 else np.zeros(2)
    if attention is None:
        return {
            "pca_variance_1": float(variance[0]),
            "pca_variance_2": float(variance[1]),
            "hard": labels.astype(np.int64),
        }
    slots = attention.shape[1]
    normalized = attention / np.clip(attention.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(normalized * np.log(np.clip(normalized, 1e-12, None))).sum(axis=1)
    entropy = entropy / math.log(slots) if slots > 1 else np.zeros_like(entropy)
    hard = attention.argmax(axis=1)
    return {
        "mean_normalized_entropy": float(entropy.mean()),
        "mean_confidence": float(attention.max(axis=1).mean()),
        # Zero when the assignment says nothing about the label; note this is a
        # *diagnostic*, and a high value is not the pilot's objective.
        "hard_slot_nmi_with_label": float(normalized_mutual_info_score(labels, hard)),
        "hard_counts": np.bincount(hard, minlength=slots).tolist(),
        "pca_variance_1": float(variance[0]),
        "pca_variance_2": float(variance[1]),
        "normalized_entropy": entropy,
        "hard": hard,
    }


def _project(u: np.ndarray, kind: str, config: DistributionConfig) -> tuple[np.ndarray, tuple[float, float]]:
    if kind == "pca":
        model = PCA(n_components=2)
        return model.fit_transform(u), tuple(float(v) for v in model.explained_variance_ratio_[:2])
    perplexity = min(config.tsne_perplexity, max(5.0, (len(u) - 1) / 3))
    projection = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=config.seed
    ).fit_transform(u)
    return projection, (float("nan"), float("nan"))


def _plot(
    kind: str, task: dict[str, Any], captures: dict[tuple[str, str], dict[str, Any]], config: DistributionConfig
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    priors, models = config.priors, config.models
    # The header and legend get a fixed inch budget rather than a fixed figure
    # fraction: a one-row grid is a fifth the height of a five-row one, and a
    # fraction that clears the title on one overlaps it on the other.
    height = 5.5 * len(models) + 1.4
    figure, axes = plt.subplots(len(models), len(priors), figsize=(6.5 * len(priors), height), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "^", "s", "D")
    for row, model_name in enumerate(models):
        for column, prior in enumerate(priors):
            axis = axes[row][column]
            capture = captures.get((prior, model_name))
            if capture is None:
                axis.set_axis_off()
                continue
            projection, variance = _project(capture["u"], kind, config)
            metrics, labels = capture["metrics"], capture["labels"]
            vanilla = capture["attention"] is None
            if vanilla:
                # No assignment to colour by, so colour keys off the true
                # class directly -- redundant with the marker shape, which is
                # the honest thing to plot for a model with no competition.
                for label in sorted(set(labels.tolist())):
                    mask = labels == label
                    if not mask.any():
                        continue
                    axis.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        s=18,
                        color=colors[label % len(colors)],
                        marker=markers[label % len(markers)],
                        # No per-row confidence exists without a competition,
                        # so opacity is fixed rather than implying one.
                        alpha=0.55,
                        linewidths=0,
                    )
                axis.set_title(f"{prior.title()} | {model_name}\nno competition; coloured by class")
            else:
                for slot in range(capture["attention"].shape[1]):
                    for label in sorted(set(labels.tolist())):
                        mask = (metrics["hard"] == slot) & (labels == label)
                        if not mask.any():
                            continue
                        axis.scatter(
                            projection[mask, 0],
                            projection[mask, 1],
                            s=18,
                            color=colors[slot % len(colors)],
                            marker=markers[label % len(markers)],
                            # Opacity carries confidence, so a panel that looks
                            # decisive because of its colours but is not shows it.
                            alpha=float(np.clip(capture["attention"][mask].max(axis=1).mean(), 0.15, 0.9)),
                            linewidths=0,
                        )
                counts = ", ".join(str(c) for c in metrics["hard_counts"])
                axis.set_title(
                    f"{prior.title()} | {model_name}\nH={metrics['mean_normalized_entropy']:.3f}; n=[{counts}]"
                )
            if kind == "pca":
                axis.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
                axis.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
            else:
                axis.set_xlabel("t-SNE 1")
                axis.set_ylabel("t-SNE 2")
            axis.grid(alpha=0.2)
    slots = config._slots
    handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[slot % len(colors)], label=f"Slot {slot}")
        for slot in range(slots)
    ] + [
        Line2D([], [], marker=markers[label % len(markers)], linestyle="", color="0.35", label=f"Label {label}")
        for label in (0, 1)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=slots + 2, frameon=False)
    figure.suptitle(
        f"{kind.upper()} of support u_i: {task['dataset']}",
        fontsize=15,
        fontweight="bold",
        y=1 - 0.35 / height,
    )
    figure.text(
        0.5,
        1 - 0.72 / height,
        f"Task {task['task_id']}; {task['sample_rows']} sampled rows, {task['support_rows']} support; "
        "color=hard slot, opacity=confidence, shape=label.",
        ha="center",
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.45 / height, 1, 1 - 0.95 / height))
    destination = config.output_dir / "plots" / f"{task['task_id']}-{task['slug']}-support-ui-{kind}.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def run(config: DistributionConfig) -> Path:
    (config.output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "representations").mkdir(parents=True, exist_ok=True)
    # Preserve the order the arms were given: it is the plot's row and column
    # order, and a sorted one would silently reorder a published figure.
    config.priors = tuple(dict.fromkeys(arm.prior for arm in config.checkpoints))
    config.models = tuple(dict.fromkeys(arm.model for arm in config.checkpoints))

    models = {}
    for arm in config.checkpoints:
        loaded = load_checkpoint_for_inference(arm.path, device=config.device)
        model = loaded.model if isinstance(loaded, SlotLogitsAdapter) else loaded
        if isinstance(model, TableSlotModel):
            models[(arm.prior, arm.model)] = (model, captured_module_name(model))
            config._slots = model.num_slots
        elif isinstance(model, NanoTabPFNModel):
            # No competition, so no module to hook and nothing to name beyond
            # the fixed source the docstring records.
            models[(arm.prior, arm.model)] = (model, VANILLA_SOURCE)
        else:
            raise ValueError(f"{arm.path} is neither a table-slot nor a plain nanoTabPFN checkpoint.")

    tasks = list(csv.DictReader(config.tasks_csv.open()))
    summary_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    query_row_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    for task in tasks:
        frame_path = Path(task["path"])
        if not frame_path.is_file():
            frame_path = config.tasks_csv.parent / frame_path.name
        features, labels = load_task_frame(frame_path)
        encoded, sampled_labels, selected, support = prepare_table(features, labels, config)
        support_labels = sampled_labels[:support]
        query_labels = sampled_labels[support:]
        slug = "".join(c if c.isalnum() else "-" for c in task["dataset"].lower()).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")

        arrays: dict[str, np.ndarray] = {
            "source_row": selected[:support].astype(np.int64),
            "label": support_labels.astype(np.int64),
            "query_source_row": selected[support:].astype(np.int64),
            "query_label": query_labels.astype(np.int64),
        }
        captures: dict[tuple[str, str], dict[str, Any]] = {}
        for (prior, model_name), (model, module_name) in models.items():
            query_gate = None
            if isinstance(model, TableSlotModel):
                u, attention, query_gate, query_probability = capture_assignment(
                    model, module_name, encoded, sampled_labels, support, config.device, config.num_mem_chunks
                )
            else:
                u, query_probability = capture_vanilla(
                    model, encoded, sampled_labels, support, config.device, config.num_mem_chunks
                )
                attention = None
            metrics = assignment_metrics(attention, u, support_labels)
            query_metrics = gate_metrics(query_gate, query_labels) if query_gate is not None else None
            # The outcome metric: every arm produces a real query prediction,
            # slot competition or not, so this is never conditional the way
            # the gate statistics are.
            query_prediction = prediction_metrics(query_probability, query_labels)
            captures[(prior, model_name)] = {
                "u": u,
                "attention": attention,
                "metrics": metrics,
                "labels": support_labels,
            }
            arrays[f"{prior}__{model_name}__u"] = u.astype(np.float32)
            arrays[f"{prior}__{model_name}__query_probability"] = query_probability.astype(np.float32)
            if attention is not None:
                arrays[f"{prior}__{model_name}__attention"] = attention.astype(np.float32)
            if query_gate is not None:
                arrays[f"{prior}__{model_name}__query_gate"] = query_gate.astype(np.float32)
            summary_rows.append(
                {
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "original_rows": task["rows"],
                    "sample_rows": len(sampled_labels),
                    "support_rows": support,
                    "query_rows": len(query_labels),
                    "raw_features": features.shape[1],
                    "encoded_features": encoded.shape[1],
                    "prior": prior,
                    "model": model_name,
                    "captured_module": module_name,
                    # Absent (blank) for the vanilla arms -- there is no
                    # assignment for these to describe.
                    "mean_normalized_entropy": metrics.get("mean_normalized_entropy"),
                    "mean_confidence": metrics.get("mean_confidence"),
                    "hard_slot_nmi_with_label": metrics.get("hard_slot_nmi_with_label"),
                    **{f"hard_slot_{k}": count for k, count in enumerate(metrics.get("hard_counts", []))},
                    "pca_variance_1": metrics["pca_variance_1"],
                    "pca_variance_2": metrics["pca_variance_2"],
                    # The outcome: the model's own query prediction, scored
                    # against the true label -- present for every arm.
                    "query_cross_entropy": query_prediction["query_cross_entropy"],
                    "query_roc_auc": query_prediction["query_roc_auc"],
                    # The query-side routing gate: a proxy for *how* that
                    # prediction was made, so absent for the vanilla arms,
                    # which have no gate at all.
                    "query_mean_normalized_entropy": query_metrics.get("mean_normalized_entropy")
                    if query_metrics
                    else None,
                    "query_mean_confidence": query_metrics.get("mean_confidence") if query_metrics else None,
                    "query_hard_gate_nmi_with_label": query_metrics.get("hard_slot_nmi_with_label")
                    if query_metrics
                    else None,
                    **{
                        f"query_hard_gate_{k}": count
                        for k, count in enumerate(query_metrics.get("hard_counts", []) if query_metrics else [])
                    },
                }
            )
            if attention is not None:
                for position in range(support):
                    row_rows.append(
                        {
                            "task_id": task["task_id"],
                            "prior": prior,
                            "model": model_name,
                            "support_position": position,
                            "source_row": int(selected[position]),
                            "label": int(support_labels[position]),
                            "hard_slot": int(metrics["hard"][position]),
                            "confidence": float(attention[position].max()),
                            "normalized_entropy": float(metrics["normalized_entropy"][position]),
                            **{
                                f"slot_{k}_weight": float(attention[position, k])
                                for k in range(attention.shape[1])
                            },
                        }
                    )
            for position in range(len(query_labels)):
                row = {
                    "task_id": task["task_id"],
                    "prior": prior,
                    "model": model_name,
                    "query_position": position,
                    "source_row": int(selected[support + position]),
                    "label": int(query_labels[position]),
                    "predicted_probability": float(query_probability[position]),
                }
                if query_gate is not None:
                    row.update(
                        {
                            "hard_gate_slot": int(query_metrics["hard"][position]),
                            "confidence": float(query_gate[position].max()),
                            "normalized_entropy": float(query_metrics["normalized_entropy"][position]),
                            **{
                                f"gate_{k}_weight": float(query_gate[position, k])
                                for k in range(query_gate.shape[1])
                            },
                        }
                    )
                query_row_rows.append(row)

        representation_path = config.output_dir / "representations" / f"{task['task_id']}-{slug}-support-ui.npz"
        np.savez_compressed(representation_path, **arrays)
        described = {**task, "slug": slug, "sample_rows": len(sampled_labels), "support_rows": support}
        pca_path = _plot("pca", described, captures, config)
        tsne_path = _plot("tsne", described, captures, config)
        index_rows.append(
            {
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "original_rows": task["rows"],
                "raw_features": features.shape[1],
                "pca_path": str(pca_path),
                "tsne_path": str(tsne_path),
                "representations_path": str(representation_path),
            }
        )
        # Written after every task, so a run killed partway keeps what it has.
        pd.DataFrame(summary_rows).to_csv(config.output_dir / "support-ui-summary.csv", index=False)
        pd.DataFrame(row_rows).to_csv(
            config.output_dir / "per-support-slot-assignment-distributions.csv", index=False
        )
        pd.DataFrame(query_row_rows).to_csv(
            config.output_dir / "per-query-slot-gate-distributions.csv", index=False
        )
        pd.DataFrame(index_rows).to_csv(config.output_dir / "plot-index.csv", index=False)
        print(json.dumps({"task_id": task["task_id"], "dataset": task["dataset"], "support_rows": support}), flush=True)

    return config.output_dir / "support-ui-summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="PRIOR=ARM=PATH",
        help="Repeatable. PRIOR is a plot column, ARM a plot row, e.g. plain=rec1=runs/.../final_checkpoint.pth",
    )
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=768)
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Ignore --sample-rows and use every row in each cached task frame.",
    )
    parser.add_argument(
        "--num-mem-chunks",
        type=int,
        default=1,
        help="Chunk the O(rows^2) datapoint attention over feature columns to bound peak memory on large tables.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> Path:
    arguments = build_parser().parse_args(argv)
    return run(
        DistributionConfig(
            checkpoints=tuple(parse_arm(text) for text in arguments.checkpoint),
            tasks_csv=arguments.tasks_csv,
            output_dir=arguments.output_dir,
            sample_rows=None if arguments.full_sample else arguments.sample_rows,
            num_mem_chunks=arguments.num_mem_chunks,
            folds=arguments.folds,
            seed=arguments.seed,
            device=arguments.device,
        )
    )


if __name__ == "__main__":
    main()
