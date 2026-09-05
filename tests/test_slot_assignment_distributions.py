"""End-to-end cover for the support-assignment artifact regeneration."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tfmplayground.experiments.pretrain_multiregime_v2 import V2TrainingConfig, _checkpoint, build_model
from tfmplayground.experiments.slot_assignment_distributions import (
    DistributionConfig,
    capture_assignment,
    capture_vanilla,
    captured_module_name,
    parse_arm,
    run,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

SUPPORT_FRACTION = 4 / 5


def _config(kind: str) -> V2TrainingConfig:
    return V2TrainingConfig(
        device="cpu",
        model_type=kind,
        embedding_size=12,
        num_attention_heads=3,
        mlp_hidden_size=24,
        num_layers=6,
        support_size=4,
        query_size=3,
        min_features=2,
        max_features=2,
        slot_layer_indices=(3, 4, 5),
        validation_episodes=1,
    )


def _write_checkpoint(kind: str, directory: Path, name: str) -> Path:
    torch.manual_seed(0)
    model = build_model(_config(kind)).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = directory / f"{name}.pth"
    torch.save(
        _checkpoint(model, optimizer, scheduler, _config(kind), 0, np.random.default_rng(1), None), path
    )
    return path


def _write_vanilla_checkpoint(directory: Path, name: str) -> Path:
    """A bare backbone checkpoint, in the plain-inference format every
    evaluator in this package reads -- no ``model_kind``, no slot keys."""
    torch.manual_seed(1)
    model = NanoTabPFNModel(embedding_size=12, num_attention_heads=3, mlp_hidden_size=24, num_layers=2, num_outputs=2)
    path = directory / f"{name}.pth"
    torch.save(
        {
            "architecture": {
                "embedding_size": 12,
                "num_attention_heads": 3,
                "mlp_hidden_size": 24,
                "num_layers": 2,
                "num_outputs": 2,
            },
            "model": model.state_dict(),
        },
        path,
    )
    return path


def _write_task(directory: Path, rows: int = 60) -> Path:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.choice(["x", "y"], size=rows),
            "__target__": rng.choice(["no", "yes"], size=rows),
        }
    )
    frame_path = directory / "task-1.pkl"
    frame.to_pickle(frame_path)
    tasks_path = directory / "tasks.csv"
    with tasks_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "dataset", "rows", "features", "path"))
        writer.writeheader()
        writer.writerow(
            {"task_id": "1", "dataset": "Tiny Set", "rows": rows, "features": 3, "path": str(frame_path)}
        )
    return tasks_path


def test_parse_arm_needs_all_three_fields():
    arm = parse_arm("plain=rec1=/tmp/a=b.pth")
    assert (arm.prior, arm.model, str(arm.path)) == ("plain", "rec1", "/tmp/a=b.pth")
    with pytest.raises(ValueError, match="prior=arm=path"):
        parse_arm("plain=only")


def test_captured_module_is_the_deepest_competition_for_every_placement():
    """The name is recorded per row, so it must resolve the way it reads."""
    expected = {
        "table_slot_head": "adapters.0.datapoint_slots",
        "table_slot_mufasa": "adapters.2.datapoint_slots",
        "table_slot_backbone": "backbone.transformer_blocks.5.table_slots.datapoint_slots",
    }
    for kind, name in expected.items():
        assert captured_module_name(build_model(_config(kind))) == name


def test_capture_returns_the_competition_input_its_attention_and_the_query_gate():
    model = build_model(_config("table_slot_head")).eval()
    encoded = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
    labels = np.array([0, 1] * 5)
    u, attention, query_gate, query_probability = capture_assignment(
        model, captured_module_name(model), encoded, labels, 8, "cpu"
    )
    assert u.shape == (8, 12)
    assert attention.shape == (8, 4)
    # Competitive attention is a distribution over slots for every row.
    np.testing.assert_allclose(attention.sum(axis=1), np.ones(8), atol=1e-5)
    # Query rows never enter the competition -- there are 2 of them here
    # (10 total, 8 support) -- and their gate is a separate softmax.
    assert query_gate.shape == (2, 4)
    np.testing.assert_allclose(query_gate.sum(axis=1), np.ones(2), atol=1e-5)
    # The outcome metric: a real probability per query row, in [0, 1].
    assert query_probability.shape == (2,)
    assert ((query_probability >= 0) & (query_probability <= 1)).all()


def test_capture_vanilla_returns_the_target_column_row_state_and_a_query_probability():
    model = NanoTabPFNModel(embedding_size=12, num_attention_heads=3, mlp_hidden_size=24, num_layers=2, num_outputs=2)
    encoded = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
    labels = np.array([0, 1] * 5)
    u, query_probability = capture_vanilla(model, encoded, labels, 8, "cpu")
    assert u.shape == (8, 12)
    assert query_probability.shape == (2,)
    assert ((query_probability >= 0) & (query_probability <= 1)).all()


def test_run_writes_the_documented_layout():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tasks = _write_task(root)
        config = DistributionConfig(
            checkpoints=(
                parse_arm(f"plain=baseline={_write_checkpoint('table_slot_head', root, 'plain')}"),
                parse_arm(f"mixed=baseline={_write_checkpoint('table_slot_head', root, 'mixed')}"),
            ),
            tasks_csv=tasks,
            output_dir=root / "out",
            sample_rows=40,
        )
        run(config)
        out = root / "out"
        support = int(40 * SUPPORT_FRACTION)

        summary = pd.read_csv(out / "support-ui-summary.csv")
        assert list(summary["prior"]) == ["plain", "mixed"]
        assert set(summary["support_rows"]) == {support}
        assert set(summary["captured_module"]) == {"adapters.0.datapoint_slots"}
        # Every row lands in exactly one slot, so the hard counts must total.
        assert (summary[[f"hard_slot_{k}" for k in range(4)]].sum(axis=1) == support).all()

        rows = pd.read_csv(out / "per-support-slot-assignment-distributions.csv")
        assert len(rows) == 2 * support
        weights = rows[[f"slot_{k}_weight" for k in range(4)]]
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(len(rows)), atol=1e-5)
        np.testing.assert_allclose(weights.max(axis=1), rows["confidence"], atol=1e-6)

        index = pd.read_csv(out / "plot-index.csv")
        assert len(index) == 1
        for column in ("pca_path", "tsne_path", "representations_path"):
            assert Path(index[column][0]).is_file()

        stored = np.load(out / "representations" / "1-tiny-set-support-ui.npz")
        assert stored["plain__baseline__u"].shape == (support, 12)
        assert stored["mixed__baseline__attention"].shape == (support, 4)
        assert stored["source_row"].shape == (support,)

        # Query rows never compete for slots, so their routing is scored
        # separately: same summary row, a distinct per-row CSV.
        query_rows_expected = 40 - support
        assert (summary["query_rows"] == query_rows_expected).all()
        assert summary["query_hard_gate_nmi_with_label"].notna().all()
        # The outcome metric -- present for every arm, gate or not.
        assert summary["query_cross_entropy"].notna().all()
        assert summary["query_roc_auc"].notna().all()
        query_rows_df = pd.read_csv(out / "per-query-slot-gate-distributions.csv")
        assert len(query_rows_df) == 2 * query_rows_expected
        gate_weights = query_rows_df[[f"gate_{k}_weight" for k in range(4)]]
        np.testing.assert_allclose(gate_weights.sum(axis=1), np.ones(len(query_rows_df)), atol=1e-5)
        assert query_rows_df["predicted_probability"].between(0, 1).all()
        assert stored["plain__baseline__query_gate"].shape == (query_rows_expected, 4)
        assert stored["plain__baseline__query_probability"].shape == (query_rows_expected,)
        assert stored["query_label"].shape == (query_rows_expected,)


def test_run_places_a_vanilla_checkpoint_in_the_same_grid_with_no_attention():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tasks = _write_task(root)
        config = DistributionConfig(
            checkpoints=(
                parse_arm(f"plain=baseline={_write_checkpoint('table_slot_head', root, 'plain')}"),
                parse_arm(f"plain=vanilla={_write_vanilla_checkpoint(root, 'plain_vanilla')}"),
            ),
            tasks_csv=tasks,
            output_dir=root / "out",
            sample_rows=40,
        )
        run(config)
        out = root / "out"
        support = int(40 * SUPPORT_FRACTION)

        summary = pd.read_csv(out / "support-ui-summary.csv")
        vanilla_row = summary[summary["model"] == "vanilla"].iloc[0]
        assert vanilla_row["captured_module"] == "encode_table[target_column]"
        # No assignment exists, so the slot-shaped columns are blank rather
        # than a fabricated number; PCA variance is still real.
        assert pd.isna(vanilla_row["mean_normalized_entropy"])
        assert pd.isna(vanilla_row["hard_slot_nmi_with_label"])
        assert vanilla_row["pca_variance_1"] > 0
        # No gate exists either, so the gate-shaped query columns are blank...
        assert pd.isna(vanilla_row["query_hard_gate_nmi_with_label"])
        # ...but a real query prediction does exist -- every arm produces one.
        assert not pd.isna(vanilla_row["query_cross_entropy"])
        assert not pd.isna(vanilla_row["query_roc_auc"])

        # The vanilla arm has no per-row assignment, so it contributes nothing
        # to the per-row distribution file -- only the slot arm does.
        rows = pd.read_csv(out / "per-support-slot-assignment-distributions.csv")
        assert set(rows["model"]) == {"baseline"}
        assert len(rows) == support

        # Both arms predict, so both contribute to the per-query file, and the
        # vanilla rows simply have no gate columns filled in.
        query_rows = pd.read_csv(out / "per-query-slot-gate-distributions.csv")
        assert set(query_rows["model"]) == {"baseline", "vanilla"}
        vanilla_query = query_rows[query_rows["model"] == "vanilla"]
        assert vanilla_query["predicted_probability"].between(0, 1).all()
        assert vanilla_query["hard_gate_slot"].isna().all()

        stored = np.load(out / "representations" / "1-tiny-set-support-ui.npz")
        assert stored["plain__vanilla__u"].shape == (support, 12)
        assert "plain__vanilla__attention" not in stored.files
        assert "plain__baseline__attention" in stored.files
        assert stored["plain__vanilla__query_probability"].shape == (40 - support,)
        assert "plain__vanilla__query_gate" not in stored.files

        index = pd.read_csv(out / "plot-index.csv")
        for column in ("pca_path", "tsne_path"):
            assert Path(index[column][0]).is_file()
