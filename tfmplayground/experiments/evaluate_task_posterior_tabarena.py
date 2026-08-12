"""Run the task-posterior adapter through the official TabArena evaluator.

This is intentionally a thin integration with TabArena.  It does not duplicate
OpenML loading, split selection, metrics, or aggregation locally.  TabArena-Lite
means canonical split 0 across every compatible classification dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

TABARENA_COMMIT = "06334097d539a5d494e56576cb973d09e251dc8c"


def _installed_tabarena_revision(package_file: str) -> str | None:
    """Resolve a source checkout or direct-URL wheel revision without invoking git."""
    for parent in Path(package_file).resolve().parents:
        git_dir = parent / ".git"
        if not git_dir.is_dir():
            continue
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        reference = head.removeprefix("ref: ")
        loose_ref = git_dir / reference
        if loose_ref.is_file():
            return loose_ref.read_text().strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    revision, name = line.split(" ", maxsplit=1)
                    if name == reference:
                        return revision
    try:
        direct_url = importlib.metadata.distribution("tabarena").read_text("direct_url.json")
        if direct_url:
            return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        pass
    return None


def run_official_tabarena(
    *,
    checkpoint: str,
    output_dir: str,
    results_dir: str,
    lite: bool = True,
    debug_mode: bool = False,
):
    """Fit/evaluate on canonical tasks and return TabArena's leaderboard frame."""
    try:
        from tabarena.benchmark.experiment import (
            ModelConstraints,
            TabArenaV0pt1ExperimentBundle,
        )
        from tabarena.contexts import TabArenaContext
    except ImportError as error:
        raise ImportError(
            "Official evaluation requires the pinned TabArena checkout; see the module CLI help."
        ) from error

    import tabarena

    from tfmplayground.models.task_posterior_adapter import load_task_posterior_checkpoint
    from tfmplayground.tabarena_model import TabArenaTaskPosteriorModel

    installed_revision = _installed_tabarena_revision(tabarena.__file__)
    if installed_revision != TABARENA_COMMIT:
        raise RuntimeError(
            f"Expected TabArena commit {TABARENA_COMMIT}, found {installed_revision or 'unverifiable install'}."
        )
    output_path = Path(output_dir)
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_path}")
    adapter, checkpoint_metadata = load_task_posterior_checkpoint(checkpoint_path)
    if adapter.context_mode != "iid_set" or adapter.particle_count != 4:
        raise ValueError("Official runs require an IID-set K=4 adapter checkpoint.")
    provenance = checkpoint_metadata.get("data_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint must document data_provenance before an official run.")
    if provenance.get("tabarena_overlap"):
        raise ValueError("Checkpoint declares TabArena overlap in its meta-training data.")
    if provenance.get("tabarena_checked_against_commit") != TABARENA_COMMIT:
        raise ValueError("Checkpoint contamination audit is not pinned to this TabArena commit.")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    output_path.mkdir(parents=True, exist_ok=False)
    protocol = {
        "tabarena_commit": TABARENA_COMMIT,
        "tabarena_installed_revision": installed_revision,
        "suite": "TabArena-v0.1",
        "subset": "lite & classification & tabpfn" if lite else "classification & tabpfn",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "checkpoint_training_config": checkpoint_metadata.get("training_config"),
        "checkpoint_lineage": checkpoint_metadata.get("lineage"),
        "data_provenance": provenance,
        "aggregation": "official TabArena task metrics and Elo",
        "context_mode": "iid_set",
        "regression": "out_of_scope",
    }
    (output_path / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    constraints = ModelConstraints(
        max_n_samples_train_per_fold=10_000,
        max_n_features=500,
        max_n_classes=10,
        regression_support=False,
    )
    generator = TabArenaTaskPosteriorModel.config_generator()
    generator.manual_configs = [{"model": str(checkpoint_path)}]
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        outer_experiments=True,
        custom_model_constraints={TabArenaTaskPosteriorModel.ag_key: constraints},
    ).build_experiments()

    subset = ["classification", "tabpfn"]
    if lite:
        subset.insert(0, "lite")
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
        debug_mode=debug_mode,
    )
    return context.compare(output_dir=output_path, subset=subset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all canonical outer splits rather than TabArena-Lite split 0.",
    )
    parser.add_argument("--debug-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    leaderboard = run_official_tabarena(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        lite=not args.full,
        debug_mode=args.debug_mode,
    )
    print(leaderboard.to_markdown(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
