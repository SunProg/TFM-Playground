# ICLR 2027 manuscript draft

`manuscript.tex` is the anonymous paper source and `manuscript.pdf` is the compiled draft. The supplied conference shell and style files are unchanged. The bibliography is copied from `../native/references.bib`; the two main figures are included from `../native/`.

Appendix tables are checked-in LaTeX files generated from the documented result summaries:

```sh
python3 build_appendix_tables.py
python3 build_diagnostic_table.py
python3 build_heart_table.py
python3 build_observed_id_training_table.py
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

`build_observed_id_training_table.py` formats the archived seed-2402, step-10,000 small and medium reports for the exploratory training-time regime-ID comparison. It checks checkpoint configs and exact episode-ID/metadata alignment across all eight reports, then writes `paper/native/observed_id_training_comparison.json` and `appendix_observed_id_table.tex`. It performs no model inference or training. The reports do not include a bank content hash or evaluator-source hash, and their model-source hashes differ, so the manuscript treats the comparison as descriptive.

The balance-sensitivity tables are regenerated from the existing matched test reports and saved standard-input bank with:

```sh
.venv/bin/python scripts/report_native_balance_slices.py
```

Run that command from the repository root, then compile from this directory. It performs no model inference or retraining, validates episode metadata for all three reports and binary class fractions against the local bank, and writes `paper/native/balance_slices.json` plus the mechanism and balance LaTeX tables. The JSON records hashes of the bank and verified training lock. The other appendix generators only format existing values.

## BeyondArena provenance archive

`beyondarena_36_tasks.csv` lists the 36 tasks reported in the manuscript, their full Data Foundry names, container UUIDs and checksums, dataset rows, raw feature and class counts, target columns, result-table dimensions, official fold counts, and SHA-256 hashes of their official split files. `beyondarena_36_official_folds.csv` lists every one of the 1,440 official `(repeat, fold)` IDs, train/test sizes, and SHA-256 hashes of the index arrays. The index hash is computed from each array using compact JSON (`json.dumps(indices, separators=(",", ":"))`) encoded as UTF-8. The pinned split files contain the full index arrays.

The archive is reconstructed from `paper/native/native_prior_results.md`, Data Foundry **0.0.5**, and the immutable [TabArena/BeyondArena revision `2ecfe882ccfb814fc27c4de10a64ceefd5d7655c`](https://huggingface.co/datasets/TabArena/BeyondArena/tree/2ecfe882ccfb814fc27c4de10a64ceefd5d7655c). Regenerate it from the repository root with:

```sh
.venv/bin/python paper/iclr2027/build_beyondarena_manifest.py
```

The script checks that the 11,000-row, 30-raw-feature, two-to-five-class filter and recorded text/high-cardinality exclusions select exactly the 36 reported tasks from the older 142-entry inventory. It checks that all 36 UUIDs and container checksums match that inventory. For the 32 tasks with saved fold-level results in `results/beyondarena-v4/all-families-37325808/fold_metrics.csv`, it also checks every recorded fold ID and train/test size. Four tasks were skipped by that older run's 10,000-row limit; they appear in the 36-task results under the later 11,000-row setting. Their fold IDs are taken from the pinned official split files but cannot be checked against an extant 36-task fold-level output. The old inventory supplies dataset row and raw feature counts; the manuscript tables supply the reported training-row and model-input feature counts. Grouped and temporal folds need not cover every dataset row.

The historical 36-task job did not record its Hugging Face cache commit. The pinned revision is the one available in the later heart-site script and matches all 36 recorded container checksums; it is a reproducible source for the archived folds, but its identity with the original job's cached revision is not independently proved. The manuscript states these provenance limits in prose. No anonymous public release or verified historical source and checkpoint hashes are available in the saved records; the draft does not present current-checkout hashes as hashes of the original training run.

## Heart-site adaptation rerun

`tfmplayground/experiments/evaluate_heart_sites_adaptation.py` evaluates three
matched conditions on the pinned three-site Heart Disease snapshot: 13 clinical
attributes, attributes with correctly aligned site tags, and attributes with
support site tags permuted. Each target site contributes five stratified draws
of 20 labelled support examples. Every support fold contains all three sites;
the query rows are the remaining target-site examples. The true and shuffled
conditions have the same input width and tag counts. The original
leave-one-site-out `site_given` comparison is retained only as a historical
record in `../native/heart_sites_group_feature.md`; its unseen query category
was not a valid test of giving the correct site identity.

The evaluator loads the nine native checkpoints with the archived inference
architecture in `tfmplayground/models/nanotabpfn_native_v4.py`, which is the
model source paired with those checkpoint files. The current working model
implementation differs at inference. The evaluator writes per-fold metrics,
summary scores, exact fold-index hashes, and source provenance. The Slurm entry
point is `scripts/slurm/evaluate_heart_sites_adaptation.sbatch`. Results used
for Appendix F are archived in `../native/heart_sites_adaptation_*.json`;
`build_heart_table.py` verifies all 15 folds for each of 20 models before
formatting the two case-study tables. These repeated folds share many query
rows and are descriptive, not 15 independent heart datasets.
