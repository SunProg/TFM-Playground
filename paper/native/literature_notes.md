# Literature and citation guide

Prepared 2026-09-24 for [the native-prior outline](outline.md). The accompanying [references.bib](references.bib) contains 29 entries checked against primary publication pages, author manuscripts, official repositories, or publisher metadata. This is a targeted reading list, not a systematic review.

External references support background, method provenance, and evaluation choices. They do **not** substantiate our experiment numbers; those remain supported by the local result files linked in the outline. Published-model and conventional-baseline references do not reintroduce non-native NanoTabPFN experiments.

## 1. Core PFN and model references

- **`muller2022pfn` — Müller et al., ICLR 2022.** [Transformers Can Do Bayesian Inference](https://openreview.net/forum?id=KSugKcbNf9). Foundational citation for learning a posterior-predictive approximation from supervised tasks sampled from a prior. Use in Sections 1 and 2.1; do not equate predictive approximation with recovery of latent mechanisms.
- **`hollmann2023tabpfn` — Hollmann et al., ICLR 2023.** [TabPFN](https://openreview.net/forum?id=cp5PvcI6w8_). Introduces the TabPFN classification model and its synthetic-prior approach. Use for the model family's origins, not as the only citation for later evaluated releases.
- **`hollmann2025tabpfn` — Hollmann et al., Nature 2025.** [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6). Main reference for the TabPFN v2 generation. Use in Sections 2.1 and 4.4 alongside exact evaluated checkpoint identifiers.
- **`pfefferle2025nanotabpfn` — Pfefferle et al., 2025 preprint.** [nanoTabPFN](https://arxiv.org/abs/2511.03634). Direct architectural/implementation starting point: a lightweight educational reimplementation of TabPFN. Cite in Sections 2.1 and 4.1 and document our modifications separately.
- **`qu2025tabicl` — Qu et al., ICML 2025.** [TabICL](https://proceedings.mlr.press/v267/qu25d.html). Main source for the TabICL model and its original synthetic pretraining approach. Cite in Sections 2.1 and 3.2. Exact native SCM/Reg2Cls behavior also requires the source-code revision used in our implementation.
- **`qu2026tabiclv2` — Qu et al., ICML 2026.** [TabICLv2](https://arxiv.org/abs/2602.11139). Citation for the published v2 comparator and its revised generation engine, architecture, and training. The current manuscript explicitly identifies ICML 2026 publication; unverified proceedings volume/pages are omitted.
- **`grinsztajn2026tabpfn3` — Grinsztajn et al., 2026 technical report.** [TabPFN-3](https://arxiv.org/abs/2605.13986). Direct source for the v3 comparator. It is recorded as a technical-report preprint, not an assumed peer-reviewed conference paper.
- **`priorlabs2026tabpfn26` — Prior Labs, official model card.** [TabPFN-2.6](https://huggingface.co/Prior-Labs/tabpfn_2_6). Release-specific provenance, not a separate research paper. Its embedded citation names TabPFN-2.5, so that citation must not be silently relabeled as a 2.6 paper.

## 2. Closest conceptual and theoretical related work

These sources are important for positioning the contribution. Avoid claiming that this is the first demonstration that transformers can learn mixtures.

- **`pathak2024mixtures` — Pathak et al., ICLR 2024.** [Transformers can optimally learn regression mixture models](https://openreview.net/forum?id=sLkj91HIZU). Closely related work on transformer prediction for regression mixtures. Discuss in Section 2.4 and contrast its regression setting with our nonlinear multiclass task generators and controlled pretraining comparison.
- **`jin2025mixtures` — Jin et al., TMLR 2025.** [In-context Learning for Mixture of Linear Regression: Existence, Generalization and Training Dynamics](https://openreview.net/forum?id=buZXVuTsHY). Especially relevant theory for in-context learning under linear mixtures. Its assumptions and guarantees do not directly cover our nonlinear SCMs, class discretization, or routing conditions. The BibTeX uses the final TMLR title rather than an earlier arXiv title.
- **`xie2022implicit` — Xie et al., ICLR 2022.** [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080). Useful conceptual connection between in-context prediction and latent-concept inference, developed in a different generative setting. It is not a theorem about our tabular generator.
- **`jacobs1991experts` — Jacobs et al., Neural Computation 1991.** [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Classical background for input-dependent allocation among local predictors. Use to introduce mixture/gating ideas in Sections 2.2 and 3.4.
- **`jordan1994hierarchical` — Jordan and Jacobs, Neural Computation 1994.** [Hierarchical Mixtures of Experts and the EM Algorithm](https://doi.org/10.1162/neco.1994.6.2.181). Additional background on gated conditional mixtures. Our mixture is in the **data-generating prior**; these citations do not make our NanoTabPFN an architectural mixture-of-experts model.
- **`jiang1999identifiability` — Jiang and Tanner, Neural Networks 1999.** [On the identifiability of mixtures-of-experts](https://www.sciencedirect.com/science/article/pii/S0893608099000660). Supports a careful discussion of identifiability as a model- and assumption-dependent issue. Do not infer either regime recovery from improved prediction or universal non-identifiability from feature-independent routing.
- **`bengio2009curriculum` — Bengio et al., ICML 2009.** [Curriculum Learning](https://doi.org/10.1145/1553374.1553380). Terminological/methodological background for changing training exposure over time. Our fixed and curriculum runs also differ in exposure proportions, so the citation does not establish a causal scheduling benefit.

Suggested positioning:

> Prior studies investigate transformer learning of regression mixtures and latent generative structure. We instead study empirical learnability of heterogeneous multiclass tabular tasks, using matched single-regime and multiregime versions of a native tabular pretraining prior in a compact PFN, and assess both synthetic behavior and real-world transfer.

Attach `pathak2024mixtures`, `jin2025mixtures`, and, where discussing Bayesian interpretation, `xie2022implicit`. Present this as the scope of our study, not an established “first” claim or a general learnability theorem.

## 3. Heterogeneity and benchmark context

- **`purucker2026beyondarena` — Purucker et al., 2026 preprint.** [Beyond IID: How General Are Tabular Foundation Models, Really?](https://arxiv.org/abs/2606.30410). Essential citation for BeyondArena and its IID/grouped/temporal evaluation context. The full benchmark and our selected **36 tasks** are not interchangeable. Our 27-binary/9-multiclass split and 35-task rank figure are local evaluation choices.
- **`erickson2025tabarena` — Erickson et al., NeurIPS 2025.** [TabArena](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html). Optional benchmark lineage and tabular comparison context. Do not substitute this citation for BeyondArena.
- **`gardner2023tableshift` — Gardner et al., NeurIPS 2023.** [TableShift](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html). Directly relevant motivation for distribution-shift evaluation in tabular data. We do not claim to evaluate TableShift.
- **`koh2021wilds` — Koh et al., ICML 2021.** [WILDS](https://proceedings.mlr.press/v139/koh21a.html). Optional broader context for real-world group/domain shifts. A grouped split alone does not establish the particular regime-dependent label mechanism assumed by our synthetic prior.
- **`grinsztajn2022trees` — Grinsztajn et al., NeurIPS 2022.** [Why do tree-based models still outperform deep learning on typical tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html). Historical evidence about tabular inductive biases and the importance of strong tree baselines. Do not turn its historical comparison into a claim that trees outperform current foundation models universally.

Use the mixture papers for **within-dataset heterogeneity**, and TableShift/BeyondArena for **evaluation under distribution shift**. These related concepts should not be treated as synonyms.

## 4. Baseline citations

Cite the algorithms actually reported and record their package versions, hyperparameters, preprocessing, and tuning budgets separately.

| Citation key | Primary source | Use |
|---|---|---|
| `breiman2001forests` | [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) | Random forest algorithm |
| `chen2016xgboost` | [XGBoost](https://arxiv.org/abs/1603.02754) | XGBoost comparator |
| `ke2017lightgbm` | [LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | LightGBM comparator |
| `prokhorenkova2018catboost` | [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | CatBoost comparator |
| `pedregosa2011sklearn` | [Scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html) | Implementation provenance, including logistic regression and histogram gradient boosting |

## 5. Metrics, comparisons, and case-study data

- **`gneiting2007scoring` — Gneiting and Raftery, JASA 2007.** [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Background for logarithmic scoring and probabilistic prediction. Define our excess-CE baseline explicitly; that normalization is part of our evaluation protocol.
- **`guo2017calibration` — Guo et al., ICML 2017.** [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Optional if discussing calibration. Lower cross-entropy alone is not a calibration-specific result; use dedicated evidence before claiming improved calibration.
- **`demsar2006comparisons` — Demšar, JMLR 2006.** [Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/v7/demsar06a.html). Background for dataset-level ranks and multi-dataset statistical comparisons. The current heatmap is descriptive: do not imply a significance test has been performed, and do not equate non-significance with equivalence.
- **`janosi1989heart` — Janosi et al., UCI dataset.** [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). Cite the original data source in Section 6.3. The year follows UCI's recommended citation. Document our site selection, preprocessing, and leave-one-site-out protocol independently; the case study does not establish clinical utility.

## 6. Version and reproducibility checks before submission

1. Record the exact NanoTabPFN source revision and all architecture changes.
2. Record the TabICL source revision used for the native SCM/Reg2Cls prior. “Native TabICL prior” does not mean the full TabICL model was retrained.
3. Record checkpoint filenames/revisions and package versions for TabICL v1/v1.1/v2 and TabPFN v2.2/v2.6/v3. A package version is not necessarily a model generation. No separate paper is invented here for a v2.2 label.
4. Cite TabPFN-2.6's official card for release provenance, alongside the appropriate model-family paper. Pin a repository revision because model cards can change.
5. Record the BeyondArena data revision and exact task/fold list. Keep the 36-task score summaries and 35-task ranks distinct.
6. Retain peer-reviewed publication status only where verified. NanoTabPFN, BeyondArena, and TabPFN-3 are stored as preprints/reports in this bibliography.

## 7. Using the bibliography

The outline uses Pandoc-style keys, for example `[@muller2022pfn; @qu2025tabicl]`. From the repository root, resolve them with:

```sh
pandoc paper/native/outline.md --bibliography=paper/native/references.bib --citeproc -t plain
```

For a LaTeX manuscript, use the same keys, e.g. `\citep{muller2022pfn,qu2025tabicl}`, and `\bibliography{references}` when the manuscript and BibTeX file share a directory. Use the bibliography style provided by the selected conference template. The optional entries need not all appear in the final paper.
