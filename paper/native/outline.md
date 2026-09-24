# Learning Heterogeneous Tabular Tasks through Multiregime Pretraining of Prior-Data Fitted Networks

## Abstract

Tabular datasets can combine observations from different populations, environments, or operating conditions, each governed by a different relationship between features and labels. These coexisting relationships define distinct regimes. Regime membership, however, may be unobserved, requiring classifiers to account for heterogeneity within a single dataset. Prior-data fitted networks (PFNs) are pretrained on synthetic datasets to predict new observations from labelled examples without task-specific retraining. We investigate whether incorporating regime heterogeneity into this pretraining enables compact PFNs to learn multiregime classification. We extend TabICL's structural-causal-model prior with regime-dependent score functions and label mappings, and compare fixed-mixture and curriculum pretraining with canonical NanoTabPFN pretrained on the matched single-regime prior, published tabular foundation models, and conventional classifiers. On held-out synthetic tasks with no explicit routing score, the largest effect occurs in feature-dependent, multiregime multiclass tasks: curriculum pretraining improves a six-layer NanoTabPFN's excess cross-entropy from −0.1562 to −0.1730 nats. Across 2,880 matched soft-gate episodes, the paired cross-entropy difference is −0.0168 nats (95% interval [−0.0179, −0.0158]) and the accuracy-gain difference is +0.84 points ([+0.76, +0.91]). Across all matched multiregime tasks with three to five classes, the curriculum model exceeds majority-class accuracy by 6.69 points, compared with 6.30 for canonical NanoTabPFN; its excess cross-entropy improves from −0.0838 to −0.0926 nats. However, these synthetic improvements do not translate into a consistent real-world advantage. We evaluate transfer on 36 BeyondArena tasks: nine multiclass tasks comprising six IID, two grouped, and one temporal task, and 27 binary tasks comprising 26 IID and one grouped task. Across the multiclass tasks, the curriculum model achieves a mean accuracy gain of 19.9 percentage points, compared with 20.0 for canonical NanoTabPFN and 23.0 for published TabICL v2. Across the binary tasks, the corresponding gains are 9.2 points for the curriculum model, 9.4 for canonical NanoTabPFN, and 11.2 for TabICL v2. The native-prior variants perform similarly on IID and grouped tasks; mean multiclass gains are lower on the grouped subset than on the IID subset. These findings provide empirical evidence that compact PFNs can learn multiregime multiclass classification and that explicit multiregime pretraining improves prediction on synthetic heterogeneous tasks, while leaving consistent real-world transfer gains an open challenge.

## Working outline

Working outline, updated 2026-09-24. Subsequent outline revisions should update this file in place.

Abstract result basis: the six-layer final-checkpoint, score-hidden synthetic results in Section 5.1 and the 36-task BeyondArena summaries in Section 6.1. The abstract consistently reports the curriculum variant; fixed-mixture results remain in the full comparisons. These replace the older non-native-prior and 32-task results.

Citation resources: [BibTeX bibliography](references.bib) and [literature/claim guide](literature_notes.md). Citation keys below use Pandoc syntax and can be reused in LaTeX.

Scope: native-prior NanoTabPFN experiments only, with published models and conventional methods retained as comparison baselines. The architecture is NanoTabPFN; “native” refers to TabICL's pretraining prior, not a different TabICL architecture. Non-native NanoTabPFN results are excluded.

Central argument: compact PFNs can learn multiregime multiclass tasks, and explicit multiregime pretraining improves prediction particularly under feature-dependent routing. These synthetic improvements do not establish a consistent real-world advantage over matched native single-regime pretraining.

### Submission constraints and page budget

Apply the supplied conference rules: the submission main text must not exceed **nine pages**; the discussion/rebuttal and camera-ready limit is **ten pages**. References do not count toward the limit. Appendices may be unlimited but must follow the bibliography, and reviewers are not required to read them. Do not rely on the later ten-page allowance for the initial submission.

Budget the title, abstract, text, equations, tables, figures, and captions within the nine-page submission limit. Target **8.5 pages of planned content**, retaining **0.5 page for layout variation**:

| Component | Target pages |
|---|---:|
| Title and abstract | 0.75 |
| 1. Introduction | 1.00 |
| 2. Background and related work | 0.50 |
| 3. Native-prior multiregime method | 1.50 |
| 4. Experimental setup | 0.75 |
| 5. Synthetic results | 1.75 |
| 6. Real-world transfer | 1.25 |
| 7. Discussion and limitations | 0.75 |
| 8. Conclusion | 0.25 |
| Planned content subtotal | **8.50** |
| Layout buffer | **0.50** |
| Submission maximum | **9.00** |

These are drafting targets, not a verified PDF page count. Check the compiled manuscript in the conference template, including float placement, before submission. The working notes, page-budget table, source pointers, and planning instructions in this file are not manuscript content.

**Main-text evidence rule:** a reviewer should understand the contribution and its limits without opening the appendices. Keep the prior definition, matched single-regime control, routing conditions and training-time score exposure, evaluation/task counts, metric definitions, canonical and published comparisons, synthetic gains, neutral real-world transfer, and single-seed caveat in the main text.

**Display budget:** plan two figures and two compact tables. Figure 1 explains the prior and routing. Figure 2 summarizes final-checkpoint synthetic effects by routing, class type, and model size; do not include the full checkpoint grid. Table 1 contains matched synthetic headline comparisons; Table 2 contains BeyondArena overall and split-specific comparisons. If space is tight, combine Table 1 with Figure 2 before removing essential comparisons. The three existing detailed plots remain attached in the appendices below; the two compact main-text figures are still to be prepared.

Use compact paragraphs or run-in headings in the manuscript, especially in Sections 2, 5, and 6. The finer subsections below are drafting guides, not a requirement to typeset every subheading.

## 1. Introduction

Target: **1.00 page**. Preserve the population/environment/operating-condition background and the accessible PFN explanation, then state the research questions and bounded contribution.

Open with the abstract's motivation: tabular datasets can combine observations from different populations, environments, or operating conditions, each governed by a different relationship between features and labels. These coexisting relationships define distinct regimes. Regime membership, however, may be unobserved, requiring classifiers to account for heterogeneity within a single dataset. Relate this framing to conditional-mixture models, without assuming that every observed population or group has a different label mechanism. [@jacobs1991experts; @jordan1994hierarchical]

Introduce PFNs in accessible terms: they are pretrained on synthetic datasets to predict new observations from labelled examples without task-specific retraining. Explain that the pretraining prior specifies the kinds of feature–label relationships and task variation the model encounters. This motivates asking whether explicitly including coexisting prediction mechanisms in that prior improves learning of heterogeneous tasks. [@muller2022pfn; @hollmann2023tabpfn; @qu2025tabicl]

- Frame the study around **within-dataset heterogeneity** first, then introduce generalization across groups or environments as a separate transfer question. Use tabular shift benchmarks to motivate the latter, rather than treating grouped evaluation as direct evidence of latent regimes. [@gardner2023tableshift; @purucker2026beyondarena]
- Present three research questions:
  1. Can compact PFNs learn multiregime classification?
  2. Does multiregime pretraining improve over matched single-regime pretraining?
  3. Do these improvements transfer to real-world datasets?
- Preview the controlled comparison: extend the native TabICL prior with regime-dependent score functions and label mappings, retain its single-regime form as the canonical control, and evaluate fixed-mixture and curriculum pretraining in NanoTabPFN alongside published and conventional baselines.
- Summarize the main finding: multiregime pretraining improves synthetic multiclass prediction, particularly under feature-dependent routing, but provides little consistent additional benefit on BeyondArena.

## 2. Background and Related Work

Target: **0.50 page**. Compress the four topics below into a connected discussion, prioritizing PFNs, the native TabICL prior, and the closest mixture-learning work. Avoid repeating the introduction or giving a separate mini-survey for each model. Retain the essential novelty distinctions in the main text; the literature guide holds the longer reading list.

### 2.1 Prior-data fitted networks

Introduce synthetic pretraining, in-context prediction, NanoTabPFN, and published TabPFN and TabICL models.

Cite the foundational PFN formulation [@muller2022pfn], TabPFN's original and v2-generation papers [@hollmann2023tabpfn; @hollmann2025tabpfn], the lightweight NanoTabPFN implementation [@pfefferle2025nanotabpfn], and TabICL [@qu2025tabicl]. Use the release-specific TabICLv2 and TabPFN-3 references for those comparators. [@qu2026tabiclv2; @grinsztajn2026tabpfn3]

### 2.2 Heterogeneous tabular prediction

Develop the introduction's population/environment/operating-condition motivation into a discussion of latent subpopulations, mixture models, and varying feature–label relationships. Explain how a predictor must account for coexisting relationships when regime membership is not supplied. Distinguish within-dataset heterogeneity from generalization to unseen groups, and avoid assuming that observed group labels exactly identify prediction regimes.

Relate regime-dependent prediction to classical conditional mixtures [@jacobs1991experts; @jordan1994hierarchical], without describing our predictor as an architectural mixture of experts. Use TableShift and BeyondArena for the separate motivation of tabular distribution shift. [@gardner2023tableshift; @purucker2026beyondarena]

### 2.3 Synthetic prior design

Explain how pretraining task distributions encode assumptions about label relationships, class imbalance, and regime structure.

Ground prior-based task generation in PFN and TabICL work [@muller2022pfn; @qu2025tabicl]. Distinguish our extension of the native prior from the broader generation changes introduced in TabICLv2. [@qu2026tabiclv2]

### 2.4 In-context learning of mixtures

Discuss transformer learning of regression mixtures [@pathak2024mixtures; @jin2025mixtures] and the related Bayesian interpretation of in-context inference [@xie2022implicit]. Position our contribution as an empirical study of nonlinear multiclass tabular mixtures and matched prior design, not the first demonstration of mixture learning or a general learnability theorem. Explain why guarantees for linear regression mixtures do not automatically extend to this setting.

## 3. Multiregime Extensions of the Native TabICL Prior

Target: **1.50 pages**, including conceptual Figure 1. Give enough notation and generator description to make the comparison understandable; put full sampling distributions and implementation-level validation in Appendix A.

### 3.1 Problem formulation

Define support/query prediction, latent regime membership, regime-dependent label mechanisms, and the information available to the predictor.

### 3.2 Native single-regime control

Describe TabICL's native SCM sampling and `Reg2Cls` label generation. Establish this prior as the matched control for all multiregime comparisons.

Cite TabICL [@qu2025tabicl] and record the exact source-code revision for implementation-level behavior.

### 3.3 Multiregime mechanisms

- **`r_z`:** Regimes use different SCM score functions.
- **`g_z`:** Regimes apply independently sampled label mappings to a shared score.
- **`rg_z`:** Pretraining mixes both mechanisms.

### 3.4 Routing and pretraining schedules

Describe feature-dependent `soft_gate` routing and latent `persistent` routing.

Compare native single-regime pretraining against:

- Fixed multiregime mixing at 30%.
- Curriculum mixing increasing to 50%.

Document routing-score exposure in 50% of training episodes. Use the score-hidden test bank for primary results and score-exposed evaluation as an ablation. Hiding the additional routing score does not imply that the features contain no routing information.

Use conditional-mixture literature for gating terminology [@jacobs1991experts; @jordan1994hierarchical] and curriculum learning for schedule context [@bengio2009curriculum]. Neither establishes the effectiveness of our particular prior or schedule.

**Planned main-text Figure 1 — not yet attached:** A compact conceptual illustration of native task generation, regime mechanisms, and routing conditions. The existing detailed empirical figures are preserved in Appendices C–E.

## 4. Experimental Setup

Target: **0.75 page**. Keep the controls, three architecture sizes, final-checkpoint reporting policy, score-hidden primary evaluation, task counts, baselines, and metric definitions here. Put full hyperparameter, checkpoint, preprocessing, and task/fold inventories in Appendices B and E.

### 4.1 Architectures and training controls

Evaluate two-, four-, and six-layer NanoTabPFN variants. Specify matched initialization, training conditions, validation procedures, and checkpoint selection. Describe all architecture settings that vary across sizes, so the comparison is not presented as a depth-only ablation.

Cite NanoTabPFN [@pfefferle2025nanotabpfn] and document the implementation revision and local changes.

### 4.2 Synthetic evaluation

Vary regime count, class count, binary class imbalance, support size, feature count, and routing family. Include single-rule and shared-rule controls. Compare models on identical native-bank episodes, with episode-weighted aggregation.

### 4.3 BeyondArena evaluation

Introduce BeyondArena with its own benchmark reference [@purucker2026beyondarena]. The following counts describe our selected evaluation subset, not the full benchmark.

Evaluate 36 tasks:

- **Binary:** 26 IID and one grouped.
- **Multiclass:** six IID, two grouped, and one temporal.

Explain official folds, preprocessing, context construction, and equal task weighting. Distinguish 36-task score summaries from rankings that exclude the temporal task. State the comparison set used for each ranking.

### 4.4 Baselines and metrics

Compare against published TabPFN/TabICL models and conventional classifiers.

Cite the corresponding model papers from Section 2.1; add the official model card for TabPFN v2.6 release provenance [@priorlabs2026tabpfn26]. Record exact checkpoint and package versions separately. Cite random forests [@breiman2001forests], XGBoost [@chen2016xgboost], LightGBM [@ke2017lightgbm], CatBoost [@prokhorenkova2018catboost], and scikit-learn for the relevant implementations [@pedregosa2011sklearn].

Report accuracy gain over majority prediction, excess cross-entropy relative to the support-class-frequency predictor, and macro one-versus-rest AUC. Express accuracy gains in percentage points and cross-entropy in nats.

Use proper scoring rules as background for cross-entropy [@gneiting2007scoring]. Do not describe lower cross-entropy alone as proof of better calibration. Dataset-level rank summaries are descriptive unless a statistical comparison is explicitly performed. [@demsar2006comparisons]

## 5. Synthetic Results

Target: **1.75 pages**, including the compact final-checkpoint Figure 2 and Table 1. Prioritize the matched multiregime-versus-single-regime comparison, routing dependence, and published baselines. Keep capacity and scheduling findings concise; move checkpoint trajectories and exploratory regime ranking to Appendices C and D.

### 5.1 Primary synthetic result: feature-dependent multiclass routing

Lead with the six-layer, score-hidden **soft-gate multiregime-multiclass** condition—the main synthetic achievement—before presenting the pooled result. Its excess CE improves from **−0.1562 to −0.1730**.

- On the matched soft-gate multiregime-multiclass subset, curriculum changes cross-entropy by **−0.0168 nats** (95% interval **[−0.0179, −0.0158]**) and accuracy gain by **+0.84 points** (**[+0.76, +0.91]**). Persistent-routing effects are much smaller and schedule-dependent: fixed mixing changes cross-entropy by **−0.0014 nats** (**[−0.0021, −0.0006]**) and accuracy gain by **+0.07 points** (**[+0.00, +0.13]**), whereas curriculum changes cross-entropy by −0.0008 nats and accuracy gain by −0.07 points.
- Binary tasks show no consistent benefit, while persistent-routing multiclass tasks show a small fixed-mixture improvement rather than the pronounced soft-gate effect.

### 5.2 Overall multiregime multiclass result and capacity

After the primary soft-gate result, report the pooled six-layer result across score-hidden multiregime multiclass tasks:

**Main-text Table 1, native-model core:** retain the three rows below and add the published comparators on this exact subset before drafting the final table. Do not substitute pooled-bank reference scores for multiregime-multiclass scores.

| Pretraining prior | Excess cross-entropy ↓ |
|---|---:|
| Native single-regime | −0.0838 |
| Native multiregime—fixed | −0.0914 |
| Native multiregime—curriculum | −0.0926 |

Accuracy gain rises from **6.30 percentage points** for the native control to **6.69 points** for the curriculum model.

Paired episode-bootstrap analysis of the **six-layer** final checkpoints on this fixed score-hidden test subset (**5,760 episodes in 720 factorial cells**) confirms the curriculum comparison: candidate minus canonical cross-entropy is **−0.0088 nats** (95% percentile interval **[−0.0094, −0.0082]**), and the accuracy-gain difference is **+0.38 percentage points** (**[+0.34, +0.43]**). Because the support-class-frequency baseline is shared within every matched episode, the cross-entropy difference is also the excess-cross-entropy difference.

The matched fixed-mixture run shows a similar but slightly smaller change (−0.0076 nats; +0.37 points). These intervals quantify finite evaluation-episode variation, not uncertainty across independent pretraining seeds. Full paired results, including routing and binary slices, are in [paired episode-bootstrap results](paired_bootstrap.md).

The interpretation is that native single-regime pretraining already supports learning these tasks, while explicit multiregime exposure improves performance on this synthetic evaluation distribution.

- Four- and six-layer models show clearer pretraining effects than two-layer models.

### 5.3 Fixed mixing versus curriculum

Compare both schedules without claiming universal curriculum superiority. Six-layer pooled test CE is nearly identical: **0.7802** for fixed mixing and **0.7803** for curriculum.

Acknowledge that exposure proportions differ, so this comparison does not isolate scheduling alone.

### 5.4 Published-model comparisons

Compare all models on the same native evaluation bank and task subsets. Separate pooled results from multiregime-specific results, and distinguish accuracy from likelihood performance.

**Planned main-text Figure 2 — not yet attached:** Compact final-checkpoint results that make the multiregime effect visible across routing families, binary versus multiclass tasks, and the two-/four-/six-layer variants. Show matched-control differences and clearly identify the evaluation subset. Use Table 1 for direct published-model comparisons without repeating the entire figure's numbers.

The full cell/checkpoint grid is attached as Figure A1 in Appendix C.

### 5.5 Within-episode regime analysis

Limit the main text to a brief pointer to the exploratory analysis in Appendix D. Do not rely on regime-ranked results to establish the central claim, and retain the prediction-versus-regime-recovery distinction in the main-text limitations.

Source for Sections 5.1–5.4: [Native-prior results](native_prior_results.md). Source for Section 5.5: [Within-episode regime breakdown](regime_breakdown.md).

## 6. Real-World Transfer

Target: **1.25 pages**, including Table 2. The absence of consistent transfer gains is a central result, not an appendix-only caveat. Keep both overall performance and IID/grouped/temporal summaries, with canonical and published comparators; move per-task tables, the native-only rank heatmap, and the full site-information case study to Appendices E and F.

### 6.1 Overall, binary, and multiclass performance

Present the six-layer models alongside conventional and published baselines:

**Main-text Table 2, panel A — overall and target-type summaries:**

| Model | Overall excess CE ↓ | Binary accuracy gain | Multiclass accuracy gain |
|---|---:|---:|---:|
| Native single-regime | −0.224 | 9.4 pp | 20.0 pp |
| Native multiregime—fixed | −0.221 | 9.3 pp | 20.2 pp |
| Native multiregime—curriculum | −0.220 | 9.2 pp | 19.9 pp |
| Random forest | −0.225 | 9.7 pp | 21.7 pp |
| Published TabICL v2 | −0.273 | 11.2 pp | 23.0 pp |
| Published TabPFN v3 | −0.267 | 11.0 pp | 22.8 pp |

These are rounded, equal-task scores over 36 tasks overall, 27 binary tasks, and nine multiclass tasks. The multiclass aggregate includes the temporal task.

Emphasize comparable performance among the native-prior variants, rather than consistent real-world gains from multiregime pretraining. Published models retain an advantage on these aggregate metrics.

### 6.2 IID, grouped, and temporal evaluation

**Main-text Table 2, panel B — split-specific summaries:** report binary IID (26 tasks), binary grouped (one), multiclass IID (six), multiclass grouped (two), and multiclass temporal (one). Retain the matched native control, multiregime variants, and published comparators. Use interpretable score differences rather than only ranks. Populate this panel from the underlying evaluation aggregates when assembling the manuscript; the current table above is panel A only.

Report the qualitative split comparison in the main text and put full per-task differences in Appendix E. Avoid broad conclusions from the single grouped binary task or single temporal task.

The native-only rank heatmap is preserved as Figure A3 in Appendix E; it is supplementary and cannot replace the main-text published-model comparison.

### 6.3 Group-information case study

Give at most a brief pointer to Appendix F: the merged heart-site experiment examines explicit group information under IID and leave-one-site-out evaluation, without establishing a consistent additional benefit from multiregime pretraining. Keep the full case study outside the main-text page budget.

Sources: [Native-prior BeyondArena results](native_prior_results.md) and [Merged heart-disease sites](heart_sites_group_feature.md).

## 7. Discussion and Limitations

Target: **0.75 page**. Keep the following limitations in the main text even if supporting analyses move to appendices.

- Distinguish successful prediction from recovery of latent regimes; mixture identifiability depends on assumptions not established by our prediction metrics. [@jiang1999identifiability]
- Explain the concentration of gains in feature-dependent multiclass tasks.
- Discuss the gap between synthetic improvements and real-world transfer.
- Acknowledge single-seed training, small effect sizes, limited grouped-task coverage, and training-time routing-score exposure.
- Separate uncertainty over evaluation episodes or tasks from uncertainty over independent training runs. In particular, paired bootstrap intervals over the fixed synthetic bank do not replace multi-seed pretraining.
- Do not interpret neutral transfer as evidence that real datasets lack regime structure.

## 8. Conclusion

Target: **0.25 page**. One concise paragraph; do not repeat every numerical result.

Compact PFNs can learn multiregime multiclass tasks, and explicit multiregime pretraining improves prediction particularly under feature-dependent routing. However, the experiments do not establish a consistent real-world advantage over matched native single-regime pretraining.

## References

Outside the main-text page budget. Generate the manuscript bibliography from [references.bib](references.bib). Place the complete bibliography before all appendices; the reference insertion point below also supports that order when rendering this Markdown with Pandoc.

<div id="refs"></div>

## Appendices

Outside the main-text page budget and placed after the bibliography. Reviewers need not read them, so no appendix should contain the only explanation or evidence for a headline claim. Keep all empirical material restricted to native-prior experiments and their comparison baselines.

### Appendix A. Native prior and implementation validation

Full SCM and label-mapping generation details, sampling distributions, routing definitions, class-count and imbalance handling, non-finite-episode handling, and native-prior validation tests. Record the exact source revision and explain how the single-regime case matches the native control.

### Appendix B. Architectures and training reproducibility

Full two-/four-/six-layer configuration tables, initialization and seed settings, optimizer and schedule details, mixing fractions, training-time routing-score exposure, checkpoint-selection rules, software versions, and published checkpoint identifiers. Essential control choices remain summarized in Sections 3–4.

### Appendix C. Complete synthetic results and training trajectories

Per-cell results, binary and multiclass breakdowns, regime counts, support-size and feature-count variations, score-exposure ablations, all size variants, and full checkpoint trajectories. Include fixed-mixture and curriculum comparisons without claiming that exposure proportions isolate scheduling effects.

![Native-prior six-layer NanoTabPFN test excess cross-entropy across task conditions, with published and conventional baselines.](../../figures/native/cells/native_large_excess_ce_cells_test.png)

*Figure A1. Native-prior six-layer test excess cross-entropy by task condition and checkpoint. Zero denotes the class-frequency baseline; lower values are better. Colors distinguish the native single-regime, fixed-mixture, and curriculum models; solid and dashed curves distinguish score-hidden and score-exposed banks. Horizontal lines show reference models evaluated on the native bank. Final-checkpoint results are marked by diamonds. Checkpoint trajectories are descriptive, not a test-set model-selection rule.*

Source: [Native-prior results](native_prior_results.md).

### Appendix D. Exploratory within-episode regime analysis

Analyze performance across easier and harder regimes. For four-regime soft-gate multiclass tasks, accuracy on the native control's hardest-ranked regime increases from **38.36% to 40.23%** with curriculum pretraining.

Treat rank-conditioned comparisons as exploratory because ranking by observed baseline loss introduces selection effects. These analyses do not establish recovery of the underlying regime assignments.

![Paired per-regime cross-entropy differences between six-layer native multiregime models and the native single-regime control.](../../figures/native/regime_ranked/native_large_test_regime_ranked_ce_delta.png)

*Figure A2. Paired cross-entropy differences against the native single-regime six-layer model, by routing family, class type, regime count, and within-episode regime rank. Negative values favor multiregime pretraining. Regimes are ranked by the control's observed cross-entropy, with rank 1 its best-fitting regime. Whiskers show the plotted 95% confidence intervals across episodes; they do not quantify variation across training seeds. The concentrated gains in soft-gate multiclass tasks contrast with the small effects in binary and persistent-routing tasks.*

Source: [Within-episode regime breakdown](regime_breakdown.md).

### Appendix E. BeyondArena task inventory and complete results

Exact data revision, task/fold list, preprocessing and context construction, per-task metrics, split-specific results, all native model sizes, conventional/published comparisons, and supplementary rankings. Explain the 36-task score versus 35-task rank distinction and state the model comparison set for each ranking.

![BeyondArena mean task ranks for the nine native-prior NanoTabPFN runs, separated by binary or multiclass targets and IID or grouped evaluation.](../../figures/beyondarena_v4/beyondarena_heatmap_rank_native.png)

*Figure A3. BeyondArena mean task ranks among the nine native-prior NanoTabPFN runs only; lower is better. Columns separate overall, binary IID, binary grouped, multiclass IID, and multiclass grouped evaluation. The overall panel contains 35 tasks and excludes the temporal task. These ranks therefore use a different task set from the 36-task score table in Section 6.1 and do not rank the native models against published or conventional baselines.*

Source: [Native-prior BeyondArena results](native_prior_results.md).

### Appendix F. Group-information case study

Identify the original UCI Heart Disease data source [@janosi1989heart], then document our site selection and evaluation protocol separately.

Use the merged heart-disease sites experiment to examine whether explicit group information helps:

- Site information has little effect under IID evaluation.
- It improves cross-entropy for several models under leave-one-site-out evaluation.
- Multiregime pretraining does not provide a consistent additional advantage.

Report native-prior NanoTabPFN runs and published/conventional comparators only, and identify any unavailable model variants rather than implying complete coverage.

Source: [Merged heart-disease sites](heart_sites_group_feature.md).
