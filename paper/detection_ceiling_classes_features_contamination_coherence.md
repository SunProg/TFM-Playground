# Detection ceiling: classes x features x contamination x coherence

Freshly measured with `detection_ceiling.measure()` (this repo), classification target, support=128, 20 episodes/cell, seed=11. The ceiling is the AUC of the strongest simple cross-fit detector (`HistGradientBoostingClassifier`) at telling contaminated rows apart from base-regime rows -- never given the true tags, but given each row's own `(x, y)` pair.

**This is a ceiling for a detector that gets to see labels, so it is the right comparison for `support_binding_auc`** (`pretrain_slot_tabpfn.py:470-504` -- scores slot competition against the regime tag on *support* rows, which come with visible `y`). **It is not a valid comparison for `validation_gate_regime_auc`.** That metric is computed on the *query* set, and a query row's true `y` is the model's prediction target -- never an input to the gate. `prediction.gate()` (`slot_regime.py:85-86`) can only be a function of query `x` plus whatever the model inferred from the labeled support set; it structurally cannot do what this file's detector does (compare a row's actual label against a fit). An earlier version of this file compared `gate_regime_auc` against these numbers anyway (first via `unrestricted`, then via `restricted`); both were wrong, not just on which column to use. The right ceiling for `gate_regime_auc` would be an **x-only** probe -- fit tag directly from `x` alone, no `y` -- which is a different measurement this file does not contain (`CONTROL_COHERENCE`'s documented 0.980 supervised x-> tag AUC, at coherence=8.0, is that kind of measurement, but not at coherence=2.0, the design actually trained on).

`measure()` returns two statistics per cell. **`restricted`** scores the detector only on rows where the two label functions (regime tag vs. the piecewise/contamination rule) actually disagree -- i.e. rows where detection is in principle possible. **`unrestricted`** folds in the rows where the rules happen to collide (correct by construction, AUC-neutral) via `identifiable_mean + (1 - identifiable_mean) * 0.5`. An earlier version of this file reported only `unrestricted` and matched it against the codebase's documented "0.742" figure; that comparison was wrong. A targeted 60-episode check at the matched design point (classes=3, features=4, contamination=0.15) showed **`restricted` is the statistic that lines up with 0.742** (`restricted=0.7418+/-0.0369` at coherence=0.0, n=60), while `unrestricted` reads meaningfully lower (`0.6751+/-0.0345` at the same point). This rewrite reports `restricted` as the headline ceiling and keeps `unrestricted` alongside for transparency.

The stored `ceiling_cells()` grid in `detection_ceiling.py` never varies `coherence` for classification cells (always defaults to 0.0), even though every real training run in this project used `regime_coherence=2.0` (`REGIME_COHERENCE`) or `8.0` (`CONTROL_COHERENCE`, the positive control). The documented "0.742" figure for `LEARNABLE_DESIGN` (`slot_tabpfn_sweep.py:128`) is therefore a coherence=0.0 measurement, not one at the coherence=2.0 actually used to train every `table_slot_head` arm. This sweep fills that gap directly, at a denser 8-point coherence grid (0, 1, 2, 3, 4, 5, 6, 8) crossed with classes {2,3,4,5} x features {4,12} x contamination {0.15,0.30} -- 128 cells total.

## Full grid

**classes=2, features=4, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6281 | +/-0.0595 | 0.5438 | +/-0.0465 |
| 1.0 | 0.5630 | +/-0.0578 | 0.5197 | +/-0.0380 |
| 2.0 | 0.5653 | +/-0.0677 | 0.5226 | +/-0.0530 |
| 3.0 | 0.5311 | +/-0.0676 | 0.5103 | +/-0.0490 |
| 4.0 | 0.5428 | +/-0.0700 | 0.5110 | +/-0.0524 |
| 5.0 | 0.5236 | +/-0.0729 | 0.5105 | +/-0.0528 |
| 6.0 | 0.5320 | +/-0.0679 | 0.5168 | +/-0.0542 |
| 8.0 | 0.5200 | +/-0.0758 | 0.4980 | +/-0.0579 |

**classes=2, features=4, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.5622 | +/-0.0416 | 0.5173 | +/-0.0267 |
| 1.0 | 0.5577 | +/-0.0537 | 0.5158 | +/-0.0389 |
| 2.0 | 0.5314 | +/-0.0590 | 0.5085 | +/-0.0432 |
| 3.0 | 0.5632 | +/-0.0551 | 0.5295 | +/-0.0453 |
| 4.0 | 0.5528 | +/-0.0550 | 0.5188 | +/-0.0460 |
| 5.0 | 0.5625 | +/-0.0529 | 0.5298 | +/-0.0459 |
| 6.0 | 0.5532 | +/-0.0554 | 0.5239 | +/-0.0485 |
| 8.0 | 0.5359 | +/-0.0530 | 0.5163 | +/-0.0460 |

**classes=2, features=12, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.5212 | +/-0.0636 | 0.5195 | +/-0.0356 |
| 1.0 | 0.4937 | +/-0.0420 | 0.4928 | +/-0.0378 |
| 2.0 | 0.4858 | +/-0.0285 | 0.4831 | +/-0.0269 |
| 3.0 | 0.4867 | +/-0.0426 | 0.4956 | +/-0.0352 |
| 4.0 | 0.5096 | +/-0.0468 | 0.5040 | +/-0.0306 |
| 5.0 | 0.5214 | +/-0.0529 | 0.4949 | +/-0.0296 |
| 6.0 | 0.5205 | +/-0.0458 | 0.4884 | +/-0.0297 |
| 8.0 | 0.5292 | +/-0.0453 | 0.4988 | +/-0.0312 |

**classes=2, features=12, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.4889 | +/-0.0259 | 0.4993 | +/-0.0245 |
| 1.0 | 0.4883 | +/-0.0347 | 0.4928 | +/-0.0262 |
| 2.0 | 0.4891 | +/-0.0324 | 0.4981 | +/-0.0264 |
| 3.0 | 0.4894 | +/-0.0239 | 0.4944 | +/-0.0254 |
| 4.0 | 0.4790 | +/-0.0343 | 0.4930 | +/-0.0297 |
| 5.0 | 0.4778 | +/-0.0341 | 0.4870 | +/-0.0346 |
| 6.0 | 0.4932 | +/-0.0336 | 0.4988 | +/-0.0272 |
| 8.0 | 0.4802 | +/-0.0324 | 0.4887 | +/-0.0262 |

**classes=3, features=4, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.7799 | +/-0.0558 | 0.6975 | +/-0.0520 |
| 1.0 | 0.7337 | +/-0.0589 | 0.6568 | +/-0.0466 |
| 2.0 | 0.6841 | +/-0.0725 | 0.6306 | +/-0.0550 |
| 3.0 | 0.6839 | +/-0.0609 | 0.6463 | +/-0.0428 |
| 4.0 | 0.6900 | +/-0.0598 | 0.6465 | +/-0.0459 |
| 5.0 | 0.6755 | +/-0.0681 | 0.6357 | +/-0.0542 |
| 6.0 | 0.6782 | +/-0.0693 | 0.6338 | +/-0.0542 |
| 8.0 | 0.6591 | +/-0.0765 | 0.6103 | +/-0.0602 |

**classes=3, features=4, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6269 | +/-0.0562 | 0.5475 | +/-0.0411 |
| 1.0 | 0.6597 | +/-0.0493 | 0.6034 | +/-0.0344 |
| 2.0 | 0.6298 | +/-0.0544 | 0.5896 | +/-0.0381 |
| 3.0 | 0.6204 | +/-0.0599 | 0.5800 | +/-0.0430 |
| 4.0 | 0.6255 | +/-0.0581 | 0.5866 | +/-0.0431 |
| 5.0 | 0.6263 | +/-0.0549 | 0.5828 | +/-0.0410 |
| 6.0 | 0.6207 | +/-0.0536 | 0.5842 | +/-0.0403 |
| 8.0 | 0.6216 | +/-0.0531 | 0.5814 | +/-0.0399 |

**classes=3, features=12, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.7241 | +/-0.0712 | 0.6515 | +/-0.0546 |
| 1.0 | 0.6996 | +/-0.0729 | 0.6177 | +/-0.0697 |
| 2.0 | 0.6625 | +/-0.0757 | 0.5993 | +/-0.0686 |
| 3.0 | 0.6459 | +/-0.0733 | 0.5860 | +/-0.0729 |
| 4.0 | 0.6604 | +/-0.0799 | 0.5987 | +/-0.0719 |
| 5.0 | 0.6451 | +/-0.0796 | 0.5879 | +/-0.0732 |
| 6.0 | 0.6496 | +/-0.0774 | 0.5806 | +/-0.0714 |
| 8.0 | 0.6509 | +/-0.0780 | 0.5806 | +/-0.0698 |

**classes=3, features=12, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6237 | +/-0.0780 | 0.5685 | +/-0.0586 |
| 1.0 | 0.6261 | +/-0.0686 | 0.5650 | +/-0.0613 |
| 2.0 | 0.6001 | +/-0.0691 | 0.5382 | +/-0.0576 |
| 3.0 | 0.5831 | +/-0.0739 | 0.5229 | +/-0.0634 |
| 4.0 | 0.5846 | +/-0.0750 | 0.5244 | +/-0.0636 |
| 5.0 | 0.5758 | +/-0.0669 | 0.5167 | +/-0.0598 |
| 6.0 | 0.5665 | +/-0.0695 | 0.5121 | +/-0.0599 |
| 8.0 | 0.5759 | +/-0.0697 | 0.5211 | +/-0.0610 |

**classes=4, features=4, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.7102 | +/-0.0567 | 0.6545 | +/-0.0625 |
| 1.0 | 0.7206 | +/-0.0593 | 0.6806 | +/-0.0533 |
| 2.0 | 0.6911 | +/-0.0675 | 0.6555 | +/-0.0560 |
| 3.0 | 0.6768 | +/-0.0674 | 0.6503 | +/-0.0618 |
| 4.0 | 0.6724 | +/-0.0696 | 0.6490 | +/-0.0609 |
| 5.0 | 0.6644 | +/-0.0744 | 0.6398 | +/-0.0642 |
| 6.0 | 0.6576 | +/-0.0792 | 0.6304 | +/-0.0679 |
| 8.0 | 0.6367 | +/-0.0795 | 0.6159 | +/-0.0706 |

**classes=4, features=4, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.5949 | +/-0.0618 | 0.5172 | +/-0.0391 |
| 1.0 | 0.6638 | +/-0.0594 | 0.6237 | +/-0.0526 |
| 2.0 | 0.6531 | +/-0.0611 | 0.6246 | +/-0.0585 |
| 3.0 | 0.6431 | +/-0.0573 | 0.6188 | +/-0.0539 |
| 4.0 | 0.6200 | +/-0.0611 | 0.6055 | +/-0.0595 |
| 5.0 | 0.6236 | +/-0.0567 | 0.6069 | +/-0.0589 |
| 6.0 | 0.6213 | +/-0.0572 | 0.6001 | +/-0.0579 |
| 8.0 | 0.6321 | +/-0.0603 | 0.6158 | +/-0.0589 |

**classes=4, features=12, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6804 | +/-0.0663 | 0.6284 | +/-0.0598 |
| 1.0 | 0.6791 | +/-0.0632 | 0.6343 | +/-0.0635 |
| 2.0 | 0.6352 | +/-0.0645 | 0.5924 | +/-0.0611 |
| 3.0 | 0.6236 | +/-0.0562 | 0.5873 | +/-0.0573 |
| 4.0 | 0.6264 | +/-0.0616 | 0.5939 | +/-0.0600 |
| 5.0 | 0.6271 | +/-0.0556 | 0.5922 | +/-0.0545 |
| 6.0 | 0.6235 | +/-0.0549 | 0.5907 | +/-0.0555 |
| 8.0 | 0.6236 | +/-0.0602 | 0.5912 | +/-0.0636 |

**classes=4, features=12, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.5947 | +/-0.0592 | 0.5626 | +/-0.0443 |
| 1.0 | 0.6037 | +/-0.0544 | 0.5625 | +/-0.0488 |
| 2.0 | 0.5747 | +/-0.0576 | 0.5410 | +/-0.0536 |
| 3.0 | 0.5595 | +/-0.0561 | 0.5203 | +/-0.0516 |
| 4.0 | 0.5608 | +/-0.0568 | 0.5259 | +/-0.0533 |
| 5.0 | 0.5479 | +/-0.0606 | 0.5145 | +/-0.0556 |
| 6.0 | 0.5384 | +/-0.0578 | 0.5083 | +/-0.0522 |
| 8.0 | 0.5253 | +/-0.0595 | 0.4969 | +/-0.0539 |

**classes=5, features=4, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6987 | +/-0.0553 | 0.6648 | +/-0.0556 |
| 1.0 | 0.7295 | +/-0.0652 | 0.6780 | +/-0.0551 |
| 2.0 | 0.6958 | +/-0.0602 | 0.6628 | +/-0.0479 |
| 3.0 | 0.6711 | +/-0.0711 | 0.6401 | +/-0.0592 |
| 4.0 | 0.6756 | +/-0.0677 | 0.6455 | +/-0.0573 |
| 5.0 | 0.6654 | +/-0.0724 | 0.6400 | +/-0.0583 |
| 6.0 | 0.6667 | +/-0.0744 | 0.6334 | +/-0.0643 |
| 8.0 | 0.6473 | +/-0.0734 | 0.6122 | +/-0.0629 |

**classes=5, features=4, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6066 | +/-0.0559 | 0.5650 | +/-0.0439 |
| 1.0 | 0.6633 | +/-0.0595 | 0.6198 | +/-0.0470 |
| 2.0 | 0.6434 | +/-0.0621 | 0.6162 | +/-0.0503 |
| 3.0 | 0.6406 | +/-0.0630 | 0.6188 | +/-0.0515 |
| 4.0 | 0.6308 | +/-0.0616 | 0.6104 | +/-0.0501 |
| 5.0 | 0.6187 | +/-0.0647 | 0.5989 | +/-0.0530 |
| 6.0 | 0.6153 | +/-0.0671 | 0.5971 | +/-0.0562 |
| 8.0 | 0.6287 | +/-0.0633 | 0.6092 | +/-0.0512 |

**classes=5, features=12, contamination=0.15**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6475 | +/-0.0565 | 0.5913 | +/-0.0536 |
| 1.0 | 0.6600 | +/-0.0643 | 0.6225 | +/-0.0611 |
| 2.0 | 0.6088 | +/-0.0612 | 0.5759 | +/-0.0569 |
| 3.0 | 0.5904 | +/-0.0579 | 0.5612 | +/-0.0562 |
| 4.0 | 0.6090 | +/-0.0542 | 0.5838 | +/-0.0516 |
| 5.0 | 0.6162 | +/-0.0532 | 0.5970 | +/-0.0523 |
| 6.0 | 0.6144 | +/-0.0493 | 0.5909 | +/-0.0507 |
| 8.0 | 0.5968 | +/-0.0526 | 0.5761 | +/-0.0560 |

**classes=5, features=12, contamination=0.3**

| coherence | ceiling AUC (restricted) | 95% CI | unrestricted AUC | 95% CI |
|---|---|---|---|---|
| 0.0 | 0.6093 | +/-0.0527 | 0.5892 | +/-0.0474 |
| 1.0 | 0.5991 | +/-0.0445 | 0.5642 | +/-0.0386 |
| 2.0 | 0.5663 | +/-0.0498 | 0.5455 | +/-0.0461 |
| 3.0 | 0.5403 | +/-0.0524 | 0.5201 | +/-0.0488 |
| 4.0 | 0.5365 | +/-0.0510 | 0.5182 | +/-0.0469 |
| 5.0 | 0.5240 | +/-0.0580 | 0.5035 | +/-0.0528 |
| 6.0 | 0.5229 | +/-0.0525 | 0.5010 | +/-0.0482 |
| 8.0 | 0.5180 | +/-0.0509 | 0.5012 | +/-0.0485 |

## Achieved, for comparison

### `support_binding_auc` -- the metric this file's ceiling actually grades

`support_binding_scores()` (`pretrain_slot_tabpfn.py:455-519`) scores slot competition against the regime tag on **support** rows, and its `identifiable` filtering (`keep = distinct | (labels==0)`) is the exact same formula `detection_ceiling`'s `restricted` uses. Support rows carry visible `y`, so this is a fair, apples-to-apples comparison against the `restricted` ceiling above -- no caveat needed.

It is logged during training as `validation_support_binding_auc` for every `table_slot_head` arm, but was never pulled into the dashboard (`fetch_15k.py`'s `VAL_KEYS` list omits it -- the same class of gap as vanilla's missing metrics). Pulled directly from `history.jsonl` on the cluster:

| model | best `support_binding_auc` achieved | step |
|---|---|---|
| mixed, base, decoder, data | 0.6476 | 3000 |
| curriculum, base, blind_similarity, data | 0.6449 | 2000 |
| curriculum, base, decoder, data | 0.6381 | 6000 |

Against the coherence=2.0 restricted ceiling (0.6841 +/-0.0725, corroborated at 0.6874 +/-0.0462): the best arm's peak (0.6476) sits comfortably inside that interval, recovering roughly (0.6476-0.5)/(0.684-0.5) &asymp; **80%** of the available signal above chance. This is a valid, ceiling-relative result, unlike anything involving `gate_regime_auc` below.

**37 of the 40 trained arms show the same peak-then-decline pattern** on this metric, independent of `gate_regime_auc` entirely -- including the best one (0.6476 at step 3000, down to 0.6231 by step 10000). So the peak-then-decline finding (see below) is not an artifact of one query-side metric; it shows up on the support-side, ceiling-comparable metric too.

### `gate_regime_auc` -- still has no valid ceiling in this file

From the compositing sweep, the same design point, best achieved:

| model | best `gate_regime_auc` achieved | step |
|---|---|---|
| mixed, base, blind_similarity, both | 0.5990 | 10500 |
| mixed, base, blind_similarity, both | 0.5963 | 9000 |
| plain, base, blind_similarity, both | 0.5818 | 4500 |

**No number in this file's grid is a valid ceiling for these.** `gate_regime_auc` is a query-side metric: the model's gate never sees a query row's true `y`, only `x` plus whatever it inferred from the labeled support set. Every ceiling in this file's grid comes from a detector that *is* given each row's own `(x, y)` pair -- fine for `support_binding_auc` above, not for this. Two earlier drafts of this file compared them anyway (once via `unrestricted`, once via `restricted`) and both were invalid, not just imprecise.

**A genuine ceiling for `gate_regime_auc` would need a different measurement**: fit the regime tag directly from query `x` alone (no `y`), cross-validated, at coherence=2.0 -- an x-only probe, not the `(x,y)` misfit detector this file uses. `CONTROL_COHERENCE`'s documented 0.980 figure is that kind of measurement, but at coherence=8.0, not the coherence=2.0 the compositing sweep trains on. Until that measurement exists at coherence=2.0, "is 0.599 close to the ceiling" has no answer here.

## Reading the coherence axis

For **classes=2**, `restricted` ceiling AUC is uniformly weak (0.52-0.63 at 4 features, 0.48-0.53 at 12 features) and drifts gently downward as coherence rises, but the whole range sits close enough to chance (0.5) that the trend is dwarfed by the CI width (+/-0.04 to +/-0.06) -- there's essentially nothing to detect here regardless of coherence.

For **classes>=3**, the picture is more interesting: most (classes, features, contamination=0.15) rows show a real step down from coherence=0.0 to coherence=1.0-2.0, then flatten out for the rest of the axis. For example: classes=3/features=4 goes 0.780 -> 0.734 -> 0.684 -> ... -> 0.659 (coherence 0 through 8); classes=4/features=4 goes 0.710 -> 0.721 -> 0.691 -> ... -> 0.637; classes=5/features=4 goes 0.699 -> 0.730 -> 0.696 -> ... -> 0.647. The drop from the coherence=0 peak to the coherence>=3 plateau is roughly 0.06-0.12 AUC across these rows -- a real, if modest, effect, not noise (the CIs at +/-0.04 to +/-0.07 don't fully cover gaps this size in most rows). At **contamination=0.30** the same rows are flatter and noisier (e.g. classes=3/features=4: 0.627, 0.660, 0.630, 0.620, 0.625, 0.626, 0.621, 0.622) -- the coherence effect is specific to the lower-contamination, higher-class-count corner of the grid, not a general property of `regime_coherence`.

This refines the codebase's own prior note (`detection_ceiling.py`: "`regime_coherence` alone moved the ceiling from 0.540 to 0.507" for a *regression* mechanism): for classification, coherence does move the classes>=3/contamination=0.15 ceiling down by a similar-sized amount, but only in that corner of the grid -- classes=2 and contamination=0.30 stay essentially flat across the whole coherence axis.

**Why coherence lowers this ceiling instead of raising it.** `CONTROL_COHERENCE`'s own docstring (`slot_tabpfn_sweep.py:117`) reports measured *supervised* x-> tag AUC of 0.980 at coherence=8.0 -- i.e. given the true tags, contaminated rows are almost perfectly separable by x alone, because coherence pushes them onto one side of a per-episode hyperplane. That is a different question from what `detection_ceiling.measure()` answers here: its detector is never given the tag, only `(x, y)`, and cross-fits to flag rows whose *label* the fit misses. Concentrating contaminated rows into one region of x-space does not help that detector -- if anything it hurts, since misfit-based detection relies on a contaminated row standing out against its neighbors, and at high coherence its neighbors are increasingly other contaminated rows rather than clean ones. So coherence raises the ceiling for a tag-supervised, x-only probe (like the 0.980 figure) and lowers it for this file's tag-blind, `(x,y)`-based detector -- two different probes moving in opposite directions, not a contradiction. (Neither is the same probe `gate_regime_auc` needs -- see above.)

Feature count (4 vs 12) does not reliably raise the ceiling -- if anything the 12-feature cells trend at or below their 4-feature counterparts across most of the grid (e.g. classes=3, contamination=0.15: 4 features peaks at 0.780, 12 features peaks at 0.724; classes=5, contamination=0.15: 4 features peaks at 0.730, 12 features peaks at 0.660), the opposite of what "more information should make detection easier" would predict.

## Peak-then-decline: confirmed on the ceiling-comparable metric too

`gate_regime_auc` peaks and declines on nearly every arm (36 of 36 `table_slot_head` arms; best case `mixed, base, similarity, both`: 0.599 at step 10500, down to 0.550 by step 15000). Since that metric has no valid ceiling in this file, on its own this could just mean the peak was noise. It is not: **`support_binding_auc` -- which does have a valid, matched ceiling -- shows the same pattern on 37 of 40 arms**, including its own best case (0.6476 at step 3000, down to 0.6231 by step 10000, comfortably inside the ceiling's CI at the peak). Two structurally different metrics (query-side, `y`-blind vs. support-side, `y`-visible), both reach near their respective peaks early and both erode afterward. That rules out "the query-side metric just never gets close to its ceiling" as the explanation, and points instead at training dynamics: nothing in the loss rewards keeping either quantity high, so there is no mechanism holding a model at a transient high once training pushes representations toward whatever the actual loss wants instead.

Tracking and checkpointing on these metrics specifically (the way `pretrain_slot_tabpfn.py` already does for `query_ce`/`multiregime_ce`) would recover the peak for both, regardless of where `gate_regime_auc`'s own ceiling turns out to sit.

## Caveats

- 20 episodes/cell is enough to see the shape but the 95% CIs are wide (+/-0.03 to +/-0.07 for the `restricted` metric, since it's computed over a smaller identifiable-rows-only subset) -- individual cell-to-cell wiggles of that size are noise, not signal.
- classes=2 cells throw sklearn stratified-fold warnings at several (contamination, coherence) combinations where a fold ends up with only 1 example of the minority class; the reported means still come from valid folds but are noisier there.
- **This whole file measures a `(x,y)`-visible ceiling, which grades `support_binding_auc`, not `gate_regime_auc`.** `restricted` (matching the documented 0.742 figure) and `unrestricted` are both valid for `support_binding_auc`; neither is valid for `gate_regime_auc`, which is a query-side, `y`-blind metric this file's detector does not model. Two earlier drafts of this file compared `gate_regime_auc` against these numbers regardless (`unrestricted`, then `restricted`) -- both wrong for the same underlying reason, not different degrees of the same mistake. A real ceiling for `gate_regime_auc` needs a fresh x-only probe (fit tag from `x` alone, cross-validated, at coherence=2.0), not yet measured.
- The matched-design coherence=2.0 unrestricted ceiling (0.631-0.639 across two independent measurements) is measurably below the coherence=0.0 figure (0.6975) -- future references to "the ceiling" for this design should specify which coherence they mean, since coherence=2.0 is what every trained arm actually sees. (Still only valid for `support_binding_auc`.)
- Coherence raises the ceiling for a detector given the true tags (`CONTROL_COHERENCE`'s documented 0.980 supervised x->tag AUC) but *lowers* the tag-blind, `(x,y)`-visible ceiling this sweep measures (see "why coherence lowers this ceiling" above) -- not a contradiction, two different probes.
- The coherence effect described above is specific to classes>=3, contamination=0.15 -- it should not be read as a general property of `regime_coherence` across the whole grid (classes=2 and contamination=0.30 rows stay flat).
