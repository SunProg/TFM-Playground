# Within-episode regime breakdown (native large runs, 2026-09-22)

Every multiregime episode routes its rows to K regimes, each with its own label rule. The evaluation bank
stores the regime id of every row, so per-episode metrics can be split by regime — this file reports that
split for the final checkpoints of the large runs on the **z-blind TEST bank** (11 443 episodes with more
than one regime among the query rows).

## Method

`tfmplayground/experiments/multiregime_v4_evaluation.py` now records, for every episode:
`regime_ids`, `regime_query_counts`, `regime_cross_entropy`, `regime_accuracy`, `regime_positive_rate`,
`regime_auc` (macro OvR, NaN when a regime's query rows are single-class), plus
`regime_cross_entropy_{spread,best,worst,rule0_minus_rest,var,sampling_var,excess_var}`, and a
`by_regime_position` summary. The same fields are produced for the published/conventional baselines by
`scratchpad/evaluate_tabpfn_versions_on_v4_bank.py`. Jobs: `scripts/slurm/evaluate_native_regime_breakdown.sbatch`
(37451487 large, 37451489 medium, 37451490 small, 37451515 rg_z-fixed-large), references 37451488,
conventional 37451484/85.

**The regime index is episode-local.** For `soft_gate`, `_route` assigns rank bins of `x[:, 0]`, so index 0
is always the lowest-score bin and positions are comparable across episodes. For `persistent`, the latent
group -> index mapping is reshuffled per episode (`rng.shuffle(group_z)`), so averaging "regime 0" there is
meaningless. The tables below therefore **rank the regimes within each episode by the reference run's
(original-large) cross entropy** — rank 1 = the regime it fits best — and apply that same permutation to every
other run, which is comparable across episodes and across models for both families
(`scratchpad/agg_regime_ranked.py`).

Caveat: ranking on the reference model's own CE means rank 1 is partly selected on noise, so the best-worst
gap within a column is inflated by regression to the mean. Differences *between models at a fixed rank* are
not affected.

## Dispersion across regimes within an episode

Range (max-min) grows with K by construction; the SD does not. Mean over episodes, z-blind TEST:

| classes | K | episodes | SD orig | SD rg_z | var orig | var rg_z | range orig | range rg_z |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 2280 | 0.0687 | 0.0694 | 0.00471 | 0.00482 | 0.0567 | 0.0580 |
| 2 | 3 | 1714 | 0.0603 | 0.0618 | 0.00363 | 0.00382 | 0.0769 | 0.0778 |
| 2 | 4 | 1723 | 0.0632 | 0.0643 | 0.00399 | 0.00413 | 0.0964 | 0.0965 |
| 3 | 2 | 761 | 0.2317 | 0.2324 | 0.05368 | 0.05403 | 0.2481 | 0.2490 |
| 3 | 3 | 571 | 0.2104 | 0.2092 | 0.04426 | 0.04378 | 0.3354 | 0.3361 |
| 3 | 4 | 571 | 0.2087 | 0.2100 | 0.04357 | 0.04408 | 0.3967 | 0.3997 |
| 4 | 2 | 763 | 0.2553 | 0.2525 | 0.06516 | 0.06376 | 0.2737 | 0.2697 |
| 4 | 3 | 574 | 0.2236 | 0.2243 | 0.05001 | 0.05033 | 0.3545 | 0.3550 |
| 4 | 4 | 574 | 0.2055 | 0.2065 | 0.04224 | 0.04266 | 0.3937 | 0.3962 |
| 5 | 2 | 762 | 0.2555 | 0.2539 | 0.06530 | 0.06448 | 0.2695 | 0.2664 |
| 5 | 3 | 574 | 0.2148 | 0.2138 | 0.04616 | 0.04572 | 0.3372 | 0.3372 |
| 5 | 4 | 576 | 0.2053 | 0.2052 | 0.04215 | 0.04209 | 0.3914 | 0.3906 |

SD falls slightly with K and is ~4x larger for multiclass than binary; the two priors are identical to within
0.002 everywhere. Part of the SD is sampling noise (a regime's mean is estimated from 256/K rows); the
`excess_var` field subtracts the estimated sampling term and is available for every run evaluated after
2026-09-22 19:00.

## Metrics per regime rank

### regimes ranked by original-large (test split, z-blind); 11443 episodes with >1 query regime; runs: ['original-large', 'rg_z-curriculum-large']


## all families (11443 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4566 | 0.7707 | 0.7684 | 0.6582 | 0.6584 | 0.5869 | 0.5866 |
| 2 | 2 | 4566 | 0.9311 | 0.9271 | 0.5448 | 0.5470 | 0.5379 | 0.5385 |
| 3 | 1 | 3433 | 0.7483 | 0.7444 | 0.6787 | 0.6789 | 0.5826 | 0.5823 |
| 3 | 2 | 3433 | 0.8533 | 0.8492 | 0.5850 | 0.5870 | 0.5505 | 0.5504 |
| 3 | 3 | 3334 | 0.9580 | 0.9525 | 0.5027 | 0.5060 | 0.5174 | 0.5203 |
| 4 | 1 | 3444 | 0.7422 | 0.7374 | 0.6894 | 0.6888 | 0.5776 | 0.5760 |
| 4 | 2 | 3444 | 0.8288 | 0.8243 | 0.6096 | 0.6129 | 0.5503 | 0.5503 |
| 4 | 3 | 3391 | 0.8992 | 0.8946 | 0.5443 | 0.5478 | 0.5303 | 0.5308 |
| 4 | 4 | 3338 | 0.9882 | 0.9805 | 0.4794 | 0.4854 | 0.5054 | 0.5080 |

## soft_gate only (5683 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2262 | 0.7339 | 0.7286 | 0.6762 | 0.6780 | 0.5990 | 0.5987 |
| 2 | 2 | 2262 | 0.8876 | 0.8810 | 0.5842 | 0.5878 | 0.5526 | 0.5530 |
| 3 | 1 | 1705 | 0.6955 | 0.6868 | 0.6995 | 0.7023 | 0.5936 | 0.5932 |
| 3 | 2 | 1705 | 0.8148 | 0.8073 | 0.6159 | 0.6206 | 0.5587 | 0.5578 |
| 3 | 3 | 1606 | 0.9098 | 0.8984 | 0.5498 | 0.5566 | 0.5321 | 0.5356 |
| 4 | 1 | 1716 | 0.6789 | 0.6681 | 0.7146 | 0.7169 | 0.5836 | 0.5826 |
| 4 | 2 | 1716 | 0.7845 | 0.7745 | 0.6401 | 0.6471 | 0.5605 | 0.5613 |
| 4 | 3 | 1663 | 0.8553 | 0.8457 | 0.5848 | 0.5915 | 0.5459 | 0.5455 |
| 4 | 4 | 1610 | 0.9436 | 0.9297 | 0.5243 | 0.5339 | 0.5182 | 0.5209 |

## persistent only (5760 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2304 | 0.8068 | 0.8075 | 0.6405 | 0.6392 | 0.5753 | 0.5750 |
| 2 | 2 | 2304 | 0.9737 | 0.9723 | 0.5061 | 0.5070 | 0.5234 | 0.5242 |
| 3 | 1 | 1728 | 0.8005 | 0.8013 | 0.6581 | 0.6558 | 0.5721 | 0.5718 |
| 3 | 2 | 1728 | 0.8913 | 0.8906 | 0.5545 | 0.5539 | 0.5425 | 0.5430 |
| 3 | 3 | 1728 | 1.0029 | 1.0027 | 0.4590 | 0.4589 | 0.5035 | 0.5057 |
| 4 | 1 | 1728 | 0.8051 | 0.8062 | 0.6643 | 0.6609 | 0.5720 | 0.5699 |
| 4 | 2 | 1728 | 0.8728 | 0.8737 | 0.5793 | 0.5789 | 0.5405 | 0.5398 |
| 4 | 3 | 1728 | 0.9415 | 0.9416 | 0.5053 | 0.5058 | 0.5153 | 0.5166 |
| 4 | 4 | 1728 | 1.0298 | 1.0279 | 0.4376 | 0.4402 | 0.4931 | 0.4956 |

## soft_gate, binary (2837 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1128 | 0.5125 | 0.5122 | 0.7352 | 0.7350 | 0.6054 | 0.6041 |
| 2 | 2 | 1128 | 0.5706 | 0.5707 | 0.6951 | 0.6953 | 0.5316 | 0.5314 |
| 3 | 1 | 850 | 0.4934 | 0.4935 | 0.7470 | 0.7468 | 0.6034 | 0.6012 |
| 3 | 2 | 850 | 0.5344 | 0.5351 | 0.7117 | 0.7107 | 0.5501 | 0.5482 |
| 3 | 3 | 800 | 0.5680 | 0.5674 | 0.6797 | 0.6805 | 0.5110 | 0.5162 |
| 4 | 1 | 859 | 0.4902 | 0.4909 | 0.7543 | 0.7541 | 0.5946 | 0.5928 |
| 4 | 2 | 859 | 0.5262 | 0.5266 | 0.7190 | 0.7199 | 0.5523 | 0.5547 |
| 4 | 3 | 835 | 0.5496 | 0.5498 | 0.6940 | 0.6938 | 0.5308 | 0.5309 |
| 4 | 4 | 804 | 0.5840 | 0.5838 | 0.6653 | 0.6659 | 0.4966 | 0.4990 |

## soft_gate, multiclass (2846 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1134 | 0.9540 | 0.9438 | 0.6175 | 0.6214 | 0.5913 | 0.5922 |
| 2 | 2 | 1134 | 1.2029 | 1.1898 | 0.4740 | 0.4808 | 0.5745 | 0.5755 |
| 3 | 1 | 855 | 0.8964 | 0.8789 | 0.6523 | 0.6580 | 0.5801 | 0.5821 |
| 3 | 2 | 855 | 1.0935 | 1.0779 | 0.5206 | 0.5309 | 0.5686 | 0.5689 |
| 3 | 3 | 806 | 1.2491 | 1.2270 | 0.4208 | 0.4337 | 0.5546 | 0.5565 |
| 4 | 1 | 857 | 0.8681 | 0.8457 | 0.6748 | 0.6797 | 0.5649 | 0.5653 |
| 4 | 2 | 857 | 1.0434 | 1.0230 | 0.5610 | 0.5742 | 0.5717 | 0.5704 |
| 4 | 3 | 828 | 1.1636 | 1.1440 | 0.4746 | 0.4884 | 0.5641 | 0.5632 |
| 4 | 4 | 806 | 1.3022 | 1.2747 | 0.3836 | 0.4023 | 0.5418 | 0.5450 |

## persistent, binary (2880 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1152 | 0.5242 | 0.5249 | 0.7268 | 0.7265 | 0.5820 | 0.5819 |
| 2 | 2 | 1152 | 0.5795 | 0.5810 | 0.6803 | 0.6814 | 0.4966 | 0.4983 |
| 3 | 1 | 864 | 0.5050 | 0.5052 | 0.7473 | 0.7458 | 0.5745 | 0.5738 |
| 3 | 2 | 864 | 0.5407 | 0.5409 | 0.7070 | 0.7062 | 0.5305 | 0.5310 |
| 3 | 3 | 864 | 0.5817 | 0.5825 | 0.6653 | 0.6664 | 0.4758 | 0.4761 |
| 4 | 1 | 864 | 0.4971 | 0.4976 | 0.7528 | 0.7515 | 0.5790 | 0.5737 |
| 4 | 2 | 864 | 0.5284 | 0.5287 | 0.7220 | 0.7216 | 0.5293 | 0.5287 |
| 4 | 3 | 864 | 0.5567 | 0.5564 | 0.6899 | 0.6897 | 0.5032 | 0.5045 |
| 4 | 4 | 864 | 0.5962 | 0.5957 | 0.6530 | 0.6541 | 0.4696 | 0.4715 |

## persistent, multiclass (2880 episodes)

| K | rank | episodes | CE original-large | CE rg_z-curriculum-large | acc original-large | acc rg_z-curriculum-large | AUC original-large | AUC rg_z-curriculum-large |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1152 | 1.0895 | 1.0901 | 0.5542 | 0.5518 | 0.5677 | 0.5674 |
| 2 | 2 | 1152 | 1.3679 | 1.3636 | 0.3318 | 0.3327 | 0.5518 | 0.5517 |
| 3 | 1 | 864 | 1.0959 | 1.0975 | 0.5689 | 0.5659 | 0.5690 | 0.5693 |
| 3 | 2 | 864 | 1.2418 | 1.2402 | 0.4021 | 0.4016 | 0.5563 | 0.5569 |
| 3 | 3 | 864 | 1.4240 | 1.4230 | 0.2526 | 0.2514 | 0.5343 | 0.5387 |
| 4 | 1 | 864 | 1.1131 | 1.1149 | 0.5758 | 0.5703 | 0.5617 | 0.5643 |
| 4 | 2 | 864 | 1.2173 | 1.2187 | 0.4366 | 0.4363 | 0.5544 | 0.5536 |
| 4 | 3 | 864 | 1.3263 | 1.3268 | 0.3207 | 0.3218 | 0.5298 | 0.5311 |
| 4 | 4 | 864 | 1.4633 | 1.4602 | 0.2222 | 0.2264 | 0.5203 | 0.5235 |

## Findings

1. **The best-worst gap inside an episode is large.** All families, K=4: CE 0.742 (rank 1) -> 0.988 (rank 4),
   accuracy 0.689 -> 0.479, AUC 0.578 -> 0.505.
2. **The whole multiregime-prior gain is in soft_gate multiclass**, and it grows with rank difficulty:
   dCE -0.010 (K=2 rank 1) to -0.028 (K=4 rank 4), daccuracy up to +0.019. soft_gate binary is +-0.0007 and both
   persistent panels +-0.002 at every rank.
3. **Binary AUC falls below chance at the worst rank** (soft_gate K=4 rank 4: 0.497; persistent K=4 rank 4:
   0.470; persistent K=3 rank 3: 0.476) — the model inverts the minority regimes rather than merely missing
   them, consistent with applying the dominant regime's rule everywhere.
4. **Multiclass AUC never drops below 0.52**, so the inversion is a binary-only failure — the same cells that
   remain positive-excess-CE in the pooled tables.
5. **persistent multiclass is effectively unlearned at the hard ranks**: accuracy 0.222 (K=4 rank 4) vs 0.576 at
   rank 1, CE 1.463, and the multiregime prior changes nothing (|d| <= 0.003).

Figures (all three sizes): `figures/native/regime_ranked/native_<size>_test_regime_ranked_{ce,acc,auc}[_delta].png` — rows = family x class
kind, columns = K, x = regime rank, one line per run.

## Medium and small (all three runs each)

The per-regime evaluation was run for every native run (jobs 37451487/89/90/515). The tables below use the same
ranking construction, with `original-<size>` as the reference in each size.

### Paired change in cross entropy vs original, by regime rank (K=4, TEST, z-blind)

Negative = the multiregime-pretrained run is better. Rank 1 = the regime original fits best.

| size | group | run | rank 1 | rank 2 | rank 3 | rank 4 |
|---|---|---|---|---|---|---|
| small | soft_gate multiclass | rg_z-curriculum | +0.0007 | -0.0005 | -0.0010 | -0.0026 |
| small | soft_gate multiclass | rg_z-fixed | -0.0011 | -0.0017 | -0.0009 | -0.0053 |
| small | soft_gate binary | rg_z-curriculum | +0.0007 | -0.0006 | -0.0024 | -0.0038 |
| small | soft_gate binary | rg_z-fixed | -0.0000 | -0.0004 | -0.0017 | -0.0020 |
| small | persistent multiclass | rg_z-curriculum | +0.0009 | +0.0004 | -0.0008 | -0.0028 |
| small | persistent multiclass | rg_z-fixed | +0.0014 | -0.0013 | -0.0010 | -0.0035 |
| small | persistent binary | rg_z-curriculum | +0.0008 | -0.0006 | -0.0021 | -0.0042 |
| small | persistent binary | rg_z-fixed | +0.0001 | -0.0005 | -0.0014 | -0.0027 |
| medium | soft_gate multiclass | rg_z-curriculum | **-0.0256** | -0.0239 | -0.0251 | -0.0275 |
| medium | soft_gate multiclass | rg_z-fixed | -0.0135 | -0.0116 | -0.0115 | -0.0200 |
| medium | soft_gate binary | rg_z-curriculum | +0.0004 | -0.0007 | -0.0011 | -0.0031 |
| medium | soft_gate binary | rg_z-fixed | +0.0002 | -0.0009 | -0.0015 | -0.0031 |
| medium | persistent multiclass | rg_z-curriculum | -0.0011 | -0.0012 | -0.0027 | -0.0021 |
| medium | persistent multiclass | rg_z-fixed | -0.0004 | -0.0001 | -0.0025 | -0.0022 |
| medium | persistent binary | rg_z-curriculum | +0.0001 | -0.0004 | -0.0018 | -0.0024 |
| medium | persistent binary | rg_z-fixed | -0.0001 | -0.0005 | -0.0021 | -0.0024 |
| large | soft_gate multiclass | rg_z-curriculum | -0.0223 | -0.0204 | -0.0196 | **-0.0276** |
| large | soft_gate multiclass | rg_z-fixed | -0.0161 | -0.0155 | -0.0162 | -0.0263 |
| large | soft_gate binary | rg_z-curriculum | +0.0007 | +0.0004 | +0.0002 | -0.0002 |
| large | soft_gate binary | rg_z-fixed | +0.0002 | +0.0002 | +0.0000 | +0.0002 |
| large | persistent multiclass | rg_z-curriculum | +0.0018 | +0.0015 | +0.0005 | -0.0031 |
| large | persistent multiclass | rg_z-fixed | +0.0002 | -0.0006 | -0.0011 | -0.0040 |
| large | persistent binary | rg_z-curriculum | +0.0005 | +0.0003 | -0.0002 | -0.0006 |
| large | persistent binary | rg_z-fixed | +0.0000 | +0.0002 | +0.0002 | +0.0001 |

### What changes with size

1. **The multiregime effect needs capacity.** On soft_gate multiclass the paired gain is ~0.001-0.005 at small,
   0.012-0.028 at medium and 0.016-0.028 at large. At small the prior essentially does nothing on the cell it is
   designed for.
2. **At medium the gain is uniform across ranks** (-0.0256 / -0.0239 / -0.0251 / -0.0275 for rg_z-curriculum),
   i.e. a level shift; at large it tilts toward the hardest regime (-0.0223 -> -0.0276), and at small the little
   there is sits almost entirely on rank 4.
3. **Exposure share matters at medium, not at large.** rg_z-curriculum (0->50 %) roughly doubles rg_z-fixed's
   (30 %) gain at medium (-0.026 vs -0.013 at rank 1) but the two converge at large (-0.022 vs -0.016), where
   rg_z-fixed's pooled test CE is in fact marginally the better of the two (0.7802 vs 0.7803).
4. **Persistent shows the redistribution pattern at every size**: rank 1 unchanged or slightly worse, rank 4
   better (small -0.003, medium -0.002, large -0.003 to -0.004), consistent with hedging rather than learning.
5. **Binary stays flat everywhere except small**, where every family shows a small worst-rank gain
   (-0.002 to -0.004) that disappears once the model has capacity.

Figures: `figures/native/regime_ranked/native_{small,medium,large}_test_regime_ranked_{ce,acc,auc}[_delta].png`
(18 files; `_delta` = paired difference vs the size's original run with 95 % CI whiskers).
