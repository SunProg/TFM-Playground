# Slot attention for multi-regime nanoTabPFN: prior composition and slot count

## Status

- Date: 2026-08-30
- Scope: binary classification, synthetic SCM episodes, from-scratch pretraining, real GPU runs on CREATE (`biomed_a30_gpu`, NVIDIA A30)
- Branch: `exp/slot-attention`. Slot arms ran at `32ff8ec` (K=2) and `432b070` (K=3, K=4)
- Slurm: array `36853727` (K=2, 4 arms) and `36854390` (K=3 and K=4, 8 arms). 11 of 12 completed; `curriculum` K=4 stopped at step 8,500 on a disk quota error
- A fourth array `36858884` trained vanilla controls; its outputs were deleted before they were read, so **no vanilla numbers are reported here**
- Run directories were deleted after the fact. The tables below are transcribed from the run logs; the underlying `history.jsonl` files no longer exist

## Motivation

`results/multiregime_backbone/eval-36553362/analysis.md` records nanoTabPFN at
chance on the minority regime of a contaminated context — other-regime AUC
0.469–0.533 against base-regime 0.75–0.90 — and full-backbone fine-tuning moving
that by −0.002 across 14 cells. `paper/tabpfn_multi_regime_results.md` adds that
real TabPFN is no better and that neither model's predictive distribution reacts
to contamination at all.

The multiregime prior is a *row-level* mixture: one shared feature matrix is
labelled by two independently drawn functions, and a contamination fraction of
support rows — and, independently, of query rows — is relabelled under the
second (`continuous_episodes._build_multiregime_item`). Its own comment is
explicit that "the truth for an individual row is a per-row mixture, not a single
resolved candidate."

Slot Attention (Locatello et al., 2020) is a mechanism for exactly that shape:
K exchangeable slots that *compete* for input elements through a softmax over
slots, refined over T iterations. Swap pixels for support rows and objects for
regimes and a minority-regime row should win a slot of its own rather than being
averaged into the majority, which is what a single softmax over all support rows
does today.

This experiment asks which mixture of single-regime and multiregime prior teaches
such a model to represent both regimes, and whether the slot count matters.

## Construction

A `NanoTabPFNSlotRegimeModel` — real slot attention over the backbone's
target-column support states, then a per-query mixture over slots:

```
p(y | x_q) = logsumexp_k [ log gate_k(x_q) + log p(y | x_q, slot_k) ]
```

The per-query gate rather than one weight per episode is what makes this a
row-level mixture. Training is from scratch, all 4.86 M parameters trainable;
the backbone is not frozen. The loss is the mixture NLL alone — no
specialization, coherence, diversity or posterior-supervision term, and no
slot-to-candidate matching anywhere in the gradient path. Slot attention is never
told which slot owns which regime.

Grid: 4 prior compositions × 3 slot counts, every other flag identical.

| arm | multiregime share per step |
|---|---|
| `plain` | 0.0 |
| `multiregime` | 1.0 |
| `mixed` | 0.30 constant |
| `curriculum` | 0.0 until 10% of steps, linear ramp to 0.5 by 50%, flat thereafter |

The three shared modes delegate to
`pretrain_plain_nanotabpfn.multiregime_probability` rather than restating it, so
they are identical to the vanilla baseline runs' definitions.

```
10,000 steps (20 epochs of 500), seed 2402, warmup 2,000, cosine to 1e-6
micro-batch 8 x accumulate 4 = 32 episodes per optimizer step
support 128, query 32, features 2-12, contamination 0.3, label noise 0.0
slot iterations 3, competitive normalization
validation every 500 steps; TabArena-small every 500 steps at 5 folds x 10 repeats
multiregime episodes streamed from a 375,000-episode HDF5 dump (0.85 cycles, no repeats)
```

## Metrics

| name | meaning | reference |
|---|---|---|
| `mr_ce` | cross entropy on held-out multiregime episodes | chance = ln 2 = 0.6931 |
| `ord_ce` | cross entropy on held-out ordinary episodes | chance = 0.6931 |
| `binding AUC` | best slot's support-attention column scored against the held-out per-row regime tag, up to the arbitrary flip | chance = 0.5, **ceiling 0.75, see below** |
| `purity / base` | rows whose argmax slot's majority regime is their own, against the majority-class fraction | equal means total collapse |
| `att_H` | normalized entropy of each row's distribution over slots | 1.0 = uniform, 0.0 = each row claimed |
| TabArena AUC | ROC AUC on 5 real OpenML tables | chance = 0.5 |

Brackets are 95% normal-approximation intervals: 8 episodes for validation
metrics, 250 fits for TabArena.

## Results

| prior | K | mr_ce | ord_ce | binding AUC | purity/base | att_H | TabArena AUC |
|---|---|---|---|---|---|---|---|
| curriculum | 2 | 0.6827 | 0.4497 | 0.510 [0.507, 0.512] | 0.703/0.703 | 0.979 | 0.728 [0.712, 0.744] |
| mixed | 2 | 0.6820 | 0.4472 | 0.506 [0.502, 0.511] | 0.703/0.703 | 0.962 | 0.723 [0.705, 0.741] |
| multiregime | 2 | 0.6823 | 0.6115 | 0.514 [0.509, 0.518] | 0.703/0.703 | 0.949 | 0.714 [0.699, 0.728] |
| plain | 2 | 0.6858 | 0.4473 | 0.505 [0.502, 0.508] | 0.703/0.703 | 0.803 | 0.720 [0.701, 0.739] |
| curriculum | 3 | 0.6830 | 0.4495 | 0.512 [0.509, 0.516] | 0.703/0.703 | 0.950 | 0.727 [0.712, 0.743] |
| mixed | 3 | 0.6821 | 0.4466 | 0.514 [0.507, 0.521] | 0.703/0.703 | 0.964 | **0.737 [0.722, 0.753]** |
| multiregime | 3 | 0.6829 | 0.6258 | 0.508 [0.507, 0.509] | 0.703/0.703 | 0.897 | 0.709 [0.693, 0.724] |
| plain | 3 | 0.6859 | 0.4466 | 0.509 [0.505, 0.514] | 0.703/0.703 | 0.774 | 0.711 [0.690, 0.732] |
| curriculum* | 4 | 0.6841 | 0.4488 | 0.519 [0.514, 0.523] | 0.703/0.703 | 0.978 | 0.721 [0.705, 0.737] |
| mixed | 4 | 0.6816 | 0.4460 | 0.516 [0.510, 0.522] | 0.703/0.703 | 0.957 | 0.725 [0.708, 0.743] |
| multiregime | 4 | 0.6827 | 0.6464 | 0.517 [0.513, 0.522] | 0.703/0.703 | 0.960 | 0.702 [0.684, 0.719] |
| plain | 4 | 0.6860 | 0.4475 | 0.518 [0.512, 0.524] | 0.703/0.703 | 0.896 | 0.718 [0.699, 0.737] |

\* stopped at step 8,500 (disk quota); all others completed 10,000.

## How much of the regime tag is even identifiable

Measured over 19,200 rows drawn from the dump:

| quantity | value |
|---|---|
| rows where the two label functions agree | 0.507 |
| rows tagged contaminated | 0.300 |
| **contaminated rows that are observationally indistinguishable** | **0.500** |

Both label functions are evaluated on the same feature matrix, so a row
relabelled to the value the base function would have produced anyway is tagged
contaminated while being identical in every observable respect. Half of all
contaminated rows are in that position.

That caps any detector. With 30% positives of which half are indistinguishable,
a perfect detector scores

```
AUC_max = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
```

and purity tops out near 0.85 against the 0.703 base rate. **The binding numbers
above were therefore measured against an unreachable target of 1.0.** Against the
real ceiling, 0.505–0.519 still uses only about 4–8% of the achievable range, so
the conclusion does not change — but any future binding metric should be
restricted to rows where the two candidate functions actually disagree.

## Analysis

**Slots did not bind, in any cell.** Binding AUC is 0.505–0.519 against a chance
of 0.5 and a ceiling of 0.75. The sharper statement is purity: it equals the base
rate 0.703 to three decimals in all twelve cells. Purity equal to the majority
fraction means every support row's argmax landed on the *same* slot. That is not
weak binding, it is complete assignment collapse, and over-provisioning to K=3
and K=4 did nothing to relieve it.

`att_H` distinguishes two failure modes. In the multiregime-heavy arms it sits at
0.95–0.98, i.e. the attention stayed near-uniform and nothing was decided either
way. In `plain` at K=3 it falls to 0.774 — the attention genuinely committed, and
still put everything in one slot. The second is the stronger negative.

**No arm learned the multiregime task.** `mr_ce` spans 0.6816–0.6860 against
chance at 0.6931. The 0.004 spread between best and worst is inside the noise, so
the prior-composition question this grid was built to answer is *unanswered*:
there is no signal to rank the four compositions by.

**Training itself worked.** `ord_ce` separates exactly as it should — ~0.447 for
the three arms that see ordinary data, 0.61–0.65 for `multiregime`, which never
does.

**TabArena** ranges 0.702–0.737, with `mixed` at K=3 highest and the
multiregime-only arms consistently lowest, consistent with never seeing clean
tables. Without the vanilla controls there is no way to say whether any of this
differs from a plain nanoTabPFN at the same budget.

## Limitations

- **10,000 steps against the 50,000 the vanilla baselines used.** A 4.86 M
  parameter model trained from scratch for 10k steps, 2k of it warmup, is
  substantially undertrained. Since nothing learned the multiregime task, there
  may have been no regime signal for slots to bind to, independent of the
  mechanism.
- **The vanilla controls were lost**, so nothing here is anchored against a
  non-slot model at matched budget.
- **No `competitive=False` ablation ran.** Without it the failure cannot be
  attributed to the competition rather than to the head, the budget or the task.
- **The query gate used in these runs has since been replaced.** These arms
  routed queries by a bilinear similarity between query state and slot, with no
  parameters shared with the decoder. It was later changed to a decoder mask
  channel, matching the vision decoder's alpha output. Runs after that change are
  a different architecture and are not comparable to this table.
- TabArena numbers here use a 2,048-row subsample at 5 folds × 10 repeats, and
  are **not** comparable to `paper/finetuned_nanotabpfn_tabarena_results.md`,
  which uses a 200-row subsample at 5 × 20.
- Single seed (2402) throughout.

## What this does and does not establish

It does not show that slot attention cannot represent multi-regime tabular
contexts. It shows that, with the mixture NLL alone, at 10k steps from scratch,
with the bilinear query gate, slots did not partition the context by regime at
any of four prior compositions or three slot counts — while the same module on a
simpler task in the same session went from 0.008 to 1.000 slot disagreement with
support rows splitting 0.035 → 0.574, and its non-competitive twin stayed
collapsed at 0.009. The mechanism works; it did not transfer here.

The two candidate explanations this grid cannot separate are budget and
architecture. The cheapest discriminating experiment is a single prior
composition at 50,000 steps with a `competitive=False` twin: if competitive and
non-competitive remain indistinguishable at full budget, the mechanism is not the
active ingredient on this task. A linear probe asking whether the regime
distinction survives the backbone's six layers of full row-attention into the
states the head reads would say whether there is anything left to bind at all —
that probe was set up but could not run, as the checkpoints had been deleted.
