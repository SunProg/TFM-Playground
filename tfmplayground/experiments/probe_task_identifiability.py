"""Zero-training probe: is the true task identifiable from streamed evidence?

By construction the h5 bimodal episodes make the SUPPORT block uninformative about which
of two candidate tasks is true (`support_disagreement_max=0.20`). The disambiguating
information lives in the stream. This probe asks whether plain frozen vanilla nanoTabPFN,
given support + a prefix of the stream as ordinary in-context data, can already pick the
true candidate by scoring both candidates' labels on the held-out stream suffix.

  - If vanilla does this well  -> the filter/particle machinery is unnecessary; just feed
                                 more context.
  - If vanilla is at chance    -> either the information is not linearly accessible to the
                                 frozen backbone (contrastive/representation work has a
                                 target), or it is not there at all.

Order-invariant: we score candidates by likelihood, never by their index.
"""

import numpy as np
import pandas as pd
import torch

from tfmplayground.experiments.train_h5_prior_bimodal_gate import (
    H5GateConfig,
    episode_config,
    finite_episode,
)
from tfmplayground.interface import init_model_from_state_dict_file

DEVICE = "mps"
N_EPISODES = 256
BATCH = 4
PREFIX = 16  # of 32 stream rows given as context; the remaining 16 are scored

vanilla = init_model_from_state_dict_file("checkpoints/nanotabpfn.pth").to(DEVICE).eval()
cfg = H5GateConfig(device=DEVICE, batch_size=BATCH)
prior_cfg = episode_config(cfg)
rng = np.random.default_rng(4242)

rows = []
done = 0
failures = 0
while done < N_EPISODES:
    size = min(BATCH, N_EPISODES - done)
    # Pairing is stochastic and occasionally exhausts its attempt budget; that is a property
    # of the generator, not an error, so retry rather than abandoning the probe.
    try:
        batch = finite_episode(cfg, prior_cfg, rng, batch_size=size)
    except RuntimeError:
        failures += 1
        if failures > 200:
            raise
        continue
    done += size

    support_x = batch.initial_support_x
    support_y = batch.initial_support_y
    stream_x = batch.stream_x
    stream_y = batch.stream_y
    cand_stream = batch.candidate_stream_y  # (b, 2, stream_count)

    # context = support + stream prefix; score the stream suffix
    context_x = torch.cat((support_x, stream_x[:, :PREFIX]), dim=1)
    context_y = torch.cat((support_y, stream_y[:, :PREFIX].float()), dim=1)
    target_x = stream_x[:, PREFIX:]

    with torch.no_grad():
        logits = vanilla(
            (torch.cat((context_x, target_x), dim=1), context_y),
            train_test_split_index=context_x.shape[1],
            num_mem_chunks=1,
        )[..., :2]
        log_prob = logits.log_softmax(-1)  # (b, suffix, 2)

    suffix_true = stream_y[:, PREFIX:].long()
    for i in range(size):
        # which candidate index actually matches the realised stream labels
        match = [int((cand_stream[i, c].long() == stream_y[i].long()).all()) for c in range(2)]
        if sum(match) != 1:
            continue  # ambiguous bookkeeping; skip
        true_c = int(np.argmax(match))
        # score both candidates on the held-out suffix under vanilla's predictive dist
        scores = []
        for c in range(2):
            lab = cand_stream[i, c, PREFIX:].long()
            scores.append(float(log_prob[i].gather(-1, lab[:, None]).squeeze(-1).sum()))
        picked = int(np.argmax(scores))
        # how different are the two candidates on the scored suffix at all?
        disagree = float((cand_stream[i, 0, PREFIX:].long() != cand_stream[i, 1, PREFIX:].long()).float().mean())
        rows.append(
            {
                "true_candidate": true_c,
                "picked": picked,
                "correct": int(picked == true_c),
                "score_margin": abs(scores[0] - scores[1]),
                "suffix_disagreement": disagree,
                "vanilla_suffix_acc": float((log_prob[i].argmax(-1) == suffix_true[i]).float().mean()),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("/tmp/probe_task_identifiability.csv", index=False)

from scipy import stats

n = len(df)
acc = df.correct.mean()
successes = int(df.correct.sum())
result = stats.binomtest(successes, n, 0.5)
print(f"episodes usable: {n}")
print(f"true-task identification accuracy: {acc:.4f}  ({successes}/{n})")
print(f"binomial test vs chance 0.5: p = {result.pvalue:.5f}")
print(f"95% CI: {result.proportion_ci(0.95)}")
print()
print(f"mean vanilla accuracy on scored suffix: {df.vanilla_suffix_acc.mean():.4f}")
print(f"mean candidate disagreement on suffix : {df.suffix_disagreement.mean():.4f}")
print()
hi = df[df.suffix_disagreement >= df.suffix_disagreement.median()]
print(f"accuracy on high-disagreement half (n={len(hi)}): {hi.correct.mean():.4f}")
print(f"pairing failures skipped: {failures}")
print("DONE")
