#!/bin/bash -l
# Pulls the live v4 regime-exposure training runs (observed_regime,
# hidden_regime_control, probability_regime_control, paired_regime_control --
# any tree whose history.jsonl was touched in the last $STALE_MINUTES) plus
# the fixed original-model checkpoint probes down from the HPC cluster, so
# compare_regime_info_vs_original.py can run against a local copy.
#
#   bash scripts/sync_observed_regime_monitor.sh
#
# Requires the "create" SSH alias (ProxyJump to erc-hpc-*) already set up in
# ~/.ssh/config, and rsync on both ends.
set -euo pipefail

readonly HOST="${OBSERVED_REGIME_SSH_HOST:-create}"
readonly REMOTE_ROOT="${OBSERVED_REGIME_REMOTE_ROOT:-/cephfs/volumes/hpc_home/k23139234/17f71989-c3b3-41d8-9315-37daa2fead54/TFM-Playground-slot-attention-v4-diagnostic}"
readonly STALE_MINUTES="${STALE_MINUTES:-20}"
readonly LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/artifacts/cluster_sync/observed_regime_monitor"

mkdir -p "${LOCAL_ROOT}/runs" "${LOCAL_ROOT}/original_probes"

readonly ARMS="observed_regime hidden_regime_control probability_regime_control paired_regime_control"

echo "discovering live regime-exposure run directories (history.jsonl touched in last ${STALE_MINUTES}m)..."
find_expr="$(for arm in ${ARMS}; do printf -- "-path '*/artifacts/runs/%s/*/history.jsonl' -o " "${arm}"; done)"
find_expr="${find_expr% -o }"
live_dirs="$(ssh "${HOST}" "find '${REMOTE_ROOT}/artifacts/runs' \\( ${find_expr} \\) -mmin -${STALE_MINUTES} 2>/dev/null | xargs -n1 dirname" || true)"

if [[ -z "${live_dirs}" ]]; then
    echo "no live observed-regime runs found (nothing updated in the last ${STALE_MINUTES} minutes)"
else
    while IFS= read -r remote_dir; do
        [[ -z "${remote_dir}" ]] && continue
        rel="${remote_dir#${REMOTE_ROOT}/artifacts/runs/}"
        dest="${LOCAL_ROOT}/runs/${rel}"
        mkdir -p "${dest}"
        echo "syncing ${rel}"
        rsync -az -e ssh \
            "${HOST}:${remote_dir}/history.jsonl" \
            "${HOST}:${remote_dir}/regime_information_progress.jsonl" \
            "${dest}/" 2>/dev/null || true
    done <<< "${live_dirs}"
fi

echo "syncing original-model checkpoint probes (small/medium/large, steps 2000/4000/6000/8000)..."
for size in small medium large; do
    dest="${LOCAL_ROOT}/original_probes/original-${size}/seed-2402/checkpoint_probes"
    mkdir -p "${dest}"
    for step in 002000 004000 006000 008000; do
        rsync -az -e ssh \
            "${HOST}:${REMOTE_ROOT}/artifacts/evaluations/regime_information_checkpoint_compatible/original-${size}/seed-2402/checkpoint_probes/step-${step}.json" \
            "${dest}/" 2>/dev/null || true
    done
done

echo "done. local copy at ${LOCAL_ROOT}"
