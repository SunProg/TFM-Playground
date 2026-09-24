#!/bin/bash
# Resubmit the BeyondArena v4 evaluation as 9 parallel jobs from the
# TFM-Playground-uncertainty checkout. The per-job settings below are a
# reconstruction (the original submissions carried them in the submitting
# shell's environment, which SLURM does not expose):
#   canonical      -> the 3 `original-*` baseline runs
#   rz / gz        -> the 6 r_z-* / 6 g_z-* baseline runs
#   z-expose-final -> the 12 z-exposed runs (runs_exp/expose_z)
#   tabpfn-* / tabicl-* -> one published baseline each (a tiny run root is
#                    still required by the CLI; original-small is used)
# Results -> results/beyondarena-v4/<RUN_MODE>-<jobid> (symlinked to scratch).
set -euo pipefail
readonly U="/cephfs/volumes/hpc_home/k23139234/17f71989-c3b3-41d8-9315-37daa2fead54/repo/TFM-Playground-uncertainty"
readonly S="${U}/scratchpad/evaluate_multiregime_v4_beyondarena_a30.sbatch"
readonly R="/scratch/users/k23139234/tfm_runs/nanotabpfn_tabicl_mix_scm_prior_scale"
readonly ORIG="${R}/37283915/original-small,${R}/37283921/original-medium,${R}/37283995/original-large"
readonly RZ="${R}/37283915/r_z-fixed-small,${R}/37283915/r_z-curriculum-small,${R}/37283921/r_z-fixed-medium,${R}/37283921/r_z-curriculum-medium,${R}/37283995/r_z-fixed-large,${R}/37283995/r_z-curriculum-large"
readonly GZ="${R}/37283915/g_z-fixed-small,${R}/37283915/g_z-curriculum-small,${R}/37283921/g_z-fixed-medium,${R}/37283921/g_z-curriculum-medium,${R}/37283995/g_z-fixed-large,${R}/37283995/g_z-curriculum-large"
readonly ZX="${U}/runs_exp/expose_z"
readonly TINY="${R}/37283915/original-small"
cd "${U}"
submit() {  # name run_mode run_roots [baselines]
  local name="$1" mode="$2" roots="$3" baselines="${4:-}"
  local exports="ALL,RUN_MODE=${mode},V4_RUN_ROOTS=${roots}"
  [[ -n "${baselines}" ]] && exports+=",BASELINES=${baselines}"
  printf '%s -> ' "${name}"
  sbatch --parsable --chdir="${U}" --job-name="${name}" --export="${exports}" "${S}"
}
submit beyond-v4-canonical      canonical      "${ORIG}"
submit beyond-v4-rz             rz             "${RZ}"
submit beyond-v4-gz             gz             "${GZ}"
submit beyond-v4-z-expose-final z-expose-final "${ZX}"
submit beyond-tabpfn-v2.2       tabpfn-v2.2    "${TINY}" tabpfn-v2.2
submit beyond-tabpfn-v2.6       tabpfn-v2.6    "${TINY}" tabpfn-v2.6
submit beyond-tabpfn-v3         tabpfn-v3      "${TINY}" tabpfn-v3
submit beyond-tabicl-v1         tabicl-v1      "${TINY}" tabicl-v1
submit beyond-tabicl-v2         tabicl-v2      "${TINY}" tabicl-v2
