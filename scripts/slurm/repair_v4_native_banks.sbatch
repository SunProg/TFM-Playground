#!/bin/bash -l
#SBATCH --job-name=repair-native-banks
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Replace non-finite-X episodes in the native banks (in place) and count them in the dumps.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="/users/k23139234/repo/TFM-Playground-uncertainty/.venv/bin/python"
readonly D="/scratch/users/k23139234/tfm_data/multiregime_v4_native"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun "${PYTHON_BIN}" scripts/repair_v4_bank_nonfinite_episodes.py --bank \
    "${D}/tabicl_mix_scm_evaluation/validation.h5" "${D}/tabicl_mix_scm_evaluation/test.h5" \
    "${D}/tabicl_mix_scm_evaluation_expose_z/validation.h5" "${D}/tabicl_mix_scm_evaluation_expose_z/test.h5"
for split in validation test; do
    "${PYTHON_BIN}" scratchpad/compute_v4_bank_prior_baselines.py --bank "${D}/tabicl_mix_scm_evaluation/${split}.h5" --output "${D}/tabicl_mix_scm_evaluation/${split}_prior.json"
    "${PYTHON_BIN}" scratchpad/compute_v4_bank_prior_baselines.py --bank "${D}/tabicl_mix_scm_evaluation_expose_z/${split}.h5" --output "${D}/tabicl_mix_scm_evaluation_expose_z/${split}_prior.json"
done
"${PYTHON_BIN}" - <<'PY'
import h5py, numpy as np
D="/scratch/users/k23139234/tfm_data/multiregime_v4_native/tabicl_mix_scm_paired_zmix5/"
for f in ("original", "r_z-multiregime", "g_z-multiregime"):
    with h5py.File(D + f + ".h5", "r") as h:
        n = h["X"].shape[0]; bad = 0
        for s in range(0, n, 1000):
            x = h["X"][s:s + 1000]
            bad += int((~np.isfinite(x).reshape(x.shape[0], -1).all(axis=1)).sum())
        print(f"dump {f}: {bad} / {n} episodes with non-finite X")
PY
