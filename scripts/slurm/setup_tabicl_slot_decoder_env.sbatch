#!/bin/bash -l
#SBATCH --job-name=tabicl-slot-env
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Install only TabICL 2.2 into a project-local overlay. The existing .venv is
# a symlink to a shared environment and remains untouched.
set -euo pipefail
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
readonly SLOT_SITE_DIR="${PROJECT_DIR}/.tabicl-slot-site"
cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing shared Python environment: ${PYTHON_BIN}" >&2; exit 2; }
mkdir -p "${SLOT_SITE_DIR}"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-deps --upgrade \
    --target "${SLOT_SITE_DIR}" 'tabicl==2.2.0'
export PYTHONPATH="${PROJECT_DIR}:${SLOT_SITE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import torch
import tabicl
from tabicl.train._run import Trainer

assert importlib.metadata.version("tabicl") == "2.2.0"
print({"torch": torch.__version__, "tabicl": tabicl.__file__, "trainer": Trainer.__module__})
PY
