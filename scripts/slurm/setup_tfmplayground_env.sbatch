#!/bin/bash -l
#SBATCH --job-name=tfm-env-setup
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR}"
readonly UV_BIN="${PROJECT_DIR}/.tools/uv"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.tools" "${PROJECT_DIR}/.uv-cache"

export UV_CACHE_DIR="${PROJECT_DIR}/.uv-cache"
export UV_INSTALL_DIR="${PROJECT_DIR}/.tools"
export UV_NO_MODIFY_PATH=1

if [[ ! -x "${UV_BIN}" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"${UV_BIN}" sync --frozen --no-dev
"${PROJECT_DIR}/.venv/bin/python" -c 'import platform, torch, tabicl, tfmplayground; print({"python": platform.python_version(), "torch": torch.__version__, "tabicl": tabicl.__file__, "tfmplayground": tfmplayground.__file__})'
