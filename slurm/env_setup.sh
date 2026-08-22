#!/bin/bash -l
#SBATCH --job-name=ais_env
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Build the aisgnn conda environment.
#
#   sbatch slurm/env_setup.sh
#
# Two things about `module` in a batch script, both of which fail silently-ish:
#
#  1. The shebang must be `#!/bin/bash -l`.  A SLURM batch script runs in a
#     non-interactive, non-login shell, which never sources the module
#     initialisation, so `module` is simply not a command.
#  2. The load must happen in this shell, not in a pipeline.  Piping it
#     (`module load anaconda/2025 | head`) runs it in a subshell and the PATH
#     change is discarded, while the load itself appears to have succeeded.

set -euo pipefail

export AISGNN_ROOT="${AISGNN_ROOT:-/p/projects/climber3/rostami/Antarctic_Ice_Shelf}"
ENV_PREFIX="${AISGNN_ROOT}/env/aisgnn"

echo "host    : $(hostname)"
echo "started : $(date -Is)"

module load anaconda/2025
echo "conda   : $(command -v conda)"

mkdir -p "${AISGNN_ROOT}/env"

# Keep package and environment storage on the project filesystem: the home
# quota is far too small for a CUDA-enabled torch build.
export CONDA_PKGS_DIRS="${AISGNN_ROOT}/env/pkgs"
export PIP_CACHE_DIR="${AISGNN_ROOT}/env/pipcache"

if [ -d "${ENV_PREFIX}" ]; then
    echo "environment exists, updating"
    conda env update --prefix "${ENV_PREFIX}" \
        --file "${AISGNN_ROOT}/code/environment.yml" --prune
else
    conda env create --prefix "${ENV_PREFIX}" \
        --file "${AISGNN_ROOT}/code/environment.yml"
fi

source activate "${ENV_PREFIX}"
pip install --no-deps -e "${AISGNN_ROOT}/code"

echo "=== versions ==="
python -c "
import numpy, scipy, xarray, matplotlib
print('numpy      ', numpy.__version__)
print('scipy      ', scipy.__version__)
print('xarray     ', xarray.__version__)
print('matplotlib ', matplotlib.__version__)
import torch
print('torch      ', torch.__version__)
print('cuda build ', torch.version.cuda)
import torch_geometric
print('pyg        ', torch_geometric.__version__)
"

echo "finished: $(date -Is)"
echo
echo "activate with:"
echo "  module load anaconda/2025 && source activate ${ENV_PREFIX}"
