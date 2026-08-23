#!/bin/bash -l
#SBATCH --job-name=ais_analyse
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Run the emulator analyses and generate the figures.
#
#   sbatch slurm/job_analyse.sh

set -euo pipefail
export AISGNN_ROOT="${AISGNN_ROOT:-/p/projects/climber3/rostami/Antarctic_Ice_Shelf}"
cd "$AISGNN_ROOT"
module load anaconda/2025
source activate "${AISGNN_ROOT}/env/aisgnn"

echo "started : $(date -Is)"
python -u code/scripts/04_analyse.py "$@"
python -u code/scripts/11_make_figures.py
echo "finished: $(date -Is)"
