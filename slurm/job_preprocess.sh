#!/bin/bash -l
#SBATCH --job-name=ais_graphs
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Build the full per-shelf, per-year graph set for both scenarios.
#
#   sbatch slurm/job_preprocess.sh          # every 5th year, all shelves
#   sbatch slurm/job_preprocess.sh --every 2
#
# Runs on `standard` rather than `io`: this is CPU and memory work reading
# already-local files, and the io QOS caps CPUs per job hard enough to reject it.

set -euo pipefail

export AISGNN_ROOT="${AISGNN_ROOT:-/p/projects/climber3/rostami/Antarctic_Ice_Shelf}"
cd "$AISGNN_ROOT"

module load anaconda/2025
source activate "${AISGNN_ROOT}/env/aisgnn"

echo "host    : $(hostname)"
echo "started : $(date -Is)"

python -u code/scripts/02_build_graphs.py \
    --all-scenarios --all-shelves --every "${EVERY:-5}" "$@"

echo "finished: $(date -Is)"
ls "${AISGNN_ROOT}/data/graphs" | wc -l | xargs echo "graph files:"
du -sh "${AISGNN_ROOT}/data/graphs"
