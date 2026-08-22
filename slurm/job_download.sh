#!/bin/bash
#SBATCH --job-name=ais_download
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Fetch the public datasets from Zenodo.
#
# Run under SLURM rather than directly on a login node: the transfer plus the
# unpacking of several multi-gigabyte archives runs for hours and is reliably
# killed by the login-node resource reaper part way through.
#
#   sbatch slurm/job_download.sh
#   sbatch slurm/job_download.sh burgard2022 misomip2_roms_utas

set -euo pipefail

export AISGNN_ROOT="${AISGNN_ROOT:-/p/projects/climber3/rostami/Antarctic_Ice_Shelf}"
cd "$AISGNN_ROOT"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "root      : $AISGNN_ROOT"
echo "started   : $(date -Is)"

# Only the standard library is used, so no environment is needed here.
python3 -u code/scripts/00_download_data.py ${@:+--records "$@"}

echo "finished  : $(date -Is)"
du -sh "$AISGNN_ROOT"/data/raw/*/ 2>/dev/null || true
