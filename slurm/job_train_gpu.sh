#!/bin/bash -l
#SBATCH --job-name=ais_train
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-59%12
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err
#
# Production training matrix: 4 architectures x 5 seeds x 3 splits = 60 jobs,
# at most 12 running at once.
#
#   sbatch slurm/job_train_gpu.sh
#   sbatch --array=0-3 slurm/job_train_gpu.sh        # one seed, all architectures
#
# The three splits answer different questions and are all needed:
#   shelf     generalisation to an unseen cavity
#   year      extrapolation in time
#   scenario  transfer from REPEAT1970 to 4xCO2, the hardest and the one the
#             tipping-point analysis rests on

set -euo pipefail

export AISGNN_ROOT="${AISGNN_ROOT:-/p/projects/climber3/rostami/Antarctic_Ice_Shelf}"
cd "$AISGNN_ROOT"

module load anaconda/2025
source activate "${AISGNN_ROOT}/env/aisgnn"

ARCHS=(mlp gcn gat egcn)
SEEDS=(0 1 2 3 4)
SPLITS=(shelf year scenario)

IDX=${SLURM_ARRAY_TASK_ID:-0}
N_ARCH=${#ARCHS[@]}
N_SEED=${#SEEDS[@]}

ARCH=${ARCHS[$(( IDX % N_ARCH ))]}
SEED=${SEEDS[$(( (IDX / N_ARCH) % N_SEED ))]}
SPLIT=${SPLITS[$(( IDX / (N_ARCH * N_SEED) ))]}

echo "host      : $(hostname)"
echo "gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "task      : ${IDX}  arch=${ARCH} seed=${SEED} split=${SPLIT}"
echo "started   : $(date -Is)"

EXTRA=""
if [ "$SPLIT" = "scenario" ]; then
    EXTRA="--test-scenario SMITH_bi646"
fi

python -u code/scripts/03_train.py \
    --arch "$ARCH" --seed "$SEED" --split "$SPLIT" $EXTRA \
    --epochs "${EPOCHS:-300}" --patience "${PATIENCE:-40}"

echo "finished  : $(date -Is)"
