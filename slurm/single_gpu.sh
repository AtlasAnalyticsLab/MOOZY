#!/bin/bash
#SBATCH -J moozy_single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/output/%x_%j.txt
#SBATCH --error=logs/error/%x_%j.txt
#SBATCH --mem=64G

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

cd "$SLURM_SUBMIT_DIR"

moozy train stage1 \
  --feature_dirs /path/to/patch_features
