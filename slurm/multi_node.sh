#!/bin/bash
#SBATCH -J moozy_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=logs/output/%x_%j.txt
#SBATCH --error=logs/error/%x_%j.txt
#SBATCH --mem=128G

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NCCL_DEBUG=warn
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$SLURM_SUBMIT_DIR"

srun --kill-on-bad-exit=1 \
  moozy train stage1 \
    --feature_dirs /path/to/patch_features
