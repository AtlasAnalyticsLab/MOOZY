#!/bin/bash
#SBATCH -J moozy_encode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/output/%x_%j.txt
#SBATCH --error=logs/error/%x_%j.txt
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# From pre-computed H5 feature files:
moozy encode \
  /path/to/slide_1.h5 \
  /path/to/slide_2.h5 \
  --output /path/to/case_embedding.h5

# From raw slides (requires atlas-patch):
# moozy encode \
#   /path/to/slide_1.svs \
#   /path/to/slide_2.svs \
#   --output /path/to/case_embedding.h5 \
#   --target_mag 20 \
#   --step_size 224
