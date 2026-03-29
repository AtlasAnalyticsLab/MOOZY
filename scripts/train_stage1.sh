#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_LIST[*]}")

# Safer single-node defaults for NCCL; override via env if needed.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=4

read -r -a TORCHRUN_CMD <<< "${TORCHRUN:-python -m torch.distributed.run}"

echo "Launching ${NUM_GPUS} processes on GPUs ${CUDA_VISIBLE_DEVICES} (master ${MASTER_ADDR}:${MASTER_PORT})"

"${TORCHRUN_CMD[@]}" \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --module moozy \
  train stage1 \
  --feature_dirs /path/to/patch_features
