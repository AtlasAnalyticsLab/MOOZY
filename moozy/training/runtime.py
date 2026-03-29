import importlib
import logging
import os
import random
import socket
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from .logging import build_training_logger


def _find_free_port() -> int:
    """Find a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize distributed training with the repository's current semantics."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % torch.cuda.device_count()))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            if rank == 0:
                os.environ["MASTER_PORT"] = str(_find_free_port())
            else:
                time.sleep(5)
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        try:
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        except Exception as exc:
            logging.error("Failed to initialize distributed training: %s", exc)
            raise

        try:
            if torch.cuda.is_available() and dist.get_backend() == "nccl":
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
        except Exception as exc:
            logging.warning("Barrier synchronization failed: %s", exc)
            try:
                dist.barrier()
            except Exception as retry_exc:
                logging.error("All barrier attempts failed: %s", retry_exc)
                raise

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if the current process is rank zero."""
    return rank == 0


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce tensor values across all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return input_dict

    world_size = dist.get_world_size()
    with torch.no_grad():
        names = []
        values = []
        for key in sorted(input_dict.keys()):
            names.append(key)
            values.append(input_dict[key])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        return {key: value for key, value in zip(names, values)}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's worker-local seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class TrainingRuntime:
    """Shared bootstrap state returned to stage entrypoints."""

    rank: int
    local_rank: int
    world_size: int
    output_dir: str
    checkpoint_dir: str
    logger: logging.Logger


def _prepare_output_dir(output_dir: str, rank: int, world_size: int) -> str:
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    return checkpoint_dir


def initialize_training_runtime(
    *,
    backend: str,
    output_dir: str,
    seed: int,
    logger_name: str,
    log_seed: bool = False,
) -> TrainingRuntime:
    """Initialize distributed state, output paths, logger, and random seed."""
    rank, local_rank, world_size = setup_distributed(backend)
    checkpoint_dir = _prepare_output_dir(output_dir, rank, world_size)
    logger = build_training_logger(
        output_dir=output_dir,
        logger_name=logger_name,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process(rank):
        logger.info("Distributed training: %s", "Enabled" if world_size > 1 else "Disabled")
        logger.info("World size: %d, Rank: %d, Local rank: %d", world_size, rank, local_rank)
        logger.info("Output directory: %s", output_dir)
        if log_seed:
            logger.info("Setting random seed to %d", seed)

    set_seed(seed + rank)
    return TrainingRuntime(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )


def initialize_wandb(
    *,
    enabled: bool,
    rank: int,
    logger: logging.Logger,
    project: str,
    config: Mapping[str, Any],
    name: str,
    output_dir: str,
    tags: Sequence[str] | None = None,
    module: Any | None = None,
    warn_on_import_error: bool = False,
) -> Any | None:
    """Initialize a W&B run when enabled for the main process."""
    if not enabled or not is_main_process(rank):
        return None

    wandb_module = module
    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except Exception as exc:
            if warn_on_import_error:
                logger.warning("Weights & Biases disabled: %s", exc)
                return None
            raise

    wandb_module.init(
        project=project,
        config=config,
        name=name,
        dir=output_dir,
        tags=tags if tags else None,
    )
    wandb_module.define_metric("*", step_metric="global_step")
    logger.info("Weights & Biases logging enabled")
    return wandb_module


def finish_wandb(wandb_module: Any | None) -> None:
    """Finish an active W&B run."""
    if wandb_module is not None:
        wandb_module.finish()


__all__ = [
    "TrainingRuntime",
    "cleanup_distributed",
    "finish_wandb",
    "initialize_training_runtime",
    "initialize_wandb",
    "is_main_process",
    "reduce_dict",
    "seed_worker",
]
