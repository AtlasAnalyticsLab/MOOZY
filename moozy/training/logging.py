import json
import logging
import os
from typing import Any, Sequence

import torch


def configure_logger_for_rank(logger: logging.Logger, rank: int, world_size: int) -> None:
    """Restrict noisy INFO logging to rank zero in distributed runs."""
    if world_size > 1 and rank != 0:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)


def build_training_logger(
    *,
    output_dir: str,
    logger_name: str,
    rank: int,
    world_size: int,
) -> logging.Logger:
    """Create the repository's standard training logger."""
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )
    logger = logging.getLogger(logger_name)
    configure_logger_for_rank(logger, rank, world_size)
    return logger


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """Count module parameters, optionally restricted to trainable ones."""
    return sum(parameter.numel() for parameter in module.parameters() if not trainable_only or parameter.requires_grad)


def log_parameter_breakdown(
    logger: logging.Logger,
    rows: Sequence[tuple[str, int, int | None]],
) -> None:
    """Log a shared parameter-count table for training entrypoints."""
    total_params = sum(total for _, total, _ in rows)
    trainable_params = sum(trainable or 0 for _, _, trainable in rows)

    logger.info("=" * 70)
    logger.info("MODEL PARAMETER BREAKDOWN:")
    logger.info("=" * 70)
    for label, total, trainable in rows:
        suffix = "(frozen)" if trainable is None else f"(trainable: {trainable:,})"
        logger.info("%-20s %12s params  %s", f"{label}:", f"{total:,}", suffix)
    logger.info("-" * 70)
    logger.info("%-20s %12s params", "Total parameters:", f"{total_params:,}")
    logger.info("%-20s %12s params", "Trainable parameters:", f"{trainable_params:,}")
    logger.info("%-20s %12s params", "Frozen parameters:", f"{total_params - trainable_params:,}")
    logger.info("=" * 70)


def save_hyperparameters(args_dict: dict, output_dir: str) -> str:
    """Save the current config dict to JSON for experiment tracking."""
    hyperparams_file = os.path.join(output_dir, "hyperparameters.json")
    hyperparams = dict(args_dict)
    for key, value in hyperparams.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            hyperparams[key] = str(value)
    with open(hyperparams_file, "w", encoding="utf-8") as handle:
        json.dump(hyperparams, handle, indent=2, sort_keys=True)
    return hyperparams_file


def log_stage2_epoch_summary(
    logger: logging.Logger | None,
    *,
    epoch: int,
    total_epochs: int,
    split_name: str,
    summary: dict[str, Any],
) -> None:
    """Log the current stage-2 epoch summary contract."""
    if logger is None:
        return
    logger.info(
        "Epoch %d/%d - %s loss %.6f (cls %.6f, surv %.6f)",
        int(epoch),
        int(total_epochs),
        split_name,
        float(summary["loss"]),
        float(summary["loss_cls_component"]),
        float(summary["loss_surv_component"]),
    )
    for task_metric in summary.get("task_metrics", []):
        if task_metric["task_type"] == "survival":
            logger.info(
                "  %s/%s c-index %.4f (n=%d)",
                split_name,
                task_metric["key"],
                float(task_metric["value"]),
                int(task_metric["count"]),
            )
        else:
            logger.info(
                "  %s/%s acc %.2f%%",
                split_name,
                task_metric["key"],
                float(task_metric["value"]),
            )
