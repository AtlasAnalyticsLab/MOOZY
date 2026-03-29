import glob
import logging
import os
import random
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from moozy.models.serialization import (
    build_slide_encoder_save_meta,
    extract_case_transformer_config,
    extract_slide_encoder_config,
)

from ..runtime import is_main_process


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    global_step: int,
    args_dict: dict,
    checkpoint_dir: str,
    filename: str | None = None,
    extra_state: dict | None = None,
    extra_payload: dict | None = None,
) -> str:
    if filename is None:
        filename = f"checkpoint_step_{global_step}.pt"

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args_dict,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
    }

    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state
    if extra_payload:
        checkpoint.update(extra_payload)

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_teacher_checkpoint(model: nn.Module, out_path: str) -> str:
    payload = {
        "teacher_slide_encoder": model.teacher_slide_encoder.state_dict(),
        "meta": build_slide_encoder_save_meta(model.teacher_slide_encoder, output_dim=model.output_dim),
    }
    torch.save(payload, out_path)
    return out_path


def save_supervised_checkpoint(model, out_path: str) -> str:
    payload = {
        "teacher_slide_encoder": model.slide_encoder.state_dict(),
        "case_transformer": model.case_transformer.state_dict(),
        "slide_encoder_config": extract_slide_encoder_config(model.slide_encoder),
        "case_transformer_config": extract_case_transformer_config(model.case_transformer),
    }
    torch.save(payload, out_path)
    return out_path


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int,
    prefixes: list[str] | None = None,
):
    if keep_last_n <= 0:
        return
    prefixes = prefixes or []

    def _prune(pattern: str):
        files = glob.glob(pattern)
        if len(files) <= keep_last_n:
            return
        files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for old_checkpoint in files[keep_last_n:]:
            try:
                os.remove(old_checkpoint)
            except OSError as e:
                logging.warning("Failed to remove old checkpoint %s: %s", old_checkpoint, e)

    _prune(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    for prefix in prefixes:
        _prune(os.path.join(checkpoint_dir, f"{prefix}_*.pt"))


class Stage1CheckpointCallback:
    """Periodic save, best-model save, teacher save, and cleanup (stage-1)."""

    def __init__(
        self,
        checkpoint_dir: str,
        args_dict: dict,
        optimizer: Any,
        lr_scheduler: Any,
        scaler: Any,
        build_extra_state: Callable[[nn.Module], dict[str, object]],
        *,
        save_every: int = 0,
        keep_last_n: int = 0,
        save_teacher: bool = False,
        teacher_save_prefix: str = "teacher_step",
        rank: int = 0,
        world_size: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.args_dict = args_dict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.build_extra_state = build_extra_state
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.save_teacher = save_teacher
        self.teacher_save_prefix = teacher_save_prefix
        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        self.best_train_loss = float("inf")
        self.best_train_step: int | None = None

    def _unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def on_step_end(
        self, *, model: nn.Module, global_step: int, eff_loss: float = float("inf"), **kwargs: object
    ) -> None:
        # Best-model tracking
        if eff_loss < self.best_train_loss and is_main_process(self.rank):
            self.best_train_loss = float(eff_loss)
            self.best_train_step = int(global_step)
            model_to_save = self._unwrap(model)
            best_filename = f"best_checkpoint_step_{global_step}.pt"
            best_path = save_checkpoint(
                model_to_save,
                self.optimizer,
                self.lr_scheduler,
                self.scaler,
                global_step,
                self.args_dict,
                self.checkpoint_dir,
                filename=best_filename,
                extra_state=self.build_extra_state(model_to_save),
            )
            if self.logger:
                self.logger.info(
                    "New best train loss %.6f at step %d; saved %s",
                    self.best_train_loss,
                    global_step,
                    best_path,
                )
            if self.save_teacher:
                teacher_best = os.path.join(self.checkpoint_dir, f"teacher_best_step_{global_step}.pt")
                save_teacher_checkpoint(model_to_save, teacher_best)

        # Periodic save
        if self.save_every <= 0 or global_step <= 0:
            return
        if global_step % self.save_every != 0:
            return

        if self.world_size > 1:
            dist.barrier()

        if not is_main_process(self.rank):
            return

        model_to_save = self._unwrap(model)
        checkpoint_path = save_checkpoint(
            model_to_save,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            global_step,
            self.args_dict,
            self.checkpoint_dir,
            extra_state=self.build_extra_state(model_to_save),
        )
        if self.logger:
            self.logger.debug("Checkpoint saved: %s", checkpoint_path)

        if self.save_teacher:
            teacher_filename = f"{self.teacher_save_prefix}_{global_step}.pt"
            teacher_path = os.path.join(self.checkpoint_dir, teacher_filename)
            save_teacher_checkpoint(model_to_save, teacher_path)

        if self.keep_last_n > 0:
            prefixes: list[str] = []
            if self.save_teacher:
                prefixes.append(self.teacher_save_prefix)
            cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last_n, prefixes)

    def on_train_end(self, *, model: nn.Module, global_step: int, **kwargs: object) -> None:
        if self.save_every <= 0:
            return

        if self.world_size > 1:
            dist.barrier()

        if not is_main_process(self.rank):
            return

        model_to_save = self._unwrap(model)
        final_path = save_checkpoint(
            model_to_save,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            global_step,
            self.args_dict,
            self.checkpoint_dir,
            "final_checkpoint.pt",
            extra_state=self.build_extra_state(model_to_save),
        )
        if self.logger:
            self.logger.debug("Final checkpoint saved: %s", final_path)

        if self.save_teacher:
            teacher_final = os.path.join(self.checkpoint_dir, "teacher_final.pt")
            save_teacher_checkpoint(model_to_save, teacher_final)


class Stage2CheckpointCallback:
    """Periodic epoch save, best-val save, and final save (stage-2)."""

    def __init__(
        self,
        output_dir: str,
        *,
        save_every_epochs: int = 1,
        keep_last_n: int = 50,
        rank: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.save_every_epochs = save_every_epochs
        self.keep_last_n = keep_last_n
        self.rank = rank
        self.logger = logger
        self.best_val_loss = float("inf")
        self.best_val_step = 0

    def on_epoch_end(
        self,
        *,
        model: nn.Module,
        epoch: int,
        global_step: int,
        val_summary: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        if is_main_process(self.rank) and val_summary is not None:
            val_loss = float(val_summary["loss"])  # type: ignore[arg-type]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_step = global_step
                best_path = os.path.join(self.output_dir, "moozy_best.pt")
                save_supervised_checkpoint(model, best_path)
                if self.logger:
                    self.logger.info(
                        "New best val loss %.6f at step %d; saved to %s",
                        self.best_val_loss,
                        self.best_val_step,
                        best_path,
                    )

        if is_main_process(self.rank) and self.save_every_epochs and epoch % self.save_every_epochs == 0:
            supervised_epoch = os.path.join(self.output_dir, f"moozy_epoch_{epoch}.pt")
            save_supervised_checkpoint(model, supervised_epoch)
            if self.logger:
                self.logger.info("Saved Stage-2 MOOZY checkpoint to %s", supervised_epoch)
            cleanup_old_checkpoints(
                self.output_dir,
                keep_last_n=self.keep_last_n,
                prefixes=["moozy_epoch"],
            )

    def on_train_end(self, *, model: nn.Module, global_step: int, **kwargs: object) -> None:
        if not is_main_process(self.rank):
            return

        save_path = os.path.join(self.output_dir, "moozy_final.pt")
        save_supervised_checkpoint(model, save_path)
        if self.logger:
            self.logger.info("Saved final MOOZY checkpoint to %s", save_path)
