import logging
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast

from moozy.data.stage1 import Stage1Batch

from ..optimization import clip_gradients_moozy_style
from ..runtime import is_main_process, reduce_dict
from .base import BaseEngine


def move_stage1_batch_to_device(batch: Stage1Batch, device: torch.device) -> None:
    """Move the canonical stage-1 batch contract onto the target device in place."""
    for key, value in batch.items():
        batch[key] = value.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_stage1(
    model: nn.Module,
    val_loader,
    device: torch.device,
    logger: logging.Logger,
    *,
    mixed_precision: bool = False,
    rank: int = 0,
) -> dict[str, float]:
    """Validate the stage-1 model against the current output contract."""
    model.eval()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_mim = 0.0
    count = 0

    for batch in val_loader:
        move_stage1_batch_to_device(batch, device)

        if mixed_precision:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                loss = outputs["loss_total"]
        else:
            outputs = model(batch)
            loss = outputs["loss_total"]

        total_loss += float(loss.item())
        total_loss_cls += float(outputs["loss_cls"].item())
        total_loss_mim += float(outputs["loss_mim"].item())
        count += 1

    if dist.is_initialized():
        metrics = {
            "loss": torch.tensor(total_loss, device=device),
            "loss_cls": torch.tensor(total_loss_cls, device=device),
            "loss_mim": torch.tensor(total_loss_mim, device=device),
            "count": torch.tensor(count, device=device),
        }
        metrics = reduce_dict(metrics, average=False)
        avg_loss = float(metrics["loss"].item()) / max(1.0, float(metrics["count"].item()))
        avg_loss_cls = float(metrics["loss_cls"].item()) / max(1.0, float(metrics["count"].item()))
        avg_loss_mim = float(metrics["loss_mim"].item()) / max(1.0, float(metrics["count"].item()))
    else:
        avg_loss = total_loss / max(1, count)
        avg_loss_cls = total_loss_cls / max(1, count)
        avg_loss_mim = total_loss_mim / max(1, count)

    if is_main_process(rank):
        logger.info(
            "Validation loss: %.6f (cls %.6f / mim %.6f)",
            avg_loss,
            avg_loss_cls,
            avg_loss_mim,
        )

    return {
        "loss": avg_loss,
        "loss_cls": avg_loss_cls,
        "loss_mim": avg_loss_mim,
    }


class Stage1Engine(BaseEngine):
    """Step-based training loop for stage-1 self-supervised pretraining.

    Scheduling and EMA are handled inline (ordering constraints).
    Logging and checkpointing are delegated to callbacks via ``self.fire()``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Any,
        callbacks: list[Any] | None = None,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        # Schedule / scheduler objects (kept inline — ordering constraints)
        lr_scheduler: Any,
        momentum_scheduler: Any,
        temperature_scheduler: Any = None,
        patch_temperature_scheduler: Any = None,
        wd_scheduler: Any = None,
        scaler: Any = None,
        # Loop parameters
        total_steps: int,
        accum_steps: int = 1,
        grad_clip: float = 0.0,
        mixed_precision: bool = False,
        val_every: int = 0,
        freeze_until_step: int = 0,
        # Distributed
        train_sampler: Any = None,
        rank: int = 0,
        logger: logging.Logger | None = None,
        wandb_module: Any = None,
        # Resume state
        start_step: int = 0,
        global_step: int = 0,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, callbacks=callbacks or [])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.temperature_scheduler = temperature_scheduler
        self.patch_temperature_scheduler = patch_temperature_scheduler
        self.wd_scheduler = wd_scheduler
        self.scaler = scaler
        self.total_steps = total_steps
        self.accum_steps = max(1, accum_steps)
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.val_every = val_every
        self.freeze_until_step = freeze_until_step
        self.train_sampler = train_sampler
        self.rank = rank
        self.logger = logger
        self.wandb_module = wandb_module
        self.global_step = global_step
        self.start_step = start_step

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Single forward/backward pass."""
        move_stage1_batch_to_device(batch, self.device)
        if self.mixed_precision:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(batch)
                loss = outputs["loss_total"] / self.accum_steps
            loss.backward()
        else:
            outputs = self.model(batch)
            loss = outputs["loss_total"] / self.accum_steps
            loss.backward()
        return {"loss": loss, "outputs": outputs}

    def _find_logging_callback(self) -> Any:
        for cb in self.callbacks:
            if hasattr(cb, "accumulate"):
                return cb
        return None

    def run(self) -> None:
        """Execute the full stage-1 training loop."""
        if is_main_process(self.rank):
            self.logger.info("Starting training from step %d...", self.start_step)  # type: ignore[union-attr]
            self.logger.info("Total steps: %d", self.total_steps)  # type: ignore[union-attr]

        self.model.train()
        logging_cb = self._find_logging_callback()
        running_loss = 0.0
        running_count = 0
        epoch = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        train_iter = iter(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        model_to_update = self.raw_model

        while self.global_step < self.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            step_result = self.training_step(batch)
            outputs = step_result["outputs"]
            loss = step_result["loss"]

            # Feed metrics to logging callback
            micro_loss = float(loss.item())
            micro_loss_cls = float((outputs["loss_cls"] / self.accum_steps).detach().item())
            micro_loss_mim = float((outputs["loss_mim"] / self.accum_steps).detach().item())
            metrics = outputs["metrics"]
            if logging_cb is not None:
                logging_cb.accumulate(micro_loss, micro_loss_cls, micro_loss_mim, metrics)
            running_loss += micro_loss
            running_count += 1

            accum_counter += 1

            if accum_counter % self.accum_steps == 0:
                freeze_active = self.global_step < self.freeze_until_step
                if freeze_active:
                    for parameter in model_to_update.student_head.last_layer.parameters():
                        parameter.grad = None

                momentum = self.momentum_scheduler.get_momentum()

                if self.grad_clip > 0:
                    clip_gradients_moozy_style(
                        chain(
                            model_to_update.student_slide_encoder.named_parameters(),
                            model_to_update.student_head.named_parameters(),
                        ),
                        self.grad_clip,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                self.momentum_scheduler.step()
                if self.temperature_scheduler is not None:
                    self.temperature_scheduler.step()
                if self.patch_temperature_scheduler is not None:
                    self.patch_temperature_scheduler.step()
                if self.wd_scheduler is not None:
                    self.wd_scheduler.step()
                    current_wd = self.wd_scheduler.get_weight_decay()
                    for param_group in self.optimizer.param_groups:
                        if param_group.get("apply_weight_decay", True):
                            param_group["weight_decay"] = current_wd
                        else:
                            param_group["weight_decay"] = 0.0

                model_to_update.update_teacher(momentum)

                if self.temperature_scheduler is not None:
                    model_to_update.tau_teacher = self.temperature_scheduler.get_temperature()
                if self.patch_temperature_scheduler is not None:
                    model_to_update.tau_teacher_patch = self.patch_temperature_scheduler.get_temperature()

                self.global_step += 1
                accum_counter = 0

                is_log_step = logging_cb is not None and self.global_step % logging_cb.log_every == 0
                if is_log_step:
                    eff_loss = (running_loss / max(1, running_count)) * self.accum_steps
                else:
                    eff_loss = float("inf")

                self.fire(
                    "on_step_end",
                    model=model_to_update,
                    global_step=self.global_step,
                    total_steps=self.total_steps,
                    optimizer=self.optimizer,
                    momentum=momentum,
                    wd_scheduler=self.wd_scheduler,
                    eff_loss=eff_loss,
                )
                if is_log_step:
                    running_loss = 0.0
                    running_count = 0

                if self.val_loader is not None and self.val_every > 0 and (self.global_step % self.val_every == 0):
                    val_metrics = evaluate_stage1(
                        self.model,
                        self.val_loader,
                        self.device,
                        self.logger,
                        mixed_precision=self.mixed_precision,
                        rank=self.rank,
                    )
                    if is_main_process(self.rank) and self.wandb_module is not None:
                        self.wandb_module.log(
                            {
                                "global_step": self.global_step,
                                "val/loss": val_metrics["loss"],
                                "val/loss_cls": val_metrics["loss_cls"],
                                "val/loss_mim": val_metrics["loss_mim"],
                            },
                            step=self.global_step,
                        )
                    self.model.train()

        if is_main_process(self.rank) and self.logger:
            self.logger.info("Training complete")

        self.fire("on_train_end", model=model_to_update, global_step=self.global_step)
