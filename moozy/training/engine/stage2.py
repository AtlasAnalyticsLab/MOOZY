import logging
import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from ..loss import compute_cindex
from ..optimization import clip_gradients_moozy_style
from ..runtime import is_main_process, reduce_dict
from .base import BaseEngine


def move_stage2_batch_to_device(batch: dict[str, Any], device: torch.device) -> None:
    """Move the current stage-2 batch contract onto the target device in place."""
    for case in batch.get("cases", []):
        if torch.is_tensor(case.get("task_labels")):
            case["task_labels"] = case["task_labels"].to(device, non_blocking=True)
        if torch.is_tensor(case.get("task_events")):
            case["task_events"] = case["task_events"].to(device, non_blocking=True)
        if torch.is_tensor(case.get("task_times")):
            case["task_times"] = case["task_times"].to(device, non_blocking=True)


def gather_survival_triplets(
    preds: torch.Tensor | None,
    events: torch.Tensor | None,
    times: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Gather survival predictions and targets to rank 0 for c-index computation."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return preds, events, times

    gathered: list[dict[str, torch.Tensor | None] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, {"preds": preds, "events": events, "times": times})

    if dist.get_rank() != 0:
        return None, None, None

    preds_parts = []
    events_parts = []
    times_parts = []
    for payload in gathered:
        if not payload:
            continue
        part_preds = payload.get("preds")
        part_events = payload.get("events")
        part_times = payload.get("times")
        if part_preds is None or part_events is None or part_times is None:
            continue
        if part_preds.numel() == 0:
            continue
        preds_parts.append(part_preds)
        events_parts.append(part_events)
        times_parts.append(part_times)

    if not preds_parts:
        return torch.empty(0), torch.empty(0, dtype=torch.int64), torch.empty(0)
    return torch.cat(preds_parts, dim=0), torch.cat(events_parts, dim=0), torch.cat(times_parts, dim=0)


def summarize_stage2_epoch_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    task_keys: list[str],
    task_types: list[str],
) -> dict[str, Any]:
    """Summarize reduced stage-2 epoch metrics into the current reporting contract."""
    sample_count = max(1.0, float(metrics["sample_count"].item()))
    summary: dict[str, Any] = {
        "loss": float(metrics["loss_sum"].item()) / sample_count,
        "loss_cls_component": float(metrics["loss_cls_component_sum"].item()) / sample_count,
        "loss_surv_component": float(metrics["loss_surv_component_sum"].item()) / sample_count,
        "task_metrics": [],
    }
    task_summaries = []
    for idx, key in enumerate(task_keys):
        task_type = task_types[idx] if idx < len(task_types) else "classification"
        if task_type == "survival":
            cindex = float(metrics.get(f"task_cindex_{key}", float("nan")))
            total = float(metrics.get(f"task_total_{key}", 0.0))
            if total > 0 and not math.isnan(cindex):
                task_summaries.append(
                    {
                        "key": key,
                        "task_type": task_type,
                        "value": cindex,
                        "count": int(total),
                    }
                )
        else:
            correct = float(metrics.get(f"task_correct_{key}", 0.0))
            total = float(metrics.get(f"task_total_{key}", 0.0))
            if total > 0:
                task_summaries.append(
                    {
                        "key": key,
                        "task_type": task_type,
                        "value": 100.0 * correct / total,
                        "count": int(total),
                    }
                )
    summary["task_metrics"] = task_summaries
    return summary


class Stage2Engine(BaseEngine):
    """Epoch-based training loop for stage-2 supervised alignment.

    Logging and checkpointing are delegated to callbacks via ``self.fire()``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Any,
        callbacks: list[Any] | None = None,
        train_loader: DataLoader,
        val_loader: Any = None,
        device: torch.device,
        scheduler: Any = None,
        # Loop parameters
        epochs: int = 30,
        grad_accumulation_steps: int = 1,
        grad_clip: float = 0.0,
        mixed_precision: bool = False,
        log_every: int = 50,
        # Distributed
        train_sampler: Any = None,
        rank: int = 0,
        logger: logging.Logger | None = None,
        wandb_module: Any = None,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, callbacks=callbacks or [])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.log_every = log_every
        self.train_sampler = train_sampler
        self.rank = rank
        self.logger = logger
        self.wandb_module = wandb_module

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
        global_step: int,
        accum_counter_state: dict[str, int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Run one train or validation epoch for stage-2 supervised training."""
        # Local aliases so the loop body matches the original standalone function.
        model = self.model
        device = self.device
        mixed_precision = self.mixed_precision
        optimizer = self.optimizer
        scheduler = self.scheduler
        grad_accumulation_steps = self.grad_accumulation_steps
        grad_clip = self.grad_clip
        log_every = self.log_every if train else 0
        logger = self.logger if train and is_main_process(self.rank) else None
        wandb_module = self.wandb_module if train else None

        if train:
            model.train()
        else:
            model.eval()

        accum_steps = max(1, int(grad_accumulation_steps))
        accum_counter = int(accum_counter_state.get("value", 0)) if train and accum_counter_state is not None else 0
        epoch_accum_start = int(accum_counter)
        expected_steps_in_epoch = (epoch_accum_start + len(loader)) // accum_steps - (epoch_accum_start // accum_steps)
        display_steps_in_epoch = max(1, int(expected_steps_in_epoch))
        loss_sum = torch.tensor(0.0, device=device)
        loss_cls_component_sum = torch.tensor(0.0, device=device)
        loss_surv_component_sum = torch.tensor(0.0, device=device)
        sample_count = torch.tensor(0.0, device=device)
        task_correct: dict[str, torch.Tensor] = {}
        task_total: dict[str, torch.Tensor] = {}
        accum_loss_sum = torch.tensor(0.0, device=device)
        accum_loss_cls_component_sum = torch.tensor(0.0, device=device)
        accum_loss_surv_component_sum = torch.tensor(0.0, device=device)
        accum_sample_count = torch.tensor(0.0, device=device)
        optimizer_steps_in_epoch = 0

        def reduce_step_window_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            log_loss_sum = accum_loss_sum.detach().clone()
            log_loss_cls_component_sum = accum_loss_cls_component_sum.detach().clone()
            log_loss_surv_component_sum = accum_loss_surv_component_sum.detach().clone()
            log_sample_count = accum_sample_count.detach().clone()
            if dist.is_available() and dist.is_initialized():
                reduced = reduce_dict(
                    {
                        "loss_sum": log_loss_sum,
                        "loss_cls_component_sum": log_loss_cls_component_sum,
                        "loss_surv_component_sum": log_loss_surv_component_sum,
                        "sample_count": log_sample_count,
                    },
                    average=False,
                )
                log_loss_sum = reduced["loss_sum"]
                log_loss_cls_component_sum = reduced["loss_cls_component_sum"]
                log_loss_surv_component_sum = reduced["loss_surv_component_sum"]
                log_sample_count = reduced["sample_count"]
            return log_loss_sum, log_loss_cls_component_sum, log_loss_surv_component_sum, log_sample_count

        model_ref = self.raw_model
        if train and not any(p.requires_grad for p in model_ref.slide_encoder.parameters()):
            model_ref.slide_encoder.eval()
        task_keys = list(model_ref.task_heads.keys())
        task_types = list(model_ref.task_types)

        survival_preds: dict[str, list[torch.Tensor]] = {
            key: [] for key, task_type in zip(task_keys, task_types) if task_type == "survival"
        }
        survival_events: dict[str, list[torch.Tensor]] = {
            key: [] for key, task_type in zip(task_keys, task_types) if task_type == "survival"
        }
        survival_times: dict[str, list[torch.Tensor]] = {
            key: [] for key, task_type in zip(task_keys, task_types) if task_type == "survival"
        }
        amp_enabled = bool(mixed_precision) and device.type == "cuda"

        def run_batch(
            batch: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            batch_loss_sum = torch.tensor(0.0, device=device)
            batch_loss_cls_component_sum = torch.tensor(0.0, device=device)
            batch_loss_surv_component_sum = torch.tensor(0.0, device=device)
            batch_sample_count = torch.tensor(0.0, device=device)
            batch_case_count = len(batch["cases"])

            amp_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else nullcontext()
            if train:
                with amp_ctx:
                    outputs = model(batch)
                    loss = outputs["loss"]
            else:
                with torch.no_grad():
                    with amp_ctx:
                        outputs = model(batch)
                        loss = outputs["loss"]

            loss_value = loss.detach().float()
            cls_component = outputs.get("loss_cls_component")
            if cls_component is None:
                cls_component = torch.tensor(0.0, device=device)
            elif not torch.is_tensor(cls_component):
                cls_component = torch.tensor(float(cls_component), device=device)
            else:
                cls_component = cls_component.detach().float()
            surv_component = outputs.get("loss_surv_component")
            if surv_component is None:
                surv_component = torch.tensor(0.0, device=device)
            elif not torch.is_tensor(surv_component):
                surv_component = torch.tensor(float(surv_component), device=device)
            else:
                surv_component = surv_component.detach().float()
            output_sample_count = outputs.get("sample_count")
            if output_sample_count is None:
                output_sample_count = torch.tensor(float(batch_case_count), device=device)
            elif not torch.is_tensor(output_sample_count):
                output_sample_count = torch.tensor(float(output_sample_count), device=device)
            weighted_sample_count = output_sample_count
            loss_sum.add_(loss_value * weighted_sample_count)
            loss_cls_component_sum.add_(cls_component * weighted_sample_count)
            loss_surv_component_sum.add_(surv_component * weighted_sample_count)
            sample_count.add_(weighted_sample_count)
            batch_loss_sum.add_(loss_value * weighted_sample_count)
            batch_loss_cls_component_sum.add_(cls_component * weighted_sample_count)
            batch_loss_surv_component_sum.add_(surv_component * weighted_sample_count)
            batch_sample_count.add_(weighted_sample_count)

            if train:
                (loss / accum_steps).backward()

            if outputs.get("logits"):
                labels = outputs.get("task_labels")
                if labels is None and "task_labels" in batch:
                    labels = batch["task_labels"]
                batch_events = outputs.get("task_events")
                if batch_events is None:
                    batch_events = batch.get("task_events")
                batch_times = outputs.get("task_times")
                if batch_times is None:
                    batch_times = batch.get("task_times")

                for idx, key in enumerate(task_keys):
                    task_type = task_types[idx] if idx < len(task_types) else "classification"
                    logits = outputs["logits"].get(key)
                    if logits is None:
                        continue

                    if task_type == "survival":
                        if batch_events is not None and batch_times is not None:
                            events = batch_events[:, idx]
                            times = batch_times[:, idx]
                            valid = (events >= 0) & (times >= 0)
                            if valid.any():
                                risk_scores = logits if logits.dim() == 1 else logits.squeeze(-1)
                                survival_preds[key].append(risk_scores[valid].detach().cpu())
                                survival_events[key].append(events[valid].detach().cpu())
                                survival_times[key].append(times[valid].detach().cpu())
                    else:
                        task_labels = labels[:, idx] if labels is not None else None
                        if task_labels is None:
                            continue
                        valid = task_labels >= 0
                        if not valid.any():
                            continue
                        preds = logits.argmax(dim=-1)
                        correct = (preds[valid] == task_labels[valid]).sum()
                        total = valid.sum()
                        task_correct[key] = task_correct.get(key, torch.tensor(0, device=device)) + correct
                        task_total[key] = task_total.get(key, torch.tensor(0, device=device)) + total
            return (
                batch_loss_sum,
                batch_loss_cls_component_sum,
                batch_loss_surv_component_sum,
                batch_sample_count,
            )

        for batch in loader:
            move_stage2_batch_to_device(batch, device)

            if train and accum_counter % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            (
                batch_loss_sum,
                batch_loss_cls_component_sum,
                batch_loss_surv_component_sum,
                batch_sample_count,
            ) = run_batch(batch=batch)
            accum_loss_sum.add_(batch_loss_sum)
            accum_loss_cls_component_sum.add_(batch_loss_cls_component_sum)
            accum_loss_surv_component_sum.add_(batch_loss_surv_component_sum)
            accum_sample_count.add_(batch_sample_count)

            if train:
                accum_counter += 1

            if train and accum_counter % accum_steps == 0:
                if grad_clip and grad_clip > 0.0:
                    trainable_params = [
                        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
                    ]
                    clip_gradients_moozy_style(trainable_params, grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                optimizer_steps_in_epoch += 1
                if train and log_every and global_step % log_every == 0:
                    (
                        reduced_loss_sum,
                        reduced_loss_cls_component_sum,
                        reduced_loss_surv_component_sum,
                        reduced_sample_count,
                    ) = reduce_step_window_tensors()
                    batch_loss = float(reduced_loss_sum.item()) / max(1.0, float(reduced_sample_count.item()))
                    batch_loss_cls_component = float(reduced_loss_cls_component_sum.item()) / max(
                        1.0, float(reduced_sample_count.item())
                    )
                    batch_loss_surv_component = float(reduced_loss_surv_component_sum.item()) / max(
                        1.0, float(reduced_sample_count.item())
                    )
                    accum_batch = int(reduced_sample_count.item())
                    current_lr = optimizer.param_groups[0]["lr"]
                    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
                    if logger:
                        logger.info(
                            "Epoch step %d/%d (global %d) - loss %.6f - cls %.6f - surv %.6f - lr %.2e - wd %.2e - accum batch %d - effective global %d",
                            optimizer_steps_in_epoch,
                            display_steps_in_epoch,
                            global_step,
                            batch_loss,
                            batch_loss_cls_component,
                            batch_loss_surv_component,
                            current_lr,
                            current_wd,
                            accum_batch,
                            accum_batch,
                        )
                    if wandb_module is not None:
                        wandb_module.log(
                            {
                                "train/loss": batch_loss,
                                "train/loss_cls_component": batch_loss_cls_component,
                                "train/loss_surv_component": batch_loss_surv_component,
                                "train/lr": current_lr,
                                "train/wd": current_wd,
                                "train/epoch_step": optimizer_steps_in_epoch,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )
                accum_loss_sum.zero_()
                accum_loss_cls_component_sum.zero_()
                accum_loss_surv_component_sum.zero_()
                accum_sample_count.zero_()

        if train and accum_counter_state is not None:
            accum_counter_state["value"] = int(accum_counter)

        if train and optimizer_steps_in_epoch != expected_steps_in_epoch and logger:
            logger.warning(
                "Optimizer step count mismatch in epoch: observed=%d, expected=%d (loader_len=%d, accum_steps=%d).",
                int(optimizer_steps_in_epoch),
                int(expected_steps_in_epoch),
                len(loader),
                int(accum_steps),
            )

        metrics = {
            "loss_sum": loss_sum,
            "loss_cls_component_sum": loss_cls_component_sum,
            "loss_surv_component_sum": loss_surv_component_sum,
            "sample_count": sample_count,
        }
        for idx, key in enumerate(task_keys):
            task_type = task_types[idx] if idx < len(task_types) else "classification"
            if task_type == "survival":
                local_preds = None
                local_events = None
                local_times = None
                if survival_preds.get(key):
                    local_preds = torch.cat(survival_preds[key], dim=0)
                    local_events = torch.cat(survival_events[key], dim=0)
                    local_times = torch.cat(survival_times[key], dim=0)

                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    all_preds, all_events, all_times = gather_survival_triplets(
                        local_preds,
                        local_events,
                        local_times,
                    )
                    if dist.get_rank() == 0:
                        if all_preds is not None and all_preds.numel() > 0:
                            cindex = compute_cindex(all_preds, all_events, all_times)
                            metrics[f"task_cindex_{key}"] = torch.tensor(cindex, device=device)
                            metrics[f"task_total_{key}"] = torch.tensor(int(all_preds.numel()), device=device)
                        else:
                            metrics[f"task_cindex_{key}"] = torch.tensor(float("nan"), device=device)
                            metrics[f"task_total_{key}"] = torch.tensor(0, device=device)
                    else:
                        metrics[f"task_cindex_{key}"] = torch.tensor(0.0, device=device)
                        metrics[f"task_total_{key}"] = torch.tensor(0, device=device)
                else:
                    if local_preds is not None and local_preds.numel() > 0:
                        cindex = compute_cindex(local_preds, local_events, local_times)
                        metrics[f"task_cindex_{key}"] = torch.tensor(cindex, device=device)
                        metrics[f"task_total_{key}"] = torch.tensor(int(local_preds.numel()), device=device)
                    else:
                        metrics[f"task_cindex_{key}"] = torch.tensor(float("nan"), device=device)
                        metrics[f"task_total_{key}"] = torch.tensor(0, device=device)
            else:
                metrics[f"task_correct_{key}"] = task_correct.get(key, torch.tensor(0, device=device))
                metrics[f"task_total_{key}"] = task_total.get(key, torch.tensor(0, device=device))
        return metrics, global_step

    def run(self) -> None:
        """Execute the full stage-2 epoch-based training loop."""
        global_step = 0
        accum_counter_state: dict[str, int] = {"value": 0}

        model_ref = self.raw_model
        task_keys = list(model_ref.task_heads.keys())
        task_types = list(model_ref.task_types)

        for epoch in range(1, self.epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_metrics, global_step = self._run_epoch(
                self.train_loader,
                train=True,
                global_step=global_step,
                accum_counter_state=accum_counter_state,
            )

            if self.val_loader is not None:
                val_metrics, _ = self._run_epoch(
                    self.val_loader,
                    train=False,
                    global_step=global_step,
                )
            else:
                val_metrics = None

            if dist.is_available() and dist.is_initialized():
                train_metrics = reduce_dict(train_metrics, average=False)
                if val_metrics is not None:
                    val_metrics = reduce_dict(val_metrics, average=False)

            train_summary = summarize_stage2_epoch_metrics(
                train_metrics,
                task_keys=task_keys,
                task_types=task_types,
            )
            val_summary: dict[str, Any] | None = None
            if val_metrics is not None:
                val_summary = summarize_stage2_epoch_metrics(
                    val_metrics,
                    task_keys=task_keys,
                    task_types=task_types,
                )

            self.fire(
                "on_epoch_end",
                model=self.raw_model,
                epoch=epoch,
                total_epochs=self.epochs,
                global_step=global_step,
                train_summary=train_summary,
                val_summary=val_summary,
            )

        self.fire("on_train_end", model=self.raw_model, global_step=global_step)
