import logging
from typing import Any

import torch

from ..logging import log_stage2_epoch_summary
from ..runtime import is_main_process, reduce_dict


class Stage1LoggingCallback:
    """Metric reduction, console logging, and wandb payload (stage-1)."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        log_every: int,
        accum_steps: int,
        device: torch.device,
        logger: logging.Logger | None = None,
        wandb_module: Any = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.log_every = log_every
        self.accum_steps = accum_steps
        self.device = device
        self.logger = logger
        self.wandb_module = wandb_module
        self._reset()

    def _reset(self) -> None:
        self.running_loss = 0.0
        self.running_loss_cls = 0.0
        self.running_loss_mim = 0.0
        self.running_count = 0
        self.running_metrics: dict[str, float] = {}
        self.running_metrics_count = 0

    def accumulate(self, micro_loss: float, loss_cls: float, loss_mim: float, metrics: dict[str, float]) -> None:
        """Accumulate per-microbatch losses and metrics (called by engine every micro-step)."""
        self.running_loss += micro_loss
        self.running_loss_cls += loss_cls
        self.running_loss_mim += loss_mim
        self.running_count += 1
        if metrics:
            for key, value in metrics.items():
                self.running_metrics[key] = self.running_metrics.get(key, 0.0) + float(value)
            self.running_metrics_count += 1

    def on_step_end(
        self,
        *,
        global_step: int,
        total_steps: int,
        model: Any,
        optimizer: Any,
        momentum: float,
        wd_scheduler: Any = None,
        **kwargs: object,
    ) -> None:
        if global_step % self.log_every != 0:
            return

        if self.world_size > 1:
            reduced: dict[str, torch.Tensor] = {
                "loss": torch.tensor(self.running_loss, device=self.device),
                "loss_cls": torch.tensor(self.running_loss_cls, device=self.device),
                "loss_mim": torch.tensor(self.running_loss_mim, device=self.device),
                "count": torch.tensor(self.running_count, device=self.device),
            }
            for key, value in self.running_metrics.items():
                reduced[f"m__{key}"] = torch.tensor(value, device=self.device)
            reduced["m__count"] = torch.tensor(self.running_metrics_count, device=self.device)
            reduced = reduce_dict(reduced, average=False)
            count = max(1.0, float(reduced["count"].item()))
            avg_loss = float(reduced["loss"].item()) / count
            avg_loss_cls = float(reduced["loss_cls"].item()) / count
            avg_loss_mim = float(reduced["loss_mim"].item()) / count
        else:
            reduced = {}
            count = max(1, self.running_count)
            avg_loss = self.running_loss / count
            avg_loss_cls = self.running_loss_cls / count
            avg_loss_mim = self.running_loss_mim / count

        if not is_main_process(self.rank):
            self._reset()
            return

        eff_loss = avg_loss * self.accum_steps
        eff_loss_cls = avg_loss_cls * self.accum_steps
        eff_loss_mim = avg_loss_mim * self.accum_steps
        current_lr = optimizer.param_groups[0]["lr"]

        if self.world_size > 1:
            metric_count = float(reduced.get("m__count", torch.tensor(0.0, device=self.device)).item())

            def metric_avg(name: str) -> float | None:
                key = f"m__{name}"
                if key in reduced and metric_count > 0:
                    return float(reduced[key].item()) / metric_count
                return None
        else:
            metric_count = float(self.running_metrics_count)

            def metric_avg(name: str) -> float | None:
                if name in self.running_metrics and metric_count > 0:
                    return self.running_metrics[name] / metric_count
                return None

        log_msg = (
            f"Step {global_step}/{total_steps} - "
            f"loss {eff_loss:.6f} (cls {eff_loss_cls:.6f} / mim {eff_loss_mim:.6f}) - "
            f"lr {current_lr:.6e} - "
            f"momentum {momentum:.6f} - "
            f"tau_teacher_cls {model.tau_teacher:.4f} - "
            f"tau_teacher_patch {model.tau_teacher_patch:.4f} - "
            f"tau_student {model.tau_student:.4f}"
        )
        if wd_scheduler is not None:
            log_msg += f" - wd {optimizer.param_groups[0]['weight_decay']:.4f}"

        _METRIC_NAMES: list[str] = [
            "t_cls_entropy",
            "t_patch_entropy_masked",
            "t_cls_perplexity",
            "s_cls_perplexity",
            "t_cls_proto_max_frac",
            "t_cls_proto_eff_num",
            "t_cls_proto_nonzero",
            "t_patch_proto_max_frac",
            "t_patch_proto_eff_num",
            "t_patch_proto_nonzero",
            "acc",
        ]
        for name in _METRIC_NAMES:
            val = metric_avg(name)
            if val is not None:
                precision = (
                    ".2f"
                    if "entropy" in name
                    else (
                        ".1f"
                        if "perplexity" in name or "eff_num" in name or "nonzero" in name
                        else (".4f" if "frac" in name else ".3f")
                    )
                )
                log_msg += f" - {name} {val:{precision}}"
        if self.logger:
            self.logger.info(log_msg)

        if self.wandb_module is not None:
            payload: dict[str, object] = {
                "global_step": global_step,
                "train/loss": eff_loss,
                "train/loss_cls": eff_loss_cls,
                "train/loss_mim": eff_loss_mim,
                "train/learning_rate": current_lr,
                "train/momentum": momentum,
                "train/tau_teacher_cls": model.tau_teacher,
                "train/tau_teacher_patch": model.tau_teacher_patch,
                "train/tau_student": model.tau_student,
            }
            if wd_scheduler is not None:
                payload["train/weight_decay"] = optimizer.param_groups[0]["weight_decay"]
            for key in list(self.running_metrics.keys()):
                metric_val = metric_avg(key)
                if metric_val is not None:
                    payload[f"train/{key}"] = metric_val
            self.wandb_module.log(payload, step=global_step)

        self._reset()


class Stage2LoggingCallback:
    """Per-epoch logging (stage-2)."""

    def __init__(
        self,
        rank: int,
        logger: logging.Logger | None = None,
        wandb_module: Any = None,
    ) -> None:
        self.rank = rank
        self.logger = logger
        self.wandb_module = wandb_module

    def on_epoch_end(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_summary: dict[str, object],
        val_summary: dict[str, object] | None = None,
        global_step: int = 0,
        **kwargs: object,
    ) -> None:
        if not is_main_process(self.rank):
            return

        log_stage2_epoch_summary(
            self.logger,
            epoch=epoch,
            total_epochs=total_epochs,
            split_name="train",
            summary=train_summary,
        )
        if val_summary is not None:
            log_stage2_epoch_summary(
                self.logger,
                epoch=epoch,
                total_epochs=total_epochs,
                split_name="val",
                summary=val_summary,
            )

        if self.wandb_module is not None:
            payload: dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "epoch/train_loss": float(train_summary["loss"]),  # type: ignore[arg-type]
                "epoch/train_loss_cls_component": float(train_summary["loss_cls_component"]),  # type: ignore[arg-type]
                "epoch/train_loss_surv_component": float(train_summary["loss_surv_component"]),  # type: ignore[arg-type]
            }
            if val_summary is not None:
                payload["epoch/val_loss"] = float(val_summary["loss"])  # type: ignore[arg-type]
                payload["epoch/val_loss_cls_component"] = float(val_summary["loss_cls_component"])  # type: ignore[arg-type]
                payload["epoch/val_loss_surv_component"] = float(val_summary["loss_surv_component"])  # type: ignore[arg-type]
            self.wandb_module.log(payload, step=global_step)
