import math

import torch


def build_optimizer(
    model,
    optimizer_name: str,
    *,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
):
    """Build the current optimizer parameter groups and optimizer instance."""
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or parameter.ndim == 1:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "apply_weight_decay": True},
        {"params": no_decay_params, "weight_decay": 0.0, "apply_weight_decay": False},
    ]

    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(beta1, beta2))
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=(beta1, beta2))
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def clip_gradients_moozy_style(named_parameters, clip: float):
    """Per-parameter gradient clipping used by the current training recipe."""
    norms = []
    clip = float(clip)
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.data.norm(2)
        norms.append(float(param_norm.item()))
        clip_coef = clip / (param_norm + 1e-6)
        if clip_coef < 1:
            parameter.grad.data.mul_(clip_coef)
    return norms


def set_fixed_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    """Match stage-1 param-group semantics with fixed, non-scheduled WD."""
    wd = float(weight_decay)
    for param_group in optimizer.param_groups:
        if param_group.get("apply_weight_decay", True):
            param_group["weight_decay"] = wd
        else:
            param_group["weight_decay"] = 0.0


def build_warmup_scheduler(
    optimizer,
    *,
    warmup_steps: int,
    total_steps: int = None,
    min_lr: float = 0.0,
    schedule: str = "cosine",
):
    """Build the current warmup + decay LR scheduler."""
    schedule_lower = schedule.lower()

    def _factory(base_lr: float):
        min_lr_factor = min_lr / max(base_lr, 1e-12)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            if total_steps is not None and current_step < total_steps:
                steps_after_warmup = max(1, total_steps - warmup_steps)
                progress = float(current_step - warmup_steps) / steps_after_warmup
                if schedule_lower == "cosine":
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    return float(min_lr_factor + (1 - min_lr_factor) * cosine)
                return float(min_lr_factor + (1 - min_lr_factor) * max(0.0, 1.0 - progress))
            return 1.0

        return lr_lambda

    base_lr = optimizer.param_groups[0]["lr"]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_factory(base_lr=base_lr),
    )


class MomentumScheduler:
    """EMA momentum scheduler (supports cosine or linear)."""

    def __init__(
        self,
        initial_momentum: float = 0.996,
        final_momentum: float = 0.9999,
        total_steps: int = 0,
        schedule: str = "cosine",
    ):
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.total_steps = max(1, total_steps)
        self.schedule = schedule.lower()
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1

    def state_dict(self) -> dict:
        return {
            "initial_momentum": self.initial_momentum,
            "final_momentum": self.final_momentum,
            "total_steps": self.total_steps,
            "schedule": self.schedule,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict):
        self.initial_momentum = float(state["initial_momentum"])
        self.final_momentum = float(state["final_momentum"])
        self.total_steps = int(state["total_steps"])
        self.schedule = str(state["schedule"]).lower()
        self.current_step = int(state["current_step"])

    def get_momentum(self) -> float:
        progress = min(1.0, self.current_step / self.total_steps)
        if self.schedule == "cosine":
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.final_momentum - (self.final_momentum - self.initial_momentum) * cosine
        return self.initial_momentum + progress * (self.final_momentum - self.initial_momentum)


class TemperatureScheduler:
    """Temperature warmup scheduler for teacher logits."""

    def __init__(
        self,
        initial_tau: float = 0.04,
        final_tau: float = 0.07,
        warmup_steps: int = 0,
    ):
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.warmup_steps = max(0, int(warmup_steps))
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1

    def state_dict(self) -> dict:
        return {
            "initial_tau": self.initial_tau,
            "final_tau": self.final_tau,
            "warmup_steps": self.warmup_steps,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict):
        self.initial_tau = float(state["initial_tau"])
        self.final_tau = float(state["final_tau"])
        self.warmup_steps = int(state["warmup_steps"])
        self.current_step = int(state["current_step"])

    def get_temperature(self) -> float:
        if self.warmup_steps == 0:
            return self.final_tau
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            return self.initial_tau + progress * (self.final_tau - self.initial_tau)
        return self.final_tau


class WeightDecayScheduler:
    """Weight-decay scheduler used by stage-1 training."""

    def __init__(
        self,
        initial_wd: float = 0.04,
        final_wd: float = 0.4,
        total_steps: int = 0,
        schedule: str = "linear",
    ):
        self.initial_wd = float(initial_wd)
        self.final_wd = float(final_wd)
        self.total_steps = max(1, int(total_steps))
        self.schedule = str(schedule).lower()
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1

    def state_dict(self) -> dict:
        return {
            "initial_wd": self.initial_wd,
            "final_wd": self.final_wd,
            "total_steps": self.total_steps,
            "schedule": self.schedule,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict):
        self.initial_wd = float(state["initial_wd"])
        self.final_wd = float(state["final_wd"])
        self.total_steps = int(state["total_steps"])
        self.schedule = str(state["schedule"]).lower()
        self.current_step = int(state["current_step"])

    def get_weight_decay(self) -> float:
        progress = min(1.0, self.current_step / self.total_steps)
        if self.schedule == "linear":
            return self.initial_wd + progress * (self.final_wd - self.initial_wd)
        if self.schedule == "cosine":
            cosine = (1 - math.cos(math.pi * progress)) / 2.0
            return self.initial_wd + cosine * (self.final_wd - self.initial_wd)
        return self.final_wd
