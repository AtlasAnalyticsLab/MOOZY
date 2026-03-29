from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and gradient parameters."""

    optimizer: str = "adamw"
    lr: float = 5e-4
    lr_min: float = 2e-6
    lr_base_batch_size: int = 256
    weight_decay: float = 0.04
    grad_clip: float = 0.3
    grad_accumulation_steps: int = 1
    mixed_precision: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning rate, weight-decay, EMA, and temperature schedule parameters."""

    # LR schedule
    lr_schedule: str = "cosine"
    warmup_steps: int = 0
    warmup_epochs: float = 5.0

    # Weight-decay schedule (stage-1 only)
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    wd_schedule: str = "cosine"

    # EMA & temperature schedule (stage-1 only)
    ema_momentum_start: float = 0.996
    ema_momentum: float = 1.0
    momentum_schedule: str = "cosine"
    student_temp: float = 0.1
    teacher_temp: float = 0.07
    teacher_patch_temp: float = 0.07
    warmup_teacher_temp: float = 0.04
    warmup_teacher_patch_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 30
    center_momentum: float = 0.9

    # Freeze last projection layer (stage-1 only)
    freeze_last_layer_steps: int | None = None
    freeze_last_layer_epochs: float = 3.0


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint saving and resumption parameters."""

    # Stage-1 fields
    save_every: int = 72
    resume_from: str | None = None
    keep_last_n: int = 50
    save_teacher: bool = True
    teacher_save_prefix: str = "teacher_step"

    # Stage-2 fields
    save_every_epochs: int = 1
