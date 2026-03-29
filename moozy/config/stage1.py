from dataclasses import dataclass

from moozy.config.data import Stage1DataConfig
from moozy.config.model import ProjectionConfig, SlideEncoderConfig
from moozy.config.training import CheckpointConfig, OptimizationConfig, SchedulerConfig


@dataclass(frozen=True)
class Stage1TrainConfig:
    """Full configuration for a stage-1 self-supervised training run."""

    slide_encoder: SlideEncoderConfig
    projection: ProjectionConfig
    data: Stage1DataConfig
    optimization: OptimizationConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig

    # Training duration
    total_steps: int = 0
    epochs: float = 200

    # Logging & validation
    log_every: int = 72
    val_ratio: float = 0.05
    val_every: int = 1000

    # Runtime
    distributed: bool = False
    backend: str = "nccl"
    local_rank: int = 0
    output_dir: str = "./results"
    seed: int = 42
    debug: bool = False

    # wandb
    wandb: bool = False
    wandb_project: str = "moozy"
    wandb_tags: list[str] | None = None

    def to_flat_dict(self) -> dict:
        """Flatten all sub-configs into a single dict for checkpoint serialization."""
        se = self.slide_encoder
        d: dict = {
            "encoder_variant": se.variant,
            "num_registers": se.num_registers,
            "dropout": se.dropout,
            "attn_dropout": se.attn_dropout,
            "layer_drop": se.layer_drop,
            "qk_norm": se.qk_norm,
            "layerscale_init": se.layerscale_init,
            "learnable_alibi": se.learnable_alibi,
            "d_model": se.d_model,
            "n_heads": se.n_heads,
            "n_layers": se.n_layers,
            "dim_feedforward": se.dim_feedforward,
            "output_dim": self.projection.output_dim,
            "proj_hidden_dim": self.projection.proj_hidden_dim,
            "proj_bottleneck_dim": self.projection.proj_bottleneck_dim,
            "proj_norm_last_layer": self.projection.proj_norm_last_layer,
            "proj_norm": self.projection.proj_norm,
            "proj_last_norm": self.projection.proj_last_norm,
            "feature_dirs": self.data.feature_dirs,
            "feature_h5_format": self.data.feature_h5_format,
            "feature_h5_key": self.data.feature_h5_key,
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "prefetch_factor": self.data.prefetch_factor,
            "lazy_feature_loading": self.data.lazy_feature_loading,
            "max_cached_slides": self.data.max_cached_slides,
            "global_crop_size": self.data.global_crop_size,
            "local_crop_size": self.data.local_crop_size,
            "num_global_crops": self.data.num_global_crops,
            "num_local_crops": self.data.num_local_crops,
            "mask_ratio_min": self.data.mask_ratio_min,
            "mask_ratio_max": self.data.mask_ratio_max,
            "min_num_mask_patches": self.data.min_num_mask_patches,
            "max_num_mask_patches": self.data.max_num_mask_patches,
            "mask_min_aspect": self.data.mask_min_aspect,
            "mask_max_aspect": self.data.mask_max_aspect,
            "mask_sample_probability": self.data.mask_sample_probability,
            "min_window_patch_ratio": self.data.min_window_patch_ratio,
            "crop_resample_attempts": self.data.crop_resample_attempts,
            "hflip_prob": self.data.hflip_prob,
            "vflip_prob": self.data.vflip_prob,
            "rotate_prob": self.data.rotate_prob,
            "optimizer": self.optimization.optimizer,
            "lr": self.optimization.lr,
            "lr_min": self.optimization.lr_min,
            "lr_base_batch_size": self.optimization.lr_base_batch_size,
            "weight_decay": self.optimization.weight_decay,
            "grad_clip": self.optimization.grad_clip,
            "grad_accumulation_steps": self.optimization.grad_accumulation_steps,
            "mixed_precision": self.optimization.mixed_precision,
            "lr_schedule": self.scheduler.lr_schedule,
            "warmup_steps": self.scheduler.warmup_steps,
            "warmup_epochs": self.scheduler.warmup_epochs,
            "weight_decay_start": self.scheduler.weight_decay_start,
            "weight_decay_end": self.scheduler.weight_decay_end,
            "wd_schedule": self.scheduler.wd_schedule,
            "ema_momentum_start": self.scheduler.ema_momentum_start,
            "ema_momentum": self.scheduler.ema_momentum,
            "momentum_schedule": self.scheduler.momentum_schedule,
            "student_temp": self.scheduler.student_temp,
            "teacher_temp": self.scheduler.teacher_temp,
            "teacher_patch_temp": self.scheduler.teacher_patch_temp,
            "warmup_teacher_temp": self.scheduler.warmup_teacher_temp,
            "warmup_teacher_patch_temp": self.scheduler.warmup_teacher_patch_temp,
            "warmup_teacher_temp_epochs": self.scheduler.warmup_teacher_temp_epochs,
            "center_momentum": self.scheduler.center_momentum,
            "freeze_last_layer_steps": self.scheduler.freeze_last_layer_steps,
            "freeze_last_layer_epochs": self.scheduler.freeze_last_layer_epochs,
            "save_every": self.checkpoint.save_every,
            "resume_from": self.checkpoint.resume_from,
            "keep_last_n": self.checkpoint.keep_last_n,
            "save_teacher": self.checkpoint.save_teacher,
            "teacher_save_prefix": self.checkpoint.teacher_save_prefix,
            "total_steps": self.total_steps,
            "epochs": self.epochs,
            "log_every": self.log_every,
            "val_ratio": self.val_ratio,
            "val_every": self.val_every,
            "distributed": self.distributed,
            "backend": self.backend,
            "local_rank": self.local_rank,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "debug": self.debug,
            "wandb": self.wandb,
            "wandb_project": self.wandb_project,
            "wandb_tags": self.wandb_tags if self.wandb_tags is not None else [],
        }
        return d
