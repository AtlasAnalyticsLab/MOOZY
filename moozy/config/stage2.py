from dataclasses import dataclass

from moozy.config.data import Stage2DataConfig
from moozy.config.model import SlideEncoderConfig
from moozy.config.training import CheckpointConfig, OptimizationConfig


@dataclass(frozen=True)
class Stage2TrainConfig:
    """Full configuration for a stage-2 supervised training run."""

    slide_encoder: SlideEncoderConfig
    data: Stage2DataConfig
    optimization: OptimizationConfig
    checkpoint: CheckpointConfig

    # Task supervision
    task_dir: str = ""
    classification_head_type: str = "mlp"
    survival_head_type: str = "mlp"
    label_smoothing: float = 0.03
    survival_num_bins: int = 8
    survival_min_bins: int = 2
    survival_max_bins: int = 16
    head_dropout: float = 0.1

    # Slide encoder init
    teacher_checkpoint: str = ""
    freeze_slide_encoder: bool = False

    # Case transformer
    case_transformer_variant: str = "base_quarter_depth"
    case_transformer_dropout: float = 0.1
    case_transformer_layerscale_init: float = 1e-5
    case_transformer_layer_drop: float = 0.0
    case_transformer_qk_norm: bool = False
    case_transformer_num_registers: int = 0

    # Training
    epochs: int = 30
    log_every: int = 50
    val_ratio: float = 0.05
    lr_schedule: str = "cosine"
    warmup_steps: int = 0
    activation_checkpointing: bool = False

    # Runtime
    output_dir: str = "outputs/moozy_stage2"
    backend: str = "nccl"
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
            "feature_dirs": self.data.feature_dirs,
            "feature_h5_format": self.data.feature_h5_format,
            "feature_h5_key": self.data.feature_h5_key,
            "batch_size": self.data.batch_size,
            "num_workers": self.data.num_workers,
            "prefetch_factor": self.data.prefetch_factor,
            "lazy_feature_loading": self.data.lazy_feature_loading,
            "max_cached_slides": self.data.max_cached_slides,
            "hflip_prob": self.data.hflip_prob,
            "vflip_prob": self.data.vflip_prob,
            "rotate_prob": self.data.rotate_prob,
            "token_dropout_ratio": self.data.token_dropout_ratio,
            "train_token_cap_sampling": self.data.train_token_cap_sampling,
            "optimizer": self.optimization.optimizer,
            "lr": self.optimization.lr,
            "lr_min": self.optimization.lr_min,
            "lr_base_batch_size": self.optimization.lr_base_batch_size,
            "weight_decay": self.optimization.weight_decay,
            "grad_clip": self.optimization.grad_clip,
            "grad_accumulation_steps": self.optimization.grad_accumulation_steps,
            "mixed_precision": self.optimization.mixed_precision,
            "save_every_epochs": self.checkpoint.save_every_epochs,
            "keep_last_n": self.checkpoint.keep_last_n,
            "task_dir": self.task_dir,
            "classification_head_type": self.classification_head_type,
            "survival_head_type": self.survival_head_type,
            "label_smoothing": self.label_smoothing,
            "survival_num_bins": self.survival_num_bins,
            "survival_min_bins": self.survival_min_bins,
            "survival_max_bins": self.survival_max_bins,
            "head_dropout": self.head_dropout,
            "teacher_checkpoint": self.teacher_checkpoint,
            "freeze_slide_encoder": self.freeze_slide_encoder,
            "case_transformer_variant": self.case_transformer_variant,
            "case_transformer_dropout": self.case_transformer_dropout,
            "case_transformer_layerscale_init": self.case_transformer_layerscale_init,
            "case_transformer_layer_drop": self.case_transformer_layer_drop,
            "case_transformer_qk_norm": self.case_transformer_qk_norm,
            "case_transformer_num_registers": self.case_transformer_num_registers,
            "epochs": self.epochs,
            "log_every": self.log_every,
            "val_ratio": self.val_ratio,
            "lr_schedule": self.lr_schedule,
            "warmup_steps": self.warmup_steps,
            "activation_checkpointing": self.activation_checkpointing,
            "output_dir": self.output_dir,
            "backend": self.backend,
            "seed": self.seed,
            "debug": self.debug,
            "wandb": self.wandb,
            "wandb_project": self.wandb_project,
            "wandb_tags": self.wandb_tags if self.wandb_tags is not None else [],
        }
        return d
