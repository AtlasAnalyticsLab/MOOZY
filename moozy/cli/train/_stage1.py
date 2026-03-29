from typing import Annotated

import typer

from .._types import Backend, H5Format, NormType, OptimizerChoice, Schedule, WDSchedule, enum_val


def stage1_command(
    feature_dirs: Annotated[
        list[str],
        typer.Option("--feature_dirs", help="Directories containing h5 feature files."),
    ],
    feature_h5_format: Annotated[
        H5Format,
        typer.Option("--feature_h5_format", help="Feature file H5 schema to load."),
    ] = H5Format.auto,
    feature_h5_key: Annotated[
        str,
        typer.Option("--feature_h5_key", help="Optional AtlasPatch feature key under features/."),
    ] = "",
    batch_size: Annotated[int, typer.Option("--batch_size", help="Batch size for training.")] = 64,
    num_workers: Annotated[int, typer.Option("--num_workers", help="Number of data-loading workers.")] = 4,
    prefetch_factor: Annotated[int, typer.Option("--prefetch_factor", help="Batches to prefetch per worker.")] = 4,
    lazy_feature_loading: Annotated[
        bool,
        typer.Option("--lazy_feature_loading/--no_lazy_feature_loading", help="Load feature grids on demand."),
    ] = False,
    max_cached_slides: Annotated[
        int,
        typer.Option("--max_cached_slides", help="Max lazily-decoded slides to cache per process (0 disables)."),
    ] = 0,
    global_crop_size: Annotated[int, typer.Option("--global_crop_size", help="Size of global crops.")] = 20,
    local_crop_size: Annotated[int, typer.Option("--local_crop_size", help="Size of local crops.")] = 12,
    num_global_crops: Annotated[int, typer.Option("--num_global_crops", help="Global crops per sample.")] = 2,
    num_local_crops: Annotated[int, typer.Option("--num_local_crops", help="Local crops per sample.")] = 4,
    mask_ratio_min: Annotated[
        float, typer.Option("--mask_ratio_min", help="Minimum mask ratio per masked global crop.")
    ] = 0.1,
    mask_ratio_max: Annotated[
        float, typer.Option("--mask_ratio_max", help="Maximum mask ratio per masked global crop.")
    ] = 0.5,
    min_num_mask_patches: Annotated[
        int, typer.Option("--min_num_mask_patches", help="Min patches per masking block.")
    ] = 4,
    max_num_mask_patches: Annotated[
        int, typer.Option("--max_num_mask_patches", help="Max patches per masking block (-1 disables cap).")
    ] = -1,
    mask_min_aspect: Annotated[
        float, typer.Option("--mask_min_aspect", help="Min aspect ratio for block-masking rectangles.")
    ] = 0.3,
    mask_max_aspect: Annotated[
        float | None,
        typer.Option("--mask_max_aspect", help="Max aspect ratio (defaults to 1/mask_min_aspect)."),
    ] = None,
    mask_sample_probability: Annotated[
        float,
        typer.Option("--mask_sample_probability", help="Probability of masking each global crop."),
    ] = 0.5,
    min_window_patch_ratio: Annotated[
        float,
        typer.Option("--min_window_patch_ratio", help="Min fraction of non-zero patches per crop."),
    ] = 0.25,
    crop_resample_attempts: Annotated[
        int, typer.Option("--crop_resample_attempts", help="Attempts to resample a sparse crop.")
    ] = 3,
    hflip_prob: Annotated[float, typer.Option("--hflip_prob", help="Horizontal flip probability.")] = 0.5,
    vflip_prob: Annotated[float, typer.Option("--vflip_prob", help="Vertical flip probability.")] = 0.5,
    rotate_prob: Annotated[float, typer.Option("--rotate_prob", help="Probability of 90/180/270° rotation.")] = 0.5,
    encoder_variant: Annotated[
        str,
        typer.Option("--encoder_variant", help="Encoder variant preset (e.g. base_half_depth, tiny, large)."),
    ] = "base_half_depth",
    output_dim: Annotated[int, typer.Option("--output_dim", help="Output dimension of the projection head.")] = 8192,
    proj_hidden_dim: Annotated[
        int, typer.Option("--proj_hidden_dim", help="Hidden dimension of the projection MLP.")
    ] = 2048,
    proj_bottleneck_dim: Annotated[
        int, typer.Option("--proj_bottleneck_dim", help="Bottleneck dimension before final weight-normalised layer.")
    ] = 256,
    proj_norm_last_layer: Annotated[
        bool,
        typer.Option(
            "--proj_norm_last_layer/--no_proj_norm_last_layer",
            help="Freeze weight_norm gain in the projection head's last layer.",
        ),
    ] = True,
    proj_norm: Annotated[
        NormType,
        typer.Option("--proj_norm", help="Normalisation inside projection MLP after fc1/fc2."),
    ] = NormType.none,
    proj_last_norm: Annotated[
        NormType,
        typer.Option("--proj_last_norm", help="Optional normalisation after the final projection layer."),
    ] = NormType.none,
    num_registers: Annotated[
        int, typer.Option("--num_registers", help="Register tokens prepended after CLS (0 disables).")
    ] = 4,
    layer_drop: Annotated[float, typer.Option("--layer_drop", help="Stochastic depth (layer drop) rate.")] = 0.1,
    dropout: Annotated[float, typer.Option("--dropout", help="MLP dropout rate inside transformer blocks.")] = 0.1,
    attn_dropout: Annotated[float, typer.Option("--attn_dropout", help="Attention dropout probability.")] = 0.0,
    qk_norm: Annotated[
        bool, typer.Option("--qk_norm/--no_qk_norm", help="Per-head LayerNorm on q/k projections.")
    ] = False,
    layerscale_init: Annotated[
        float, typer.Option("--layerscale_init", help="Initial LayerScale gamma (<=0 disables).")
    ] = 0.0,
    learnable_alibi: Annotated[
        bool,
        typer.Option("--learnable_alibi/--no_learnable_alibi", help="Make ALiBi slopes trainable."),
    ] = False,
    ema_momentum_start: Annotated[
        float, typer.Option("--ema_momentum_start", help="Initial EMA momentum for teacher update.")
    ] = 0.996,
    ema_momentum: Annotated[float, typer.Option("--ema_momentum", help="Final EMA momentum for teacher update.")] = 1.0,
    momentum_schedule: Annotated[
        Schedule, typer.Option("--momentum_schedule", help="Teacher EMA momentum schedule.")
    ] = Schedule.cosine,
    student_temp: Annotated[float, typer.Option("--student_temp", help="Student temperature.")] = 0.1,
    teacher_temp: Annotated[float, typer.Option("--teacher_temp", help="Final teacher CLS temperature.")] = 0.07,
    teacher_patch_temp: Annotated[
        float, typer.Option("--teacher_patch_temp", help="Final teacher patch temperature.")
    ] = 0.07,
    warmup_teacher_temp: Annotated[
        float, typer.Option("--warmup_teacher_temp", help="Starting teacher CLS temperature during warmup.")
    ] = 0.04,
    warmup_teacher_patch_temp: Annotated[
        float, typer.Option("--warmup_teacher_patch_temp", help="Starting teacher patch temperature during warmup.")
    ] = 0.04,
    warmup_teacher_temp_epochs: Annotated[
        int, typer.Option("--warmup_teacher_temp_epochs", help="Epochs to warm up teacher temperatures (0 disables).")
    ] = 30,
    center_momentum: Annotated[float, typer.Option("--center_momentum", help="Momentum for center updates.")] = 0.9,
    total_steps: Annotated[
        int,
        typer.Option("--total_steps", help="Total training steps when --epochs is 0 or negative."),
    ] = 0,
    epochs: Annotated[
        float,
        typer.Option("--epochs", help="Number of epochs. Positive values override --total_steps. Supports fractional."),
    ] = 200,
    lr: Annotated[float, typer.Option("--lr", help="Base learning rate.")] = 5e-4,
    lr_min: Annotated[float, typer.Option("--lr_min", help="Minimum learning rate at end of decay.")] = 2e-6,
    lr_schedule: Annotated[
        Schedule, typer.Option("--lr_schedule", help="LR decay schedule after warmup.")
    ] = Schedule.cosine,
    lr_base_batch_size: Annotated[
        int, typer.Option("--lr_base_batch_size", help="Reference global batch size for LR scaling.")
    ] = 256,
    weight_decay: Annotated[float, typer.Option("--weight_decay", help="Global weight decay.")] = 0.04,
    weight_decay_start: Annotated[float, typer.Option("--weight_decay_start", help="Starting weight decay.")] = 0.04,
    weight_decay_end: Annotated[float, typer.Option("--weight_decay_end", help="Final weight decay.")] = 0.4,
    wd_schedule: Annotated[
        WDSchedule, typer.Option("--wd_schedule", help="Weight-decay schedule type.")
    ] = WDSchedule.cosine,
    warmup_steps: Annotated[int, typer.Option("--warmup_steps", help="Linear warmup steps for LR.")] = 0,
    warmup_epochs: Annotated[
        float,
        typer.Option("--warmup_epochs", help="LR warmup epochs when --warmup_steps is 0."),
    ] = 5,
    grad_clip: Annotated[float, typer.Option("--grad_clip", help="Gradient clipping max norm (0 disables).")] = 0.3,
    grad_accumulation_steps: Annotated[
        int, typer.Option("--grad_accumulation_steps", help="Gradient accumulation steps.")
    ] = 1,
    freeze_last_layer_steps: Annotated[
        int | None,
        typer.Option("--freeze_last_layer_steps", help="Freeze last projection layer for N steps."),
    ] = None,
    freeze_last_layer_epochs: Annotated[
        float,
        typer.Option("--freeze_last_layer_epochs", help="Freeze last projection layer for N epochs."),
    ] = 3.0,
    optimizer: Annotated[
        OptimizerChoice, typer.Option("--optimizer", help="Optimiser algorithm.")
    ] = OptimizerChoice.adamw,
    log_every: Annotated[int, typer.Option("--log_every", help="Log training loss every N steps.")] = 72,
    val_ratio: Annotated[float, typer.Option("--val_ratio", help="Validation split ratio.")] = 0.05,
    val_every: Annotated[int, typer.Option("--val_every", help="Validate every N steps.")] = 1000,
    mixed_precision: Annotated[
        bool,
        typer.Option("--mixed_precision/--no_mixed_precision", help="Enable mixed-precision training (bf16)."),
    ] = False,
    save_every: Annotated[int, typer.Option("--save_every", help="Save checkpoint every N steps (0 disables).")] = 72,
    resume_from: Annotated[str | None, typer.Option("--resume_from", help="Path to checkpoint to resume from.")] = None,
    keep_last_n: Annotated[int, typer.Option("--keep_last_n", help="Keep only last N checkpoints.")] = 50,
    save_teacher: Annotated[
        bool,
        typer.Option("--save_teacher/--no_save_teacher", help="Save teacher-only checkpoint at each save."),
    ] = True,
    teacher_save_prefix: Annotated[
        str, typer.Option("--teacher_save_prefix", help="Filename prefix for teacher-only checkpoints.")
    ] = "teacher_step",
    distributed: Annotated[
        bool, typer.Option("--distributed/--no_distributed", help="Enable distributed training.")
    ] = False,
    backend: Annotated[Backend, typer.Option("--backend", help="Distributed backend.")] = Backend.nccl,
    local_rank: Annotated[int, typer.Option("--local_rank", help="Local rank for distributed training.")] = 0,
    output_dir: Annotated[
        str, typer.Option("--output_dir", help="Output directory for logs and checkpoints.")
    ] = "./results",
    wandb: Annotated[bool, typer.Option("--wandb/--no_wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb_project", help="Weights & Biases project name.")] = "moozy",
    wandb_tags: Annotated[
        str,
        typer.Option("--wandb_tags", help="Space or comma-separated tags for Weights & Biases run."),
    ] = "",
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    debug: Annotated[bool, typer.Option("--debug/--no_debug", help="Debug mode with limited data.")] = False,
) -> None:
    """Train the Stage-1 MOOZY self-supervised pretraining model."""
    from moozy.config.data import Stage1DataConfig
    from moozy.config.model import ProjectionConfig, SlideEncoderConfig
    from moozy.config.stage1 import Stage1TrainConfig
    from moozy.config.training import CheckpointConfig, OptimizationConfig, SchedulerConfig

    parsed_wandb_tags = [t for raw in wandb_tags.replace(",", " ").split() if (t := raw.strip())] if wandb_tags else []

    config = Stage1TrainConfig(
        slide_encoder=SlideEncoderConfig(
            variant=encoder_variant,
            num_registers=num_registers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            layer_drop=layer_drop,
            qk_norm=qk_norm,
            layerscale_init=layerscale_init,
            learnable_alibi=learnable_alibi,
        ).resolve_variant(),
        projection=ProjectionConfig(
            output_dim=output_dim,
            proj_hidden_dim=proj_hidden_dim,
            proj_bottleneck_dim=proj_bottleneck_dim,
            proj_norm_last_layer=proj_norm_last_layer,
            proj_norm=enum_val(proj_norm),
            proj_last_norm=enum_val(proj_last_norm),
        ),
        data=Stage1DataConfig(
            feature_dirs=list(feature_dirs),
            feature_h5_format=enum_val(feature_h5_format),
            feature_h5_key=feature_h5_key,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            lazy_feature_loading=lazy_feature_loading,
            max_cached_slides=max_cached_slides,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            min_num_mask_patches=min_num_mask_patches,
            max_num_mask_patches=max_num_mask_patches,
            mask_min_aspect=mask_min_aspect,
            mask_max_aspect=mask_max_aspect,
            mask_sample_probability=mask_sample_probability,
            min_window_patch_ratio=min_window_patch_ratio,
            crop_resample_attempts=crop_resample_attempts,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            rotate_prob=rotate_prob,
        ),
        optimization=OptimizationConfig(
            optimizer=enum_val(optimizer),
            lr=lr,
            lr_min=lr_min,
            lr_base_batch_size=lr_base_batch_size,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            grad_accumulation_steps=grad_accumulation_steps,
            mixed_precision=mixed_precision,
        ),
        scheduler=SchedulerConfig(
            lr_schedule=enum_val(lr_schedule),
            warmup_steps=warmup_steps,
            warmup_epochs=warmup_epochs,
            weight_decay_start=weight_decay_start,
            weight_decay_end=weight_decay_end,
            wd_schedule=enum_val(wd_schedule),
            ema_momentum_start=ema_momentum_start,
            ema_momentum=ema_momentum,
            momentum_schedule=enum_val(momentum_schedule),
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            teacher_patch_temp=teacher_patch_temp,
            warmup_teacher_temp=warmup_teacher_temp,
            warmup_teacher_patch_temp=warmup_teacher_patch_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            center_momentum=center_momentum,
            freeze_last_layer_steps=freeze_last_layer_steps,
            freeze_last_layer_epochs=freeze_last_layer_epochs,
        ),
        checkpoint=CheckpointConfig(
            save_every=save_every,
            resume_from=resume_from,
            keep_last_n=keep_last_n,
            save_teacher=save_teacher,
            teacher_save_prefix=teacher_save_prefix,
        ),
        total_steps=total_steps,
        epochs=epochs,
        log_every=log_every,
        val_ratio=val_ratio,
        val_every=val_every,
        distributed=distributed,
        backend=enum_val(backend),
        local_rank=local_rank,
        output_dir=output_dir,
        seed=seed,
        debug=debug,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_tags=parsed_wandb_tags,
    )

    from moozy.training.runners.stage1 import run_stage1

    run_stage1(config)
