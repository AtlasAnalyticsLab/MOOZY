from typing import Annotated

import typer

from .._types import Backend, H5Format, HeadType, OptimizerChoice, Schedule, TokenCapSampling, enum_val


def stage2_command(
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
    batch_size: Annotated[int, typer.Option("--batch_size", help="Batch size for supervised fine-tuning.")] = 1,
    num_workers: Annotated[int, typer.Option("--num_workers", help="Number of data-loading workers.")] = 4,
    prefetch_factor: Annotated[int, typer.Option("--prefetch_factor", help="Batches to prefetch per worker.")] = 2,
    lazy_feature_loading: Annotated[
        bool,
        typer.Option("--lazy_feature_loading/--no_lazy_feature_loading", help="Load slide feature grids on demand."),
    ] = False,
    max_cached_slides: Annotated[
        int,
        typer.Option("--max_cached_slides", help="Max lazily-loaded slides to keep in LRU cache (0 disables)."),
    ] = 0,
    hflip_prob: Annotated[float, typer.Option("--hflip_prob", help="Horizontal flip probability.")] = 0.5,
    vflip_prob: Annotated[float, typer.Option("--vflip_prob", help="Vertical flip probability.")] = 0.5,
    rotate_prob: Annotated[float, typer.Option("--rotate_prob", help="Probability of 90/180/270° rotation.")] = 0.5,
    token_dropout_ratio: Annotated[
        float,
        typer.Option("--token_dropout_ratio", help="Fraction of valid patches to drop per slide during training."),
    ] = 0.1,
    train_token_cap_sampling: Annotated[
        TokenCapSampling,
        typer.Option(
            "--train_token_cap_sampling",
            help="Sampling strategy when training slide exceeds token cap.",
        ),
    ] = TokenCapSampling.random_stratified,
    task_dir: Annotated[
        str,
        typer.Option("--task_dir", help="Directory containing task subdirectories with task.csv and config.yaml."),
    ] = "",  # sentinel — replaced by DEFAULT_TASK_DIR below
    classification_head_type: Annotated[
        HeadType,
        typer.Option("--classification_head_type", help="Head type for classification tasks."),
    ] = HeadType.mlp,
    survival_head_type: Annotated[
        HeadType,
        typer.Option("--survival_head_type", help="Head type for survival tasks."),
    ] = HeadType.mlp,
    label_smoothing: Annotated[
        float,
        typer.Option("--label_smoothing", help="Label smoothing for supervised task loss."),
    ] = 0.03,
    survival_num_bins: Annotated[
        int, typer.Option("--survival_num_bins", help="Target discrete time bins per survival task.")
    ] = 8,
    survival_min_bins: Annotated[int, typer.Option("--survival_min_bins", help="Minimum bins per survival task.")] = 2,
    survival_max_bins: Annotated[int, typer.Option("--survival_max_bins", help="Maximum bins per survival task.")] = 16,
    teacher_checkpoint: Annotated[
        str,
        typer.Option(
            "--teacher_checkpoint",
            help="Path to exported Stage-1 encoder checkpoint (empty = train from scratch).",
        ),
    ] = "",
    encoder_variant: Annotated[
        str,
        typer.Option(
            "--encoder_variant",
            help="Slide encoder variant preset (used when --teacher_checkpoint is empty).",
        ),
    ] = "base_half_depth",
    num_registers: Annotated[
        int,
        typer.Option("--num_registers", help="Register tokens in slide encoder (0 disables)."),
    ] = 4,
    dropout: Annotated[float, typer.Option("--dropout", help="MLP dropout inside transformer blocks.")] = 0.1,
    attn_dropout: Annotated[float, typer.Option("--attn_dropout", help="Attention dropout probability.")] = 0.0,
    layer_drop: Annotated[float, typer.Option("--layer_drop", help="Stochastic depth rate.")] = 0.1,
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
    head_dropout: Annotated[
        float,
        typer.Option("--head_dropout", help="Dropout before task heads."),
    ] = 0.1,
    case_transformer_variant: Annotated[
        str,
        typer.Option("--case_transformer_variant", help="Case-transformer architecture preset."),
    ] = "base_quarter_depth",
    case_transformer_dropout: Annotated[
        float,
        typer.Option("--case_transformer_dropout", help="Dropout inside the case transformer."),
    ] = 0.1,
    case_transformer_layerscale_init: Annotated[
        float,
        typer.Option("--case_transformer_layerscale_init", help="LayerScale init for case transformer."),
    ] = 1e-5,
    case_transformer_layer_drop: Annotated[
        float,
        typer.Option("--case_transformer_layer_drop", help="Stochastic depth rate for case transformer."),
    ] = 0.0,
    case_transformer_qk_norm: Annotated[
        bool,
        typer.Option(
            "--case_transformer_qk_norm/--no_case_transformer_qk_norm",
            help="Per-head LayerNorm on Q/K in case transformer.",
        ),
    ] = False,
    case_transformer_num_registers: Annotated[
        int,
        typer.Option("--case_transformer_num_registers", help="Register tokens for case transformer."),
    ] = 0,
    freeze_slide_encoder: Annotated[
        bool,
        typer.Option("--freeze_slide_encoder/--no_freeze_slide_encoder", help="Freeze slide encoder weights."),
    ] = False,
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs.")] = 30,
    val_ratio: Annotated[float, typer.Option("--val_ratio", help="Validation split ratio.")] = 0.05,
    optimizer: Annotated[
        OptimizerChoice, typer.Option("--optimizer", help="Optimiser algorithm.")
    ] = OptimizerChoice.adamw,
    lr: Annotated[float, typer.Option("--lr", help="Base learning rate.")] = 5e-5,
    lr_min: Annotated[float, typer.Option("--lr_min", help="Minimum learning rate.")] = 2e-7,
    lr_schedule: Annotated[Schedule, typer.Option("--lr_schedule", help="Learning rate schedule.")] = Schedule.cosine,
    warmup_steps: Annotated[int, typer.Option("--warmup_steps", help="LR warmup steps.")] = 0,
    lr_base_batch_size: Annotated[
        int,
        typer.Option("--lr_base_batch_size", help="Reference batch size for linear LR scaling."),
    ] = 256,
    grad_accumulation_steps: Annotated[
        int, typer.Option("--grad_accumulation_steps", help="Gradient accumulation steps.")
    ] = 1,
    weight_decay: Annotated[float, typer.Option("--weight_decay", help="Weight decay.")] = 0.4,
    grad_clip: Annotated[
        float, typer.Option("--grad_clip", help="Per-parameter gradient clipping norm (0 disables).")
    ] = 0.3,
    log_every: Annotated[int, typer.Option("--log_every", help="Log metrics every N optimizer steps.")] = 50,
    save_every_epochs: Annotated[
        int, typer.Option("--save_every_epochs", help="Save checkpoint every N epochs (0 disables).")
    ] = 1,
    keep_last_n: Annotated[int, typer.Option("--keep_last_n", help="Keep only last N checkpoints.")] = 50,
    mixed_precision: Annotated[
        bool,
        typer.Option("--mixed_precision/--no_mixed_precision", help="Enable mixed-precision training (bf16)."),
    ] = False,
    activation_checkpointing: Annotated[
        bool,
        typer.Option(
            "--activation_checkpointing/--no_activation_checkpointing",
            help="Enable activation checkpointing to reduce GPU memory.",
        ),
    ] = False,
    output_dir: Annotated[
        str, typer.Option("--output_dir", help="Output directory for logs and checkpoints.")
    ] = "outputs/moozy_stage2",
    wandb: Annotated[bool, typer.Option("--wandb/--no_wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb_project", help="Weights & Biases project name.")] = "moozy",
    wandb_tags: Annotated[
        str,
        typer.Option("--wandb_tags", help="Space or comma-separated tags for Weights & Biases run."),
    ] = "",
    backend: Annotated[Backend, typer.Option("--backend", help="Distributed backend.")] = Backend.nccl,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    debug: Annotated[bool, typer.Option("--debug/--no_debug", help="Debug mode with limited data.")] = False,
) -> None:
    """Train the Stage-2 MOOZY supervised alignment model."""
    from moozy.config.data import Stage2DataConfig
    from moozy.config.model import SlideEncoderConfig
    from moozy.config.stage2 import Stage2TrainConfig
    from moozy.config.training import CheckpointConfig, OptimizationConfig

    parsed_wandb_tags = [t for raw in wandb_tags.replace(",", " ").split() if (t := raw.strip())] if wandb_tags else []

    resolved_task_dir = task_dir
    if not resolved_task_dir:
        from moozy.hf_hub import ensure_tasks_dir

        resolved_task_dir = ensure_tasks_dir()

    clamped_min_bins = max(1, int(survival_min_bins))
    clamped_max_bins = max(clamped_min_bins, int(survival_max_bins))
    clamped_num_bins = min(clamped_max_bins, max(clamped_min_bins, int(survival_num_bins)))

    config = Stage2TrainConfig(
        slide_encoder=SlideEncoderConfig(
            variant=encoder_variant,
            num_registers=num_registers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            layer_drop=layer_drop,
            qk_norm=qk_norm,
            layerscale_init=layerscale_init,
            learnable_alibi=learnable_alibi,
        ),
        data=Stage2DataConfig(
            feature_dirs=list(feature_dirs),
            feature_h5_format=enum_val(feature_h5_format),
            feature_h5_key=feature_h5_key,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            lazy_feature_loading=lazy_feature_loading,
            max_cached_slides=max_cached_slides,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            rotate_prob=rotate_prob,
            token_dropout_ratio=token_dropout_ratio,
            train_token_cap_sampling=enum_val(train_token_cap_sampling),
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
        checkpoint=CheckpointConfig(
            save_every_epochs=save_every_epochs,
            keep_last_n=keep_last_n,
        ),
        task_dir=resolved_task_dir,
        classification_head_type=enum_val(classification_head_type),
        survival_head_type=enum_val(survival_head_type),
        label_smoothing=label_smoothing,
        survival_num_bins=clamped_num_bins,
        survival_min_bins=clamped_min_bins,
        survival_max_bins=clamped_max_bins,
        head_dropout=head_dropout,
        teacher_checkpoint=teacher_checkpoint,
        freeze_slide_encoder=freeze_slide_encoder,
        case_transformer_variant=case_transformer_variant,
        case_transformer_dropout=case_transformer_dropout,
        case_transformer_layerscale_init=case_transformer_layerscale_init,
        case_transformer_layer_drop=case_transformer_layer_drop,
        case_transformer_qk_norm=case_transformer_qk_norm,
        case_transformer_num_registers=case_transformer_num_registers,
        epochs=epochs,
        log_every=log_every,
        val_ratio=val_ratio,
        lr_schedule=enum_val(lr_schedule),
        warmup_steps=warmup_steps,
        activation_checkpointing=activation_checkpointing,
        output_dir=output_dir,
        backend=enum_val(backend),
        seed=seed,
        debug=debug,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_tags=parsed_wandb_tags,
    )

    from moozy.training.runners.stage2 import run_stage2

    run_stage2(config)
