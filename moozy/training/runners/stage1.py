import math
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from moozy.config.stage1 import Stage1TrainConfig
from moozy.data.features import detect_feat_dim, list_feature_paths
from moozy.data.stage1 import MOOZYDataset, build_stage1_dataloader
from moozy.models.factory import build_ssl_model
from moozy.models.serialization import load_stage1_training_checkpoint
from moozy.training import (
    MomentumScheduler,
    TemperatureScheduler,
    WeightDecayScheduler,
    build_optimizer,
    build_warmup_scheduler,
    cleanup_distributed,
    count_parameters,
    finish_wandb,
    initialize_training_runtime,
    initialize_wandb,
    is_main_process,
    log_parameter_breakdown,
    save_hyperparameters,
    seed_worker,
    set_fixed_weight_decay,
)
from moozy.training.engine.stage1 import Stage1Engine


def _reduce_size(local_size: int, device: torch.device) -> int:
    size_tensor = torch.tensor(int(local_size), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)
    return int(size_tensor.item())


def run_stage1(config: Stage1TrainConfig) -> None:
    """Run stage-1 self-supervised training from a typed config."""
    args_dict = config.to_flat_dict()

    runtime = initialize_training_runtime(
        backend=config.backend,
        output_dir=config.output_dir,
        seed=config.seed,
        logger_name="moozy.stage_1_ssl",
        log_seed=True,
    )
    rank = runtime.rank
    local_rank = runtime.local_rank
    world_size = runtime.world_size
    checkpoint_dir = runtime.checkpoint_dir
    logger = runtime.logger
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    max_num_mask_patches = (
        None
        if config.data.max_num_mask_patches is None or config.data.max_num_mask_patches < 0
        else config.data.max_num_mask_patches
    )

    if is_main_process(rank):
        logger.info(f"Loading dataset from {config.data.feature_dirs}")

    dataset = MOOZYDataset(
        feature_dirs=config.data.feature_dirs,
        feature_h5_format=config.data.feature_h5_format,
        feature_h5_key=config.data.feature_h5_key,
        global_crop_size=config.data.global_crop_size,
        local_crop_size=config.data.local_crop_size,
        num_global_crops=config.data.num_global_crops,
        num_local_crops=config.data.num_local_crops,
        min_window_patch_ratio=config.data.min_window_patch_ratio,
        crop_resample_attempts=config.data.crop_resample_attempts,
        hflip_prob=config.data.hflip_prob,
        vflip_prob=config.data.vflip_prob,
        rotate_prob=config.data.rotate_prob,
        debug=config.debug,
        preload_features=not config.data.lazy_feature_loading,
        max_cached_slides=config.data.max_cached_slides,
        rank=rank,
        world_size=world_size,
    )

    local_dataset_size = len(dataset)
    global_dataset_size = _reduce_size(local_dataset_size, device)
    if is_main_process(rank):
        logger.info(f"Dataset size (local): {local_dataset_size}")
        if world_size > 1:
            logger.info(f"Global dataset size across shards: {global_dataset_size}")
        else:
            logger.info(f"Global dataset size: {global_dataset_size}")

    train_dataset: Any = dataset
    global_train_size = global_dataset_size
    val_loader = None

    if config.val_ratio and config.val_ratio > 0.0:
        from torch.utils.data import random_split

        total_len = len(dataset)
        val_len = max(1, int(total_len * config.val_ratio))
        train_len = max(0, total_len - val_len)
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
        global_train_size = _reduce_size(train_len, device)
        global_val_size = _reduce_size(val_len, device)
        if is_main_process(rank):
            logger.info(f"Train split (global): {global_train_size} - Val split (global): {global_val_size}")

    loader_kwargs: dict[str, Any] = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        worker_init_fn=seed_worker,
        mask_ratio_min=config.data.mask_ratio_min,
        mask_ratio_max=config.data.mask_ratio_max,
        min_num_mask_patches=config.data.min_num_mask_patches,
        max_num_mask_patches=max_num_mask_patches,
        mask_min_aspect=config.data.mask_min_aspect,
        mask_max_aspect=config.data.mask_max_aspect,
        mask_sample_probability=config.data.mask_sample_probability,
    )
    train_loader = build_stage1_dataloader(
        dataset=train_dataset, shuffle=True, drop_last=True, sampler=None, **loader_kwargs
    )
    if config.val_ratio and config.val_ratio > 0.0:
        val_loader = build_stage1_dataloader(
            dataset=val_dataset,
            shuffle=False,
            sampler=None,
            **loader_kwargs,
        )

    if is_main_process(rank):
        logger.info(
            f"Dataloader created with {len(train_loader)} batches "
            f"(workers={config.data.num_workers}, prefetch_factor={config.data.prefetch_factor})"
        )

    h5_paths = list_feature_paths(config.data.feature_dirs)
    if not h5_paths:
        raise ValueError(f"No .h5 feature files found in {config.data.feature_dirs}")
    feat_dim = detect_feat_dim(
        h5_paths[0],
        feature_h5_format=config.data.feature_h5_format,
        feature_h5_key=config.data.feature_h5_key,
    )
    args_dict["feat_dim"] = feat_dim

    if is_main_process(rank):
        logger.info("Detected feat_dim=%d from %s", feat_dim, h5_paths[0])
        logger.info("Initializing the Stage-1 MOOZY model...")

    model = build_ssl_model(config.slide_encoder, config.projection, config.scheduler, feat_dim=feat_dim)
    model = model.to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    raw = model.module if isinstance(model, DDP) else model
    if is_main_process(rank):
        rows = [
            (
                "Student Slide Encoder",
                count_parameters(raw.student_slide_encoder),
                count_parameters(raw.student_slide_encoder, trainable_only=True),
            ),
            (
                "Student Head",
                count_parameters(raw.student_head),
                count_parameters(raw.student_head, trainable_only=True),
            ),
            ("Teacher Slide Encoder", count_parameters(raw.teacher_slide_encoder), None),
            ("Teacher Head", count_parameters(raw.teacher_head), None),
        ]
        log_parameter_breakdown(logger, rows)

    accum_steps = max(1, config.optimization.grad_accumulation_steps)
    global_batch = config.data.batch_size * max(1, world_size)
    batches_per_epoch = max(1, math.ceil(global_train_size / float(global_batch)))
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / accum_steps))

    total_steps = config.total_steps
    if config.epochs is not None and float(config.epochs) > 0:
        total_steps = math.ceil(float(config.epochs) * steps_per_epoch)
        if is_main_process(rank):
            logger.info(f"Using epochs={config.epochs} -> {steps_per_epoch} steps/epoch, total_steps={total_steps}")
    elif total_steps <= 0:
        raise ValueError("Stage 1 requires either positive --epochs or positive --total_steps.")
    args_dict["total_steps"] = total_steps

    total_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    if is_main_process(rank):
        logger.info(
            f"Schedule: steps_per_epoch={steps_per_epoch}, total_epochs~{total_epochs} "
            f"(global_train_size={global_train_size}, global_batch={global_batch}, accum={accum_steps})"
        )

    effective_global_batch = global_batch * accum_steps
    scaled_lr = config.optimization.lr * (effective_global_batch / float(config.optimization.lr_base_batch_size))
    if is_main_process(rank):
        logger.info(
            f"Using scaled learning rate {scaled_lr:.6e} for effective global batch {effective_global_batch} "
            f"(micro batch {config.data.batch_size}, world size {world_size}, accum {accum_steps}; "
            f"base {config.optimization.lr} @ {config.optimization.lr_base_batch_size})"
        )

    optimizer = build_optimizer(
        model, config.optimization.optimizer, lr=scaled_lr, weight_decay=config.optimization.weight_decay
    )

    if is_main_process(rank):
        hp_path = save_hyperparameters(args_dict, config.output_dir)
        logger.info("Saved hyperparameters to %s", hp_path)

    if config.scheduler.warmup_steps > 0:
        lr_warmup_steps = config.scheduler.warmup_steps
    elif config.scheduler.warmup_epochs > 0:
        lr_warmup_steps = max(1, round(config.scheduler.warmup_epochs * steps_per_epoch))
    else:
        lr_warmup_steps = 0
    lr_warmup_steps = min(lr_warmup_steps, total_steps)

    lr_scheduler = build_warmup_scheduler(
        optimizer,
        warmup_steps=lr_warmup_steps,
        total_steps=total_steps,
        min_lr=config.optimization.lr_min,
        schedule=config.scheduler.lr_schedule,
    )
    momentum_scheduler = MomentumScheduler(
        initial_momentum=config.scheduler.ema_momentum_start,
        final_momentum=config.scheduler.ema_momentum,
        total_steps=total_steps,
        schedule=config.scheduler.momentum_schedule,
    )

    wd_scheduler = None
    if config.scheduler.wd_schedule != "none":
        wd_scheduler = WeightDecayScheduler(
            initial_wd=config.scheduler.weight_decay_start,
            final_wd=config.scheduler.weight_decay_end,
            total_steps=total_steps,
            schedule=config.scheduler.wd_schedule,
        )
        set_fixed_weight_decay(optimizer, config.scheduler.weight_decay_start)
        if is_main_process(rank):
            logger.info(
                f"Weight decay schedule enabled: {config.scheduler.wd_schedule}, "
                f"start={config.scheduler.weight_decay_start}, end={config.scheduler.weight_decay_end}"
            )

    temperature_scheduler = None
    patch_temperature_scheduler = None
    if config.scheduler.warmup_teacher_temp_epochs > 0:
        teacher_temp_warmup_steps = max(1, round(config.scheduler.warmup_teacher_temp_epochs * steps_per_epoch))
        teacher_temp_warmup_steps = min(teacher_temp_warmup_steps, total_steps)
        temperature_scheduler = TemperatureScheduler(
            initial_tau=config.scheduler.warmup_teacher_temp,
            final_tau=config.scheduler.teacher_temp,
            warmup_steps=teacher_temp_warmup_steps,
        )
        patch_temperature_scheduler = TemperatureScheduler(
            initial_tau=config.scheduler.warmup_teacher_patch_temp,
            final_tau=config.scheduler.teacher_patch_temp,
            warmup_steps=teacher_temp_warmup_steps,
        )
        if is_main_process(rank):
            logger.info(
                f"Teacher temperature warmup enabled: {config.scheduler.warmup_teacher_temp_epochs} epochs "
                f"(~{teacher_temp_warmup_steps} steps), "
                f"CLS {config.scheduler.warmup_teacher_temp}->{config.scheduler.teacher_temp}, "
                f"PATCH {config.scheduler.warmup_teacher_patch_temp}->{config.scheduler.teacher_patch_temp}"
            )

    if temperature_scheduler is not None:
        raw.tau_teacher = temperature_scheduler.get_temperature()
    if patch_temperature_scheduler is not None:
        raw.tau_teacher_patch = patch_temperature_scheduler.get_temperature()

    if config.optimization.mixed_precision and is_main_process(rank):
        logger.info("Mixed precision (bf16) enabled")

    freeze_until_step = 0
    if config.scheduler.freeze_last_layer_epochs is not None:
        freeze_until_step = math.ceil(max(0.0, float(config.scheduler.freeze_last_layer_epochs)) * steps_per_epoch)
    elif config.scheduler.freeze_last_layer_steps is None:
        freeze_until_step = 0
    elif config.scheduler.freeze_last_layer_steps < 0:
        freeze_until_step = steps_per_epoch
    else:
        freeze_until_step = int(config.scheduler.freeze_last_layer_steps)

    if is_main_process(rank):
        if freeze_until_step > 0:
            logger.info(
                f"Freezing student_head.last_layer for the first {freeze_until_step} optimizer steps"
                f" (~{freeze_until_step / steps_per_epoch:.2f} epochs)."
            )
        else:
            logger.info("No last-layer freezing configured.")

    global_step = 0
    start_step = 0
    best_train_loss = float("inf")
    best_train_step = None

    if config.checkpoint.resume_from is not None and os.path.exists(config.checkpoint.resume_from):
        if is_main_process(rank):
            logger.info(f"Resuming from checkpoint: {config.checkpoint.resume_from}")
        start_step, extra_state = load_stage1_training_checkpoint(
            config.checkpoint.resume_from,
            raw,
            optimizer,
            lr_scheduler,
            None,
            restore_rng=True,
        )
        global_step = start_step
        momentum_scheduler.load_state_dict(extra_state["momentum_scheduler"])
        if temperature_scheduler is not None:
            temperature_scheduler.load_state_dict(extra_state["temperature_scheduler"])
            raw.tau_teacher = temperature_scheduler.get_temperature()
        if patch_temperature_scheduler is not None:
            patch_temperature_scheduler.load_state_dict(extra_state["patch_temperature_scheduler"])
            raw.tau_teacher_patch = patch_temperature_scheduler.get_temperature()
        if wd_scheduler is not None:
            wd_scheduler.load_state_dict(extra_state["wd_scheduler"])
            set_fixed_weight_decay(optimizer, wd_scheduler.get_weight_decay())
        raw.tau_teacher = float(extra_state["tau_teacher"])
        raw.tau_teacher_patch = float(extra_state["tau_teacher_patch"])
        best_train_loss = float(extra_state.get("best_train_loss", best_train_loss))
        best_train_step = extra_state.get("best_train_step", best_train_step)
        if is_main_process(rank):
            logger.info(f"Resumed from step {start_step}")

    if start_step == 0 and lr_warmup_steps > 0:
        lr_scheduler.step(0)
        if is_main_process(rank):
            logger.info(f"Applied warmup LR for step 0: {optimizer.param_groups[0]['lr']:.6e}")

    wandb_module = initialize_wandb(
        enabled=config.wandb,
        rank=rank,
        logger=logger,
        project=config.wandb_project,
        config=args_dict,
        name=f"moozy-stage1-{raw.d_model}d-{raw.n_layers}l",
        output_dir=config.output_dir,
        tags=config.wandb_tags if config.wandb_tags else None,
        warn_on_import_error=True,
    )

    from moozy.training.callbacks import Stage1CheckpointCallback, Stage1LoggingCallback

    logging_cb = Stage1LoggingCallback(
        rank=rank,
        world_size=world_size,
        log_every=config.log_every,
        accum_steps=accum_steps,
        device=device,
        logger=logger,
        wandb_module=wandb_module,
    )

    checkpoint_cb = Stage1CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        args_dict=args_dict,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=None,
        build_extra_state=lambda _: {},  # replaced below
        save_every=config.checkpoint.save_every,
        keep_last_n=config.checkpoint.keep_last_n,
        save_teacher=config.checkpoint.save_teacher,
        teacher_save_prefix=config.checkpoint.teacher_save_prefix,
        rank=rank,
        world_size=world_size,
        logger=logger,
    )
    checkpoint_cb.best_train_loss = best_train_loss
    checkpoint_cb.best_train_step = best_train_step

    # Closure reads live state from checkpoint_cb at save time.
    def build_extra_state(model_ref: Any) -> dict:
        return {
            "momentum_scheduler": momentum_scheduler.state_dict() if momentum_scheduler is not None else None,
            "temperature_scheduler": temperature_scheduler.state_dict() if temperature_scheduler is not None else None,
            "patch_temperature_scheduler": (
                patch_temperature_scheduler.state_dict() if patch_temperature_scheduler is not None else None
            ),
            "wd_scheduler": wd_scheduler.state_dict() if wd_scheduler is not None else None,
            "tau_teacher": model_ref.tau_teacher,
            "tau_teacher_patch": model_ref.tau_teacher_patch,
            "best_train_loss": checkpoint_cb.best_train_loss,
            "best_train_step": checkpoint_cb.best_train_step,
        }

    checkpoint_cb.build_extra_state = build_extra_state

    engine = Stage1Engine(
        model=model,
        optimizer=optimizer,
        callbacks=[logging_cb, checkpoint_cb],
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_scheduler=lr_scheduler,
        momentum_scheduler=momentum_scheduler,
        temperature_scheduler=temperature_scheduler,
        patch_temperature_scheduler=patch_temperature_scheduler,
        wd_scheduler=wd_scheduler,
        scaler=None,
        total_steps=total_steps,
        accum_steps=accum_steps,
        grad_clip=config.optimization.grad_clip,
        mixed_precision=config.optimization.mixed_precision,
        val_every=config.val_every,
        freeze_until_step=freeze_until_step,
        train_sampler=None,
        rank=rank,
        logger=logger,
        wandb_module=wandb_module,
        start_step=start_step,
        global_step=global_step,
    )
    engine.run()

    finish_wandb(wandb_module)
    cleanup_distributed()
