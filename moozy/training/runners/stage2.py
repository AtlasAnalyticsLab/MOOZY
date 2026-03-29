import math
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from moozy.config.stage2 import Stage2TrainConfig
from moozy.data.features import (
    TRAINING_TOKEN_PRESETS_BF16,
    TRAINING_TOKEN_PRESETS_FP32,
    detect_feat_dim,
    list_feature_paths,
    resolve_vram_token_cap,
)
from moozy.data.stage2 import SupervisedCaseDataset, build_stage2_dataloader
from moozy.models.factory import build_supervised_model, load_teacher_slide_encoder
from moozy.tasks import (
    build_case_event_matrix,
    build_case_label_matrix,
    build_case_time_matrix,
    build_supervised_cases,
    build_survival_bin_edges,
    load_task_supervision,
    log_task_coverage,
    split_train_val_indices_task_stratified,
)
from moozy.tasks.loader import discover_task_csvs
from moozy.training import (
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
from moozy.training.engine.stage2 import Stage2Engine


def run_stage2(config: Stage2TrainConfig) -> None:
    """Run stage-2 supervised training from a typed config."""
    args_dict = config.to_flat_dict()

    runtime = initialize_training_runtime(
        backend=config.backend,
        output_dir=config.output_dir,
        seed=config.seed,
        logger_name="moozy.stage_2_supervised_alignment",
    )
    rank = runtime.rank
    local_rank = runtime.local_rank
    world_size = runtime.world_size
    logger = runtime.logger

    wandb_module = initialize_wandb(
        enabled=config.wandb,
        rank=rank,
        logger=logger,
        project=config.wandb_project,
        config=args_dict,
        name=f"moozy-stage2-{os.path.basename(config.output_dir)}",
        output_dir=config.output_dir,
        tags=config.wandb_tags if config.wandb_tags else None,
        warn_on_import_error=True,
    )

    if is_main_process(rank):
        hp_path = save_hyperparameters(args_dict, config.output_dir)
        logger.info("Saved hyperparameters to %s", hp_path)

    task_csvs = discover_task_csvs(config.task_dir)
    task_info: Any = None
    if rank == 0:
        logger.info("Discovered %d task CSVs in %s", len(task_csvs), config.task_dir)
        task_info = load_task_supervision(config.data.feature_dirs, task_csvs, logger)
    if dist.is_available() and dist.is_initialized():
        objs = [task_info]
        dist.broadcast_object_list(objs, src=0)
        task_info = objs[0]
    if not task_info:
        raise ValueError("Task supervision enabled but no tasks were resolved.")
    if is_main_process(rank):
        logger.info("Supervision task count: %d", len(task_info.get("tasks", [])))

    cases, dropped_empty_cases = build_supervised_cases(task_info)
    total_cases = len(cases)
    if config.debug:
        cases = cases[:5]
    if not cases:
        raise ValueError("No labeled cases found for the provided task CSVs.")

    total_slides = sum(len(case["paths"]) for case in cases)
    case_ids = [case["case_id"] for case in cases]
    labels = build_case_label_matrix(case_ids, task_info["case_labels"])
    events = build_case_event_matrix(case_ids, task_info["case_events"])
    times = build_case_time_matrix(case_ids, task_info["case_times"])
    if is_main_process(rank):
        if dropped_empty_cases:
            logger.info(
                "Dropped %d cases with no slides (examples): %s",
                len(dropped_empty_cases),
                ", ".join(dropped_empty_cases[:5]),
            )
        if config.debug and total_cases > len(cases):
            logger.info("Debug mode truncation: using first %d cases out of %d.", len(cases), total_cases)
        logger.info("Loaded %d labeled cases (%d slides) for supervision.", len(cases), total_slides)
        log_task_coverage(logger, task_info, labels, events=events, times=times)

    training_presets = (
        TRAINING_TOKEN_PRESETS_BF16 if config.optimization.mixed_precision else TRAINING_TOKEN_PRESETS_FP32
    )
    max_valid_tokens_per_slide = resolve_vram_token_cap(
        presets=training_presets,
        logger=logger if is_main_process(rank) else None,
        local_rank=int(local_rank),
    )

    train_dataset = None
    val_dataset = None
    train_events = None
    train_times = None
    preload = not config.data.lazy_feature_loading

    if config.val_ratio and config.val_ratio > 0.0:
        task_types_for_split = [t.get("task_type", "classification") for t in task_info["tasks"]]
        train_indices, val_indices, remaining = split_train_val_indices_task_stratified(
            labels,
            config.val_ratio,
            config.seed,
            events=events,
            task_types=task_types_for_split,
        )
        if is_main_process(rank) and remaining:
            remaining_total = sum(max(0, v) for v in remaining.values())
            if remaining_total > 0:
                logger.warning(
                    "Per-task stratified split left %d label targets unmet; consider increasing --val_ratio.",
                    remaining_total,
                )
        train_cases = [cases[i] for i in train_indices]
        train_labels = labels[train_indices]
        train_events = events[train_indices]
        train_times = times[train_indices]
        val_cases = [cases[i] for i in val_indices]
        val_labels = labels[val_indices]
        val_events = events[val_indices]
        val_times = times[val_indices]
        train_dataset = SupervisedCaseDataset(
            train_cases,
            train_labels,
            events=train_events,
            times=train_times,
            augment=True,
            hflip_prob=config.data.hflip_prob,
            vflip_prob=config.data.vflip_prob,
            rotate_prob=config.data.rotate_prob,
            token_dropout_ratio=config.data.token_dropout_ratio,
            max_valid_tokens_per_slide=max_valid_tokens_per_slide,
            token_cap_sampling=config.data.train_token_cap_sampling,
            preload_features=preload,
            max_cached_slides=config.data.max_cached_slides,
            feature_h5_format=config.data.feature_h5_format,
            feature_h5_key=config.data.feature_h5_key,
        )
        val_dataset = SupervisedCaseDataset(
            val_cases,
            val_labels,
            events=val_events,
            times=val_times,
            augment=False,
            hflip_prob=0.0,
            vflip_prob=0.0,
            rotate_prob=0.0,
            token_dropout_ratio=0.0,
            max_valid_tokens_per_slide=max_valid_tokens_per_slide,
            token_cap_sampling="deterministic",
            preload_features=preload,
            max_cached_slides=config.data.max_cached_slides,
            feature_h5_format=config.data.feature_h5_format,
            feature_h5_key=config.data.feature_h5_key,
        )
    else:
        train_events = events
        train_times = times
        train_dataset = SupervisedCaseDataset(
            cases,
            labels,
            events=events,
            times=times,
            augment=True,
            hflip_prob=config.data.hflip_prob,
            vflip_prob=config.data.vflip_prob,
            rotate_prob=config.data.rotate_prob,
            token_dropout_ratio=config.data.token_dropout_ratio,
            max_valid_tokens_per_slide=max_valid_tokens_per_slide,
            token_cap_sampling=config.data.train_token_cap_sampling,
            preload_features=preload,
            max_cached_slides=config.data.max_cached_slides,
            feature_h5_format=config.data.feature_h5_format,
            feature_h5_key=config.data.feature_h5_key,
        )

    survival_bin_edges = build_survival_bin_edges(
        task_info=task_info,
        train_events=train_events,
        train_times=train_times,
        target_bins=config.survival_num_bins,
        min_bins=config.survival_min_bins,
        max_bins=config.survival_max_bins,
        logger=logger if is_main_process(rank) else None,
    )

    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = build_stage2_dataloader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        prefetch_factor=config.data.prefetch_factor,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = None
        if world_size > 1:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = build_stage2_dataloader(
            dataset=val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            prefetch_factor=config.data.prefetch_factor,
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

    teacher_checkpoint = str(config.teacher_checkpoint or "").strip()
    if is_main_process(rank):
        logger.info("Detected feat_dim=%d from %s", feat_dim, h5_paths[0])
        if teacher_checkpoint:
            logger.info("Building slide encoder from checkpoint: %s", teacher_checkpoint)
        else:
            se = config.slide_encoder
            logger.info(
                "Teacher checkpoint is empty; initializing slide encoder from scratch "
                "(variant=%s feat_dim=%d registers=%d qk_norm=%s layerscale_init=%.1e learnable_alibi=%s).",
                se.variant,
                feat_dim,
                se.num_registers,
                str(se.qk_norm),
                se.layerscale_init,
                str(se.learnable_alibi),
            )

    slide_encoder, _meta = load_teacher_slide_encoder(config.slide_encoder, teacher_checkpoint, feat_dim=feat_dim)
    slide_encoder.mask_token.requires_grad = False

    model = build_supervised_model(
        slide_encoder,
        task_info=task_info,
        survival_bin_edges=survival_bin_edges,
        case_transformer_variant=config.case_transformer_variant,
        case_transformer_dropout=config.case_transformer_dropout,
        case_transformer_layerscale_init=config.case_transformer_layerscale_init,
        case_transformer_layer_drop=config.case_transformer_layer_drop,
        case_transformer_qk_norm=config.case_transformer_qk_norm,
        case_transformer_num_registers=config.case_transformer_num_registers,
        classification_head_type=config.classification_head_type,
        survival_head_type=config.survival_head_type,
        head_dropout=config.head_dropout,
        label_smoothing=config.label_smoothing,
        survival_num_bins=config.survival_num_bins,
    )
    model.set_activation_checkpointing(config.activation_checkpointing)
    if is_main_process(rank):
        logger.info("Activation checkpointing: %s", "enabled" if config.activation_checkpointing else "disabled")

    if config.freeze_slide_encoder:
        for p in model.slide_encoder.parameters():
            p.requires_grad = False

    if is_main_process(rank):
        log_parameter_breakdown(
            logger,
            [
                (
                    "Slide Encoder",
                    count_parameters(model.slide_encoder),
                    count_parameters(model.slide_encoder, trainable_only=True),
                ),
                (
                    "Case Transformer",
                    count_parameters(model.case_transformer),
                    count_parameters(model.case_transformer, trainable_only=True),
                ),
                (
                    "Task Heads",
                    count_parameters(model.task_heads),
                    count_parameters(model.task_heads, trainable_only=True),
                ),
            ],
        )
        ct = model.case_transformer
        logger.info(
            "Case transformer: variant=%s layers=%d heads=%d ffn_dim=%d dropout=%.3f "
            "layerscale=%.1e layer_drop=%.3f qk_norm=%s registers=%d",
            config.case_transformer_variant,
            int(ct.num_layers),
            int(ct.num_heads),
            int(ct.dim_feedforward),
            config.case_transformer_dropout,
            config.case_transformer_layerscale_init,
            config.case_transformer_layer_drop,
            str(config.case_transformer_qk_norm),
            config.case_transformer_num_registers,
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for _, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    effective_global_batch = (
        config.data.batch_size * max(1, world_size) * max(1, config.optimization.grad_accumulation_steps)
    )
    scaled_lr = config.optimization.lr * (effective_global_batch / config.optimization.lr_base_batch_size)
    if is_main_process(rank):
        logger.info(
            "LR scaling: base_lr=%.2e * (effective_batch=%d / base_batch=%d) = scaled_lr=%.2e",
            config.optimization.lr,
            effective_global_batch,
            config.optimization.lr_base_batch_size,
            scaled_lr,
        )

    optimizer = build_optimizer(
        model, config.optimization.optimizer, lr=scaled_lr, weight_decay=config.optimization.weight_decay
    )
    set_fixed_weight_decay(optimizer, config.optimization.weight_decay)
    if is_main_process(rank):
        decay_groups = [pg for pg in optimizer.param_groups if pg.get("apply_weight_decay", True)]
        no_decay_groups = [pg for pg in optimizer.param_groups if not pg.get("apply_weight_decay", True)]
        decay_params = int(sum(p.numel() for pg in decay_groups for p in pg.get("params", [])))
        no_decay_params = int(sum(p.numel() for pg in no_decay_groups for p in pg.get("params", [])))
        logger.info(
            "Fixed weight decay enabled (stage-1 parity groups): wd=%.3e on %d params, no_wd on %d params.",
            config.optimization.weight_decay,
            decay_params,
            no_decay_params,
        )

    accum_steps = max(1, config.optimization.grad_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / float(accum_steps)))
    total_steps = int(max(1, config.epochs) * steps_per_epoch)
    scheduler = build_warmup_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr=config.optimization.lr_min,
        schedule=config.lr_schedule,
    )

    if is_main_process(rank):
        train_batches_per_rank = len(train_loader)
        if train_sampler is not None:
            samples_per_rank = train_sampler.num_samples
            global_cases_per_epoch = samples_per_rank * max(1, world_size)
        else:
            samples_per_rank = len(train_dataset)
            global_cases_per_epoch = len(train_dataset)
        remainder_microsteps = int(train_batches_per_rank % max(1, accum_steps))
        logger.info(
            "Training setup: epochs=%d, steps/epoch=%d, total_steps=%d", config.epochs, steps_per_epoch, total_steps
        )
        logger.info(
            "Survival objective: discrete_hazard (target_bins=%d, min_bins=%d, max_bins=%d, tasks_with_edges=%d)",
            config.survival_num_bins,
            config.survival_min_bins,
            config.survival_max_bins,
            len(survival_bin_edges),
        )
        logger.info(
            "Batch/step accounting: train_batches_per_rank=%d, accum_steps=%d, optimizer_steps_per_epoch_ceiling=%d, "
            "samples_per_rank=%d, approx_global_cases_per_epoch=%d",
            train_batches_per_rank,
            accum_steps,
            steps_per_epoch,
            samples_per_rank,
            global_cases_per_epoch,
        )
        if remainder_microsteps > 0:
            logger.info(
                "Per-epoch accumulation remainder is %d/%d micro-batches; no epoch-end flush "
                "(stage-1 parity), so gradients carry into the next epoch.",
                remainder_microsteps,
                accum_steps,
            )
        logger.info(
            "Effective global batch size: %d (per-rank=%d, accum=%d)",
            effective_global_batch,
            config.data.batch_size,
            config.optimization.grad_accumulation_steps,
        )

    from moozy.training.callbacks import Stage2CheckpointCallback, Stage2LoggingCallback

    logging_cb = Stage2LoggingCallback(
        rank=rank,
        logger=logger,
        wandb_module=wandb_module,
    )
    checkpoint_cb = Stage2CheckpointCallback(
        output_dir=config.output_dir,
        save_every_epochs=config.checkpoint.save_every_epochs,
        keep_last_n=config.checkpoint.keep_last_n,
        rank=rank,
        logger=logger,
    )

    engine = Stage2Engine(
        model=model,
        optimizer=optimizer,
        callbacks=[logging_cb, checkpoint_cb],
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        epochs=config.epochs,
        grad_accumulation_steps=config.optimization.grad_accumulation_steps,
        grad_clip=config.optimization.grad_clip,
        mixed_precision=config.optimization.mixed_precision,
        log_every=config.log_every,
        train_sampler=train_sampler,
        rank=rank,
        logger=logger,
        wandb_module=wandb_module,
    )
    engine.run()

    if is_main_process(rank):
        finish_wandb(wandb_module)
    cleanup_distributed()
