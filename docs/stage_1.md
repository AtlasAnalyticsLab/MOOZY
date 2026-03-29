# Stage 1: Self-Supervised Pretraining

- [Overview](#overview)
- [Usage Guide](#usage-guide)
  - [Distributed training](#distributed-training)
  - [Single-GPU training](#single-gpu-training)
  - [Resuming from a checkpoint](#resuming-from-a-checkpoint)
- [Arguments](#arguments)
  - [Data](#data)
  - [Cropping](#cropping)
  - [Masking](#masking)
  - [Augmentation](#augmentation)
  - [Slide Encoder](#slide-encoder)
  - [Projection Head](#projection-head)
  - [Self-Distillation](#self-distillation)
  - [Optimization](#optimization)
  - [Checkpointing](#checkpointing)
  - [Logging and Validation](#logging-and-validation)
  - [Runtime](#runtime)

## Overview

Stage 1 pretrains a slide encoder on unlabeled whole-slide image feature grids using masked self-distillation. A student slide encoder and an EMA teacher are jointly trained via CLS-level cross-view distillation (e.g., DINO) and masked image modeling (e.g., iBOT). The student processes both global and local crops (with block masking applied to global crops), while the teacher only processes unmasked global crops. Teacher outputs are centered with momentum-updated running averages to prevent mode collapse.

The slide encoder is a Vision Transformer adapted for precomputed patch feature grids. It uses 2-D ALiBi for spatial position encoding, a learnable CLS token, optional register tokens, and a projection head that maps encoder tokens to prototype logits via an MLP with an L2-normalized bottleneck and a weight-normalized prototype layer.

Input data consists of H5 feature files containing per-patch feature vectors and coordinates, arranged into spatial grids at training time. The patch encoder used to produce these features is external to MOOZY (`lunit_vit_small_patch8_dino` with `feat_dim=384`).

## Usage Guide

### Distributed training

The recommended way to launch Stage 1 is via `torchrun` with the provided shell script. The script sets NCCL environment variables for single-node multi-GPU training and passes through to `moozy train stage1`.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_stage1.sh
```

The shell script uses placeholder paths. Override arguments on the command line:

```bash
GPU_IDS=0,1,2,3 bash scripts/train_stage1.sh \
  --feature_dirs /data/tcga_features /data/cptac_features \
  --batch_size 64 \
  --epochs 200 \
  --grad_accumulation_steps 2 \
  --mixed_precision \
  --output_dir ./results/stage1_run \
  --wandb
```

Or call `torchrun` directly:

```bash
torchrun --nproc_per_node=8 --module moozy train stage1 \
  --feature_dirs /data/features \
  --epochs 200 \
  --mixed_precision \
  --output_dir ./results/stage1
```

### Single-GPU training

For a single GPU, run `moozy` as a module without `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m moozy train stage1 \
  --feature_dirs /data/features \
  --batch_size 64 \
  --epochs 200 \
  --output_dir ./results/stage1_single
```

### Resuming from a checkpoint

Pass `--resume_from` with the path to a full training checkpoint. All state is restored: model weights, optimizer, LR scheduler, EMA momentum scheduler, temperature schedulers, weight-decay scheduler, RNG states, and the best-loss tracker.

```bash
torchrun --nproc_per_node=8 --module moozy train stage1 \
  --feature_dirs /data/features \
  --resume_from ./results/stage1/checkpoints/checkpoint_step_7200.pt \
  --output_dir ./results/stage1
```

## Arguments

### Data

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--feature_dirs` | str (list) | ✓ | - | One or more directories containing H5 feature files. Each directory is scanned recursively for `.h5` files. All slides across all directories form the training set. When running distributed, each rank sees the full set of paths (the DataLoader handles sharding). |
| `--feature_h5_format` | str | ✗ | `auto` | H5 schema to use when reading feature files. `auto` detects the format from file contents. `atlaspatch` reads the AtlasPatch layout. `trident` reads the TRIDENT layout. Valid values: `auto`, `atlaspatch`, `trident`. |
| `--feature_h5_key` | str | ✗ | `""` | AtlasPatch-specific feature key under the `features/` group. Only relevant when `--feature_h5_format` is `atlaspatch` or `auto` (and the file uses AtlasPatch layout). |
| `--batch_size` | int | ✗ | `64` | Micro batch size per GPU. The effective global batch size is `batch_size * world_size * grad_accumulation_steps`. The learning rate is scaled linearly relative to `--lr_base_batch_size`. |
| `--num_workers` | int | ✗ | `4` | Number of DataLoader worker processes per GPU. |
| `--prefetch_factor` | int | ✗ | `4` | Number of batches each worker prefetches ahead of time. |
| `--lazy_feature_loading` / `--no_lazy_feature_loading` | flag | ✗ | `--no_lazy_feature_loading` | When enabled, feature grids are loaded from disk on demand rather than preloaded into memory at startup. Reduces initial memory and startup time at the cost of slower per-sample I/O. |
| `--max_cached_slides` | int | ✗ | `0` | Maximum number of lazily-loaded slides to keep in an LRU cache per worker process. Only takes effect when `--lazy_feature_loading` is enabled. 0 disables caching (every access reads from disk). |

### Cropping

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--global_crop_size` | int | ✗ | `20` | Side length (in tokens) of global crops sampled from the feature grid. Each global crop is a `global_crop_size x global_crop_size` window. Larger crops capture more spatial context but increase memory. |
| `--local_crop_size` | int | ✗ | `12` | Side length (in tokens) of local crops. Must be smaller than `--global_crop_size`. Local crops capture fine-grained detail and are processed by the student only. |
| `--num_global_crops` | int | ✗ | `2` | Number of global crops sampled per slide per training step. Both student and teacher process all global crops. The teacher provides soft targets from these views for the CLS distillation loss. |
| `--num_local_crops` | int | ✗ | `4` | Number of local crops sampled per slide per training step. Only the student processes local crops. The student predicts the teacher's global CLS distributions from these views. |
| `--min_window_patch_ratio` | float | ✗ | `0.25` | Minimum fraction of non-zero (tissue) patches required in a sampled crop. Crops below this ratio are discarded and resampled up to `--crop_resample_attempts` times. Prevents training on mostly-background windows. |
| `--crop_resample_attempts` | int | ✗ | `3` | Maximum number of resampling attempts when a crop fails the `--min_window_patch_ratio` constraint. If all attempts fail, the last sampled crop is used regardless. |

### Masking

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--mask_ratio_min` | float | ✗ | `0.1` | Lower bound of the per-crop mask ratio range. Within each batch, mask ratios are distributed uniformly across `[mask_ratio_min, mask_ratio_max]` and shuffled across the selected global crops, so the batch covers the full masking spectrum. |
| `--mask_ratio_max` | float | ✗ | `0.5` | Upper bound of the per-crop mask ratio range. |
| `--min_num_mask_patches` | int | ✗ | `4` | Minimum number of patches per rectangular masking block. Controls the smallest block that can be placed during iterative block masking. |
| `--max_num_mask_patches` | int | ✗ | `-1` | Maximum number of patches per rectangular masking block. -1 removes the cap, allowing blocks as large as needed to fill the target mask ratio. |
| `--mask_min_aspect` | float | ✗ | `0.3` | Minimum aspect ratio for masking blocks. Block aspect ratios are drawn log-uniformly from `[mask_min_aspect, mask_max_aspect]`. Lower values produce more elongated blocks. |
| `--mask_max_aspect` | float or null | ✗ | `None` | Maximum aspect ratio for masking blocks. When not set, defaults to `1/mask_min_aspect` (approximately 3.33 for the default `mask_min_aspect=0.3`). |
| `--mask_sample_probability` | float | ✗ | `0.5` | Probability that each global crop is selected for masking. Unselected global crops are processed without any mask. The MIM loss is computed only over masked valid positions in the selected crops. |

### Augmentation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--hflip_prob` | float | ✗ | `0.5` | Probability of applying a horizontal flip to each crop (both global and local). Spatial augmentations are applied independently per crop after sampling. |
| `--vflip_prob` | float | ✗ | `0.5` | Probability of applying a vertical flip to each crop. |
| `--rotate_prob` | float | ✗ | `0.5` | Probability of applying a 90/180/270-degree rotation to each crop. When triggered, one of the three rotation angles is chosen uniformly at random. |

### Slide Encoder

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--encoder_variant` | str | ✗ | `base_half_depth` | Architecture preset for the slide encoder. Determines `d_model`, `n_heads`, `n_layers`, and `dim_feedforward` from a lookup table. Available variants: `tiny` (192d, 3h, 12L), `small` (384d, 6h, 12L), `base` (768d, 12h, 12L), `base_half_depth` (768d, 12h, 6L), `base_quarter_depth` (768d, 12h, 3L), `large` (1024d, 16h, 24L), `large_half_depth` (1024d, 16h, 12L), `large_quarter_depth` (1024d, 16h, 6L). |
| `--num_registers` | int | ✗ | `4` | Number of register tokens prepended after the CLS token. Register tokens participate in attention but are not used in the loss. They absorb global information and reduce artifact attention patterns. 0 disables register tokens. |
| `--layer_drop` | float | ✗ | `0.1` | Maximum stochastic depth (DropPath) rate. Drop rates are linearly interpolated from 0 at the first block to this value at the last block. Applied during training only. |
| `--dropout` | float | ✗ | `0.1` | Dropout rate inside the MLP sub-layers of each transformer block. |
| `--attn_dropout` | float | ✗ | `0.0` | Dropout rate applied to attention weights in scaled dot-product attention. |
| `--qk_norm` / `--no_qk_norm` | flag | ✗ | `--no_qk_norm` | Apply per-head LayerNorm to Q and K projections before computing attention scores. Can stabilize training at larger model scales. |
| `--layerscale_init` | float | ✗ | `0.0` | Initial value for LayerScale gamma parameters. Each transformer block multiplies its residual branch by a learnable scalar initialized to this value. 0 or negative disables LayerScale. |
| `--learnable_alibi` / `--no_learnable_alibi` | flag | ✗ | `--no_learnable_alibi` | Make the ALiBi head slopes learnable parameters instead of fixed geometric values. Fixed slopes (the default) follow the standard ALiBi initialization. |

### Projection Head

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output_dim` | int | ✗ | `8192` | Number of prototype logits (output dimension of the projection head). This is the dimensionality of the softmax distributions used in both the CLS distillation loss and the MIM loss. |
| `--proj_hidden_dim` | int | ✗ | `2048` | Hidden dimension of the 3-layer projection MLP (the two intermediate linear layers). |
| `--proj_bottleneck_dim` | int | ✗ | `256` | Dimension of the L2-normalized bottleneck between the MLP and the final weight-normalized prototype layer. |
| `--proj_norm_last_layer` / `--no_proj_norm_last_layer` | flag | ✗ | `--proj_norm_last_layer` | Freeze the weight-norm gain parameter in the projection head's final prototype layer. When enabled (the default), the gain is fixed at 1.0, and only the direction of prototype vectors is learned. Disabling allows the gain to be trained. |
| `--proj_norm` | str | ✗ | `none` | Normalization type applied after the first two linear layers inside the projection MLP. Valid values: `none`, `ln` (LayerNorm). |
| `--proj_last_norm` | str | ✗ | `none` | Optional normalization applied after the final prototype layer output. Valid values: `none`, `ln` (LayerNorm). |

### Self-Distillation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--ema_momentum_start` | float | ✗ | `0.996` | Initial EMA momentum for the teacher update. The teacher parameters are updated each optimizer step as `teacher = momentum * teacher + (1 - momentum) * student`. |
| `--ema_momentum` | float | ✗ | `1.0` | Final EMA momentum at the end of training. A value of 1.0 means the teacher eventually stops updating and freezes. |
| `--momentum_schedule` | str | ✗ | `cosine` | Schedule for interpolating between `--ema_momentum_start` and `--ema_momentum` over the full training run. Valid values: `linear`, `cosine`. |
| `--student_temp` | float | ✗ | `0.1` | Temperature applied to the student's softmax distributions for both CLS and patch logits. Higher values produce softer distributions. |
| `--teacher_temp` | float | ✗ | `0.07` | Final teacher temperature for CLS-level softmax distributions. Reached after the warmup period defined by `--warmup_teacher_temp_epochs`. |
| `--teacher_patch_temp` | float | ✗ | `0.07` | Final teacher temperature for patch-level softmax distributions used in the MIM loss. Reached after the same warmup period as the CLS temperature. |
| `--warmup_teacher_temp` | float | ✗ | `0.04` | Starting teacher CLS temperature. The temperature is linearly interpolated from this value to `--teacher_temp` over `--warmup_teacher_temp_epochs`. |
| `--warmup_teacher_patch_temp` | float | ✗ | `0.04` | Starting teacher patch temperature. Linearly interpolated to `--teacher_patch_temp` over the same warmup period. |
| `--warmup_teacher_temp_epochs` | int | ✗ | `30` | Number of epochs over which teacher temperatures are warmed up. Converted to optimizer steps internally using the computed steps-per-epoch. 0 disables temperature warmup (teacher starts at the final temperatures). |
| `--center_momentum` | float | ✗ | `0.9` | Momentum for the exponential moving average updates of the teacher centering vectors. The center is subtracted from teacher logits before computing soft targets. Higher values make the center more stable. |

### Optimization

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--epochs` | float | ✗ | `200` | Number of training epochs. Positive values override `--total_steps`. Supports fractional values (e.g., 0.5 for half an epoch). The total optimizer step count is `ceil(epochs * steps_per_epoch)`. Set `--epochs 0` to drive training by `--total_steps` instead. |
| `--total_steps` | int | ✗ | `0` | Total number of optimizer steps when `--epochs` is 0 or negative. The default run is epoch-driven, so this fallback is disabled by default. |
| `--optimizer` | str | ✗ | `adamw` | Optimizer algorithm. Valid values: `adamw`, `adam`, `sgd`. |
| `--lr` | float | ✗ | `5e-4` | Base learning rate before linear scaling. The effective learning rate is `lr * (effective_global_batch / lr_base_batch_size)`. |
| `--lr_min` | float | ✗ | `2e-6` | Minimum learning rate at the end of the cosine/linear decay. The scheduler decays from the scaled LR to this floor. |
| `--lr_schedule` | str | ✗ | `cosine` | Learning rate decay schedule applied after warmup. Valid values: `linear`, `cosine`. |
| `--lr_base_batch_size` | int | ✗ | `256` | Reference global batch size for linear LR scaling. The learning rate is multiplied by `effective_global_batch / lr_base_batch_size`. For example, with `lr=5e-4`, `batch_size=64`, 8 GPUs, and `grad_accumulation_steps=2`, the effective batch is 1024, and the scaled LR is `5e-4 * 1024/256 = 2e-3`. |
| `--warmup_steps` | int | ✗ | `0` | Number of linear LR warmup steps. Takes precedence over `--warmup_epochs` when greater than 0. |
| `--warmup_epochs` | float | ✗ | `5` | Number of epochs for LR warmup. Converted to steps using the computed steps-per-epoch. Only used when `--warmup_steps` is 0. |
| `--weight_decay` | float | ✗ | `0.04` | Global weight decay passed to the optimizer. When `--wd_schedule` is not `none`, this value is overridden by the weight-decay scheduler (see `--weight_decay_start` and `--weight_decay_end`). Bias and normalization parameters are always exempt from weight decay. |
| `--weight_decay_start` | float | ✗ | `0.04` | Starting weight decay when `--wd_schedule` is enabled. |
| `--weight_decay_end` | float | ✗ | `0.4` | Final weight decay when `--wd_schedule` is enabled. |
| `--wd_schedule` | str | ✗ | `cosine` | Weight-decay schedule type. `cosine` interpolates from `--weight_decay_start` to `--weight_decay_end` over training. `linear` does the same with linear interpolation. `none` keeps weight decay fixed at `--weight_decay`. Valid values: `linear`, `cosine`, `none`. |
| `--grad_clip` | float | ✗ | `0.3` | Per-parameter gradient clipping max norm. Applied to the student slide encoder and student projection head after each backward pass. 0 disables clipping. |
| `--grad_accumulation_steps` | int | ✗ | `1` | Number of micro-batches to accumulate before each optimizer step. Increases the effective batch size without increasing per-GPU memory. |
| `--mixed_precision` / `--no_mixed_precision` | flag | ✗ | `--no_mixed_precision` | Enable bfloat16 automatic mixed precision. Reduces GPU memory usage and can speed up training on Ampere+ GPUs. |
| `--freeze_last_layer_steps` | int or null | ✗ | `None` | Freeze the final weight-normalized layer of the student projection head for this many optimizer steps. Prevents early training instability from large prototype updates. When set, takes precedence over `--freeze_last_layer_epochs`. |
| `--freeze_last_layer_epochs` | float | ✗ | `3.0` | Freeze the final student projection layer for this many epochs. Converted to steps internally. Only used when `--freeze_last_layer_steps` is not set. |

### Checkpointing

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--save_every` | int | ✗ | `72` | Save a full training checkpoint every N optimizer steps. Each checkpoint contains model weights (student + teacher), optimizer state, scheduler states, RNG states, and all schedule metadata needed for exact resumption. 0 disables periodic saving. |
| `--resume_from` | path or null | ✗ | `None` | Path to a full training checkpoint to resume from. Restores all model, optimizer, scheduler, and RNG state. Training continues from the saved global step. |
| `--keep_last_n` | int | ✗ | `50` | Maximum number of full training checkpoints to keep on disk. Older checkpoints are deleted after each save. |
| `--save_teacher` / `--no_save_teacher` | flag | ✗ | `--save_teacher` | Save a lightweight teacher-only checkpoint alongside each full checkpoint. Teacher checkpoints contain only the teacher slide encoder weights and architectural metadata (no optimizer state, no projection head). These are the checkpoints used to initialize Stage 2. |
| `--teacher_save_prefix` | str | ✗ | `teacher_step` | Filename prefix for teacher-only checkpoints. The file is saved as `{prefix}_{step}.pt` in the checkpoint directory. |

### Logging and Validation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--log_every` | int | ✗ | `72` | Log training metrics (loss, CLS/MIM components, LR, momentum, weight decay, teacher-student agreement) every N optimizer steps. |
| `--val_ratio` | float | ✗ | `0.05` | Fraction of the dataset to hold out for validation. The split is random with a fixed seed. Set to 0 to disable validation. |
| `--val_every` | int | ✗ | `1000` | Run validation every N optimizer steps. Only takes effect when `--val_ratio` is greater than 0. |
| `--wandb` / `--no_wandb` | flag | ✗ | `--no_wandb` | Enable Weights & Biases logging. Requires the `wandb` package. Training and validation metrics are logged at each logging/validation step. |
| `--wandb_project` | str | ✗ | `moozy` | Weights & Biases project name. |
| `--wandb_tags` | str | ✗ | `""` | Space or comma-separated tags for the Weights & Biases run. |

### Runtime

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output_dir` | path | ✗ | `./results` | Output directory for logs, hyperparameters, and checkpoints. Checkpoints are saved under `{output_dir}/checkpoints/`. Created automatically if it does not exist. |
| `--distributed` / `--no_distributed` | flag | ✗ | `--no_distributed` | Enable distributed training. When using `torchrun`, distribution is handled by the PyTorch distributed launcher and this flag is not required. |
| `--backend` | str | ✗ | `nccl` | Distributed communication backend. Valid values: `nccl`, `gloo`. Use `nccl` for GPU training. |
| `--local_rank` | int | ✗ | `0` | Local rank for distributed training. Typically set automatically by `torchrun`. |
| `--seed` | int | ✗ | `42` | Random seed for reproducibility. Controls dataset shuffling, validation split, crop sampling, and weight initialization. |
| `--debug` / `--no_debug` | flag | ✗ | `--no_debug` | Debug mode. Limits the dataset to a small subset for quick iteration. |
