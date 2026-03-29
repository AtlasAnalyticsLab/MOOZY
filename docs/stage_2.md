# Stage 2: Supervised Alignment

- [Overview](#overview)
- [Usage Guide](#usage-guide)
  - [Distributed training](#distributed-training)
  - [Single-GPU training](#single-gpu-training)
  - [Task definitions](#task-definitions)
  - [Training without a Stage 1 checkpoint](#training-without-a-stage-1-checkpoint)
- [Arguments](#arguments)
  - [Data](#data)
  - [Augmentation](#augmentation)
  - [Task Supervision](#task-supervision)
  - [Slide Encoder](#slide-encoder)
  - [Case Transformer](#case-transformer)
  - [Optimization](#optimization)
  - [Checkpointing](#checkpointing)
  - [Logging and Validation](#logging-and-validation)
  - [Runtime](#runtime)

## Overview

Stage 2 steers the slide encoder pretrained in Stage 1 toward clinical semantics through multi-task supervision over classification and survival endpoints. The fundamental unit of representation is the patient case, not the individual slide. Each case contains one or more slides, and a case transformer explicitly models dependencies across all slides of the same patient, producing a single case-level embedding.

Full slide feature grids (no crop sampling as in stage 1) are passed through the slide encoder to produce one CLS embedding per slide. A hardware-adaptive token cap automatically limits the number of valid tokens per slide to fit GPU memory, using stratified random sampling to preserve spatial coverage. All per-slide CLS embeddings for a case are then aggregated by a case transformer, a lightweight transformer that prepends a learnable CASE token and outputs a case-level representation. This representation is routed to task-specific heads for classification (weighted cross-entropy with label smoothing) and survival prediction (discrete-hazard negative log-likelihood). Losses are averaged uniformly over all tasks with valid labels in each batch.

Task definitions are loaded from a task directory containing per-task subdirectories, each with a `task.csv` and a `config.yaml`. By default, the task directory is downloaded from [HuggingFace](https://huggingface.co/AtlasAnalyticsLab/MOOZY) (`AtlasAnalyticsLab/MOOZY`). The bundled tasks cover 333 tasks from 56 public datasets, including 205 classification tasks and 128 survival tasks across four endpoints (OS, DSS, DFI, PFI).

At inference, the slide encoder and case transformer output the case embedding, and all task heads are discarded.

## Usage Guide

### Distributed training

The recommended way to launch Stage 2 is via `torchrun` with the provided shell script:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_stage2.sh
```

Override arguments on the command line:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_stage2.sh \
  --feature_dirs /data/tcga_features /data/cptac_features \
  --teacher_checkpoint ./results/stage1/checkpoints/teacher_step_14400.pt \
  --epochs 30 \
  --batch_size 1 \
  --grad_accumulation_steps 128 \
  --mixed_precision \
  --activation_checkpointing \
  --output_dir ./results/stage2_run \
  --wandb
```

Or call `torchrun` directly:

```bash
torchrun --nproc_per_node=8 --module moozy train stage2 \
  --feature_dirs /data/features \
  --teacher_checkpoint ./results/stage1/checkpoints/teacher_step_14400.pt \
  --epochs 30 \
  --mixed_precision \
  --output_dir ./results/stage2
```

Stage 2 uses per-case batch sizes (typically `--batch_size 1`) with large gradient accumulation to achieve an effective batch size of hundreds or thousands of cases. Because each case can contain variable numbers of tokens across multiple slides, memory usage varies per sample. Mixed precision and activation checkpointing are both recommended.

### Single-GPU training

```bash
CUDA_VISIBLE_DEVICES=0 python -m moozy train stage2 \
  --feature_dirs /data/features \
  --teacher_checkpoint ./results/stage1/checkpoints/teacher_step_14400.pt \
  --epochs 30 \
  --batch_size 1 \
  --grad_accumulation_steps 128 \
  --mixed_precision \
  --activation_checkpointing \
  --output_dir ./results/stage2_single
```

### Task definitions

Task supervision is loaded from a directory of per-task subdirectories. Each subdirectory contains a `task.csv` (mapping case IDs to labels or survival times) and a `config.yaml` (task type, class names, etc.). By default, when `--task_dir` is empty, the bundled tasks are downloaded from [HuggingFace](https://huggingface.co/AtlasAnalyticsLab/MOOZY). To use custom tasks:

```bash
torchrun --nproc_per_node=8 --module moozy train stage2 \
  --feature_dirs /data/features \
  --teacher_checkpoint ./results/stage1/checkpoints/teacher_step_14400.pt \
  --task_dir /data/my_tasks \
  --output_dir ./results/stage2_custom
```

### Training without a Stage 1 checkpoint

When `--teacher_checkpoint` is empty, the slide encoder is initialized from scratch using the architecture defined by `--encoder_variant`. This is useful for ablation experiments but will produce weaker representations than initializing from a pretrained Stage 1 teacher.

```bash
torchrun --nproc_per_node=8 --module moozy train stage2 \
  --feature_dirs /data/features \
  --encoder_variant base_half_depth \
  --epochs 30 \
  --output_dir ./results/stage2_scratch
```

## Arguments

### Data

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--feature_dirs` | str (list) | ✓ | - | One or more directories containing H5 feature files. Each directory is scanned recursively for `.h5` files. Unlike Stage 1, the feature paths here are matched against case IDs in the task CSVs to build labeled training samples. Slides not referenced by any task are ignored. |
| `--feature_h5_format` | str | ✗ | `auto` | H5 schema to use when reading feature files. `auto` detects the format from file contents. `atlaspatch` reads the AtlasPatch layout. `trident` reads the TRIDENT layout. Valid values: `auto`, `atlaspatch`, `trident`. |
| `--feature_h5_key` | str | ✗ | `""` | AtlasPatch-specific feature key under the `features/` group. Only relevant when the H5 file uses AtlasPatch layout. |
| `--batch_size` | int | ✗ | `1` | Micro batch size per GPU, in cases. Each case may contain multiple slides with variable token counts. A batch size of 1 is typical for Stage 2 because full-slide grids can be very large. Use `--grad_accumulation_steps` to increase the effective batch size. |
| `--num_workers` | int | ✗ | `4` | Number of DataLoader worker processes per GPU. |
| `--prefetch_factor` | int | ✗ | `2` | Number of batches each worker prefetches ahead of time. |
| `--lazy_feature_loading` / `--no_lazy_feature_loading` | flag | ✗ | `--no_lazy_feature_loading` | When enabled, slide feature grids are loaded from disk on demand rather than preloaded into memory at startup. |
| `--max_cached_slides` | int | ✗ | `0` | Maximum number of lazily-loaded slides to keep in an LRU cache per worker process. Only takes effect when `--lazy_feature_loading` is enabled. 0 disables caching. |

### Augmentation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--hflip_prob` | float | ✗ | `0.5` | Probability of applying a horizontal flip to each slide's full feature grid during training. Augmentations are applied independently per slide. Disabled during validation. |
| `--vflip_prob` | float | ✗ | `0.5` | Probability of applying a vertical flip. |
| `--rotate_prob` | float | ✗ | `0.5` | Probability of applying a 90/180/270-degree rotation. When triggered, one of the three angles is chosen uniformly at random. |
| `--token_dropout_ratio` | float | ✗ | `0.1` | Fraction of valid (tissue) tokens to randomly drop from each slide during training. Acts as a regularizer by forcing the encoder to be robust to missing patches. Disabled during validation. |
| `--train_token_cap_sampling` | str | ✗ | `random_stratified` | Sampling strategy when a training slide exceeds the hardware-adaptive token cap. `random_stratified` partitions valid tokens into equal-width spatial bins and samples one token per bin, preserving whole-slide spatial coverage. `deterministic` always selects the same tokens for a given slide (useful for debugging). The token cap itself is computed automatically from GPU VRAM. Valid values: `random_stratified`, `deterministic`. |

### Task Supervision

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--task_dir` | path | ✗ | `""` (auto-download) | Directory containing task subdirectories, each with a `task.csv` and `config.yaml`. When empty, the bundled task definitions are downloaded from [HuggingFace](https://huggingface.co/AtlasAnalyticsLab/MOOZY) (`AtlasAnalyticsLab/MOOZY`). |
| `--classification_head_type` | str | ✗ | `mlp` | Head architecture for classification tasks. `linear` uses a single linear layer. `mlp` uses a 3-layer MLP with LayerNorm, GELU, and internal dropout (0.25). Valid values: `linear`, `mlp`. |
| `--survival_head_type` | str | ✗ | `mlp` | Head architecture for survival tasks. Same options as `--classification_head_type`. Each survival head outputs one logit per discrete time bin. |
| `--label_smoothing` | float | ✗ | `0.03` | Label smoothing coefficient for classification cross-entropy loss. Redistributes probability mass from the ground-truth class to all classes. 0 disables smoothing. |
| `--survival_num_bins` | int | ✗ | `8` | Target number of discrete time bins per survival task. The actual bin count adapts per task based on the number of observed events in the training set (see `--survival_min_bins` and `--survival_max_bins`). Bin edges are placed at equal quantiles of the training event-time distribution. When event times are tied, duplicate edges are merged, so the effective count may be lower. |
| `--survival_min_bins` | int | ✗ | `2` | Minimum number of time bins per survival task. Tasks with very few events are clamped to this lower bound. |
| `--survival_max_bins` | int | ✗ | `16` | Maximum number of time bins per survival task. The bin count is capped at this upper bound regardless of event count. |
| `--head_dropout` | float | ✗ | `0.1` | Dropout applied to the case embedding before it is passed to task heads. Acts as regularization at the interface between the case transformer and the prediction heads. |

### Slide Encoder

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--teacher_checkpoint` | path | ✗ | `""` | Path to a teacher-only checkpoint exported during Stage 1 (e.g., `teacher_step_14400.pt`). The checkpoint must contain `teacher_slide_encoder` (state dict) and `meta` (architecture config). When provided, the slide encoder is rebuilt from the checkpoint and initialized with the pretrained weights. When empty, the encoder is initialized from scratch using `--encoder_variant`. |
| `--encoder_variant` | str | ✗ | `base_half_depth` | Slide encoder architecture preset. Only used when `--teacher_checkpoint` is empty. Same variant table as Stage 1: `tiny`, `small`, `base`, `base_half_depth`, `base_quarter_depth`, `large`, `large_half_depth`, `large_quarter_depth`. |
| `--num_registers` | int | ✗ | `4` | Number of register tokens in the slide encoder. Only used when `--teacher_checkpoint` is empty (when loading from checkpoint, registers are determined by the checkpoint). |
| `--dropout` | float | ✗ | `0.1` | MLP dropout inside slide encoder transformer blocks. When loading from a teacher checkpoint, this value overrides the dropout stored in the checkpoint, allowing different regularization during Stage 2. |
| `--attn_dropout` | float | ✗ | `0.0` | Attention dropout probability in the slide encoder. |
| `--layer_drop` | float | ✗ | `0.1` | Stochastic depth rate for the slide encoder. |
| `--qk_norm` / `--no_qk_norm` | flag | ✗ | `--no_qk_norm` | Per-head LayerNorm on Q/K projections in the slide encoder. |
| `--layerscale_init` | float | ✗ | `0.0` | Initial LayerScale gamma for the slide encoder. 0 or negative disables LayerScale. |
| `--learnable_alibi` / `--no_learnable_alibi` | flag | ✗ | `--no_learnable_alibi` | Make ALiBi slopes trainable in the slide encoder. |
| `--freeze_slide_encoder` / `--no_freeze_slide_encoder` | flag | ✗ | `--no_freeze_slide_encoder` | Freeze all slide encoder parameters during Stage 2. The encoder runs in eval mode and only the case transformer and task heads are trained. Useful for probing experiments. |

### Case Transformer

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--case_transformer_variant` | str | ✗ | `base_quarter_depth` | Architecture preset for the case transformer. Uses the same variant table as the slide encoder to determine the number of layers, heads, and feed-forward dimension. The `d_model` is inherited from the slide encoder (not from the variant table). The variant must have `n_heads` that divides the slide encoder's `d_model`. |
| `--case_transformer_dropout` | float | ✗ | `0.1` | Dropout rate inside the case transformer's attention and MLP sub-layers. |
| `--case_transformer_layerscale_init` | float | ✗ | `1e-5` | Initial LayerScale gamma for case transformer blocks. The case transformer uses LayerScale by default (unlike the slide encoder default). |
| `--case_transformer_layer_drop` | float | ✗ | `0.0` | Stochastic depth rate for the case transformer. Drop rates are linearly interpolated across layers. |
| `--case_transformer_qk_norm` / `--no_case_transformer_qk_norm` | flag | ✗ | `--no_case_transformer_qk_norm` | Per-head LayerNorm on Q/K in the case transformer. |
| `--case_transformer_num_registers` | int | ✗ | `0` | Number of register tokens for the case transformer. These are separate from the slide encoder's registers. 0 disables them. |

### Optimization

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--epochs` | int | ✗ | `30` | Number of training epochs. The total optimizer step count is `epochs * ceil(len(train_loader) / grad_accumulation_steps)`. |
| `--optimizer` | str | ✗ | `adamw` | Optimizer algorithm. Valid values: `adamw`, `adam`, `sgd`. |
| `--lr` | float | ✗ | `5e-5` | Base learning rate before linear scaling. The effective learning rate is `lr * (effective_global_batch / lr_base_batch_size)`. With `batch_size=1`, 8 GPUs, and `grad_accumulation_steps=128`, the effective batch is 1024, so the scaled LR is `5e-5 * 1024/256 = 2e-4`. |
| `--lr_min` | float | ✗ | `2e-7` | Minimum learning rate at the end of the decay schedule. |
| `--lr_schedule` | str | ✗ | `cosine` | Learning rate schedule. Cosine annealing from the scaled LR to `--lr_min` over the total training steps. Valid values: `linear`, `cosine`. |
| `--warmup_steps` | int | ✗ | `0` | Number of linear LR warmup steps. 0 disables warmup (training starts at the scaled LR). |
| `--lr_base_batch_size` | int | ✗ | `256` | Reference global batch size for linear LR scaling. The learning rate is multiplied by `effective_global_batch / lr_base_batch_size`. |
| `--weight_decay` | float | ✗ | `0.4` | Weight decay. Applied as a fixed value throughout training (Stage 2 does not use a weight-decay schedule). Bias and normalization parameters are exempt. |
| `--grad_clip` | float | ✗ | `0.3` | Per-parameter gradient clipping max norm. Applied to all trainable parameters. 0 disables clipping. |
| `--grad_accumulation_steps` | int | ✗ | `1` | Number of micro-batches to accumulate before each optimizer step. With `batch_size=1` and 8 GPUs, setting this to 128 yields an effective batch of 1024 cases. |
| `--mixed_precision` / `--no_mixed_precision` | flag | ✗ | `--no_mixed_precision` | Enable bfloat16 automatic mixed precision. Reduces GPU memory and enables a higher hardware-adaptive token cap per slide. |
| `--activation_checkpointing` / `--no_activation_checkpointing` | flag | ✗ | `--no_activation_checkpointing` | Enable gradient checkpointing in both the slide encoder and the case transformer. Trades compute for memory by recomputing intermediate activations during the backward pass. Recommended for large slides. |

### Checkpointing

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--save_every_epochs` | int | ✗ | `1` | Save a checkpoint after every N epochs. Each checkpoint contains the slide encoder, case transformer, and task head state dicts, plus architectural metadata needed for inference loading. 0 disables periodic saving. |
| `--keep_last_n` | int | ✗ | `50` | Maximum number of checkpoints to keep on disk. Older checkpoints are deleted after each save. |

### Logging and Validation

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--log_every` | int | ✗ | `50` | Log training metrics (loss, classification component, survival component, LR, weight decay) every N optimizer steps. |
| `--val_ratio` | float | ✗ | `0.05` | Fraction of cases to hold out for validation. The split is stratified by task labels and survival events to ensure representation across all tasks. Set to 0 to disable validation. |
| `--wandb` / `--no_wandb` | flag | ✗ | `--no_wandb` | Enable Weights & Biases logging. Training and validation metrics, per-task accuracy, and per-task c-index (survival) are logged each epoch. |
| `--wandb_project` | str | ✗ | `moozy` | Weights & Biases project name. |
| `--wandb_tags` | str | ✗ | `""` | Space or comma-separated tags for the Weights & Biases run. |

### Runtime

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output_dir` | path | ✗ | `outputs/moozy_stage2` | Output directory for logs, hyperparameters, and checkpoints. Created automatically if it does not exist. |
| `--backend` | str | ✗ | `nccl` | Distributed communication backend. Valid values: `nccl`, `gloo`. Use `nccl` for GPU training. |
| `--seed` | int | ✗ | `42` | Random seed for reproducibility. Controls dataset splitting, augmentation sampling, and weight initialization. |
| `--debug` / `--no_debug` | flag | ✗ | `--no_debug` | Debug mode. Truncates the dataset to the first 5 cases for quick iteration and pipeline testing. |
