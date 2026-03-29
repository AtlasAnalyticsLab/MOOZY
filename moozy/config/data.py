from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage1DataConfig:
    """Stage-1 self-supervised data pipeline parameters."""

    # Feature loading
    feature_dirs: list[str] = field(default_factory=list)
    feature_h5_format: str = "auto"
    feature_h5_key: str = ""
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 4
    lazy_feature_loading: bool = False
    max_cached_slides: int = 0

    # Cropping & masking
    global_crop_size: int = 20
    local_crop_size: int = 12
    num_global_crops: int = 2
    num_local_crops: int = 4
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    min_num_mask_patches: int = 4
    max_num_mask_patches: int = -1
    mask_min_aspect: float = 0.3
    mask_max_aspect: float | None = None
    mask_sample_probability: float = 0.5
    min_window_patch_ratio: float = 0.25
    crop_resample_attempts: int = 3

    # Augmentation
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_prob: float = 0.5


@dataclass(frozen=True)
class Stage2DataConfig:
    """Stage-2 supervised data pipeline parameters."""

    # Feature loading
    feature_dirs: list[str] = field(default_factory=list)
    feature_h5_format: str = "auto"
    feature_h5_key: str = ""
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    lazy_feature_loading: bool = False
    max_cached_slides: int = 0

    # Augmentation
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_prob: float = 0.5

    # Token capping
    token_dropout_ratio: float = 0.1
    train_token_cap_sampling: str = "random_stratified"
