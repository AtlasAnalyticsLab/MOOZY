from functools import partial

from torch.utils.data import DataLoader, Dataset

from .collate import collate_stage1_batch
from .masking import BlockMaskGenerator


def build_stage1_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    min_num_mask_patches: int = 4,
    max_num_mask_patches: int | None = None,
    mask_min_aspect: float = 0.3,
    mask_max_aspect: float | None = None,
    mask_sample_probability: float = 0.5,
    **kwargs,
) -> DataLoader:
    """Build the stage-1 SSL dataloader with batch-level block masking."""
    base_dataset = getattr(dataset, "dataset", dataset)
    mask_generator = BlockMaskGenerator(
        window_h=base_dataset.global_crop_size,
        window_w=base_dataset.global_crop_size,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        min_num_patches=min_num_mask_patches,
        max_num_patches=max_num_mask_patches,
        min_aspect=mask_min_aspect,
        max_aspect=mask_max_aspect,
    )
    collate_fn = partial(
        collate_stage1_batch,
        mask_generator=mask_generator,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        mask_sample_probability=mask_sample_probability,
    )

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs,
    )
    if num_workers is not None and num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**dataloader_kwargs)
