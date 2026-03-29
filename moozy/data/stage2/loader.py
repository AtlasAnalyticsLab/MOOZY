from torch.utils.data import DataLoader, Dataset

from .batching import collate_stage2_batch


def build_stage2_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    sampler=None,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    worker_init_fn=None,
) -> DataLoader:
    """Build the stage-2 supervised dataloader with the canonical collate function."""
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_stage2_batch,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**dataloader_kwargs)
