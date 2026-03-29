import torch

from moozy.data.features.transforms import apply_grid_spatial_augmentation, sample_augmentation_params


def apply_random_crop_augmentations(
    crop: torch.Tensor,
    mask: torch.Tensor | None,
    valid: torch.Tensor | None,
    coords: torch.Tensor,
    *,
    hflip_prob: float,
    vflip_prob: float,
    rotate_prob: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Apply random hflip, vflip, and 90-degree rotations to one crop."""
    hflip, vflip, angle = sample_augmentation_params(
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rotate_prob=rotate_prob,
    )
    if not (hflip or vflip or angle):
        return crop, mask, valid, coords

    grids, coords_out = apply_grid_spatial_augmentation(
        [crop, mask, valid],
        coords,
        hflip=hflip,
        vflip=vflip,
        angle=angle,
    )
    return grids[0], grids[1], grids[2], coords_out
