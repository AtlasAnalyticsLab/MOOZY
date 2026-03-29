import random

import numpy as np
import torch


def sample_crop(
    grid: np.ndarray,
    crop_size: int,
    valid_grid_mask: np.ndarray | None = None,
    x_axis: np.ndarray | None = None,
    y_axis: np.ndarray | None = None,
    min_valid_ratio: float = 0.0,
    max_attempts: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random crop from a slide feature grid."""
    actual_grid_h, actual_grid_w = grid.shape[:2]
    max_h = actual_grid_h - crop_size + 1
    max_w = actual_grid_w - crop_size + 1

    if max_h <= 0 or max_w <= 0:
        raise ValueError(f"Crop size {crop_size} exceeds grid size ({actual_grid_h}, {actual_grid_w})")

    best_candidate = None
    chosen = None
    for _ in range(max(1, max_attempts)):
        h_start = random.randint(0, max_h - 1)
        w_start = random.randint(0, max_w - 1)
        crop = grid[h_start : h_start + crop_size, w_start : w_start + crop_size, :]

        if valid_grid_mask is not None:
            valid_mask_np = valid_grid_mask[h_start : h_start + crop_size, w_start : w_start + crop_size]
        else:
            valid_mask_np = np.ones((crop_size, crop_size), dtype=bool)

        valid_ratio = float(valid_mask_np.mean())
        meets_ratio = (min_valid_ratio <= 0.0) or (valid_ratio >= min_valid_ratio)
        if meets_ratio:
            chosen = (h_start, w_start, crop, valid_mask_np)
            break

        if best_candidate is None or valid_ratio > best_candidate[0]:
            best_candidate = (valid_ratio, h_start, w_start, crop, valid_mask_np)

    if chosen is not None:
        h_start, w_start, crop, valid_mask_np = chosen
    else:
        _, h_start, w_start, crop, valid_mask_np = best_candidate

    mask = torch.zeros(crop_size, crop_size, dtype=torch.bool)

    if x_axis is not None and y_axis is not None:
        cx = x_axis[w_start : w_start + crop_size]
        cy = y_axis[h_start : h_start + crop_size]
        xx, yy = np.meshgrid(cx, cy, indexing="xy")
        coords_np = np.stack([xx, yy], axis=-1).astype(np.int64)
    else:
        coords_np = np.zeros((crop_size, crop_size, 2), dtype=np.int64)

    crop_tensor = torch.from_numpy(crop).float()
    valid_tensor = torch.from_numpy(valid_mask_np.astype(np.bool_))
    coords_tensor = torch.from_numpy(coords_np.astype(np.int64))
    return crop_tensor, mask, valid_tensor, coords_tensor
