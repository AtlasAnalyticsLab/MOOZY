import random

import torch


def _compute_grid_center(
    coords: torch.Tensor,
) -> tuple[float, float]:
    coords_f = coords.to(dtype=torch.float32)
    cx = float(0.5 * (coords_f[..., 0].min() + coords_f[..., 0].max()))
    cy = float(0.5 * (coords_f[..., 1].min() + coords_f[..., 1].max()))
    return cx, cy


def _rotate_coords(
    coords_f: torch.Tensor,
    cx: float,
    cy: float,
    angle: int,
    k: int,
) -> torch.Tensor:
    dx = coords_f[..., 0] - cx
    dy = coords_f[..., 1] - cy
    if angle == 90:
        rot_x = cx - dy
        rot_y = cy + dx
    elif angle == 180:
        rot_x = cx - dx
        rot_y = cy - dy
    else:  # 270
        rot_x = cx + dy
        rot_y = cy - dx
    return torch.rot90(torch.stack([rot_x, rot_y], dim=-1), k=k, dims=[0, 1])


def apply_grid_spatial_augmentation(
    grids: list[torch.Tensor | None],
    coords: torch.Tensor,
    *,
    hflip: bool,
    vflip: bool,
    angle: int,
) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    """Apply hflip, vflip, and 90-degree rotation to a list of [H, W, ...] grids and coords [H, W, 2].

    Each non-None grid in `grids` must have spatial dimensions H and W at axes 0 and 1.
    Returns the transformed grids and updated integer coords.
    """
    if not (hflip or vflip or angle % 360):
        return grids, coords

    coords_f = coords.to(dtype=torch.float32)
    cx, cy = _compute_grid_center(coords_f)

    if hflip:
        coords_f = torch.stack([2 * cx - coords_f[..., 0], coords_f[..., 1]], dim=-1)
        coords_f = torch.flip(coords_f, dims=[1])
        grids = [torch.flip(g, dims=[1]) if g is not None else None for g in grids]

    if vflip:
        coords_f = torch.stack([coords_f[..., 0], 2 * cy - coords_f[..., 1]], dim=-1)
        coords_f = torch.flip(coords_f, dims=[0])
        grids = [torch.flip(g, dims=[0]) if g is not None else None for g in grids]

    if angle % 360:
        k = (angle // 90) % 4
        coords_f = _rotate_coords(coords_f, cx, cy, angle, k)
        grids = [torch.rot90(g, k=k, dims=[0, 1]) if g is not None else None for g in grids]

    return grids, torch.round(coords_f).to(torch.int64)


def sample_augmentation_params(
    *,
    hflip_prob: float,
    vflip_prob: float,
    rotate_prob: float,
) -> tuple[bool, bool, int]:
    """Sample hflip, vflip, and rotation angle given per-transform probabilities."""
    hflip = hflip_prob > 0.0 and random.random() < hflip_prob
    vflip = vflip_prob > 0.0 and random.random() < vflip_prob
    angle = random.choice([90, 180, 270]) if rotate_prob > 0.0 and random.random() < rotate_prob else 0
    return hflip, vflip, angle
