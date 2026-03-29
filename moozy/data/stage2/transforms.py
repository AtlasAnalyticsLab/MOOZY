import torch

from moozy.data.features.transforms import apply_grid_spatial_augmentation, sample_augmentation_params

from .types import SlideSample


def _drop_flat_tokens(slide: SlideSample, drop_mask: torch.Tensor) -> SlideSample:
    if not drop_mask.any():
        return slide

    x = slide["x"]
    invalid = slide["invalid"]

    flat_drop = drop_mask.reshape(-1).to(dtype=torch.bool)
    flat_invalid = invalid.reshape(-1).clone()
    flat_invalid[flat_drop] = True
    slide["invalid"] = flat_invalid.view_as(invalid)

    flat_x = x.reshape(-1, x.shape[-1]).clone()
    flat_x[flat_drop] = 0.0
    slide["x"] = flat_x.view_as(x)
    return slide


def apply_slide_augmentations(
    slide: SlideSample,
    hflip_prob: float,
    vflip_prob: float,
    rotate_prob: float,
) -> SlideSample:
    if hflip_prob <= 0.0 and vflip_prob <= 0.0 and rotate_prob <= 0.0:
        return slide

    x = slide["x"]
    coords = slide["coords"]
    invalid = slide.get("invalid")

    hflip, vflip, angle = sample_augmentation_params(
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rotate_prob=rotate_prob,
    )
    if not (hflip or vflip or angle):
        return slide

    grids, coords_out = apply_grid_spatial_augmentation(
        [x, invalid],
        coords,
        hflip=hflip,
        vflip=vflip,
        angle=angle,
    )
    slide["x"] = grids[0]
    if grids[1] is not None:
        slide["invalid"] = grids[1]
    slide["coords"] = coords_out.to(dtype=coords.dtype)
    return slide


def apply_token_dropout(
    slide: SlideSample,
    dropout_ratio: float,
) -> SlideSample:
    if dropout_ratio <= 0.0:
        return slide
    invalid = slide["invalid"]
    valid = ~invalid
    valid_count = int(valid.sum().item())
    if valid_count <= 1:
        return slide

    drop_count = round(valid_count * dropout_ratio)
    if drop_count <= 0:
        return slide
    drop_count = min(drop_count, valid_count - 1)

    flat_valid = valid.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    perm = torch.randperm(valid_indices.numel())
    drop_indices = valid_indices[perm[:drop_count]]
    drop_mask = torch.zeros_like(flat_valid)
    drop_mask[drop_indices] = True
    return _drop_flat_tokens(slide, drop_mask)


def apply_max_valid_tokens(
    slide: SlideSample,
    max_valid_tokens: int,
    sampling: str = "deterministic",
) -> SlideSample:
    if max_valid_tokens <= 0:
        return slide
    invalid = slide["invalid"]
    valid_flat = (~invalid).reshape(-1)
    valid_indices = valid_flat.nonzero(as_tuple=False).flatten()
    valid_count = int(valid_indices.numel())
    if valid_count <= max_valid_tokens:
        return slide

    sampling_mode = str(sampling).strip().lower()
    if sampling_mode == "deterministic":
        keep_positions = torch.div(
            torch.arange(max_valid_tokens, device=valid_indices.device, dtype=torch.long) * valid_count,
            max_valid_tokens,
            rounding_mode="floor",
        )
    elif sampling_mode == "random_stratified":
        starts = torch.div(
            torch.arange(max_valid_tokens, device=valid_indices.device, dtype=torch.long) * valid_count,
            max_valid_tokens,
            rounding_mode="floor",
        )
        ends = (
            torch.div(
                (torch.arange(max_valid_tokens, device=valid_indices.device, dtype=torch.long) + 1) * valid_count,
                max_valid_tokens,
                rounding_mode="floor",
            )
            - 1
        )
        widths = (ends - starts + 1).clamp(min=1)
        offsets = torch.floor(
            torch.rand(max_valid_tokens, device=valid_indices.device) * widths.to(dtype=torch.float32)
        ).to(dtype=torch.long)
        keep_positions = starts + offsets
    else:
        raise ValueError(
            f"Unsupported token cap sampling mode: {sampling!r}. "
            "Expected one of {'deterministic', 'random_stratified'}."
        )

    keep_indices = valid_indices.index_select(0, keep_positions)
    keep_mask = torch.zeros_like(valid_flat)
    keep_mask[keep_indices] = True
    drop_mask = valid_flat & (~keep_mask)
    return _drop_flat_tokens(slide, drop_mask)


def compact_slide_to_valid_tokens(
    slide: SlideSample,
) -> SlideSample:
    """Drop invalid/background tokens before model forward."""
    x = slide["x"]
    coords = slide["coords"]
    invalid = slide["invalid"]

    flat_valid = (~invalid).reshape(-1)
    keep_indices = flat_valid.nonzero(as_tuple=False).flatten()
    if int(keep_indices.numel()) <= 0:
        return slide

    flat_x = x.reshape(-1, x.shape[-1])
    flat_coords = coords.reshape(-1, coords.shape[-1])
    slide["x"] = flat_x.index_select(0, keep_indices).unsqueeze(1).contiguous()
    slide["coords"] = flat_coords.index_select(0, keep_indices).unsqueeze(1).contiguous()
    slide.pop("invalid", None)
    return slide
