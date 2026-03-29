import logging
import math
from typing import Sequence

import torch

INFERENCE_TOKEN_PRESETS: Sequence[tuple[float, int]] = (
    (16.0, 12000),
    (20.0, 13500),
    (24.0, 14500),
    (40.0, 19000),
    (48.0, 21000),
    (80.0, 27000),
)

TRAINING_TOKEN_PRESETS_FP32: Sequence[tuple[float, int]] = (
    (16.0, 5500),
    (20.0, 6500),
    (24.0, 7000),
    (40.0, 9500),
    (48.0, 10500),
    (80.0, 14000),
)

TRAINING_TOKEN_PRESETS_BF16: Sequence[tuple[float, int]] = (
    (16.0, 7000),
    (20.0, 8500),
    (24.0, 9000),
    (40.0, 12000),
    (48.0, 13500),
    (80.0, 17500),
)


def interpolate_from_vram_presets(
    total_vram_gib: float,
    presets: Sequence[tuple[float, int]],
) -> int:
    """Interpolate a max-token cap from calibrated VRAM presets."""
    points = sorted((float(gb), int(tokens)) for gb, tokens in presets)
    if not points:
        return 0

    total_vram_gib = float(max(0.0, total_vram_gib))
    first_gb, first_tokens = points[0]
    if total_vram_gib <= first_gb:
        scaled = first_tokens * (total_vram_gib / max(first_gb, 1e-6))
        return max(1, int(round(scaled)))

    for (lo_gb, lo_tokens), (hi_gb, hi_tokens) in zip(points[:-1], points[1:]):
        if total_vram_gib <= hi_gb:
            alpha = (total_vram_gib - lo_gb) / max(1e-6, hi_gb - lo_gb)
            interp = lo_tokens + alpha * (hi_tokens - lo_tokens)
            return max(1, int(round(interp)))

    last_gb, last_tokens = points[-1]
    extrap = last_tokens * math.sqrt(total_vram_gib / max(last_gb, 1e-6))
    return max(1, int(round(extrap)))


def resolve_vram_token_cap(
    *,
    presets: Sequence[tuple[float, int]],
    logger: logging.Logger | None = None,
    device: torch.device | None = None,
    local_rank: int | None = None,
) -> int:
    """Auto-detect GPU VRAM and return a max-token cap from calibrated presets."""
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        if logger is not None:
            logger.warning("CUDA is unavailable; disabling token cap.")
        return 0

    if device is not None:
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    else:
        device_index = int(max(0, local_rank or 0))

    props = torch.cuda.get_device_properties(device_index)
    total_vram_gib = float(props.total_memory) / float(1024**3)
    cap = interpolate_from_vram_presets(total_vram_gib, presets)

    if logger is not None:
        logger.info(
            "Token cap for GPU %d (%s, %.1f GiB): max_valid_tokens_per_slide=%d.",
            device_index,
            str(props.name),
            total_vram_gib,
            cap,
        )
    return cap
