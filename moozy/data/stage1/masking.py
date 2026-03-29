import math

import torch


class BlockMaskGenerator:
    """Batch-level block masking with exact token budget."""

    def __init__(
        self,
        window_h: int,
        window_w: int,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        min_num_patches: int = 4,
        max_num_patches: int | None = None,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ):
        self.window_h = int(window_h)
        self.window_w = int(window_w)
        self.num_patches = self.window_h * self.window_w
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.min_num_patches = int(min_num_patches)
        self.max_num_patches = None if max_num_patches is None else int(max_num_patches)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect) if max_aspect is not None else 1.0 / self.min_aspect
        self.log_aspect_min = math.log(self.min_aspect)
        self.log_aspect_max = math.log(self.max_aspect)

    def _mask(self, mask: torch.Tensor, max_mask_patches: int) -> int:
        if max_mask_patches <= 0:
            return 0
        max_mask_patches = int(max_mask_patches)
        min_patches = min(self.min_num_patches, max_mask_patches)
        delta = 0
        for _ in range(10):
            target_area = float(torch.empty(1).uniform_(min_patches, max_mask_patches).item())
            aspect_ratio = float(torch.empty(1).uniform_(self.log_aspect_min, self.log_aspect_max).exp().item())
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if h < 1 or w < 1:
                continue
            if w < self.window_w and h < self.window_h:
                top = int(torch.randint(0, self.window_h - h + 1, (1,)).item())
                left = int(torch.randint(0, self.window_w - w + 1, (1,)).item())
                region = mask[top : top + h, left : left + w]
                num_masked = int(region.sum().item())
                if 0 < h * w - num_masked <= max_mask_patches:
                    newly_masked = (~region).sum().item()
                    region[:] = True
                    delta += int(newly_masked)
                    break
        return delta

    def __call__(self, num_masking_patches: int | None = None) -> torch.Tensor:
        """
        Generate a mask.

        If `num_masking_patches` is provided, mask exactly that many tokens
        (clamped to the number of patches) using block sampling, then fill any
        leftover with random tokens. Otherwise, sample a target ratio uniformly
        in [mask_ratio_min, mask_ratio_max] and mask to that budget.
        """
        if num_masking_patches is None:
            ratio = float(torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item())
            target = int(round(ratio * self.num_patches))
        else:
            target = int(num_masking_patches)
        target = max(0, min(target, self.num_patches))
        if target == 0:
            return torch.zeros(self.window_h, self.window_w, dtype=torch.bool)

        mask = torch.zeros(self.window_h, self.window_w, dtype=torch.bool)
        max_per_block = self.max_num_patches if self.max_num_patches is not None else target
        mask_count = 0

        while mask_count < target:
            remaining = target - mask_count
            max_mask_patches = min(remaining, max_per_block)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta

        if mask_count < target:
            flat = mask.view(-1)
            available = (~flat).nonzero(as_tuple=False).squeeze(1)
            if available.numel() > 0:
                take = min(target - mask_count, available.numel())
                idx = available[torch.randperm(available.numel())[:take]]
                flat[idx] = True

        return mask.view(self.window_h, self.window_w)


def apply_mask_budget(
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_count: int,
) -> torch.Tensor:
    """Restrict a mask to valid positions and enforce an exact mask count."""
    mask = mask.to(dtype=torch.bool)
    valid_mask = valid_mask.to(dtype=torch.bool)
    mask_flat = mask.view(-1) & valid_mask.view(-1)
    valid_flat = valid_mask.view(-1)
    valid_total = int(valid_flat.sum().item())
    if valid_total == 0:
        return torch.zeros_like(mask_flat).view_as(mask)

    target = max(0, min(valid_total, int(target_count)))
    current = mask_flat.nonzero(as_tuple=False).squeeze(1)

    if current.numel() > target:
        drop = current[torch.randperm(current.numel())[: current.numel() - target]]
        mask_flat[drop] = False
    elif current.numel() < target:
        needed = target - current.numel()
        available = valid_flat.nonzero(as_tuple=False).squeeze(1)
        available = available[~mask_flat[available]]
        if available.numel() > 0:
            take = min(needed, available.numel())
            pick = available[torch.randperm(available.numel())[:take]]
            mask_flat[pick] = True

    return mask_flat.view_as(mask)
