import random

import torch

from .masking import BlockMaskGenerator, apply_mask_budget
from .types import Stage1Batch, Stage1Sample


def _stack_stage1_batch(batch: list[Stage1Sample]) -> Stage1Batch:
    payload: Stage1Batch = {
        "global_crops": torch.stack([item["global_crops"] for item in batch], dim=0),
        "local_crops": torch.stack([item["local_crops"] for item in batch], dim=0),
        "global_masks": torch.stack([item["global_masks"] for item in batch], dim=0),
        "global_valids": torch.stack([item["global_valids"] for item in batch], dim=0),
        "local_valids": torch.stack([item["local_valids"] for item in batch], dim=0),
        "global_coords": torch.stack([item["global_coords"] for item in batch], dim=0),
        "local_coords": torch.stack([item["local_coords"] for item in batch], dim=0),
        "patch_sizes": torch.stack([item["patch_sizes"] for item in batch], dim=0),
    }
    return payload


def collate_stage1_batch(
    batch: list[Stage1Sample],
    *,
    mask_generator: BlockMaskGenerator,
    mask_ratio_min: float,
    mask_ratio_max: float,
    mask_sample_probability: float,
) -> Stage1Batch:
    """Collate a batch and generate block masks across global views."""
    payload = _stack_stage1_batch(batch)

    global_crops = payload["global_crops"]
    global_valids = payload["global_valids"]
    batch_size, num_global, h, w = global_crops.shape[:4]
    total_global = batch_size * num_global
    n_masked = int(total_global * mask_sample_probability)
    if n_masked > 0:
        probs = torch.linspace(mask_ratio_min, mask_ratio_max, steps=n_masked + 1)
        budgets = [float(probs[i + 1].item()) for i in range(n_masked)]
    else:
        budgets = []
    budgets.extend(0.0 for _ in range(total_global - n_masked))
    random.shuffle(budgets)

    masks_list = []
    valids_flat = global_valids.reshape(total_global, h, w)
    for idx in range(total_global):
        ratio = budgets[idx]
        valid_mask = valids_flat[idx]
        valid_count = int(valid_mask.sum().item())
        target = int(round(ratio * valid_count))
        if target > 0:
            mask = mask_generator(target)
        else:
            mask = torch.zeros(h, w, dtype=torch.bool)
        masks_list.append(apply_mask_budget(mask, valid_mask, target))

    payload["global_masks"] = torch.stack(masks_list).reshape(batch_size, num_global, h, w)
    return payload
