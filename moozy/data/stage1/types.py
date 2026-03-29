from typing import TypedDict

import torch


class Stage1Sample(TypedDict):
    global_crops: torch.Tensor
    local_crops: torch.Tensor
    global_masks: torch.Tensor
    global_valids: torch.Tensor
    local_valids: torch.Tensor
    global_coords: torch.Tensor
    local_coords: torch.Tensor
    patch_sizes: torch.Tensor


class Stage1Batch(TypedDict):
    global_crops: torch.Tensor
    local_crops: torch.Tensor
    global_masks: torch.Tensor
    global_valids: torch.Tensor
    local_valids: torch.Tensor
    global_coords: torch.Tensor
    local_coords: torch.Tensor
    patch_sizes: torch.Tensor
