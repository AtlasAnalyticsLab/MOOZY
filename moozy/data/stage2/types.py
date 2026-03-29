from typing import TypedDict

import numpy as np
import torch


class SlideSample(TypedDict, total=False):
    name: str
    x: torch.Tensor
    invalid: torch.Tensor
    coords: torch.Tensor
    patch_size: torch.Tensor
    center: np.ndarray
    grid_h: int
    grid_w: int
    patch_size_level0: int
    patch_size_value: int


class CaseRecord(TypedDict, total=False):
    case_id: str
    paths: list[str]
    path_options: list[list[str]]


class CaseSample(TypedDict, total=False):
    case_id: str
    slides: list[SlideSample]
    task_labels: torch.Tensor
    task_events: torch.Tensor
    task_times: torch.Tensor


class CaseBatch(TypedDict):
    cases: list[CaseSample]
