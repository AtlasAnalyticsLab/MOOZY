import torch
import torch.nn as nn

from moozy.data.stage2 import CaseBatch


def _stack_optional_case_tensors(
    values: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if not values:
        return None
    moved = [v.to(device, non_blocking=True) if v.device != device else v for v in values]
    return torch.stack(moved, dim=0)


def encode_case_batch(
    slide_encoder: nn.Module,
    case_transformer: nn.Module,
    batch: CaseBatch,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Encode the canonical case-batch representation into case embeddings."""
    device = next(slide_encoder.parameters()).device
    cases = batch["cases"]
    case_cls = []
    case_labels = []
    case_events = []
    case_times = []

    for case in cases:
        slides = case["slides"]
        if not slides:
            continue
        slide_cls = []
        for slide in slides:
            x = slide["x"].to(device, non_blocking=True).unsqueeze(0)
            invalid = slide.get("invalid")
            if invalid is not None:
                invalid = invalid.to(device, non_blocking=True).unsqueeze(0)
            coords = slide.get("coords")
            if coords is not None:
                coords = coords.to(device, non_blocking=True).unsqueeze(0)
            patch_sizes = slide.get("patch_size")
            if patch_sizes is not None and patch_sizes.dim() == 0:
                patch_sizes = patch_sizes.to(device, non_blocking=True).unsqueeze(0)
            elif patch_sizes is not None:
                patch_sizes = patch_sizes.to(device, non_blocking=True)
            cls_slide, _, _ = slide_encoder(
                x,
                mask=None,
                invalid_mask=invalid,
                coords_xy=coords,
                patch_sizes=patch_sizes,
            )
            slide_cls.append(cls_slide.squeeze(0))
        slide_tokens = torch.stack(slide_cls, dim=0)
        case_cls.append(case_transformer(slide_tokens))
        if case.get("task_labels") is not None:
            case_labels.append(case["task_labels"])
        if case.get("task_events") is not None:
            case_events.append(case["task_events"])
        if case.get("task_times") is not None:
            case_times.append(case["task_times"])

    if not case_cls:
        raise ValueError("Empty case batch; no slides were provided.")

    cls_out = torch.stack(case_cls, dim=0)
    task_labels = _stack_optional_case_tensors(case_labels, device=device)
    task_events = (
        _stack_optional_case_tensors(case_events, device=device)
        if case_events and len(case_events) == len(case_cls)
        else None
    )
    task_times = (
        _stack_optional_case_tensors(case_times, device=device)
        if case_times and len(case_times) == len(case_cls)
        else None
    )
    sample_count = torch.tensor(len(case_cls), device=cls_out.device)
    return cls_out, task_labels, task_events, task_times, sample_count
