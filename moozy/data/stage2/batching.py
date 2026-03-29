from typing import Sequence

import torch

from .types import CaseBatch, CaseSample, SlideSample


def build_case_sample(
    *,
    slides: list[SlideSample],
    case_id: str | None = None,
    task_labels: torch.Tensor | None = None,
    task_events: torch.Tensor | None = None,
    task_times: torch.Tensor | None = None,
) -> CaseSample:
    """Build the canonical stage-2 case sample payload."""
    payload: CaseSample = {"slides": slides}
    if case_id is not None:
        payload["case_id"] = case_id
    if task_labels is not None:
        payload["task_labels"] = task_labels
    if task_events is not None:
        payload["task_events"] = task_events
    if task_times is not None:
        payload["task_times"] = task_times
    return payload


def collate_stage2_batch(cases: Sequence[CaseSample]) -> CaseBatch:
    """Collate a list of CaseSamples into the canonical stage-2 batch payload."""
    return {"cases": list(cases)}
