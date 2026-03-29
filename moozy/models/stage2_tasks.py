from typing import Sequence

import torch
import torch.nn as nn

from moozy.training.loss import (
    compute_classification_task_loss,
    compute_survival_task_loss,
    hazard_logits_to_risk_scores,
    reduce_task_loss_components,
)


def run_stage2_task_heads(
    *,
    cls_out: torch.Tensor,
    feature_dropout: nn.Module,
    task_heads: nn.ModuleDict,
    task_types: Sequence[str],
    task_head_types: Sequence[str],
    task_weights: Sequence[torch.Tensor | None],
    task_labels: torch.Tensor | None,
    task_events: torch.Tensor | None,
    task_times: torch.Tensor | None,
    label_smoothing: float,
    hazard_loss_fn,
    survival_bin_edges: dict[str, torch.Tensor | None],
) -> dict[str, object]:
    """Run stage-2 task heads and aggregate the current output contract."""
    logits_by_task: dict[str, torch.Tensor] = {}
    loss_task_sum = torch.tensor(0.0, device=cls_out.device)
    loss_task_cls_sum = torch.tensor(0.0, device=cls_out.device)
    loss_task_surv_sum = torch.tensor(0.0, device=cls_out.device)
    task_count = 0
    cls_task_count = 0
    surv_task_count = 0

    cls_in_linear = feature_dropout(cls_out)
    for idx, (key, head) in enumerate(task_heads.items()):
        task_type = task_types[idx] if idx < len(task_types) else "classification"
        task_head_type = task_head_types[idx] if idx < len(task_head_types) else "linear"
        cls_in = cls_in_linear if task_head_type == "linear" else cls_out
        logits = head(cls_in)

        if task_type == "survival":
            if task_events is not None and task_times is not None:
                events = task_events[:, idx]
                times = task_times[:, idx]
            else:
                events = None
                times = None

            hazard_logits = logits if logits.dim() == 2 else logits.unsqueeze(-1)
            logits_by_task[key] = hazard_logits_to_risk_scores(hazard_logits)
            loss_t, has_valid = compute_survival_task_loss(
                hazard_logits,
                events,
                times,
                hazard_loss_fn=hazard_loss_fn,
                bin_edges=survival_bin_edges.get(key),
            )
            loss_task_sum = loss_task_sum + loss_t
            if has_valid:
                loss_task_surv_sum = loss_task_surv_sum + loss_t
                task_count += 1
                surv_task_count += 1
        else:
            logits_by_task[key] = logits
            weight = task_weights[idx] if idx < len(task_weights) else None
            labels = task_labels[:, idx] if task_labels is not None and task_labels.numel() > 0 else None
            loss_t, has_valid = compute_classification_task_loss(
                logits,
                labels,
                weight=weight,
                label_smoothing=label_smoothing,
            )
            loss_task_sum = loss_task_sum + loss_t
            if has_valid:
                loss_task_cls_sum = loss_task_cls_sum + loss_t
                task_count += 1
                cls_task_count += 1

    loss_task, loss_cls_component, loss_surv_component = reduce_task_loss_components(
        loss_task_sum,
        loss_task_cls_sum,
        loss_task_surv_sum,
        task_count=task_count,
    )
    return {
        "logits": logits_by_task,
        "loss_task": loss_task,
        "loss_cls_component": loss_cls_component,
        "loss_surv_component": loss_surv_component,
        "task_count_cls": torch.tensor(float(cls_task_count), device=cls_out.device),
        "task_count_surv": torch.tensor(float(surv_task_count), device=cls_out.device),
    }
