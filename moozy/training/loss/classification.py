import torch
import torch.nn.functional as F


def compute_classification_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, bool]:
    """Compute one classification-task loss using the current missing-label contract."""
    if labels is None or labels.numel() == 0:
        return logits.sum() * 0.0, False

    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0.0, False

    loss = F.cross_entropy(
        logits[valid],
        labels[valid].long(),
        weight=weight,
        label_smoothing=float(label_smoothing),
    )
    return loss, True


def reduce_task_loss_components(
    loss_task_sum: torch.Tensor,
    loss_task_cls_sum: torch.Tensor,
    loss_task_surv_sum: torch.Tensor,
    *,
    task_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce accumulated task losses into the current output contract."""
    if task_count > 0:
        loss_task = loss_task_sum / task_count
        loss_cls_component = loss_task_cls_sum / task_count
        loss_surv_component = loss_task_surv_sum / task_count
    else:
        loss_task = loss_task_sum
        loss_cls_component = loss_task_sum * 0.0
        loss_surv_component = loss_task_sum * 0.0
    return loss_task, loss_cls_component, loss_surv_component
