import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def update_teacher_centers(
    center_cls: torch.Tensor,
    center_patch: torch.Tensor,
    cls_features: torch.Tensor,
    patch_features: torch.Tensor,
    center_momentum: float,
) -> None:
    """Update teacher centers via distributed momentum averages."""
    device = cls_features.device
    cls_sum = cls_features.float().sum(dim=0, keepdim=True)
    cls_count = torch.tensor([cls_features.shape[0]], device=device, dtype=torch.float32)

    patch_sum = patch_features.float().mean(dim=1).sum(dim=0, keepdim=True)
    patch_count = torch.tensor([patch_features.shape[0]], device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(cls_sum)
        dist.all_reduce(cls_count)
        dist.all_reduce(patch_sum)
        dist.all_reduce(patch_count)

    mean_cls = cls_sum / cls_count.clamp(min=1.0)
    mean_patch = patch_sum / patch_count.clamp(min=1.0)

    center_cls.mul_(center_momentum).add_(mean_cls, alpha=1 - center_momentum)
    center_patch.mul_(center_momentum).add_(mean_patch, alpha=1 - center_momentum)


def compute_cls_distillation_loss_terms(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    tau_student: float,
    tau_teacher: float,
    skip_same_view: bool = False,
) -> tuple[torch.Tensor, int]:
    """Return the accumulated CLS distillation loss and number of active pairs."""
    _, num_student_views, num_classes = student_logits.shape
    _, num_teacher_views, num_classes_2 = teacher_logits.shape
    if num_student_views == 0 or num_teacher_views == 0:
        return torch.tensor(0.0, device=student_logits.device), 0
    if num_classes != num_classes_2:
        raise ValueError("Student/teacher CLS dims must match.")

    p_t = F.softmax(teacher_logits / float(tau_teacher), dim=-1)
    q_s = F.log_softmax(student_logits / float(tau_student), dim=-1)

    loss_sum = torch.tensor(0.0, device=student_logits.device)
    num_terms = 0
    for teacher_view in range(num_teacher_views):
        for student_view in range(num_student_views):
            if skip_same_view and num_student_views == num_teacher_views and student_view == teacher_view:
                continue
            loss_sum = loss_sum + (-(p_t[:, teacher_view, :] * q_s[:, student_view, :]).sum(dim=-1).mean())
            num_terms += 1

    return loss_sum, num_terms


def compute_masked_image_modeling_loss(
    student_patch_logits: torch.Tensor,
    teacher_patch_logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    tau_student: float,
    tau_teacher_patch: float,
) -> torch.Tensor:
    """Masked image modeling loss over masked tokens only."""
    batch_crops = student_patch_logits.shape[0]
    if masks.ndim == 3:
        crop_h, crop_w = masks.shape[-2:]
        num_tokens = crop_h * crop_w
        masks_flat = masks.reshape(batch_crops, num_tokens)
    elif masks.ndim == 2:
        masks_flat = masks
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    teacher_probs = F.softmax(teacher_patch_logits / float(tau_teacher_patch), dim=-1)
    student_log_probs = F.log_softmax(student_patch_logits / float(tau_student), dim=-1)

    loss_per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
    masked = masks_flat.to(dtype=loss_per_token.dtype)
    masked_counts = masked.sum(dim=1)
    loss_per_crop = (loss_per_token * masked).sum(dim=1) / masked_counts.clamp(min=1.0)

    valid_crops = masked_counts > 0
    if valid_crops.any():
        return loss_per_crop[valid_crops].mean()
    return torch.tensor(0.0, device=student_patch_logits.device)
