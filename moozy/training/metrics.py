import torch
import torch.nn.functional as F


@torch.no_grad()
def build_stage1_ssl_metrics(
    *,
    teacher_cls_logits: torch.Tensor,
    teacher_patch_logits: torch.Tensor,
    student_cls_logits: torch.Tensor,
    student_patch_logits: torch.Tensor,
    invalid_global_flat: torch.Tensor,
    masks_g: torch.Tensor,
    tau_teacher: float,
    tau_teacher_patch: float,
    tau_student: float,
    agreement: torch.Tensor | None = None,
) -> dict[str, float]:
    """Build the current stage-1 diagnostic metric payload."""
    eps = 1e-12
    metrics: dict[str, float] = {}

    def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        return (-(probs * probs.clamp_min(eps).log()).sum(dim=-1)).mean()

    t_cls_probs = F.softmax(teacher_cls_logits / float(tau_teacher), dim=-1)
    h_t_cls = entropy_from_probs(t_cls_probs)
    metrics["t_cls_entropy"] = float(h_t_cls.detach().item())
    metrics["t_cls_perplexity"] = float(torch.exp(h_t_cls).detach().item())

    t_top1_idx = teacher_cls_logits.argmax(dim=-1)
    proto_counts = torch.bincount(
        t_top1_idx,
        minlength=int(teacher_cls_logits.shape[-1]),
    ).to(dtype=torch.float32)
    total_assign = proto_counts.sum().clamp_min(1.0)
    max_frac = proto_counts.max() / total_assign
    proto_probs = proto_counts / total_assign
    eff_num = 1.0 / (torch.sum(proto_probs * proto_probs) + eps)
    metrics["t_cls_proto_max_frac"] = float(max_frac.detach().item())
    metrics["t_cls_proto_eff_num"] = float(eff_num.detach().item())
    metrics["t_cls_proto_nonzero"] = float((proto_counts > 0).sum().item())

    t_patch_probs = F.softmax(teacher_patch_logits / float(tau_teacher_patch), dim=-1)
    valid_g_flat = torch.logical_not(invalid_global_flat).reshape(teacher_patch_logits.shape[0], -1)
    masks_g_flat = masks_g.reshape(teacher_patch_logits.shape[0], -1)

    num_classes = t_patch_probs.shape[-1]
    t_flat = t_patch_probs.reshape(-1, num_classes)
    valid_flat = valid_g_flat.reshape(-1)
    masked_flat = masks_g_flat.reshape(-1)

    if valid_flat.any():
        t_valid = t_flat[valid_flat]
        h_t_patch_valid = entropy_from_probs(t_valid)
        metrics["t_patch_entropy_valid"] = float(h_t_patch_valid.detach().item())
        metrics["t_patch_perplexity_valid"] = float(torch.exp(h_t_patch_valid).detach().item())
    else:
        metrics["t_patch_entropy_valid"] = 0.0
        metrics["t_patch_perplexity_valid"] = 0.0

    mask_valid = valid_flat & masked_flat
    if mask_valid.any():
        t_masked = t_flat[mask_valid]
        h_t_patch_masked = entropy_from_probs(t_masked)
        metrics["t_patch_entropy_masked"] = float(h_t_patch_masked.detach().item())
        metrics["t_patch_perplexity_masked"] = float(torch.exp(h_t_patch_masked).detach().item())
    else:
        metrics["t_patch_entropy_masked"] = 0.0
        metrics["t_patch_perplexity_masked"] = 0.0

    if mask_valid.any():
        t_patch_logits_all = teacher_patch_logits.reshape(-1, num_classes)
        t_patch_top1 = t_patch_logits_all[mask_valid].argmax(dim=-1)
        patch_proto_counts = torch.bincount(
            t_patch_top1,
            minlength=int(num_classes),
        ).to(dtype=torch.float32)
        total_assign_patch = patch_proto_counts.sum().clamp_min(1.0)
        max_frac_patch = patch_proto_counts.max() / total_assign_patch
        patch_probs = patch_proto_counts / total_assign_patch
        eff_num_patch = 1.0 / (torch.sum(patch_probs * patch_probs) + eps)
        metrics["t_patch_proto_max_frac"] = float(max_frac_patch.detach().item())
        metrics["t_patch_proto_eff_num"] = float(eff_num_patch.detach().item())
        metrics["t_patch_proto_nonzero"] = float((patch_proto_counts > 0).sum().item())
    else:
        metrics["t_patch_proto_max_frac"] = 0.0
        metrics["t_patch_proto_eff_num"] = 0.0
        metrics["t_patch_proto_nonzero"] = 0.0

    s_cls_probs = F.softmax(student_cls_logits / float(tau_student), dim=-1)
    h_s_cls = entropy_from_probs(s_cls_probs)
    metrics["s_cls_entropy"] = float(h_s_cls.detach().item())
    metrics["s_cls_perplexity"] = float(torch.exp(h_s_cls).detach().item())

    s_patch_probs = F.softmax(student_patch_logits / float(tau_student), dim=-1)
    s_flat = s_patch_probs.reshape(-1, num_classes)
    if mask_valid.any():
        s_masked = s_flat[mask_valid]
        h_s_patch_masked = entropy_from_probs(s_masked)
        metrics["s_patch_entropy_masked"] = float(h_s_patch_masked.detach().item())
        metrics["s_patch_perplexity_masked"] = float(torch.exp(h_s_patch_masked).detach().item())
    else:
        metrics["s_patch_entropy_masked"] = 0.0
        metrics["s_patch_perplexity_masked"] = 0.0

    if agreement is not None:
        metrics["acc"] = float(agreement.detach().item())

    return metrics
