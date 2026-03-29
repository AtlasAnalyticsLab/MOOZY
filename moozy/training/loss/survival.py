import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_survival_task_loss(
    hazard_logits: torch.Tensor,
    events: torch.Tensor | None,
    times: torch.Tensor | None,
    *,
    hazard_loss_fn,
    bin_edges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    """Compute one survival-task loss using the current missing-label contract."""
    if events is None or times is None or events.numel() == 0 or times.numel() == 0:
        return hazard_logits.sum() * 0.0, False

    valid = (events >= 0) & (times >= 0)
    if not valid.any():
        return hazard_logits.sum() * 0.0, False

    loss = hazard_loss_fn(
        hazard_logits[valid],
        events[valid],
        times[valid],
        bin_edges=bin_edges,
    )
    return loss, True


def _time_to_bin_index(times: torch.Tensor, bin_edges: torch.Tensor | None) -> torch.Tensor:
    """Map time values to discrete bin indices in [0, K-1]."""
    if bin_edges is None or not torch.is_tensor(bin_edges) or bin_edges.numel() == 0:
        return torch.zeros_like(times, dtype=torch.long)
    edges = bin_edges.to(device=times.device, dtype=times.dtype)
    return torch.bucketize(times, edges, right=False).long()


def hazard_logits_to_risk_scores(hazard_logits: torch.Tensor) -> torch.Tensor:
    """Convert per-bin hazard logits [N, K] to scalar risk scores [N]."""
    if hazard_logits.dim() != 2:
        raise ValueError(f"Expected hazard_logits shape [N, K], got {tuple(hazard_logits.shape)}")
    log_survival = torch.cumsum(F.logsigmoid(-hazard_logits), dim=1)
    return -log_survival[:, -1]


class DiscreteHazardLoss(nn.Module):
    """Negative log-likelihood for discrete-time survival hazards."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        hazard_logits: torch.Tensor,
        events: torch.Tensor,
        times: torch.Tensor,
        bin_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hazard_logits.dim() != 2:
            raise ValueError(f"Expected hazard_logits shape [N, K], got {tuple(hazard_logits.shape)}")
        if events.dim() != 1 or times.dim() != 1:
            raise ValueError(
                f"Expected events/times shape [N], got events={tuple(events.shape)} times={tuple(times.shape)}"
            )
        if hazard_logits.shape[0] != events.shape[0] or hazard_logits.shape[0] != times.shape[0]:
            raise ValueError(
                "Batch size mismatch for hazard loss: "
                f"logits={hazard_logits.shape[0]} events={events.shape[0]} times={times.shape[0]}"
            )
        if hazard_logits.numel() == 0:
            return hazard_logits.sum() * 0.0

        events_i64 = events.to(dtype=torch.long)
        times_fp = times.to(dtype=hazard_logits.dtype)
        bin_idx = _time_to_bin_index(times_fp, bin_edges).clamp_(0, hazard_logits.shape[1] - 1)

        log_h = F.logsigmoid(hazard_logits)
        log_1mh = F.logsigmoid(-hazard_logits)
        cum_log_surv = torch.cumsum(log_1mh, dim=1)

        idx = bin_idx.unsqueeze(1)
        log_h_at_idx = log_h.gather(1, idx).squeeze(1)
        log_surv_at_idx = cum_log_surv.gather(1, idx).squeeze(1)

        prev_idx = (bin_idx - 1).clamp(min=0).unsqueeze(1)
        prev_log_surv = torch.where(
            bin_idx > 0,
            cum_log_surv.gather(1, prev_idx).squeeze(1),
            torch.zeros_like(log_surv_at_idx),
        )

        event_mask = events_i64 == 1
        event_loss = -(prev_log_surv + log_h_at_idx)
        censored_loss = -log_surv_at_idx
        loss = torch.where(event_mask, event_loss, censored_loss)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_cindex(
    risk_scores: torch.Tensor,
    events: torch.Tensor,
    times: torch.Tensor,
) -> float:
    """Compute concordance index for survival predictions."""
    from sksurv.metrics import concordance_index_censored

    risk = risk_scores.detach().cpu().numpy()
    event = events.detach().cpu().numpy().astype(bool)
    time = times.detach().cpu().numpy()

    try:
        cindex, _, _, _, _ = concordance_index_censored(event, time, risk)
        return float(cindex)
    except Exception:
        return float("nan")
