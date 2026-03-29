import copy

import torch
import torch.nn as nn

from moozy.data.stage1 import Stage1Batch
from moozy.training.loss import (
    compute_cls_distillation_loss_terms,
    compute_masked_image_modeling_loss,
    update_teacher_centers,
)
from moozy.training.metrics import build_stage1_ssl_metrics

from .heads import ProjectionHead
from .moozy_slide_encoder import MOOZYSlideEncoder
from .stage1_encoding import (
    encode_stage1_student_views,
    encode_stage1_teacher_global_views,
    prepare_stage1_batch_views,
)


class MOOZYSSLModel(nn.Module):
    """
    Self-supervised MOOZY pretraining module.

    Combines student-teacher distillation with masked image modeling (MIM).
    The teacher is an EMA of the student.
    """

    def __init__(
        self,
        feat_dim: int = 384,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        dim_feedforward: int = 3072,
        num_registers: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        output_dim: int = 8192,
        ema_momentum: float = 0.999,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        teacher_patch_temp: float = 0.07,
        center_momentum: float = 0.9,
        layer_drop: float = 0.0,
        qk_norm: bool = True,
        layerscale_init: float = 0.0,
        learnable_alibi: bool = False,
        proj_hidden_dim: int = 2048,
        proj_bottleneck_dim: int = 256,
        proj_norm_last_layer: bool = True,
        proj_norm: str = "none",
        proj_last_norm: str = "none",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.ema_momentum = ema_momentum
        self.tau_student = student_temp
        self.tau_teacher = teacher_temp
        self.tau_teacher_patch = teacher_patch_temp
        self.center_momentum = center_momentum
        self.n_layers = n_layers
        self.num_registers = max(0, int(num_registers))
        self.learnable_alibi = bool(learnable_alibi)

        self.student_slide_encoder = MOOZYSlideEncoder(
            feat_dim=feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            num_registers=self.num_registers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            layer_drop=layer_drop,
            qk_norm=qk_norm,
            layerscale_init=layerscale_init,
            learnable_alibi=self.learnable_alibi,
        )
        self.student_head = ProjectionHead(
            d_model=d_model,
            d_hidden=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            output_dim=output_dim,
            norm_last_layer=proj_norm_last_layer,
            norm_type=proj_norm,
            last_norm_type=proj_last_norm,
        )

        self.teacher_slide_encoder = copy.deepcopy(self.student_slide_encoder)
        self.teacher_head = ProjectionHead(
            d_model=d_model,
            d_hidden=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            output_dim=output_dim,
            norm_last_layer=proj_norm_last_layer,
            norm_type=proj_norm,
            last_norm_type=proj_last_norm,
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_slide_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.register_buffer("center_cls", torch.zeros(1, output_dim))
        self.register_buffer("center_patch", torch.zeros(1, output_dim))

    def train(self, mode: bool = True):
        """Ensure teacher stays in eval mode regardless of student mode."""
        super().train(mode)
        self.teacher_slide_encoder.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Update teacher weights via EMA."""
        for student_p, teacher_p in zip(
            self.student_slide_encoder.parameters(),
            self.teacher_slide_encoder.parameters(),
        ):
            teacher_p.data = teacher_p.data * momentum + student_p.data * (1 - momentum)

        for student_p, teacher_p in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters(),
        ):
            teacher_p.data = teacher_p.data * momentum + student_p.data * (1 - momentum)

    def forward(self, batch: Stage1Batch) -> dict[str, torch.Tensor]:
        """Forward pass expecting the canonical stage-1 training batch."""
        views = prepare_stage1_batch_views(batch)
        student_outputs = encode_stage1_student_views(
            self.student_slide_encoder,
            self.student_head,
            views,
        )

        with torch.no_grad():
            self.teacher_slide_encoder.eval()
            self.teacher_head.eval()
            teacher_outputs = encode_stage1_teacher_global_views(
                self.teacher_slide_encoder,
                self.teacher_head,
                views,
                center_cls=self.center_cls,
                center_patch=self.center_patch,
            )

        agreement = None
        if self.training and views.num_global > 1:
            with torch.no_grad():
                teacher_pred = teacher_outputs.cls_logits_raw.reshape(views.batch_size, views.num_global, -1)[
                    :, 0
                ].argmax(dim=-1)
                student_pred = student_outputs.global_views.cls_logits.reshape(
                    views.batch_size,
                    views.num_global,
                    -1,
                )[:, 1].argmax(dim=-1)
                agreement = (teacher_pred == student_pred).float().mean()

        student_cls_global = student_outputs.global_views.cls_logits.reshape(
            views.batch_size,
            views.num_global,
            -1,
        )
        teacher_cls_global = teacher_outputs.cls_logits.reshape(
            views.batch_size,
            views.num_global,
            -1,
        )
        loss_cls_sum, cls_terms = compute_cls_distillation_loss_terms(
            student_logits=student_cls_global,
            teacher_logits=teacher_cls_global,
            tau_student=self.tau_student,
            tau_teacher=self.tau_teacher,
            skip_same_view=True,
        )

        student_cls_local = student_outputs.local_views.cls_logits.reshape(
            views.batch_size,
            views.num_local,
            student_outputs.local_views.cls_logits.shape[-1],
        )
        loss_local_sum, local_terms = compute_cls_distillation_loss_terms(
            student_logits=student_cls_local,
            teacher_logits=teacher_cls_global,
            tau_student=self.tau_student,
            tau_teacher=self.tau_teacher,
            skip_same_view=False,
        )
        loss_cls_sum = loss_cls_sum + loss_local_sum
        cls_terms += local_terms

        if cls_terms > 0:
            loss_cls = loss_cls_sum / cls_terms
        else:
            loss_cls = torch.tensor(0.0, device=student_outputs.global_views.cls_logits.device)

        n_global_tokens = views.num_global_tokens
        masks_g = torch.logical_and(
            views.global_masks_flat.reshape(views.batch_size * views.num_global, n_global_tokens),
            torch.logical_not(views.invalid_global_flat.reshape(views.batch_size * views.num_global, n_global_tokens)),
        )
        loss_mim = compute_masked_image_modeling_loss(
            student_outputs.global_views.patch_logits,
            teacher_outputs.patch_logits,
            masks_g,
            tau_student=self.tau_student,
            tau_teacher_patch=self.tau_teacher_patch,
        )

        loss_total = loss_cls + loss_mim

        if self.training:
            with torch.no_grad():
                update_teacher_centers(
                    center_cls=self.center_cls,
                    center_patch=self.center_patch,
                    cls_features=teacher_outputs.cls_logits_raw.reshape(
                        views.batch_size * views.num_global,
                        -1,
                    ),
                    patch_features=teacher_outputs.patch_logits_raw,
                    center_momentum=self.center_momentum,
                )

        metrics = build_stage1_ssl_metrics(
            teacher_cls_logits=teacher_outputs.cls_logits,
            teacher_patch_logits=teacher_outputs.patch_logits,
            student_cls_logits=student_outputs.global_views.cls_logits,
            student_patch_logits=student_outputs.global_views.patch_logits,
            invalid_global_flat=views.invalid_global_flat,
            masks_g=masks_g,
            tau_teacher=self.tau_teacher,
            tau_teacher_patch=self.tau_teacher_patch,
            tau_student=self.tau_student,
            agreement=agreement,
        )

        return {
            "loss_cls": loss_cls,
            "loss_mim": loss_mim,
            "loss_total": loss_total,
            "metrics": metrics,
        }
