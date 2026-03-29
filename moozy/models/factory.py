from typing import Any

import torch

from moozy.config.model import ProjectionConfig, SlideEncoderConfig
from moozy.config.training import SchedulerConfig

from .moozy_slide_encoder import MOOZYSlideEncoder
from .serialization import (
    build_slide_encoder_from_payload,
    build_slide_encoder_save_meta,
    load_checkpoint_payload,
)
from .stage1_ssl import MOOZYSSLModel
from .stage2_supervised import MOOZY
from .variants import ENCODER_VARIANTS


def build_ssl_model(
    slide_encoder: SlideEncoderConfig,
    projection: ProjectionConfig,
    scheduler: SchedulerConfig,
    feat_dim: int,
) -> MOOZYSSLModel:
    """Construct a Stage-1 SSL model from typed configs.

    ``slide_encoder`` must have d_model/n_heads/n_layers/dim_feedforward
    resolved (call :meth:`SlideEncoderConfig.resolve_variant` first).
    """
    se = slide_encoder
    if se.d_model is None or se.n_heads is None or se.n_layers is None or se.dim_feedforward is None:
        raise ValueError(
            "SlideEncoderConfig must have d_model/n_heads/n_layers/dim_feedforward resolved. "
            "Call .resolve_variant() first."
        )
    return MOOZYSSLModel(
        feat_dim=feat_dim,
        d_model=se.d_model,
        n_heads=se.n_heads,
        n_layers=se.n_layers,
        dim_feedforward=se.dim_feedforward,
        num_registers=se.num_registers,
        dropout=se.dropout,
        attn_dropout=se.attn_dropout,
        layer_drop=se.layer_drop,
        qk_norm=se.qk_norm,
        layerscale_init=se.layerscale_init,
        learnable_alibi=se.learnable_alibi,
        output_dim=projection.output_dim,
        proj_hidden_dim=projection.proj_hidden_dim,
        proj_bottleneck_dim=projection.proj_bottleneck_dim,
        proj_norm_last_layer=projection.proj_norm_last_layer,
        proj_norm=projection.proj_norm,
        proj_last_norm=projection.proj_last_norm,
        ema_momentum=scheduler.ema_momentum,
        student_temp=scheduler.student_temp,
        teacher_temp=scheduler.teacher_temp,
        teacher_patch_temp=scheduler.teacher_patch_temp,
        center_momentum=scheduler.center_momentum,
    )


def load_slide_encoder_from_checkpoint(
    checkpoint_path: str,
    *,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    layer_drop: float = 0.0,
) -> tuple[MOOZYSlideEncoder, dict[str, object]]:
    """Load a ``MOOZYSlideEncoder`` from a teacher encoder export checkpoint.

    This is the single source of truth for "checkpoint → slide encoder".  Both
    training (stage-2 init) and inference (stage-1 slide extraction) call
    this function.
    """
    ckpt = load_checkpoint_payload(checkpoint_path)
    slide_encoder = build_slide_encoder_from_payload(
        state=ckpt["teacher_slide_encoder"],
        config=ckpt["meta"],
        dropout=dropout,
        attn_dropout=attn_dropout,
        layer_drop=layer_drop,
    )
    return slide_encoder, ckpt["meta"]


def load_teacher_slide_encoder(
    slide_encoder: SlideEncoderConfig,
    teacher_checkpoint: str,
    feat_dim: int = 384,
) -> tuple[MOOZYSlideEncoder, dict[str, object]]:
    """Load or create a slide encoder for stage-2 training.

    If *teacher_checkpoint* is empty, builds a fresh slide encoder from the
    variant table using *feat_dim*.  Otherwise delegates to
    :func:`load_slide_encoder_from_checkpoint` (feat_dim comes from checkpoint).
    """
    checkpoint_path = str(teacher_checkpoint or "").strip()
    if not checkpoint_path:
        resolved = slide_encoder.resolve_variant()
        if resolved.d_model is None:
            raise ValueError(
                f"Unknown encoder variant for scratch init: {slide_encoder.variant!r}. "
                "Expected one of ENCODER_VARIANTS."
            )
        se = resolved
        se_module = MOOZYSlideEncoder(
            feat_dim=feat_dim,
            d_model=se.d_model,
            n_heads=se.n_heads,
            n_layers=se.n_layers,
            dim_feedforward=se.dim_feedforward,
            num_registers=se.num_registers,
            dropout=se.dropout,
            attn_dropout=se.attn_dropout,
            layer_drop=se.layer_drop,
            qk_norm=se.qk_norm,
            layerscale_init=se.layerscale_init,
            learnable_alibi=se.learnable_alibi,
        )
        return se_module, build_slide_encoder_save_meta(se_module)

    return load_slide_encoder_from_checkpoint(
        checkpoint_path,
        dropout=slide_encoder.dropout,
        attn_dropout=slide_encoder.attn_dropout,
        layer_drop=slide_encoder.layer_drop,
    )


def load_stage2_inference_model(
    checkpoint_path: str,
    device: torch.device,
) -> MOOZY:
    """Load a stage-2 MOOZY model for inference."""
    payload = load_checkpoint_payload(checkpoint_path)

    slide_encoder_module = build_slide_encoder_from_payload(
        state=payload["teacher_slide_encoder"],
        config=payload["slide_encoder_config"],
    )

    case_cfg = payload["case_transformer_config"]
    model = MOOZY(
        slide_encoder=slide_encoder_module,
        case_transformer_layers=int(case_cfg["num_layers"]),
        case_transformer_heads=int(case_cfg["num_heads"]),
        case_transformer_ffn_dim=int(case_cfg["dim_feedforward"]),
        case_transformer_dropout=float(case_cfg["dropout"]),
        case_transformer_layerscale_init=float(case_cfg["layerscale_init"]),
        case_transformer_layer_drop=float(case_cfg["layer_drop"]),
        case_transformer_qk_norm=bool(case_cfg["qk_norm"]),
        case_transformer_num_registers=int(case_cfg["num_registers"]),
    )
    model.case_transformer.load_state_dict(payload["case_transformer"], strict=True)
    model.to(device)
    model.eval()
    return model


def build_supervised_model(
    slide_encoder_module: MOOZYSlideEncoder,
    *,
    task_info: dict[str, Any],
    survival_bin_edges: dict[str, list[float]],
    case_transformer_variant: str = "base_quarter_depth",
    case_transformer_dropout: float = 0.1,
    case_transformer_layerscale_init: float = 1e-5,
    case_transformer_layer_drop: float = 0.0,
    case_transformer_qk_norm: bool = False,
    case_transformer_num_registers: int = 0,
    classification_head_type: str = "mlp",
    survival_head_type: str = "mlp",
    head_dropout: float = 0.1,
    label_smoothing: float = 0.03,
    survival_num_bins: int = 8,
) -> MOOZY:
    """Construct a Stage-2 supervised model from a slide encoder and task info."""
    case_variant_cfg = ENCODER_VARIANTS.get(str(case_transformer_variant).lower().strip())
    if case_variant_cfg is None:
        raise ValueError(
            f"Unknown case_transformer_variant: {case_transformer_variant!r}. Expected one of ENCODER_VARIANTS."
        )
    case_layers = int(case_variant_cfg["n_layers"])
    case_heads = int(case_variant_cfg["n_heads"])
    case_ffn_dim = int(case_variant_cfg["dim_feedforward"])
    if int(slide_encoder_module.d_model) % case_heads != 0:
        raise ValueError(
            "Invalid case_transformer_variant for this slide encoder: "
            f"d_model={int(slide_encoder_module.d_model)} is not divisible by num_heads={case_heads} "
            f"(variant={case_transformer_variant})."
        )

    tasks = task_info.get("tasks", [])
    task_names = [t["name"] for t in tasks]
    task_keys = [t["key"] for t in tasks]
    task_num_classes = [int(t["num_classes"]) for t in tasks]
    task_class_weights = [t["class_weights"] for t in tasks]
    task_types = [t.get("task_type", "classification") for t in tasks]

    return MOOZY(
        slide_encoder=slide_encoder_module,
        task_names=task_names,
        task_keys=task_keys,
        task_num_classes=task_num_classes,
        task_class_weights=task_class_weights,
        task_types=task_types,
        classification_head_type=classification_head_type,
        survival_head_type=survival_head_type,
        head_dropout=head_dropout,
        label_smoothing=label_smoothing,
        case_transformer_layers=case_layers,
        case_transformer_heads=case_heads,
        case_transformer_ffn_dim=case_ffn_dim,
        case_transformer_dropout=case_transformer_dropout,
        case_transformer_layerscale_init=case_transformer_layerscale_init,
        case_transformer_layer_drop=case_transformer_layer_drop,
        case_transformer_qk_norm=case_transformer_qk_norm,
        case_transformer_num_registers=case_transformer_num_registers,
        survival_num_bins=survival_num_bins,
        survival_bin_edges=survival_bin_edges,
    )
