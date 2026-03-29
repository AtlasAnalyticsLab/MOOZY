import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .layers import DropPath, LayerScale
from .moozy_slide_encoder import MOOZYSlideEncoder


def load_checkpoint_payload(checkpoint_path: str) -> dict[str, Any]:
    """Load a checkpoint payload as a dictionary from disk."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    return payload


def load_stage1_training_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
) -> tuple:
    """Load a stage-1 training checkpoint and return (global step, extra_state)."""
    checkpoint = load_checkpoint_payload(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if restore_rng:
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

        if "numpy_rng_state" in checkpoint and checkpoint["numpy_rng_state"] is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])

        if "random_rng_state" in checkpoint and checkpoint["random_rng_state"] is not None:
            random.setstate(checkpoint["random_rng_state"])

    return checkpoint["global_step"], checkpoint["extra_state"]


@dataclass(frozen=True)
class SlideEncoderSpec:
    feat_dim: int
    d_model: int
    n_layers: int
    n_heads: int
    dim_feedforward: int
    num_registers: int
    learnable_alibi: bool
    qk_norm: bool
    layerscale_enabled: bool
    layerscale_init: float
    dropout: float = 0.0
    attn_dropout: float = 0.0
    layer_drop: float = 0.0


def _extract_slide_encoder_meta(slide_encoder: MOOZYSlideEncoder) -> dict[str, object]:
    """Introspect architectural fields from a live slide encoder instance."""
    first_block = slide_encoder.blocks[0]

    ls_attn = first_block.ls_attn
    layerscale_enabled = isinstance(ls_attn, LayerScale)
    layerscale_init = float(ls_attn.gamma.detach().float().mean().item()) if layerscale_enabled else 0.0

    return {
        "feat_dim": int(slide_encoder.feat_dim),
        "d_model": int(slide_encoder.d_model),
        "n_layers": int(slide_encoder.n_layers),
        "n_heads": int(slide_encoder.n_heads),
        "dim_feedforward": int(first_block.fc1.out_features),
        "num_registers": int(slide_encoder.num_registers),
        "learnable_alibi": bool(slide_encoder.learnable_alibi),
        "qk_norm": bool(first_block.qk_norm),
        "layerscale_enabled": layerscale_enabled,
        "layerscale_init": layerscale_init,
    }


def build_slide_encoder_save_meta(
    slide_encoder: MOOZYSlideEncoder,
    *,
    output_dim: int | None = None,
) -> dict[str, object]:
    """Collect metadata for a slide encoder checkpoint save.

    When *output_dim* is provided (stage-1 teacher export), it is included in
    the metadata alongside the slide encoder's ``feat_dim`` and ``d_model``.
    """
    meta = _extract_slide_encoder_meta(slide_encoder)
    meta["patch_size_dynamic"] = True
    if output_dim is not None:
        meta["output_dim"] = int(output_dim)
    return meta


def _parse_slide_encoder_spec(
    config: Mapping[str, object],
    *,
    state: Mapping[str, torch.Tensor] | None = None,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    layer_drop: float = 0.0,
) -> SlideEncoderSpec:
    """Parse slide encoder architecture from a checkpoint metadata dict.

    Required keys: feat_dim, d_model, n_layers, n_heads, dim_feedforward.
    Optional keys: num_registers, learnable_alibi, qk_norm, layerscale_enabled,
    layerscale_init, dropout, attn_dropout, layer_drop.

    When dropout/attn_dropout/layer_drop are present in *config* they take
    precedence over the explicit keyword arguments (stage-2 exports store them;
    stage-1 exports do not).
    """
    try:
        feat_dim = int(config["feat_dim"])
        d_model = int(config["d_model"])
        n_layers = int(config["n_layers"])
        n_heads = int(config["n_heads"])
        dim_feedforward = int(config["dim_feedforward"])
    except KeyError as exc:
        raise ValueError(f"Checkpoint config missing required key '{exc.args[0]}'") from exc

    num_registers = int(config.get("num_registers", 0))
    learnable_alibi = bool(config.get("learnable_alibi", False))
    qk_norm = bool(config.get("qk_norm", False))

    has_ls = state is not None and any(key.endswith(".ls_attn.gamma") or key.endswith(".ls_mlp.gamma") for key in state)
    ls_enabled_raw = config.get("layerscale_enabled")
    layerscale_enabled = bool(ls_enabled_raw) if ls_enabled_raw is not None else has_ls
    if layerscale_enabled:
        layerscale_init = max(1e-5, float(config.get("layerscale_init", 0) or 0))
    else:
        layerscale_init = 0.0

    resolved_dropout = float(config["dropout"]) if "dropout" in config else dropout
    resolved_attn_dropout = float(config["attn_dropout"]) if "attn_dropout" in config else attn_dropout
    resolved_layer_drop = float(config["layer_drop"]) if "layer_drop" in config else layer_drop

    return SlideEncoderSpec(
        feat_dim=feat_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        num_registers=num_registers,
        learnable_alibi=learnable_alibi,
        qk_norm=qk_norm,
        layerscale_enabled=layerscale_enabled,
        layerscale_init=layerscale_init,
        dropout=resolved_dropout,
        attn_dropout=resolved_attn_dropout,
        layer_drop=resolved_layer_drop,
    )


def _build_slide_encoder_from_spec(spec: SlideEncoderSpec) -> MOOZYSlideEncoder:
    """Construct a fresh slide encoder from a parsed spec."""
    return MOOZYSlideEncoder(
        feat_dim=spec.feat_dim,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        dim_feedforward=spec.dim_feedforward,
        num_registers=spec.num_registers,
        dropout=spec.dropout,
        attn_dropout=spec.attn_dropout,
        layer_drop=spec.layer_drop,
        qk_norm=spec.qk_norm,
        layerscale_init=spec.layerscale_init if spec.layerscale_enabled else 0.0,
        learnable_alibi=spec.learnable_alibi,
    )


def build_slide_encoder_from_payload(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    layer_drop: float = 0.0,
) -> MOOZYSlideEncoder:
    """Rebuild a slide encoder from a checkpoint payload."""
    spec = _parse_slide_encoder_spec(
        config, state=state, dropout=dropout, attn_dropout=attn_dropout, layer_drop=layer_drop
    )
    slide_encoder = _build_slide_encoder_from_spec(spec)
    slide_encoder.load_state_dict(state, strict=True)
    return slide_encoder


def extract_slide_encoder_config(slide_encoder: MOOZYSlideEncoder) -> dict[str, object]:
    """Collect the full stage-2 slide encoder config."""
    meta = _extract_slide_encoder_meta(slide_encoder)

    first_block = slide_encoder.blocks[0]
    meta["dropout"] = float(first_block.mlp_dropout.p)
    meta["attn_dropout"] = float(first_block.attn_drop_prob)

    layer_drop = 0.0
    for block in slide_encoder.blocks:
        drop_prob = float(block.drop_path.drop_prob) if isinstance(block.drop_path, DropPath) else 0.0
        layer_drop = max(layer_drop, drop_prob)
    meta["layer_drop"] = layer_drop

    return meta


def extract_case_transformer_config(case_transformer) -> dict[str, object]:
    """Collect the case-transformer config."""
    return {
        "d_model": int(case_transformer.d_model),
        "num_layers": int(case_transformer.num_layers),
        "num_heads": int(case_transformer.num_heads),
        "dim_feedforward": int(case_transformer.dim_feedforward),
        "dropout": float(case_transformer.dropout),
        "layerscale_init": float(case_transformer.layerscale_init),
        "layer_drop": float(case_transformer.layer_drop),
        "qk_norm": bool(case_transformer.qk_norm),
        "num_registers": int(case_transformer.num_registers),
    }
