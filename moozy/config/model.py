from dataclasses import dataclass

from moozy.models.variants import ENCODER_VARIANTS


@dataclass(frozen=True)
class SlideEncoderConfig:
    """MOOZYSlideEncoder architecture parameters."""

    variant: str = "base_half_depth"
    num_registers: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.0
    layer_drop: float = 0.1
    qk_norm: bool = False
    layerscale_init: float = 0.0
    learnable_alibi: bool = False

    d_model: int | None = None
    n_heads: int | None = None
    n_layers: int | None = None
    dim_feedforward: int | None = None

    def resolve_variant(self) -> "SlideEncoderConfig":
        """Return a new config with d_model/n_heads/n_layers/dim_feedforward filled from the variant table."""
        variant_cfg = ENCODER_VARIANTS.get(str(self.variant).lower().strip())
        if variant_cfg is None:
            return self
        return SlideEncoderConfig(
            variant=self.variant,
            num_registers=self.num_registers,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            layer_drop=self.layer_drop,
            qk_norm=self.qk_norm,
            layerscale_init=self.layerscale_init,
            learnable_alibi=self.learnable_alibi,
            d_model=int(variant_cfg["d_model"]),
            n_heads=int(variant_cfg["n_heads"]),
            n_layers=int(variant_cfg["n_layers"]),
            dim_feedforward=int(variant_cfg["dim_feedforward"]),
        )


@dataclass(frozen=True)
class ProjectionConfig:
    """Stage-1 projection head parameters."""

    output_dim: int = 8192
    proj_hidden_dim: int = 2048
    proj_bottleneck_dim: int = 256
    proj_norm_last_layer: bool = True
    proj_norm: str = "none"
    proj_last_norm: str = "none"
