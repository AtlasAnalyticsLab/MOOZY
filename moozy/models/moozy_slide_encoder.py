import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from .layers import DropPath, LayerScale, init_linear_and_layernorm_weights


def _get_alibi_slopes(n_heads: int):
    """Compute fixed, decreasing ALiBi slopes as in Press et al.

    This follows the commonly used reference implementation that creates a
    head-wise decreasing sequence. For non power-of-two head counts, it
    stitches sequences to preserve the overall monotone decrease.
    Returns a list of positive slopes (we apply a minus sign at registration).
    """

    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if n_heads & (n_heads - 1) == 0:
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
        slopes += extra_slopes
    return slopes


class ALiBi2D(nn.Module):
    """
    2-D Attention with Linear Biases (ALiBi) for Vision Transformer.

    Extends attention with distance-dependent bias based on Euclidean distance
    in the 2-D feature grid. This enables train-short, test-long behavior.
    """

    def __init__(self, num_heads: int, learnable: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.learnable = bool(learnable)

        slopes_pos = torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float32).unsqueeze(1)
        # Bias is added directly to logits, so larger distances need more negative values.
        slopes_init = -slopes_pos
        if self.learnable:
            self.slopes = nn.Parameter(slopes_init)
        else:
            self.register_buffer("slopes", slopes_init)

    def build_bias(
        self,
        positions: torch.Tensor,
        patch_sizes: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        """Construct an additive ALiBi bias tensor [B, H, N, N].

        Positions must include the CLS token at index 0. Bias for CLS row/col
        is set to 0 to keep CLS spatially neutral.
        Register tokens (if any) also receive zero relative bias to remain
        unbiased anchors.
        """
        assert positions.ndim == 3 and positions.shape[-1] == 2, "positions must be [B, N, 2]"
        if patch_sizes is None:
            raise ValueError("patch_sizes must be provided for ALiBi bias computation.")
        B, _, _ = positions.shape
        patch_sizes = torch.as_tensor(patch_sizes, device=positions.device, dtype=positions.dtype).view(-1)
        if patch_sizes.numel() == 1 and B > 1:
            patch_sizes = patch_sizes.expand(B)
        if patch_sizes.numel() != B:
            raise ValueError(f"patch_sizes must have 1 or {B} elements, got {patch_sizes.numel()}")
        patch_sizes = torch.clamp(patch_sizes, min=1e-6)
        patch_sizes_dist = patch_sizes.view(B, 1, 1)
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.norm(pos_diff, dim=-1)
        distances = distances / patch_sizes_dist
        slopes = self.slopes.view(1, self.num_heads, 1, 1)
        bias = slopes * distances.unsqueeze(1)

        bias[:, :, 0, :] = 0.0
        bias[:, :, :, 0] = 0.0
        if num_registers > 0:
            reg_start = 1
            reg_end = 1 + int(num_registers)
            bias[:, :, reg_start:reg_end, :] = 0.0
            bias[:, :, :, reg_start:reg_end] = 0.0
        return bias


class VisionTransformerBlock(nn.Module):
    """Single transformer block with 2-D positional bias and SDPA."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        alibi: ALiBi2D | None = None,
        drop_path_rate: float = 0.0,
        qk_norm: bool = True,
        layerscale_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = bool(qk_norm)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if alibi is None:
            alibi = ALiBi2D(n_heads)
        self.pos_bias = alibi

        self.attn_drop_prob = float(attn_dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.ls_attn = (
            LayerScale(d_model, init_values=layerscale_init)
            if layerscale_init and layerscale_init > 0
            else nn.Identity()
        )
        self.ls_mlp = (
            LayerScale(d_model, init_values=layerscale_init)
            if layerscale_init and layerscale_init > 0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        patch_sizes: torch.Tensor | None = None,
        num_registers: int = 0,
    ) -> torch.Tensor:
        B, N, d = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        assert positions is not None, "positions [B,N,2] required for ALiBi"
        alibi_bias = self.pos_bias.build_bias(positions, patch_sizes, num_registers=num_registers)

        if attn_mask is not None:
            if attn_mask.dim() == 4 and attn_mask.shape[1] == 1:
                attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)
            attn_bias = (alibi_bias + attn_mask).to(dtype=q.dtype)
        else:
            attn_bias = alibi_bias.to(dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, N, d)
        attn_out = self.out_proj(attn_out)

        x = x + self.drop_path(self.ls_attn(attn_out))

        x_norm2 = self.norm2(x)
        mlp_out = self.fc1(x_norm2)
        mlp_out = self.activation(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out)
        mlp_out = self.fc2(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out)

        x = x + self.drop_path(self.ls_mlp(mlp_out))
        return x


class MOOZYSlideEncoder(nn.Module):
    """
    Vision Transformer slide encoder for MOOZY.

    Takes feature crops as input and applies a sequence of transformer blocks
    with 2-D distance-dependent positional biases.
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
        layer_drop: float = 0.0,
        qk_norm: bool = True,
        layerscale_init: float = 0.0,
        learnable_alibi: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_registers = max(0, int(num_registers))
        self.learnable_alibi = bool(learnable_alibi)
        self.activation_checkpointing = False

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_registers, d_model))
        else:
            self.register_tokens = None

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_bias = ALiBi2D(
            n_heads,
            learnable=self.learnable_alibi,
        )

        drop_path_rates = torch.linspace(0, layer_drop, steps=n_layers).tolist() if n_layers > 0 else []
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    alibi=self.pos_bias,
                    drop_path_rate=drop_path_rates[i] if i < len(drop_path_rates) else 0.0,
                    qk_norm=qk_norm,
                    layerscale_init=layerscale_init,
                )
                for i in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        # Zero-init keeps masked-token prediction stable at startup.
        nn.init.zeros_(self.mask_token)
        self.apply(init_linear_and_layernorm_weights)

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        self.activation_checkpointing = bool(enabled)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        invalid_mask: torch.Tensor | None = None,
        coords_xy: torch.Tensor = None,
        patch_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : [B, crop_h, crop_w, feat_dim]
            Feature crops
        mask : [B, crop_h, crop_w], optional
            Masking mask (True = masked)
        patch_sizes : [B] float tensor
            Patch size (in level-0 pixels) for each sample; required for ALiBi scaling

        Returns
        -------
        cls_output : [B, d_model]
            CLS token output
        patch_output : [B, N, d_model]
            Patch token outputs (excluding CLS)
        mask : [B, N] bool
            Flattened mask
        """
        B, crop_h, crop_w, feat_dim = x.shape
        N = crop_h * crop_w

        if patch_sizes is None:
            raise ValueError("patch_sizes must be provided per window for ALiBi scaling.")
        patch_sizes = torch.as_tensor(patch_sizes, device=x.device, dtype=torch.float32).view(-1)
        if patch_sizes.numel() == 1 and B > 1:
            patch_sizes = patch_sizes.expand(B)
        if patch_sizes.numel() != B:
            raise ValueError(f"patch_sizes must have length {B}, got {patch_sizes.numel()}")

        x = x.reshape(B, N, feat_dim)
        x = self.input_proj(x)
        if mask is None:
            mask_flat = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        else:
            mask_flat = mask.reshape(B, N)

        if invalid_mask is None:
            invalid_flat = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        else:
            invalid_flat = invalid_mask.reshape(B, N)

        if mask_flat.any():
            x = torch.where(
                mask_flat.unsqueeze(-1),
                self.mask_token.expand(B, N, -1),
                x,
            )

        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.num_registers > 0 and self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            tokens.append(reg)
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        if coords_xy is None:
            raise ValueError("coords_xy must be provided and contain real-space positions for tokens.")
        coords_xy = coords_xy.to(device=x.device, dtype=torch.float32)
        pos_tokens = coords_xy.reshape(B, N, 2)
        zeros_cls = torch.zeros(B, 1, 2, dtype=pos_tokens.dtype, device=pos_tokens.device)
        if self.num_registers > 0:
            zeros_registers = torch.zeros(B, self.num_registers, 2, dtype=pos_tokens.dtype, device=pos_tokens.device)
            positions = torch.cat([zeros_cls, zeros_registers, pos_tokens], dim=1)
        else:
            positions = torch.cat([zeros_cls, pos_tokens], dim=1)

        # Always mask background pairs in attention. For unmasked single-sample
        # forwards, prune invalid tokens entirely to reduce O(N^2) memory.
        valid_flat = torch.logical_not(invalid_flat)
        register_valid = (
            torch.ones(B, self.num_registers, dtype=torch.bool, device=x.device)
            if self.num_registers > 0
            else torch.zeros(B, 0, dtype=torch.bool, device=x.device)
        )
        valid_with_cls = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=x.device), register_valid, valid_flat],
            dim=1,
        )

        if mask is None and B == 1:
            keep = valid_with_cls[0]
            if keep.sum() > 0:
                x = x[:, keep, :]
                positions = positions[:, keep, :]
            attn_mask = None
        else:
            pair_valid = valid_with_cls.unsqueeze(2) & valid_with_cls.unsqueeze(1)
            diag = torch.eye(pair_valid.shape[-1], dtype=torch.bool, device=x.device).unsqueeze(0)
            pair_valid = pair_valid | diag
            mask_fill = torch.finfo(x.dtype).min
            attn_mask = (~pair_valid).to(x.dtype) * mask_fill
            attn_mask = attn_mask.unsqueeze(1)

        use_checkpoint = bool(
            self.activation_checkpointing and self.training and torch.is_grad_enabled() and x.requires_grad
        )
        for block in self.blocks:
            if use_checkpoint:

                def _run_block(x_in: torch.Tensor, block=block) -> torch.Tensor:
                    return block(
                        x_in,
                        positions=positions,
                        attn_mask=attn_mask,
                        patch_sizes=patch_sizes,
                        num_registers=self.num_registers,
                    )

                x = checkpoint_fn(_run_block, x, use_reentrant=False)
            else:
                x = block(
                    x,
                    positions=positions,
                    attn_mask=attn_mask,
                    patch_sizes=patch_sizes,
                    num_registers=self.num_registers,
                )

        x = self.norm(x)
        cls_output = x[:, 0, :]
        patch_output = x[:, 1 + self.num_registers :, :]
        return cls_output, patch_output, mask_flat
