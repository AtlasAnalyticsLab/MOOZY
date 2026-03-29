import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from .layers import DropPath, LayerScale


class CaseTransformerBlock(nn.Module):
    """Single transformer block for case-level aggregation."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        qk_norm: bool = False,
        layerscale_init: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = float(dropout)
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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, d_model = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if not isinstance(self.q_norm, nn.Identity):
            q = self.q_norm(q)
        if not isinstance(self.k_norm, nn.Identity):
            k = self.k_norm(k)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0).to(dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, d_model)
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


class CaseAggregator(nn.Module):
    """Aggregate per-slide embeddings into a single case embedding."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        layerscale_init: float = 1e-5,
        layer_drop: float = 0.0,
        qk_norm: bool = False,
        num_registers: int = 0,
    ):
        super().__init__()
        num_layers = max(1, num_layers)
        num_heads = max(1, num_heads)
        if d_model % num_heads != 0:
            raise ValueError(f"case transformer heads ({num_heads}) must divide d_model ({d_model}).")
        if dim_feedforward is None or dim_feedforward <= 0:
            dim_feedforward = 4 * d_model
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layerscale_init = layerscale_init
        self.layer_drop = layer_drop
        self.qk_norm = qk_norm
        self.num_registers = num_registers
        self.activation_checkpointing = False

        self.case_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.case_token, std=0.02)

        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, d_model))
            nn.init.normal_(self.register_tokens, std=0.02)

        dpr = [float(x) for x in torch.linspace(0, layer_drop, num_layers)]
        self.blocks = nn.ModuleList(
            [
                CaseTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    drop_path_rate=dpr[i],
                    qk_norm=qk_norm,
                    layerscale_init=layerscale_init,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        self.activation_checkpointing = bool(enabled)

    def forward(
        self,
        slide_tokens: torch.Tensor,
        slide_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze_batch = False
        if slide_tokens.dim() == 2:
            slide_tokens = slide_tokens.unsqueeze(0)
            squeeze_batch = True
        if slide_tokens.dim() != 3:
            raise ValueError(f"Expected slide_tokens [B, S, D], got {slide_tokens.shape}")
        batch_size = slide_tokens.shape[0]

        case_token = self.case_token.expand(batch_size, -1, -1)
        tokens = torch.cat([case_token, slide_tokens], dim=1)

        key_padding_mask = None
        num_prefix = 1
        if self.num_registers > 0:
            reg = self.register_tokens.expand(batch_size, -1, -1)
            tokens = torch.cat([tokens[:, :1], reg, tokens[:, 1:]], dim=1)
            num_prefix += self.num_registers

        if slide_mask is not None:
            if slide_mask.dim() == 1:
                slide_mask = slide_mask.unsqueeze(0)
            if slide_mask.dim() != 2:
                raise ValueError(f"Expected slide_mask [B, S], got {slide_mask.shape}")
            prefix_mask = torch.zeros(
                (slide_mask.shape[0], num_prefix),
                dtype=slide_mask.dtype,
                device=slide_mask.device,
            )
            key_padding_mask = torch.cat([prefix_mask, slide_mask], dim=1)

        use_checkpoint = bool(
            self.activation_checkpointing and self.training and torch.is_grad_enabled() and tokens.requires_grad
        )
        for block in self.blocks:
            if use_checkpoint:

                def _run_block(tokens_in: torch.Tensor, block=block) -> torch.Tensor:
                    return block(tokens_in, key_padding_mask=key_padding_mask)

                tokens = checkpoint_fn(_run_block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, key_padding_mask=key_padding_mask)

        tokens = self.norm(tokens)
        case_out = tokens[:, 0]
        if squeeze_batch:
            case_out = case_out.squeeze(0)
        return case_out
