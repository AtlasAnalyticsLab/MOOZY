import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import init_linear_and_layernorm_weights


class ProjectionHead(nn.Module):
    """Stage-1 prototype projection head."""

    def __init__(
        self,
        d_model: int = 768,
        d_hidden: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 8192,
        norm_last_layer: bool = True,
        norm_type: str = "none",
        last_norm_type: str = "none",
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.norm_last_layer = bool(norm_last_layer)
        self.norm_type = str(norm_type)
        self.last_norm_type = str(last_norm_type)
        self.n_layers = 3

        norm = self._build_norm(self.norm_type, d_hidden)
        act = nn.GELU()
        layers = [nn.Linear(d_model, d_hidden)]
        if norm is not None:
            layers.append(norm)
        layers.append(act)
        for _ in range(self.n_layers - 2):
            layers.append(nn.Linear(d_hidden, d_hidden))
            if norm is not None:
                layers.append(norm)
            layers.append(act)
        layers.append(nn.Linear(d_hidden, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        self.last_norm = self._build_norm(self.last_norm_type, output_dim, affine=False)
        self.apply(init_linear_and_layernorm_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        if self.norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            cls_tok = x[:, 0]
            patch_tok = x[:, 1:]

            cls_logits = self._forward_tokens(cls_tok)
            patch_logits = self._forward_tokens(patch_tok.reshape(-1, self.d_model))
            patch_logits = patch_logits.reshape(patch_tok.shape[0], patch_tok.shape[1], self.output_dim)
            return torch.cat([cls_logits.unsqueeze(1), patch_logits], dim=1)

        original_shape = x.shape[:-1]
        logits = self._forward_tokens(x.reshape(-1, self.d_model))
        return logits.reshape(*original_shape, self.output_dim)

    def _forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(tokens)
        projected = F.normalize(projected, dim=-1, p=2)
        projected = self.last_layer(projected)
        if self.last_norm is not None:
            projected = self.last_norm(projected)
        return projected

    @staticmethod
    def _build_norm(norm_type: str, dim: int, affine: bool = True):
        nt = (norm_type or "none").lower()
        if nt == "none":
            return None
        if nt == "ln":
            return nn.LayerNorm(dim, elementwise_affine=affine)
        raise ValueError(f"Unsupported norm type for projection head: {norm_type}")


def build_task_head(d_model: int, num_classes: int, head_type: str) -> nn.Module:
    """Build a classification or survival head with the current shapes."""
    if head_type == "linear":
        return nn.Linear(d_model, num_classes)
    if head_type == "mlp":
        h1 = max(4, int(round(d_model * 0.66)))
        h2 = max(2, int(round(h1 * 0.5)))
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h1),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(h2, num_classes),
        )
    raise ValueError(f"Unsupported head_type: {head_type}")
