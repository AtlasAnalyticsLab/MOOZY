import logging
from typing import Any

import timm
import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    """Thin wrapper around the supported patch encoder backend."""

    SUPPORTED_ENCODER = "DINO_p8"
    HF_HUB_MODEL_NAME = "hf-hub:1aurent/vit_small_patch8_224.lunit_dino"

    def __init__(self):
        super().__init__()
        self.model_name = self.SUPPORTED_ENCODER
        self.backbone = timm.create_model(
            model_name=self.HF_HUB_MODEL_NAME,
            pretrained=True,
        )
        self.data_config: dict[str, Any] = timm.data.resolve_model_data_config(self.backbone)
        self.eval_transform = timm.data.create_transform(
            **self.data_config,
            is_training=False,
        )
        logging.info(
            "Loaded patch encoder %s from %s",
            self.model_name,
            self.HF_HUB_MODEL_NAME,
        )

    def create_transform(self, is_training: bool = False):
        """Create the timm transform that matches the loaded backbone."""
        return timm.data.create_transform(
            **self.data_config,
            is_training=bool(is_training),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings for a BCHW tensor."""
        return self.backbone(x)
