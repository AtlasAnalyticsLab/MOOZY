from dataclasses import dataclass

import torch
import torch.nn as nn

from moozy.data.stage1 import Stage1Batch


@dataclass(frozen=True)
class Stage1BatchViews:
    """Flattened canonical stage-1 batch views used by the SSL model."""

    batch_size: int
    num_global: int
    num_local: int
    global_height: int
    global_width: int
    global_crops_flat: torch.Tensor
    local_crops_flat: torch.Tensor
    global_masks_flat: torch.Tensor
    invalid_global_flat: torch.Tensor
    invalid_local_flat: torch.Tensor
    global_coords_flat: torch.Tensor
    local_coords_flat: torch.Tensor
    patch_sizes_global: torch.Tensor
    patch_sizes_local: torch.Tensor

    @property
    def num_global_tokens(self) -> int:
        return self.global_height * self.global_width


@dataclass(frozen=True)
class Stage1ProjectedViews:
    """Backbone features plus projection-head logits for a view collection."""

    cls_features: torch.Tensor
    patch_features: torch.Tensor
    cls_logits: torch.Tensor
    patch_logits: torch.Tensor


@dataclass(frozen=True)
class Stage1StudentEncoding:
    """Student outputs for the canonical global and local view sets."""

    global_views: Stage1ProjectedViews
    local_views: Stage1ProjectedViews


@dataclass(frozen=True)
class Stage1TeacherEncoding:
    """Teacher outputs for global views before and after centering."""

    cls_features: torch.Tensor
    cls_logits_raw: torch.Tensor
    patch_logits_raw: torch.Tensor
    cls_logits: torch.Tensor
    patch_logits: torch.Tensor


def prepare_stage1_batch_views(batch: Stage1Batch) -> Stage1BatchViews:
    """Normalize the canonical stage-1 batch into flattened view tensors."""
    global_crops = batch["global_crops"]
    local_crops = batch["local_crops"]
    global_masks = batch["global_masks"]
    global_valids = batch["global_valids"]
    local_valids = batch["local_valids"]
    global_coords = batch["global_coords"]
    local_coords = batch["local_coords"]
    patch_sizes = batch["patch_sizes"]

    batch_size, num_global, global_height, global_width, feat_dim_global = global_crops.shape
    batch_size_local, num_local, local_height, local_width, feat_dim_local = local_crops.shape
    assert batch_size_local == batch_size, "Batch size mismatch between global and local crops"
    assert feat_dim_global == feat_dim_local, "Feature dim mismatch between global and local crops"

    patch_sizes = patch_sizes.to(device=global_crops.device, dtype=torch.float32).view(-1)
    if patch_sizes.numel() == 1 and batch_size > 1:
        patch_sizes = patch_sizes.expand(batch_size)
    if patch_sizes.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} patch sizes, got {patch_sizes.numel()}")

    global_crops_flat = global_crops.reshape(
        batch_size * num_global,
        global_height,
        global_width,
        feat_dim_global,
    )
    local_crops_flat = local_crops.reshape(
        batch_size * num_local,
        local_height,
        local_width,
        feat_dim_global,
    )
    patch_sizes_global = patch_sizes.unsqueeze(1).expand(-1, num_global).reshape(batch_size * num_global)
    patch_sizes_local = patch_sizes.unsqueeze(1).expand(-1, num_local).reshape(batch_size * num_local)
    global_coords_flat = global_coords.reshape(batch_size * num_global, global_height, global_width, 2)
    local_coords_flat = local_coords.reshape(batch_size * num_local, local_height, local_width, 2)

    global_valids = global_valids.to(dtype=torch.bool)
    local_valids = local_valids.to(dtype=torch.bool)

    return Stage1BatchViews(
        batch_size=batch_size,
        num_global=num_global,
        num_local=num_local,
        global_height=global_height,
        global_width=global_width,
        global_crops_flat=global_crops_flat,
        local_crops_flat=local_crops_flat,
        global_masks_flat=global_masks.reshape(batch_size * num_global, global_height, global_width),
        invalid_global_flat=torch.logical_not(
            global_valids.reshape(batch_size * num_global, global_height, global_width)
        ),
        invalid_local_flat=torch.logical_not(local_valids.reshape(batch_size * num_local, local_height, local_width)),
        global_coords_flat=global_coords_flat,
        local_coords_flat=local_coords_flat,
        patch_sizes_global=patch_sizes_global,
        patch_sizes_local=patch_sizes_local,
    )


def _project_stage1_views(
    slide_encoder: nn.Module,
    head: nn.Module,
    *,
    crops: torch.Tensor,
    invalid_mask: torch.Tensor,
    coords_xy: torch.Tensor,
    patch_sizes: torch.Tensor,
    mask: torch.Tensor | None,
) -> Stage1ProjectedViews:
    cls_features, patch_features, _ = slide_encoder(
        crops,
        mask=mask,
        invalid_mask=invalid_mask,
        coords_xy=coords_xy,
        patch_sizes=patch_sizes,
    )
    logits = head(torch.cat([cls_features.unsqueeze(1), patch_features], dim=1))
    return Stage1ProjectedViews(
        cls_features=cls_features,
        patch_features=patch_features,
        cls_logits=logits[:, 0],
        patch_logits=logits[:, 1:],
    )


def encode_stage1_student_views(
    slide_encoder: nn.Module,
    head: nn.Module,
    views: Stage1BatchViews,
) -> Stage1StudentEncoding:
    """Encode the canonical stage-1 batch through the student encoder."""
    return Stage1StudentEncoding(
        global_views=_project_stage1_views(
            slide_encoder,
            head,
            crops=views.global_crops_flat,
            invalid_mask=views.invalid_global_flat,
            coords_xy=views.global_coords_flat,
            patch_sizes=views.patch_sizes_global,
            mask=views.global_masks_flat,
        ),
        local_views=_project_stage1_views(
            slide_encoder,
            head,
            crops=views.local_crops_flat,
            invalid_mask=views.invalid_local_flat,
            coords_xy=views.local_coords_flat,
            patch_sizes=views.patch_sizes_local,
            mask=None,
        ),
    )


def encode_stage1_teacher_global_views(
    slide_encoder: nn.Module,
    head: nn.Module,
    views: Stage1BatchViews,
    *,
    center_cls: torch.Tensor,
    center_patch: torch.Tensor,
) -> Stage1TeacherEncoding:
    """Encode the canonical stage-1 global views through the teacher encoder."""
    projected = _project_stage1_views(
        slide_encoder,
        head,
        crops=views.global_crops_flat.detach(),
        invalid_mask=views.invalid_global_flat,
        coords_xy=views.global_coords_flat,
        patch_sizes=views.patch_sizes_global,
        mask=None,
    )
    return Stage1TeacherEncoding(
        cls_features=projected.cls_features,
        cls_logits_raw=projected.cls_logits,
        patch_logits_raw=projected.patch_logits,
        cls_logits=projected.cls_logits - center_cls,
        patch_logits=projected.patch_logits - center_patch,
    )
