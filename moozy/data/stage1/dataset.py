import logging
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from moozy.data.features import SLIDE_GRID_STEP_TOLERANCE, build_grid_from_coords, list_feature_paths, load_feature_h5

from .crops import sample_crop
from .transforms import apply_random_crop_augmentations
from .types import Stage1Sample

SlideGrid = tuple[np.ndarray, tuple[int, int, int, int, int, int], int, np.ndarray]


class MOOZYDataset(Dataset):
    """
    Stage-1 MOOZY dataset that loads precomputed slide feature grids.

    Features are assumed to be from a fixed patch encoder backbone and stored
    as grids. We generate multi-crop views while batch-level masking is applied
    later during collation.
    Features can be preloaded at startup or lazily loaded per-sample. Optionally
    applies per-crop random horizontal/vertical flips and rotations. The full
    slide grid is never transformed in-place.
    """

    def __init__(
        self,
        feature_dirs: str | list[str],
        feature_h5_format: str = "auto",
        feature_h5_key: str | None = None,
        global_crop_size: int = 20,
        local_crop_size: int = 12,
        num_global_crops: int = 2,
        num_local_crops: int = 4,
        min_window_patch_ratio: float = 0.0,
        crop_resample_attempts: int = 3,
        debug: bool = False,
        preload_features: bool = True,
        max_cached_slides: int = 0,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.min_window_patch_ratio = float(min_window_patch_ratio)
        self.crop_resample_attempts = max(1, int(crop_resample_attempts))
        self.preload_features = bool(preload_features)
        self.max_cached_slides = max(0, int(max_cached_slides))
        self._slide_cache: "OrderedDict[int, SlideGrid]" = OrderedDict()
        self.patch_step_tolerance = SLIDE_GRID_STEP_TOLERANCE
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate_prob = float(rotate_prob)

        if isinstance(feature_dirs, str):
            feature_dirs = [feature_dirs]
        self.feature_h5_format = str(feature_h5_format or "auto")
        self.feature_h5_key = str(feature_h5_key).strip() if feature_h5_key else None

        h5_files = list_feature_paths(feature_dirs)

        if world_size > 1:
            h5_files = h5_files[rank::world_size]
            logging.info(
                "Sharding feature files across ranks: rank %s/%s loading %s files",
                rank,
                world_size,
                len(h5_files),
            )

        if debug:
            h5_files = h5_files[:400]

        if not h5_files:
            raise ValueError(f"No h5 files found in {feature_dirs}")

        self.features_list = []
        self.h5_files: list[str] = []
        self.axes_params: list[tuple[int, int, int, int, int, int]] = []
        self.patch_sizes = []
        self.valid_masks: list[np.ndarray] = []
        if self.preload_features:
            logging.info("Preloading %d feature files into memory...", len(h5_files))
            for h5_path in h5_files:
                try:
                    features_grid, axes_params, patch_size, valid_mask = self._load_slide_from_h5(h5_path)
                    g_h, g_w = features_grid.shape[:2]
                    valid_ratio = float(valid_mask.mean())
                    self.features_list.append(features_grid)
                    self.axes_params.append(axes_params)
                    self.patch_sizes.append(patch_size)
                    self.valid_masks.append(valid_mask)
                    self.h5_files.append(h5_path)
                    logging.debug(
                        "  Loaded grid for %s (grid %dx%d, valid_ratio=%.3f)",
                        h5_path,
                        g_h,
                        g_w,
                        valid_ratio,
                    )
                except Exception as exc:
                    logging.warning("  Failed to load %s: %s", h5_path, exc)

            if not self.features_list:
                raise ValueError(f"No features were successfully loaded from {feature_dirs}")

            logging.info("Preloaded %d feature grids into memory", len(self.features_list))
        else:
            self.h5_files = list(h5_files)
            logging.info(
                "Lazy feature loading enabled for %d files (max_cached_slides=%d)",
                len(self.h5_files),
                self.max_cached_slides,
            )

    def __len__(self) -> int:
        return len(self.features_list) if self.preload_features else len(self.h5_files)

    def _load_slide_from_h5(self, h5_path: str) -> SlideGrid:
        features_raw, coords, meta = load_feature_h5(
            h5_path,
            feature_h5_format=self.feature_h5_format,
            feature_h5_key=self.feature_h5_key,
        )
        patch_size = int(meta["patch_size_level0"])

        features_grid, xs_axis, ys_axis = build_grid_from_coords(
            features_raw,
            coords,
            expected_step=patch_size,
            step_tolerance=self.patch_step_tolerance,
        )
        zero_mask = np.all(features_grid == 0, axis=2)
        num_zeros = int(np.sum(zero_mask))
        if num_zeros > 0:
            logging.debug(
                "Found %d empty lattice positions after uniformization out of %d total.",
                num_zeros,
                features_grid.shape[0] * features_grid.shape[1],
            )
        axes_x_params = (int(xs_axis[0]), int(patch_size), int(xs_axis.shape[0]))
        axes_y_params = (int(ys_axis[0]), int(patch_size), int(ys_axis.shape[0]))

        g_h, g_w = features_grid.shape[:2]
        min_required = max(self.global_crop_size, self.local_crop_size)
        if g_h < min_required or g_w < min_required:
            pad_h = max(0, min_required - g_h)
            pad_w = max(0, min_required - g_w)
            features_grid = np.pad(
                features_grid,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            min_x, step_x, w = axes_x_params
            min_y, step_y, h = axes_y_params
            axes_x_params = (min_x, step_x, w + pad_w)
            axes_y_params = (min_y, step_y, h + pad_h)
            g_h, g_w = features_grid.shape[:2]
            logging.debug(
                "  Padded %s: grid %s -> %s to meet min crop %s",
                h5_path,
                (g_h - pad_h, g_w - pad_w),
                (g_h, g_w),
                min_required,
            )

        min_x, step_x, w = axes_x_params
        min_y, step_y, h = axes_y_params
        axes_x = (min_x + np.arange(w) * step_x).astype(np.int64)
        axes_y = (min_y + np.arange(h) * step_y).astype(np.int64)
        dx = np.diff(axes_x)
        dy = np.diff(axes_y)
        dx = dx[dx != 0]
        dy = dy[dy != 0]
        if dx.size > 0 and dy.size > 0:
            step_x_med = float(np.median(np.abs(dx)))
            step_y_med = float(np.median(np.abs(dy)))
            step = 0.5 * (step_x_med + step_y_med)
            tol = float(self.patch_step_tolerance)
            if not (patch_size * (1 - tol) <= step <= patch_size * (1 + tol)):
                logging.warning(
                    "  Grid step %.1f (x~%.1f, y~%.1f) mismatches patch_size_level0 %.1f",
                    step,
                    step_x_med,
                    step_y_med,
                    patch_size,
                )

        zero_mask = np.all(features_grid == 0, axis=2)
        valid_mask = ~zero_mask
        valid_ratio = float(valid_mask.mean())
        if self.min_window_patch_ratio > 0.0 and valid_ratio < self.min_window_patch_ratio:
            logging.debug(
                "  Low tissue ratio for %s: %.3f below threshold %.3f (will rely on crop resampling)",
                h5_path,
                valid_ratio,
                self.min_window_patch_ratio,
            )

        min_x, step_x, w = axes_x_params
        min_y, step_y, h = axes_y_params
        axes_params = (int(min_x), int(step_x), int(w), int(min_y), int(step_y), int(h))
        return features_grid, axes_params, patch_size, valid_mask

    def _get_slide(self, idx: int) -> SlideGrid:
        if self.preload_features:
            return (
                self.features_list[idx],
                self.axes_params[idx],
                self.patch_sizes[idx],
                self.valid_masks[idx],
            )

        if self.max_cached_slides > 0:
            cached = self._slide_cache.get(idx)
            if cached is not None:
                self._slide_cache.move_to_end(idx)
                return cached

        h5_path = self.h5_files[idx]
        try:
            slide = self._load_slide_from_h5(h5_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to lazily load slide {h5_path}: {exc}") from exc

        if self.max_cached_slides > 0:
            self._slide_cache[idx] = slide
            self._slide_cache.move_to_end(idx)
            while len(self._slide_cache) > self.max_cached_slides:
                self._slide_cache.popitem(last=False)

        return slide

    def __getitem__(self, idx: int) -> Stage1Sample:
        """Generate multi-crop views from a precomputed slide grid."""
        full_grid, axes_params, patch_size, window_valid = self._get_slide(idx)
        min_x, step_x, g_w, min_y, step_y, g_h = axes_params

        axes_x = (min_x + step_x * np.arange(g_w)).astype(np.int64)
        axes_y = (min_y + step_y * np.arange(g_h)).astype(np.int64)

        global_crops = []
        global_masks = []
        global_valids = []
        global_coords = []
        for _ in range(self.num_global_crops):
            crop, mask, valid, coords = sample_crop(
                full_grid,
                self.global_crop_size,
                valid_grid_mask=window_valid,
                x_axis=axes_x,
                y_axis=axes_y,
                min_valid_ratio=self.min_window_patch_ratio,
                max_attempts=self.crop_resample_attempts,
            )
            crop, mask, valid, coords = apply_random_crop_augmentations(
                crop,
                mask,
                valid,
                coords,
                hflip_prob=self.hflip_prob,
                vflip_prob=self.vflip_prob,
                rotate_prob=self.rotate_prob,
            )
            global_crops.append(crop)
            global_masks.append(mask)
            global_valids.append(valid)
            global_coords.append(coords)

        local_crops = []
        local_valids = []
        local_coords = []
        if self.num_local_crops > 0:
            for _ in range(self.num_local_crops):
                crop, mask, valid, coords = sample_crop(
                    full_grid,
                    self.local_crop_size,
                    valid_grid_mask=window_valid,
                    x_axis=axes_x,
                    y_axis=axes_y,
                    min_valid_ratio=self.min_window_patch_ratio,
                    max_attempts=self.crop_resample_attempts,
                )
                crop, mask, valid, coords = apply_random_crop_augmentations(
                    crop,
                    mask,
                    valid,
                    coords,
                    hflip_prob=self.hflip_prob,
                    vflip_prob=self.vflip_prob,
                    rotate_prob=self.rotate_prob,
                )
                local_crops.append(crop)
                local_valids.append(valid)
                local_coords.append(coords)

            local_crops_tensor = torch.stack(local_crops)
            local_valids_tensor = torch.stack(local_valids)
            local_coords_tensor = torch.stack(local_coords).to(torch.int64)
        else:
            feat_dim = full_grid.shape[2]
            size = self.local_crop_size
            local_crops_tensor = torch.zeros((0, size, size, feat_dim), dtype=torch.float32)
            local_valids_tensor = torch.zeros((0, size, size), dtype=torch.bool)
            local_coords_tensor = torch.zeros((0, size, size, 2), dtype=torch.int64)

        sample: Stage1Sample = {
            "global_crops": torch.stack(global_crops),
            "local_crops": local_crops_tensor,
            "global_masks": torch.stack(global_masks),
            "global_valids": torch.stack(global_valids),
            "local_valids": local_valids_tensor,
            "global_coords": torch.stack(global_coords).to(torch.int64),
            "local_coords": local_coords_tensor,
            "patch_sizes": torch.tensor(float(patch_size), dtype=torch.float32),
        }
        return sample
