import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from moozy.data.features import SLIDE_GRID_STEP_TOLERANCE, build_grid_from_coords, load_feature_h5

from .batching import build_case_sample
from .transforms import (
    apply_max_valid_tokens,
    apply_slide_augmentations,
    apply_token_dropout,
    compact_slide_to_valid_tokens,
)
from .types import CaseRecord, CaseSample, SlideSample


def load_stage2_slide_sample(
    name: str,
    h5_path: str,
    *,
    include_geometry_meta: bool = False,
    max_valid_tokens_per_slide: int = 0,
    feature_h5_format: str = "auto",
    feature_h5_key: str | None = None,
) -> SlideSample:
    features, coords, meta = load_feature_h5(
        h5_path,
        feature_h5_format=feature_h5_format,
        feature_h5_key=feature_h5_key,
    )
    patch_size = float(meta["patch_size_level0"])
    grid, xs_sorted, ys_sorted = build_grid_from_coords(
        features,
        coords,
        expected_step=patch_size,
        step_tolerance=SLIDE_GRID_STEP_TOLERANCE,
    )
    valid_mask = (grid != 0).any(axis=2)
    xx, yy = np.meshgrid(xs_sorted, ys_sorted, indexing="xy")
    slide: SlideSample = {
        "name": name,
        "x": torch.from_numpy(grid).float(),
        "invalid": torch.from_numpy(~valid_mask).bool(),
        "coords": torch.from_numpy(np.stack([xx, yy], axis=-1)).float(),
        "patch_size": torch.tensor(patch_size, dtype=torch.float32),
    }
    if include_geometry_meta:
        center_x = int(np.floor((float(xs_sorted.min()) + float(xs_sorted.max())) / 2.0))
        center_y = int(np.floor((float(ys_sorted.min()) + float(ys_sorted.max())) / 2.0))
        slide.update(
            {
                "center": np.asarray([center_x, center_y], dtype=np.int64),
                "grid_h": int(grid.shape[0]),
                "grid_w": int(grid.shape[1]),
                "patch_size_level0": int(meta["patch_size_level0"]),
                "patch_size_value": int(meta["patch_size"]),
            }
        )
    if int(max_valid_tokens_per_slide) > 0:
        slide = apply_max_valid_tokens(slide, int(max_valid_tokens_per_slide), sampling="deterministic")
    return slide


class SupervisedCaseDataset(Dataset):
    """Dataset that loads all slides for a case and attaches task labels."""

    def __init__(
        self,
        cases: list[CaseRecord],
        labels,
        *,
        events=None,
        times=None,
        augment: bool = False,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rotate_prob: float = 0.0,
        token_dropout_ratio: float = 0.0,
        max_valid_tokens_per_slide: int = 0,
        token_cap_sampling: str = "deterministic",
        preload_features: bool = False,
        max_cached_slides: int = 0,
        feature_h5_format: str = "auto",
        feature_h5_key: str | None = None,
    ):
        self.cases = cases
        self.labels = labels
        self.events = events
        self.times = times
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate_prob = float(rotate_prob)
        self.token_dropout_ratio = float(token_dropout_ratio)
        self.max_valid_tokens_per_slide = max(0, int(max_valid_tokens_per_slide))
        self.token_cap_sampling = str(token_cap_sampling).strip().lower()
        if self.token_cap_sampling not in {"deterministic", "random_stratified"}:
            raise ValueError(
                f"Unsupported token_cap_sampling={token_cap_sampling!r}. "
                "Expected one of {'deterministic', 'random_stratified'}."
            )
        self.preload_features = bool(preload_features)
        self.max_cached_slides = max(0, int(max_cached_slides))
        self.feature_h5_format = str(feature_h5_format or "auto")
        self.feature_h5_key = str(feature_h5_key).strip() if feature_h5_key else None
        self._slide_cache: "OrderedDict[str, SlideSample]" = OrderedDict()

        if self.preload_features:
            self._preloaded: dict[str, SlideSample] = {}
            all_paths = set()
            for case in cases:
                path_options = case.get("path_options")
                if path_options is not None:
                    for options in path_options:
                        all_paths.update(options)
                else:
                    all_paths.update(case["paths"])
            logging.info("Preloading %d unique slide files into memory...", len(all_paths))
            for path in sorted(all_paths):
                self._preloaded[path] = load_stage2_slide_sample(
                    os.path.basename(path),
                    path,
                    feature_h5_format=self.feature_h5_format,
                    feature_h5_key=self.feature_h5_key,
                )
            logging.info("Preloaded %d slide files.", len(self._preloaded))
        else:
            self._preloaded = {}
            logging.info(
                "Lazy feature loading for supervised dataset (max_cached_slides=%d).",
                self.max_cached_slides,
            )

    def _get_slide(self, name: str, h5_path: str) -> SlideSample:
        if self.preload_features:
            return {**self._preloaded[h5_path], "name": name}

        if self.max_cached_slides > 0:
            cached = self._slide_cache.get(h5_path)
            if cached is not None:
                self._slide_cache.move_to_end(h5_path)
                return {**cached, "name": name}

        slide = load_stage2_slide_sample(
            name,
            h5_path,
            feature_h5_format=self.feature_h5_format,
            feature_h5_key=self.feature_h5_key,
        )
        if self.max_cached_slides > 0:
            self._slide_cache[h5_path] = slide
            if len(self._slide_cache) > self.max_cached_slides:
                self._slide_cache.popitem(last=False)
        return {**slide, "name": name}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> CaseSample:
        case = self.cases[idx]
        case_id = str(case["case_id"])
        path_options = case.get("path_options")
        if path_options is not None:
            paths = []
            for options in path_options:
                if not options:
                    continue
                if self.augment and len(options) > 1:
                    paths.append(random.choice(options))
                else:
                    paths.append(options[0])
        else:
            paths = case["paths"]

        slides = []
        for path in paths:
            slide = self._get_slide(f"{case_id}:{os.path.basename(path)}", path)
            if self.augment:
                slide = apply_slide_augmentations(
                    slide,
                    hflip_prob=self.hflip_prob,
                    vflip_prob=self.vflip_prob,
                    rotate_prob=self.rotate_prob,
                )
            if self.token_dropout_ratio > 0.0:
                slide = apply_token_dropout(slide, self.token_dropout_ratio)
            if self.max_valid_tokens_per_slide > 0:
                slide = apply_max_valid_tokens(
                    slide,
                    self.max_valid_tokens_per_slide,
                    sampling=self.token_cap_sampling,
                )
            slides.append(compact_slide_to_valid_tokens(slide))

        return build_case_sample(
            case_id=case_id,
            slides=slides,
            task_labels=torch.as_tensor(self.labels[idx], dtype=torch.int64),
            task_events=(torch.as_tensor(self.events[idx], dtype=torch.int64) if self.events is not None else None),
            task_times=(torch.as_tensor(self.times[idx], dtype=torch.float32) if self.times is not None else None),
        )
