import json
import os
import uuid
from contextlib import suppress
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


def detect_feat_dim(h5_path: str, *, feature_h5_format: str = "auto", feature_h5_key: str = "") -> int:
    """Read the feature dimension from a single H5 file without loading all data."""
    features, _, _ = load_feature_h5(h5_path, feature_h5_format=feature_h5_format, feature_h5_key=feature_h5_key)
    return int(features.shape[1])


def _normalize_feature_h5_format(feature_h5_format: str) -> str:
    fmt = str(feature_h5_format or "auto").strip().lower()
    if fmt not in {"auto", "trident", "atlaspatch"}:
        raise ValueError(
            f"Unsupported feature_h5_format={feature_h5_format!r}. Expected one of {{'auto', 'trident', 'atlaspatch'}}."
        )
    return fmt


def _extract_meta(h5_path: str, meta_keys: Iterable[str], *attr_sources: Mapping[str, Any]) -> dict[str, int]:
    meta: dict[str, int] = {}
    for key in meta_keys:
        for attrs in attr_sources:
            if key not in attrs:
                continue
            meta[key] = int(attrs[key])
            break
        else:
            raise ValueError(f"{h5_path} is missing required metadata '{key}'.")
    return meta


def load_feature_h5(
    h5_path: str,
    *,
    required_meta_keys: Iterable[str] = (),
    validate_shapes: bool = False,
    feature_h5_format: str = "auto",
    feature_h5_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load precomputed features, normalized XY coordinates, and requested metadata from an H5 file."""
    meta_keys = ["patch_size_level0", "patch_size"]
    for key in required_meta_keys:
        if key not in meta_keys:
            meta_keys.append(key)

    format_hint = _normalize_feature_h5_format(feature_h5_format)
    requested_key = str(feature_h5_key or "").strip()
    if requested_key.startswith("features/"):
        requested_key = requested_key.split("/", 1)[1]

    with h5py.File(h5_path, "r") as handle:
        coords_ds = handle.get("coords")
        if not isinstance(coords_ds, h5py.Dataset):
            raise ValueError(f"{h5_path} is missing required dataset 'coords'.")

        features_node = handle.get("features")
        if isinstance(features_node, h5py.Dataset):
            detected_format = "trident"
            if format_hint not in {"auto", "trident"}:
                raise ValueError(
                    f"{h5_path} looks like a trident H5, but feature_h5_format={feature_h5_format!r} was requested."
                )
            features = features_node[()]
            meta = _extract_meta(h5_path, meta_keys, coords_ds.attrs, handle.attrs)
        elif isinstance(features_node, h5py.Group):
            detected_format = "atlaspatch"
            if format_hint not in {"auto", "atlaspatch"}:
                raise ValueError(
                    f"{h5_path} looks like an atlaspatch H5, but feature_h5_format={feature_h5_format!r} was requested."
                )
            available = sorted(name for name, obj in features_node.items() if isinstance(obj, h5py.Dataset))
            if not available:
                raise ValueError(f"{h5_path} does not contain any AtlasPatch feature datasets under 'features/'.")
            if requested_key:
                if requested_key not in available:
                    raise ValueError(
                        f"{h5_path} does not contain AtlasPatch feature dataset 'features/{requested_key}'. "
                        f"Available datasets: {', '.join(available)}"
                    )
                selected_key = requested_key
            else:
                if len(available) > 1:
                    raise ValueError(
                        f"{h5_path} contains multiple AtlasPatch feature datasets under 'features/': {', '.join(available)}. "
                        "Set feature_h5_key to select one."
                    )
                selected_key = available[0]
            features = features_node[selected_key][()]
            meta = _extract_meta(h5_path, meta_keys, handle.attrs, coords_ds.attrs)
        else:
            raise ValueError(
                f"{h5_path} is missing a supported 'features' node. Expected a dataset (TRIDENT) or group (AtlasPatch)."
            )

        features = np.asarray(features, dtype=np.float32)
        coords = np.asarray(coords_ds[()], dtype=np.int64)
        if coords.ndim != 2:
            raise ValueError(f"{h5_path} has invalid coords shape {coords.shape}; expected 2-D coordinates.")
        if detected_format == "atlaspatch":
            if coords.shape[1] < 2:
                raise ValueError(
                    f"{h5_path} has invalid AtlasPatch coords shape {coords.shape}; expected at least 2 columns."
                )
            coords = coords[:, :2]

        if validate_shapes:
            if features.ndim != 2:
                raise ValueError(f"{h5_path} has invalid features shape {features.shape}; expected 2-D [N, d]")
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"{h5_path} has invalid coords shape {coords.shape}; expected 2-D [N, 2]")
            if coords.shape[0] != features.shape[0]:
                raise ValueError(
                    f"{h5_path} has mismatched features/coords lengths: "
                    f"features {features.shape[0]} vs coords {coords.shape[0]}"
                )

    return features, coords, meta


def save_h5(
    save_path: str,
    assets: Mapping[str, np.ndarray],
    attributes: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Save arrays to HDF5 atomically with optional per-dataset attributes."""
    target_path = save_path
    tmp_path = None

    try:
        dir_name = os.path.dirname(os.path.abspath(target_path)) or "."
        base_name = os.path.basename(target_path)
        tmp_name = f".{base_name}.tmp.{uuid.uuid4().hex}"
        tmp_path = os.path.join(dir_name, tmp_name)

        with h5py.File(tmp_path, "w") as file:
            for key, value in assets.items():
                data_shape = value.shape
                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=(None, *data_shape[1:]),
                    chunks=(1, *data_shape[1:]),
                    dtype=value.dtype,
                )
                dset[:] = value
                if attributes is not None and key in attributes:
                    for attr_key, attr_val in attributes[key].items():
                        if isinstance(attr_val, dict):
                            attr_val = json.dumps(attr_val)
                        elif attr_val is None:
                            attr_val = "None"
                        dset.attrs[attr_key] = attr_val

        os.replace(tmp_path, target_path)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            with suppress(OSError):
                os.remove(tmp_path)


def is_valid_h5(
    path: str,
    required_datasets: dict[str, int | None] | None = None,
) -> bool:
    """Validate that an HDF5 file is readable and contains required datasets."""
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as handle:
            if required_datasets is None:
                return True
            for name, ndim in required_datasets.items():
                if name not in handle:
                    return False
                dset = handle[name]
                if ndim is not None and dset.ndim != ndim:
                    return False
        return True
    except Exception:
        return False
