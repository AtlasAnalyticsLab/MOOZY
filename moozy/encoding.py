import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
from torch.amp import autocast

from moozy.data.features import INFERENCE_TOKEN_PRESETS, resolve_vram_token_cap, save_h5
from moozy.data.stage2 import SlideSample, build_case_sample, collate_stage2_batch, load_stage2_slide_sample
from moozy.hf_hub import ensure_checkpoint
from moozy.models.factory import load_stage2_inference_model
from moozy.models.stage2_supervised import MOOZY

_FEATURE_H5_KEY = "lunit_vit_small_patch8_dino"
_PATCH_SIZE = 224
_H5_EXTENSIONS = frozenset({".h5", ".hdf5"})

_ATLASPATCH_INSTALL_URL = "https://github.com/AtlasAnalyticsLab/AtlasPatch#installation"

# Calibrated for lunit ViT-S/8 at 224x224: ~7 MB/sample fp16, ~14 MB/sample fp32.
_BYTES_PER_SAMPLE_FP16 = 7 * 1024 * 1024
_BYTES_PER_SAMPLE_FP32 = 14 * 1024 * 1024


def _is_h5_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in _H5_EXTENSIONS


def _atlaspatch_error(stderr: str, returncode: int) -> RuntimeError:
    """Build a helpful RuntimeError from AtlasPatch failure output."""
    hint = ""
    lower = stderr.lower()
    if "no module named" in lower and "sam2" in lower:
        hint = (
            "\nSAM2 is not installed. Install it with:\n  pip install git+https://github.com/facebookresearch/sam2.git"
        )
    elif "openslide" in lower:
        hint = (
            "\nOpenSlide is not installed. Install it with:"
            "\n  conda install -c conda-forge openslide   # or"
            "\n  sudo apt-get install openslide-tools"
        )
    if not hint:
        hint = f"\nSee installation guide: {_ATLASPATCH_INSTALL_URL}"
    return RuntimeError(f"AtlasPatch failed (exit code {returncode}):\n{stderr.strip()}{hint}")


def _extract_features_with_atlaspatch(
    slide_paths: list[str],
    output_dir: str,
    target_mag: int,
    step_size: int,
    mixed_precision: bool,
    mpp_csv: str | None,
    logger: logging.Logger,
) -> list[str]:
    """Run AtlasPatch to extract lunit features from raw slides into *output_dir*."""
    for path in slide_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Slide file not found: {path}")
    if mpp_csv and not os.path.isfile(mpp_csv):
        raise FileNotFoundError(f"MPP CSV not found: {mpp_csv}")
    if shutil.which("atlaspatch") is None:
        raise RuntimeError(
            "atlaspatch command not found. Install it with:"
            "\n  pip install atlas-patch"
            "\n  pip install git+https://github.com/facebookresearch/sam2.git"
            f"\nSee: {_ATLASPATCH_INSTALL_URL}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = min(os.cpu_count() or 4, 8)
    precision = "bfloat16" if mixed_precision else "float32"

    if device.startswith("cuda"):
        free_bytes, _ = torch.cuda.mem_get_info(0)
        usable = int(free_bytes * 0.6)  # room for patch encoder itself, SAM2, etc
        per_sample = _BYTES_PER_SAMPLE_FP32 if precision == "float32" else _BYTES_PER_SAMPLE_FP16
        batch_size = max(32, min(4096, int(usable / per_sample)))
    else:
        batch_size = 32

    if len(slide_paths) == 1:
        input_path = os.path.abspath(slide_paths[0])
    else:
        input_dir = os.path.join(output_dir, "_slides")
        os.makedirs(input_dir)
        stems_seen: set[str] = set()
        for path in slide_paths:
            basename = os.path.basename(path)
            stem = os.path.splitext(basename)[0]
            if stem in stems_seen:
                raise ValueError(f"Duplicate slide stem '{stem}' — all slides must have unique filenames.")
            stems_seen.add(stem)
            os.symlink(os.path.abspath(path), os.path.join(input_dir, basename))
        input_path = input_dir

    cmd = [
        "atlaspatch",
        "process",
        input_path,
        "--output",
        output_dir,
        "--patch-size",
        str(_PATCH_SIZE),
        "--target-mag",
        str(target_mag),
        "--step-size",
        str(step_size),
        "--fast-mode",
        "--feature-extractors",
        _FEATURE_H5_KEY,
        "--device",
        device,
        "--feature-batch-size",
        str(batch_size),
        "--feature-num-workers",
        str(num_workers),
        "--feature-precision",
        precision,
    ]
    if mpp_csv:
        cmd.extend(["--mpp-csv", mpp_csv])
    logger.info("Running AtlasPatch: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise _atlaspatch_error(result.stderr, result.returncode)

    patches_dir = os.path.join(output_dir, "patches")
    h5_paths: list[str] = []
    for slide_path in slide_paths:
        stem = os.path.splitext(os.path.basename(slide_path))[0]
        h5_path = os.path.join(patches_dir, f"{stem}.h5")
        if not os.path.exists(h5_path):
            raise RuntimeError(f"AtlasPatch did not produce expected output: {h5_path}")
        h5_paths.append(h5_path)

    return h5_paths


def run_case_encoding(
    model: MOOZY,
    slides: list[SlideSample],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, torch.Tensor]:
    """Run forward pass on a single case (one or more slides)."""
    if not slides:
        raise ValueError("Empty case batch; no slides were provided.")

    model.eval()
    model.slide_encoder.eval()
    model.case_transformer.eval()

    batch = collate_stage2_batch([build_case_sample(slides=slides)])
    amp_ctx = (
        autocast(device_type="cuda", dtype=torch.bfloat16)
        if mixed_precision and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        with amp_ctx:
            return model(batch)


def run_encoding(
    slide_paths: list[str],
    output_path: str,
    *,
    mixed_precision: bool = False,
    target_mag: int = 20,
    step_size: int = 224,
    mpp_csv: str | None = None,
) -> None:
    """Encode one or more slides into a case-level embedding using MOOZY.

    Accepts either pre-computed H5 feature files (.h5, .hdf5) or raw whole-slide
    images (.svs, .tiff, .ndpi, etc.). When raw slides are provided, feature
    extraction is performed automatically via AtlasPatch using the
    ``lunit_vit_small_patch8_dino`` encoder at 224x224 patch size. The input type
    is determined by file extension. Mixing H5 and raw slides is not allowed.

    The output is an H5 file containing a ``features`` dataset with the case
    embedding vector and a ``coords`` dataset with slide-level metadata.

    Args:
        slide_paths: Paths to H5 feature files or raw slide images. All paths
            must be the same type (all H5 or all slides).
        output_path: Destination path for the output H5 file. Parent directories
            are created automatically.
        mixed_precision: Use bfloat16 autocast for MOOZY inference and feature
            extraction. When False, both run in float32.
        target_mag: Target magnification for patch extraction from raw slides
            (e.g., 5, 10, 20, 40). Ignored for H5 inputs.
        step_size: Stride in pixels between patch centers at target magnification.
            Defaults to 224 (non-overlapping). Ignored for H5 inputs.
        mpp_csv: Path to a CSV file with ``wsi,mpp`` columns providing custom
            microns-per-pixel overrides. Ignored for H5 inputs.

    Raises:
        ValueError: If ``slide_paths`` is empty or mixes H5 and raw slide files.
        FileNotFoundError: If a slide path or ``mpp_csv`` does not exist.
        RuntimeError: If AtlasPatch is not installed or fails during extraction.

    Example::

        from moozy.encoding import run_encoding

        # From pre-computed H5 feature files
        run_encoding(["slide_1.h5", "slide_2.h5"], "case.h5")

        # From raw slides with mixed precision
        run_encoding(
            ["tumor.svs", "normal.svs"],
            "case.h5",
            mixed_precision=True,
            target_mag=20,
        )
    """
    logger = logging.getLogger(__name__)

    if not slide_paths:
        raise ValueError("No slide paths provided.")

    h5_inputs = [p for p in slide_paths if _is_h5_file(p)]
    slide_inputs = [p for p in slide_paths if not _is_h5_file(p)]
    if h5_inputs and slide_inputs:
        raise ValueError("Cannot mix H5 feature files and raw slide files. Provide all H5 or all slides.")

    tmp_dir = None
    try:
        if slide_inputs:
            logger.info("Detected raw slide inputs — extracting features with AtlasPatch.")
            tmp_dir = tempfile.TemporaryDirectory(prefix="moozy_features_")
            feature_paths = _extract_features_with_atlaspatch(
                slide_inputs,
                tmp_dir.name,
                target_mag=target_mag,
                step_size=step_size,
                mixed_precision=mixed_precision,
                mpp_csv=mpp_csv,
                logger=logger,
            )
        else:
            feature_paths = slide_paths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_tokens = resolve_vram_token_cap(presets=INFERENCE_TOKEN_PRESETS, logger=logger, device=device)
        checkpoint_path = ensure_checkpoint()
        model = load_stage2_inference_model(checkpoint_path, device=device)

        slides: list[SlideSample] = []
        for path in feature_paths:
            slides.append(
                load_stage2_slide_sample(
                    os.path.basename(path),
                    path,
                    include_geometry_meta=True,
                    max_valid_tokens_per_slide=max_tokens,
                    feature_h5_format="auto",
                    feature_h5_key=_FEATURE_H5_KEY,
                )
            )

        outputs = run_case_encoding(model, slides, device=device, mixed_precision=mixed_precision)

        case_vec = outputs.get("cls")
        if case_vec is None:
            raise RuntimeError("Model did not return a 'cls' embedding.")

        case_np = case_vec.detach().float().cpu().numpy().squeeze(0)
        features = case_np.astype(np.float32)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        slide_meta = slides[0]
        attributes = {
            "coords": {
                "columns": ["x", "y"],
                "patch_size_level0": slide_meta["patch_size_level0"],
                "patch_size": slide_meta["patch_size_value"],
                "num_slides": len(slides),
            },
        }
        save_h5(
            output_path,
            assets={"features": features, "coords": np.zeros((2,), dtype=np.int64)},
            attributes=attributes,
        )
        logger.info("Wrote case embedding to %s (%d slides).", output_path, len(slides))
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
