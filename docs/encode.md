# Encoding

- [Usage Guide](#usage-guide)
  - [Pre-computed H5 feature files](#pre-computed-h5-feature-files)
  - [Raw whole-slide images](#raw-whole-slide-images)
- [Arguments](#arguments)
- [Python API](#python-api)

## Usage Guide

`moozy encode` accepts two types of input, determined automatically by file extension.

### Pre-computed H5 feature files

Pass `.h5` or `.hdf5` files that already contain patch-level feature vectors. This is the faster path since feature extraction is already done. MOOZY is compatible with H5 files produced by both [AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch) and [TRIDENT](https://github.com/mahmoodlab/TRIDENT). In either case, the patch encoder used must be `lunit_vit_small_patch8_dino` (called `lunit-vits8` in TRIDENT), this is the patch encoder MOOZY was trained with.

```bash
moozy encode slide_1.h5 slide_2.h5 --output case_embedding.h5
```

### Raw whole-slide images

Pass slide files directly (`.svs`, `.tiff`, `.ndpi`, `.mrxs`, `.scn`, `.dcm`, etc.). MOOZY calls [AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch) under the hood to segment tissue, extract patches, and compute `lunit_vit_small_patch8_dino` features for those extracted patches. The patch size is fixed at 224x224 pixels because this is what MOOZY was trained on. Device, batch size, and worker count are resolved automatically from the system.

This requires `atlas-patch`, `sam2`, and the OpenSlide system library. See the [AtlasPatch installation guide](https://github.com/AtlasAnalyticsLab/AtlasPatch#installation) for full setup instructions.

```bash
moozy encode slide_1.svs slide_2.svs --output case_embedding.h5 --target_mag 20
```

For full control over the feature extraction parameters (segmentation thresholds, visualization, etc.), run [`atlaspatch process`](https://github.com/AtlasAnalyticsLab/AtlasPatch#process) directly with `--feature-extractors lunit_vit_small_patch8_dino` and pass the resulting H5 files to `moozy encode`.

> You cannot mix H5 files and raw slides in a single invocation.

Encode multiple slides (one case) from raw WSIs at 40x with overlapping patches:

```bash
moozy encode tumor.svs normal.svs \
  --output case_42.h5 \
  --target_mag 40 \
  --step_size 112 \
  --mixed_precision
```

Encode with a custom MPP override file:

```bash
moozy encode slide.ndpi --output out.h5 --target_mag 20 --mpp_csv overrides.csv
```

The MPP CSV is a two-column file mapping slide filenames to their microns-per-pixel values:

```csv
wsi,mpp
slide_A.ndpi,0.2528
slide_B.svs,0.5016
```

Slides are matched by stem (filename without extension), so `slide_A.ndpi` and `slide_A` both match the same row.

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `SLIDES` | path(s) | yes | - | One or more input files that together form a single case. All files must be either H5 feature files (`.h5`, `.hdf5`) or raw whole-slide images (`.svs`, `.tiff`, etc.). Mixing the two types is not allowed. Each file represents one slide. The model aggregates all slides into one case-level embedding. |
| `--output`, `-o` | path | yes | - | Destination path for the output H5 file. Parent directories are created if they do not exist. |
| `--mixed_precision` / `--no_mixed_precision` | flag | no | `--no_mixed_precision` | Enable bfloat16 automatic mixed precision during both MOOZY inference and feature extraction (when using raw slides). Reduces VRAM usage and can speed up inference on Ampere+ GPUs. When disabled, MOOZY runs in float32 and feature extraction uses float32. |
| `--target_mag` | int | no | `20` | Target magnification level for patch extraction from raw slides (e.g., 5, 10, 20, 40). Controls the resolution at which tissue patches are read from the whole-slide image pyramid. Higher magnification yields more detail per patch but produces more patches. Ignored when passing H5 files. |
| `--step_size` | int | no | `224` | Stride in pixels between adjacent patch centers at the target magnification. Equal to the patch size (224) by default, producing a non-overlapping grid. Set lower than 224 for overlapping patches (e.g., 112 for 50% overlap). Ignored when passing H5 files. |
| `--mpp_csv` | path | no | - | Path to a CSV file with columns `wsi,mpp` that provides custom microns-per-pixel values for slides whose metadata is missing or inaccurate. Passed directly to AtlasPatch. Ignored when passing H5 files. See the example above for the expected format. |

## Python API

The encoding pipeline can also be called directly from Python:

```python
from moozy.encoding import run_encoding

# From pre-computed H5 feature files
run_encoding(
    slide_paths=["slide_1.h5", "slide_2.h5"],
    output_path="case_embedding.h5",
)

# From raw whole-slide images (requires atlas-patch)
run_encoding(
    slide_paths=["slide_1.svs", "slide_2.svs"],
    output_path="case_embedding.h5",
    target_mag=20,
    step_size=224,
)

# With mixed precision and MPP override
run_encoding(
    slide_paths=["slide.ndpi"],
    output_path="case_embedding.h5",
    mixed_precision=True,
    target_mag=40,
    mpp_csv="overrides.csv",
)
```
