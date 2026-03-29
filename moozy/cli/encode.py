from typing import Annotated

import typer


def encode_command(
    slides: Annotated[
        list[str],
        typer.Argument(
            help="H5 feature files or raw slides (.svs, .tiff, etc.) to process as a single case.",
        ),
    ],
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output H5 path for the case embedding."),
    ],
    mixed_precision: Annotated[
        bool,
        typer.Option("--mixed_precision/--no_mixed_precision", help="Enable bf16 autocast."),
    ] = False,
    target_mag: Annotated[
        int,
        typer.Option("--target_mag", help="Target magnification for patching (raw slides only)."),
    ] = 20,
    step_size: Annotated[
        int,
        typer.Option("--step_size", help="Stride between patches in pixels (raw slides only)."),
    ] = 224,
    mpp_csv: Annotated[
        str,
        typer.Option("--mpp_csv", help="CSV with wsi,mpp columns for microns-per-pixel override (raw slides only)."),
    ] = "",
) -> None:
    """Encode slides into a case-level embedding with a MOOZY checkpoint."""
    import logging

    from moozy.encoding import run_encoding

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_encoding(
        slide_paths=list(slides),
        output_path=output,
        mixed_precision=mixed_precision,
        target_mag=target_mag,
        step_size=step_size,
        mpp_csv=mpp_csv or None,
    )
