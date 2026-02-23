"""Worker script for non-rigid registration on an Azure ML compute node.

Called by the ``register_f3d`` component defined in ``pipeline.py``.
Each invocation registers one floating image to the reference using
``reg_f3d`` via *niftyregw*.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from niftyregw import run

app = typer.Typer(add_completion=False)


@app.command()
def main(
    reference: Path = typer.Option(..., help="Reference image."),
    floating: Path = typer.Option(..., help="Floating image."),
    output_cpp: Path = typer.Option(..., help="Output control point grid."),
    output_result: Path = typer.Option(..., help="Output resampled image."),
    input_affine: Optional[Path] = typer.Option(None, help="Input affine."),
    reference_mask: Optional[Path] = typer.Option(None, help="Reference mask."),
    floating_mask: Optional[Path] = typer.Option(None, help="Floating mask."),
    extra_args: str = typer.Option("", help="Extra reg_f3d arguments."),
    save_result: bool = typer.Option(False, help="Save resampled image."),
) -> None:
    """Register a floating image to a reference using reg_f3d."""
    args: list[str] = []

    if extra_args:
        args.extend(extra_args.split())
    if input_affine is not None:
        args.extend(["-aff", str(input_affine)])
    if reference_mask is not None:
        args.extend(["-rmask", str(reference_mask)])
    if floating_mask is not None:
        args.extend(["-fmask", str(floating_mask)])

    res = str(output_result) if save_result else os.devnull

    args.extend([
        "-ref", str(reference),
        "-flo", str(floating),
        "-cpp", str(output_cpp),
        "-res", res,
    ])

    run("reg_f3d", *args)


if __name__ == "__main__":
    app()
