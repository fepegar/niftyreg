"""Worker script for averaging on an Azure ML compute node.

Called by the ``average_images`` component defined in ``pipeline.py``.
Gathers transformation/image pairs and calls ``reg_average`` via
*niftyregw*.
"""

from __future__ import annotations

from pathlib import Path

import typer
from niftyregw import run

app = typer.Typer(add_completion=False)


@app.command()
def main(
    transforms_and_images: Path = typer.Option(
        ..., help="Folder with transformation and image files."
    ),
    reference: Path = typer.Option(..., help="Reference image."),
    output_average: Path = typer.Option(..., help="Output average image."),
    mode: str = typer.Option(
        "demean",
        help="Averaging mode: avg, demean, or demean_noaff.",
    ),
) -> None:
    """Average or demean registered images/transformations."""
    folder = Path(transforms_and_images)

    if mode == "avg":
        images = sorted(folder.glob("*.nii*"))
        args = [str(output_average), "-avg", *[str(p) for p in images]]
    elif mode == "demean":
        # Expect pairs: transform.txt + image.nii.gz
        transforms = sorted(folder.glob("*.txt"))
        images = sorted(folder.glob("*.nii*"))
        args = [str(output_average), "-demean", str(reference)]
        for trans, img in zip(transforms, images):
            args.extend([str(trans), str(img)])
    elif mode == "demean_noaff":
        # Expect triples: affine.txt + cpp.nii.gz + image.nii.gz
        affines = sorted(folder.glob("aff_*.txt"))
        cpps = sorted(folder.glob("nrr_*.nii*"))
        images = sorted(folder.glob("img_*.nii*"))
        args = [str(output_average), "-demean_noaff", str(reference)]
        for aff, cpp, img in zip(affines, cpps, images):
            args.extend([str(aff), str(cpp), str(img)])
    else:
        typer.echo(f"Unknown mode: {mode}", err=True)
        raise typer.Exit(code=1)

    run("reg_average", *args)


if __name__ == "__main__":
    app()
