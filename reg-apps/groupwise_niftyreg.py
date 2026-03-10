"""Groupwise image registration using NiftyReg.

Python equivalent of ``groupwise_niftyreg_run.sh``, built with Typer for an
excellent CLI experience and *niftyregw* for calling the NiftyReg binaries.
"""

from __future__ import annotations

import glob as globmod
import sys
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from loguru import logger
from niftyregw import run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Groupwise image registration using NiftyReg.",
)

IMAGE_EXTENSIONS = (".nii", ".nii.gz", ".hdr", ".img")


# ── helpers ──────────────────────────────────────────────────────────


def _setup_logger() -> None:
    """Configure *loguru* for CLI output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )


def _strip_image_extensions(path: Path) -> str:
    """Return the file *stem* with all known imaging extensions removed."""
    name = path.name
    for ext in (".nii.gz",):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _check_inputs(
    images: list[Path],
    template: Path,
    template_mask: Path | None,
    floating_masks: list[Path] | None,
) -> None:
    """Validate inputs, aborting on errors."""
    if len(images) < 2:
        logger.error("Less than 2 images have been specified")
        raise typer.Exit(code=1)
    if not template.exists():
        logger.error(f"The template image ({template}) does not exist")
        raise typer.Exit(code=1)
    if template_mask is not None and not template_mask.is_file():
        logger.error(
            f"The template image mask ({template_mask}) does not exist"
        )
        raise typer.Exit(code=1)
    if floating_masks:
        if len(floating_masks) != len(images):
            logger.error(
                "The number of images is different from the number of"
                " floating masks"
            )
            raise typer.Exit(code=1)
        for mask in floating_masks:
            if not mask.is_file():
                logger.error(f"Floating mask not found: {mask}")
                raise typer.Exit(code=1)


def _run_reg_average(args: list[str]) -> None:
    """Call ``reg_average`` via *niftyregw*."""
    tool_logger = logger.bind(executable="reg_average")
    run("reg_average", *args, tool_logger=tool_logger)


# ── affine loop ──────────────────────────────────────────────────────


def _affine_loop(
    images: list[Path],
    template: Path,
    template_mask: Path | None,
    floating_masks: list[Path] | None,
    result_dir: Path,
    affine_iterations: int,
    affine_extra_args: list[str],
) -> Path:
    """Run the rigid / affine registration loop.

    Returns the path to the last average image.
    """
    average_image = template
    tool_logger = logger.bind(executable="reg_aladin")

    for cur_it in range(1, affine_iterations + 1):
        it_dir = result_dir / f"aff_{cur_it}"
        avg_path = it_dir / f"average_affine_it_{cur_it}.nii.gz"

        if avg_path.is_file():
            logger.info(f"{avg_path} already exists")
            average_image = avg_path
            continue

        it_dir.mkdir(parents=True, exist_ok=True)

        # ── register each image ──────────────────────────────────
        for i, img in enumerate(images):
            name = _strip_image_extensions(img)
            mat_path = it_dir / f"aff_mat_{name}_it{cur_it}.txt"

            if mat_path.is_file():
                continue

            args: list[str] = list(affine_extra_args)

            # First iteration is rigid-only
            if cur_it == 1:
                args.append("-rigOnly")
            else:
                # Use previous affine for initialisation
                prev_mat = (
                    result_dir
                    / f"aff_{cur_it - 1}"
                    / f"aff_mat_{name}_it{cur_it - 1}.txt"
                )
                if prev_mat.is_file():
                    args.extend(["-inaff", str(prev_mat)])

            if template_mask is not None:
                args.extend(["-rmask", str(template_mask)])
            if floating_masks:
                args.extend(["-fmask", str(floating_masks[i])])

            # Only save the resampled result on the last iteration
            if cur_it == affine_iterations:
                res_path = str(
                    it_dir / f"aff_res_{name}_it{cur_it}.nii.gz"
                )
            else:
                res_path = "/dev/null"

            log_path = it_dir / f"aff_log_{name}_it{cur_it}.txt"

            args.extend([
                "-ref", str(average_image),
                "-flo", str(img),
                "-aff", str(mat_path),
                "-res", res_path,
            ])

            logger.info(
                f"Affine iteration {cur_it}/{affine_iterations} – "
                f"image {i + 1}/{len(images)} ({name})"
            )
            run("reg_aladin", *args, tool_logger=tool_logger)

            if not mat_path.is_file():
                logger.error(f"Error when creating {mat_path}")
                raise typer.Exit(code=1)

        # ── create the average image ─────────────────────────────
        if cur_it != affine_iterations:
            # Intermediate iterations: demean transformations
            avg_args: list[str] = [
                str(avg_path),
                "-demean",
                str(average_image),
            ]
            for img in images:
                name = _strip_image_extensions(img)
                avg_args.append(
                    str(it_dir / f"aff_mat_{name}_it{cur_it}.txt")
                )
                avg_args.append(str(img))

            logger.info(f"Creating demeaned average (affine it {cur_it})")
            _run_reg_average(avg_args)
        else:
            # Last iteration: direct average of resampled images
            res_images = sorted(
                globmod.glob(
                    str(it_dir / f"aff_res_*_it{cur_it}.nii*")
                )
            )
            avg_args = [str(avg_path), "-avg", *res_images]
            logger.info(
                f"Creating average of resampled images"
                f" (affine it {cur_it})"
            )
            _run_reg_average(avg_args)

        if not avg_path.is_file():
            logger.error(f"Error when creating {avg_path}")
            raise typer.Exit(code=1)

        average_image = avg_path

    return average_image


# ── non-rigid loop ───────────────────────────────────────────────────


def _nonrigid_loop(
    images: list[Path],
    template_mask: Path | None,
    floating_masks: list[Path] | None,
    result_dir: Path,
    average_image: Path,
    affine_iterations: int,
    nrr_iterations: int,
    nrr_extra_args: list[str],
) -> Path:
    """Run the non-rigid registration loop.

    Returns the path to the last average image.
    """
    tool_logger = logger.bind(executable="reg_f3d")

    for cur_it in range(1, nrr_iterations + 1):
        it_dir = result_dir / f"nrr_{cur_it}"
        avg_path = it_dir / f"average_nonrigid_it_{cur_it}.nii.gz"

        if avg_path.is_file():
            logger.info(f"{avg_path} already exists")
            average_image = avg_path
            continue

        it_dir.mkdir(parents=True, exist_ok=True)

        # ── register each image ──────────────────────────────────
        for i, img in enumerate(images):
            name = _strip_image_extensions(img)
            cpp_path = it_dir / f"nrr_cpp_{name}_it{cur_it}.nii.gz"

            if cpp_path.is_file():
                continue

            args: list[str] = list(nrr_extra_args)

            if template_mask is not None:
                args.extend(["-rmask", str(template_mask)])
            if floating_masks:
                args.extend(["-fmask", str(floating_masks[i])])

            # Use affine from the last affine iteration
            if affine_iterations > 0:
                aff_mat = (
                    result_dir
                    / f"aff_{affine_iterations}"
                    / f"aff_mat_{name}_it{affine_iterations}.txt"
                )
                args.extend(["-aff", str(aff_mat)])

            # Only save the resampled result on the last iteration
            if cur_it == nrr_iterations:
                res_path = str(
                    it_dir / f"nrr_res_{name}_it{cur_it}.nii.gz"
                )
            else:
                res_path = "/dev/null"

            args.extend([
                "-ref", str(average_image),
                "-flo", str(img),
                "-cpp", str(cpp_path),
                "-res", res_path,
            ])

            logger.info(
                f"Non-rigid iteration {cur_it}/{nrr_iterations} – "
                f"image {i + 1}/{len(images)} ({name})"
            )
            run("reg_f3d", *args, tool_logger=tool_logger)

        # ── create the average image ─────────────────────────────
        if cur_it != nrr_iterations:
            # Intermediate iterations: demean_noaff
            avg_args: list[str] = [
                str(avg_path),
                "-demean_noaff",
                str(average_image),
            ]
            for img in images:
                name = _strip_image_extensions(img)
                avg_args.append(
                    str(
                        result_dir
                        / f"aff_{affine_iterations}"
                        / f"aff_mat_{name}_it{affine_iterations}.txt"
                    )
                )
                avg_args.append(
                    str(it_dir / f"nrr_cpp_{name}_it{cur_it}.nii.gz")
                )
                avg_args.append(str(img))

            logger.info(
                f"Creating demeaned average (non-rigid it {cur_it})"
            )
            _run_reg_average(avg_args)
        else:
            # Last iteration: direct average of resampled images
            res_images = sorted(
                globmod.glob(
                    str(it_dir / f"nrr_res_*_it{cur_it}.nii*")
                )
            )
            avg_args = [str(avg_path), "-avg", *res_images]
            logger.info(
                f"Creating average of resampled images"
                f" (non-rigid it {cur_it})"
            )
            _run_reg_average(avg_args)

        if not avg_path.is_file():
            logger.error(f"Error when creating {avg_path}")
            raise typer.Exit(code=1)

        average_image = avg_path

    return average_image


# ── main command ─────────────────────────────────────────────────────


@app.command()
def main(
    images: Annotated[
        List[Path],
        typer.Argument(
            help="Input images to groupwise register (at least 2).",
        ),
    ],
    template: Annotated[
        Path,
        typer.Option(
            "--template",
            "-t",
            help=(
                "Template image used to initialise the registration."
                " Defaults to the first input image."
            ),
        ),
    ] = None,  # type: ignore[assignment]
    result_dir: Annotated[
        Path,
        typer.Option(
            "--result-dir",
            "-d",
            help="Directory where results are saved.",
        ),
    ] = Path("groupwise_result"),
    template_mask: Annotated[
        Optional[Path],
        typer.Option(
            "--template-mask",
            help="Mask in the template (reference) space.",
        ),
    ] = None,
    floating_masks: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--floating-mask",
            help=(
                "Masks in the floating spaces.  Provide one per input"
                " image, in the same order."
            ),
        ),
    ] = None,
    affine_iterations: Annotated[
        int,
        typer.Option(
            "--affine-iterations",
            "-a",
            min=0,
            help=(
                "Number of affine iterations (the first is always rigid)."
            ),
        ),
    ] = 5,
    nrr_iterations: Annotated[
        int,
        typer.Option(
            "--nrr-iterations",
            "-n",
            min=0,
            help="Number of non-rigid iterations.",
        ),
    ] = 10,
    affine_extra_args: Annotated[
        Optional[str],
        typer.Option(
            "--affine-args",
            help=(
                "Extra arguments forwarded to reg_aladin"
                ' (e.g. "--affine-args=\'-omp 4\'").'
            ),
        ),
    ] = None,
    nrr_extra_args: Annotated[
        Optional[str],
        typer.Option(
            "--nrr-args",
            help=(
                "Extra arguments forwarded to reg_f3d"
                ' (e.g. "--nrr-args=\'-omp 4\'").'
            ),
        ),
    ] = None,
) -> None:
    """Run groupwise image registration using NiftyReg.

    Iteratively registers all input images to a common average template
    using affine (rigid on the first step) and non-rigid registration.
    """
    _setup_logger()

    # Default template is the first input image
    if template is None:
        template = images[0]

    _check_inputs(images, template, template_mask, floating_masks)

    logger.info(
        f"There are {len(images)} input images to groupwise register"
    )
    logger.info(
        f"The template image to initialise the registration is {template}"
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    aff_extra = affine_extra_args.split() if affine_extra_args else []
    nrr_extra = nrr_extra_args.split() if nrr_extra_args else []

    average_image = _affine_loop(
        images=images,
        template=template,
        template_mask=template_mask,
        floating_masks=floating_masks,
        result_dir=result_dir,
        affine_iterations=affine_iterations,
        affine_extra_args=aff_extra,
    )

    _nonrigid_loop(
        images=images,
        template_mask=template_mask,
        floating_masks=floating_masks,
        result_dir=result_dir,
        average_image=average_image,
        affine_iterations=affine_iterations,
        nrr_iterations=nrr_iterations,
        nrr_extra_args=nrr_extra,
    )


if __name__ == "__main__":
    app()
