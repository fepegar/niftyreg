"""Azure Machine Learning pipeline for groupwise template generation.

Distributes pairwise registrations across many CPU nodes using Azure ML
parallel jobs, then averages the results to build a population template.

Workflow
--------
For each affine iteration::

    ┌─────────────────────────────────────────────────────┐
    │  parallel: register each floating → current average │
    │            (one reg_aladin per CPU node)             │
    └───────────────────────┬─────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │  reg_average  │
                    │  (demean/avg) │
                    └───────┬───────┘
                            │
                  next iteration or NRR
                            ▼

For each non-rigid iteration the same pattern applies using reg_f3d.

Usage
-----
::

    # Set environment variables or use Azure CLI auth
    export AZURE_SUBSCRIPTION_ID="..."
    export AZURE_RESOURCE_GROUP="..."
    export AZURE_WORKSPACE_NAME="..."

    uv run python azure/pipeline.py \\
        --images-uri azureml://datastores/images/paths/study/ \\
        --compute cpu-cluster \\
        --affine-iterations 5 \\
        --nrr-iterations 10

Requirements
------------
``azure-ai-ml``, ``azure-identity`` (install via
``uv add azure-ai-ml azure-identity``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from azure.ai.ml import Input, MLClient, Output, command, dsl, parallel_run_function
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import AmlCompute, Environment
from azure.ai.ml.parallel import RunFunction
from azure.identity import DefaultAzureCredential

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Submit a groupwise registration pipeline to Azure ML.",
)

CONDA_ENV = dict(
    name="niftyreg",
    channels=["defaults"],
    dependencies=[
        "python=3.12",
        {"pip": ["niftyregw>=0.1.3"]},
    ],
)

NIFTYREG_ENV = Environment(
    name="niftyreg-env",
    description="NiftyReg environment with niftyregw.",
    conda_file=CONDA_ENV,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
)

SRC_DIR = str(Path(__file__).resolve().parent / "src")


# ── helper: build ML client ─────────────────────────────────────────


def _get_ml_client(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> MLClient:
    subscription_id = subscription_id or os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = resource_group or os.environ["AZURE_RESOURCE_GROUP"]
    workspace_name = workspace_name or os.environ["AZURE_WORKSPACE_NAME"]
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


# ── components ───────────────────────────────────────────────────────

_register_aladin = command(
    name="register_aladin",
    display_name="Affine registration (reg_aladin)",
    description="Register one floating image to a reference using reg_aladin.",
    inputs={
        "reference": Input(type=AssetTypes.URI_FILE),
        "floating": Input(type=AssetTypes.URI_FILE),
        "input_affine": Input(type=AssetTypes.URI_FILE, optional=True),
        "reference_mask": Input(type=AssetTypes.URI_FILE, optional=True),
        "floating_mask": Input(type=AssetTypes.URI_FILE, optional=True),
        "rigid_only": Input(type="boolean", default=False),
        "extra_args": Input(type="string", default=""),
        "save_result": Input(type="boolean", default=False),
    },
    outputs={
        "output_affine": Output(type=AssetTypes.URI_FILE),
        "output_result": Output(type=AssetTypes.URI_FILE),
    },
    code=SRC_DIR,
    command=(
        "python register_aladin.py"
        " --reference ${{inputs.reference}}"
        " --floating ${{inputs.floating}}"
        " $[[--input-affine ${{inputs.input_affine}}]]"
        " $[[--reference-mask ${{inputs.reference_mask}}]]"
        " $[[--floating-mask ${{inputs.floating_mask}}]]"
        " $[[--rigid-only ${{inputs.rigid_only}}]]"
        " $[[--extra-args ${{inputs.extra_args}}]]"
        " $[[--save-result ${{inputs.save_result}}]]"
        " --output-affine ${{outputs.output_affine}}"
        " --output-result ${{outputs.output_result}}"
    ),
    environment=NIFTYREG_ENV,
)

_register_f3d = command(
    name="register_f3d",
    display_name="Non-rigid registration (reg_f3d)",
    description="Register one floating image to a reference using reg_f3d.",
    inputs={
        "reference": Input(type=AssetTypes.URI_FILE),
        "floating": Input(type=AssetTypes.URI_FILE),
        "input_affine": Input(type=AssetTypes.URI_FILE, optional=True),
        "reference_mask": Input(type=AssetTypes.URI_FILE, optional=True),
        "floating_mask": Input(type=AssetTypes.URI_FILE, optional=True),
        "extra_args": Input(type="string", default=""),
        "save_result": Input(type="boolean", default=False),
    },
    outputs={
        "output_cpp": Output(type=AssetTypes.URI_FILE),
        "output_result": Output(type=AssetTypes.URI_FILE),
    },
    code=SRC_DIR,
    command=(
        "python register_f3d.py"
        " --reference ${{inputs.reference}}"
        " --floating ${{inputs.floating}}"
        " $[[--input-affine ${{inputs.input_affine}}]]"
        " $[[--reference-mask ${{inputs.reference_mask}}]]"
        " $[[--floating-mask ${{inputs.floating_mask}}]]"
        " $[[--extra-args ${{inputs.extra_args}}]]"
        " $[[--save-result ${{inputs.save_result}}]]"
        " --output-cpp ${{outputs.output_cpp}}"
        " --output-result ${{outputs.output_result}}"
    ),
    environment=NIFTYREG_ENV,
)

_average = command(
    name="average_images",
    display_name="Average images (reg_average)",
    description="Average or demean a set of registered images/transformations.",
    inputs={
        "transforms_and_images": Input(type=AssetTypes.URI_FOLDER),
        "reference": Input(type=AssetTypes.URI_FILE),
        "mode": Input(type="string", default="demean"),
    },
    outputs={
        "output_average": Output(type=AssetTypes.URI_FILE),
    },
    code=SRC_DIR,
    command=(
        "python average.py"
        " --transforms-and-images ${{inputs.transforms_and_images}}"
        " --reference ${{inputs.reference}}"
        " --mode ${{inputs.mode}}"
        " --output-average ${{outputs.output_average}}"
    ),
    environment=NIFTYREG_ENV,
)


# ── pipeline definition ─────────────────────────────────────────────


@dsl.pipeline(
    description="Groupwise registration: build a population template from many images.",
)
def groupwise_pipeline(
    images_folder: Input,
    template: Input,
    affine_iterations: int = 5,
    nrr_iterations: int = 10,
    affine_extra_args: str = "",
    nrr_extra_args: str = "",
):
    """Build a population template via iterative groupwise registration.

    The pipeline fans out pairwise registrations across cluster nodes,
    then gathers and averages the results before the next iteration.
    """
    # NOTE: Azure ML pipelines are declarative DAGs.  True dynamic
    # looping (variable iteration count) requires submitting each
    # iteration programmatically.  This pipeline hard-codes a single
    # affine + NRR iteration as a template; callers should wrap this
    # in a Python loop that re-submits with updated inputs, or use the
    # ``submit_iterative`` helper below.

    # ── Affine registration (one iteration) ──
    aff_step = _register_aladin(
        reference=template,
        floating=images_folder,
        rigid_only=True,
        extra_args=affine_extra_args,
        save_result=True,
    )
    aff_step.compute = "cpu-cluster"

    avg_step = _average(
        transforms_and_images=aff_step.outputs.output_affine,
        reference=template,
        mode="avg",
    )
    avg_step.compute = "cpu-cluster"

    return {"template": avg_step.outputs.output_average}


# ── iterative submission helper ──────────────────────────────────────


def submit_iterative(
    ml_client: MLClient,
    images_uri: str,
    template_uri: str,
    compute: str,
    affine_iterations: int,
    nrr_iterations: int,
    affine_extra_args: str,
    nrr_extra_args: str,
    experiment_name: str,
) -> None:
    """Submit the groupwise pipeline iteratively.

    Because Azure ML pipelines are static DAGs, we implement the outer
    iteration loop in Python: each iteration submits a pipeline job that
    fans out pairwise registrations, waits for it to finish, then feeds
    the resulting average back as the template for the next iteration.
    """
    from azure.ai.ml.entities import PipelineJob

    current_template = template_uri

    for aff_it in range(1, affine_iterations + 1):
        is_last_aff = aff_it == affine_iterations
        rigid_only = aff_it == 1
        mode = "avg" if is_last_aff else "demean"

        typer.echo(
            f"── Affine iteration {aff_it}/{affine_iterations}"
            f" (rigid={rigid_only}, avg_mode={mode}) ──"
        )

        @dsl.pipeline(
            description=f"Affine iteration {aff_it}/{affine_iterations}",
            compute=compute,
        )
        def _aff_pipeline(images_folder: Input, template: Input):
            reg = _register_aladin(
                reference=template,
                floating=images_folder,
                rigid_only=rigid_only,
                extra_args=affine_extra_args,
                save_result=is_last_aff,
            )
            avg = _average(
                transforms_and_images=reg.outputs.output_affine,
                reference=template,
                mode=mode,
            )
            return {"template": avg.outputs.output_average}

        job = ml_client.jobs.create_or_update(
            _aff_pipeline(
                images_folder=Input(type=AssetTypes.URI_FOLDER, path=images_uri),
                template=Input(type=AssetTypes.URI_FILE, path=current_template),
            ),
            experiment_name=experiment_name,
        )
        typer.echo(f"  Submitted job: {job.name}")
        ml_client.jobs.stream(job.name)

        current_template = (
            f"azureml://jobs/{job.name}/outputs/template"
        )

    for nrr_it in range(1, nrr_iterations + 1):
        is_last_nrr = nrr_it == nrr_iterations
        mode = "avg" if is_last_nrr else "demean_noaff"

        typer.echo(
            f"── NRR iteration {nrr_it}/{nrr_iterations}"
            f" (avg_mode={mode}) ──"
        )

        @dsl.pipeline(
            description=f"NRR iteration {nrr_it}/{nrr_iterations}",
            compute=compute,
        )
        def _nrr_pipeline(images_folder: Input, template: Input):
            reg = _register_f3d(
                reference=template,
                floating=images_folder,
                extra_args=nrr_extra_args,
                save_result=is_last_nrr,
            )
            avg = _average(
                transforms_and_images=reg.outputs.output_cpp,
                reference=template,
                mode=mode,
            )
            return {"template": avg.outputs.output_average}

        job = ml_client.jobs.create_or_update(
            _nrr_pipeline(
                images_folder=Input(type=AssetTypes.URI_FOLDER, path=images_uri),
                template=Input(type=AssetTypes.URI_FILE, path=current_template),
            ),
            experiment_name=experiment_name,
        )
        typer.echo(f"  Submitted job: {job.name}")
        ml_client.jobs.stream(job.name)

        current_template = (
            f"azureml://jobs/{job.name}/outputs/template"
        )

    typer.echo(f"✓ Final template: {current_template}")


# ── CLI ──────────────────────────────────────────────────────────────


@app.command()
def main(
    images_uri: Annotated[
        str,
        typer.Option(
            help="Azure ML data URI to the folder of input images.",
        ),
    ],
    template_uri: Annotated[
        str,
        typer.Option(
            help=(
                "Azure ML data URI to the initial template image."
                " If not provided, you should create one first"
                " (e.g., by averaging the input images)."
            ),
        ),
    ],
    compute: Annotated[
        str,
        typer.Option(help="Name of the Azure ML CPU compute cluster."),
    ] = "cpu-cluster",
    affine_iterations: Annotated[
        int,
        typer.Option(min=0, help="Number of affine iterations."),
    ] = 5,
    nrr_iterations: Annotated[
        int,
        typer.Option(min=0, help="Number of non-rigid iterations."),
    ] = 10,
    affine_extra_args: Annotated[
        str,
        typer.Option(help="Extra arguments for reg_aladin."),
    ] = "",
    nrr_extra_args: Annotated[
        str,
        typer.Option(help="Extra arguments for reg_f3d."),
    ] = "",
    experiment_name: Annotated[
        str,
        typer.Option(help="Azure ML experiment name."),
    ] = "groupwise-registration",
    subscription_id: Annotated[
        Optional[str],
        typer.Option(
            help="Azure subscription ID (or set AZURE_SUBSCRIPTION_ID).",
        ),
    ] = None,
    resource_group: Annotated[
        Optional[str],
        typer.Option(
            help="Azure resource group (or set AZURE_RESOURCE_GROUP).",
        ),
    ] = None,
    workspace_name: Annotated[
        Optional[str],
        typer.Option(
            help="Azure ML workspace (or set AZURE_WORKSPACE_NAME).",
        ),
    ] = None,
) -> None:
    """Submit groupwise registration to Azure Machine Learning.

    Distributes pairwise registrations across many CPU nodes in an Azure
    ML compute cluster, iterating through affine and non-rigid stages to
    build a population template.
    """
    ml_client = _get_ml_client(subscription_id, resource_group, workspace_name)

    submit_iterative(
        ml_client=ml_client,
        images_uri=images_uri,
        template_uri=template_uri,
        compute=compute,
        affine_iterations=affine_iterations,
        nrr_iterations=nrr_iterations,
        affine_extra_args=affine_extra_args,
        nrr_extra_args=nrr_extra_args,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    app()
