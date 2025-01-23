import os
import urllib.parse

import typer

import wandb


def link_model(artifact_path: str, aliases: list[str] = ["staging"]) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model entity/project/artifact_name:version -a staging -a best

    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    _, _, artifact_name_version = artifact_path.split("/")

    artifact = api.artifact(artifact_path)
    artifact.link(
        target_path=f"vbranica-danmarks-tekniske-universitet-dtu-org/wandb-registry-face_classification_registry/Model_collection:latest",
        aliases=aliases,
    )
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
