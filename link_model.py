import os
import urllib.parse

import typer
from omegaconf import OmegaConf

import wandb


def link_model(artifact_path: str, aliases: list[str] = ["staging"]) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: path of the artifact to stage.
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model entity/project/artifact_name:version -a staging -a

    """
    config = OmegaConf.load("configs/urls/urls_config.yaml")

    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    model_registry_url = config.wandb_registry

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=model_registry_url, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
