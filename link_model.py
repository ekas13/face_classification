import os
import urllib.parse

import typer
from omegaconf import OmegaConf

import wandb


def link_model(version: str, aliases: list[str] = ["staging"]) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        version: Version of the artifact to stage (e.g., "v8").
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model v8 -a staging -a best

    """
    config = OmegaConf.load("configs/default_config.yaml")

    if version == "":
        typer.echo("No artifact version provided. Exiting.")
        return

    model_registry_url = config.urls.wandb_registry
    artifact_path = f"{model_registry_url}:{version}"

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
