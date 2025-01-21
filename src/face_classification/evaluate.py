import logging
import os
import sys

import hydra
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from face_classification.data import FaceDataset
from face_classification.model import PretrainedResNet34

app = typer.Typer()


@app.command()
def evaluate(model_path: str, config_name: str = "default_config") -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Evaluating with model: {model_path}")

    with hydra.initialize(config_path="../../configs", version_base=None, job_name="evaluate_model"):
        cfg = hydra.compose(config_name=config_name)

    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.evaluate

    # Define the test dataset and dataloader
    test_dataset = FaceDataset(mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    model = PretrainedResNet34(cfg)
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator=hparams.accelerator, logger=WandbLogger(project="face_classification"))

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
