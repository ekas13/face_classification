import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import typer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from data import FaceDataset
from model import PretrainedResNet34

# Define a specific checkpoint from the checkpoints directory
model_checkpoint: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "model-epoch=01-val_loss=1.68.ckpt"
)

app = typer.Typer()

def evaluate(model_path: str, cfg: DictConfig) -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    print("Evaluating with model:", model_path)
    
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.evaluate

    # Define the test dataset and dataloader
    test_dataset = FaceDataset(mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    model = PretrainedResNet34(cfg)
    # Uncomment these when you have a saved model to load:
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator=hparams.accelerator)

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)

@app.command()
def main(model_path: str = None, config_name: str = "evaluate_config"):
    hydra.initialize(config_path="../../configs")
    cfg = hydra.compose(config_name=config_name)
    evaluate(model_path, cfg)

if __name__ == "__main__":
    typer.run(main)
