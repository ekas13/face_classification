import os
import sys

import hydra
from omegaconf import OmegaConf

import torch
import typer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from face_classification.data import FaceDataset
from face_classification.model import PretrainedResNet34

# Define a specific checkpoint from the checkpoints directory
model_checkpoint: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "model-epoch=01-val_loss=1.68.ckpt"
)

app = typer.Typer()
@app.command()
def evaluate(model_path: str, config_name: str = "default_config") -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    print("Evaluating with model:", model_path)
    with hydra.initialize(config_path="../../configs", version_base = None, job_name="evaluate_model"):
        cfg = hydra.compose(config_name=config_name)

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

def run_evaluate():
    if len(sys.argv) < 2:
        model_path = None
    else:
        model_path = sys.argv[1]
    evaluate(model_path)


if __name__ == "__main__":
    typer.run(evaluate)
