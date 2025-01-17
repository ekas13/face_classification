import logging
import torch
import typer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from data import FaceDataset
from model import PretrainedResNet34

app = typer.Typer()

@app.command()
def evaluate(model_path: str) -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluating with model:", model_path)

    map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the test dataset and dataloader
    test_dataset = FaceDataset(mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = PretrainedResNet34(num_classes=16)
    # Uncomment these when you have a saved model to load:
    if model_path:
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint["state_dict"])

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator="auto", logger=WandbLogger(project="face_classification"))

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
