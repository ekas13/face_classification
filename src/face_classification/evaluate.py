import torch
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import typer
from model import PretrainedResNet34
import os
from typing import Union

# Define a specific checkpoint from the checkpoints directory
model_checkpoint: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "model-epoch=01-val_loss=1.68.ckpt"
    )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model_checkpoint: str = model_checkpoint) -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    print("Evaluating with model checkpoint:", model_checkpoint)

    # Define the test dataset and dataloader
    # TODO: Replace with our dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="../../data", train=False, download=True, transform=ToTensor()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = PretrainedResNet34(num_classes=10)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator="auto")

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
