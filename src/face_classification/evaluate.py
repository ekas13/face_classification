import torch
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import typer
from model import PretrainedResNet34

def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    print("Evaluating with model checkpoint:", model_checkpoint)

    # Define the test dataset and dataloader
    # TODO: Replace with our dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="../../data", train=False, download=True, transform=ToTensor()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PretrainedResNet34(num_classes=10)
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator="auto")

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
