import torch
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import typer
from model import PretrainedResNet34
from data import FaceDataset

def evaluate(model_path: str) -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    print("Evaluating with model:", model_path)

    # Define the test dataset and dataloader
    test_dataset = FaceDataset(mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = PretrainedResNet34(num_classes=16)
    # Uncomment these when you have a saved model to load:
    '''
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    '''

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator="auto")

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
