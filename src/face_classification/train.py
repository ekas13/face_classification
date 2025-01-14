import os

import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torchvision import datasets, transforms

from face_classification.model import PretrainedResNet34
from face_classification.visualize import plot_accuracy, plot_loss

transform = transforms.Compose([transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()
figures_path = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 2


class MetricTracker(Callback):
    """
    A PyTorch Lightning callback for tracking and logging training, validation, and test metrics.

    This callback collects metrics such as loss and accuracy during the training process
    and stores them in a dictionary for later use. At the end of training, the metrics
    are visualized using custom plotting functions.

    Attributes:
        collection (dict): A dictionary containing the tracked metrics:
            - "train_loss" (list): List of training loss values.
            - "train_acc" (list): List of training accuracy values.
            - "val_loss" (list): List of validation loss values.
            - "val_acc" (list): List of validation accuracy values.
            - "test_loss" (list): List of test loss values.
            - "test_acc" (list): List of test accuracy values.

    Methods:
        on_train_epoch_end(trainer, pl_module):
            Called at the end of each training epoch to log and store the current
            training and validation metrics.

        on_train_end(trainer, pl_module):
            Called at the end of training to visualize and save the collected metrics
            using the `plot_loss` and `plot_accuracy` functions.
    """

    def __init__(self):
        self.collection = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.callback_metrics
        train_loss = elogs.get("train_loss")
        train_acc = elogs.get("train_acc")
        val_loss = elogs.get("val_loss")
        val_acc = elogs.get("val_acc")
        if train_loss is not None:
            self.collection["train_loss"].append(train_loss.item())
        if train_acc is not None:
            self.collection["train_acc"].append(train_acc.item())
        if val_loss is not None:
            self.collection["val_loss"].append(val_loss.item())
        if val_acc is not None:
            self.collection["val_acc"].append(val_acc.item())

    def on_train_end(self, trainer, pl_module):
        plot_loss(self.collection, figures_path)
        plot_accuracy(self.collection, figures_path)


@app.command()
def train(batch_size: int = BATCH_SIZE, epochs: int = EPOCHS) -> None:
    """Train a model on CIFAR10."""
    print("Training day and night")

    train_set = datasets.CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = PretrainedResNet34(num_classes=10).to(device)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="max",
    )
    metric_tracker = MetricTracker()

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback, metric_tracker])
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    print("Training finished")


if __name__ == "__main__":
    typer.run(train)
