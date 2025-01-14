import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import PretrainedResNet34
from metric_tracker import MetricTracker

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10


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
