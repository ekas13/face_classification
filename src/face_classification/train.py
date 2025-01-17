import torch
import typer
import wandb
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from metric_tracker import MetricTracker
from model import PretrainedResNet34
from data import FaceDataset

app = typer.Typer()

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30


@app.command()
def train(batch_size: int = BATCH_SIZE, epochs: int = EPOCHS) -> None:
    """Train a model on CIFAR10."""

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Training day and night")
    logger.info(f"Training with batch size: {batch_size}, and epochs: {epochs}")

    params = {"batch_size": batch_size, "epochs": epochs}
    run = wandb.init(project="face_classification", config=params) # type: ignore
    wandb.log(params)

    train_set = FaceDataset(mode="train")
    val_set = FaceDataset(mode="val")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = PretrainedResNet34(num_classes=16)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    metric_tracker = MetricTracker()

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback, metric_tracker],
                      accelerator="auto", logger=WandbLogger(project="face_classification"))
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    typer.run(train)
