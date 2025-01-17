import torch
import typer
import wandb
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from metric_tracker import MetricTracker
from model import PretrainedResNet34
from data import FaceDataset
import os

app = typer.Typer()

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 1
USE_CHROMIUM_PROFILER = True
USE_TENSORFLOW_PROFILER = True

@app.command()
def train(batch_size: int = BATCH_SIZE, epochs: int = EPOCHS) -> None:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    if USE_TENSORFLOW_PROFILER:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler(f"reports/profiler_logs/tensorboard/PretrainedResNet34_train_{epochs}_epoch")) as prof:
            run_training(batch_size, epochs)
        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    if USE_CHROMIUM_PROFILER:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            run_training(batch_size, epochs)
        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
        logdir = f"reports/profiler_logs/chromium_trace/PretrainedResNet34_train_{epochs}_epoch"
        os.makedirs(logdir, exist_ok=True)
        filename= f"{logdir}/train_trace.json"
        prof.export_chrome_trace(filename)
    if not USE_CHROMIUM_PROFILER and not USE_TENSORFLOW_PROFILER:
        run_training(batch_size, epochs)

def run_training(batch_size: int, epochs: int) -> None:
    """Train our model on the Face Dataset."""

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Training with batch size: {batch_size}, and epochs: {epochs}")

    params = {"batch_size": batch_size, "epochs": epochs}
    run = wandb.init(project="face_classification", config=params) # type: ignore
    wandb.log(params)

    train_set = FaceDataset(mode="train")
    val_set = FaceDataset(mode="val")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = PretrainedResNet34(num_classes=16)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="models/checkpoints/",
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
