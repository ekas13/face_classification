import torch
import typer
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

from face_classification.metric_tracker import MetricTracker
from face_classification.model import PretrainedResNet34
from face_classification.data import FaceDataset
import os

app = typer.Typer()

@app.command()
def train(config_name: str = "default_config") -> None:
    """Train a model on the Face Dataset."""
    print("Training day and night")
    
    # Initialize Hydra configuration
    with hydra.initialize(config_path="../../configs", version_base=None, job_name="train_model"):
        cfg = hydra.compose(config_name=config_name)

    # Print configuration settings
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.train

    if hparams.use_tensorflow_profiler:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler(f"reports/profiler_logs/tensorboard/PretrainedResNet34_train_{hparams.epochs}_epoch")) as prof:
            run_training(cfg, hparams)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    if hparams.use_chromium_profiler:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            run_training(cfg, hparams)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
        logdir = f"reports/profiler_logs/chromium_trace/PretrainedResNet34_train_{hparams.epochs}_epoch"
        os.makedirs(logdir, exist_ok=True)
        filename= f"{logdir}/train_trace.json"
        prof.export_chrome_trace(filename)
    if not hparams.use_chromium_profiler and not hparams.use_tensorflow_profiler:
        run_training(cfg, hparams)

def run_training(cfg, hparams) -> None:
    
    train_set = FaceDataset(mode="train")
    val_set = FaceDataset(mode="val")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    hparams_model = cfg.model
    optimizer = hparams_model.optimizer
    model = PretrainedResNet34(num_classes = hparams_model.num_classes, fine_tuning = hparams_model.fine_tuning, optimizer_type = optimizer.type, optimizer_lr = optimizer.lr)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="models/checkpoints/",
        filename="model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=hparams.save_top_k,
        mode=hparams.mode,
    )
    metric_tracker = MetricTracker()

    trainer = Trainer(max_epochs=hparams.epochs, callbacks=[checkpoint_callback, metric_tracker], accelerator="auto")
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    print("Training finished")

if __name__ == "__main__":
    typer.run(train)
