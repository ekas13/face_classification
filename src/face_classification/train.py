import torch
import typer
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from face_classification.metric_tracker import MetricTracker
from face_classification.model import PretrainedResNet34
from face_classification.data import FaceDataset

app = typer.Typer()

@app.command()
def train(config_name: str = "default_config") -> None:
    """Train a model on CIFAR10."""
    print("Training day and night")
    with hydra.initialize(config_path="../../configs", version_base = None, job_name="train_model"):
        cfg = hydra.compose(config_name=config_name)

    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.train

    train_set = FaceDataset(mode="train")
    val_set = FaceDataset(mode="val")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    model = PretrainedResNet34(cfg)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
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
